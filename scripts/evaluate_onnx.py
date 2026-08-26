from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from lesion_ml.inference.calibration import softmax
from lesion_ml.inference.onnx_predictor import (
    SkinLesionONNXPredictor,
)
from lesion_ml.paths import get_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Evaluate the exported ONNX model using deployment preprocessing.")
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        required=True,
    )

    return parser.parse_args()


def load_config(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def build_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, float]:
    label_to_index = {label: index for index, label in enumerate(labels)}

    y_true_indices = np.array(
        [label_to_index[str(label)] for label in y_true],
        dtype=np.int64,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "nll": float(
            log_loss(
                y_true_indices,
                probabilities,
                labels=list(range(len(labels))),
            )
        ),
    }


def optional_value(
    row: pd.Series,
    column: str,
) -> Any:
    if column not in row:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = get_project_paths(config)

    split_csv = Path(config["data"][f"{args.split}_csv"])

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    predictor = SkinLesionONNXPredictor.from_export_dir(
        paths.onnx_dir,
        provider=str(
            config.get(
                "inference",
                {},
            ).get("onnx_provider", "cpu")
        ),
    )

    dataframe = pd.read_csv(split_csv)

    required_columns = {
        "image_id",
        "image_path",
        "label",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing_columns)}")

    records: list[dict[str, Any]] = []
    all_probabilities: list[np.ndarray] = []

    for _, row in dataframe.iterrows():
        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            logits = predictor.predict_logits(
                image.convert("RGB"),
                age=optional_value(row, "age"),
                sex=optional_value(row, "sex"),
                anatom_site=optional_value(
                    row,
                    "anatom_site",
                ),
            )

        probabilities = softmax(logits)[0]

        predicted_index = int(np.argmax(probabilities))

        predicted_label = predictor.labels[predicted_index]

        record: dict[str, Any] = {
            "image_id": str(row["image_id"]),
            "image_path": str(image_path),
            "true_label": str(row["label"]),
            "predicted_label": predicted_label,
            "confidence": float(probabilities[predicted_index]),
            "correct": bool(predicted_label == str(row["label"])),
        }

        for index, label in enumerate(predictor.labels):
            record[f"logit_{label}"] = float(logits[0, index])
            record[f"prob_{label}"] = float(probabilities[index])

        records.append(record)
        all_probabilities.append(probabilities)

    predictions = pd.DataFrame(records)

    probabilities_array = np.stack(
        all_probabilities,
        axis=0,
    )

    metrics = build_metrics(
        y_true=predictions["true_label"].to_numpy(),
        y_pred=predictions["predicted_label"].to_numpy(),
        probabilities=probabilities_array,
        labels=predictor.labels,
    )

    output_dir = paths.deployment_eval_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = output_dir / f"{args.split}_predictions.csv"

    metrics_path = output_dir / f"{args.split}_metrics.json"

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] Predictions: {predictions_path}")
    print(f"[DONE] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
