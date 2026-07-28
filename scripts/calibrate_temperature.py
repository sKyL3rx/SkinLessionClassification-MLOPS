from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from lesion_ml.inference.calibration import (
    apply_temperature_to_logits,
    softmax,
)
from lesion_ml.paths import get_project_paths

DEFAULT_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling from ONNX validation logits.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to the canonical project configuration.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return config


def logit_columns(labels: list[str]) -> list[str]:
    return [f"logit_{label}" for label in labels]


def validate_labels(labels: list[str]) -> list[str]:
    normalized = [str(label).strip().lower() for label in labels]

    if not normalized:
        raise ValueError("data.labels cannot be empty.")

    if any(not label for label in normalized):
        raise ValueError("data.labels cannot contain empty values.")

    if len(normalized) != len(set(normalized)):
        raise ValueError(f"data.labels contains duplicates: {normalized}")

    return normalized


def load_predictions(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    dataframe = pd.read_csv(path)
    required_columns = [
        "image_id",
        "true_label",
        *logit_columns(labels),
    ]

    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path} missing columns: {missing_columns}")

    dataframe = dataframe.copy()
    dataframe["image_id"] = dataframe["image_id"].astype(str)
    dataframe["true_label"] = dataframe["true_label"].astype(str).str.strip().str.lower()

    unknown_labels = sorted(set(dataframe["true_label"]) - set(labels))
    if unknown_labels:
        raise ValueError(f"{path} contains unknown true labels: {unknown_labels}")

    logits = dataframe[logit_columns(labels)].to_numpy(dtype=np.float64)

    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError(f"Expected logits shape [N, {len(labels)}] in {path}, got {logits.shape}")

    if logits.shape[0] == 0:
        raise ValueError(f"Predictions file is empty: {path}")

    if not np.isfinite(logits).all():
        raise ValueError(f"{path} contains NaN or infinite logits.")

    return dataframe


def labels_to_indices(y_true: list[str], labels: list[str]) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(labels)}

    try:
        return np.asarray([mapping[label] for label in y_true], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown label: {error.args[0]}") from error


def negative_log_likelihood(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    eps: float = 1e-12,
) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=np.int64)

    if probabilities.ndim != 2:
        raise ValueError(f"Expected probabilities with shape [N, C], got {probabilities.shape}")

    if probabilities.shape[0] != target_indices.shape[0]:
        raise ValueError(
            "Probability row count does not match target count: "
            f"{probabilities.shape[0]} != {target_indices.shape[0]}"
        )

    selected = probabilities[np.arange(len(target_indices)), target_indices]
    selected = np.clip(selected, eps, 1.0)
    return float(-np.log(selected).mean())


def brier_score_multiclass(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    num_classes: int,
) -> float:
    one_hot = np.zeros((len(target_indices), num_classes), dtype=np.float64)
    one_hot[np.arange(len(target_indices)), target_indices] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == target_indices).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        if not np.any(mask):
            continue

        bin_accuracy = float(correct[mask].mean())
        bin_confidence = float(confidence[mask].mean())
        bin_weight = float(mask.mean())
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_reliability_diagram(
    before_probabilities: np.ndarray,
    after_probabilities: np.ndarray,
    y_true: list[str],
    labels: list[str],
    output_path: Path,
    n_bins: int = 15,
) -> None:
    target_indices = labels_to_indices(y_true, labels)

    def bin_statistics(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        confidence = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        correct = (predictions == target_indices).astype(np.float64)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        mean_confidences: list[float] = []
        accuracies: list[float] = []

        for lower, upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
            if upper == 1.0:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)

            if not np.any(mask):
                continue

            mean_confidences.append(float(confidence[mask].mean()))
            accuracies.append(float(correct[mask].mean()))

        return np.asarray(mean_confidences), np.asarray(accuracies)

    before_x, before_y = bin_statistics(before_probabilities)
    after_x, after_y = bin_statistics(after_probabilities)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(before_x, before_y, marker="o", label="Before calibration")
    plt.plot(after_x, after_y, marker="o", label="After calibration")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Validation Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def fit_temperature(
    validation_logits: np.ndarray,
    validation_target_indices: np.ndarray,
    *,
    min_temperature: float,
    max_temperature: float,
    max_iter: int,
    eps: float,
) -> float:
    if min_temperature <= 0:
        raise ValueError("calibration.min_temperature must be positive.")

    if max_temperature <= min_temperature:
        raise ValueError("calibration.max_temperature must be greater than min_temperature.")

    if max_iter <= 0:
        raise ValueError("calibration.max_iter must be positive.")

    def objective(temperature: float) -> float:
        calibrated_probabilities = apply_temperature_to_logits(
            validation_logits,
            temperature,
        )
        return negative_log_likelihood(
            calibrated_probabilities,
            validation_target_indices,
            eps=eps,
        )

    result = minimize_scalar(
        objective,
        bounds=(min_temperature, max_temperature),
        method="bounded",
        options={
            "xatol": 1e-4,
            "maxiter": max_iter,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Temperature optimization failed: status={result.status}, message={result.message}"
        )

    temperature = float(result.x)
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"Invalid optimized temperature: {temperature}")

    return temperature


def compute_metrics(
    probabilities: np.ndarray,
    y_true: list[str],
    labels: list[str],
    *,
    eps: float,
) -> dict[str, float]:
    target_indices = labels_to_indices(y_true, labels)
    prediction_indices = probabilities.argmax(axis=1)
    predictions = [labels[index] for index in prediction_indices]

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "nll": negative_log_likelihood(
            probabilities,
            target_indices,
            eps=eps,
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "brier": brier_score_multiclass(
            probabilities,
            target_indices,
            len(labels),
        ),
        "ece": expected_calibration_error(probabilities, target_indices),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def add_calibrated_columns(
    dataframe: pd.DataFrame,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    output = dataframe.copy()

    raw_prediction_indices = raw_probabilities.argmax(axis=1)
    calibrated_prediction_indices = calibrated_probabilities.argmax(axis=1)

    output["predicted_label"] = [labels[index] for index in raw_prediction_indices]
    output["confidence"] = raw_probabilities.max(axis=1)
    output["correct"] = output["predicted_label"] == output["true_label"]

    output["calibrated_pred_label"] = [labels[index] for index in calibrated_prediction_indices]
    output["calibrated_confidence"] = calibrated_probabilities.max(axis=1)
    output["calibrated_correct"] = output["calibrated_pred_label"] == output["true_label"]

    for index, label in enumerate(labels):
        output[f"prob_{label}"] = raw_probabilities[:, index]
        output[f"calibrated_prob_{label}"] = calibrated_probabilities[:, index]

    return output


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    labels = validate_labels(config.get("data", {}).get("labels", DEFAULT_LABELS))
    calibration_config = config.get("calibration", {})

    eps = float(calibration_config.get("eps", 1e-12))
    min_temperature = float(calibration_config.get("min_temperature", 0.05))
    max_temperature = float(calibration_config.get("max_temperature", 10.0))
    max_iter = int(calibration_config.get("max_iter", 500))
    optimize_metric = str(calibration_config.get("optimize_metric", "nll")).lower()

    if optimize_metric != "nll":
        raise ValueError(
            "Only calibration.optimize_metric='nll' is currently supported, "
            f"got {optimize_metric!r}."
        )

    paths = get_project_paths(config)

    validation_path = paths.deployment_eval_dir / "val_predictions.csv"
    test_path = paths.deployment_eval_dir / "test_predictions.csv"

    output_dir = paths.calibration_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading ONNX validation predictions: {validation_path}")
    validation_dataframe = load_predictions(validation_path, labels)

    print(f"[INFO] Loading ONNX test predictions: {test_path}")
    test_dataframe = load_predictions(test_path, labels)

    columns = logit_columns(labels)
    validation_logits = validation_dataframe[columns].to_numpy(dtype=np.float64)
    test_logits = test_dataframe[columns].to_numpy(dtype=np.float64)

    validation_probabilities_before = softmax(validation_logits)
    test_probabilities_before = softmax(test_logits)

    validation_targets = validation_dataframe["true_label"].tolist()
    test_targets = test_dataframe["true_label"].tolist()
    validation_target_indices = labels_to_indices(validation_targets, labels)

    temperature = fit_temperature(
        validation_logits,
        validation_target_indices,
        min_temperature=min_temperature,
        max_temperature=max_temperature,
        max_iter=max_iter,
        eps=eps,
    )
    print(f"[DONE] Optimized validation temperature: {temperature:.6f}")

    validation_probabilities_after = apply_temperature_to_logits(
        validation_logits,
        temperature,
    )
    test_probabilities_after = apply_temperature_to_logits(
        test_logits,
        temperature,
    )

    metrics = {
        "schema_version": 1,
        "temperature": temperature,
        "fit_split": "val",
        "source_backend": "onnxruntime",
        "optimize_metric": optimize_metric,
        "validation": {
            "num_samples": int(len(validation_dataframe)),
            "source_predictions": str(validation_path),
            "before": compute_metrics(
                validation_probabilities_before,
                validation_targets,
                labels,
                eps=eps,
            ),
            "after": compute_metrics(
                validation_probabilities_after,
                validation_targets,
                labels,
                eps=eps,
            ),
        },
        "test": {
            "num_samples": int(len(test_dataframe)),
            "source_predictions": str(test_path),
            "before": compute_metrics(
                test_probabilities_before,
                test_targets,
                labels,
                eps=eps,
            ),
            "after": compute_metrics(
                test_probabilities_after,
                test_targets,
                labels,
                eps=eps,
            ),
        },
    }

    temperature_payload = {
        "schema_version": 1,
        "temperature": float(temperature),
        "fit_split": "val",
        "source_backend": "onnxruntime",
        "source_predictions": str(validation_path),
        "optimize_metric": optimize_metric,
    }

    temperature_path = output_dir / "temperature.json"
    metrics_path = output_dir / "calibration_metrics.json"
    reliability_path = output_dir / "reliability_diagram.png"
    calibrated_validation_path = output_dir / "calibrated_val_predictions.csv"
    calibrated_test_path = output_dir / "calibrated_test_predictions.csv"

    temperature_path.write_text(
        json.dumps(temperature_payload, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_reliability_diagram(
        before_probabilities=validation_probabilities_before,
        after_probabilities=validation_probabilities_after,
        y_true=validation_targets,
        labels=labels,
        output_path=reliability_path,
    )

    calibrated_validation = add_calibrated_columns(
        validation_dataframe,
        validation_probabilities_before,
        validation_probabilities_after,
        labels,
    )
    calibrated_test = add_calibrated_columns(
        test_dataframe,
        test_probabilities_before,
        test_probabilities_after,
        labels,
    )

    calibrated_validation.to_csv(calibrated_validation_path, index=False)
    calibrated_test.to_csv(calibrated_test_path, index=False)

    print(f"[DONE] Wrote temperature: {temperature_path}")
    print(f"[DONE] Wrote metrics: {metrics_path}")
    print(f"[DONE] Wrote validation reliability diagram: {reliability_path}")
    print(f"[DONE] Wrote calibrated validation predictions: {calibrated_validation_path}")
    print(f"[DONE] Wrote calibrated test predictions: {calibrated_test_path}")
    print(
        "[INFO] Validation ECE before -> after: "
        f"{metrics['validation']['before']['ece']:.6f} -> "
        f"{metrics['validation']['after']['ece']:.6f}"
    )
    print(
        "[INFO] Test ECE before -> after: "
        f"{metrics['test']['before']['ece']:.6f} -> "
        f"{metrics['test']['after']['ece']:.6f}"
    )
    print(
        "[INFO] Test NLL before -> after: "
        f"{metrics['test']['before']['nll']:.6f} -> "
        f"{metrics['test']['after']['nll']:.6f}"
    )


if __name__ == "__main__":
    main()
