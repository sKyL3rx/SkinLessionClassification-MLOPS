from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from lesion_ml.paths import get_project_paths

DEFAULT_LABELS = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML root to be a mapping, got: {type(data).__name__}")

    return data


def compute_metrics(
    df: pd.DataFrame,
    labels: list[str],
) -> dict[str, float | int | None]:
    """Compute classification metrics for the provided predictions."""
    if df.empty:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "balanced_accuracy": None,
        }

    y_true = df["true_label"].astype(str)
    y_pred = df["calibrated_pred_label"].astype(str)

    return {
        "n": int(len(df)),
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
    }


def class_recall(
    df: pd.DataFrame,
    class_name: str,
) -> float | None:
    """Compute recall for one class on the provided predictions."""
    subset = df[df["true_label"].astype(str) == class_name]

    if subset.empty:
        return None

    correct = subset["calibrated_pred_label"].astype(str) == class_name

    return float(correct.mean())


def validate_predictions(
    df: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """Validate and normalize the calibrated prediction dataframe."""
    required_columns = [
        "true_label",
        "calibrated_pred_label",
        "calibrated_confidence",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in calibrated predictions: {missing_columns}")

    validated = df.copy()

    null_counts = validated[required_columns].isna().sum()

    columns_with_nulls = {column: int(count) for column, count in null_counts.items() if count > 0}

    if columns_with_nulls:
        raise ValueError(f"Calibrated predictions contain missing values: {columns_with_nulls}")

    validated["true_label"] = validated["true_label"].astype(str)
    validated["calibrated_pred_label"] = validated["calibrated_pred_label"].astype(str)

    try:
        validated["calibrated_confidence"] = pd.to_numeric(
            validated["calibrated_confidence"],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Column 'calibrated_confidence' must contain numeric values.") from error

    invalid_confidence = validated[
        ~validated["calibrated_confidence"].between(
            0.0,
            1.0,
            inclusive="both",
        )
    ]

    if not invalid_confidence.empty:
        invalid_values = invalid_confidence["calibrated_confidence"].head(10).tolist()

        raise ValueError(
            f"Column 'calibrated_confidence' contains values outside [0, 1]: {invalid_values}"
        )

    allowed_labels = set(labels)

    unknown_true_labels = set(validated["true_label"].unique()) - allowed_labels
    unknown_predicted_labels = set(validated["calibrated_pred_label"].unique()) - allowed_labels

    if unknown_true_labels:
        raise ValueError(f"Unknown true labels found in predictions: {sorted(unknown_true_labels)}")

    if unknown_predicted_labels:
        raise ValueError(
            f"Unknown calibrated prediction labels found: {sorted(unknown_predicted_labels)}"
        )

    return validated


def make_markdown(
    df: pd.DataFrame,
    split: str,
) -> str:
    """Create a Markdown report for the threshold sweep."""
    display = df.copy()

    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].round(4)

    preferred_columns = [
        "split",
        "threshold",
        "total_n",
        "accepted_n",
        "review_n",
        "coverage",
        "needs_review_rate",
        "overall_accuracy",
        "overall_macro_f1",
        "overall_balanced_accuracy",
        "accepted_accuracy",
        "accepted_macro_f1",
        "accepted_balanced_accuracy",
        "accepted_mel_recall",
        "accepted_akiec_recall",
    ]

    preferred_columns = [column for column in preferred_columns if column in display.columns]

    display = display[preferred_columns]

    try:
        table = display.to_markdown(index=False)
    except Exception:
        table = display.to_string(index=False)

    text = f"# Selective Prediction Summary ({split})\n\n"

    text += (
        "The system accepts predictions above a calibrated confidence "
        "threshold and routes lower-confidence cases for review.\n\n"
    )

    text += table
    text += "\n\n"

    text += "Notes:\n"
    text += "- `coverage` is the fraction of samples accepted for automatic prediction.\n"
    text += "- `needs_review_rate` is the fraction routed to review.\n"
    text += "- `accepted_*` metrics are computed only on accepted predictions.\n"
    text += "- Threshold selection is performed only on validation data.\n"
    text += "- Test data must be used only for final reporting, not for threshold selection.\n"

    return text


def evaluate_thresholds(
    df: pd.DataFrame,
    *,
    thresholds: list[float],
    labels: list[str],
    split: str,
) -> pd.DataFrame:
    """Evaluate selective prediction performance at each threshold."""
    validated_df = validate_predictions(
        df,
        labels,
    )

    total = len(validated_df)

    overall_metrics = compute_metrics(
        validated_df,
        labels,
    )

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        accepted = validated_df[validated_df["calibrated_confidence"] >= threshold].copy()

        review = validated_df[validated_df["calibrated_confidence"] < threshold].copy()

        accepted_metrics = compute_metrics(
            accepted,
            labels,
        )

        rows.append(
            {
                "split": split,
                "threshold": float(threshold),
                "total_n": int(total),
                "accepted_n": int(len(accepted)),
                "review_n": int(len(review)),
                "coverage": (float(len(accepted) / total) if total else 0.0),
                "needs_review_rate": (float(len(review) / total) if total else 0.0),
                "overall_accuracy": (overall_metrics["accuracy"]),
                "overall_macro_f1": (overall_metrics["macro_f1"]),
                "overall_weighted_f1": (overall_metrics["weighted_f1"]),
                "overall_balanced_accuracy": (overall_metrics["balanced_accuracy"]),
                "accepted_accuracy": (accepted_metrics["accuracy"]),
                "accepted_macro_f1": (accepted_metrics["macro_f1"]),
                "accepted_weighted_f1": (accepted_metrics["weighted_f1"]),
                "accepted_balanced_accuracy": (accepted_metrics["balanced_accuracy"]),
                "accepted_mel_recall": class_recall(
                    accepted,
                    "mel",
                ),
                "accepted_akiec_recall": class_recall(
                    accepted,
                    "akiec",
                ),
            }
        )

    return pd.DataFrame(rows)


def select_validation_threshold(
    result_df: pd.DataFrame,
    *,
    minimum_coverage: float,
) -> dict[str, Any]:
    """
    Select the best validation threshold subject to minimum coverage.

    Primary metric:
        accepted_macro_f1

    Tie breaker:
        higher threshold
    """
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError(f"minimum_coverage must be between 0 and 1, got {minimum_coverage}")

    eligible = result_df[
        (result_df["coverage"] >= minimum_coverage) & result_df["accepted_macro_f1"].notna()
    ].copy()

    if eligible.empty:
        raise RuntimeError(f"No threshold satisfies minimum_coverage={minimum_coverage}")

    selected = eligible.sort_values(
        by=[
            "accepted_macro_f1",
            "threshold",
        ],
        ascending=[
            False,
            False,
        ],
    ).iloc[0]

    return {
        "schema_version": 1,
        "threshold": float(selected["threshold"]),
        "selection_split": "val",
        "selection_metric": "accepted_macro_f1",
        "minimum_coverage": float(minimum_coverage),
        "validation_coverage": float(selected["coverage"]),
        "validation_accepted_macro_f1": float(selected["accepted_macro_f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Evaluate selective prediction and needs-review triage.")
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Project configuration YAML.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "val",
            "test",
        ],
        help=(
            "Calibrated prediction split to evaluate. "
            "Threshold selection is performed only for val."
        ),
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    paths = get_project_paths(cfg)

    data_config = cfg.get("data", {})

    labels = [
        str(label)
        for label in data_config.get(
            "labels",
            DEFAULT_LABELS,
        )
    ]

    selective_config = cfg.get("selective_prediction")

    if not isinstance(selective_config, dict):
        raise ValueError("Missing 'selective_prediction' mapping in config.")

    raw_thresholds = selective_config.get("thresholds")

    if not isinstance(raw_thresholds, list) or not raw_thresholds:
        raise ValueError("'selective_prediction.thresholds' must be a non-empty list.")

    thresholds = [float(value) for value in raw_thresholds]

    invalid_thresholds = [threshold for threshold in thresholds if not 0.0 <= threshold <= 1.0]

    if invalid_thresholds:
        raise ValueError(
            "All selective prediction thresholds must be between "
            f"0 and 1. Invalid values: {invalid_thresholds}"
        )

    if args.split == "val":
        predictions_path = paths.calibration_dir / "calibrated_val_predictions.csv"
    else:
        predictions_path = paths.calibration_dir / "calibrated_test_predictions.csv"

    output_dir = paths.selective_prediction_dir / args.split

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing calibrated predictions: {predictions_path}. "
            "Run scripts/calibrate_temperature.py first."
        )

    predictions_df = pd.read_csv(predictions_path)

    result_df = evaluate_thresholds(
        predictions_df,
        thresholds=thresholds,
        labels=labels,
        split=args.split,
    )

    csv_path = output_dir / f"selective_prediction_curve_{args.split}.csv"

    markdown_path = output_dir / f"selective_prediction_summary_{args.split}.md"

    result_df.to_csv(
        csv_path,
        index=False,
    )

    markdown_path.write_text(
        make_markdown(
            result_df,
            args.split,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] Wrote selective prediction CSV: {csv_path}")
    print(f"[DONE] Wrote selective prediction Markdown: {markdown_path}")

    if args.split == "val":
        selection_config = selective_config.get(
            "selection",
            {},
        )

        if not isinstance(selection_config, dict):
            raise ValueError("'selective_prediction.selection' must be a mapping.")

        minimum_coverage = float(
            selection_config.get(
                "minimum_coverage",
                0.80,
            )
        )

        decision = select_validation_threshold(
            result_df,
            minimum_coverage=minimum_coverage,
        )

        decision_path = paths.calibration_dir / "decision.json"

        decision_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        decision_path.write_text(
            json.dumps(
                decision,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(f"[DONE] Deployment decision: {decision_path}")
        print(
            "[SELECTED] threshold="
            f"{decision['threshold']:.4f}, "
            "validation_coverage="
            f"{decision['validation_coverage']:.4f}, "
            "validation_accepted_macro_f1="
            f"{decision['validation_accepted_macro_f1']:.4f}"
        )

    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
