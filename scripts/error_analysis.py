from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from lesion_ml.paths import get_project_paths

DEFAULT_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)
    required = [
        "image_id",
        "true_label",
        "calibrated_pred_label",
        "calibrated_confidence",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    if df.empty:
        raise ValueError(f"Predictions file is empty: {path}")

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["true_label"] = df["true_label"].astype(str)
    df["calibrated_pred_label"] = df["calibrated_pred_label"].astype(str)

    try:
        df["calibrated_confidence"] = pd.to_numeric(
            df["calibrated_confidence"],
            errors="raise",
        ).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} contains non-numeric calibrated confidence values.") from error

    invalid_confidence = df[
        df["calibrated_confidence"].isna() | ~df["calibrated_confidence"].between(0.0, 1.0)
    ]
    if not invalid_confidence.empty:
        raise ValueError(
            f"{path} contains calibrated confidence values outside [0, 1] or missing values."
        )

    return df


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")

    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{path} must contain image_id column.")

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    return df


def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "age" not in out.columns:
        out["age_bucket"] = "unknown"
        return out

    age = pd.to_numeric(out["age"], errors="coerce")

    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels = ["<=20", "21-40", "41-60", "61-80", "80+"]

    out["age_bucket"] = pd.cut(age, bins=bins, labels=labels).astype(str)
    out.loc[age.isna(), "age_bucket"] = "unknown"
    return out


def compute_basic_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "balanced_accuracy": None,
        }

    return {
        "n": int(len(y_true)),
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
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def per_class_metrics(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                }
                for label in labels
            ]
        )

    report = classification_report(
        df["true_label"].astype(str),
        df["calibrated_pred_label"].astype(str),
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label in labels:
        r = report.get(label, {})
        rows.append(
            {
                "label": label,
                "precision": float(r.get("precision", 0.0)),
                "recall": float(r.get("recall", 0.0)),
                "f1": float(r.get("f1-score", 0.0)),
                "support": int(r.get("support", 0)),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(
    df: pd.DataFrame,
    group_col: str,
    labels: list[str],
    min_n: int = 10,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for group_value, group_df in df.groupby(group_col, dropna=False):
        group_df = group_df.copy()
        n = len(group_df)

        metrics = compute_basic_metrics(
            group_df["true_label"].astype(str),
            group_df["calibrated_pred_label"].astype(str),
            labels,
        )

        row = {
            "group_col": group_col,
            "group_value": str(group_value),
            "n": n,
            "is_small_group": bool(n < min_n),
            **{k: v for k, v in metrics.items() if k != "n"},
        }

        # Important class-specific recall.
        for cls in ["mel", "akiec", "bcc", "bkl", "df", "vasc"]:
            cls_df = group_df[group_df["true_label"].astype(str) == cls]
            if len(cls_df) == 0:
                row[f"{cls}_support"] = 0
                row[f"{cls}_recall"] = None
            else:
                row[f"{cls}_support"] = int(len(cls_df))
                row[f"{cls}_recall"] = float(
                    (cls_df["calibrated_pred_label"].astype(str) == cls).mean()
                )

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    return out.sort_values(["n", "group_value"], ascending=[False, True])


def confusion_outputs(df: pd.DataFrame, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cm = confusion_matrix(
        df["true_label"].astype(str),
        df["calibrated_pred_label"].astype(str),
        labels=labels,
    )

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true_label"

    row_sums = cm_df.sum(axis=1).replace(0, np.nan)
    cm_norm = cm_df.div(row_sums, axis=0).fillna(0.0)

    return cm_df, cm_norm


def top_errors(df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    errors = df[df["true_label"].astype(str) != df["calibrated_pred_label"].astype(str)].copy()

    if len(errors) == 0:
        return errors

    preferred_cols = [
        "image_id",
        "true_label",
        "calibrated_pred_label",
        "calibrated_confidence",
        "age",
        "sex",
        "anatom_site",
        "dx_type",
        "image_path",
    ]

    cols = [c for c in preferred_cols if c in errors.columns]
    errors = errors.sort_values("calibrated_confidence", ascending=False)
    return errors[cols].head(top_k)


def mel_bkl_analysis(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["true_label"].isin(["mel", "bkl"])].copy()

    rows = []
    for true_label in ["mel", "bkl"]:
        true_df = subset[subset["true_label"] == true_label]
        if len(true_df) == 0:
            continue

        pred_counts = (
            true_df["calibrated_pred_label"]
            .value_counts()
            .rename_axis("pred_label")
            .reset_index(name="count")
        )
        pred_counts["true_label"] = true_label
        pred_counts["rate"] = pred_counts["count"] / len(true_df)
        rows.append(pred_counts)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)[["true_label", "pred_label", "count", "rate"]]


def make_summary(
    overall_metrics: dict[str, Any],
    per_class_df: pd.DataFrame,
    top_errors_df: pd.DataFrame,
    threshold: float,
) -> str:
    per_class_display = per_class_df.copy()
    for col in ["precision", "recall", "f1"]:
        per_class_display[col] = per_class_display[col].round(4)

    try:
        per_class_table = per_class_display.to_markdown(index=False)
    except Exception:
        per_class_table = per_class_display.to_string(index=False)

    if len(top_errors_df) > 0:
        err_display = top_errors_df.head(10).copy()
        if "calibrated_confidence" in err_display.columns:
            err_display["calibrated_confidence"] = err_display["calibrated_confidence"].round(4)

        try:
            err_table = err_display.to_markdown(index=False)
        except Exception:
            err_table = err_display.to_string(index=False)
    else:
        err_table = "No errors found."

    lines = []
    lines.append("# Error Analysis Summary")
    lines.append("")
    lines.append(f"Final triage threshold: `{threshold}`")
    lines.append("")
    lines.append("## Overall test metrics")
    lines.append("")
    lines.append(f"- Accuracy: `{overall_metrics['accuracy']:.4f}`")
    lines.append(f"- Macro-F1: `{overall_metrics['macro_f1']:.4f}`")
    lines.append(f"- Balanced accuracy: `{overall_metrics['balanced_accuracy']:.4f}`")
    lines.append("")
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append(per_class_table)
    lines.append("")
    lines.append("## Top high-confidence errors")
    lines.append("")
    lines.append(err_table)
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `dx_type` is used only for analysis, not as a model input.")
    lines.append("- Small subgroup metrics should be interpreted carefully.")
    lines.append(
        "- High-confidence errors are useful for qualitative case studies and demo examples."
    )
    lines.append("")
    return "\n".join(lines)


def load_selected_threshold(decision_path: Path) -> float:
    """Load and validate the threshold selected on the validation split."""
    if not decision_path.exists():
        raise FileNotFoundError(
            f"Missing deployment decision: {decision_path}. "
            "Run scripts/evaluate_selective_prediction.py "
            "--config params.yaml --split val first."
        )

    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in deployment decision: {decision_path}") from error

    if not isinstance(decision, dict):
        raise ValueError(f"Deployment decision must be a JSON object: {decision_path}")

    if "threshold" not in decision:
        raise ValueError(f"Deployment decision is missing 'threshold': {decision_path}")

    if decision.get("selection_split") not in (None, "val"):
        raise ValueError(
            "Error-analysis threshold must be selected from validation, "
            f"got selection_split={decision.get('selection_split')!r}."
        )

    raw_threshold = decision["threshold"]
    if isinstance(raw_threshold, bool):
        raise ValueError("Deployment threshold must be numeric, not boolean.")

    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid deployment threshold: {raw_threshold!r}") from error

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Deployment threshold must be between 0 and 1, got {threshold}.")

    return threshold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run error and subgroup analysis for calibrated test predictions.")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Project configuration YAML.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/metadata.csv"),
        help="Metadata CSV joined to test predictions by image_id.",
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=50,
        help="Maximum number of high-confidence errors to export.",
    )
    args = parser.parse_args()

    if args.top_k_errors < 1:
        raise ValueError("--top-k-errors must be at least 1.")

    cfg = load_yaml(args.config)
    paths = get_project_paths(cfg)
    labels = [str(label) for label in cfg.get("data", {}).get("labels", DEFAULT_LABELS)]

    calibration_dir = paths.calibration_dir
    predictions_path = calibration_dir / "calibrated_test_predictions.csv"

    decision_path = calibration_dir / "decision.json"
    threshold = load_selected_threshold(decision_path)

    pred_df = load_predictions(predictions_path)
    meta_df = load_metadata(args.metadata_csv)

    df = pred_df.merge(
        meta_df,
        on="image_id",
        how="left",
        suffixes=("", "_meta"),
    )
    df = add_age_bucket(df)

    # Apply the deployment threshold selected only from validation data.
    df["accepted"] = df["calibrated_confidence"] >= threshold
    df["needs_review"] = ~df["accepted"]

    output_dir = paths.error_analysis_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = "test"

    overall_metrics = compute_basic_metrics(
        df["true_label"].astype(str),
        df["calibrated_pred_label"].astype(str),
        labels,
    )

    accepted_df = df[df["accepted"]].copy()
    accepted_metrics = compute_basic_metrics(
        accepted_df["true_label"].astype(str),
        accepted_df["calibrated_pred_label"].astype(str),
        labels,
    )

    coverage = float(len(accepted_df) / len(df)) if len(df) else 0.0
    needs_review_rate = float(df["needs_review"].mean()) if len(df) else 0.0

    metrics_df = pd.DataFrame(
        [
            {
                "split": split,
                "subset": "all",
                "threshold": threshold,
                **overall_metrics,
            },
            {
                "split": split,
                "subset": "accepted_only",
                "threshold": threshold,
                **accepted_metrics,
                "coverage": coverage,
                "needs_review_rate": needs_review_rate,
            },
        ]
    )
    metrics_df.to_csv(
        output_dir / "test_overall_and_accepted_metrics.csv",
        index=False,
    )

    # Per-class metrics.
    per_class_df = per_class_metrics(df, labels)
    per_class_df.to_csv(
        output_dir / "test_per_class_metrics.csv",
        index=False,
    )

    accepted_per_class_df = per_class_metrics(accepted_df, labels)
    accepted_per_class_df.to_csv(
        output_dir / "test_accepted_per_class_metrics.csv",
        index=False,
    )

    # Confusion matrices.
    cm_df, cm_norm_df = confusion_outputs(df, labels)
    cm_df.to_csv(output_dir / "test_confusion_matrix.csv")
    cm_norm_df.to_csv(output_dir / "test_confusion_matrix_normalized.csv")

    # Highest-confidence incorrect predictions.
    top_err_df = top_errors(
        df,
        top_k=args.top_k_errors,
    )
    top_err_df.to_csv(
        output_dir / "test_top_high_confidence_errors.csv",
        index=False,
    )

    # Focused melanoma / benign  confusion analysis.
    mel_bkl_df = mel_bkl_analysis(df)
    mel_bkl_df.to_csv(
        output_dir / "test_mel_bkl_confusion_analysis.csv",
        index=False,
    )

    # Subgroup metrics.
    for group_col in [
        "sex",
        "age_bucket",
        "anatom_site",
        "dx_type",
    ]:
        subgroup_df = subgroup_metrics(
            df,
            group_col,
            labels,
        )
        if not subgroup_df.empty:
            subgroup_df.to_csv(
                output_dir / f"test_subgroup_metrics_by_{group_col}.csv",
                index=False,
            )

    summary = make_summary(
        overall_metrics=overall_metrics,
        per_class_df=per_class_df,
        top_errors_df=top_err_df,
        threshold=threshold,
    )
    (output_dir / "test_error_analysis_summary.md").write_text(
        summary,
        encoding="utf-8",
    )

    print(f"[DONE] Predictions: {predictions_path}")
    print(f"[DONE] Deployment decision: {decision_path}")
    print(f"[DONE] Selected threshold: {threshold:.4f}")
    print(f"[DONE] Wrote error analysis outputs to: {output_dir}")
    print(f"[DONE] Overall macro-F1: {overall_metrics['macro_f1']:.4f}")

    accepted_macro_f1 = accepted_metrics["macro_f1"]
    if accepted_macro_f1 is None:
        print("[DONE] Accepted-only macro-F1: N/A (no accepted predictions)")
    else:
        print(f"[DONE] Accepted-only macro-F1 @ {threshold:.4f}: {accepted_macro_f1:.4f}")

    print(f"[DONE] Needs-review rate @ {threshold:.4f}: {needs_review_rate:.4f}")


if __name__ == "__main__":
    main()
