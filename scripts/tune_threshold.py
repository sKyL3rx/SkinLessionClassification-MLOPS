from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CRITICAL_LABELS = ["mel", "bcc", "akiec"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep confidence thresholds for triage/abstention."
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        required=True,
        help="CSV with true_label, pred_label, confidence, and prob_* columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/reports/threshold_calibration",
        help="Output directory.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--target-mel-recall",
        type=float,
        default = 0.90,
    )
    parser.add_argument(
        "--max-review-rate",
        type=float,
        default=0.35,
        help="Preferred maximum needs_review rate for selecting threshold.",
    )
    
    return parser.parse_args()

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path)

    required = ["true_label", "pred_label", "confidence"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df["true_label"] = df["true_label"].astype("str").str.lower().str.strip()
    df["pred_label"] = df["pred_label"].astype(str).str.lower().str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors = "coerce")

    if df["confidence"].isna().any():
        raise ValueError("Some confidence values could not be parsed as numeric.")

    return df

def compute_threshold_metrics(df: pd.DataFrame, threshold: float) -> dict[str,Any]:
    needs_review = df["confidence"] < threshold
    covered = ~needs_review

    covered_df = df[covered]
    reviewed_df = df[needs_review]

    num_samples = len(df)
    num_covered = len(covered_df)
    num_reviewed = len(reviewed_df)


    row: dict[str, Any] = {
        "threshold": float(threshold),
        "num_samples": int(num_samples),
        "num_covered": int(num_covered),
        "num_reviewed": int(num_reviewed),
        "coverage": float(num_covered / num_samples) if num_samples else 0.0,
        "needs_review_rate": float(num_reviewed / num_samples) if num_samples else 0.0,
    }

    if num_covered > 0:
        row["accuracy_on_covered"] = float(
            accuracy_score(covered_df["true_label"], covered_df["pred_label"])
        )
        row["macro_f1_on_covered"] = float(
            f1_score(
                covered_df["true_label"],
                covered_df["pred_label"],
                average="macro",
                zero_division=0,
            )
        )
    else:
        row["accuracy_on_covered"] = np.nan
        row["macro_f1_on_covered"] = np.nan

    for label in CRITICAL_LABELS:
        class_df = df[df["true_label"] == label]

        if len(class_df) == 0:
            row[f"{label}_support"] = 0
            row[f"{label}_classifier_recall_all"] = np.nan
            row[f"{label}_review_rate"] = np.nan
            row[f"{label}_triage_recall"] = np.nan
            continue

        true_positive_predictions = (
            (class_df["pred_label"] == label)
            & (~needs_review[class_df.index])
        ).sum()

        reviewed_positives = needs_review[class_df.index].sum()

        triage_caught = true_positive_predictions + reviewed_positives
        # Model can predict true label w/ high confidence + wrong/ right reviewed by experts

        row[f"{label}_support"] = int(len(class_df))
        row[f"{label}_classifier_recall_all"] = (
                float((class_df["pred_label"] == label).sum())
                / len(class_df)
                )

        row[f"{label}_review_rate"] = float(reviewed_positives / len(class_df))

        row[f"{label}_triage_recall"] = float(triage_caught / len(class_df))

    # confident >= threshold prediction but wrong
    wrong_covered = covered_df[covered_df["true_label"] != covered_df["pred_label"]]

    row["covered_error_count"] = int(len(wrong_covered))
    row["covered_error_rate"] = (
    float(len(wrong_covered) / len(covered_df)) if len(covered_df) else 0.0
    )

    row["critical_miss_count"] = int(
        len(wrong_covered[wrong_covered["true_label"].isin(CRITICAL_LABELS)])
    )

    mel_as_nv = wrong_covered[
        (wrong_covered["true_label"] == "mel") & (wrong_covered["pred_label"] == "nv")
    ]
    row["mel_as_nv_count"] = int(len(mel_as_nv))

    return row

def select_threshold(
    sweep_df: pd.DataFrame,
    target_mel_recall: float,
    max_review_rate: float,
) -> dict[str, Any]:
    candidates = sweep_df[
        (sweep_df["mel_triage_recall"] >= target_mel_recall)
        & (sweep_df["needs_review_rate"] <= max_review_rate)
    ].copy()

    selection_reason = (
        f"mel_triage_recall >= {target_mel_recall} and "
        f"needs_review_rate <= {max_review_rate}; maximize macro_f1_on_covered"
    )

    if candidates.empty:
        candidates = sweep_df[sweep_df["mel_triage_recall"] >= target_mel_recall].copy()
        selection_reason = (
            f"No threshold met max_review_rate <= {max_review_rate}; "
            f"using mel_triage_recall >= {target_mel_recall} and lowest review rate"
        )
    
    if candidates.empty:
        candidates = sweep_df.copy()
        selection_reason = (
            f"No threshold met mel_triage_recall >= {target_mel_recall}; "
            "using lowest critical_miss_count then highest macro_f1_on_covered"
        )
        candidates = candidates.sort_values(
            ["critical_miss_count", "macro_f1_on_covered", "coverage"],
            ascending=[True, False, False],
        )
    else:
        candidates = candidates.sort_values(
            ["macro_f1_on_covered", "needs_review_rate", "coverage"],
            ascending=[False, True, False],
        )
    
    best = candidates.iloc[0].to_dict()
    return {
        "selected_threshold": float(best["threshold"]),
        "selection_reason": selection_reason,
        "selected_metrics": best,
    }

def main() -> None:
    args = parse_args()

    predictions_path = Path(args.predictions_csv)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_path)

    thresholds = np.arange(
        args.min_threshold, 
        args.max_threshold + args.step / 2,
        args.step,
    )

    rows = [compute_threshold_metrics(df, float(t)) for t in thresholds]
    sweep_df = pd.DataFrame(rows)

    selected = select_threshold(
        sweep_df=sweep_df,
        target_mel_recall=args.target_mel_recall,
        max_review_rate=args.max_review_rate,
    )

    sweep_path = out_dir / "threshold_sweep.csv"
    selected_path = out_dir / "selected_threshold.json"

    sweep_df.to_csv(sweep_path, index=False)

    payload = {
        "predictions_csv": str(predictions_path),
        "target_mel_recall": float(args.target_mel_recall),
        "max_review_rate": float(args.max_review_rate),
        **selected,
    }

    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Wrote threshold sweep to: {sweep_path}")
    print(f"[INFO] Wrote selected threshold to: {selected_path}")
    print()
    print("[INFO] Selected threshold:")
    print(json.dumps(payload, indent=2))
    print()
    print("[INFO] Sweep preview:")
    preview_cols = [
        "threshold",
        "coverage",
        "needs_review_rate",
        "accuracy_on_covered",
        "macro_f1_on_covered",
        "mel_triage_recall",
        "mel_review_rate",
        "critical_miss_count",
        "mel_as_nv_count",
    ]
    preview_cols = [col for col in preview_cols if col in sweep_df.columns]
    print(sweep_df[preview_cols].to_string(index=False))

if __name__ == "__main__":
    main()

