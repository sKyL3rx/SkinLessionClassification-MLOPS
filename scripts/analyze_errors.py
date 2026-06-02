from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CRITICAL_LABELS = ["mel", "bcc", "akiec"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze model errors from archived prediction artifacts."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Archived run directory, e.g. artifacts/runs/convnext_tiny_ep50_recipe_cw.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="artifacts/reports/experiment_summary.csv",
        help="Experiment summary CSV. Used to auto-select best run if --run-dir is not provided.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/reports/error_analysis",
        help="Output directory for error analysis reports.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="best_val_macro_f1",
        help="Metric used to auto-select best run from experiment_summary.csv.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override confidence threshold. If omitted, read from test_metrics.json when possible.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top examples to save for ranked reports.",
    )
    return parser.parse_args()

def auto_select_run(summary_csv: Path, sort_by: str) -> Path:
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"Missing {summary_csv}. Provide --run-dir or run scripts/compare_runs.py first."
        )
    
    df = pd.read_csv(summary_csv)

    if df.empty:
        raise RuntimeError(f"Empty summary file: {summary_csv}")
    
    if sort_by not in df.columns:
        raise ValueError(f"Sort column '{sort_by}' not found in {summary_csv}")


    df = df.sort_values(by=sort_by, ascending=False, na_position="last")

    if "run_dir" in df.columns and pd.notna(df.iloc[0]["run_dir"]):
        return Path(str(df.iloc[0]["run_dir"]))

    if "experiment_name" not in df.columns:
        raise ValueError("experiment_summary.csv must contain either run_dir or experiment_name.")

    return Path("artifacts/runs") / str(df.iloc[0]["experiment_name"])

def metric_file(run_dir: Path, filename: str) -> Path:
    candidates = [
        run_dir / filename,
        run_dir / "reports" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"File named {filename} under {run_dir} not found. Tried:\n{tried}"
    )

def params_file(run_dir: Path, filename: str = "params.yaml") -> Path:
    candidates = [
        run_dir / filename,
        run_dir / "reports" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"File named {filename} under {run_dir} not found. Tried:\n{tried}"
    )

def load_json(path: Path) -> dict[str,Any]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)

def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.astype("str").str.lower().str.strip()

def load_predictions(predictions_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    df = pd.read_csv(predictions_path)

    if df.empty:
        raise RuntimeError(f"Predictions file is empty: {predictions_path}")

    true_col = "true_label"
    pred_col = "pred_label" 
    confidence_col = "confidence"

    image_col = "image_path"

    needs_review_col = "needs_review"

    columns = {
        "true": true_col,
        "pred": pred_col,
        "confidence": confidence_col,
    }

    if image_col is not None:
        columns["image"] = image_col
    
    if needs_review_col is not None:
        columns["needs_review"] = needs_review_col
    
    df["_true"] = normalize_label_series(df[true_col])
    df["_pred"] = normalize_label_series(df[pred_col])
    df["_confidence"] = pd.to_numeric(df[confidence_col], errors="coerce")
    df["_correct"] = df["_true"] == df["_pred"]

    if needs_review_col is not None:
        df["_needs_review"] = df[needs_review_col].astype(bool)
    else:
        df["_needs_review"] = False
    
    return df,columns

def build_per_class_error_summary(df: pd.DataFrame, confidence_threshold: float) -> pd.DataFrame:
    
    rows = []

    for label in LABELS:
        subset = df[df["_true"] == label]
        if subset.empty:
            continue
            
        wrong = subset[~subset["_correct"]]
        correct = subset[subset["_correct"]]

        support = len(subset)
        num_correct = len(correct)
        num_wrong = len(wrong)

        recall = num_correct / support if support else 0.0

        high_conf_wrong = wrong[wrong["_confidence"] >= confidence_threshold]
        low_conf = subset[subset["_confidence"] < confidence_threshold]

        most_common_wrong_pred = ""
        most_common_wrong_count = 0

        if not wrong.empty:
            counts = wrong["_pred"].value_counts()
            most_common_wrong_pred = str(counts.index[0])
            most_common_wrong_count = int(counts.iloc[0])
        rows.append(
            {
                "class": label,
                "support": support,
                "correct": num_correct,
                "wrong": num_wrong,
                "recall_from_predictions": recall,
                "avg_confidence": subset["_confidence"].mean(),
                "avg_confidence_correct": correct["_confidence"].mean() if not correct.empty else None,
                "avg_confidence_wrong": wrong["_confidence"].mean() if not wrong.empty else None,
                "low_confidence_count": len(low_conf),
                "low_confidence_rate": len(low_conf) / support if support else 0.0,
                "high_confidence_wrong_count": len(high_conf_wrong),
                "high_confidence_wrong_rate": len(high_conf_wrong) / support if support else 0.0,
                "most_common_wrong_pred": most_common_wrong_pred,
                "most_common_wrong_count": most_common_wrong_count,
            }
        )

    return pd.DataFrame(rows).sort_values("recall_from_predictions", ascending=True)

def load_confusion_pairs(confusion_path: Path) -> pd.DataFrame:
    if not confusion_path.exists():
        return pd.DataFrame(columns=["true_label", "pred_label", "count"])

    cm = pd.read_csv(confusion_path, index_col=0)

    if cm.empty:
        return pd.DataFrame(columns=["true_label", "pred_label", "count"])

    cm.index = cm.index.astype(str).str.lower().str.strip()
    cm.columns = [str(col).lower().strip() for col in cm.columns]

    rows = []

    for true_label in cm.index:
        for pred_label in cm.columns:
            if true_label == pred_label:
                continue

            count = int(cm.loc[true_label, pred_label])

            if count > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": count,
                    }
                )

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame(columns=["true_label", "pred_label", "count"])

    return out.sort_values("count", ascending=False)

def select_columns_for_examples(df: pd.DataFrame, columns: dict[str, str]) -> list[str]:
    preferred = []

    # Core columns 
    for key in ["image", "true", "pred", "confidence", "needs_review"]:
        if key in columns:
            preferred.append(columns[key])

    extra_cols = []

    for col in df.columns:
        col_l = str(col).lower()

        if col in preferred:
            continue

        if col.startswith("_"):
            continue
        # top_1_label, top_1_prob, top_2_label, top_2_prob, top_3_label, top_3_prob
        if col_l.startswith("top_"):
            extra_cols.append(col)

        # Mb future formats like prob_mel or p_mel.
        elif col_l.startswith("prob_") or col_l.startswith("p_") or col_l.endswith("_prob"):
            extra_cols.append(col)

    return preferred + extra_cols
    

def write_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
def main() -> None:
    args = parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        run_dir = auto_select_run(Path(args.summary_csv), args.sort_by)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = metric_file(run_dir, "predictions.csv")
    confusion_path = metric_file(run_dir, "confusion_matrix.csv")
    per_class_path = metric_file(run_dir, "per_class_metrics.csv")
    test_metrics_path = metric_file(run_dir, "test_metrics.json")

    test_metrics = load_json(test_metrics_path)

    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = float(test_metrics.get("confidence_threshold", 0.65))
    
    print(f"[INFO] Run dir: {run_dir}")
    print(f"[INFO] Predictions: {predictions_path}")
    print(f"[INFO] Confidence threshold: {confidence_threshold}")

    df, columns = load_predictions(predictions_path)

    example_cols = select_columns_for_examples(df, columns)

    wrong = df[~df["_correct"]].copy()
    correct = df[df["_correct"]].copy()

    top_confident_errors = wrong.sort_values("_confidence", ascending=False).head(args.top_k)

    low_confidence_predictions = df[df["_confidence"] < confidence_threshold].sort_values(
        "_confidence",
        ascending=True,
    )

    low_confidence_correct = correct[correct["_confidence"] < confidence_threshold].sort_values(
        "_confidence",
        ascending=True
    )
    high_confidence_correct = correct[correct["_confidence"] >= confidence_threshold].sort_values(
        "_confidence",
        ascending=False,
    )

    critical_misses = wrong[wrong["_true"].isin(CRITICAL_LABELS)].sort_values(
        ["_true", "_confidence"],
        ascending = [True, False]
    )

    per_class_summary = build_per_class_error_summary(df, confidence_threshold)

    confusion_pairs = load_confusion_pairs(confusion_path)

    top_confident_errors[example_cols].to_csv(out_dir / "top_confident_errors.csv", index=False)

    low_confidence_predictions[example_cols].to_csv(
        out_dir / "low_confidence_predictions.csv",
        index=False,
    )

    low_confidence_correct[example_cols].head(args.top_k).to_csv(
        out_dir / "low_confidence_correct.csv",
        index=False,
    )

    high_confidence_correct[example_cols].head(args.top_k).to_csv(
        out_dir / "high_confidence_correct.csv",
        index=False,
    )

    critical_misses[example_cols].to_csv(out_dir / "critical_class_misses.csv", index=False)
    per_class_summary.to_csv(out_dir / "per_class_error_summary.csv", index=False)
    confusion_pairs.to_csv(out_dir / "confusion_pairs.csv", index=False)

    if per_class_path.exists():
        per_class_metrics = pd.read_csv(per_class_path)
        per_class_metrics.to_csv(out_dir / "original_per_class_metrics.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "num_samples": int(len(df)),
        "num_correct": int(df["_correct"].sum()),
        "num_wrong": int((~df["_correct"]).sum()),
        "accuracy_from_predictions": float(df["_correct"].mean()),
        "confidence_threshold": confidence_threshold,
        "low_confidence_count": int((df["_confidence"] < confidence_threshold).sum()),
        "low_confidence_rate": float((df["_confidence"] < confidence_threshold).mean()),
        "high_confidence_error_count": int(
            ((~df["_correct"]) & (df["_confidence"] >= confidence_threshold)).sum()
        ),
        "high_confidence_error_rate": float(
            ((~df["_correct"]) & (df["_confidence"] >= confidence_threshold)).mean()
        ),
        "critical_miss_count": int(len(critical_misses)),
        "test_metrics": test_metrics,
        "column_mapping": columns,
    }

    write_json(out_dir / "error_analysis_summary.json", summary)

    print(f"[INFO] Wrote error analysis to: {out_dir}")
    print()
    print("[INFO] Top confusion pairs:")
    if confusion_pairs.empty:
        print("No confusion pairs found.")
    else:
        print(confusion_pairs.head(10).to_string(index=False))

    print()
    print("[INFO] Lowest recall classes:")
    print(
        per_class_summary[
            [
                "class",
                "support",
                "recall_from_predictions",
                "wrong",
                "high_confidence_wrong_count",
                "most_common_wrong_pred",
                "most_common_wrong_count",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()




