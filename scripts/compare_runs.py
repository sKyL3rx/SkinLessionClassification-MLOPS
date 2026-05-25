from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare archived skin-lesion experiment runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="artifacts/runs",
        help=(
            "Directory containing archived runs. Supports either "
            "artifacts/runs/<exp>/reports/*.json or artifacts/reports/runs/<run>/*.json."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="artifacts/reports/experiment_summary.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="artifacts/reports/experiment_summary.md",
        help="Output Markdown table path.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="best_val_macro_f1",
        help="Column used to sort experiments.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}

def find_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")
    
    run_dirs = []

    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir():
            continue

        has_report_subdir = (path / "reports").exists()

        has_direct_metrics = any(
            (path / name).exists()
            for name in [
                "train_metrics.json",
                "test_metrics.json",
                "per_class_metrics.csv",
                "confusion_matrix.csv",
            ]
        )

        if has_report_subdir or has_direct_metrics:
            run_dirs.append(path)

    return run_dirs

def metric_file(run_dir: Path, filename: str) -> Path:
    
    path = run_dir / "reports" / filename

    if path.exists():
        return path
    else:
        raise FileNotFoundError(f"File named {filename} under {run_dir} not found")

def params_file(run_dir: Path, filename: str = "params.yaml") -> Path:
    
    path = run_dir / filename

    if path.exists():
        return path
    else:
        raise FileNotFoundError(f"File named {filename} under {run_dir} not found")

def load_json(path: Path) -> dict[str,Any]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}

def extract_history_metrics(history_path: Path) -> dict[str,Any]:
    if not history_path.exists():
        return {}

    df = pd.read_csv(history_path)

    if df.empty:
        return {}
    
    out: dict[str, Any] = {}

    if "val_macro_f1" in df.columns:
        best_idx = df["val_macro_f1"].idxmax()
        best_row = df.loc[best_idx]
        out["best_epoch_from_history"] = int(best_row["epoch"])
        out["best_val_macro_f1_from_history"] = float(best_row["val_macro_f1"])
    
    if "train_macro_f1" in df.columns:
        out["final_train_macro_f1"] = float(df.iloc[-1]["train_macro_f1"])

    if "val_macro_f1" in df.columns:
        out["final_val_macro_f1"] = float(df.iloc[-1]["val_macro_f1"])

    if "lr" in df.columns:
        out["final_lr"] = float(df.iloc[-1]["lr"])

    return out

def extract_per_class_metrics(per_class_path: Path) -> dict[str, Any]:

    if not per_class_path.exists():
        return {}
    
    df = pd.read_csv(per_class_path)

    if df.empty:
        return {}
    
    columns_l = {str(col).lower(): col for col in df.columns}

    class_col = columns_l.get("class") or df.columns[0]
    precision_col = columns_l.get("precision")
    recall_col = columns_l.get("recall")
    f1_col = columns_l.get("f1")
    support_col = columns_l.get("support")

    out: dict[str, Any] = {}

    label_series = df[class_col].astype(str).str.lower().str.strip()

    for label in LABELS:
        matched = df[label_series == label]

        if matched.empty:
            continue

        row = matched.iloc[0]

        if precision_col is not None:
            out[f"{label}_precision"] = float(row[precision_col])

        if recall_col is not None:
            out[f"{label}_recall"] = float(row[recall_col])

        if f1_col is not None:
            out[f"{label}_f1"] = float(row[f1_col])

        if support_col is not None:
            out[f"{label}_support"] = int(row[support_col])

    return out

def extract_confusion_metrics(confusion_path: Path) -> dict[str, Any]:
    if not confusion_path.exists():
        return {}

    df = pd.read_csv(confusion_path, index_col=0)

    if df.empty:
        return {}

    df.index = df.index.astype(str).str.lower().str.strip()
    df.columns = [str(col).lower().strip() for col in df.columns]

    out: dict[str, Any] = {}

    for true_label, pred_label in [
        ("mel", "nv"),
        ("mel", "bkl"),
        ("mel", "bcc"),
        ("bcc", "nv"),
        ("bcc", "bkl"),
        ("akiec", "bcc"),
        ("akiec", "bkl"),
    ]:
        if true_label in df.index and pred_label in df.columns:
            out[f"{true_label}_as_{pred_label}"] = int(df.loc[true_label, pred_label])
        top_pair = None
    top_count = -1

    for true_label in df.index:
        for pred_label in df.columns:
            if true_label == pred_label:
                continue

            count = int(df.loc[true_label, pred_label])

            if count > top_count:
                top_count = count
                top_pair = f"{true_label}->{pred_label}"

    if top_pair is not None:
        out["top_confusion_pair"] = top_pair
        out["top_confusion_count"] = top_count

    return out

def build_row(run_dir: Path) -> dict[str,Any]:
    train_metrics_path = metric_file(run_dir, "train_metrics.json")
    test_metrics_path = metric_file(run_dir, "test_metrics.json")
    per_class_path = metric_file(run_dir, "per_class_metrics.csv")
    confusion_path = metric_file(run_dir, "confusion_matrix.csv")
    history_path = metric_file(run_dir, "train_history.csv")

    params_path = params_file(run_dir, "params.yaml")

    train_metrics = load_json(train_metrics_path)
    test_metrics = load_json(test_metrics_path)
    params = load_yaml(params_path)

    project_cfg = params.get("project", {})
    train_cfg = params.get("train", {})

    experiment_name = params.get("experiment_name", {})

    loss_type = train_cfg.get("loss", {})

    class_weights = train_cfg.get("class_weights", {})

    weighted_sampler = train_cfg.get("weighted_sampler", {})

    row: dict[str,Any] = {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),

        "has_train_metrics": train_metrics_path.exists(),
        "has_test_metrics": test_metrics_path.exists(),
        "has_per_class_metrics": per_class_path.exists(),
        "has_confusion_matrix": confusion_path.exists(),
        "has_train_history": history_path.exists(),
        "has_params": params_path.exists(),

        "backbone": train_cfg.get("backbone", {}),
        "loss": loss_type,
        "class_weights": class_weights,
        "weighted_sampler": weighted_sampler,
        "label_smoothing": train_cfg.get("label_smoothing"),
        "scheduler" : train_cfg.get("scheduler", {}),
        "lr": train_cfg.get("lr"),
        "weight_decay": train_cfg.get("weight_decay"),
        "epochs_requested":train_cfg.get("epochs",{}),

        "epochs_completed": train_metrics.get("epochs_completed"),
        "stopped_early": train_metrics.get("stopped_early"),
        "early_stop_epoch": train_metrics.get("early_stop_epoch"),

         # Model selection
        "selection_metric": train_metrics.get("selection_metric", "val_macro_f1"),
        "best_val_macro_f1": train_metrics.get("best_val_macro_f1"),

        # Test metrics
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_weighted_f1": test_metrics.get("weighted_f1"),
        "balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "accuracy": test_metrics.get("accuracy"),
        "mean_confidence": test_metrics.get("mean_confidence"),
        "needs_review_rate": test_metrics.get("needs_review_rate"),
        "confidence_threshold": test_metrics.get("confidence_threshold"),
        "checkpoint_epoch": test_metrics.get("checkpoint_epoch"),
        "num_test_samples": test_metrics.get("num_samples"),
        "run_name": test_metrics.get("run_name"),
        "evaluation_split": test_metrics.get("evaluation_split"),
    }


    row.update(extract_history_metrics(history_path))
    row.update(extract_per_class_metrics(per_class_path))
    row.update(extract_confusion_metrics(confusion_path))

    return row


def format_cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)



def write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    display_cols = [
        "experiment_name",
        "backbone",
        "loss",
        "class_weights",
        "weighted_sampler",
        "label_smoothing",
        "best_val_macro_f1",
        "test_macro_f1",
        "balanced_accuracy",
        "accuracy",
        "mel_recall",
        "bcc_recall",
        "akiec_recall",
        "needs_review_rate",
        "checkpoint_epoch",
        "top_confusion_pair",
        "top_confusion_count",
    ]

    cols = [col for col in display_cols if col in df.columns]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df[cols].iterrows():
        values = [format_cell(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")





def main() -> None:
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    run_dirs = find_run_dirs(runs_dir)

    if not run_dirs:
        raise RuntimeError(f"No valid run directories found under: {runs_dir}")
    
    rows = []

    for run_dir in run_dirs:
        row = build_row(run_dir)

        rows.append(row)

    df = pd.DataFrame(rows)
    sort_by = args.sort_by

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False, na_position="last")
    else:
        print(f"[WARN] Sort column not found: {sort_by}. Keeping original order.")

    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    write_markdown_table(df, out_md)

    preview_cols = [
        "experiment_name",
        "backbone",
        "loss",
        "class_weights",
        "weighted_sampler",
        "best_val_macro_f1",
        "test_macro_f1",
        "balanced_accuracy",
        "accuracy",
        "mel_recall",
        "bcc_recall",
        "needs_review_rate",
        "top_confusion_pair",
        "top_confusion_count",
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]

    print(f"[INFO] Compared {len(df)} run(s).")
    print(f"[INFO] Wrote CSV: {out_csv}")
    print(f"[INFO] Wrote Markdown: {out_md}")
    print()
    print(df[preview_cols].to_string(index=False))

if __name__ == "__main__":
    main()



