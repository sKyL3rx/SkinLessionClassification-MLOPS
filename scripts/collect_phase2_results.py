from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"[WARN] Could not read YAML: {path} ({exc})")
        return {}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] Could not read JSON: {path} ({exc})")
        return {}


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    return out


def summarize_config(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)

    project = cfg.get("project", {})
    data = cfg.get("data", {})
    train = cfg.get("train", {})
    preprocess = cfg.get("preprocess", {})
    augment = cfg.get("augment", {})
    evaluate = cfg.get("evaluate", {})

    experiment_name = project.get("experiment_name", config_path.stem)

    return {
        "config_path": str(config_path),
        "experiment": experiment_name,
        "backbone": train.get("backbone"),
        "image_size": data.get("image_size"),
        "use_metadata": train.get("use_metadata"),
        "fusion_type": train.get("fusion_type"),
        "metadata_dim": train.get("metadata_dim"),
        "freeze_backbone": train.get("freeze_backbone"),
        "batch_size": train.get("batch_size"),
        "epochs": train.get("epochs"),
        "lr": train.get("lr"),
        "backbone_lr_mult": train.get("backbone_lr_mult"),
        "head_lr_mult": train.get("head_lr_mult"),
        "loss": train.get("loss"),
        "resize_mode": preprocess.get("resize_mode"),
        "preprocess_mode": preprocess.get("mode"),
        "use_random_resized_crop": augment.get("use_random_resized_crop"),
        "use_tta": evaluate.get("use_tta"),
    }


def read_train_summary(run_dir: Path) -> dict[str, Any]:
    train_metrics_file = first_existing(
        [
            run_dir / "train_metrics.json",
            run_dir / "reports" / "train_metrics.json",
            run_dir / "summary.json",
            run_dir / "reports" / "summary.json",
        ]
    )

    if train_metrics_file is None:
        print(f"[WARN] Missing train metrics in: {run_dir}")
        return {
            "best_val_macro_f1": "",
            "checkpoint_source": "",
            "selection_metric": "",
            "epochs_completed": "",
            "stopped_early": "",
            "train_metrics_path": "",
        }

    data = load_json(train_metrics_file)

    return {
        "best_val_macro_f1": data.get("best_val_macro_f1", ""),
        "checkpoint_source": data.get("checkpoint_source", data.get("best_checkpoint_source", "")),
        "selection_metric": data.get("selection_metric", ""),
        "epochs_completed": data.get("epochs_completed", ""),
        "stopped_early": data.get("stopped_early", ""),
        "train_metrics_path": str(train_metrics_file),
    }


def read_eval_metrics(run_dir: Path, split: str) -> dict[str, Any] | None:
    eval_metrics_file = first_existing(
        [
            run_dir / "reports" / f"{split}_metrics.json",
            run_dir / f"{split}_metrics.json",
        ]
    )

    if eval_metrics_file is None:
        print(f"[WARN] Missing {split} metrics in: {run_dir}")
        return None

    data = load_json(eval_metrics_file)

    return {
        f"{split}_accuracy": data.get("accuracy", ""),
        f"{split}_macro_f1": data.get("macro_f1", ""),
        f"{split}_weighted_f1": data.get("weighted_f1", ""),
        f"{split}_balanced_accuracy": data.get("balanced_accuracy", ""),
        f"{split}_mean_confidence": data.get("mean_confidence", ""),
        f"{split}_needs_review_rate": data.get("needs_review_rate", ""),
        f"{split}_num_samples": data.get("num_samples", ""),
        f"{split}_checkpoint_source": data.get("checkpoint_source", ""),
        f"{split}_best_val_macro_f1_from_checkpoint": data.get("best_val_macro_f1", ""),
        f"{split}_metrics_path": str(eval_metrics_file),
    }


def read_per_class_metrics(run_dir: Path, split: str) -> dict[str, Any] | None:
    per_class_metrics_file = first_existing(
        [
            run_dir / "reports" / f"{split}_per_class_metrics.csv",
            run_dir / f"{split}_per_class_metrics.csv",
        ]
    )

    if per_class_metrics_file is None:
        print(f"[WARN] Missing {split} per-class metrics in: {run_dir}")
        return None

    try:
        df = normalize_columns(pd.read_csv(per_class_metrics_file))
    except Exception as exc:
        print(f"[WARN] Could not read per-class CSV: {per_class_metrics_file} ({exc})")
        return {}

    required_cols = {"class", "recall", "precision", "f1", "support"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        print(
            f"[WARN] Missing columns in per-class CSV: "
            f"{per_class_metrics_file} missing={sorted(missing_cols)}"
        )
        return {}

    out: dict[str, Any] = {}
    df["class"] = df["class"].astype(str)

    for label in LABELS:
        row = df[df["class"] == label]
        if row.empty:
            continue

        row0 = row.iloc[0]

        out[f"{label}_recall"] = float(row0["recall"])
        out[f"{label}_f1"] = float(row0["f1"])
        out[f"{label}_precision"] = float(row0["precision"])
        out[f"{label}_support"] = int(row0["support"])

    out["per_class_metrics_path"] = str(per_class_metrics_file)
    return out


def make_markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)


def round_numeric_columns(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def add_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out


def fill_checkpoint_fields_from_eval(
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Fill checkpoint fields when train metrics are missing.
    """
    if is_missing(row.get("checkpoint_source")):
        if not is_missing(row.get("val_checkpoint_source")):
            row["checkpoint_source"] = row.get("val_checkpoint_source")
        elif not is_missing(row.get("test_checkpoint_source")):
            row["checkpoint_source"] = row.get("test_checkpoint_source")

    if is_missing(row.get("best_val_macro_f1")):
        if not is_missing(row.get("val_best_val_macro_f1_from_checkpoint")):
            row["best_val_macro_f1"] = row.get("val_best_val_macro_f1_from_checkpoint")
        elif not is_missing(row.get("test_best_val_macro_f1_from_checkpoint")):
            row["best_val_macro_f1"] = row.get("test_best_val_macro_f1_from_checkpoint")

    return row


def collect_results(
    configs_dir: Path,
    runs_dir: Path,
    output_dir: Path,
    include_stage1: bool = False,
) -> pd.DataFrame:
    config_paths = sorted(configs_dir.glob("*.yaml")) + sorted(configs_dir.glob("*.yml"))

    if not config_paths:
        raise FileNotFoundError(f"No YAML configs found in {configs_dir}")

    rows: list[dict[str, Any]] = []

    for config_path in config_paths:
        row = summarize_config(config_path)
        exp = str(row["experiment"])

        if not include_stage1 and "stage1" in exp:
            print(f"[INFO] Skipping intermediate Stage1 run: {exp}")
            continue

        run_dir = runs_dir / exp

        row["run_dir"] = str(run_dir)
        row["run_exists"] = run_dir.exists()
        row["best_ckpt_exists"] = (run_dir / "checkpoints" / "best.ckpt").exists()

        if not run_dir.exists():
            print(f"[WARN] Missing run dir for {exp}: {run_dir}")
            rows.append(row)
            continue

        row.update(read_train_summary(run_dir))

        val_metrics_loaded = read_eval_metrics(run_dir, split="val")
        if val_metrics_loaded is not None:
            row.update(val_metrics_loaded)

        test_metrics_loaded = read_eval_metrics(run_dir, split="test")
        if test_metrics_loaded is not None:
            row.update(test_metrics_loaded)

        test_per_class_loaded = read_per_class_metrics(run_dir, split="test")
        if test_per_class_loaded is not None:
            row.update(test_per_class_loaded)

        row = fill_checkpoint_fields_from_eval(row)
        rows.append(row)

    df = pd.DataFrame(rows)

    always_show_cols = [
        "experiment",
        "backbone",
        "image_size",
        "use_metadata",
        "fusion_type",
        "checkpoint_source",
        "best_val_macro_f1",
        "val_macro_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
        "mel_recall",
        "akiec_recall",
        "test_needs_review_rate",
    ]
    df = add_missing_columns(df, always_show_cols)

    preferred_cols = [
        "experiment",
        "backbone",
        "image_size",
        "use_metadata",
        "fusion_type",
        "freeze_backbone",
        "resize_mode",
        "use_random_resized_crop",
        "use_tta",
        "best_val_macro_f1",
        "checkpoint_source",
        "selection_metric",
        "val_macro_f1",
        "val_balanced_accuracy",
        "val_accuracy",
        "val_weighted_f1",
        "val_mean_confidence",
        "val_needs_review_rate",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_weighted_f1",
        "test_mean_confidence",
        "test_needs_review_rate",
        "akiec_recall",
        "bcc_recall",
        "bkl_recall",
        "df_recall",
        "mel_recall",
        "nv_recall",
        "vasc_recall",
        "akiec_f1",
        "bcc_f1",
        "bkl_f1",
        "df_f1",
        "mel_f1",
        "nv_f1",
        "vasc_f1",
        "best_ckpt_exists",
        "run_exists",
        "config_path",
        "run_dir",
        "train_metrics_path",
        "val_metrics_path",
        "test_metrics_path",
        "per_class_metrics_path",
    ]

    existing_preferred = [c for c in preferred_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_preferred]
    df = df[existing_preferred + remaining]

    # Sort for readability and model selection.
    if "best_val_macro_f1" in df.columns:
        df["_sort_best_val"] = pd.to_numeric(df["best_val_macro_f1"], errors="coerce")
        df = df.sort_values(
            by="_sort_best_val",
            ascending=False,
            na_position="last",
        ).drop(columns=["_sort_best_val"])

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "experiment_summary.csv"
    compact_md_path = output_dir / "experiment_summary.md"
    full_md_path = output_dir / "experiment_summary_full.md"

    # Full CSV
    df.to_csv(csv_path, index=False)

    df_for_md = df.copy().fillna("")

    # Full Markdown table.
    full_report_cols = [
        "experiment",
        "backbone",
        "image_size",
        "use_metadata",
        "fusion_type",
        "checkpoint_source",
        "best_val_macro_f1",
        "val_macro_f1",
        "val_balanced_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_accuracy",
        "test_weighted_f1",
        "mel_recall",
        "akiec_recall",
        "test_needs_review_rate",
    ]
    full_report_cols = [c for c in full_report_cols if c in df_for_md.columns]
    full_report_df = round_numeric_columns(df_for_md[full_report_cols])

    full_markdown = "# Full Experiment Summary\n\n"
    full_markdown += make_markdown_table(full_report_df)
    full_markdown += "\n\n"
    full_markdown += "Notes:\n"
    full_markdown += "- `best_val_macro_f1` is used for model selection when available.\n"
    full_markdown += (
        "- `val_macro_f1` is the validation metric from the saved best checkpoint evaluation.\n"
    )
    full_markdown += (
        "- Test metrics should be treated as final evaluation, not for choosing hyperparameters.\n"
    )
    full_markdown += (
        "- `checkpoint_source` indicates whether the best checkpoint "
        "came from raw or EMA weights.\n"
    )
    full_md_path.write_text(full_markdown, encoding="utf-8")

    # Compact Markdown table for README / preview.
    compact_cols = [
        "experiment",
        "backbone",
        "use_metadata",
        "fusion_type",
        "val_macro_f1",
        "test_macro_f1",
        "test_balanced_accuracy",
        "mel_recall",
        "akiec_recall",
    ]
    compact_cols = [c for c in compact_cols if c in df_for_md.columns]
    compact_df = round_numeric_columns(df_for_md[compact_cols])

    compact_markdown = "# Experiment Summary\n\n"
    compact_markdown += make_markdown_table(compact_df)
    compact_markdown += "\n\n"
    compact_markdown += (
        "- See `experiment_summary.csv` for checkpoint source, confidence, "
        "review-rate, and full metrics.\n"
    )
    compact_md_path.write_text(compact_markdown, encoding="utf-8")

    print(f"[OK] Wrote CSV: {csv_path}")
    print(f"[OK] Wrote compact Markdown: {compact_md_path}")
    print(f"[OK] Wrote full Markdown: {full_md_path}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Phase 2 experiment metrics into CSV and Markdown summaries."
    )

    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs/experiments"),
        help="Directory containing phase2 YAML configs.",
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Directory containing run artifacts.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Output directory for summary files.",
    )

    parser.add_argument(
        "--include-stage1",
        action="store_true",
        help="Include intermediate Stage1 runs in the summary table.",
    )

    args = parser.parse_args()

    collect_results(
        configs_dir=args.configs_dir,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        include_stage1=args.include_stage1,
    )


if __name__ == "__main__":
    main()
