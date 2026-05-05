#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LOGGER = logging.getLogger("prepare_ham10000")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000 metadata.")
    parser.add_argument("--config", type=str, default="params.yaml")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def resolve_raw_root(raw_dir: str | Path) -> Path:
    raw_root = Path(raw_dir)

    if (raw_root / "images").exists() and (raw_root / "metadata").exists():
        return raw_root

    ham_subdir = raw_root / "ham10000"
    if (ham_subdir / "images").exists() and (ham_subdir / "metadata").exists():
        return ham_subdir

    raise FileNotFoundError(
        f"Cannot find HAM10000 raw root from '{raw_root}'. "
        "Expected either data/raw/ham10000/{images,metadata} "
        "or data/raw/{ham10000/images, ham10000/metadata}."
    )


def score_metadata_csv(csv_path: Path) -> int:
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return -1

    cols = {normalize_col(c) for c in df.columns}
    score = 0

    if "image_id" in cols or "image" in cols:
        score += 5
    if "dx" in cols or "label" in cols or "class" in cols:
        score += 5
    if "lesion_id" in cols:
        score += 2
    if "dx_type" in cols:
        score += 1
    if "age" in cols:
        score += 1
    if "sex" in cols:
        score += 1
    if "localization" in cols or "anatom_site" in cols or "anatom_site_general" in cols:
        score += 1

    one_hot_labels = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    if len(one_hot_labels.intersection(cols)) >= 2:
        score += 4

    return score


def choose_metadata_csv(metadata_dir: Path) -> Path:
    csv_files = sorted(metadata_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {metadata_dir}")

    scored = [(score_metadata_csv(p), p) for p in csv_files]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_path = scored[0]
    if best_score < 0:
        raise RuntimeError(f"Could not parse any metadata CSV in {metadata_dir}")

    LOGGER.info("Using metadata CSV: %s (score=%s)", best_path.name, best_score)
    return best_path


def normalize_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    mapping = {
        "akiec": "akiec",
        "ak": "akiec",
        "actinic_keratosis": "akiec",
        "actinic_keratoses": "akiec",
        "bcc": "bcc",
        "basal_cell_carcinoma": "bcc",
        "bkl": "bkl",
        "benign_keratosis": "bkl",
        "benign_keratosis_like_lesions": "bkl",
        "df": "df",
        "dermatofibroma": "df",
        "mel": "mel",
        "melanoma": "mel",
        "nv": "nv",
        "melanocytic_nevus": "nv",
        "melanocytic_nevi": "nv",
        "vasc": "vasc",
        "vascular_lesion": "vasc",
        "vascular_lesions": "vasc",
    }
    return mapping.get(raw, raw)


def infer_label_from_one_hot(df: pd.DataFrame) -> pd.Series | None:
    label_cols = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    available = [c for c in label_cols if c in df.columns]
    if len(available) < 2:
        return None

    numeric = df[available].apply(pd.to_numeric, errors="coerce").fillna(0)
    if numeric.sum(axis=1).eq(0).all():
        return None

    return numeric.idxmax(axis=1)


def standardize_metadata(df: pd.DataFrame, allowed_labels: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    # image_id
    if "image_id" in df.columns:
        image_col = "image_id"
    elif "image" in df.columns:
        image_col = "image"
    else:
        raise ValueError("Metadata CSV must contain 'image_id' or 'image' column.")

    out = pd.DataFrame()
    out["image_id"] = df[image_col].astype(str).str.strip()

    # label
    if "dx" in df.columns:
        label_series = df["dx"]
    elif "label" in df.columns:
        label_series = df["label"]
    elif "class" in df.columns:
        label_series = df["class"]
    else:
        label_series = infer_label_from_one_hot(df)
        if label_series is None:
            raise ValueError(
                "Could not infer labels. Expected one of: "
                "dx, label, class, or one-hot label columns."
            )

    out["label"] = label_series.map(normalize_label)

    # optional columns
    if "lesion_id" in df.columns:
        out["lesion_id"] = df["lesion_id"]
    else:
        out["lesion_id"] = pd.NA

    if "dx_type" in df.columns:
        out["dx_type"] = df["dx_type"]
    else:
        out["dx_type"] = pd.NA

    if "age" in df.columns:
        out["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        out["age"] = pd.NA

    if "sex" in df.columns:
        out["sex"] = df["sex"].astype("string").str.lower()
    else:
        out["sex"] = pd.NA

    if "localization" in df.columns:
        out["anatom_site"] = df["localization"].astype("string").str.lower()
    elif "anatom_site_general" in df.columns:
        out["anatom_site"] = df["anatom_site_general"].astype("string").str.lower()
    elif "anatom_site" in df.columns:
        out["anatom_site"] = df["anatom_site"].astype("string").str.lower()
    else:
        out["anatom_site"] = pd.NA

    out["dataset"] = "ham10000"

    allowed = {normalize_label(x) for x in allowed_labels}
    out = out[out["label"].isin(allowed)].copy()

    return out


def build_image_lookup(images_dir: Path) -> dict[str, str]:
    image_lookup: dict[str, str] = {}

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_lookup[path.stem] = path.as_posix()

    if not image_lookup:
        raise FileNotFoundError(f"No image files found under {images_dir}")

    return image_lookup


def attach_image_paths(df: pd.DataFrame, image_lookup: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["image_path"] = df["image_id"].map(image_lookup)

    missing = df["image_path"].isna().sum()
    if missing > 0:
        LOGGER.warning("Dropping %d rows because image files were not found.", missing)

    df = df.dropna(subset=["image_path", "label"]).copy()
    df["split_group"] = df["lesion_id"].fillna(df["image_id"])

    return (
        df[
            [
                "image_id",
                "image_path",
                "label",
                "lesion_id",
                "dx_type",
                "age",
                "sex",
                "anatom_site",
                "dataset",
                "split_group",
            ]
        ]
        .sort_values(["label", "image_id"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    config = load_config(args.config)

    raw_root = resolve_raw_root(config["data"]["raw_dir"])
    metadata_dir = raw_root / "metadata"
    images_dir = raw_root / "images"
    output_csv = Path(config["data"]["metadata_csv"])
    allowed_labels = config["data"]["labels"]

    LOGGER.info("Resolved raw root: %s", raw_root)
    LOGGER.info("Metadata dir: %s", metadata_dir)
    LOGGER.info("Images dir: %s", images_dir)
    LOGGER.info("Output CSV: %s", output_csv)

    metadata_csv = choose_metadata_csv(metadata_dir)
    raw_df = pd.read_csv(metadata_csv)
    clean_df = standardize_metadata(raw_df, allowed_labels)

    image_lookup = build_image_lookup(images_dir)
    prepared_df = attach_image_paths(clean_df, image_lookup)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output_csv, index=False)

    LOGGER.info("Saved %d rows to %s", len(prepared_df), output_csv)
    LOGGER.info("Class distribution:\n%s", prepared_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
