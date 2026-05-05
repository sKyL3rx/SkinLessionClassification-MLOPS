from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

LOGGER = logging.getLogger("make_grouped_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped train/val/test splits.")
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


def ensure_required_columns(df: pd.DataFrame, group_col: str) -> None:
    required = {"image_id", "image_path", "label"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {missing}")

    if group_col not in df.columns:
        raise ValueError(
            f"Configured group column '{group_col}' not found in metadata.csv. "
            f"Available columns: {list(df.columns)}"
        )


def can_use_sgkf_fraction(frac: float, tol: float = 1e-6) -> bool:
    reciprocal = 1.0 / frac
    return abs(reciprocal - round(reciprocal)) < tol


def split_with_sgkf(
    df: pd.DataFrame, label_col: str, group_col: str, holdout_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:

    n_splits = int(round(1.0 / holdout_frac))

    if df[group_col].nunique() < n_splits:
        raise ValueError(
            f"Not enough unique groups for StratifiedGroupKFold: "
            f"{df[group_col].nunique()} groups < {n_splits} folds"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    X = df.index.to_numpy()
    y = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()

    train_idx, holdout_idx = next(splitter.split(X=X, y=y, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[holdout_idx].copy()


def split_with_group_shuffle(
    df: pd.DataFrame,
    group_col: str,
    holdout_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_frac,
        random_state=seed,
    )

    X = df.index.to_numpy()
    groups = df[group_col].to_numpy()

    train_idx, holdout_idx = next(splitter.split(X=X, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[holdout_idx].copy()


def grouped_split(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    holdout_frac: float,
    seed: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if can_use_sgkf_fraction(holdout_frac):
        LOGGER.info(
            "Using StratifiedGroupKFold for %s split with holdout_frac=%.4f",
            split_name,
            holdout_frac,
        )
        try:
            return split_with_sgkf(df, label_col, group_col, holdout_frac, seed)
        except Exception as exc:
            LOGGER.warning(
                "StratifiedGroupKFold failed for %s split (%s). Falling back to GroupShuffleSplit.",
                split_name,
                exc,
            )

    LOGGER.info(
        "Using GroupShuffleSplit for %s split with holdout_frac=%.4f",
        split_name,
        holdout_frac,
    )
    return split_with_group_shuffle(df, group_col, holdout_frac, seed)


def assert_no_group_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
) -> None:

    train_groups = set(train_df[group_col].astype(str))
    val_groups = set(val_df[group_col].astype(str))
    test_groups = set(test_df[group_col].astype(str))

    overlaps = {
        "train_val": train_groups & val_groups,
        "train_test": train_groups & test_groups,
        "val_test": val_groups & test_groups,
    }

    for name, values in overlaps.items():
        if values:
            raise RuntimeError(
                f"Group leakage detected in {name}: {len(values)} overlapping groups"
            )


def add_split_column(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split_name
    return out


def summarize_split(df: pd.DataFrame, name: str, group_col: str) -> None:
    LOGGER.info(
        "%s -> rows=%d | unique_images=%d | unique_groups=%d",
        name,
        len(df),
        df["image_id"].nunique(),
        df[group_col].nunique(),
    )
    LOGGER.info("%s class distribution:\n%s", name, df["label"].value_counts().sort_index())


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    config = load_config(args.config)

    # Config
    seed = int(config["project"]["seed"])
    metadata_csv = Path(config["data"]["metadata_csv"])
    splits_dir = Path(config["data"]["splits_dir"])
    val_size = float(config["data"]["val_size"])
    test_size = float(config["data"]["test_size"])
    group_col = str(config["data"].get("group_by", "split_group"))

    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata_csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    ensure_required_columns(df, group_col)

    LOGGER.info("Loaded metadata: %s rows from %s", len(df), metadata_csv)
    LOGGER.info("Using group column: %s", group_col)

    train_val_df, test_df = grouped_split(
        df=df,
        label_col="label",
        group_col=group_col,
        holdout_frac=test_size,
        seed=seed,
        split_name="test",
    )

    val_frac_within_train_val = val_size / (1.0 - test_size)
    train_df, val_df = grouped_split(
        df=train_val_df,
        label_col="label",
        group_col=group_col,
        holdout_frac=val_frac_within_train_val,
        seed=seed + 1,
        split_name="val",
    )

    assert_no_group_overlap(train_df, val_df, test_df, group_col)

    train_df = add_split_column(train_df, "train")
    val_df = add_split_column(val_df, "val")
    test_df = add_split_column(test_df, "test")

    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summarize_split(train_df, "train", group_col)
    summarize_split(val_df, "val", group_col)
    summarize_split(test_df, "test", group_col)

    LOGGER.info("Saved train split to %s", train_path)
    LOGGER.info("Saved val split to %s", val_path)
    LOGGER.info("Saved test split to %s", test_path)


if __name__ == "__main__":
    main()
