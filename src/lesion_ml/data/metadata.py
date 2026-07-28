from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UNKNOWN_TOKEN = "__unknown__"


@dataclass(frozen=True)
class MetadataSchema:
    age_median: float
    age_mean: float
    age_std: float
    sex_categories: list[str]
    anatom_site_categories: list[str]

    @property
    def dim(self) -> int:
        return 1 + len(self.sex_categories) + len(self.anatom_site_categories)


def clean_category(value: Any) -> str:
    if pd.isna(value):
        return UNKNOWN_TOKEN

    value_str = str(value).strip().lower()

    if value_str in {"", "nan", "none", "unknown", "unk"}:
        return UNKNOWN_TOKEN

    return value_str


def build_metadata_schema(
    train_csv: str | Path,
    age_col: str = "age",
    sex_col: str = "sex",
    anatom_site_col: str = "anatom_site",
) -> MetadataSchema:
    df = pd.read_csv(train_csv)

    required_cols = [age_col, sex_col, anatom_site_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing metadata columns in {train_csv}: {missing_cols}")

    age = pd.to_numeric(df[age_col], errors="coerce")

    age_median = float(age.median())
    age_filled = age.fillna(age_median)

    age_mean = float(age_filled.mean())
    age_std = float(age_filled.std())

    if not np.isfinite(age_std) or age_std <= 0:
        age_std = 1.0

    sex_categories = sorted({clean_category(value) for value in df[sex_col].tolist()})

    anatom_site_categories = sorted(
        {clean_category(value) for value in df[anatom_site_col].tolist()}
    )

    if UNKNOWN_TOKEN not in sex_categories:
        sex_categories.append(UNKNOWN_TOKEN)

    if UNKNOWN_TOKEN not in anatom_site_categories:
        anatom_site_categories.append(UNKNOWN_TOKEN)

    return MetadataSchema(
        age_median=age_median,
        age_mean=age_mean,
        age_std=age_std,
        sex_categories=sex_categories,
        anatom_site_categories=anatom_site_categories,
    )


def encode_metadata_row(
    row: pd.Series,
    schema: MetadataSchema,
    age_col: str = "age",
    sex_col: str = "sex",
    anatom_site_col: str = "anatom_site",
) -> np.ndarray:
    age_value = pd.to_numeric(row.get(age_col), errors="coerce")

    if pd.isna(age_value):
        age_value = schema.age_median

    age_norm = (float(age_value) - schema.age_mean) / schema.age_std

    features: list[float] = [age_norm]

    sex_value = clean_category(row.get(sex_col))
    if sex_value not in schema.sex_categories:
        sex_value = UNKNOWN_TOKEN

    features.extend([1.0 if category == sex_value else 0.0 for category in schema.sex_categories])

    anatom_site_value = clean_category(row.get(anatom_site_col))
    if anatom_site_value not in schema.anatom_site_categories:
        anatom_site_value = UNKNOWN_TOKEN

    features.extend(
        [
            1.0 if category == anatom_site_value else 0.0
            for category in schema.anatom_site_categories
        ]
    )

    return np.asarray(features, dtype=np.float32)


def save_metadata_schema(schema: MetadataSchema, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(schema), indent=2), encoding="utf-8")


def load_metadata_schema(path: str | Path) -> MetadataSchema:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return MetadataSchema(**data)
