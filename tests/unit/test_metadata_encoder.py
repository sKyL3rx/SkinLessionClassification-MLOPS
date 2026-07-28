from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lesion_ml.data.metadata import (
    UNKNOWN_TOKEN,
    build_metadata_schema,
    encode_metadata_row,
    load_metadata_schema,
    save_metadata_schema,
)


def test_build_schema_and_encode_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"

    df = pd.DataFrame(
        [
            {"age": 45, "sex": "male", "anatom_site": "back"},
            {"age": 60, "sex": "female", "anatom_site": "lower extremity"},
            {"age": np.nan, "sex": "unknown", "anatom_site": "unknown"},
        ]
    )

    df.to_csv(csv_path, index=False)

    schema = build_metadata_schema(csv_path)

    assert schema.age_median == 52.5
    assert UNKNOWN_TOKEN in schema.sex_categories
    assert UNKNOWN_TOKEN in schema.anatom_site_categories
    assert schema.dim == 1 + len(schema.sex_categories) + len(schema.anatom_site_categories)

    encoded = encode_metadata_row(df.iloc[0], schema)

    assert encoded.shape == (schema.dim,)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()


def test_metadata_schema_handles_unseen_categories(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"

    train_df = pd.DataFrame(
        [
            {"age": 45, "sex": "male", "anatom_site": "back"},
            {"age": 60, "sex": "female", "anatom_site": "lower extremity"},
        ]
    )
    train_df.to_csv(csv_path, index=False)

    schema = build_metadata_schema(csv_path)

    unseen_row = pd.Series(
        {
            "age": None,
            "sex": "new-sex-category",
            "anatom_site": "new-site-category",
        }
    )

    encoded = encode_metadata_row(unseen_row, schema)

    assert encoded.shape == (schema.dim,)
    assert np.isfinite(encoded).all()


def test_metadata_schema_save_and_load(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    schema_path = tmp_path / "metadata_schema.json"

    df = pd.DataFrame(
        [
            {"age": 45, "sex": "male", "anatom_site": "back"},
            {"age": 60, "sex": "female", "anatom_site": "lower extremity"},
        ]
    )
    df.to_csv(csv_path, index=False)

    schema = build_metadata_schema(csv_path)
    save_metadata_schema(schema, schema_path)

    loaded = load_metadata_schema(schema_path)

    assert loaded == schema
    assert loaded.dim == schema.dim
