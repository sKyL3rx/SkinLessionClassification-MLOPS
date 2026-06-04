from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from lesion_ml.data.dataset import SkinLesionDataset, build_label_mapping_from_csv
from lesion_ml.data.metadata import build_metadata_schema


def create_dummy_metadata_dataset(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels = ["akiec", "bcc", "mel", "nv"]

    for idx, label in enumerate(labels):
        image_id = f"dummy_{idx}"
        image_path = image_dir / f"{image_id}.jpg"

        image_array = np.random.randint(
            low=0,
            high=255,
            size=(64, 64, 3),
            dtype=np.uint8,
        )
        Image.fromarray(image_array).save(image_path)

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "label": label,
                "lesion_id": f"lesion_{idx}",
                "age": 40 + idx * 5,
                "sex": "male" if idx % 2 == 0 else "female",
                "anatom_site": "back" if idx % 2 == 0 else "lower extremity",
            }
        )

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return csv_path

def test_dataset_return_metadata_tensor(tmp_path: Path) -> None:
    csv_path = create_dummy_metadata_dataset(tmp_path)

    label_to_idx = build_label_mapping_from_csv(csv_path)
    metadata_schema = build_metadata_schema(csv_path)

    dataset = SkinLesionDataset(
        csv_path=csv_path,
        transform=None,
        label_to_idx=label_to_idx,
        return_metadata=True,
        metadata_schema=metadata_schema,
        validate_paths=True,
    )

    sample = dataset[0]

    assert "metadata" in sample
    assert isinstance(sample["metadata"], torch.Tensor)
    assert sample["metadata"].dtype == torch.float32
    assert sample["metadata"].shape == (metadata_schema.dim,)

def test_dataloader_collates_metadata_batch(tmp_path: Path) -> None:
    csv_path = create_dummy_metadata_dataset(tmp_path)

    label_to_idx = build_label_mapping_from_csv(csv_path)
    metadata_schema = build_metadata_schema(csv_path)

    dataset = SkinLesionDataset(
        csv_path=csv_path,
        transform=None,
        label_to_idx=label_to_idx,
        return_metadata=True,
        metadata_schema=metadata_schema,
        validate_paths=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert "metadata" in batch
    assert isinstance(batch["metadata"], torch.Tensor)
    assert batch["metadata"].dtype == torch.float32
    assert batch["metadata"].shape == (4, metadata_schema.dim)