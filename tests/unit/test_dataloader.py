from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from lesion_ml.data.dataset import SkinLesionDataset, build_label_mapping_from_csv
from lesion_ml.data.transforms import build_transforms_from_config

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def config() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def train_csv_path() -> Path:
    path = Path("data/splits/train.csv")
    if not path.exists():
        pytest.skip("data/splits/train.csv not found. Run make prepare && make split first.")
    return path


@pytest.fixture(scope="module")
def label_to_idx(train_csv_path: Path) -> dict[str, int]:
    return build_label_mapping_from_csv(train_csv_path)


@pytest.fixture(scope="module")
def train_dataset(
    config: dict, train_csv_path: Path, label_to_idx: dict[str, int]
) -> SkinLesionDataset:
    ds = SkinLesionDataset(
        csv_path=train_csv_path,
        transform=build_transforms_from_config(config, "train"),
        label_to_idx=label_to_idx,
        return_metadata=False,
        validate_paths=False,
    )
    if len(ds) == 0:
        pytest.skip("Train dataset is empty.")
    return ds


def test_dataset_can_be_created(train_dataset: SkinLesionDataset) -> None:
    assert len(train_dataset) > 0


def test_dataset_item_structure(train_dataset: SkinLesionDataset) -> None:
    sample = train_dataset[0]

    assert "image" in sample
    assert "label" in sample
    assert "label_name" in sample
    assert "image_id" in sample
    assert "image_path" in sample

    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["label"], torch.Tensor)
    assert sample["image"].ndim == 3
    assert sample["image"].shape[0] == 3
    assert sample["label"].dtype == torch.long
    assert isinstance(sample["label_name"], str)
    assert Path(sample["image_path"]).exists()


def test_dataloader_returns_valid_batch(config: dict, train_dataset: SkinLesionDataset) -> None:
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    expected_img_size = int(config["data"]["image_size"])

    assert "image" in batch
    assert "label" in batch
    assert "label_name" in batch
    assert "image_id" in batch
    assert "image_path" in batch

    assert isinstance(batch["image"], torch.Tensor)
    assert isinstance(batch["label"], torch.Tensor)

    assert batch["image"].ndim == 4
    assert batch["image"].shape[0] == 4
    assert batch["image"].shape[1] == 3
    assert batch["image"].shape[2] == expected_img_size
    assert batch["image"].shape[3] == expected_img_size

    assert batch["label"].ndim == 1
    assert batch["label"].shape[0] == 4
    assert batch["label"].dtype == torch.long

    assert len(batch["label_name"]) == 4
    assert len(batch["image_id"]) == 4
    assert len(batch["image_path"]) == 4
