from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lesion_ml.data.metadata import MetadataSchema, encode_metadata_row
from lesion_ml.data.preprocess import apply_preprocess

DEFAULT_CLASS_ORDER = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform: Callable | None = None,
        label_to_idx: dict[str, int] | None = None,
        return_metadata: bool = False,
        metadata_schema: MetadataSchema | None = None,
        validate_paths: bool = False,
        preprocess_mode: str = "none",
        lesion_crop_margin: float = 0.20,
        dark_border_threshold: int = 10,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        self.return_metadata = return_metadata
        self.metadata_schema = metadata_schema

        if self.return_metadata and self.metadata_schema is None:
            raise ValueError("metadata_schema must be provided when return_metadata=True")

        self._validate_dataframe()

        if label_to_idx is None:
            self.label_to_idx = self._build_label_mapping(self.df["label"].tolist())
        else:
            self.label_to_idx = label_to_idx

        unknown_labels = set(self.df["label"].unique()) - set(self.label_to_idx.keys())
        if unknown_labels:
            raise ValueError(f"Found labels not present in label_to_idx: {sorted(unknown_labels)}")

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        if validate_paths:
            self._validate_image_paths()

        self.preprocess_mode = preprocess_mode
        self.lesion_crop_margin = lesion_crop_margin
        self.dark_border_threshold = dark_border_threshold

    def _validate_dataframe(self) -> None:
        required_cols = {"image_id", "image_path", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")
        self.df["image_id"] = self.df["image_id"].astype(str).str.strip()
        self.df["image_path"] = self.df["image_path"].astype(str).str.strip()
        self.df["label"] = self.df["label"].astype(str).str.strip().str.lower()

    def _validate_image_paths(self) -> None:
        missing_paths = [p for p in self.df["image_path"].tolist() if not Path(p).exists()]

        if missing_paths:
            preview = missing_paths[:5]
            raise FileNotFoundError(
                f"Found {len(missing_paths)} missing image paths. Examples: {preview}"
            )

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            return torch.from_numpy(image).float()

        transformed = None

        try:
            transformed = self.transform(image=image)
        except TypeError:
            transformed = self.transform(image)

        if isinstance(transformed, dict) and "image" in transformed:
            image_out = transformed["image"]
        else:
            image_out = transformed

        if isinstance(image_out, torch.Tensor):
            return image_out.float()

        if isinstance(image_out, np.ndarray):
            if image_out.ndim == 3:
                image_out = np.transpose(image_out, (2, 0, 1))
            return torch.from_numpy(image_out).float()

        raise TypeError(
            f"Unsupported transform output type: {type(image_out)}. "
            "Expected torch.Tensor or numpy.ndarray."
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _build_label_mapping(labels: list[str]) -> dict[str, int]:
        unique_labels = set(labels)

        if unique_labels.issubset(set(DEFAULT_CLASS_ORDER)):
            ordered = [label for label in DEFAULT_CLASS_ORDER if label in unique_labels]
        else:
            ordered = sorted(unique_labels)

        return {label: idx for idx, label in enumerate(ordered)}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        image_id = row["image_id"]
        image_path = row["image_path"]
        label_name = row["label"]
        label_idx = self.label_to_idx[label_name]

        image = self._load_image(image_path)

        image = apply_preprocess(
            image=image,
            mode=self.preprocess_mode,
            lesion_crop_margin=self.lesion_crop_margin,
            dark_border_threshold=self.dark_border_threshold,
        )

        image_tensor = self._apply_transform(image)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "label": label_tensor,
            "label_name": label_name,
            "image_id": image_id,
            "image_path": image_path,
        }

        if self.return_metadata:
            if self.metadata_schema is None:
                raise ValueError("metadata_schema must be provided when return_metadata=True")

            metadata = encode_metadata_row(
                row=row,
                schema=self.metadata_schema,
            )

            sample["metadata"] = torch.tensor(metadata, dtype=torch.float32)

        return sample


def build_label_mapping_from_csv(csv_path: str | Path) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(str).str.strip().str.lower().tolist()

    unique_labels = set(labels)
    if unique_labels.issubset(set(DEFAULT_CLASS_ORDER)):
        ordered = [label for label in DEFAULT_CLASS_ORDER if label in unique_labels]
    else:
        ordered = sorted(unique_labels)

    return {label: idx for idx, label in enumerate(ordered)}
