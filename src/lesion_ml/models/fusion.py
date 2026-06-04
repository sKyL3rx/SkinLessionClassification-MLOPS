from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MetadataFusionClassifier(nn.Module):
    """V1: Image + metadata classifier using MLP-concat fusion."""

    def __init__(
        self,
        image_backbone: nn.Module,
        image_feature_dim: int,
        metadata_dim: int,
        num_classes: int,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if metadata_dim <= 0:
            raise ValueError(f"metadata_dim must be positive, got {metadata_dim}")

        self.image_backbone = image_backbone
        self.image_feature_dim = image_feature_dim
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + metadata_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self._forward_image_features(image)

        if metadata.ndim != 2:
            raise ValueError(
                f"Metadata must have shape [batch_size, metadata_dim], got {metadata.shape}"
            )

        if metadata.shape[0] != image_features.shape[0]:
            raise ValueError(
                "Metadata batch size must match image batch size: "
                f"metadata={metadata.shape[0]}, image={image_features.shape[0]}"
            )

        if metadata.shape[1] != self.metadata_dim:
            raise ValueError(
                f"Metadata feature dim must be {self.metadata_dim}, got {metadata.shape[1]}"
            )

        metadata = metadata.to(device=image_features.device, dtype=image_features.dtype)
        metadata_features = self.metadata_mlp(metadata)

        fused_features = torch.cat([image_features, metadata_features], dim=1)
        logits = self.classifier(fused_features)

        return logits

    def _forward_image_features(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.image_backbone, "forward_features"):
            features = self.image_backbone.forward_features(image)
        else:
            features = self.image_backbone(image)

        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        elif features.ndim == 3:
            features = features.mean(dim=1)

        if features.ndim != 2:
            raise ValueError(f"Expected image features with shape [B, C], got {features.shape}")

        if features.shape[1] != self.image_feature_dim:
            raise ValueError(
                f"Expected image feature dim {self.image_feature_dim}, got {features.shape[1]}"
            )

        return features

    def extra_repr(self) -> str:
        return (
            f"image_feature_dim={self.image_feature_dim}, "
            f"metadata_dim={self.metadata_dim}, "
            f"num_classes={self.num_classes}"
        )


def build_metadata_fusion_model(
    image_backbone: nn.Module,
    image_feature_dim: int,
    metadata_dim: int,
    num_classes: int,
    metadata_hidden_dim: int = 64,
    fusion_hidden_dim: int = 256,
    dropout: float = 0.2,
    **_: Any,
) -> MetadataFusionClassifier:
    return MetadataFusionClassifier(
        image_backbone=image_backbone,
        image_feature_dim=image_feature_dim,
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        metadata_hidden_dim=metadata_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=dropout,
    )