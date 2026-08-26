from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MetadataFusionClassifier(nn.Module):
    """Image + metadata classifier with concat or GMU fusion.

    fusion_type:
        - "concat": simple image-feature + metadata-feature concatenation.
        - "gmu": Gated Multimodal Unit style fusion.

    GMU equation, adapted from Arevalo et al. (2017):

        h_img  = tanh(W_img x_img)
        h_meta = tanh(W_meta x_meta)
        z      = sigmoid(W_z [x_img, x_meta])
        h      = z * h_img + (1 - z) * h_meta

        x_img  = dermoscopic image embedding from CNN/ViT backbone
        x_meta = structured metadata embedding from age/sex/anatomical-site features
    """

    SUPPORTED_FUSION_TYPES = {"concat", "gmu"}

    def __init__(
        self,
        image_backbone: nn.Module,
        image_feature_dim: int,
        metadata_dim: int,
        num_classes: int,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        fusion_type: str = "concat",
    ) -> None:
        super().__init__()

        if image_feature_dim <= 0:
            raise ValueError(f"image_feature_dim must be positive, got {image_feature_dim}")

        if metadata_dim <= 0:
            raise ValueError(f"metadata_dim must be positive, got {metadata_dim}")

        if num_classes <= 1:
            raise ValueError(f"num_classes must be greater than 1, got {num_classes}")

        if metadata_hidden_dim <= 0:
            raise ValueError(f"metadata_hidden_dim must be positive, got {metadata_hidden_dim}")

        if fusion_hidden_dim <= 0:
            raise ValueError(f"fusion_hidden_dim must be positive, got {fusion_hidden_dim}")

        fusion_type = fusion_type.lower().strip()
        if fusion_type not in self.SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"Unsupported fusion_type='{fusion_type}'. "
                f"Supported: {sorted(self.SUPPORTED_FUSION_TYPES)}"
            )

        self.image_backbone = image_backbone
        self.image_feature_dim = image_feature_dim
        self.metadata_dim = metadata_dim
        self.num_classes = num_classes
        self.metadata_hidden_dim = metadata_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout = dropout
        self.fusion_type = fusion_type

        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.LayerNorm(metadata_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if fusion_type == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(image_feature_dim + metadata_hidden_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, num_classes),
            )

        elif fusion_type == "gmu":
            self.image_projection = nn.Sequential(
                nn.Linear(image_feature_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.Tanh(),
            )

            self.metadata_projection = nn.Sequential(
                nn.Linear(metadata_hidden_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.Tanh(),
            )

            # GMU gate:
            # z = sigmoid(W_z [x_img, x_meta])
            self.gate = nn.Sequential(
                nn.Linear(image_feature_dim + metadata_hidden_dim, fusion_hidden_dim),
                nn.Sigmoid(),
            )

            self.classifier = nn.Sequential(
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden_dim, num_classes),
            )

    def _forward_metadata_features(
        self,
        metadata: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
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
        return self.metadata_mlp(metadata)

    def _forward_image_features(self, image: torch.Tensor) -> torch.Tensor:
        if hasattr(self.image_backbone, "forward_features"):
            features = self.image_backbone.forward_features(image)

            if features.ndim == 3 and hasattr(self.image_backbone, "forward_head"):
                features = self.image_backbone.forward_head(features, pre_logits=True)
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

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self._forward_image_features(image)
        metadata_features = self._forward_metadata_features(metadata, image_features)

        if self.fusion_type == "concat":
            fused_features = torch.cat([image_features, metadata_features], dim=1)
            return self.classifier(fused_features)

        if self.fusion_type == "gmu":
            image_hidden = self.image_projection(image_features)
            metadata_hidden = self.metadata_projection(metadata_features)

            gate_input = torch.cat([image_features, metadata_features], dim=1)
            gate = self.gate(gate_input)

            fused_features = gate * image_hidden + (1.0 - gate) * metadata_hidden
            return self.classifier(fused_features)

        raise RuntimeError(f"Unsupported fusion_type: {self.fusion_type}")

    def extra_repr(self) -> str:
        return (
            f"image_feature_dim={self.image_feature_dim}, "
            f"metadata_dim={self.metadata_dim}, "
            f"num_classes={self.num_classes}, "
            f"metadata_hidden_dim={self.metadata_hidden_dim}, "
            f"fusion_hidden_dim={self.fusion_hidden_dim}, "
            f"fusion_type='{self.fusion_type}'"
        )


def build_metadata_fusion_model(
    image_backbone: nn.Module,
    image_feature_dim: int,
    metadata_dim: int,
    num_classes: int,
    metadata_hidden_dim: int = 64,
    fusion_hidden_dim: int = 256,
    dropout: float = 0.2,
    fusion_type: str = "concat",
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
        fusion_type=fusion_type,
    )
