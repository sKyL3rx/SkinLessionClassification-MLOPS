from __future__ import annotations

import torch
from torch import nn

from lesion_ml.models.fusion import MetadataFusionClassifier


class DummyImageBackbone(nn.Module):
    def __init__(self, feature_dim: int = 16) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Conv2d(3, feature_dim, kernel_size=3, padding=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_metadata_fusion_classifier_forward() -> None:
    batch_size = 4
    image_feature_dim = 16
    metadata_dim = 10
    num_classes = 7

    image_backbone = DummyImageBackbone(feature_dim=image_feature_dim)

    model = MetadataFusionClassifier(
        image_backbone=image_backbone,
        image_feature_dim=image_feature_dim,
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        metadata_hidden_dim=8,
        fusion_hidden_dim=12,
        dropout=0.0,
    )

    images = torch.randn(batch_size, 3, 64, 64)
    metadata = torch.randn(batch_size, metadata_dim)

    logits = model(images, metadata)

    assert logits.shape == (batch_size, num_classes)


def test_metadata_fusion_classifier_rejects_bad_metadata_shape() -> None:
    image_backbone = DummyImageBackbone(feature_dim=16)

    model = MetadataFusionClassifier(
        image_backbone=image_backbone,
        image_feature_dim=16,
        metadata_dim=10,
        num_classes=7,
        dropout=0.0,
    )

    images = torch.randn(4, 3, 64, 64)
    metadata = torch.randn(4, 10, 1)

    try:
        model(images, metadata)
    except ValueError as exc:
        assert "Metadata must have shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid metadata shape.")