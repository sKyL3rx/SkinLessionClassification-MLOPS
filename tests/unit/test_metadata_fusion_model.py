from __future__ import annotations

import pytest
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


@pytest.mark.parametrize("fusion_type", ["concat", "gmu"])
def test_metadata_fusion_classifier_forward(fusion_type: str) -> None:
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
        fusion_type=fusion_type,
    )

    images = torch.randn(batch_size, 3, 64, 64)
    metadata = torch.randn(batch_size, metadata_dim)

    logits = model(images, metadata)

    assert logits.shape == (batch_size, num_classes)


def test_gmu_has_expected_gate_shape() -> None:
    batch_size = 4
    image_feature_dim = 16
    metadata_dim = 10
    metadata_hidden_dim = 8
    fusion_hidden_dim = 12
    num_classes = 7

    image_backbone = DummyImageBackbone(feature_dim=image_feature_dim)

    model = MetadataFusionClassifier(
        image_backbone=image_backbone,
        image_feature_dim=image_feature_dim,
        metadata_dim=metadata_dim,
        num_classes=num_classes,
        metadata_hidden_dim=metadata_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        dropout=0.0,
        fusion_type="gmu",
    )

    images = torch.randn(batch_size, 3, 64, 64)
    metadata = torch.randn(batch_size, metadata_dim)

    with torch.no_grad():
        image_features = model._forward_image_features(images)
        metadata_features = model._forward_metadata_features(metadata, image_features)
        gate_input = torch.cat([image_features, metadata_features], dim=1)
        gate = model.gate(gate_input)

    assert gate.shape == (batch_size, fusion_hidden_dim)
    assert torch.all(gate >= 0.0)
    assert torch.all(gate <= 1.0)


def test_metadata_fusion_classifier_rejects_bad_metadata_shape() -> None:
    image_backbone = DummyImageBackbone(feature_dim=16)

    model = MetadataFusionClassifier(
        image_backbone=image_backbone,
        image_feature_dim=16,
        metadata_dim=10,
        num_classes=7,
        dropout=0.0,
        fusion_type="gmu",
    )

    images = torch.randn(4, 3, 64, 64)
    metadata = torch.randn(4, 10, 1)

    with pytest.raises(ValueError, match="Metadata must have shape"):
        model(images, metadata)


def test_metadata_fusion_classifier_rejects_invalid_fusion_type() -> None:
    image_backbone = DummyImageBackbone(feature_dim=16)

    with pytest.raises(ValueError, match="Unsupported fusion_type"):
        MetadataFusionClassifier(
            image_backbone=image_backbone,
            image_feature_dim=16,
            metadata_dim=10,
            num_classes=7,
            dropout=0.0,
            fusion_type="gated",
        )
