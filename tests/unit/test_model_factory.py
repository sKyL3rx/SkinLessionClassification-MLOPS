from __future__ import annotations

import pytest
import torch
import yaml

from lesion_ml.models.factory import build_model, build_model_from_config


@pytest.fixture(scope="module")
def config() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_build_model_resnet50() -> None:
    model = build_model(
        backbone="resnet50",
        num_classes=7,
        pretrained=False,
    )

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 7)


def test_build_model_from_config(config: dict) -> None:
    model = build_model_from_config(config)

    batch_size = 2
    image_size = int(config["data"]["image_size"])
    num_classes = int(config["data"]["num_classes"])

    x = torch.randn(batch_size, 3, image_size, image_size)

    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, num_classes)
