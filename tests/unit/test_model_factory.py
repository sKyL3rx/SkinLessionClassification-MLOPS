from __future__ import annotations

import copy

import torch
import yaml

from lesion_ml.models.factory import build_model, build_model_from_config


def load_config() -> dict:
    with open("params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_build_model_resnet50() -> None:
    model = build_model(
        backbone="resnet50",
        num_classes=7,
        pretrained=False,
    )

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 7)

def test_build_model_from_config() -> None:
    config = copy.deepcopy(load_config())
    config["train"]["pretrained"] = False

    model = build_model_from_config(config)

    batch_size = 1
    image_size = int(config["data"]["image_size"])
    num_classes = int(config["data"]["num_classes"])

    x = torch.randn(batch_size, 3, image_size, image_size)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, num_classes)