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

def test_build_metadata_fusion_model_from_config() -> None:
    config = copy.deepcopy(load_config())

    config["train"]["pretrained"] = False
    config["train"]["use_metadata"] = True
    config["train"]["metadata_dim"] = 10
    config["train"]["metadata_hidden_dim"] = 8
    config["train"]["fusion_hidden_dim"] = 12

    model = build_model_from_config(config)

    batch_size = 2
    image_size = int(config["data"]["image_size"])
    num_classes = int(config["data"]["num_classes"])
    metadata_dim = int(config["train"]["metadata_dim"])

    images = torch.randn(batch_size, 3, image_size, image_size)
    metadata = torch.randn(batch_size, metadata_dim)

    with torch.no_grad():
        out = model(images, metadata)

    assert out.shape == (batch_size, num_classes)

    