from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn

SUPPORTED_BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def get_backbone_names() -> list[str]:
    return sorted(SUPPORTED_BACKBONES.keys())


def get_real_model_name(backbone_name: str) -> str:
    key = backbone_name.lower().strip()
    if key not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Supported backbones: {get_backbone_names()}"
        )
    return SUPPORTED_BACKBONES[key]


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    in_chans: int = 3,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:

    model_name = get_real_model_name(backbone)

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    return model


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})

    backbone = str(train_cfg.get("backbone", "resnet50"))
    pretrained = bool(train_cfg.get("pretrained", True))
    in_chans = int(train_cfg.get("in_chans", 3))
    drop_rate = float(train_cfg.get("drop_rate", 0.0))
    drop_path_rate = float(train_cfg.get("drop_path_rate", 0.0))
    num_classes = int(data_cfg["num_classes"])

    return build_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        in_chans=in_chans,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
