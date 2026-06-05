from __future__ import annotations

import timm

from lesion_ml.models.fusion import MetadataFusionClassifier

SUPPORTED_BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",

    # ConvNeXt family
    "convnext_tiny": "convnext_tiny",
    "convnext_tiny_hnf": "convnext_tiny_hnf",

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
    dropout: float = 0.0,
    use_metadata: bool = False,
    metadata_dim: int | None = None,
    metadata_hidden_dim: int = 64,
    fusion_hidden_dim: int = 256,
):
    if use_metadata:
        if metadata_dim is None:
            raise ValueError("metadata_dim must be provided when use_metadata=True")

        image_backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        image_feature_dim = image_backbone.num_features

        return MetadataFusionClassifier(
            image_backbone=image_backbone,
            image_feature_dim=image_feature_dim,
            metadata_dim=metadata_dim,
            num_classes=num_classes,
            metadata_hidden_dim=metadata_hidden_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )


def build_model_from_config(config: dict):
    train_cfg = config["train"]
    data_cfg = config["data"]

    return build_model(
        backbone=train_cfg["backbone"],
        num_classes=int(data_cfg["num_classes"]),
        pretrained=bool(train_cfg.get("pretrained", True)),
        dropout=float(train_cfg.get("dropout", 0.0)),
        use_metadata=bool(train_cfg.get("use_metadata", False)),
        metadata_dim=train_cfg.get("metadata_dim"),
        metadata_hidden_dim=int(train_cfg.get("metadata_hidden_dim", 64)),
        fusion_hidden_dim=int(train_cfg.get("fusion_hidden_dim", 256)),
    )