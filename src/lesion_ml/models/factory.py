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
    # Optional aliases 
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    # DINOV2 FAM
    "dinov2_small": "hf_hub:timm/vit_small_patch14_dinov2.lvd142m",
    "dinov2_base": "hf_hub:timm/vit_base_patch14_dinov2.lvd142m",
}


def get_backbone_names() -> list[str]:
    return sorted(SUPPORTED_BACKBONES.keys())


def get_real_model_name(backbone_name: str) -> str:
    key = backbone_name.lower().strip()

    if key in SUPPORTED_BACKBONES:
        return SUPPORTED_BACKBONES[key]

    return backbone_name


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
    use_metadata: bool = False,
    metadata_dim: int | None = None,
    metadata_hidden_dim: int = 64,
    fusion_hidden_dim: int = 256,
    fusion_type: str = "concat",
    drop_path_rate: float = 0.0,
    freeze_backbone: bool = False,
):
    real_backbone = get_real_model_name(backbone)

    if use_metadata:
        if metadata_dim is None:
            raise ValueError("metadata_dim must be provided when use_metadata=True")

        is_dinov2 = "dinov2" in real_backbone.lower()

        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_rate": dropout,
            "drop_path_rate": drop_path_rate,
        }

        if not is_dinov2:
            create_kwargs["global_pool"] = "avg"

        image_backbone = timm.create_model(
            real_backbone,
            **create_kwargs,
        )

        if freeze_backbone:
            for p in image_backbone.parameters():
                p.requires_grad = False

        image_feature_dim = image_backbone.num_features

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

    return timm.create_model(
        real_backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
        drop_path_rate=drop_path_rate,
    )


def build_model_from_config(config: dict):
    train_cfg = config["train"]
    data_cfg = config["data"]

    return build_model(
        backbone=train_cfg["backbone"],
        num_classes=int(data_cfg["num_classes"]),
        pretrained=bool(train_cfg.get("pretrained", True)),
        dropout=float(train_cfg.get("dropout", 0.0)),
        drop_path_rate=float(train_cfg.get("drop_path_rate", 0.0)),
        use_metadata=bool(train_cfg.get("use_metadata", False)),
        metadata_dim=train_cfg.get("metadata_dim"),
        metadata_hidden_dim=int(train_cfg.get("metadata_hidden_dim", 64)),
        fusion_hidden_dim=int(train_cfg.get("fusion_hidden_dim", 256)),
        fusion_type=str(train_cfg.get("fusion_type", "concat")),
        freeze_backbone=bool(train_cfg.get("freeze_backbone", False)),
    )
