from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resize_transforms(
    image_size: int,
    resize_mode: str = "resize_pad",
) -> list[Any]:
    """Build resize transforms.

    resize:
        Directly resize to image_size x image_size. 

    resize_pad:
        Preserve aspect ratio with LongestMaxSize, then pad to image_size x image_size.
    """
    resize_mode = resize_mode.lower().strip()

    if resize_mode == "resize":
        return [
            A.Resize(height=image_size, width=image_size),
        ]

    if resize_mode == "resize_pad":
        return [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
            ),
        ]

    raise ValueError(
        f"Unsupported resize_mode='{resize_mode}'. Supported values: resize, resize_pad"
    )


def get_train_transforms(
    image_size: int = 224,
    use_random_resized_crop: bool = False,
    crop_scale: tuple[float, float] = (0.7, 1.0),
    crop_ratio: tuple[float, float] = (0.75, 1.33),
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    shift_limit: float = 0.05,
    scale_limit: float = 0.10,
    rotate_limit: int = 45,
    geometric_p: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05,
    color_jitter_p: float = 0.3,
    normalize: bool = True,
    resize_mode: str = "resize_pad",
) -> A.Compose:
    transforms: list[Any] = []

    if use_random_resized_crop:
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=crop_scale,
                ratio=crop_ratio,
                p=1.0,
            )
        )
    else:
        transforms.extend(
            build_resize_transforms(
                image_size=image_size,
                resize_mode=resize_mode,
            )
        )

    transforms.extend(
        [
            A.HorizontalFlip(p=horizontal_flip_p),
            A.VerticalFlip(p=vertical_flip_p),
            A.Affine(
                translate_percent={
                    "x": (-shift_limit, shift_limit),
                    "y": (-shift_limit, shift_limit),
                },
                scale={
                    "x": (1.0 - scale_limit, 1.0 + scale_limit),
                    "y": (1.0 - scale_limit, 1.0 + scale_limit),
                },
                rotate=(-rotate_limit, rotate_limit),
                shear={"x": (0.0, 0.0), "y": (0.0, 0.0)},
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=geometric_p,
            ),
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=color_jitter_p,
            ),
        ]
    )

    if normalize:
        transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        transforms.append(
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            )
        )

    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_eval_transforms(
    image_size: int = 224,
    normalize: bool = True,
    resize_mode: str = "resize_pad",
) -> A.Compose:
    transforms: list[Any] = []

    transforms.extend(
        build_resize_transforms(
            image_size=image_size,
            resize_mode=resize_mode,
        )
    )

    if normalize:
        transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        transforms.append(
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            )
        )

    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_tta_transforms(
    image_size: int = 224,
    normalize: bool = True,
    resize_mode: str = "resize_pad",
) -> list[A.Compose]:
    """Light TTA.

    - original
    - hflip
    - vflip
    - hflip + vflip
.
    """
    resize_transforms = build_resize_transforms(
        image_size=image_size,
        resize_mode=resize_mode,
    )

    normalize_transforms: list[Any] = []
    if normalize:
        normalize_transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        normalize_transforms.append(
            A.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
            )
        )

    tail = normalize_transforms + [ToTensorV2()]

    return [
        A.Compose(resize_transforms + tail),
        A.Compose([A.HorizontalFlip(p=1.0)] + resize_transforms + tail),
        A.Compose([A.VerticalFlip(p=1.0)] + resize_transforms + tail),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + resize_transforms + tail),
    ]


def build_transforms_from_config(config: dict[str, Any], split: str) -> A.Compose:
    image_size = int(config["data"].get("image_size", 224))
    normalize = bool(config.get("augment", {}).get("normalize", True))
    split = split.lower()

    aug = config.get("augment", {})
    preprocess = config.get("preprocess", {})
    resize_mode = str(preprocess.get("resize_mode", "resize_pad"))

    if split == "train":
        return get_train_transforms(
            image_size=image_size,
            use_random_resized_crop=bool(aug.get("use_random_resized_crop", False)),
            crop_scale=tuple(aug.get("crop_scale", (0.7, 1.0))),
            crop_ratio=tuple(aug.get("crop_ratio", (0.75, 1.33))),
            horizontal_flip_p=float(aug.get("horizontal_flip_p", 0.5)),
            vertical_flip_p=float(aug.get("vertical_flip_p", 0.5)),
            shift_limit=float(aug.get("shift_limit", 0.05)),
            scale_limit=float(aug.get("scale_limit", 0.10)),
            rotate_limit=int(aug.get("rotate_limit", 45)),
            geometric_p=float(aug.get("geometric_p", 0.5)),
            brightness=float(aug.get("brightness", 0.2)),
            contrast=float(aug.get("contrast", 0.2)),
            saturation=float(aug.get("saturation", 0.2)),
            hue=float(aug.get("hue", 0.05)),
            color_jitter_p=float(aug.get("color_jitter_p", 0.3)),
            normalize=normalize,
            resize_mode=resize_mode,
        )

    if split in {"val", "valid", "validation", "test"}:
        return get_eval_transforms(
            image_size=image_size,
            normalize=normalize,
            resize_mode=resize_mode,
        )

    raise ValueError(f"Unsupported split: {split}")
