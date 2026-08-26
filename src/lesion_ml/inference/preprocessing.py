from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image

from lesion_ml.data.preprocess import apply_preprocess

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def resize_pad_rgb(
    image: np.ndarray,
    image_size: int,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got {image.shape}")

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = float(image_size) / float(max(height, width))

    resized_width = max(
        1,
        int(round(width * scale)),
    )

    resized_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    remaining_width = image_size - resized_width
    remaining_height = image_size - resized_height

    left = remaining_width // 2
    right = remaining_width - left
    top = remaining_height // 2
    bottom = remaining_height - top

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def preprocess_pil_for_onnx(
    image: Image.Image,
    *,
    image_size: int,
    preprocessing: dict[str, Any],
) -> np.ndarray:
    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    rgb = apply_preprocess(
        image=rgb,
        mode=str(preprocessing.get("mode", "none")),
        lesion_crop_margin=float(
            preprocessing.get(
                "lesion_crop_margin",
                0.20,
            )
        ),
        dark_border_threshold=int(
            preprocessing.get(
                "dark_border_threshold",
                10,
            )
        ),
    )

    resize_mode = str(
        preprocessing.get(
            "resize_mode",
            "resize_pad",
        )
    )

    if resize_mode == "resize_pad":
        rgb = resize_pad_rgb(
            rgb,
            image_size=image_size,
        )

    elif resize_mode == "resize":
        rgb = cv2.resize(
            rgb,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )

    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    array = rgb.astype(np.float32) / 255.0

    if bool(preprocessing.get("normalize", True)):
        array = (array - IMAGENET_MEAN) / IMAGENET_STD

    array = np.transpose(
        array,
        (2, 0, 1),
    )

    array = np.expand_dims(
        array,
        axis=0,
    )

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )
