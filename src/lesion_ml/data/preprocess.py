from __future__ import annotations

import cv2
import numpy as np


def crop_dark_border(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove near-black border from an RGB image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold

    if mask.sum() == 0:
        return image

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    if y2 <= y1 or x2 <= x1:
        return image

    return image[y1 : y2 + 1, x1 : x2 + 1]


def approximate_lesion_crop(
    image: np.ndarray,
    margin: float = 0.20,
    min_area_ratio: float = 0.03,
) -> np.ndarray:
    """
    Approximate lesion crop with thresholding.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image

    if not 0.0 <= margin <= 1.0:
        raise ValueError(f"margin must be in [0, 1], got {margin}")

    image = crop_dark_border(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return image

    image_area = image.shape[0] * image.shape[1]
    if image_area <= 0:
        return image

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    crop_area = w * h
    if crop_area < min_area_ratio * image_area:
        return image

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x1 = max(x - pad_x, 0)
    y1 = max(y - pad_y, 0)
    x2 = min(x + w + pad_x, image.shape[1])
    y2 = min(y + h + pad_y, image.shape[0])

    if y2 <= y1 or x2 <= x1:
        return image

    return image[y1:y2, x1:x2]


def apply_preprocess(
    image: np.ndarray,
    mode: str = "none",
    lesion_crop_margin: float = 0.20,
    dark_border_threshold: int = 10,
) -> np.ndarray:
    mode = mode.lower().strip()

    if mode in {"none", "null", ""}:
        return image

    if mode == "crop_dark_border":
        return crop_dark_border(
            image=image,
            threshold=dark_border_threshold,
        )

    if mode == "lesion_crop":
        return approximate_lesion_crop(
            image=image,
            margin=lesion_crop_margin,
        )

    raise ValueError(
        f"Unsupported preprocess mode='{mode}'. "
        "Supported these values only: none, crop_dark_border, lesion_crop"
    )
