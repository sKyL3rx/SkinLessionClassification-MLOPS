from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def load_temperature(path: str | Path) -> float:
    # Load temp val from calibration json that has been done previously
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing temperature JSON: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    temperature = float(data["temperature"])

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return temperature


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def apply_temperature_to_logits(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Apply temperature scaling to raw logits."""
    logits = np.asarray(logits, dtype=np.float64)
    temperature = float(temperature)

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return softmax(logits / temperature, axis=-1)


def apply_temperature_to_probs(
    probs: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply temperature scaling to probabilities"""

    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)

    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), eps, None)
    pseudo_logits = np.log(probs)

    return apply_temperature_to_logits(pseudo_logits, temperature)


def top_k_prediction(
    probs: np.ndarray,
    labels: Sequence[str],
    k: int = 3,
) -> list[dict[str, float | str]]:
    """Return top-k predictions from a probability vector."""
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 1:
        raise ValueError(f"Expected 1D probability vector, got shape {probs.shape}")

    if len(labels) != probs.shape[0]:
        raise ValueError(
            f"Number of labels ({len(labels)}) does not match probs shape ({probs.shape[0]})"
        )

    k = min(int(k), len(labels))

    order = np.argsort(probs)[::-1][:k]

    return [
        {
            "label": str(labels[i]),
            "probability": float(probs[i]),
        }
        for i in order
    ]
