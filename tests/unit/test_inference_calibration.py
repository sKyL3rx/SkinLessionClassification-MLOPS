from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lesion_ml.inference.calibration import (
    apply_temperature_to_logits,
    apply_temperature_to_probs,
    load_temperature,
    top_k_prediction,
)


def test_load_temperature(tmp_path: Path) -> None:
    path = tmp_path / "temperature.json"
    path.write_text(json.dumps({"temperature": 1.561}), encoding="utf-8")

    assert load_temperature(path) == 1.561


def test_apply_temperature_to_logits_returns_probability_distribution() -> None:
    logits = np.array([2.0, 1.0, 0.0])
    probs = apply_temperature_to_logits(logits, temperature=1.5)

    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)


def test_apply_temperature_to_probs_returns_probability_distribution() -> None:
    raw_probs = np.array([0.8, 0.15, 0.05])
    probs = apply_temperature_to_probs(raw_probs, temperature=1.5)

    assert probs.shape == (3,)
    assert np.all(probs >= 0.0)
    assert np.isclose(probs.sum(), 1.0)


def test_top_k_prediction_sorted_descending() -> None:
    probs = np.array([0.2, 0.7, 0.1])
    labels = ["a", "b", "c"]

    top = top_k_prediction(probs, labels, k=2)

    assert len(top) == 2
    assert top[0]["label"] == "b"
    assert top[0]["probability"] >= top[1]["probability"]
