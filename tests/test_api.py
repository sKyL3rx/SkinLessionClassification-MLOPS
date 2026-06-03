from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ONNX_PATH = Path("deployment/onnx/model.onnx")

pytestmark = pytest.mark.skipif(
    not ONNX_PATH.exists(),
    reason = "ONNX model artifact is not available."
)

from deployment.api.main import app

client = TestClient(app)

def test_health() -> None:
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["backend"] == "onnxruntime"
    assert "labels" in payload
    assert len(payload["labels"]) == 7
    assert "providers" in payload
    assert "input_name" in payload
    assert "output_name" in payload


def test_predict_dummy_image(tmp_path) -> None:
    image_array = np.random.randint(
        low=0,
        high=255,
        size=(224, 224, 3),
        dtype=np.uint8,
    )

    image = Image.fromarray(image_array)
    image_path = tmp_path / "dummy.jpg"
    image.save(image_path)

    with open(image_path, "rb") as f:
        response = client.post(
            "/predict?top_k=3",
            files={"file": ("dummy.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200

    payload = response.json()

    assert "predicted_class" in payload
    assert "confidence" in payload
    assert "needs_review" in payload
    assert "top_k" in payload
    assert len(payload["top_k"]) == 3
    assert 0.0 <= payload["confidence"] <= 1.0


def test_predict_rejects_non_image_file(tmp_path) -> None:
    text_path = tmp_path / "not_image.txt"
    text_path.write_text("hello", encoding="utf-8")

    with open(text_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("not_image.txt", f, "text/plain")},
        )

    assert response.status_code == 400
