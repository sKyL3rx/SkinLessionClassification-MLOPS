from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from deployment.api.main import app, get_predictor

ONNX_PATH = Path("deployment/onnx/model.onnx")

pytestmark = pytest.mark.artifact

if not ONNX_PATH.exists():
    pytest.skip(
        "ONNX model artifact is not available.",
        allow_module_level=True,
    )

get_predictor.cache_clear()
app.dependency_overrides = {}
client = TestClient(app)

def test_onnx_predict_dummy_image(tmp_path) -> None:
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