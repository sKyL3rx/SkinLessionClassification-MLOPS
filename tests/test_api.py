from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from deployment.api.main import app, get_predictor


class FakePredictor:
    def model_info(self) -> dict:
        return {
            "backend": "fake",
            "onnx_path": "fake.onnx",
            "labels": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            "image_size": 224,
            "confidence_threshold": 0.65,
            "providers": ["FakeProvider"],
            "input_name": "image",
            "output_name": "logits",
        }

    def predict_pil(self, image, top_k: int = 3) -> dict:
        return {
            "predicted_class": "nv",
            "confidence": 0.91,
            "needs_review": False,
            "top_k": [
                {"label": "nv", "probability": 0.91},
                {"label": "mel", "probability": 0.06},
                {"label": "bkl", "probability": 0.03},
            ][:top_k],
            "model_backend": "fake",
            "onnx_path": "fake.onnx",
            "image_size": 224,
            "confidence_threshold": 0.65,
        }
    
def override_predictor() -> FakePredictor:
    return FakePredictor()

app.dependency_overrides[get_predictor] = override_predictor
client = TestClient(app)

def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info() -> None:
    response = client.get("/model-info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["backend"] == "fake"
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

    assert payload["predicted_class"] == "nv"
    assert payload["confidence"] == 0.91
    assert payload["needs_review"] is False
    assert len(payload["top_k"]) == 3


def test_predict_rejects_non_image_file(tmp_path) -> None:
    text_path = tmp_path / "not_image.txt"
    text_path.write_text("hello", encoding="utf-8")

    with open(text_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("not_image.txt", f, "text/plain")},
        )

    assert response.status_code == 400