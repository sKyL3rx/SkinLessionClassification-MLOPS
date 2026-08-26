from __future__ import annotations

from io import BytesIO
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from lesion_ml.api.main import create_app

LABELS = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]


class FakePredictor:
    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "fake",
            "experiment_name": "fake-experiment",
            "labels": LABELS,
            "image_size": 224,
            "temperature": 1.0,
            "confidence_threshold": 0.65,
            "uses_metadata": True,
            "providers": ["FakeProvider"],
            "input_names": [
                "image",
                "metadata",
            ],
            "output_name": "logits",
        }

    def predict_pil(
        self,
        image: Image.Image,
        *,
        age: float | None = None,
        sex: str | None = None,
        anatom_site: str | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        assert isinstance(image, Image.Image)

        predictions = [
            {
                "label": "nv",
                "probability": 0.91,
            },
            {
                "label": "mel",
                "probability": 0.06,
            },
            {
                "label": "bkl",
                "probability": 0.03,
            },
        ]

        return {
            "predicted_class": "nv",
            "confidence": 0.91,
            "needs_review": False,
            "top_k": predictions[:top_k],
            "model_backend": "fake",
            "experiment_name": "fake-experiment",
            "image_size": 224,
            "temperature": 1.0,
            "confidence_threshold": 0.65,
            "metadata_used": True,
            "providers": ["FakeProvider"],
        }


def create_test_client() -> TestClient:
    app = create_app(
        predictor_factory=FakePredictor,
    )

    return TestClient(app)


def make_image_bytes() -> bytes:
    buffer = BytesIO()

    Image.new(
        mode="RGB",
        size=(32, 32),
        color=(120, 80, 60),
    ).save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


def test_health() -> None:
    with create_test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info() -> None:
    with create_test_client() as client:
        response = client.get("/model-info")

    assert response.status_code == 200

    payload = response.json()

    assert payload["backend"] == "fake"
    assert payload["experiment_name"] == "fake-experiment"
    assert payload["labels"] == LABELS
    assert payload["image_size"] == 224
    assert payload["temperature"] == 1.0
    assert payload["confidence_threshold"] == 0.65
    assert payload["uses_metadata"] is True
    assert payload["providers"] == ["FakeProvider"]
    assert payload["input_names"] == [
        "image",
        "metadata",
    ]
    assert payload["output_name"] == "logits"


def test_predict_image_with_metadata() -> None:
    with create_test_client() as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    make_image_bytes(),
                    "image/jpeg",
                )
            },
            data={
                "age": "40",
                "sex": "male",
                "anatom_site": "back",
                "top_k": "3",
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["predicted_class"] == "nv"
    assert payload["confidence"] == 0.91
    assert payload["needs_review"] is False
    assert len(payload["top_k"]) == 3

    assert payload["top_k"][0] == {
        "label": "nv",
        "probability": 0.91,
    }

    assert payload["model_backend"] == "fake"
    assert payload["metadata_used"] is True


def test_predict_respects_top_k() -> None:
    with create_test_client() as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    make_image_bytes(),
                    "image/jpeg",
                )
            },
            data={
                "top_k": "2",
            },
        )

    assert response.status_code == 200
    assert len(response.json()["top_k"]) == 2


def test_predict_rejects_non_image_content_type() -> None:
    with create_test_client() as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "not_image.txt",
                    b"hello",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Uploaded file must be an image.")


def test_predict_rejects_invalid_image_bytes() -> None:
    with create_test_client() as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "broken.jpg",
                    b"this-is-not-a-real-image",
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Invalid image file.")


def test_predict_rejects_invalid_top_k() -> None:
    with create_test_client() as client:
        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    make_image_bytes(),
                    "image/jpeg",
                )
            },
            data={
                "top_k": "8",
            },
        )

    assert response.status_code == 422
