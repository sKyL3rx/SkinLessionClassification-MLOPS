from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lesion_ml.api.main import app

BUNDLE_DIR = Path(
    os.getenv(
        "MODEL_BUNDLE_DIR",
        "artifacts/deployment/model",
    )
)

pytestmark = pytest.mark.artifact

if not (BUNDLE_DIR / "model.onnx").exists():
    pytest.skip(
        "Deployment bundle is unavailable.",
        allow_module_level=True,
    )


def test_onnx_predict_dummy_image(
    tmp_path: Path,
) -> None:
    image_array = np.random.randint(
        low=0,
        high=255,
        size=(224, 224, 3),
        dtype=np.uint8,
    )

    image = Image.fromarray(
        image_array,
    )

    image_path = tmp_path / "dummy.jpg"

    image.save(
        image_path,
    )

    with TestClient(app) as client:
        with image_path.open("rb") as file:
            response = client.post(
                "/predict",
                files={
                    "file": (
                        "dummy.jpg",
                        file,
                        "image/jpeg",
                    ),
                },
                data={
                    "top_k": "3",
                },
            )

    assert response.status_code == 200, response.text

    payload = response.json()

    assert "predicted_class" in payload
    assert "confidence" in payload
    assert "needs_review" in payload
    assert "top_k" in payload

    assert len(payload["top_k"]) == 3

    assert 0.0 <= payload["confidence"] <= 1.0
