from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image

from deployment.api.predictor import SkinLessionONNXPredictor

ONNX_PATH = os.getenv("ONNX_PATH", "deployment/onnx/model.onnx")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))

app = FastAPI(
    title="Skin Lesion Classification API",
    description=(
        "ONNX Runtime inference API for 7-class skin lesion classification. "
        "This project is for ML engineering demonstration only, not clinical diagnosis."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> SkinLessionONNXPredictor:
    return SkinLessionONNXPredictor.from_metadata(
        onnx_path=ONNX_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info(
    predictor: SkinLessionONNXPredictor = Depends(get_predictor),
) -> dict[str, Any]:
    return predictor.model_info()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = 3,
    predictor: SkinLessionONNXPredictor = Depends(get_predictor),
) -> dict[str, Any]:
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected image file, got content_type={file.content_type}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    return predictor.predict_pil(image=image, top_k=top_k)