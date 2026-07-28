from __future__ import annotations

import io
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from PIL import (
    Image,
    UnidentifiedImageError,
)

from lesion_ml.inference.onnx_predictor import (
    SkinLesionONNXPredictor,
)

PredictorFactory = Callable[
    [],
    SkinLesionONNXPredictor,
]


def default_predictor_factory() -> SkinLesionONNXPredictor:
    bundle_dir = Path(
        os.getenv(
            "MODEL_BUNDLE_DIR",
            "artifacts/deployment/model",
        )
    )

    provider = os.getenv(
        "ONNX_PROVIDER",
        "cpu",
    )

    return SkinLesionONNXPredictor.from_bundle(
        bundle_dir,
        provider=provider,
    )


def create_app(
    predictor_factory: PredictorFactory = (default_predictor_factory),
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        app.state.predictor = predictor_factory()

        yield

        app.state.predictor = None

    app = FastAPI(
        title=("Skin Lesion Classification API"),
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_predictor(
        request: Request,
    ) -> SkinLesionONNXPredictor:
        predictor = getattr(
            request.app.state,
            "predictor",
            None,
        )

        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded.",
            )

        return predictor

    @app.get("/health")
    def health(
        request: Request,
    ) -> dict[str, Any]:
        loaded = (
            getattr(
                request.app.state,
                "predictor",
                None,
            )
            is not None
        )

        return {
            "status": ("ok" if loaded else "not_ready"),
            "model_loaded": loaded,
        }

    @app.get("/model-info")
    def model_info(
        request: Request,
    ) -> dict[str, Any]:
        return get_predictor(request).model_info()

    @app.post("/predict")
    async def predict(
        request: Request,
        file: Annotated[
            UploadFile,
            File(),
        ],
        age: Annotated[
            float | None,
            Form(ge=0, le=120),
        ] = None,
        sex: Annotated[
            str | None,
            Form(),
        ] = None,
        anatom_site: Annotated[
            str | None,
            Form(),
        ] = None,
        top_k: Annotated[
            int,
            Form(ge=1, le=7),
        ] = 3,
    ) -> dict[str, Any]:
        if file.content_type is not None and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=("Uploaded file must be an image."),
            )

        max_upload_mb = int(
            os.getenv(
                "MAX_UPLOAD_MB",
                "10",
            )
        )

        max_bytes = max_upload_mb * 1024 * 1024

        contents = await file.read(max_bytes + 1)

        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(f"Image exceeds {max_upload_mb} MB."),
            )

        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
            image = image.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as error:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file.",
            ) from error

        predictor = get_predictor(request)

        return predictor.predict_pil(
            image,
            age=age,
            sex=sex,
            anatom_site=anatom_site,
            top_k=top_k,
        )

    return app


app = create_app()
