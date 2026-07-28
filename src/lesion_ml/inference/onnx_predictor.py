from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from lesion_ml.data.metadata import (
    encode_metadata_row,
    load_metadata_schema,
)
from lesion_ml.inference.calibration import (
    apply_temperature_to_logits,
    load_temperature,
    top_k_prediction,
)
from lesion_ml.inference.preprocessing import (
    preprocess_pil_for_onnx,
)


class SkinLesionONNXPredictor:
    def __init__(
        self,
        *,
        onnx_path: str | Path,
        labels: list[str],
        image_size: int,
        preprocessing: dict[str, Any],
        confidence_threshold: float,
        temperature: float,
        experiment_name: str,
        providers: list[str] | None = None,
        metadata_schema_path: str | Path | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.labels = [str(label) for label in labels]
        self.image_size = int(image_size)
        self.preprocessing = dict(preprocessing)
        self.confidence_threshold = float(confidence_threshold)
        self.temperature = float(temperature)
        self.experiment_name = str(experiment_name)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        if not self.labels:
            raise ValueError("labels cannot be empty.")

        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in between [0,1].")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers,
        )

        active_providers = self.session.get_providers()

        print(
            "[INFO] Active ONNX providers: "
            f"{active_providers}"
        )

        if (
            providers[0] == "CUDAExecutionProvider"
            and active_providers[0]
            != "CUDAExecutionProvider"
        ):
            raise RuntimeError(
                "CUDA was requested, but ONNX Runtime "
                "falling back to CPU. "
                f"Active providers: {active_providers}"
            )



        self.inputs = {input_meta.name: input_meta for input_meta in self.session.get_inputs()}

        self.input_names = list(self.inputs.keys())

        self.outputs = {output_meta.name: output_meta for output_meta in self.session.get_outputs()}

        self.image_input_name = "image" if "image" in self.inputs else self.input_names[0]

        self.metadata_input_name = "metadata" if "metadata" in self.inputs else None

        self.uses_metadata = self.metadata_input_name is not None

        self.output_name = "logits" if "logits" in self.outputs else next(iter(self.outputs))

        self.metadata_schema = None

        if self.uses_metadata:
            if metadata_schema_path is None:
                raise FileNotFoundError(
                    "Model requires metadata, but metadata_schema_path is missing."
                )

            metadata_schema_path = Path(metadata_schema_path)

            if not metadata_schema_path.exists():
                raise FileNotFoundError(f"Metadata schema not found: {metadata_schema_path}")

            self.metadata_schema = load_metadata_schema(metadata_schema_path)

    @staticmethod
    def _providers_from_name(
        provider: str,
    ) -> list[str]:
        provider = provider.lower().strip()

        if provider == "cpu":
            return ["CPUExecutionProvider"]
    
        if provider != "cuda":
            raise ValueError(
                f"Unsupported ONNX provider: {provider!r}"
            )
        
        ort.preload_dlls(directory="")

        available = ort.get_available_providers()

        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. "
                f"Available providers: {available}"
            )

        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    @classmethod
    def from_export_dir(
        cls,
        export_dir: str | Path,
        *,
        provider: str = "cpu",
    ) -> SkinLesionONNXPredictor:
        export_dir = Path(export_dir)

        onnx_path = export_dir / "model.onnx"
        metadata_path = export_dir / "model.metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing ONNX metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        metadata_schema_path = None

        if bool(metadata.get("uses_metadata", False)):
            metadata_schema_path = export_dir / str(
                metadata.get(
                    "metadata_schema_path",
                    "metadata_schema.json",
                )
            )

        return cls(
            onnx_path=onnx_path,
            labels=list(metadata["labels"]),
            image_size=int(metadata["image_size"]),
            preprocessing=dict(metadata["preprocessing"]),
            confidence_threshold=0.0,
            temperature=1.0,
            experiment_name=str(metadata["experiment_name"]),
            providers=cls._providers_from_name(provider),
            metadata_schema_path=(metadata_schema_path),
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str | Path,
        *,
        provider: str = "cpu",
    ) -> SkinLesionONNXPredictor:
        bundle_dir = Path(bundle_dir)

        onnx_path = bundle_dir / "model.onnx"
        metadata_path = bundle_dir / "model.metadata.json"
        temperature_path = bundle_dir / "temperature.json"
        decision_path = bundle_dir / "decision.json"

        required_paths = [
            onnx_path,
            metadata_path,
            temperature_path,
            decision_path,
        ]

        missing = [str(path) for path in required_paths if not path.exists()]

        if missing:
            raise FileNotFoundError(
                "Deployment bundle is incomplete. Missing: " + ", ".join(missing)
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        decision = json.loads(decision_path.read_text(encoding="utf-8"))

        labels = [str(label) for label in metadata["labels"]]

        metadata_schema_path = None

        if bool(metadata.get("uses_metadata", False)):
            metadata_schema_path = bundle_dir / str(
                metadata.get(
                    "metadata_schema_path",
                    "metadata_schema.json",
                )
            )

            if not metadata_schema_path.exists():
                raise FileNotFoundError(f"Metadata schema is missing: {metadata_schema_path}")

        return cls(
            onnx_path=onnx_path,
            labels=labels,
            image_size=int(metadata["image_size"]),
            preprocessing=dict(metadata["preprocessing"]),
            confidence_threshold=float(decision["threshold"]),
            temperature=load_temperature(temperature_path),
            experiment_name=str(metadata["experiment_name"]),
            providers=cls._providers_from_name(provider),
            metadata_schema_path=(metadata_schema_path),
        )

    def preprocess_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        return preprocess_pil_for_onnx(
            image,
            image_size=self.image_size,
            preprocessing=self.preprocessing,
        )

    def build_metadata_input(
        self,
        *,
        age: float | None,
        sex: str | None,
        anatom_site: str | None,
    ) -> np.ndarray:
        if not self.uses_metadata:
            return np.empty(
                (1, 0),
                dtype=np.float32,
            )

        if self.metadata_schema is None:
            raise RuntimeError("metadata_schema is required.")

        row = pd.Series(
            {
                "age": age,
                "sex": sex,
                "anatom_site": anatom_site,
            }
        )

        encoded = encode_metadata_row(
            row=row,
            schema=self.metadata_schema,
        )

        return encoded.astype(np.float32)[None, :]

    def predict_logits(
        self,
        image: Image.Image,
        *,
        age: float | None = None,
        sex: str | None = None,
        anatom_site: str | None = None,
    ) -> np.ndarray:
        feed: dict[str, np.ndarray] = {self.image_input_name: self.preprocess_image(image)}

        if self.uses_metadata:
            if self.metadata_input_name is None:
                raise RuntimeError("Metadata input name is unavailable.")

            feed[self.metadata_input_name] = self.build_metadata_input(
                age=age,
                sex=sex,
                anatom_site=anatom_site,
            )

        logits = self.session.run(
            [self.output_name],
            feed,
        )[0]

        logits = np.asarray(
            logits,
            dtype=np.float64,
        )

        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(self.labels):
            raise RuntimeError(
                f"Unexpected logits shape. Expected [1, {len(self.labels)}], got {logits.shape}"
            )

        return logits

    def predict_pil(
        self,
        image: Image.Image,
        *,
        age: float | None = None,
        sex: str | None = None,
        anatom_site: str | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        if not 1 <= int(top_k) <= len(self.labels):
            raise ValueError(f"top_k must be between 1 and {len(self.labels)}.")

        logits = self.predict_logits(
            image,
            age=age,
            sex=sex,
            anatom_site=anatom_site,
        )

        probabilities = apply_temperature_to_logits(
            logits,
            self.temperature,
        )[0]

        top_predictions = top_k_prediction(
            probabilities,
            self.labels,
            k=int(top_k),
        )

        predicted_class = str(top_predictions[0]["label"])

        confidence = float(top_predictions[0]["probability"])

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "needs_review": bool(confidence < self.confidence_threshold),
            "top_k": top_predictions,
            "model_backend": "onnxruntime",
            "experiment_name": (self.experiment_name),
            "image_size": self.image_size,
            "temperature": self.temperature,
            "confidence_threshold": (self.confidence_threshold),
            "metadata_used": self.uses_metadata,
            "providers": (self.session.get_providers()),
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "onnxruntime",
            "experiment_name": (self.experiment_name),
            "labels": self.labels,
            "image_size": self.image_size,
            "temperature": self.temperature,
            "confidence_threshold": (self.confidence_threshold),
            "uses_metadata": (self.uses_metadata),
            "providers": (self.session.get_providers()),
            "input_names": self.input_names,
            "output_name": self.output_name,
        }
