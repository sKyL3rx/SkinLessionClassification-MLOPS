from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class SkinLessionONNXPredictor:
    def __init__(
        self,
        onnx_path: str | Path,
        labels: list[str],
        image_size: int = 224,
        confidence_threshold: float = 0.65,
        providers: list[str] | None = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.labels = labels
        self.image_size = int(image_size)
        self.confidence_threshold = float(confidence_threshold)

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @classmethod
    def from_metadata(
        cls,
        onnx_path: str | Path = "deployment/onnx/model.onnx",
        metadata_path: str | Path | None = None,
        confidence_threshold: float = 0.65,
    ) -> "SkinLessionONNXPredictor":
        
        onnx_path = Path(onnx_path)

        if metadata_path is None:
            metadata_path = onnx_path.with_suffix(".metadata.json")
        
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = list(metadata.get("labels", []))
            image_size = int(metadata.get("image_size", 224))
        else:
            labels = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
            image_size = 224
        
        if not labels:
            raise ValueError("Labels are empty. Check ONNX metadata.")

        return cls(
            onnx_path=onnx_path,
            labels=labels,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
        )
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  
        arr = np.transpose(arr, (2,0,1))
        arr = np.expand_dims(arr, axis = 0)

        return arr.astype(np.float32)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits = logits - np.max(logits, axis = 1, keepdims=True)

        exp = np.exp(logits)

        return exp / np.sum(exp, axis = 1, keepdims=True)

    def predict_pil(self, image: Image.Image, top_k: int = 3) -> dict[str,Any]:
        x = self.preprocess_image(image)

        logits=  self.session.run([self.output_name], {self.input_name: x})[0]

        probs = self.softmax(logits)[0]

        top_k = min(top_k, len(self.labels))
        top_indices = np.argsort(-probs)[:top_k]

        pred_idx = int(top_indices[0])
        confidence = float(probs[pred_idx])

        top_predictions = [
            {
                "label": self.labels[int(idx)],
                "probability": float(probs[int(idx)]),
            }
            for idx in top_indices
        ]

        return {
            "predicted_class": self.labels[pred_idx],
            "confidence": confidence,
            "needs_review": bool(confidence < self.confidence_threshold),
            "top_k": top_predictions,
            "model_backend": "onnxruntime",
            "onnx_path": str(self.onnx_path),
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
        }
    
    def model_info(self) -> dict[str, Any]:
        return {
            "backend": "onnxruntime",
            "onnx_path": str(self.onnx_path),
            "labels": self.labels,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "providers": self.session.get_providers(),
            "input_name": self.input_name,
            "output_name": self.output_name,
        }