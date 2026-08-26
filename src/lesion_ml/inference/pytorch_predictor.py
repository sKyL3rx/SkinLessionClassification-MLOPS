from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from lesion_ml.data.metadata import MetadataSchema, build_metadata_schema, encode_metadata_row
from lesion_ml.data.preprocess import apply_preprocess
from lesion_ml.data.transforms import build_transforms_from_config
from lesion_ml.models.factory import build_model_from_config
from lesion_ml.models.forward import forward_batch
from lesion_ml.paths import get_project_paths

DEFAULT_CONFIG_PATH = "params.yaml"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TOP_K = 3
DISCLAIMER = "Personal project only. Not for clinical use."


@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    calibrated_confidence: float
    needs_review: bool
    threshold: float
    top_k: list[dict[str, float | str]]
    raw_confidence: float
    temperature: float
    use_tta: bool
    tta_transforms: list[str]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower().strip()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_temperature(path: str | Path | None) -> float:
    if path is None:
        return 1.0

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Temperature JSON not found: {path}. Run scripts/calibrate_temperature.py first."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if "temperature" not in data:
        raise KeyError(f"Temperature JSON must contain key 'temperature': {path}")

    temperature = float(data["temperature"])
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return temperature


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint)}")

    return checkpoint


def build_inference_config(
    runtime_config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:

    checkpoint_config = copy.deepcopy(checkpoint.get("config", runtime_config))
    checkpoint_config.setdefault("train", {})
    checkpoint_config["train"]["pretrained"] = False

    inference_config = copy.deepcopy(checkpoint_config)

    for key in ["inference", "evaluate", "preprocess", "project"]:
        if key in runtime_config:
            if isinstance(runtime_config[key], dict):
                inference_config.setdefault(key, {})
                inference_config[key].update(copy.deepcopy(runtime_config[key]))
            else:
                inference_config[key] = copy.deepcopy(runtime_config[key])

    if "data" in runtime_config:
        inference_config.setdefault("data", {})
        inference_config["data"].update(copy.deepcopy(runtime_config["data"]))

    return inference_config


def make_label_maps(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, int], list[str]]:
    if "label_to_idx" in checkpoint:
        label_to_idx = {str(k): int(v) for k, v in checkpoint["label_to_idx"].items()}
    else:
        labels = config.get("data", {}).get(
            "labels",
            ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        )
        label_to_idx = {str(label): idx for idx, label in enumerate(labels)}

    labels_by_idx = [
        label for label, _idx in sorted(label_to_idx.items(), key=lambda item: item[1])
    ]
    return label_to_idx, labels_by_idx


def get_tta_config(
    config: dict[str, Any], override_use_tta: bool | None = None
) -> tuple[bool, list[str]]:
    eval_cfg = config.get("evaluate", {})
    use_tta = bool(eval_cfg.get("use_tta", False))

    if override_use_tta is not None:
        use_tta = bool(override_use_tta)

    tta_transforms = eval_cfg.get("tta_transforms", ["original"])

    tta_transforms = [str(name).lower().strip() for name in tta_transforms]
    if "original" not in tta_transforms:
        tta_transforms = ["original", *tta_transforms]

    if not use_tta:
        return False, ["original"]

    return True, tta_transforms


def apply_tta_transform(images: torch.Tensor, transform_name: str) -> torch.Tensor:
    transform_name = transform_name.lower().strip()

    if transform_name == "original":
        return images

    if transform_name in {"hflip", "horizontal_flip"}:
        return torch.flip(images, dims=[3])

    if transform_name in {"vflip", "vertical_flip"}:
        return torch.flip(images, dims=[2])

    if transform_name in {"hvflip", "hflip_vflip", "vflip_hflip"}:
        return torch.flip(images, dims=[2, 3])

    raise ValueError(f"Unsupported TTA transform: {transform_name}")


def pil_to_rgb_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def load_image_as_rgb_numpy(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ValueError(f"Expected HWC RGB/RGBA image array, got shape {image.shape}")
        if image.shape[2] == 4:
            image = image[:, :, :3]
        return image.astype(np.uint8)
    if isinstance(image, Image.Image):
        return pil_to_rgb_numpy(image)
    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_single_image_preprocessor_tensor(
    image: str | Path | Image.Image | np.ndarray,
    config: dict[str, Any],
) -> torch.Tensor:
    image_np = load_image_as_rgb_numpy(image)

    preprocess_cfg = config.get("preprocess", {})

    image_np = apply_preprocess(
        image=image_np,
        mode=str(preprocess_cfg.get("mode", "none")),
        lesion_crop_margin=float(preprocess_cfg.get("lesion_crop_margin", 0.20)),
        dark_border_threshold=int(preprocess_cfg.get("dark_border_threshold", 10)),
    )

    transform = build_transforms_from_config(config, split="test")
    transformed = transform(image=image_np)

    tensor = transformed["image"]

    return tensor.float().unsqueeze(0)


def build_metadata_tensor(
    *,
    age: float | int | None,
    sex: str,
    anatom_site: str,
    metadata_schema: MetadataSchema,
) -> torch.Tensor:
    row = pd.Series(
        {
            "age": age,
            "sex": sex,
            "anatom_site": anatom_site,
        }
    )

    metadata = encode_metadata_row(row=row, schema=metadata_schema)

    return torch.tensor(metadata, dtype=torch.float32).unsqueeze(0)


def softmax(logits: np.ndarray, axis: int = -1) -> torch.Tensor:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def calibrate_probs_from_probs(
    probs: np.ndarray,
    temperature: float,
    eps: float = 1e-12,
) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), eps, None)
    pseudo_logits = np.log(probs)
    return softmax(pseudo_logits / float(temperature), axis=-1)


def calibrate_probs_from_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return softmax(logits / float(temperature), axis=-1)


def top_k_predictions(
    probs: np.ndarray,
    labels: list[str],
    top_k: int,
) -> list[dict[str, float | str]]:
    probs = np.asarray(probs, dtype=np.float64)
    top_k = min(top_k, len(labels))
    indices = np.argsort(probs)[::-1][:top_k]

    return [
        {
            "label": str(labels[idx]),
            "probability": float(probs[idx]),
        }
        for idx in indices
    ]


class SkinLesionPredictor:
    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: str | Path | None = None,
        temperature_json: str | Path | None = None,
        threshold: float | None = None,
        device: str = "auto",
        use_tta: bool | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.runtime_config = load_yaml(self.config_path)

        paths = get_project_paths(self.runtime_config)

        if checkpoint_path is None:
            checkpoint_path = paths.checkpoint_path

        self.checkpoint_path = Path(checkpoint_path)
        self.device = get_device(
            device or self.runtime_config.get("inference", {}).get("device", "auto")
        )

        self.temperature = load_temperature(temperature_json)

        if threshold is None:
            threshold = self.runtime_config.get(
                "evaluate",
                {},
            ).get(
                "confidence_threshold",
                DEFAULT_THRESHOLD,
            )

        self.threshold = float(threshold)

        checkpoint = load_checkpoint(self.checkpoint_path, self.device)
        self.config = build_inference_config(self.runtime_config, checkpoint)
        self.label_to_idx, self.labels = make_label_maps(checkpoint, self.config)

        use_metadata = bool(self.config["train"].get("use_metadata", False))
        self.metadata_schema: MetadataSchema | None = None

        if use_metadata:
            train_csv = Path(self.config["data"]["train_csv"])
            self.metadata_schema = build_metadata_schema(train_csv)
            self.config["train"]["metadata_dim"] = self.metadata_schema.dim
            print(f"[INFO] Metadata fusion enabled. metadata_dim={self.metadata_schema.dim}")

        self.use_tta, self.tta_transforms = get_tta_config(self.config, override_use_tta=use_tta)

        self.model = build_model_from_config(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"[INFO] Loaded checkpoint: {self.checkpoint_path}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Labels: {self.labels}")
        print(f"[INFO] Temperature: {self.temperature:.6f}")
        print(f"[INFO] Threshold: {self.threshold:.2f}")
        print(f"[INFO] TTA: {self.use_tta}, transforms={self.tta_transforms}")

    @torch.inference_mode()
    def predict_proba(
        self,
        image_tensor: torch.Tensor,
        metadata_tensor: torch.Tensor | None,
    ) -> np.ndarray:
        batch: dict[str, Any] = {"image": image_tensor.to(self.device)}

        if metadata_tensor is not None:
            batch["metadata"] = metadata_tensor.to(self.device)

        if not self.use_tta:
            logits = forward_batch(self.model, batch, self.device)
            probs = torch.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()[0]

        probs_sum: torch.Tensor | None = None
        base_images = batch["image"]

        for transform_name in self.tta_transforms:
            tta_batch = dict(batch)
            tta_batch["image"] = apply_tta_transform(base_images, transform_name)
            logits = forward_batch(self.model, tta_batch, self.device)
            probs = torch.softmax(logits, dim=1)

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum = probs_sum + probs

        if probs_sum is None:
            raise RuntimeError("TTA produced no probabilities.")

        probs = probs_sum / float(len(self.tta_transforms))
        return probs.detach().cpu().numpy()[0]

    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        *,
        age: float | int | None,
        sex: str,
        anatom_site: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> PredictionResult:

        image_tensor = build_single_image_preprocessor_tensor(image, self.config)

        metadata_tensor = None
        if self.metadata_schema is not None:
            metadata_tensor = build_metadata_tensor(
                age=age,
                sex=sex,
                anatom_site=anatom_site,
                metadata_schema=self.metadata_schema,
            )

        raw_probs = self.predict_proba(image_tensor, metadata_tensor)
        calibrated_probs = calibrate_probs_from_probs(raw_probs, self.temperature)

        pred_idx = int(np.argmax(calibrated_probs))
        prediction = self.labels[pred_idx]
        calibrated_confidence = float(calibrated_probs[pred_idx])
        raw_confidence = float(np.max(raw_probs))

        return PredictionResult(
            prediction=prediction,
            calibrated_confidence=calibrated_confidence,
            needs_review=calibrated_confidence < self.threshold,
            threshold=self.threshold,
            top_k=top_k_predictions(calibrated_probs, self.labels, top_k),
            raw_confidence=raw_confidence,
            temperature=self.temperature,
            use_tta=self.use_tta,
            tta_transforms=self.tta_transforms,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated skin lesion inference.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--sex", type=str, default="unknown")
    parser.add_argument("--anatom-site", type=str, default="unknown")

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=("Optional checkpoint override. Defaults to the checkpoint derived from --config."),
    )

    parser.add_argument(
        "--temperature-json",
        type=Path,
        default=None,
        help=(
            "Optional calibration file for this PyTorch inference flow. "
            "No calibration is applied when omitted."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=("Optional confidence threshold override. Defaults to evaluate.confidence_threshold."),
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--no-tta", action="store_true", help="Disable TTA even if config enables it."
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)

    args = parser.parse_args()

    predictor = SkinLesionPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        temperature_json=args.temperature_json,
        threshold=args.threshold,
        device=args.device,
        use_tta=False if args.no_tta else None,
    )

    result = predictor.predict(
        image=args.image,
        age=args.age,
        sex=args.sex,
        anatom_site=args.anatom_site,
        top_k=args.top_k,
    )

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
