from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from lesion_ml.data.metadata import build_metadata_schema
from lesion_ml.models.factory import build_model_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to ONNX.")
    parser.add_argument("--config", type=str, default="params.yaml")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def prepare_model_config(
    checkpoint: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    model_config = copy.deepcopy(checkpoint.get("config", runtime_config))

    model_config.setdefault("train", {})
    model_config["train"]["pretrained"] = False

    if "inference" in runtime_config:
        model_config["inference"] = copy.deepcopy(runtime_config["inference"])

    if "project" in runtime_config:
        model_config.setdefault("project", {})
        model_config["project"].update(runtime_config["project"])

    return model_config


def get_metadata_dim(model_config: dict[str, Any]) -> int | None:
    use_metadata = bool(model_config["train"].get("use_metadata", False))

    if not use_metadata:
        return None

    metadata_dim = model_config["train"].get("metadata_dim")

    if metadata_dim is None:
        metadata_schema = build_metadata_schema(model_config["data"]["train_csv"])
        metadata_dim = metadata_schema.dim
        model_config["train"]["metadata_dim"] = metadata_dim

    return int(metadata_dim)


def export_image_only_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    image_size: int,
) -> None:
    dummy_image = torch.randn(
        1,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        dummy_image,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def export_metadata_fusion_onnx(
    model: torch.nn.Module,
    onnx_path: Path,
    image_size: int,
    metadata_dim: int,
) -> None:
    dummy_image = torch.randn(
        1,
        3,
        image_size,
        image_size,
        dtype=torch.float32,
    )
    dummy_metadata = torch.randn(
        1,
        metadata_dim,
        dtype=torch.float32,
    )

    torch.onnx.export(
        model,
        (dummy_image, dummy_metadata),
        onnx_path.as_posix(),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["image", "metadata"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "metadata": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def main() -> None:
    args = parse_args()
    runtime_config = load_config(args.config)

    checkpoint_path = Path(runtime_config["inference"]["checkpoint_path"])
    onnx_path = Path(runtime_config["inference"]["onnx_path"])

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)

    model_config = prepare_model_config(
        checkpoint=checkpoint,
        runtime_config=runtime_config,
    )

    image_size = int(model_config["data"]["image_size"])
    metadata_dim = get_metadata_dim(model_config)
    use_metadata = metadata_dim is not None

    if use_metadata:
        print(f"[INFO] Exporting metadata-fusion ONNX. metadata_dim={metadata_dim}")
    else:
        print("[INFO] Exporting image-only ONNX.")

    model = build_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting ONNX model to: {onnx_path}")

    if use_metadata:
        export_metadata_fusion_onnx(
            model=model,
            onnx_path=onnx_path,
            image_size=image_size,
            metadata_dim=metadata_dim,
        )
    else:
        export_image_only_onnx(
            model=model,
            onnx_path=onnx_path,
            image_size=image_size,
        )

    print("[INFO] ONNX export finished.")


if __name__ == "__main__":
    main()