from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import yaml

from lesion_ml.models.factory import build_model_from_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to ONNX.")
    parser.add_argument("--config", type=str, default="params.yaml")
    return parser.parse_args()

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")
    

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    checkpoint_path = Path(config["inference"]["checkpoint_path"])
    onnx_path = Path(config["inference"]["onnx_path"])
    image_size = int(config["data"]["image_size"])

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path)

    model_config = copy.deepcopy(checkpoint.get("config", config))

    model_config.setdefault("train", {})
    model_config["train"]["pretrained"] = False

    model = build_model_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    print(f"[INFO] Exporting ONNX model to: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
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

    print("[INFO] ONNX export finished.")

if __name__ == "__main__":
    main()
