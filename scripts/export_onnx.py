from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
import yaml

from lesion_ml.data.metadata import build_metadata_schema, save_metadata_schema
from lesion_ml.models.factory import build_model_from_config
from lesion_ml.paths import get_project_paths


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


def write_onnx_sidecars(
    model_config: dict[str, Any],
    onnx_path: Path,
    image_size: int,
    use_metadata: bool,
    metadata_dim: int | None,
) -> None:
    onnx_dir = onnx_path.parent
    onnx_dir.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "schema_version": 1,
        "experiment_name": model_config["project"]["experiment_name"],
        "labels": model_config["data"]["labels"],
        "image_size": int(image_size),
        "uses_metadata": bool(use_metadata),
        "metadata_dim": (int(metadata_dim) if metadata_dim is not None else None),
        "metadata_schema_path": ("metadata_schema.json" if use_metadata else None),
        "onnx_inputs": (["image", "metadata"] if use_metadata else ["image"]),
        "onnx_output": "logits",
        "preprocessing": {
            "mode": model_config.get(
                "preprocess",
                {},
            ).get("mode", "none"),
            "resize_mode": model_config.get(
                "preprocess",
                {},
            ).get("resize_mode", "resize_pad"),
            "normalize": bool(
                model_config.get(
                    "augment",
                    {},
                ).get("normalize", True)
            ),
            "lesion_crop_margin": float(
                model_config.get(
                    "preprocess",
                    {},
                ).get("lesion_crop_margin", 0.20)
            ),
            "dark_border_threshold": int(
                model_config.get(
                    "preprocess",
                    {},
                ).get("dark_border_threshold", 10)
            ),
        },
    }

    metadata_json_path = onnx_dir / "model.metadata.json"
    metadata_json_path.write_text(
        json.dumps(sidecar, indent=2),
        encoding="utf-8",
    )

    if use_metadata:
        schema = build_metadata_schema(model_config["data"]["train_csv"])
        if metadata_dim is not None and schema.dim != int(metadata_dim):
            raise ValueError(f"metadata_dim mismatch: config={metadata_dim}, schema={schema.dim}")

        schema_path = onnx_dir / "metadata_schema.json"
        save_metadata_schema(schema, schema_path)

        print(f"[INFO] Wrote metadata schema: {schema_path}")

    print(f"[INFO] Wrote ONNX metadata: {metadata_json_path}")


def main() -> None:
    args = parse_args()
    runtime_config = load_config(args.config)

    paths = get_project_paths(runtime_config)
    checkpoint_path = paths.checkpoint_path
    onnx_path = paths.onnx_path

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

    write_onnx_sidecars(
        model_config=model_config,
        onnx_path=onnx_path,
        image_size=image_size,
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
    )

    onnx_model = onnx.load(
        str(onnx_path),
        load_external_data=True,
    )
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    actual_inputs = {input_meta.name for input_meta in session.get_inputs()}

    expected_inputs = {"image", "metadata"} if use_metadata else {"image"}

    if actual_inputs != expected_inputs:
        raise RuntimeError(
            "Unexpected ONNX inputs. "
            f"expected={sorted(expected_inputs)}, "
            f"actual={sorted(actual_inputs)}"
        )

    actual_outputs = {output_meta.name for output_meta in session.get_outputs()}

    if "logits" not in actual_outputs:
        raise RuntimeError(
            f"ONNX output 'logits' was not found. actual_outputs={sorted(actual_outputs)}"
        )

    print(f"[INFO] ONNX validation passed: {onnx_path}")

    print("[INFO] ONNX export finished.")


if __name__ == "__main__":
    main()
