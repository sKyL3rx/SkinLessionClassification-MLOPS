from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from lesion_ml.paths import get_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
    )

    return parser.parse_args()


def load_config(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def copy_required(
    source: Path,
    destination_dir: Path,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Required bundle file missing: {source}")

    destination = destination_dir / source.name

    shutil.copy2(
        source,
        destination,
    )

    return destination


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = get_project_paths(config)

    bundle_dir = paths.deployment_bundle_dir

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = paths.onnx_dir / "model.metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required_sources = [
        paths.onnx_path,
        metadata_path,
        paths.calibration_dir / "temperature.json",
        paths.calibration_dir / "decision.json",
    ]

    if bool(metadata.get("uses_metadata", False)):
        required_sources.append(paths.onnx_dir / "metadata_schema.json")

    external_data_path = Path(str(paths.onnx_path) + ".data")

    if external_data_path.exists():
        required_sources.append(external_data_path)

    copied_files = [
        copy_required(
            source,
            bundle_dir,
        )
        for source in required_sources
    ]

    config_snapshot = bundle_dir / "params.yaml"

    shutil.copy2(
        args.config,
        config_snapshot,
    )

    copied_files.append(config_snapshot)

    manifest = {
        "schema_version": 1,
        "experiment_name": (config["project"]["experiment_name"]),
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": (path.stat().st_size),
            }
            for path in sorted(copied_files)
        },
    }

    manifest_path = bundle_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] Bundle: {bundle_dir}")

    for path in sorted(bundle_dir.iterdir()):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
