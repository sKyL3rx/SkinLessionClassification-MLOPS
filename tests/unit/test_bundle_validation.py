from pathlib import Path

import pytest

REQUIRED_FILES = {
    "model.onnx",
    "model.metadata.json",
    "temperature.json",
    "decision.json",
    "manifest.json",
}


@pytest.mark.artifact
def test_bundle_has_required_files() -> None:
    bundle_dir = Path("artifacts/deployment/model")

    existing = {path.name for path in bundle_dir.iterdir()}

    assert REQUIRED_FILES <= existing
