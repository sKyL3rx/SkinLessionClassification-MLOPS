from pathlib import Path

from lesion_ml.paths import get_project_paths


def test_project_paths() -> None:
    config = {
        "project": {
            "experiment_name": "example",
            "artifact_dir": "artifacts",
        },
        "deployment": {"bundle_dir": ("artifacts/deployment/model")},
    }

    paths = get_project_paths(config)

    assert paths.checkpoint_path == (
        Path("artifacts") / "runs" / "example" / "checkpoints" / "best.ckpt"
    )

    assert paths.onnx_path == (Path("artifacts") / "runs" / "example" / "onnx" / "model.onnx")
