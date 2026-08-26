from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    artifact_dir: Path
    run_dir: Path
    checkpoint_path: Path
    reports_dir: Path
    onnx_dir: Path
    onnx_path: Path
    deployment_eval_dir: Path
    calibration_dir: Path
    selective_prediction_dir: Path
    error_analysis_dir: Path
    benchmark_dir: Path
    deployment_bundle_dir: Path


def get_project_paths(config: dict[str, Any]) -> ProjectPaths:
    project = config.get("project", {})

    experiment_name = str(project.get("experiment_name", "")).strip()

    if not experiment_name:
        raise ValueError("project.experiment_name must be a non empty string.")

    artifact_dir = Path(project.get("artifact_dir", "artifacts"))

    run_dir = artifact_dir / "runs" / experiment_name

    deployment_config = config.get("deployment", {})
    configured_bundle = deployment_config.get("bundle_dir")

    deployment_bundle_dir = (
        Path(configured_bundle) if configured_bundle else artifact_dir / "deployment" / "model"
    )

    return ProjectPaths(
        artifact_dir=artifact_dir,
        run_dir=run_dir,
        checkpoint_path=run_dir / "checkpoints" / "best.ckpt",
        reports_dir=run_dir / "reports",
        onnx_dir=run_dir / "onnx",
        onnx_path=run_dir / "onnx" / "model.onnx",
        deployment_eval_dir=run_dir / "deployment_eval",
        calibration_dir=run_dir / "calibration",
        selective_prediction_dir=run_dir / "selective_prediction",
        error_analysis_dir=run_dir / "error_analysis",
        benchmark_dir=run_dir / "benchmark",
        deployment_bundle_dir=deployment_bundle_dir,
    )
