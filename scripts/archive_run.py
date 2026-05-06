from __future__ import annotations

import argparse
import shutil
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive latest training artifacts into a named run folder.")
    parser.add_argument("--experiment-name", required=True, help="Name of the experiment/run.")
    parser.add_argument("--artifact-dir", default="artifacts", help="Root artifact directory.")
    parser.add_argument("--onnx-path", default="deployment/onnx/model.onnx", help="Latest ONNX model path.")
    parser.add_argument(
        "--dvc-add",
        action="store_true",
        help="Run `dvc add` on the archived run folder after copying artifacts.",
    )
    return parser.parse_args()

def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[WARN] Missing artifact, skipping: {src}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[INFO] Copied {src} -> {dst}")



def main() -> None:
    args = parse_args()

    artifact_dir = Path(args.artifact_dir)
    exp_name = args.experiment_name

    run_dir = artifact_dir / "runs" / exp_name
    checkpoint_dir = run_dir / "checkpoints"
    report_dir = run_dir / "reports"
    onnx_dir = run_dir / "onnx"

    copy_if_exists(
        artifact_dir / "models" / "best.ckpt",
        checkpoint_dir / "best.ckpt",
    )

    # Reports
    report_files = [
        "train_metrics.json",
        "train_history.csv",
        "test_metrics.json",
        "confusion_matrix.csv",
        "per_class_metrics.csv",
        "predictions.csv",
    ]

    for file_name in report_files:
        copy_if_exists(
            artifact_dir / "reports" / file_name,
            report_dir / file_name,
        )

    # ONNX
    onnx_path = Path(args.onnx_path)
    copy_if_exists(
        onnx_path,
        onnx_dir / "model.onnx",
    )

    # Optional ONNX external data file
    onnx_data_path = Path(str(onnx_path) + ".data")
    copy_if_exists(
        onnx_data_path,
        onnx_dir / "model.onnx.data",
    )

    copy_if_exists(
        Path("params.yaml"),
        run_dir / "params.yaml",
    )

    print(f"[INFO] Archived run to: {run_dir}")

    if args.dvc_add:
        import subprocess

        print(f"[INFO] Running: dvc add {run_dir}")
        subprocess.run(["dvc", "add", str(run_dir)], check=True)
        print(f"[INFO] DVC pointer created: {run_dir}.dvc")


if __name__ == "__main__":
    main()