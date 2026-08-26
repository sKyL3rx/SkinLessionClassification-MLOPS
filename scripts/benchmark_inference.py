from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime inference.")

    parser.add_argument(
        "--onnx-path",
        type=Path,
        required=True,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32],
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution provider to use.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_providers(
    provider: str,
) -> list[str]:
    available = ort.get_available_providers()

    if provider == "cpu":
        return [
            "CPUExecutionProvider",
        ]

    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available. Available providers: {available}"
            )

        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    if "CUDAExecutionProvider" in available:
        return [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    return [
        "CPUExecutionProvider",
    ]


def percentile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        return float("nan")

    return float(
        np.percentile(
            np.array(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def resolve_static_dimension(
    value: object,
    fallback: int,
) -> int:
    if isinstance(value, int) and value > 0:
        return value

    return fallback


def make_dummy_feed(
    session: ort.InferenceSession,
    *,
    batch_size: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}

    for input_meta in session.get_inputs():
        if input_meta.name == "image":
            feed[input_meta.name] = np.random.randn(
                batch_size,
                3,
                image_size,
                image_size,
            ).astype(np.float32)

        elif input_meta.name == "metadata":
            metadata_dim = resolve_static_dimension(
                input_meta.shape[-1],
                fallback=19,
            )

            feed[input_meta.name] = np.zeros(
                (
                    batch_size,
                    metadata_dim,
                ),
                dtype=np.float32,
            )

        else:
            raise RuntimeError(f"Unsupported ONNX input: {input_meta.name}")

    if "image" not in feed:
        raise RuntimeError("ONNX model does not expose an 'image' input.")

    return feed


def benchmark_batch(
    session: ort.InferenceSession,
    *,
    batch_size: int,
    image_size: int,
    warmup_runs: int,
    runs: int,
) -> dict[str, Any]:
    feed = make_dummy_feed(
        session,
        batch_size=batch_size,
        image_size=image_size,
    )

    for _ in range(warmup_runs):
        session.run(
            None,
            feed,
        )

    latencies_ms: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()

        session.run(
            None,
            feed,
        )

        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000.0)

    mean_ms = statistics.mean(latencies_ms)

    std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    return {
        "batch_size": int(batch_size),
        "runs": int(runs),
        "warmup_runs": int(warmup_runs),
        "mean_latency_ms": float(mean_ms),
        "std_latency_ms": float(std_ms),
        "min_latency_ms": float(min(latencies_ms)),
        "p50_latency_ms": percentile(
            latencies_ms,
            50,
        ),
        "p95_latency_ms": percentile(
            latencies_ms,
            95,
        ),
        "p99_latency_ms": percentile(
            latencies_ms,
            99,
        ),
        "max_latency_ms": float(max(latencies_ms)),
        "throughput_images_per_sec": (float(batch_size / (mean_ms / 1000.0))),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def get_onnx_artifact_size_mb(
    onnx_path: Path,
) -> float:
    total_bytes = onnx_path.stat().st_size

    external_data_path = onnx_path.with_name(onnx_path.name + ".data")

    if external_data_path.exists():
        total_bytes += external_data_path.stat().st_size

    return float(total_bytes / 1024 / 1024)


def main() -> None:
    args = parse_args()

    onnx_path = args.onnx_path

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model_size_mb = get_onnx_artifact_size_mb(onnx_path)

    requested_provider = args.provider.strip().lower()

    if requested_provider == "cuda":
        ort.preload_dlls(directory="")

    providers = resolve_providers(requested_provider)

    print(f"[INFO] ONNX path: {onnx_path}")
    print(f"[INFO] ONNX artifact size: {model_size_mb:.2f} MB")
    print(f"[INFO] Available providers: {ort.get_available_providers()}")
    print(f"[INFO] Requested provider: {requested_provider}")
    print(f"[INFO] Requested session providers: {providers}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=providers,
    )

    active_providers = session.get_providers()

    print(f"[INFO] Active session providers: {active_providers}")

    if requested_provider == "cuda" and (
        not active_providers or active_providers[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "CUDA was requested, but ONNX Runtime "
            "fell back to CPU. "
            f"Active providers: {active_providers}"
        )

    for input_meta in session.get_inputs():
        print(
            "[INFO] Input: "
            f"name={input_meta.name}, "
            f"shape={input_meta.shape}, "
            f"type={input_meta.type}"
        )

    results: list[dict[str, Any]] = []

    for batch_size in args.batch_sizes:
        print(f"[INFO] Benchmarking batch_size={batch_size}")

        row = benchmark_batch(
            session=session,
            batch_size=int(batch_size),
            image_size=int(args.image_size),
            warmup_runs=int(args.warmup_runs),
            runs=int(args.runs),
        )

        row = {
            "backend": "onnxruntime",
            "providers": ",".join(active_providers),
            "onnx_path": str(onnx_path),
            "model_size_mb": (model_size_mb),
            "image_size": int(args.image_size),
            **row,
        }

        results.append(row)

        print(
            f"mean={row['mean_latency_ms']:.2f} ms | "
            f"p50={row['p50_latency_ms']:.2f} ms | "
            f"p95={row['p95_latency_ms']:.2f} ms | "
            "throughput="
            f"{row['throughput_images_per_sec']:.2f} img/s"
        )

    out_json = args.out_json
    out_csv = args.out_csv

    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "onnx_path": str(onnx_path),
        "requested_provider": (requested_provider),
        "available_providers": (ort.get_available_providers()),
        "session_providers": (active_providers),
        "results": results,
    }

    with out_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )

    write_csv(
        out_csv,
        results,
    )

    print(f"[INFO] Wrote JSON benchmark to: {out_json}")
    print(f"[INFO] Wrote CSV benchmark to: {out_csv}")


if __name__ == "__main__":
    main()
