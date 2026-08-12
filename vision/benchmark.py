from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _percentile(samples: list[float], percent: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _runtime_identity(model: Any) -> dict[str, Any]:
    identity = getattr(model, "runtime_identity", None)
    if not callable(identity):
        return {"backend": "unknown", "provider": type(model).__qualname__, "device": "unknown"}
    value = identity()
    if not isinstance(value, dict):
        raise TypeError("runtime_identity() must return a dict")
    return value


def _synchronize(identity: dict[str, Any]) -> None:
    if identity.get("device") != "cuda":
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA benchmark requires torch synchronization support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("runtime reports CUDA but torch cannot synchronize CUDA")
    torch.cuda.synchronize()


def _image_identity(image: Any) -> dict[str, Any]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.is_symlink() or not path.is_file():
            raise ValueError("benchmark image path must be a regular non-symlink file")
        with Image.open(path) as opened:
            shape = [opened.width, opened.height]
        return {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": shape,
            "mode": "file",
        }
    if isinstance(image, Image.Image):
        converted = image.convert("RGB")
        return {
            "sha256": hashlib.sha256(converted.tobytes()).hexdigest(),
            "size": [converted.width, converted.height],
            "mode": "memory-rgb",
        }
    return {"sha256": _digest(repr(image)), "size": None, "mode": type(image).__qualname__}


def benchmark_model(
    model: Any,
    image: Any,
    repeats: int = 20,
    warmup: int = 3,
    batch: int = 1,
) -> dict[str, Any]:
    """Measure end-to-end inference with explicit workload provenance."""

    if repeats <= 0 or warmup < 0 or batch <= 0:
        raise ValueError("repeats/batch must be positive and warmup non-negative")
    identity = _runtime_identity(model)
    if batch == 1:

        def invoke():
            return model(image)
    else:
        predict_batch = getattr(model, "predict_batch", None)
        if not callable(predict_batch):
            raise ValueError("batch benchmark requires a predict_batch implementation")

        def invoke():
            return predict_batch([image] * batch)

    workload = {
        "kind": "end_to_end_detection",
        "batch": batch,
        "repeats": repeats,
        "warmup": warmup,
        "input": _image_identity(image),
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    for _ in range(warmup):
        invoke()
        _synchronize(identity)

    samples: list[float] = []
    for _ in range(repeats):
        _synchronize(identity)
        started = time.perf_counter_ns()
        invoke()
        _synchronize(identity)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise RuntimeError("benchmark produced a non-positive or non-finite latency")
        samples.append(elapsed_ms)

    mean_ms = statistics.fmean(samples)
    report: dict[str, Any] = {
        "schema_version": 1,
        "samples_ms": samples,
        "end_to_end_ms": {
            "p50": _percentile(samples, 50),
            "p95": _percentile(samples, 95),
            "p99": _percentile(samples, 99),
            "mean": mean_ms,
            "std": statistics.pstdev(samples),
        },
        "throughput_items_per_s": batch * 1000 / mean_ms,
        "runtime": identity,
        "environment": environment,
        "hardware_digest": _digest(environment),
        "workload": workload,
        "workload_digest": _digest(workload),
        "harness": {
            "clock": "time.perf_counter_ns",
            "synchronization": "torch.cuda.synchronize" if identity.get("device") == "cuda" else "blocking-call",
        },
    }
    report["harness_digest"] = _digest(report["harness"])
    report["digest"] = _digest(report)
    return report


def verify_benchmark_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate a benchmark report's derived statistics and provenance digests."""

    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("benchmark report must be an object with schema_version=1")
    samples = report.get("samples_ms")
    if (
        not isinstance(samples, list)
        or not samples
        or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in samples)
    ):
        raise ValueError("benchmark samples_ms must be a non-empty list of finite positive numbers")
    stats = report.get("end_to_end_ms")
    if not isinstance(stats, dict):
        raise ValueError("benchmark report lacks end_to_end_ms statistics")
    expected_stats = {
        "p50": _percentile(samples, 50),
        "p95": _percentile(samples, 95),
        "p99": _percentile(samples, 99),
        "mean": statistics.fmean(samples),
        "std": statistics.pstdev(samples),
    }
    for key, expected in expected_stats.items():
        actual = stats.get(key)
        if (
            not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"benchmark statistic {key} does not match samples_ms")
    workload = report.get("workload")
    environment = report.get("environment")
    harness = report.get("harness")
    if not all(isinstance(value, dict) for value in (workload, environment, harness)):
        raise ValueError("benchmark workload, environment, and harness must be objects")
    if report.get("workload_digest") != _digest(workload):
        raise ValueError("benchmark workload digest mismatch")
    if report.get("hardware_digest") != _digest(environment):
        raise ValueError("benchmark hardware digest mismatch")
    if report.get("harness_digest") != _digest(harness):
        raise ValueError("benchmark harness digest mismatch")
    if workload.get("repeats") != len(samples):
        raise ValueError("benchmark workload repeat count does not match samples_ms")
    batch = workload.get("batch")
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError("benchmark workload batch must be a positive integer")
    throughput = report.get("throughput_items_per_s")
    expected_throughput = batch * 1000 / expected_stats["mean"]
    if (
        not isinstance(throughput, (int, float))
        or not math.isfinite(throughput)
        or not math.isclose(throughput, expected_throughput, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("benchmark throughput does not match samples_ms and batch")
    digest = report.get("digest")
    unsigned = dict(report)
    unsigned.pop("digest", None)
    if digest != _digest(unsigned):
        raise ValueError("benchmark report digest mismatch")
    return {
        "schema_version": 1,
        "samples": len(samples),
        "p50_ms": expected_stats["p50"],
        "p95_ms": expected_stats["p95"],
        "workload_digest": report["workload_digest"],
        "hardware_digest": report["hardware_digest"],
        "harness_digest": report["harness_digest"],
        "report_digest": digest,
    }
