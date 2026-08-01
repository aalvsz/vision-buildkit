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
