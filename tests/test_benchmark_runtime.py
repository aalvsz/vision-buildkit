import math

import pytest
from PIL import Image

from vision.benchmark import _digest, benchmark_model, verify_benchmark_report


class Runner:
    def __call__(self, image):
        return image.size

    def runtime_identity(self):
        return {
            "backend": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "device": "cpu",
            "precision": "fp32",
            "input_shape": [1, 3, 8, 8],
        }

    def predict_batch(self, images):
        return [image.size for image in images]


def test_benchmark_has_raw_samples_and_bound_fingerprints():
    report = benchmark_model(Runner(), Image.new("RGB", (8, 8)), repeats=5, warmup=1)
    assert len(report["samples_ms"]) == 5
    assert all(math.isfinite(value) and value > 0 for value in report["samples_ms"])
    assert report["runtime"]["provider"] == "CPUExecutionProvider"
    assert report["workload"]["input"]["size"] == [8, 8]
    assert report["throughput_items_per_s"] == pytest.approx(1000 / report["end_to_end_ms"]["mean"])
    assert all(len(report[key]) == 64 for key in ("digest", "hardware_digest", "workload_digest", "harness_digest"))


def test_benchmark_measures_real_batch_and_derives_throughput():
    report = benchmark_model(Runner(), Image.new("RGB", (8, 8)), repeats=3, warmup=1, batch=4)
    assert report["workload"]["batch"] == 4
    assert report["throughput_items_per_s"] == pytest.approx(4000 / report["end_to_end_ms"]["mean"])


@pytest.mark.parametrize(
    ("repeats", "warmup"),
    [(0, 0), (-1, 0), (1, -1)],
)
def test_benchmark_rejects_invalid_counts(repeats, warmup):
    with pytest.raises(ValueError):
        benchmark_model(Runner(), Image.new("RGB", (8, 8)), repeats=repeats, warmup=warmup)


def test_cuda_identity_fails_without_synchronization(monkeypatch):
    class CudaRunner(Runner):
        def runtime_identity(self):
            return Runner.runtime_identity(self) | {"device": "cuda"}

    import vision.benchmark as module

    monkeypatch.setattr(module, "_synchronize", lambda identity: (_ for _ in ()).throw(RuntimeError("no sync")))
    with pytest.raises(RuntimeError, match="no sync"):
        benchmark_model(CudaRunner(), Image.new("RGB", (8, 8)), repeats=1, warmup=0)


def test_benchmark_report_verifier_recomputes_provenance_and_statistics():
    report = benchmark_model(Runner(), Image.new("RGB", (8, 8)), repeats=5, warmup=1)
    verified = verify_benchmark_report(report)
    assert verified["samples"] == 5
    forged = dict(report)
    forged["end_to_end_ms"] = dict(report["end_to_end_ms"], p95=999.0)
    with pytest.raises(ValueError, match="p95"):
        verify_benchmark_report(forged)


def test_benchmark_report_verifier_rejects_forged_throughput():
    report = benchmark_model(Runner(), Image.new("RGB", (8, 8)), repeats=3, warmup=1)
    forged = dict(report, throughput_items_per_s=1.0)
    with pytest.raises(ValueError, match="throughput"):
        verify_benchmark_report(forged)


def test_benchmark_report_verifier_rejects_repeat_count_mismatch():
    report = benchmark_model(Runner(), Image.new("RGB", (8, 8)), repeats=3, warmup=1)
    workload = dict(report["workload"], repeats=2)
    forged = dict(report, workload=workload)
    forged["workload_digest"] = _digest(workload)
    with pytest.raises(ValueError, match="repeat count"):
        verify_benchmark_report(forged)
