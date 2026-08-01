"""Independent regression gates for the fail-closed optimization primitives."""

import hashlib
import json
import math

import pytest

from vision.manifest import VisionManifest
from vision.optimization import (
    BenchmarkReport,
    BenchmarkRequest,
    CandidateEvaluation,
    CandidatePlan,
    ResolvedArtifactEvidence,
    RuntimeFingerprint,
    SearchPolicy,
    SearchSpace,
    ShapeProfile,
    Stage,
    StageEvidence,
    _comparable,
    digest,
    expected_build_request_digest,
)

HEX = "a" * 64


def plan(**changes):
    value = {
        "backend": "pytorch",
        "precision": "fp32",
        "device": "cpu",
        "provider": "torch",
        "resolution": (640, 640),
        "batch": 1,
        "source_model_digest": HEX,
    }
    value.update(changes)
    return CandidatePlan(**value)


def profile():
    return ShapeProfile(
        input_name="images", min_shape=(1, 3, 320, 320), opt_shape=(1, 3, 640, 640), max_shape=(2, 3, 1280, 1280)
    )


def fingerprint(**changes):
    value = {
        "hardware_digest": HEX,
        "harness_digest": "b" * 64,
        "workload_digest": "c" * 64,
        "input_digest": "d" * 64,
        "artifact_digest": "e" * 64,
        "backend": "pytorch",
        "precision": "fp32",
        "runtime_version": "test",
        "device": "cpu",
        "provider": "torch",
        "threads": 1,
        "mode": "e2e",
    }
    value.update(changes)
    return RuntimeFingerprint(**value)


def material(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"weights")
    manifest_path = tmp_path / "model.yaml"
    manifest = VisionManifest(source=str(artifact), family="torch")
    manifest.to_yaml(manifest_path)
    source = hashlib.sha256(artifact.read_bytes()).hexdigest()
    item = plan(source_model_digest=source)
    template = "b" * 64
    raw = {
        "schema_version": 1,
        "kind": "original",
        "candidate_id": item.candidate_id,
        "plan_digest": item.candidate_id,
        "source_model_digest": source,
        "build_template_digest": template,
        "build_request_digest": expected_build_request_digest(item, template),
        "artifact_path": str(artifact),
        "artifact_digest": source,
        "manifest_path": str(manifest_path),
        "manifest_digest": digest(manifest.model_dump(mode="json")),
        "target_digest": "c" * 64,
        "backend": item.backend,
        "precision": item.precision,
        "device": item.device,
        "provider": item.provider,
        "resolution": item.resolution,
        "batch": item.batch,
        "dynamic": False,
        "profiles": (),
        "upstream_build_report_path": None,
        "upstream_build_report_file_sha256": None,
        "upstream_build_report_self_digest": None,
    }
    record = raw | {"self_digest": digest(raw)}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    evidence = ResolvedArtifactEvidence(
        candidate_id=item.candidate_id,
        artifact_digest=source,
        build_request_digest=raw["build_request_digest"],
        manifest_digest=raw["manifest_digest"],
        target_digest=raw["target_digest"],
        build_report_digest=record["self_digest"],
        artifact_path=str(artifact),
        artifact_sha256=source,
        manifest_path=str(manifest_path),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        build_report_path=str(path),
        build_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    request = BenchmarkRequest(
        fingerprint=fingerprint(artifact_digest=source),
        resolved=evidence,
        resolution=item.resolution,
        dynamic=False,
        warmup=0,
        repeats=2,
        batch=1,
    )
    return request


def test_candidate_plan_rejects_invalid_axes_and_profiles():
    for changes in ({"resolution": (0, 640)}, {"batch": 0}, {"dynamic": True}, {"profiles": (profile(),)}):
        with pytest.raises(ValueError):
            plan(**changes)
    with pytest.raises(ValueError):
        CandidatePlan.model_validate({**plan().model_dump(), "artifact_digest": HEX})


def test_shape_profile_and_search_space_fail_closed():
    with pytest.raises(ValueError):
        ShapeProfile(input_name="x", min_shape=(2,), opt_shape=(1,), max_shape=(3,))
    with pytest.raises(ValueError):
        SearchSpace(backends=())
    with pytest.raises(ValueError):
        SearchSpace(dynamics=(False,), profiles=(profile(),))
    assert SearchSpace(backends=("pytorch", "pytorch"), max_candidates=2).max_candidates > 0


def test_stage_prefix_and_terminal_error_contracts():
    item = plan()
    with pytest.raises(ValueError, match="consecutive"):
        CandidateEvaluation(plan=item, stages=(StageEvidence(stage=Stage.benchmark),))
    stages = tuple(
        StageEvidence(stage=x)
        for x in (
            Stage.capability,
            Stage.build,
            Stage.inspect,
            Stage.parity,
            Stage.dataset_map,
            Stage.benchmark,
            Stage.accepted,
        )
    )
    with pytest.raises(ValueError, match="error"):
        CandidateEvaluation(plan=item, stages=stages, error="bad")


def test_real_resolved_material_positive_control_and_benchmark_derivation(tmp_path):
    report = BenchmarkReport(request=material(tmp_path), samples_ms=(1.0, 3.0), artifact_size_bytes=1)
    assert report.p50_ms == 2.0
    with pytest.raises(ValueError, match="derived"):
        BenchmarkReport(request=report.request, samples_ms=(1.0, 3.0), artifact_size_bytes=1, p50_ms=9)
    with pytest.raises(ValueError, match="sample count"):
        BenchmarkReport(request=report.request, samples_ms=(1.0,), artifact_size_bytes=1)
    with pytest.raises(ValueError, match="digest"):
        BenchmarkReport(request=report.request, samples_ms=(1.0, 3.0), artifact_size_bytes=1, digest="0" * 64)


@pytest.mark.parametrize("sample", [-1.0, 0.0, math.nan, math.inf])
def test_benchmark_sample_domain_is_explicit(tmp_path, sample):
    with pytest.raises(ValueError, match="finite positive"):
        BenchmarkReport(request=material(tmp_path), samples_ms=(sample, 1.0), artifact_size_bytes=1)


@pytest.mark.parametrize(
    "field", ["hardware_digest", "harness_digest", "workload_digest", "input_digest", "threads", "mode"]
)
def test_comparability_requires_identical_harness_and_workload(tmp_path, field):
    left = BenchmarkReport(request=material(tmp_path), samples_ms=(1.0, 1.0), artifact_size_bytes=1)
    value = 2 if field == "threads" else ("other" if field == "mode" else "f" * 64)
    right_request = left.request.model_copy(
        update={"fingerprint": left.request.fingerprint.model_copy(update={field: value})}
    )
    right = BenchmarkReport(request=right_request, samples_ms=(1.0, 1.0), artifact_size_bytes=1)
    assert not _comparable(left, right, SearchPolicy(max_map_drop=0))


def test_comparability_permits_cross_backend_on_same_harness(tmp_path):
    left = BenchmarkReport(request=material(tmp_path), samples_ms=(1.0, 1.0), artifact_size_bytes=1)
    fingerprint_value = left.request.fingerprint.model_copy(
        update={"backend": "onnxruntime", "provider": "CPUExecutionProvider"}
    )
    right = BenchmarkReport(
        request=left.request.model_copy(update={"fingerprint": fingerprint_value}),
        samples_ms=(1.0, 1.0),
        artifact_size_bytes=1,
    )
    assert _comparable(left, right, SearchPolicy(max_map_drop=0))
