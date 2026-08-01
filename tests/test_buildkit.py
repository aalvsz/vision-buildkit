import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from vision.buildkit import BuildPlanner
from vision.capsule import RuntimeRequest, create_certified_capsule, select_runtime, verify_capsule
from vision.cli import app
from vision.conformance import (
    AccuracyEvidence,
    CertificationReport,
    ConformanceEvidenceBundle,
    ConformancePolicy,
    EvidenceFile,
    GateDetail,
    GoldenCase,
    GoldenCorpus,
    RuntimeEvidence,
    _parity_within_policy,
    _prediction_parity_diagnostics,
    _prediction_parity_report,
    _repeatable,
    conform,
)
from vision.contract import VisionContract
from vision.manifest import VisionManifest, manifest_from_contract
from vision.plugins import (
    MaterializedOnnx,
    OnnxRuntimeTargetPlugin,
    PluginRegistry,
    PreparedArtifact,
    PreparedSource,
    SourceAdapter,
    TargetPlugin,
    TensorRTTargetPlugin,
)
from vision.providers import UltralyticsSourceAdapter
from vision.types import Box, DetectionResult


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capsule_id(root):
    """Read a preconfigured trust pin; never treat it as verification."""
    return json.loads((Path(root) / "capsule.json").read_text())["capsule_id"]


def test_checkpoint_provider_records_only_tensor_observed_execution_device():
    class Tensor:
        def __init__(self, value, device=None):
            self.value, self.device = value, device

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    raw = SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=Tensor([[0, 0, 1, 1]], "cuda:7"), conf=Tensor([0.9], "cuda:7"), cls=Tensor([0], "cuda:7")
        ),
        names={0: "person"},
    )
    result = UltralyticsSourceAdapter()._result(raw, (1, 1))
    assert result.metadata["execution_device"] == "cuda:7"

    unknown = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=Tensor([[0, 0, 1, 1]]), conf=Tensor([0.9]), cls=Tensor([0])), names={0: "person"}
    )
    assert "execution_device" not in UltralyticsSourceAdapter()._result(unknown, (1, 1)).metadata


def test_determinism_is_not_governed_by_candidate_parity_tolerances(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    value = contract(sha(source)).model_copy(update={"tolerances": {"box_iou": 0.95, "score_delta": 0.12}})
    left = DetectionResult([Box(0, 0, 10, 10)], [0.9], [0], image_size=(32, 32))
    # This would pass the historical 0.95 IoU / 0.12 score semantic gate.
    right = DetectionResult([Box(0.1, 0, 10.1, 10)], [0.95], [0], image_size=(32, 32))
    assert not _repeatable(left, right, value)


def put(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return EvidenceFile(path=str(path), sha256=sha(path))


def contract(source_digest, *, preprocessing=None):
    return VisionContract.model_validate(
        {
            "input": {"color": "RGB", "layout": "NCHW", "dtype": "float32", "input_name": "x", "shape": [1, 3, 32, 32]},
            "preprocessing": preprocessing
            or [
                {
                    "kind": "letterbox",
                    "height": 32,
                    "width": 32,
                    "pad_value": 114,
                    "interpolation": "bilinear",
                    "placement": "center",
                    "rounding": "round_half_up",
                    "scale": "min_ratio",
                }
            ],
            "raw_output": {
                "semantics": "boxes_scores_classes",
                "tensors": [
                    {"name": "b", "dtype": "float32", "shape": [1, "N", 4]},
                    {"name": "s", "dtype": "float32", "shape": [1, "N"]},
                    {"name": "c", "dtype": "int64", "shape": [1, "N"]},
                ],
                "roles": {"boxes": "b", "scores": "s", "classes": "c"},
            },
            "result": {
                "box_format": "xyxy",
                "coordinate_space": "original_image",
                "nms_owner": "source",
                "nms_iou_threshold": 0.5,
                "ordering": "score_desc",
                "tie_policy": "stable",
                "score_threshold": 0.0,
                "class_aware_nms": True,
                "clip_boxes": True,
                "max_detections": 10,
                "suppression": "greedy_iou",
            },
            "labels": ["person"],
            "coco_category_ids": [1],
            "tolerances": {"box_iou": 0.9, "score_delta": 0.01},
            "provenance": {"source": "synthetic", "source_sha256": source_digest},
        }
    )


class Source(SourceAdapter):
    plugin_id, plugin_version = "source", "1"
    capabilities = frozenset({"fp32", "static", "nms:source"})

    def capture(self, source, contract):
        return Path(source).read_bytes()

    def run(self, captured, sample):
        if captured.material_path.read_bytes() != b"source":
            raise ValueError("synthetic source material changed")
        return DetectionResult([Box(0, 0, 1, 1)], [1.0], [0], image_size=(1, 1))


class Target(TargetPlugin):
    plugin_id, plugin_version = "target", "1"
    capabilities = frozenset({"fp32", "static", "nms:source"})

    def build(self, captured, candidate, output=None):
        return captured

    def run(self, artifact, sample):
        if artifact.material_path.read_bytes() != b"artifact":
            raise ValueError("synthetic artifact material changed")
        return DetectionResult([Box(0, 0, 1, 1)], [1.0], [0], image_size=(1, 1))

    def verify_artifact(self, artifact, contract):
        return artifact.material_path.is_file() and artifact.material_path.read_bytes() == b"artifact"

    def prepare(self, path, contract):
        path = Path(path)
        return PreparedArtifact(
            path,
            sha(path),
            self.plugin_id,
            self.plugin_version,
            contract.digest,
        )


def files(
    tmp_path,
    source_digest,
    artifact_digest,
    *,
    metric=1.0,
    image_bytes=b"image",
    case_bytes=b"case",
    source_plugin="source",
    target_plugin="target",
    source_version="1",
    target_version="1",
    parity_schema_version=2,
):
    case = tmp_path / "case.bin"
    case.write_bytes(case_bytes)
    manifest = put(tmp_path, "golden.json", {"cases": [{"case_id": "a", "sha256": sha(case)}]})
    image = tmp_path / "x"
    image.write_bytes(image_bytes)
    dataset = put(
        tmp_path,
        "dataset.json",
        {
            "images": [{"id": 1, "width": 1, "height": 1, "file_name": "x", "sha256": sha(image)}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}],
        },
    )
    policy = put(
        tmp_path,
        "policy.json",
        {
            "metric_name": "map",
            "maximum_drop": 0.5,
            "golden_parity_lost": 0,
            "golden_parity_added": 0,
            "mean_iou_min": 0.95,
            "min_iou_min": 0.85,
            "max_score_delta": 0.12,
        },
    )
    predictions = put(
        tmp_path, "predictions.json", [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0}]
    )
    source_predictions = put(
        tmp_path, "source-predictions.json", [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0}]
    )
    metrics = put(
        tmp_path,
        "metrics.json",
        {
            **({"parity_schema_version": 2} if parity_schema_version == 2 else {}),
            "metric_name": "map",
            "metric_value": metric,
            "source_metric_value": metric,
            "source_predictions_path": Path(source_predictions.path).name,
            "source_predictions_sha256": source_predictions.sha256,
            "parity": _prediction_parity_report(
                Path(source_predictions.path), Path(predictions.path), (1,), schema_version=parity_schema_version
            ),
            **(
                {
                    "parity_diagnostics": _prediction_parity_diagnostics(
                        Path(source_predictions.path), Path(predictions.path), (1,), schema_version=2
                    )
                }
                if parity_schema_version == 2
                else {}
            ),
        },
    )
    environment = put(tmp_path, "environment.json", {"python": "test"})
    runtime = RuntimeEvidence(
        command=("synthetic",),
        hardware={"cpu": "test"},
        environment={"os": "test"},
        source_digest=source_digest,
        artifact_digest=artifact_digest,
        source_plugin_version=source_version,
        target_plugin_version=target_version,
        backend="target",
        provider="target",
        device="cpu",
        precision="fp32",
        resolution=(32, 32),
        batch=1,
        dynamic=False,
        runtime_version="1",
    )
    record = put(
        tmp_path,
        "runtime.json",
        {
            "source_digest": source_digest,
            "artifact_digest": artifact_digest,
            "source_plugin": source_plugin,
            "target_plugin": target_plugin,
            "source_plugin_version": source_version,
            "target_plugin_version": target_version,
            "command": ["synthetic"],
            "hardware": {"cpu": "test"},
            "environment": {"os": "test"},
            "environment_digest": environment.sha256,
            "backend": "target",
            "provider": "target",
            "device": "cpu",
            "precision": "fp32",
            "resolution": [32, 32],
            "batch": 1,
            "dynamic": False,
            "runtime_version": "1",
            "p50_ms": None,
            "p95_ms": None,
        },
    )
    accuracy = AccuracyEvidence(
        dataset_digest=dataset.sha256,
        prediction_digest=predictions.sha256,
        report_digest=metrics.sha256,
        metric_name="map",
        metric_value=metric,
        evaluator_digest=hashlib.sha256(b"vision.coco-map.v1.pycocotools").hexdigest(),
        parity_schema_version=parity_schema_version,
    )
    golden = GoldenCorpus(cases=(GoldenCase(case_id="a", sha256=sha(case)),), manifest_digest=manifest.sha256)
    bundle = ConformanceEvidenceBundle(
        golden_manifest=manifest,
        case_inputs=(EvidenceFile(path=str(case), sha256=sha(case)),),
        dataset_manifest=dataset,
        policy=policy,
        predictions=predictions,
        source_predictions=source_predictions,
        accuracy_report=metrics,
        runtime_record=record,
        environment_record=environment,
    )
    return golden, runtime, accuracy, bundle


def report(tmp_path, *, parity_schema_version=2, report_schema_version=2):
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    value = contract(sha(source))
    golden, runtime, accuracy, bundle = files(
        tmp_path, sha(source), sha(artifact), parity_schema_version=parity_schema_version
    )

    captured, prepared = Source().prepare(source, value), Target().prepare(artifact, value)

    result = conform(
        Source(),
        Target(),
        captured,
        prepared,
        [b"case"],
        value,
        golden=golden,
        policy_digest=bundle.policy.sha256,
        source_digest=sha(source),
        artifact_digest=sha(artifact),
        runtime=runtime,
        accuracy=accuracy,
        evidence=bundle,
        report_schema_version=report_schema_version,
    )
    return source, artifact, value, result, bundle


def test_contract_and_planner_binds_mutating_source(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"one")
    c = contract(sha(source))
    registry = PluginRegistry()
    registry.register_source(Source())
    registry.register_target(Target())
    first = BuildPlanner().plan(Source(), Target(), c, sha(source)).candidates[0].candidate_id
    source.write_bytes(b"two")
    with pytest.raises(ValueError):
        BuildPlanner().plan(Source(), Target(), c, sha(source))
    assert (
        first != BuildPlanner().plan(Source(), Target(), contract(sha(source)), sha(source)).candidates[0].candidate_id
    )


def test_report_requires_digest_and_file_backed_pass(tmp_path):
    _, _, _, value, _ = report(tmp_path)
    with pytest.raises(ValueError):
        CertificationReport.model_validate(value.model_dump(exclude={"digest"}))
    with pytest.raises(ValueError):
        CertificationReport.trusted_create(**(value.model_dump(exclude={"digest"}) | {"evidence": None}))


def test_generated_schema_v1_capsule_verifies_in_a_clean_fixture(tmp_path):
    """A v1 certificate is self-contained; Phase16/17 verification is external evidence."""
    source, artifact, value, result, _ = report(tmp_path, parity_schema_version=1, report_schema_version=1)
    assert result.schema_version == 1
    assert "parity_schema_version" not in json.loads(Path(result.evidence.accuracy_report.path).read_text())
    serialized = result.model_dump(mode="json")
    assert "parity_schema_version" not in serialized["accuracy"]
    assert CertificationReport.model_validate_json(json.dumps(serialized)).digest == result.digest
    root = create_certified_capsule(
        tmp_path / "v1.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    before = sha(root / "certification-report.json")
    assert verify_capsule(root, expected_capsule_id=capsule_id(root)).certification_status == "PASS"
    assert sha(root / "certification-report.json") == before


@pytest.mark.parametrize(
    "path",
    [
        "autoresearch/vision-buildkit-phase16-260731/run/certified.vcap",
        "autoresearch/vision-buildkit-phase17-axelera-260731/run/certified.vcap",
    ],
)
def test_external_phase_v1_capsule_reports_keep_legacy_digest_shape(path):
    """External preserved evidence; skipped in a clean clone without it."""
    root = Path(path)
    if not root.is_dir():
        pytest.skip("external preserved Phase16/17 evidence is not present")
    raw = json.loads((root / "certification-report.json").read_text())
    report = CertificationReport.model_validate(raw)
    assert report.schema_version == 1
    assert report.model_dump(mode="json") == raw
    assert verify_capsule(root, expected_capsule_id=capsule_id(root)).certification_status == "PASS"


def test_report_and_prediction_parity_schema_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="report and prediction parity schema versions must match"):
        report(tmp_path, parity_schema_version=1, report_schema_version=2)


def test_parity_schema_marker_rejects_boolean(tmp_path):
    source = tmp_path / "source"
    artifact = tmp_path / "artifact"
    source.write_bytes(b"source")
    artifact.write_bytes(b"artifact")
    golden, runtime, accuracy, bundle = files(tmp_path, sha(source), sha(artifact), parity_schema_version=2)
    metrics_path = Path(bundle.accuracy_report.path)
    payload = json.loads(metrics_path.read_text())
    payload["parity_schema_version"] = True
    metrics_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    changed = bundle.model_copy(
        update={"accuracy_report": EvidenceFile(path=str(metrics_path), sha256=sha(metrics_path))}
    )
    changed_accuracy = accuracy.model_copy(update={"report_digest": changed.accuracy_report.sha256})
    with pytest.raises(ValueError, match="exact int 1 or 2"):
        changed.verify(golden, changed_accuracy, runtime, source_plugin="source", target_plugin="target")


def test_v2_report_details_name_exactly_the_failed_gate(tmp_path):
    _, _, _, result, _ = report(tmp_path)
    payload = result.model_dump(exclude={"digest"}, mode="json")
    payload["gates"]["runtime"] = "FAIL"
    payload["status"] = "FAIL"
    with pytest.raises(ValueError, match="exactly the failed gates"):
        CertificationReport.trusted_create(**payload)
    payload["details"] = {
        "runtime": GateDetail(
            gate="artifacts", stage="wrong", reason="wrong owner", exception_type="SyntheticError"
        ).model_dump(mode="json")
    }
    with pytest.raises(ValueError, match="detail gate must match"):
        CertificationReport.trusted_create(**payload)


def test_empty_operational_reason_is_typed_and_unexpected_defects_propagate(tmp_path):
    class EmptyFailure(Target):
        def verify_artifact(self, artifact, contract):
            raise RuntimeError()

    failed, _ = _conform_fixture(tmp_path, EmptyFailure())
    assert failed.details["artifacts"].reason == "RuntimeError raised without a message"

    class ProgrammerDefect(Target):
        def verify_artifact(self, artifact, contract):
            raise AssertionError("must reach launcher")

    with pytest.raises(AssertionError, match="must reach launcher"):
        _conform_fixture(tmp_path, ProgrammerDefect())


def test_evidence_negative_policy_tamper_and_missing(tmp_path):
    source, artifact, value, result, bundle = report(tmp_path)
    assert result.status == "PASS", result.details
    negative = tmp_path / "negative"
    negative.mkdir()
    golden, runtime, accuracy, bad = files(negative, sha(source), sha(artifact), metric=0.1)
    with pytest.raises(ValueError):
        bad.verify(golden, accuracy, runtime, source_plugin="source", target_plugin="target")
    Path(bundle.predictions.path).write_text("tampered")
    captured, prepared = Source().prepare(source, value), Target().prepare(artifact, value)
    failed = conform(
        Source(),
        Target(),
        captured,
        prepared,
        [b"case"],
        value,
        golden=result.golden,
        policy_digest=bundle.policy.sha256,
        source_digest=sha(source),
        artifact_digest=sha(artifact),
        runtime=result.runtime,
        accuracy=result.accuracy,
        evidence=bundle,
    )
    assert failed.status == "FAIL"


def test_v2_parity_diagnostics_are_required_and_bound_for_a_passing_report(tmp_path):
    source = tmp_path / "source"
    artifact = tmp_path / "artifact"
    source.write_bytes(b"source")
    artifact.write_bytes(b"artifact")
    golden, runtime, accuracy, bundle = files(tmp_path, sha(source), sha(artifact), parity_schema_version=2)
    payload = json.loads(Path(bundle.accuracy_report.path).read_text())
    assert "parity_diagnostics" in payload
    payload.pop("parity_diagnostics")
    report_path = Path(bundle.accuracy_report.path)
    report_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    changed = bundle.model_copy(
        update={"accuracy_report": EvidenceFile(path=str(report_path), sha256=sha(report_path))}
    )
    changed_accuracy = accuracy.model_copy(update={"report_digest": changed.accuracy_report.sha256})
    with pytest.raises(ValueError, match="diagnostics are required"):
        changed.verify(golden, changed_accuracy, runtime, source_plugin="source", target_plugin="target")


def _conform_fixture(tmp_path, target=None):
    source, artifact, value, result, bundle = report(tmp_path)
    return conform(
        Source(),
        target or Target(),
        Source().prepare(source, value),
        Target().prepare(artifact, value),
        [b"case"],
        value,
        golden=result.golden,
        policy_digest=bundle.policy.sha256,
        source_digest=sha(source),
        artifact_digest=sha(artifact),
        runtime=result.runtime,
        accuracy=result.accuracy,
        evidence=bundle,
    ), bundle


def test_conformance_artifact_failure_does_not_masquerade_as_runtime_failure(tmp_path):
    class RejectedArtifact(Target):
        def verify_artifact(self, artifact, contract):
            return False

    result, _ = _conform_fixture(tmp_path, RejectedArtifact())
    assert result.gates == {
        "artifacts": "FAIL",
        "accuracy_dataset": "PASS",
        "runtime": "UNTESTED",
        "determinism": "UNTESTED",
    }
    assert result.details["artifacts"].stage == "artifact_verification"


def test_conformance_semantic_and_determinism_failures_are_separate(tmp_path):
    class SemanticFailure(Target):
        def run(self, artifact, sample):
            return DetectionResult([], [], [], image_size=(1, 1))

    semantic, _ = _conform_fixture(tmp_path, SemanticFailure())
    assert semantic.gates["runtime"] == "FAIL"
    assert semantic.gates["determinism"] == "PASS"
    assert semantic.gates["accuracy_dataset"] == "FAIL"  # generated predictions cannot bind to supplied evidence
    assert semantic.details["runtime"].exception_type == "SemanticMismatch"

    class NonDeterministic(Target):
        calls = 0

        def run(self, artifact, sample):
            self.calls += 1
            x = 0 if self.calls == 1 else 0.1
            return DetectionResult([Box(x, 0, 1 + x, 1)], [1.0], [0], image_size=(1, 1))

    nondeterministic, _ = _conform_fixture(tmp_path, NonDeterministic())
    assert nondeterministic.gates["runtime"] == "PASS"
    assert nondeterministic.gates["determinism"] == "FAIL"
    assert nondeterministic.details["determinism"].stage == "repeatability"


def test_phase18_style_prediction_drift_is_accuracy_failure_not_runtime_failure(tmp_path):
    """Equal top-score AP cannot hide four lost and four added output rows."""
    source, artifact, value, result, bundle = report(tmp_path)
    source_rows = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0}] + [
        {"image_id": 1, "category_id": 1, "bbox": [10 + index, 0, 1, 1], "score": 0.1 - index / 100}
        for index in range(4)
    ]
    candidate_rows = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0}] + [
        {"image_id": 1, "category_id": 1, "bbox": [20 + index, 0, 1, 1], "score": 0.1 - index / 100}
        for index in range(4)
    ]
    source_path, candidate_path = Path(bundle.source_predictions.path), Path(bundle.predictions.path)
    source_path.write_text(json.dumps(source_rows, sort_keys=True, separators=(",", ":")))
    candidate_path.write_text(json.dumps(candidate_rows, sort_keys=True, separators=(",", ":")))
    parity = _prediction_parity_report(source_path, candidate_path, (1,), schema_version=2)
    diagnostics = _prediction_parity_diagnostics(source_path, candidate_path, (1,))
    assert parity["lost"] == parity["added"] == 4
    assert [row["rank"] for row in diagnostics["lost_predictions"]] == [2, 3, 4, 5]
    assert all(row["image_id"] == 1 and row["category_id"] == 1 for row in diagnostics["added_predictions"])

    metrics_path = Path(bundle.accuracy_report.path)
    metrics = json.loads(metrics_path.read_text())
    metrics.update(
        {
            "source_predictions_path": source_path.name,
            "source_predictions_sha256": sha(source_path),
            "parity": parity,
            "parity_diagnostics": diagnostics,
        }
    )
    metrics_path.write_text(json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    changed = bundle.model_copy(
        update={
            "source_predictions": EvidenceFile(path=str(source_path), sha256=sha(source_path)),
            "predictions": EvidenceFile(path=str(candidate_path), sha256=sha(candidate_path)),
            "accuracy_report": EvidenceFile(path=str(metrics_path), sha256=sha(metrics_path)),
        }
    )
    changed_accuracy = result.accuracy.model_copy(
        update={"prediction_digest": changed.predictions.sha256, "report_digest": changed.accuracy_report.sha256}
    )
    failed = conform(
        Source(),
        Target(),
        Source().prepare(source, value),
        Target().prepare(artifact, value),
        [b"case"],
        value,
        golden=result.golden,
        policy_digest=changed.policy.sha256,
        source_digest=sha(source),
        artifact_digest=sha(artifact),
        runtime=result.runtime,
        accuracy=changed_accuracy,
        evidence=changed,
    )
    assert failed.gates == {
        "artifacts": "PASS",
        "accuracy_dataset": "FAIL",
        "runtime": "UNTESTED",
        "determinism": "UNTESTED",
    }
    assert failed.details["accuracy_dataset"].stage == "evidence_binding"


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_drop", True),
        ("maximum_drop", "0.5"),
        ("maximum_drop", float("nan")),
        ("maximum_drop", float("inf")),
        ("golden_parity_lost", 0.0),
        ("golden_parity_added", -1),
        ("mean_iou_min", float("-inf")),
        ("min_iou_min", 1.01),
        ("max_score_delta", -0.01),
    ],
)
def test_conformance_policy_rejects_untyped_or_nonfinite_thresholds(tmp_path, field, value):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"artifact")
    golden, runtime, accuracy, bundle = files(tmp_path, sha(source), sha(artifact))
    payload = json.loads(Path(bundle.policy.path).read_text())
    payload[field] = value
    # ``allow_nan=True`` is deliberate: it exercises the hostile JSON bytes
    # accepted by Python's default decoder but forbidden by certificate input.
    policy_path = tmp_path / "bad-policy.json"
    policy_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True))
    bad = bundle.model_copy(update={"policy": EvidenceFile(path=str(policy_path), sha256=sha(policy_path))})
    with pytest.raises(ValueError):
        bad.verify(golden, accuracy, runtime, source_plugin="source", target_plugin="target")


def test_conformance_policy_requires_all_and_only_threshold_fields(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"artifact")
    golden, runtime, accuracy, bundle = files(tmp_path, sha(source), sha(artifact))
    payload = json.loads(Path(bundle.policy.path).read_text())
    payload.pop("min_iou_min")
    payload["not_a_threshold"] = 0
    policy_path = tmp_path / "bad-policy.json"
    policy_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    bad = bundle.model_copy(update={"policy": EvidenceFile(path=str(policy_path), sha256=sha(policy_path))})
    with pytest.raises(ValueError):
        bad.verify(golden, accuracy, runtime, source_plugin="source", target_plugin="target")


def test_evidence_verify_recomputes_parity_views_and_rejects_nonfinite_report(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"artifact")
    golden, runtime, accuracy, bundle = files(tmp_path, sha(source), sha(artifact))
    metrics_path = Path(bundle.accuracy_report.path)
    payload = json.loads(metrics_path.read_text())
    payload["parity"]["per_image"] = []  # forged aggregate used to be accepted
    metrics_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    forged = bundle.model_copy(
        update={"accuracy_report": EvidenceFile(path=str(metrics_path), sha256=sha(metrics_path))}
    )
    forged_accuracy = accuracy.model_copy(update={"report_digest": forged.accuracy_report.sha256})
    with pytest.raises(ValueError, match="parity report mismatch"):
        forged.verify(golden, forged_accuracy, runtime, source_plugin="source", target_plugin="target")

    payload["parity"] = _prediction_parity_report(
        Path(bundle.source_predictions.path), Path(bundle.predictions.path), (1,)
    )
    payload["source_metric_value"] = float("nan")
    metrics_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True))
    nonfinite = bundle.model_copy(
        update={"accuracy_report": EvidenceFile(path=str(metrics_path), sha256=sha(metrics_path))}
    )
    nonfinite_accuracy = accuracy.model_copy(update={"report_digest": nonfinite.accuracy_report.sha256})
    with pytest.raises(ValueError, match="non-finite"):
        nonfinite.verify(golden, nonfinite_accuracy, runtime, source_plugin="source", target_plugin="target")


def test_per_image_and_class_policy_cannot_hide_one_bad_image(tmp_path):
    source_path = tmp_path / "source-predictions.json"
    candidate_path = tmp_path / "candidate-predictions.json"
    source_path.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0},
            ]
        )
    )
    candidate_path.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1], "score": 1.0},
                {"image_id": 2, "category_id": 1, "bbox": [2, 2, 1, 1], "score": 1.0},
            ]
        )
    )
    parity = _prediction_parity_report(source_path, candidate_path, (1, 2), schema_version=2)
    assert parity["mean_iou"] == 1.0  # matched boxes alone appear acceptable
    assert parity["per_image"][1]["mean_iou"] == 0.0
    policy = ConformancePolicy.model_validate(
        {
            "metric_name": "map",
            "maximum_drop": 0,
            "golden_parity_lost": 0,
            "golden_parity_added": 0,
            "mean_iou_min": 0.4,
            "min_iou_min": 0.0,
            "max_score_delta": 0,
        }
    )
    assert not _parity_within_policy(parity, policy)
    # Classes are also reconstructed over all their member detections rather
    # than trusting a producer-supplied aggregate.
    assert parity["per_class"] == [
        {
            "category_id": 1,
            "lost": 1,
            "added": 1,
            "mean_iou": 1.0,
            "min_iou": 1.0,
            "max_score_delta": 0.0,
        }
    ]


def test_schema_v1_zero_overlap_keeps_the_historical_same_class_matcher(tmp_path):
    source_path = tmp_path / "source.json"
    candidate_path = tmp_path / "candidate.json"
    source_path.write_text('[{"image_id":1,"category_id":1,"bbox":[0,0,1,1],"score":0.9}]')
    candidate_path.write_text('[{"image_id":1,"category_id":1,"bbox":[3,3,1,1],"score":0.8}]')
    v1 = _prediction_parity_report(source_path, candidate_path, (1,), schema_version=1)
    v2 = _prediction_parity_report(source_path, candidate_path, (1,), schema_version=2)
    assert (v1["lost"], v1["added"], v1["mean_iou"], v1["min_iou"]) == (0, 0, 0.0, 0.0)
    assert (v2["lost"], v2["added"], v2["mean_iou"], v2["min_iou"]) == (1, 1, 0.0, 0.0)


def test_certified_capsule_evidence_tamper_and_selection(tmp_path):
    source, artifact, value, result, _ = report(tmp_path)
    root = create_certified_capsule(
        tmp_path / "ok.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    index = verify_capsule(root, expected_capsule_id=capsule_id(root))
    assert select_runtime(root, RuntimeRequest(backend="target", expected_capsule_id=index.capsule_id)).selected
    assert CliRunner().invoke(app, ["certify", str(root), "--expected-capsule-id", index.capsule_id]).exit_code == 0
    evidence = next((root / "evidence").iterdir())
    evidence.write_bytes(b"tamper")
    with pytest.raises(ValueError):
        verify_capsule(root)


def test_certified_capsule_is_relocatable_and_rejects_external_source_evidence(tmp_path):
    """Certified evidence cannot retain a producer-local source-prediction path."""
    source, artifact, value, result, bundle = report(tmp_path)
    root = create_certified_capsule(
        tmp_path / "published.vcap",
        VisionManifest(source=str(artifact)),
        value,
        result,
        source=source,
        artifact=artifact,
    )
    report_payload = json.loads((root / "certification-report.json").read_text())
    source_evidence = report_payload["evidence"]["source_predictions"]
    accuracy_payload = json.loads((root / report_payload["evidence"]["accuracy_report"]["path"]).read_text())
    assert source_evidence["path"].startswith("evidence/")
    assert accuracy_payload["source_predictions_path"] == source_evidence["path"]
    assert accuracy_payload["source_predictions_sha256"] == source_evidence["sha256"]

    relocated = tmp_path / "relocated.vcap"
    shutil.copytree(root, relocated)
    # No source-side material remains.  The packaged report must still verify.
    source.unlink()
    artifact.unlink()
    for evidence in (
        bundle.golden_manifest,
        *bundle.case_inputs,
        bundle.dataset_manifest,
        bundle.policy,
        bundle.predictions,
        bundle.source_predictions,
        bundle.accuracy_report,
        bundle.runtime_record,
        bundle.environment_record,
    ):
        Path(evidence.path).unlink(missing_ok=True)
    relocated_id = capsule_id(relocated)
    verify_capsule(relocated, expected_capsule_id=relocated_id)

    source_logical = source_evidence["path"]
    missing = tmp_path / "missing-source-evidence.vcap"
    shutil.copytree(relocated, missing)
    (missing / source_logical).unlink()
    with pytest.raises(ValueError):
        verify_capsule(missing, expected_capsule_id=relocated_id)
    tampered = tmp_path / "tampered-source-evidence.vcap"
    shutil.copytree(relocated, tampered)
    (tampered / source_logical).write_bytes(b"tamper")
    with pytest.raises(ValueError):
        verify_capsule(tampered, expected_capsule_id=relocated_id)
    symlinked = tmp_path / "symlinked-source-evidence.vcap"
    shutil.copytree(relocated, symlinked)
    (symlinked / source_logical).unlink()
    outside = tmp_path / "outside-source-predictions.json"
    outside.write_bytes(b"[]")
    (symlinked / source_logical).symlink_to(outside)
    with pytest.raises(ValueError):
        verify_capsule(symlinked, expected_capsule_id=relocated_id)
    extra = tmp_path / "extra-source-evidence.vcap"
    shutil.copytree(relocated, extra)
    (extra / "evidence" / "unreferenced-source-predictions.bin").write_bytes(b"[]")
    with pytest.raises(ValueError):
        verify_capsule(extra, expected_capsule_id=relocated_id)

    # Simulate a malicious re-indexing attempt that makes source predictions
    # point outside the capsule.  Digest/index recomputation must not make it
    # valid because packaged evidence paths are structurally constrained.
    from vision.capsule import CapsuleIndex, _digest, _inventory, _write_json

    forged_payload = json.loads((relocated / "certification-report.json").read_text())
    forged_payload["evidence"]["source_predictions"]["path"] = str(relocated / "evidence" / "source_predictions.bin")
    forged_payload.pop("digest")
    forged = CertificationReport.trusted_create(**forged_payload)
    _write_json(relocated / "certification-report.json", forged.model_dump(mode="json"))
    index = CapsuleIndex.model_validate_json((relocated / "capsule.json").read_text())
    body = index.model_dump(exclude={"capsule_id"}, mode="json") | {
        "files": _inventory(relocated),
        "certification_digest": forged.digest,
    }
    _write_json(relocated / "checksums.json", body["files"])
    _write_json(relocated / "capsule.json", CapsuleIndex(capsule_id=_digest(body), **body).model_dump(mode="json"))
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(relocated, expected_capsule_id=relocated_id)


def test_certified_publish_rejects_mutated_evidence_and_swapped_material(tmp_path):
    source, artifact, value, result, bundle = report(tmp_path)
    Path(bundle.policy.path).write_text("{}")
    output = tmp_path / "mutated.vcap"
    with pytest.raises(ValueError):
        create_certified_capsule(
            output, VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
        )
    assert not output.exists()
    # A source/artifact argument swap cannot be made valid by their matching
    # individual handles: the report binds each role separately.
    clean = tmp_path / "clean"
    clean.mkdir()
    clean_source, clean_artifact, clean_contract, clean_report, _ = report(clean)
    # The runtime manifest has to name the artifact.  Naming the independently
    # retained checkpoint is semantically wrong even when both report roles are
    # otherwise valid.
    with pytest.raises(ValueError, match="runnable artifact"):
        create_certified_capsule(
            tmp_path / "checkpoint-as-runtime.vcap",
            VisionManifest(source=str(clean_source)),
            clean_contract,
            clean_report,
            source=clean_source,
            artifact=clean_artifact,
        )
    with pytest.raises(ValueError):
        create_certified_capsule(
            tmp_path / "swapped.vcap",
            VisionManifest(source=str(clean_artifact)),
            clean_contract,
            clean_report,
            source=clean_artifact,
            artifact=clean_source,
        )


def test_certified_capsule_binds_checkpoint_and_artifact_roles_and_runtime_manifest(tmp_path):
    source, artifact, value, result, _ = report(tmp_path)
    root = create_certified_capsule(
        tmp_path / "role-bound.vcap",
        VisionManifest(source=str(artifact)),
        value,
        result,
        source=source,
        artifact=artifact,
    )
    packaged = VisionManifest.from_yaml(root / "manifest.yaml")
    assert packaged.source.startswith("artifacts/")
    assert (root / packaged.source).read_bytes() == artifact.read_bytes()
    assert (root / packaged.source).read_bytes() != source.read_bytes()

    # A checkpoint mutation is independently detected against source_digest.
    source.write_bytes(b"tampered-source")
    with pytest.raises(ValueError, match="source checkpoint"):
        create_certified_capsule(
            tmp_path / "tampered-source.vcap",
            VisionManifest(source=str(artifact)),
            value,
            result,
            source=source,
            artifact=artifact,
        )

    clean = tmp_path / "artifact-clean"
    clean.mkdir()
    clean_source, clean_artifact, clean_contract, clean_report, _ = report(clean)
    clean_artifact.write_bytes(b"tampered-artifact")
    with pytest.raises(ValueError, match="runnable artifact"):
        create_certified_capsule(
            tmp_path / "tampered-artifact.vcap",
            VisionManifest(source=str(clean_artifact)),
            clean_contract,
            clean_report,
            source=clean_source,
            artifact=clean_artifact,
        )


def test_result_contract_invariants_and_cli_id_pin(tmp_path):
    source, artifact, value, result, _ = report(tmp_path)
    root = create_certified_capsule(
        tmp_path / "ok.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    index = verify_capsule(root, expected_capsule_id=capsule_id(root))
    runner = CliRunner()
    assert runner.invoke(app, ["certify", str(root)]).exit_code != 0
    assert runner.invoke(app, ["certify", str(root), "--expected-capsule-id", "0" * 64]).exit_code != 0
    assert runner.invoke(app, ["certify", str(root), "--expected-capsule-id", index.capsule_id]).exit_code == 0
    from vision.conformance import _valid_result

    assert not _valid_result(DetectionResult([Box(-1, 0, 1, 1)], [1.0], [0], image_size=(1, 1)), value)
    assert not _valid_result(DetectionResult([Box(0, 0, 2, 1)], [1.0], [0], image_size=(1, 1)), value)
    threshold = value.model_copy(update={"result": value.result.model_copy(update={"score_threshold": 0.5})})
    assert not _valid_result(DetectionResult([Box(0, 0, 1, 1)], [0.4], [0], image_size=(1, 1)), threshold)
    duplicate = DetectionResult([Box(0, 0, 1, 1), Box(0, 0, 1, 1)], [1.0, 0.9], [0, 0], image_size=(1, 1))
    assert not _valid_result(duplicate, value)


def test_handles_reject_arbitrary_runners_and_manual_predictions_do_not_certify(tmp_path):
    source, artifact, value, result, bundle = report(tmp_path)
    with pytest.raises(TypeError):
        PreparedArtifact(artifact, sha(artifact), "target", "1", value.digest, runner=lambda _: None)

    class DifferentArtifact(Target):
        def run(self, artifact, sample):
            return DetectionResult([], [], [], image_size=(1, 1))

    failed = conform(
        Source(),
        DifferentArtifact(),
        Source().prepare(source, value),
        Target().prepare(artifact, value),
        [b"case"],
        value,
        golden=result.golden,
        policy_digest=bundle.policy.sha256,
        source_digest=sha(source),
        artifact_digest=sha(artifact),
        runtime=result.runtime,
        accuracy=result.accuracy,
        evidence=bundle,
    )
    assert failed.status == "FAIL"


def test_packaged_manifest_is_contract_generated_not_default_640(tmp_path):
    source, artifact, value, result, _ = report(tmp_path)
    root = create_certified_capsule(
        tmp_path / "contract.vcap",
        VisionManifest(source=str(artifact)),
        value,
        result,
        source=source,
        artifact=artifact,
    )
    packaged = VisionManifest.from_yaml(root / "manifest.yaml")
    assert packaged.input.shape == [1, 3, 32, 32]


def test_manifest_effective_limit_is_bounded_by_max_detections(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"source")
    base = contract(sha(source))
    value = base.model_copy(update={"result": base.result.model_copy(update={"top_k": 5, "max_detections": 2})})
    manifest = manifest_from_contract(str(source), value)
    assert manifest.decoder.top_k == 2
    with pytest.raises(ValueError, match="manifest NMS"):
        from vision.manifest import validate_contract_equivalence

        validate_contract_equivalence(
            manifest.model_copy(update={"decoder": manifest.decoder.model_copy(update={"top_k": 5})}), value
        )


def test_certified_reindexed_extra_evidence_and_runtime_axis_are_rejected(tmp_path):
    source, artifact, value, result, _ = report(tmp_path)
    root = create_certified_capsule(
        tmp_path / "ok.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    root_id = capsule_id(root)
    from vision.capsule import CapsuleIndex, _digest, _inventory, _write_json

    # A checksum/index recomputation does not authorize unreferenced evidence.
    (root / "evidence" / "extra.bin").write_bytes(b"extra")
    index = CapsuleIndex.model_validate_json((root / "capsule.json").read_text())
    body = index.model_dump(exclude={"capsule_id"}, mode="json") | {"files": _inventory(root)}
    forged = CapsuleIndex(capsule_id=_digest(body), **body)
    _write_json(root / "checksums.json", body["files"])
    _write_json(root / "capsule.json", forged.model_dump(mode="json"))
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(root, expected_capsule_id=root_id)

    axis_root = create_certified_capsule(
        tmp_path / "axis.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    axis_id = capsule_id(axis_root)
    axis_index = CapsuleIndex.model_validate_json((axis_root / "capsule.json").read_text())
    changed_artifact = axis_index.artifacts[0].model_copy(update={"device": "forged"})
    axis_body = axis_index.model_dump(exclude={"capsule_id"}, mode="json") | {
        "artifacts": (changed_artifact.model_dump(mode="json"),)
    }
    forged_axis = CapsuleIndex(capsule_id=_digest(axis_body), **axis_body)
    _write_json(axis_root / "capsule.json", forged_axis.model_dump(mode="json"))
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(axis_root, expected_capsule_id=axis_id)

    sidecar_root = create_certified_capsule(
        tmp_path / "sidecar.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    sidecar_id = capsule_id(sidecar_root)
    # Recomputing both inventory and capsule index cannot make an independently
    # edited sidecar authoritative over the embedded canonical contract.
    _write_json(sidecar_root / "preprocessing.json", {"contract_digest": value.digest, "transforms": []})
    sidecar_index = CapsuleIndex.model_validate_json((sidecar_root / "capsule.json").read_text())
    sidecar_body = sidecar_index.model_dump(exclude={"capsule_id"}, mode="json") | {"files": _inventory(sidecar_root)}
    _write_json(sidecar_root / "checksums.json", sidecar_body["files"])
    _write_json(
        sidecar_root / "capsule.json",
        CapsuleIndex(capsule_id=_digest(sidecar_body), **sidecar_body).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(sidecar_root, expected_capsule_id=sidecar_id)

    decoder_root = create_certified_capsule(
        tmp_path / "decoder.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    decoder_id = capsule_id(decoder_root)
    # An attacker that updates every index binding still cannot switch the
    # runtime decoder away from the contract's separate-output semantics.
    original_manifest = VisionManifest.from_yaml(decoder_root / "manifest.yaml")
    forged_manifest = original_manifest.model_copy(
        update={"decoder": original_manifest.decoder.model_copy(update={"kind": "yolo_end2end"})}
    )
    forged_manifest.to_yaml(decoder_root / "manifest.yaml")
    decoder_index = CapsuleIndex.model_validate_json((decoder_root / "capsule.json").read_text())
    decoder_body = decoder_index.model_dump(exclude={"capsule_id"}, mode="json") | {
        "files": _inventory(decoder_root),
        "manifest_digest": _digest(forged_manifest.model_dump(mode="json")),
    }
    _write_json(decoder_root / "checksums.json", decoder_body["files"])
    _write_json(
        decoder_root / "capsule.json",
        CapsuleIndex(capsule_id=_digest(decoder_body), **decoder_body).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(decoder_root, expected_capsule_id=decoder_id)

    limit_root = create_certified_capsule(
        tmp_path / "limit.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    limit_id = capsule_id(limit_root)
    original_manifest = VisionManifest.from_yaml(limit_root / "manifest.yaml")
    forged_manifest = original_manifest.model_copy(
        update={"decoder": original_manifest.decoder.model_copy(update={"top_k": 11})}
    )
    forged_manifest.to_yaml(limit_root / "manifest.yaml")
    limit_index = CapsuleIndex.model_validate_json((limit_root / "capsule.json").read_text())
    limit_body = limit_index.model_dump(exclude={"capsule_id"}, mode="json") | {
        "files": _inventory(limit_root),
        "manifest_digest": _digest(forged_manifest.model_dump(mode="json")),
    }
    _write_json(limit_root / "checksums.json", limit_body["files"])
    _write_json(
        limit_root / "capsule.json",
        CapsuleIndex(capsule_id=_digest(limit_body), **limit_body).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(limit_root, expected_capsule_id=limit_id)

    roles_root = create_certified_capsule(
        tmp_path / "roles.vcap", VisionManifest(source=str(artifact)), value, result, source=source, artifact=artifact
    )
    roles_id = capsule_id(roles_root)
    # Clearing every name and reindexing every mutable integrity field cannot
    # turn a three-output contract into a decoder-defaulted runtime.
    original_manifest = VisionManifest.from_yaml(roles_root / "manifest.yaml")
    forged_manifest = original_manifest.model_copy(
        update={
            "output": original_manifest.output.model_copy(
                update={"boxes_name": None, "scores_name": None, "labels_name": None}
            )
        }
    )
    forged_manifest.to_yaml(roles_root / "manifest.yaml")
    roles_index = CapsuleIndex.model_validate_json((roles_root / "capsule.json").read_text())
    roles_body = roles_index.model_dump(exclude={"capsule_id"}, mode="json") | {
        "files": _inventory(roles_root),
        "manifest_digest": _digest(forged_manifest.model_dump(mode="json")),
    }
    _write_json(roles_root / "checksums.json", roles_body["files"])
    _write_json(
        roles_root / "capsule.json",
        CapsuleIndex(capsule_id=_digest(roles_body), **roles_body).model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(roles_root, expected_capsule_id=roles_id)


def test_cli_contract_plugins(tmp_path):
    source = tmp_path / "s"
    source.write_bytes(b"x")
    p = tmp_path / "c.json"
    contract(sha(source)).to_json(p)
    runner = CliRunner()
    assert runner.invoke(app, ["contract", "validate", str(p)]).exit_code == 0
    assert runner.invoke(app, ["plugins", "list"]).exit_code == 0


def test_onnx_runtime_real_constant_smoke_and_invalid(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import numpy as np
    from onnx import TensorProto, helper

    path = tmp_path / "tiny.onnx"
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])]
    outputs = [
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4]),
        helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1]),
        helper.make_tensor_value_info("c", TensorProto.INT64, [1, 1]),
    ]
    nodes = [
        helper.make_node(
            "Constant", [], ["b"], value=helper.make_tensor("b", TensorProto.FLOAT, [1, 1, 4], [0, 0, 1, 1])
        ),
        helper.make_node("Constant", [], ["s"], value=helper.make_tensor("s", TensorProto.FLOAT, [1, 1], [1])),
        helper.make_node("Constant", [], ["c"], value=helper.make_tensor("c", TensorProto.INT64, [1, 1], [0])),
    ]
    onnx.save(
        helper.make_model(helper.make_graph(nodes, "g", inputs, outputs), opset_imports=[helper.make_opsetid("", 13)]),
        path,
    )
    plugin = OnnxRuntimeTargetPlugin()
    prepared = plugin.prepare(path, contract(sha(path)))
    from vision.preprocess import PreparedInput

    smoke = PreparedInput(np.zeros((1, 3, 32, 32), dtype=np.float32), (32, 32), "0" * 64)
    assert plugin.run(prepared, smoke).classes == [0]
    bad = contract(sha(path)).model_copy(
        update={
            "raw_output": contract(sha(path)).raw_output.model_copy(
                update={"roles": {"boxes": "b", "scores": "s", "classes": "missing"}}
            )
        }
    )
    with pytest.raises(ValueError):
        plugin.prepare(path, bad)


@pytest.mark.parametrize("interpolation", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("with_transpose", [False, True])
def test_onnx_image_file_contract_preprocessing_and_generated_predictions(tmp_path, interpolation, with_transpose):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("pycocotools")
    import numpy as np
    from onnx import TensorProto, helper
    from PIL import Image

    model = tmp_path / "tiny.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["b"], value=helper.make_tensor("b", TensorProto.FLOAT, [1, 1, 4], [0, 0, 1, 1])
            ),
            helper.make_node("Constant", [], ["c"], value=helper.make_tensor("c", TensorProto.INT64, [1, 1], [0])),
            helper.make_node("ReduceMean", ["x"], ["mean"], axes=[1, 2, 3], keepdims=0),
            helper.make_node(
                "Constant", [], ["score_shape"], value=helper.make_tensor("score_shape", TensorProto.INT64, [2], [1, 1])
            ),
            helper.make_node("Reshape", ["mean", "score_shape"], ["s"]),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("c", TensorProto.INT64, [1, 1]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), model)
    image = tmp_path / "input.png"
    Image.new("RGB", (5, 3), (20, 30, 40)).save(image)
    bright = tmp_path / "bright.png"
    Image.new("RGB", (5, 3), (255, 255, 255)).save(bright)
    transforms = [
        {"kind": "color_convert", "source": "RGB", "target": "RGB"},
        {
            "kind": "letterbox",
            "height": 32,
            "width": 32,
            "pad_value": 114,
            "interpolation": interpolation,
            "placement": "center",
            "rounding": "round_half_up",
            "scale": "min_ratio",
        },
        {"kind": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
    ]
    if with_transpose:
        transforms.append({"kind": "transpose", "order": [0, 3, 1, 2]})
    value = contract(
        sha(model),
        preprocessing=transforms,
    )
    plugin = OnnxRuntimeTargetPlugin()
    prepared = plugin.prepare(model, value)
    tensor = plugin.prepare_input(prepared, image, value)
    from vision.adapters.processor import DetectionProcessor

    packaged_tensor, _ = DetectionProcessor(manifest_from_contract(str(model), value))(Image.open(image))
    assert tensor.tensor.shape == (1, 3, 32, 32)
    assert np.array_equal(tensor.tensor, packaged_tensor)
    # Pixel-dependent graph plus exact contract preprocessing: the /255
    # normalization and center letterbox affect observable runtime output.
    dark = plugin.run(prepared, tensor)
    from vision.preprocess import PreparedInput

    packaged_result = plugin.run(prepared, PreparedInput(packaged_tensor, tensor.image_size, tensor.input_sha256))
    assert packaged_result.scores == dark.scores
    bright_tensor = plugin.prepare_input(prepared, bright, value)
    bright_result = plugin.run(prepared, bright_tensor)
    assert tensor.tensor.shape == bright_tensor.tensor.shape == (1, 3, 32, 32)
    assert dark.metadata["input_sha256"] == sha(image)
    assert dark.image_size == (5, 3)
    assert bright_result.scores[0] > dark.scores[0]


def test_input_dependent_onnx_conform_pass_with_generated_coco_predictions(tmp_path):
    """Certification executes real pixels, then binds generated COCO bytes."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("pycocotools")
    from onnx import TensorProto, helper
    from PIL import Image

    from vision.conformance import _canonical_predictions
    from vision.preprocess import preprocess

    model = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["b"], value=helper.make_tensor("b", TensorProto.FLOAT, [1, 1, 4], [0, 0, 1, 1])
            ),
            helper.make_node("Constant", [], ["c"], value=helper.make_tensor("c", TensorProto.INT64, [1, 1], [0])),
            helper.make_node("ReduceMean", ["x"], ["mean"], axes=[1, 2, 3], keepdims=0),
            helper.make_node(
                "Constant", [], ["shape"], value=helper.make_tensor("shape", TensorProto.INT64, [2], [1, 1])
            ),
            helper.make_node("Reshape", ["mean", "shape"], ["s"]),
        ],
        "input-dependent",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 4]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("c", TensorProto.INT64, [1, 1]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), model)
    image = tmp_path / "case.png"
    Image.new("RGB", (5, 3), (255, 128, 0)).save(image)
    payload = image.read_bytes()
    value = contract(
        sha(model),
        preprocessing=[
            {
                "kind": "letterbox",
                "height": 32,
                "width": 32,
                "pad_value": 114,
                "interpolation": "bilinear",
                "placement": "center",
                "rounding": "round_half_up",
                "scale": "min_ratio",
            },
            {"kind": "normalize", "mean": [0, 0, 0], "std": [255, 255, 255]},
        ],
    )
    target = OnnxRuntimeTargetPlugin()
    artifact = target.prepare(model, value)

    class OnnxSource(SourceAdapter):
        plugin_id, plugin_version = "onnx-source", "1"
        capabilities = target.capabilities

        def capture(self, source, contract):
            return Path(source)

        def prepare_input(self, captured, sample, contract):
            return preprocess(contract, sample)

        def run(self, captured, sample):
            descriptor = dict(captured.descriptors)["contract_json"]
            return target.run(
                target.prepare(captured.material_path, VisionContract.model_validate_json(descriptor)), sample
            )

        def prepare(self, source, contract):
            return PreparedSource(
                Path(source),
                sha(Path(source)),
                self.plugin_id,
                self.plugin_version,
                contract.digest,
                (("contract_json", contract.canonical_json()),),
            )

    source = OnnxSource()
    captured = source.prepare(model, value)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    golden, runtime, accuracy, bundle = files(
        evidence_root,
        sha(model),
        sha(model),
        image_bytes=payload,
        case_bytes=payload,
        source_plugin=source.plugin_id,
        target_plugin=target.plugin_id,
        target_version=target.plugin_version,
    )
    generated = _canonical_predictions(target, artifact, value, Path(bundle.dataset_manifest.path))
    prediction_path = Path(bundle.predictions.path)
    prediction_path.write_bytes(generated)
    source_prediction_path = Path(bundle.source_predictions.path)
    source_prediction_path.write_bytes(generated)
    metric_payload = json.loads(Path(bundle.accuracy_report.path).read_text())
    metric_payload["source_predictions_sha256"] = sha(source_prediction_path)
    from vision.conformance import _prediction_parity_report

    metric_payload["parity"] = _prediction_parity_report(source_prediction_path, prediction_path, (1,))
    Path(bundle.accuracy_report.path).write_text(json.dumps(metric_payload, sort_keys=True, separators=(",", ":")))
    prediction = EvidenceFile(path=str(prediction_path), sha256=sha(prediction_path))
    report_file = Path(bundle.accuracy_report.path)
    bundle = bundle.model_copy(
        update={
            "predictions": prediction,
            "source_predictions": EvidenceFile(path=str(source_prediction_path), sha256=sha(source_prediction_path)),
            "accuracy_report": EvidenceFile(path=str(report_file), sha256=sha(report_file)),
        }
    )
    accuracy = accuracy.model_copy(update={"prediction_digest": prediction.sha256, "report_digest": sha(report_file)})
    runtime_record = Path(bundle.runtime_record.path)
    runtime_payload = json.loads(runtime_record.read_text())
    runtime_payload.update(
        {
            "backend": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "device": "cpu",
            "runtime_version": target.plugin_version,
        }
    )
    runtime_record.write_text(json.dumps(runtime_payload, sort_keys=True, separators=(",", ":")))
    bundle = bundle.model_copy(
        update={"runtime_record": EvidenceFile(path=str(runtime_record), sha256=sha(runtime_record))}
    )
    runtime = runtime.model_copy(
        update={
            "backend": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "device": "cpu",
            "runtime_version": target.plugin_version,
        }
    )
    result = conform(
        source,
        target,
        captured,
        artifact,
        [payload],
        value,
        golden=golden,
        policy_digest=bundle.policy.sha256,
        source_digest=sha(model),
        artifact_digest=sha(model),
        runtime=runtime,
        accuracy=accuracy,
        evidence=bundle,
    )
    assert result.status == "PASS", result.details
    capsule = create_certified_capsule(
        tmp_path / "runtime.vcap", VisionManifest(source=str(model)), value, result, source=model, artifact=model
    )
    index = verify_capsule(capsule, expected_capsule_id=capsule_id(capsule))
    from vision import load

    loaded = load(
        str(capsule), backend="onnxruntime", provider="CPUExecutionProvider", expected_capsule_id=index.capsule_id
    )
    certified = loaded(image)
    expected = target.run(artifact, target.prepare_input(artifact, image, value))
    assert certified.boxes == expected.boxes
    assert certified.scores == expected.scores
    assert certified.classes == expected.classes
    assert certified.image_size == expected.image_size

    generated = VisionManifest.from_yaml(capsule / "manifest.yaml")
    wrong_input = generated.model_copy(update={"input": generated.input.model_copy(update={"input_name": "wrong"})})
    with pytest.raises(ValueError, match="input"):
        from vision.adapters.backend import BackendAdapter

        BackendAdapter(wrong_input.model_copy(update={"source": str(model)}), "onnxruntime").load(str(model), "cpu")
    wrong_role = generated.model_copy(update={"output": generated.output.model_copy(update={"labels_name": "wrong"})})
    with pytest.raises(ValueError, match="outputs|roles"):
        from vision.adapters.backend import BackendAdapter

        BackendAdapter(wrong_role.model_copy(update={"source": str(model)}), "onnxruntime").load(str(model), "cpu")


def test_mock_tensorrt_conformance_capsule_receipt_relocates_and_selects(tmp_path, monkeypatch):
    """A real target instance emits every TensorRT attestation under a mocked builder."""
    source = tmp_path / "source.pt"
    artifact = tmp_path / "model.engine"
    source.write_bytes(b"source")
    value = contract(sha(source))
    value = value.model_copy(
        update={
            "raw_output": value.raw_output.model_copy(
                update={
                    "tensors": tuple(
                        item.model_copy(update={"shape": (1, 2, *item.shape[2:])}) for item in value.raw_output.tensors
                    )
                }
            )
        }
    )

    identity = {
        "cuda_device": "Mock GPU",
        "cuda_compute_capability": "8.7",
        "cuda_runtime": "12.4",
        "tensorrt_python_version": "10.0",
        "driver_or_jetpack": "JP6",
    }
    onnx = tmp_path / "materialized.onnx"
    onnx.write_bytes(b"mock-onnx")
    lineage_path = tmp_path / "materialized.onnx.lineage.json"
    lineage_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "source_sha256": sha(source),
                "onnx_sha256": sha(onnx),
                "source_plugin": "source",
                "source_plugin_version": "1",
                "contract_digest": value.digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    source_plugin = Source()
    captured = source_plugin.prepare(source, value)
    raw_materialized = MaterializedOnnx(
        onnx, sha(onnx), sha(source), "source", "1", value.digest, lineage_path, sha(lineage_path), ""
    )
    materialized = source_plugin._issue_materialization(raw_materialized, captured, value)
    monkeypatch.setattr("vision.plugins._validate_onnx_contract", lambda *args: None)
    monkeypatch.setattr(
        "vision.backends.TensorRTBackend.resolve_trtexec", classmethod(lambda cls, path=None: str(Path(__file__)))
    )
    monkeypatch.setattr(
        TensorRTTargetPlugin,
        "_build_host_metadata",
        staticmethod(
            lambda executable, version: {
                "os": "mock",
                "architecture": "arm64",
                "builder_target": "mock-arm64",
                "trtexec_sha256": sha(Path(executable)),
                "trtexec_version": version,
                **identity,
            }
        ),
    )

    def build(cls, source_path, output, request):
        output.write_bytes(b"engine")
        return (
            output,
            [request.trtexec_path, "--onnx=<bound>", "--saveEngine=<bound>", "--noTF32", "--skipInference"],
            "mock build",
            {"trtexec": "10.0"},
        )

    monkeypatch.setattr("vision.backends.TensorRTBackend.build_atomic", classmethod(build))
    target = TensorRTTargetPlugin(engine_output=artifact)
    prepared = target.prepare_materialized(materialized, captured, source_plugin, value)
    golden, base_runtime, accuracy, bundle = files(
        tmp_path, sha(source), sha(artifact), target_plugin="tensorrt", target_version="1"
    )
    runtime = base_runtime.model_copy(
        update={
            "hardware": identity,
            "environment": {"os": "mock"},
            "backend": "tensorrt",
            "provider": "TensorRTExecutionProvider",
            "device": "cuda",
            "runtime_version": "10.0",
        }
    )
    record_data = json.loads(Path(bundle.runtime_record.path).read_text())
    record_data.update(
        {
            "target_plugin": "tensorrt",
            "hardware": identity,
            "environment": {"os": "mock"},
            "backend": "tensorrt",
            "provider": "TensorRTExecutionProvider",
            "device": "cuda",
            "runtime_version": "10.0",
        }
    )
    Path(bundle.runtime_record.path).write_text(json.dumps(record_data, sort_keys=True, separators=(",", ":")))
    record = EvidenceFile(path=bundle.runtime_record.path, sha256=sha(Path(bundle.runtime_record.path)))
    bundle = bundle.model_copy(update={"runtime_record": record})
    monkeypatch.setattr(target, "prepare_input", lambda artifact, sample, contract: sample)
    monkeypatch.setattr(
        target,
        "run",
        lambda artifact, sample: DetectionResult([Box(0, 0, 1, 1)], [1.0], [0], image_size=(1, 1)),
    )
    result = conform(
        source_plugin,
        target,
        captured,
        prepared,
        [b"case"],
        value,
        golden=golden,
        policy_digest=bundle.policy.sha256,
        source_digest=sha(source),
        artifact_digest=sha(artifact),
        runtime=runtime,
        accuracy=accuracy,
        evidence=bundle,
        report_schema_version=2,
    )
    assert result.status == "PASS", result.details
    root = create_certified_capsule(
        tmp_path / "trt.vcap",
        VisionManifest(source=str(artifact)),
        value,
        result,
        source=source,
        artifact=artifact,
        build_artifact=prepared,
        target_plugin=target,
        materialized_onnx=materialized,
    )
    expected_id = capsule_id(root)
    moved = tmp_path / "moved.vcap"
    shutil.copytree(root, moved)
    source.unlink()
    artifact.unlink()
    index = verify_capsule(moved, expected_capsule_id=expected_id)
    assert index.target_plugin_id == "tensorrt" and (moved / "evidence/tensorrt-build-receipt.json").is_file()
    packaged_source = moved / next(name for name in index.files if name.startswith("original_weights/"))
    selected = select_runtime(moved, RuntimeRequest(backend="tensorrt", expected_capsule_id=expected_id))
    packaged_engine = moved / selected.selected.logical_path
    assert packaged_engine.read_bytes() == b"engine"
    assert target.run(prepared, b"case").classes == [0]
    packaged_onnx = moved / "evidence/tensorrt-materialized.onnx"
    packaged_onnx.write_bytes(b"forged-onnx")
    # A coherent local rewrite of ONNX, lineage, receipt, checksums, and index
    # still has no external trust anchor and cannot inherit the original pin.
    lineage = moved / "evidence/tensorrt-materialized.onnx.lineage.json"
    lineage_payload = json.loads(lineage.read_text())
    lineage_payload["onnx_sha256"] = sha(packaged_onnx)
    lineage.write_text(json.dumps(lineage_payload, sort_keys=True, separators=(",", ":")))
    receipt_path = moved / "evidence/tensorrt-build-receipt.json"
    receipt_payload = json.loads(receipt_path.read_text())
    receipt_payload["onnx_sha256"] = sha(packaged_onnx)
    receipt_payload["onnx_lineage"] = lineage_payload
    receipt_payload["onnx_lineage_sha256"] = sha(lineage)
    receipt_path.write_text(json.dumps(receipt_payload, sort_keys=True, separators=(",", ":")))
    from vision.capsule import CapsuleIndex, _digest, _inventory, _write_json

    index_payload = json.loads((moved / "capsule.json").read_text())
    original = CapsuleIndex.model_validate(index_payload)
    body = original.model_dump(exclude={"capsule_id"}, mode="json") | {"files": _inventory(moved)}
    repaired = CapsuleIndex(capsule_id=_digest(body), **body)
    _write_json(moved / "checksums.json", body["files"])
    _write_json(moved / "capsule.json", repaired.model_dump(mode="json"))
    with pytest.raises(ValueError, match="requires expected_capsule_id"):
        verify_capsule(moved)
    with pytest.raises(ValueError, match="unexpected capsule id"):
        verify_capsule(moved, expected_capsule_id=expected_id)
    forged = PreparedArtifact(packaged_engine, sha(packaged_engine), "tensorrt", "1", value.digest)
    with pytest.raises(ValueError, match="target-attested|invalid"):
        create_certified_capsule(
            tmp_path / "bad.vcap",
            VisionManifest(source=str(packaged_engine)),
            value,
            result,
            source=packaged_source,
            artifact=packaged_engine,
            build_artifact=forged,
            target_plugin=target,
        )
