import errno
import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

import vision
import vision.validation as validation_module
from vision.adapters.rfdetr import RfDetrAdapter
from vision.adapters.rtdetr import RTDetrAdapter
from vision.adapters.ultralytics import UltralyticsAdapter
from vision.cli import app
from vision.experiment_evidence import ExperimentReport
from vision.manifest import VisionManifest
from vision.types import Box, DetectionResult
from vision.validation import (
    DatasetManifest,
    ModelTarget,
    ValidationPolicy,
    ValidationReport,
    _digest,
    _validate_materialized,
    compare_detections,
    recover_orphan_validations,
    runtime_policy_contract,
    validate_dataset,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


RUNTIME_IDENTITY = {
    "backend": "pytorch",
    "provider": "torch",
    "device": "cpu",
    "precision": "fp32",
    "input_shape": [1, 3, 640, 640],
}


def dataset(tmp_path, count=1):
    images, entries = [], []
    for ident in range(1, count + 1):
        path = tmp_path / f"{ident}.jpg"
        Image.new("RGB", (10, 10)).save(path)
        images.append({"id": ident, "file_name": path.name, "width": 10, "height": 10})
        entries.append({"id": ident, "path": str(path), "sha256": sha(path), "width": 10, "height": 10})
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}]
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"images": images, "categories": [{"id": 1, "name": "x"}], "annotations": annotations}))
    m = VisionManifest(source="source.onnx")
    data = {
        "split": "tiny",
        "annotations_path": str(ann),
        "annotations_sha256": sha(ann),
        "images": entries,
        "categories": [1],
        "preprocessing_digest": digest(
            {"input": m.input.model_dump(mode="json"), "preprocessing": m.preprocessing.model_dump(mode="json")}
        ),
        "postprocessing_digest": digest(
            {
                "output": m.output.model_dump(mode="json"),
                "decoder": m.decoder.model_dump(mode="json"),
                "postprocessing": m.postprocessing,
                "class_mapping": {"0": 1},
            }
        ),
        "expected_image_count": count,
        "expected_annotation_count": 1,
        "format": "coco",
        "cocoeval_iou_type": "bbox",
        "cocoeval_version": "pycocotools",
    }
    return DatasetManifest(**(data | {"digest": digest(data)}))


def target(tmp_path, name):
    source = tmp_path / f"{name}.pt"
    source.write_bytes(name.encode())
    manifest = tmp_path / f"{name}.yaml"
    value = VisionManifest(source=str(source), family="torch")
    value.to_yaml(manifest)
    return ModelTarget(
        source=str(source),
        source_sha256=sha(source),
        manifest_path=str(manifest),
        manifest_sha256=sha(manifest),
        manifest_digest=digest(value.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )


def test_public_loads_twice_full_inventory_and_zero_boxes(monkeypatch, tmp_path):
    data = dataset(tmp_path, 2)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    calls = []
    loads = []

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            calls.append(path)
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: loads.append((args, kwargs)) or Model())
    report = validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "out")
    assert (
        report.passed
        and not report.publishable
        and report.evidence_kind == "observed"
        and len(loads) == 2
        and len(calls) == 4
        and report.reference.evaluated_image_ids == [1, 2]
    )
    assert report.reference.detection_image_ids == [] and (tmp_path / "out" / "COMMITTED").exists()
    experiment_path = tmp_path / "out" / "result" / "experiment_report.json"
    experiment = ExperimentReport.model_validate_json(experiment_path.read_text())
    assert experiment.status == "PASS"
    assert experiment.verify_validation_binding(report)
    assert (tmp_path / "out" / "result" / "EXPERIMENT_COMMITTED").is_file()
    assert report.verify(tmp_path / "out", report.policy_digest)


def test_public_validation_rejects_forged_canonical_experiment(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    report = validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "out")
    experiment_path = tmp_path / "out" / "result" / "experiment_report.json"
    payload = json.loads(experiment_path.read_text())
    payload["status"] = "FAIL"
    experiment_path.write_text(json.dumps(payload))
    marker = tmp_path / "out" / "result" / "EXPERIMENT_COMMITTED"
    marker.write_text(hashlib.sha256(experiment_path.read_bytes()).hexdigest())

    with pytest.raises(ValueError):
        report.verify(tmp_path / "out", report.policy_digest)


def test_canonical_experiment_failure_rolls_back_validation_publication(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    monkeypatch.setattr(
        "vision.experiment_evidence.experiment_report_from_validation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("canonical evidence failed")),
    )

    with pytest.raises(RuntimeError, match="canonical evidence failed"):
        validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "out")

    assert not (tmp_path / "out").exists()
    assert not (tmp_path / ".out.validation.lock").exists()


def test_public_runtime_session_identity_rejects_forged_or_inconsistent_metadata(monkeypatch, tmp_path):
    data = dataset(tmp_path, 2)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    trusted = {
        "backend": "pytorch",
        "provider": "torch",
        "device": "cpu",
        "precision": "fp32",
        "input_shape": [1, 3, 640, 640],
    }

    class Model:
        def __init__(self, bad=False, inconsistent=False):
            self.bad, self.inconsistent, self.calls = bad, inconsistent, 0

        def runtime_identity(self):
            return trusted

        def __call__(self, path):
            self.calls += 1
            meta = dict(trusted)
            if self.bad or (self.inconsistent and self.calls == 2):
                meta["provider"] = "forged"
            return DetectionResult(metadata=meta)

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    report = validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "valid")
    assert report.candidate.runtime.backend == "pytorch" and report.verify(tmp_path / "valid", report.policy_digest)
    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model(bad=True))
    with pytest.raises(ValueError, match="session identity"):
        validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "forged")
    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model(inconsistent=True))
    with pytest.raises(ValueError, match="session identity"):
        validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "inconsistent")


def _runtime_contract(target, **changes):
    return runtime_policy_contract(target, **(RUNTIME_IDENTITY | changes))


def test_public_observed_runtime_contract_is_publishable_only_when_policy_matches(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    policy = ValidationPolicy(
        max_map_drop=0,
        reference_runtime=_runtime_contract(left),
        candidate_runtime=_runtime_contract(right),
    )
    report = validate_dataset(left, right, data, policy, tmp_path / "publishable")
    assert report.evidence_kind == "observed" and report.publishable
    assert report.verify(tmp_path / "publishable", report.policy_digest)

    mismatched = ValidationPolicy(
        max_map_drop=0,
        reference_runtime=_runtime_contract(left, provider="ultralytics-torch"),
        candidate_runtime=_runtime_contract(right),
    )
    with pytest.raises(ValueError, match="reference runtime does not match policy"):
        validate_dataset(left, right, data, mismatched, tmp_path / "mismatched")


def test_public_int8_validation_requires_bound_calibration_build_report(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    reference = target(tmp_path, "reference")
    source = tmp_path / "candidate.onnx"
    source.write_bytes(b"candidate-int8")
    manifest_path = tmp_path / "candidate.yaml"
    manifest = VisionManifest(
        source=str(source),
        family="onnx",
        artifacts=[
            {
                "path": str(source),
                "backend": "onnxruntime",
                "precision": "int8",
                "validated": True,
                "checksum": sha(source),
                "compatible": {"device": "cpu", "provider": "CPUExecutionProvider"},
            }
        ],
    )
    manifest.to_yaml(manifest_path)
    candidate = ModelTarget(
        source=str(source),
        source_sha256=sha(source),
        manifest_path=str(manifest_path),
        manifest_sha256=sha(manifest_path),
        manifest_digest=digest(manifest.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )

    class Model:
        def __init__(self, runtime):
            self.runtime = runtime

        def runtime_identity(self):
            return dict(self.runtime)

        def __call__(self, path):
            return DetectionResult(metadata=dict(self.runtime))

    candidate_runtime = {
        "backend": "onnxruntime",
        "provider": "CPUExecutionProvider",
        "device": "cpu",
        "precision": "int8",
        "input_shape": [1, 3, 640, 640],
    }
    monkeypatch.setattr(
        vision,
        "load",
        lambda source, **kwargs: Model(
            candidate_runtime if str(source).endswith("candidate.onnx") else RUNTIME_IDENTITY
        ),
    )

    with pytest.raises(ValueError, match="bound calibration build report"):
        validate_dataset(reference, candidate, data, ValidationPolicy(max_map_drop=0), tmp_path / "out")

    assert not (tmp_path / "out").exists()


def test_public_validation_rejects_materialized_runtime_sentinel(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    sentinel = dict(RUNTIME_IDENTITY | {"backend": "materialized", "provider": "materialized"})

    class Model:
        def runtime_identity(self):
            return dict(sentinel)

        def __call__(self, path):
            return DetectionResult(metadata=dict(sentinel))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    with pytest.raises(ValueError, match="rejects materialized"):
        validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "sentinel")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "onnxruntime"),
        ("provider", "CPUExecutionProvider"),
        ("device", "cuda"),
        ("precision", "fp16"),
        ("input_shape", [1, 3, 320, 320]),
    ],
)
def test_public_runtime_session_identity_rejects_each_forged_field(monkeypatch, tmp_path, field, value):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            metadata = dict(RUNTIME_IDENTITY)
            metadata[field] = value
            return DetectionResult(metadata=metadata)

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    with pytest.raises(ValueError, match="session identity"):
        validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / field)


@pytest.mark.parametrize("mode", ["missing", "raises"])
def test_public_runtime_session_identity_is_mandatory(monkeypatch, tmp_path, mode):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")

    class Model:
        def runtime_identity(self):
            if mode == "raises":
                raise RuntimeError("session unavailable")
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    if mode == "missing":
        delattr(Model, "runtime_identity")
    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    with pytest.raises(ValueError, match="runtime session identity"):
        validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / mode)


def test_native_adapter_runtime_identity_overrides_are_family_specific():
    yolo = UltralyticsAdapter(VisionManifest(source="yolo.pt", family="yolo26"))
    yolo.device = "cuda"
    rtdetr = RTDetrAdapter(VisionManifest(source="rt.pt", family="rtdetr"))
    rtdetr.device = "cpu"
    rfdetr = RfDetrAdapter(
        VisionManifest(
            source="rf.pt",
            family="rfdetr",
            input={"shape": [1, 3, 384, 384]},
            preprocessing={
                "resize": "resize",
                "implementation": "torchvision_bilinear_antialias_false",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            output={"box_format": "cxcywh"},
            decoder={"kind": "rfdetr", "boxes_normalized": True},
        )
    )
    rfdetr.device = "cpu"
    assert yolo.runtime_identity()["provider"] == "ultralytics-torch"
    assert rtdetr.runtime_identity()["provider"] == "ultralytics-torch"
    assert rfdetr.runtime_identity() == {
        "backend": "pytorch",
        "provider": "rfdetr-torch",
        "device": "cpu",
        "precision": "fp32",
        "input_shape": [1, 3, 384, 384],
    }


def test_materialized_quality_failure_and_forged_metric(tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    rows = [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "score": 0.9}]
    report = _validate_materialized(
        rows,
        [],
        data,
        ValidationPolicy(max_map_drop=0),
        tmp_path / "out",
        reference_target=left,
        candidate_target=right,
        reference_evaluated_image_ids=[1],
        candidate_evaluated_image_ids=[1],
    )
    assert not report.passed and (tmp_path / "out" / "validation_report.json").exists()
    forged = report.model_copy(update={"map_candidate": report.map_reference, "digest": "0" * 64})
    with pytest.raises(ValueError):
        forged.verify(tmp_path / "out", report.policy_digest)


def test_hungarian_and_class_mismatch_are_deterministic():
    policy = ValidationPolicy(max_map_drop=0, iou_threshold=0.1)
    ref = DetectionResult([Box(0, 0, 10, 10), Box(6, 0, 16, 10)], [0.9, 0.8], [1, 1])
    cand = DetectionResult([Box(0, 0, 16, 10), Box(6, 0, 16, 10)], [0.9, 0.8], [1, 1])
    assert compare_detections(ref, cand, policy)["matched"] == 2
    mismatch = compare_detections(
        DetectionResult([Box(0, 0, 4, 4)], [0.9], [1]), DetectionResult([Box(0, 0, 4, 4)], [0.9], [2]), policy
    )
    assert mismatch["class_mismatches"]


def test_coco_path_dimensions_and_mapping_rejected(tmp_path):
    data = dataset(tmp_path)
    value = data.model_dump()
    value["images"][0]["width"] = 11
    value["digest"] = digest({k: v for k, v in value.items() if k != "digest"})
    with pytest.raises(ValueError, match="dimensions"):
        DatasetManifest(**value)
    broken = target(tmp_path, "broken").model_copy(update={"class_mapping": {"0": 2}})
    with pytest.raises(ValueError, match="class mapping"):
        validate_dataset(broken, target(tmp_path, "ok"), data, ValidationPolicy(max_map_drop=0), tmp_path / "x")


def materialize(tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    rows = [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "score": 0.9}]
    out = tmp_path / "out"
    report = _validate_materialized(
        rows,
        rows,
        data,
        ValidationPolicy(max_map_drop=0),
        out,
        reference_target=left,
        candidate_target=right,
        reference_evaluated_image_ids=[1],
        candidate_evaluated_image_ids=[1],
    )
    return data, report, out


def test_committed_result_uses_relative_records_and_policy_binding(tmp_path):
    _, report, out = materialize(tmp_path)
    assert not os.path.isabs(report.reference.predictions_path)
    assert (out / "result" / "reference_prediction_manifest.json").is_file()
    assert report.evidence_kind == "materialized" and not report.publishable
    with pytest.raises(ValueError, match="explicit development authorization"):
        report.verify(out, report.policy_digest)
    assert report.verify(out, report.policy_digest, allow_materialized=True)
    with pytest.raises(ValueError, match="external policy"):
        report.verify(out, "0" * 64)


def test_coordinated_materialized_to_observed_rewrite_is_ineligible(tmp_path):
    _, report, _ = materialize(tmp_path)
    payload = report.model_dump(mode="json")
    payload["evidence_kind"] = "observed"
    payload["publishable"] = False
    payload["digest"] = _digest({key: value for key, value in payload.items() if key != "digest"})
    with pytest.raises(ValueError, match="cannot contain materialized sentinels"):
        ValidationReport.model_validate(payload)


def test_private_stage_and_lock_permissions(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    observed = []

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            lock = next(tmp_path.glob(".out.validation.lock"))
            stage = next(tmp_path.glob(".out.validation-stage-*"))
            observed.append((lock.stat().st_mode & 0o777, stage.stat().st_mode & 0o777))
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), tmp_path / "out")
    assert observed and all(item == (0o600, 0o700) for item in observed)


def test_report_or_prediction_forgery_is_rejected(tmp_path):
    _, report, out = materialize(tmp_path)
    (out / "result" / "reference_predictions.json").write_text("[]")
    with pytest.raises(ValueError, match="prediction bytes"):
        report.verify(out, report.policy_digest, allow_materialized=True)


def test_relative_escape_and_symlinked_prediction_are_rejected(tmp_path):
    _, report, out = materialize(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        report.reference.model_copy(update={"predictions_path": "../outside.json"}).__class__.model_validate(
            report.reference.model_copy(update={"predictions_path": "../outside.json"}).model_dump()
        )
    (out / "result" / "reference_predictions.json").unlink()
    (out / "result" / "reference_predictions.json").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="unsafe"):
        report.verify(out, report.policy_digest, allow_materialized=True)


def test_output_collision_never_overwrites_existing(tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    out = tmp_path / "out"
    out.mkdir()
    (out / "foreign").write_text("keep")
    with pytest.raises(FileExistsError):
        _validate_materialized(
            [],
            [],
            data,
            ValidationPolicy(max_map_drop=0),
            out,
            reference_target=left,
            candidate_target=right,
            reference_evaluated_image_ids=[1],
            candidate_evaluated_image_ids=[1],
        )
    assert (out / "foreign").read_text() == "keep"


def test_reservation_happens_before_public_model_inference(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    out, seen = tmp_path / "out", []

    class Model:
        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            seen.append((tmp_path / ".out.validation.lock").is_file() and not out.exists())
            return DetectionResult(metadata=dict(RUNTIME_IDENTITY))

    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    validate_dataset(left, right, data, ValidationPolicy(max_map_drop=0), out)
    assert seen and all(seen)


def test_untrusted_symlink_parent_is_rejected(tmp_path):
    parent = tmp_path / "real"
    parent.mkdir()
    link = tmp_path / "link"
    link.symlink_to(parent, target_is_directory=True)
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    with pytest.raises(ValueError, match="symlink parent"):
        _validate_materialized(
            [],
            [],
            data,
            ValidationPolicy(max_map_drop=0),
            link / "out",
            reference_target=left,
            candidate_target=right,
            reference_evaluated_image_ids=[1],
            candidate_evaluated_image_ids=[1],
        )


def test_recovery_requires_age_dead_pid_and_valid_token(monkeypatch, tmp_path):
    def owner(token, pid):
        return {
            "token": token,
            "pid": pid,
            "created_at": 0,
            "policy_digest": "a" * 64,
            "stage_name": f".out.validation-stage-{token}",
            "lock_name": ".out.validation.lock",
            "output_name": "out",
        }

    stale_owner = owner("a" * 32, 99999)
    stale = tmp_path / stale_owner["stage_name"]
    stale.mkdir()
    stale.joinpath(".validation-owner.json").write_text(json.dumps(stale_owner))
    (tmp_path / stale_owner["lock_name"]).write_text(json.dumps(stale_owner))
    live_owner = owner("b" * 32, 123)
    live_owner["lock_name"] = ".live.validation.lock"
    live_owner["stage_name"] = f".live.validation-stage-{live_owner['token']}"
    live = tmp_path / live_owner["stage_name"]
    live.mkdir()
    live.joinpath(".validation-owner.json").write_text(json.dumps(live_owner))
    (tmp_path / live_owner["lock_name"]).write_text(json.dumps(live_owner))
    bad = tmp_path / "bad"
    bad.mkdir()
    bad.joinpath(".validation-owner.json").write_text("not-json")
    monkeypatch.setattr("vision.validation._pid_alive", lambda pid: pid == 123)
    assert recover_orphan_validations(tmp_path, min_age_seconds=1) == [stale]
    assert live.exists() and bad.exists()


def test_coco_duplicate_annotation_and_out_of_bounds_rejected(tmp_path):
    data = dataset(tmp_path)
    raw = json.loads(Path(data.annotations_path).read_text())
    raw["annotations"].append(raw["annotations"][0].copy())
    Path(data.annotations_path).write_text(json.dumps(raw))
    values = data.model_dump()
    values["annotations_sha256"] = sha(Path(data.annotations_path))
    values["expected_annotation_count"] = 2
    values["digest"] = digest({k: v for k, v in values.items() if k != "digest"})
    with pytest.raises(ValueError, match="duplicate COCO annotation"):
        DatasetManifest(**values)


def test_manifest_source_binding_and_canonical_class_keys(tmp_path):
    item = target(tmp_path, "bound")
    forged = item.model_copy(update={"class_mapping": {"00": 1}})
    with pytest.raises(ValueError, match="canonical decimal"):
        ModelTarget.model_validate(forged.model_dump())
    manifest = VisionManifest(source="wrong.bin")
    Path(item.manifest_path).write_text("source: wrong.bin\n")
    values = item.model_dump()
    values["manifest_sha256"] = sha(Path(item.manifest_path))
    values["manifest_digest"] = digest(manifest.model_dump(mode="json"))
    with pytest.raises(ValueError, match="source is not bound"):
        ModelTarget(**values)


def test_model_target_rejects_source_suffix_incompatible_with_manifest_family(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    manifest = tmp_path / "model.yaml"
    value = VisionManifest(source=str(source), family="onnx")
    value.to_yaml(manifest)
    with pytest.raises(ValueError, match="suffix is incompatible"):
        ModelTarget(
            source=str(source),
            source_sha256=sha(source),
            manifest_path=str(manifest),
            manifest_sha256=sha(manifest),
            manifest_digest=digest(value.model_dump(mode="json")),
            device="cpu",
            class_mapping={"0": 1},
        )


def test_openvino_runtime_is_bound_to_validated_artifact_precision(tmp_path):
    source = tmp_path / "model.xml"
    source.write_bytes(b"openvino")
    manifest = tmp_path / "model.yaml"
    value = VisionManifest(
        source=str(source),
        family="onnx",
        artifacts=[
            {
                "path": str(source),
                "backend": "openvino",
                "precision": "fp16",
                "validated": True,
                "checksum": sha(source),
                "compatible": {"device": "cpu", "provider": "CPU"},
            }
        ],
    )
    value.to_yaml(manifest)
    item = ModelTarget(
        source=str(source),
        source_sha256=sha(source),
        manifest_path=str(manifest),
        manifest_sha256=sha(manifest),
        manifest_digest=digest(value.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )
    runtime = runtime_policy_contract(
        item,
        backend="openvino",
        provider="CPU",
        device="cpu",
        precision="fp16",
        input_shape=(1, 3, 640, 640),
    )
    assert runtime.backend == "openvino" and runtime.precision == "fp16"
    with pytest.raises(ValueError, match="validated manifest artifact"):
        runtime_policy_contract(
            item,
            backend="openvino",
            provider="CPU",
            device="cpu",
            precision="int8",
            input_shape=(1, 3, 640, 640),
        )


def test_materialized_runtime_contract_rejects_coordinated_digest_tamper(tmp_path):
    _, report, _ = materialize(tmp_path)
    payload = report.model_dump(mode="json")
    runtime = payload["reference"]["runtime"]
    runtime.update({"backend": "onnxruntime", "provider": "CPUExecutionProvider"})
    runtime["digest"] = digest({key: value for key, value in runtime.items() if key != "digest"})
    reference = payload["reference"]
    reference["digest"] = digest({key: value for key, value in reference.items() if key != "digest"})
    payload["digest"] = digest({key: value for key, value in payload.items() if key != "digest"})
    with pytest.raises(ValueError, match="requires internal sentinels"):
        ValidationReport.model_validate(payload)


def test_build_failure_leaves_no_output_or_lock(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    monkeypatch.setattr("vision.validation._score", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        _validate_materialized(
            [],
            [],
            data,
            ValidationPolicy(max_map_drop=0),
            tmp_path / "out",
            reference_target=left,
            candidate_target=right,
            reference_evaluated_image_ids=[1],
            candidate_evaluated_image_ids=[1],
        )
    assert not (tmp_path / "out").exists() and not (tmp_path / ".out.validation.lock").exists()


def test_partial_stage_marker_is_not_a_published_result(tmp_path):
    data, report, out = materialize(tmp_path)
    (out / "COMMITTED").unlink()
    with pytest.raises(ValueError, match="not published"):
        report.verify(out, report.policy_digest, allow_materialized=True)
    assert data.categories == [1]


def test_duplicate_categories_and_annotation_orphan_rejected(tmp_path):
    data = dataset(tmp_path)
    values = data.model_dump()
    values["categories"] = [1, 1]
    values["digest"] = digest({k: v for k, v in values.items() if k != "digest"})
    with pytest.raises(ValueError, match="duplicate dataset category"):
        DatasetManifest(**values)
    raw = json.loads(Path(data.annotations_path).read_text())
    raw["annotations"][0]["image_id"] = 99
    Path(data.annotations_path).write_text(json.dumps(raw))
    values = data.model_dump()
    values["annotations_sha256"] = sha(Path(data.annotations_path))
    values["digest"] = digest({k: v for k, v in values.items() if k != "digest"})
    with pytest.raises(ValueError, match="reference mismatch"):
        DatasetManifest(**values)


def test_no_clobber_rename_preserves_foreign_output(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    foreign = tmp_path / "out"
    original = __import__("vision.validation", fromlist=["_noreplace_rename"])._noreplace_rename

    def race(parent_fd, stage, destination):
        foreign.mkdir()
        (foreign / "foreign").write_text("preserve")
        return original(parent_fd, stage, destination)

    monkeypatch.setattr("vision.validation._noreplace_rename", race)
    with pytest.raises(FileExistsError):
        _validate_materialized(
            [],
            [],
            data,
            ValidationPolicy(max_map_drop=0),
            foreign,
            reference_target=left,
            candidate_target=right,
            reference_evaluated_image_ids=[1],
            candidate_evaluated_image_ids=[1],
        )
    assert (foreign / "foreign").read_text() == "preserve"


def test_postrename_fsync_failure_rolls_back_output_and_lock(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    original_rename = validation_module._noreplace_rename
    original_fsync = validation_module._fsync_fd
    state = {"published": False}

    def rename(parent_fd, source, destination):
        original_rename(parent_fd, source, destination)
        if destination == "out":
            state["published"] = True

    def fsync(fd):
        if state["published"]:
            raise OSError("postrename fsync")
        return original_fsync(fd)

    monkeypatch.setattr(validation_module, "_noreplace_rename", rename)
    monkeypatch.setattr(validation_module, "_fsync_fd", fsync)
    with pytest.raises(OSError, match="postrename fsync"):
        _validate_materialized(
            [],
            [],
            data,
            ValidationPolicy(max_map_drop=0),
            tmp_path / "out",
            reference_target=left,
            candidate_target=right,
            reference_evaluated_image_ids=[1],
            candidate_evaluated_image_ids=[1],
        )
    assert not (tmp_path / "out").exists() and not (tmp_path / ".out.validation.lock").exists()


def test_primary_error_survives_rollback_and_cleanup_failure(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    original_rename = validation_module._noreplace_rename
    original_fsync = validation_module._fsync_fd
    state = {"published": False}
    primary = OSError("primary postrename fsync")

    def rename(parent_fd, source, destination):
        if state["published"]:
            raise RuntimeError("rollback failure")
        original_rename(parent_fd, source, destination)
        if destination == "out":
            state["published"] = True

    def fsync(fd):
        if state["published"]:
            raise primary
        return original_fsync(fd)

    monkeypatch.setattr(validation_module, "_noreplace_rename", rename)
    monkeypatch.setattr(validation_module, "_fsync_fd", fsync)
    monkeypatch.setattr(validation_module, "_rmtree_at", lambda *args: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(OSError, match="primary postrename fsync") as caught:
        _validate_materialized(
            [],
            [],
            data,
            ValidationPolicy(max_map_drop=0),
            tmp_path / "out",
            reference_target=left,
            candidate_target=right,
            reference_evaluated_image_ids=[1],
            candidate_evaluated_image_ids=[1],
        )
    assert caught.value is primary
    assert any("rollback failed" in note for note in getattr(caught.value, "__notes__", []))
    assert (tmp_path / "out").exists() and (tmp_path / ".out.validation.lock").exists()
    lock = tmp_path / ".out.validation.lock"
    owner = json.loads(lock.read_text())
    owner["created_at"] = 0
    lock.write_text(json.dumps(owner))
    monkeypatch.setattr(validation_module, "_fsync_fd", original_fsync)
    monkeypatch.setattr(validation_module, "_pid_alive", lambda pid: False)
    recover_orphan_validations(tmp_path, min_age_seconds=1)
    assert (tmp_path / "out").exists() and not lock.exists()


def strict_owner(output="out", token="c" * 32, pid=99999, created=0, policy="a" * 64):
    return {
        "token": token,
        "pid": pid,
        "created_at": created,
        "policy_digest": policy,
        "output_name": output,
        "stage_name": f".{output}.validation-stage-{token}",
        "lock_name": f".{output}.validation.lock",
    }


def test_recovery_matrix_stage_lock_only_invalid_and_live(monkeypatch, tmp_path):
    dead = strict_owner()
    stage = tmp_path / dead["stage_name"]
    stage.mkdir()
    stage.joinpath(".validation-owner.json").write_text(json.dumps(dead))
    lock = tmp_path / dead["lock_name"]
    lock.write_text(json.dumps(dead))
    pre = strict_owner("pre", "d" * 32)
    (tmp_path / pre["lock_name"]).write_text(json.dumps(pre))
    bad = tmp_path / ".bad.validation.lock"
    bad.write_text(json.dumps({"token": "not-strict"}))
    live = strict_owner("live", "e" * 32, pid=123)
    (tmp_path / live["lock_name"]).write_text(json.dumps(live))
    monkeypatch.setattr(validation_module, "_pid_alive", lambda pid: pid == 123)
    removed = recover_orphan_validations(tmp_path, min_age_seconds=1)
    assert removed == [stage]
    assert not stage.exists() and not lock.exists() and not (tmp_path / pre["lock_name"]).exists()
    assert bad.exists() and (tmp_path / live["lock_name"]).exists()


def test_recovery_postpublish_lock_only_preserves_committed_output(monkeypatch, tmp_path):
    _, report, out = materialize(tmp_path)
    owner = strict_owner(policy=report.policy_digest)
    lock = tmp_path / owner["lock_name"]
    lock.write_text(json.dumps(owner))
    monkeypatch.setattr(validation_module, "_pid_alive", lambda pid: False)
    assert recover_orphan_validations(tmp_path, min_age_seconds=1) == []
    assert out.exists() and not lock.exists() and report.verify(out, report.policy_digest, allow_materialized=True)


@pytest.mark.parametrize("machine,number", [("x86_64", 316), ("amd64", 316), ("aarch64", 276), ("arm64", 276)])
def test_linux_renameat2_fallback_uses_explicit_numbers(monkeypatch, machine, number):
    calls = []

    class Libc:
        def syscall(self, *args):
            calls.append(args)
            return 0

    monkeypatch.setattr(validation_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(validation_module.platform, "machine", lambda: machine)
    monkeypatch.setattr(validation_module, "CDLL", lambda *args, **kwargs: Libc())
    validation_module._noreplace_rename(1, "a", "b")
    assert calls[0][0] == number


def test_linux_renameat2_unknown_arch_rejects(monkeypatch):
    class Libc:
        pass

    monkeypatch.setattr(validation_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(validation_module.platform, "machine", lambda: "ppc64le")
    monkeypatch.setattr(validation_module, "CDLL", lambda *args, **kwargs: Libc())
    with pytest.raises(RuntimeError, match="no verified"):
        validation_module._noreplace_rename(1, "a", "b")


def test_linux_renameat2_exported_symbol_is_preferred(monkeypatch):
    calls = []

    class Function:
        argtypes = None

        def __call__(self, *args):
            calls.append(args)
            return 0

    class Libc:
        renameat2 = Function()

        def syscall(self, *args):
            raise AssertionError("syscall fallback must not run")

    monkeypatch.setattr(validation_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(validation_module, "CDLL", lambda *args, **kwargs: Libc())
    validation_module._noreplace_rename(1, "a", "b")
    assert calls and calls[0][-1] == 1


def test_hungarian_equal_ties_are_canonical_across_permutations():
    policy = ValidationPolicy(max_map_drop=0, iou_threshold=0.1)
    items = [(Box(0, 0, 4, 4), 0.9, 1), (Box(0, 0, 4, 4), 0.8, 1)]
    canonical = None
    for refs in itertools.permutations(items):
        for cands in itertools.permutations(items):
            result = compare_detections(
                DetectionResult([x[0] for x in refs], [x[1] for x in refs], [x[2] for x in refs]),
                DetectionResult([x[0] for x in cands], [x[1] for x in cands], [x[2] for x in cands]),
                policy,
            )
            descriptors = tuple(
                sorted(
                    (
                        refs[pair["reference"]][0].as_xyxy(),
                        refs[pair["reference"]][1],
                        refs[pair["reference"]][2],
                        cands[pair["candidate"]][0].as_xyxy(),
                        cands[pair["candidate"]][1],
                        cands[pair["candidate"]][2],
                    )
                    for pair in result["pairs"]
                )
            )
            summary = (
                result["matched"],
                result["mean_iou"],
                result["mean_score_delta"],
                result["coverage"],
                descriptors,
            )
            canonical = summary if canonical is None else canonical
            assert summary == canonical


def test_cleanup_lost_owner_closes_every_fd(tmp_path):
    reservation = validation_module._reserve_output(tmp_path / "out", "a" * 64)
    result_fd = os.open(".", os.O_RDONLY, dir_fd=reservation["stage_fd"])
    reservation["result_fd"] = result_fd
    original_stage_fd, original_parent_fd = reservation["stage_fd"], reservation["parent_fd"]
    os.unlink(reservation["lock_name"], dir_fd=reservation["parent_fd"])
    validation_module._cleanup_owned(reservation)
    for fd in (result_fd, original_stage_fd, original_parent_fd):
        with pytest.raises(OSError) as exc:
            os.fstat(fd)
        assert exc.value.errno == errno.EBADF
    assert (tmp_path / reservation["stage_name"]).exists()


def test_cleanup_transient_rmtree_failure_retains_then_recovery_removes(monkeypatch, tmp_path):
    reservation = validation_module._reserve_output(tmp_path / "out", "a" * 64)
    original = validation_module._rmtree_at
    original_fds = (reservation["stage_fd"], reservation["parent_fd"])
    monkeypatch.setattr(validation_module, "_rmtree_at", lambda *args: (_ for _ in ()).throw(OSError("transient")))
    validation_module._cleanup_owned(reservation)
    for fd in original_fds:
        with pytest.raises(OSError) as exc:
            os.fstat(fd)
        assert exc.value.errno == errno.EBADF
    stage, lock = tmp_path / reservation["stage_name"], tmp_path / reservation["lock_name"]
    assert stage.exists() and lock.exists()
    monkeypatch.setattr(validation_module, "_rmtree_at", original)
    owner = json.loads(lock.read_text())
    owner["created_at"] = 0
    lock.write_text(json.dumps(owner))
    stage.joinpath(".validation-owner.json").write_text(json.dumps(owner))
    monkeypatch.setattr(validation_module, "_pid_alive", lambda pid: False)
    assert recover_orphan_validations(tmp_path, min_age_seconds=1) == [stage]
    assert not stage.exists() and not lock.exists()


def test_recovery_distinct_partial_and_committed_stages(monkeypatch, tmp_path):
    partial = strict_owner("partial", "f" * 32)
    committed = strict_owner("committed", "1" * 32)
    for owner, complete in ((partial, False), (committed, True)):
        stage = tmp_path / owner["stage_name"]
        stage.mkdir()
        stage.joinpath(".validation-owner.json").write_text(json.dumps(owner))
        if complete:
            result = stage / "result"
            result.mkdir()
            result.joinpath("validation_report.json").write_text("{}")
            result.joinpath("COMMITTED").write_text("x")
            stage.joinpath("validation_report.json").write_text("{}")
            stage.joinpath("COMMITTED").write_text("x")
        (tmp_path / owner["lock_name"]).write_text(json.dumps(owner))
    monkeypatch.setattr(validation_module, "_pid_alive", lambda pid: False)
    removed = recover_orphan_validations(tmp_path, min_age_seconds=1)
    assert set(removed) == {tmp_path / partial["stage_name"], tmp_path / committed["stage_name"]}
    for owner in (partial, committed):
        assert not (tmp_path / owner["stage_name"]).exists() and not (tmp_path / owner["lock_name"]).exists()


def test_cli_validate_dataset_pass_failure_and_runtime_cleanup(monkeypatch, tmp_path):
    data = dataset(tmp_path)
    left, right = target(tmp_path, "left"), target(tmp_path, "right")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(data.model_dump_json())
    runner = CliRunner()

    class Model:
        def __init__(self, empty=False):
            self.empty = empty

        def runtime_identity(self):
            return dict(RUNTIME_IDENTITY)

        def __call__(self, path):
            common = dict(RUNTIME_IDENTITY)
            return (
                DetectionResult(metadata=common)
                if self.empty
                else DetectionResult([Box(1, 1, 5, 5)], [0.9], [0], metadata=common)
            )

    base = [
        "validate-dataset",
        str(dataset_path),
        str(tmp_path / "out"),
        "0",
        "--reference-source",
        left.source,
        "--reference-manifest",
        left.manifest_path,
        "--candidate-source",
        right.source,
        "--candidate-manifest",
        right.manifest_path,
        "--reference-class-mapping",
        '{"0": 1}',
        "--candidate-class-mapping",
        '{"0": 1}',
    ]
    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: Model())
    passed = runner.invoke(app, base)
    assert passed.exit_code == 0 and (tmp_path / "out" / "COMMITTED").exists()
    monkeypatch.setattr(
        vision, "load", lambda *args, **kwargs: Model(kwargs.get("device") == "cpu" and args[0] == right.source)
    )
    # Candidate mismatch through a source-aware loader.
    monkeypatch.setattr(vision, "load", lambda source, **kwargs: Model(source == right.source))
    failed = runner.invoke(app, base[:2] + [str(tmp_path / "failed")] + base[3:])
    assert failed.exit_code == 1 and (tmp_path / "failed" / "COMMITTED").exists()
    monkeypatch.setattr(vision, "load", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("runtime")))
    broken = runner.invoke(app, base[:2] + [str(tmp_path / "broken")] + base[3:])
    assert (
        broken.exit_code != 0 and not (tmp_path / "broken").exists() and not list(tmp_path.glob(".broken.validation-*"))
    )


def test_detection_ids_are_recomputed_even_for_coordinated_report_forgery(tmp_path):
    _, report, out = materialize(tmp_path)
    payload = report.model_dump(mode="json")
    payload["reference"]["detection_image_ids"] = []
    payload["reference"]["digest"] = _digest({k: v for k, v in payload["reference"].items() if k != "digest"})
    payload["digest"] = _digest({k: v for k, v in payload.items() if k != "digest"})
    forged = ValidationReport(**payload)
    exact = json.dumps(forged.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    (out / "result" / "reference_prediction_manifest.json").write_text(
        json.dumps(forged.reference.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    )
    (out / "result" / "validation_report.json").write_text(exact)
    (out / "validation_report.json").write_text(exact)
    marker = hashlib.sha256(exact.encode()).hexdigest()
    (out / "result" / "COMMITTED").write_text(marker)
    (out / "COMMITTED").write_text(marker)
    with pytest.raises(ValueError, match="detection image IDs"):
        forged.verify(out, forged.policy_digest, allow_materialized=True)


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b"])
def test_publication_basename_rejects_traversal(name):
    with pytest.raises(ValueError, match="basename"):
        validation_module._basename(name)


@pytest.mark.parametrize(
    "mutation",
    [
        {"pid": 0},
        {"created_at": float("nan")},
        {"policy_digest": "g" * 64},
        {"token": "x" * 31},
        {"stage_name": ".wrong.validation-stage-" + "a" * 32},
    ],
)
def test_owner_schema_rejects_partial_or_unbound_records(mutation):
    token = "a" * 32
    owner = {
        "token": token,
        "pid": 1,
        "created_at": 1.0,
        "policy_digest": "b" * 64,
        "output_name": "out",
        "stage_name": f".out.validation-stage-{token}",
        "lock_name": ".out.validation.lock",
    }
    owner.update(mutation)
    assert not validation_module._strict_owner(owner)
