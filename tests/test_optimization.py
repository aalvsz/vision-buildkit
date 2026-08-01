import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

import vision
from vision.export import BuildReport
from vision.manifest import VisionManifest
from vision.optimization import (
    BenchmarkReport,
    BenchmarkRequest,
    CandidateEvaluation,
    CandidatePlan,
    ResolvedArtifactEvidence,
    RuntimeFingerprint,
    SearchPolicy,
    SearchRequest,
    SearchSpace,
    Stage,
    StageEvidence,
    ValidationEvidence,
    _bound,
    digest,
    expand_candidates,
    expected_build_request_digest,
    optimize_search,
)
from vision.types import DetectionResult
from vision.validation import DatasetManifest, ModelTarget, ValidationPolicy, runtime_policy_contract, validate_dataset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(root, name, family="torch"):
    suffix = ".onnx" if family == "onnx" else ".pt"
    source = root / f"{name}{suffix}"
    source.write_bytes(name.encode())
    manifest = root / f"{name}.yaml"
    model = VisionManifest(source=str(source), family=family)
    model.to_yaml(manifest)
    return ModelTarget(
        source=str(source),
        source_sha256=sha(source),
        manifest_path=str(manifest),
        manifest_sha256=sha(manifest),
        manifest_digest=digest(model.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )


def dataset(root):
    image = root / "image.jpg"
    Image.new("RGB", (10, 10)).save(image)
    annotation = root / "ann.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image.name, "width": 10, "height": 10}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [],
            }
        )
    )
    manifest = VisionManifest(source="source.onnx")
    payload = {
        "split": "tiny",
        "annotations_path": str(annotation),
        "annotations_sha256": sha(annotation),
        "images": [{"id": 1, "path": str(image), "sha256": sha(image), "width": 10, "height": 10}],
        "categories": [1],
        "preprocessing_digest": digest(
            {
                "input": manifest.input.model_dump(mode="json"),
                "preprocessing": manifest.preprocessing.model_dump(mode="json"),
            }
        ),
        "postprocessing_digest": digest(
            {
                "output": manifest.output.model_dump(mode="json"),
                "decoder": manifest.decoder.model_dump(mode="json"),
                "postprocessing": manifest.postprocessing,
                "class_mapping": {"0": 1},
            }
        ),
        "expected_image_count": 1,
        "expected_annotation_count": 0,
        "format": "coco",
        "cocoeval_iou_type": "bbox",
        "cocoeval_version": "pycocotools",
    }
    return DatasetManifest(**(payload | {"digest": digest(payload)}))


def plan(source, **changes):
    value = {
        "backend": "pytorch",
        "precision": "fp32",
        "device": "cpu",
        "provider": "torch",
        "resolution": (640, 640),
        "batch": 1,
        "source_model_digest": source,
    }
    value.update(changes)
    return CandidatePlan(**value)


def report_for(root, data, reference, candidate, item):
    ref_runtime = runtime_policy_contract(
        reference, backend="pytorch", provider="torch", device="cpu", precision="fp32", input_shape=(1, 3, 640, 640)
    )
    candidate_runtime = runtime_policy_contract(
        candidate,
        backend=item.backend,
        provider=item.provider,
        device=item.device,
        precision=item.precision,
        input_shape=(item.batch, 3, *item.resolution),
    )
    policy = ValidationPolicy(max_map_drop=0, reference_runtime=ref_runtime, candidate_runtime=candidate_runtime)

    class Model:
        def __init__(self, runtime):
            self.runtime = runtime

        def runtime_identity(self):
            return self.runtime

        def __call__(self, path):
            return DetectionResult(metadata=self.runtime)

    original = vision.load
    models = iter(
        (
            Model(ref_runtime.model_dump(exclude={"source_sha256", "manifest_digest", "digest"})),
            Model(candidate_runtime.model_dump(exclude={"source_sha256", "manifest_digest", "digest"})),
        )
    )
    vision.load = lambda *args, **kwargs: next(models)
    try:
        report = validate_dataset(reference, candidate, data, policy, root / f"validation-{item.candidate_id}")
    finally:
        vision.load = original
    return report, str(root / f"validation-{item.candidate_id}")


def request(root):
    reference = target(root, "reference")
    data = dataset(root)
    base = plan(reference.source_sha256)
    ref_runtime = runtime_policy_contract(
        reference, backend="pytorch", provider="torch", device="cpu", precision="fp32", input_shape=(1, 3, 640, 640)
    )
    return SearchRequest(
        baseline=base,
        space=SearchSpace(
            backends=("pytorch", "onnxruntime"),
            precisions=("fp32",),
            devices=("cpu",),
            providers=("torch", "CPUExecutionProvider"),
            resolutions=((640, 640),),
            batches=(1,),
            dynamics=(False,),
            max_candidates=4,
        ),
        policy=SearchPolicy(max_map_drop=0),
        validation_policy=ValidationPolicy(max_map_drop=0),
        dataset_digest=data.digest,
        reference_target=reference,
        reference_runtime=ref_runtime,
        reference_source_model_digest=reference.source_sha256,
        build_template_digest="b" * 64,
        output=str(root / "out"),
    ), data


def evaluate(root, req, data, item, latency):
    exported = item != req.baseline
    candidate = (
        req.reference_target
        if not exported
        else target(root, f"artifact-{item.backend}", "onnx" if item.backend == "onnxruntime" else "torch")
    )
    upstream_path = None
    if exported:
        upstream_path = root / f"upstream-{item.candidate_id}.json"
        upstream = BuildReport(
            source=req.reference_target.source,
            intermediate=None,
            output=candidate.source,
            target=item.backend,
            requested_precision=item.precision,
            effective_precision=item.precision,
            source_digest=item.source_model_digest,
            intermediate_digest=item.source_model_digest,
            output_digest=candidate.source_sha256,
            artifact_digests={Path(candidate.source).name: candidate.source_sha256},
            dynamic=item.dynamic,
            profiles=[profile.model_dump(mode="json") for profile in item.profiles],
        ).finalized()
        upstream_path.write_text(json.dumps(upstream.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
    build_path = root / f"build-{item.candidate_id}.json"
    raw = {
        "schema_version": 1,
        "kind": "exported" if exported else "original",
        "candidate_id": item.candidate_id,
        "plan_digest": item.candidate_id,
        "source_model_digest": item.source_model_digest,
        "build_template_digest": req.build_template_digest,
        "build_request_digest": expected_build_request_digest(item, req.build_template_digest),
        "artifact_digest": candidate.source_sha256,
        "artifact_path": candidate.source,
        "manifest_path": candidate.manifest_path,
        "manifest_digest": candidate.manifest_digest,
        "target_digest": digest(candidate.model_dump(mode="json")),
        "backend": item.backend,
        "precision": item.precision,
        "device": item.device,
        "provider": item.provider,
        "resolution": item.resolution,
        "batch": item.batch,
        "dynamic": item.dynamic,
        "profiles": item.profiles,
        "upstream_build_report_path": str(upstream_path) if upstream_path else None,
        "upstream_build_report_file_sha256": sha(upstream_path) if upstream_path else None,
        "upstream_build_report_self_digest": (
            json.loads(upstream_path.read_text())["self_digest"] if upstream_path else None
        ),
    }
    record = raw | {"self_digest": digest(raw)}
    build_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    resolved = ResolvedArtifactEvidence(
        candidate_id=item.candidate_id,
        artifact_digest=candidate.source_sha256,
        build_request_digest=expected_build_request_digest(item, req.build_template_digest),
        build_report_digest=record["self_digest"],
        manifest_digest=candidate.manifest_digest,
        target_digest=digest(candidate.model_dump(mode="json")),
        artifact_path=candidate.source,
        artifact_sha256=sha(Path(candidate.source)),
        manifest_path=candidate.manifest_path,
        manifest_sha256=sha(Path(candidate.manifest_path)),
        build_report_path=str(build_path),
        build_report_sha256=sha(build_path),
    )
    report, report_root = report_for(root, data, req.reference_target, candidate, item)
    binding = digest(
        {
            "report_digest": report.digest,
            "resolved": resolved.model_dump(mode="json"),
            "policy_digest": report.policy_digest,
        }
    )
    evidence = ValidationEvidence(
        report=report,
        root=report_root,
        expected_policy_digest=report.policy_digest,
        resolved=resolved,
        binding_digest=binding,
    )
    fingerprint = RuntimeFingerprint(
        hardware_digest="d" * 64,
        harness_digest="e" * 64,
        workload_digest="f" * 64,
        input_digest="a" * 64,
        artifact_digest=resolved.artifact_digest,
        backend=item.backend,
        precision=item.precision,
        runtime_version="test",
        device=item.device,
        provider=item.provider,
        threads=1,
        mode="e2e",
    )
    bench = BenchmarkReport(
        request=BenchmarkRequest(
            fingerprint=fingerprint,
            resolved=resolved,
            resolution=item.resolution,
            dynamic=item.dynamic,
            profiles=item.profiles,
            warmup=0,
            repeats=2,
            batch=item.batch,
        ),
        samples_ms=(latency, latency),
        artifact_size_bytes=10,
    )
    return CandidateEvaluation(
        plan=item,
        resolved=resolved,
        stages=tuple(
            StageEvidence(stage=x)
            for x in (Stage.capability, Stage.build, Stage.inspect, Stage.parity, Stage.dataset_map, Stage.benchmark)
        ),
        validation=evidence,
        benchmark=bench,
    )


def test_plan_matrix_is_prebuild_only_and_dedupes(tmp_path):
    req, _ = request(tmp_path)
    plans = expand_candidates(req)
    assert len(plans) == 4 and all(plan.source_model_digest == req.reference_source_model_digest for plan in plans)
    assert not hasattr(plans[0], "artifact_digest")
    with pytest.raises(ValueError):
        CandidatePlan.model_validate({**plans[0].model_dump(), "artifact_digest": "a" * 64})


def test_distinct_onnx_artifact_is_accepted_and_wins(tmp_path):
    req, data = request(tmp_path)
    outcome = optimize_search(
        req, lambda item: evaluate(tmp_path, req, data, item, 2.0 if item.backend == "pytorch" else 1.0)
    )
    winner = next(x for x in outcome.evaluations if x.plan.candidate_id == outcome.selected_candidate_id)
    assert (
        winner.plan.backend == "onnxruntime" and winner.resolved.artifact_digest != req.reference_target.source_sha256
    )


def test_plan_and_resolved_reuse_are_rejected(tmp_path):
    req, data = request(tmp_path)
    good = evaluate(tmp_path, req, data, req.baseline, 1.0)
    other = plan(req.reference_source_model_digest, backend="onnxruntime")
    with pytest.raises(ValueError, match="different candidate plan"):
        CandidateEvaluation(
            plan=other, resolved=good.resolved, stages=good.stages, validation=good.validation, benchmark=good.benchmark
        )
    tampered = good.resolved.model_copy(update={"artifact_digest": "0" * 64})
    assert not hasattr(good.plan, "build_request_digest") and tampered != good.resolved


def test_external_anchors_reject_coordinated_report_policy_rewrite(tmp_path):
    req, data = request(tmp_path)
    good = evaluate(tmp_path, req, data, req.baseline, 1.0)
    forged = good.validation.report.model_copy(update={"policy_digest": "0" * 64})
    bad = good.validation.model_copy(update={"report": forged, "expected_policy_digest": "0" * 64})
    assert not _bound(good.model_copy(update={"validation": bad}), req)[0]


def test_failure_and_persistence_paths(tmp_path):
    req, _ = request(tmp_path)
    with pytest.raises(RuntimeError, match="baseline PyTorch fp32 failed"):
        optimize_search(req, lambda item: CandidateEvaluation(plan=item, stages=(), error="no runtime"))
