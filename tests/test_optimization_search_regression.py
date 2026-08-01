"""Independent search-selection regressions with signed public evidence."""

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
    SearchReport,
    SearchRequest,
    SearchSpace,
    ShapeProfile,
    Stage,
    StageEvidence,
    ValidationEvidence,
    _comparable,
    digest,
    expand_candidates,
    expected_build_request_digest,
    optimize_search,
)
from vision.types import Box, DetectionResult
from vision.validation import DatasetManifest, ModelTarget, ValidationPolicy, runtime_policy_contract, validate_dataset

HEX = "a" * 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(**changes) -> CandidatePlan:
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


def profile(**changes) -> ShapeProfile:
    value = {
        "input_name": "images",
        "min_shape": (1, 3, 320, 320),
        "opt_shape": (1, 3, 640, 640),
        "max_shape": (2, 3, 1280, 1280),
    }
    value.update(changes)
    return ShapeProfile(**value)


def target(
    root: Path, name: str, family: str = "torch", input_shape: tuple[int, int, int, int] = (1, 3, 640, 640)
) -> ModelTarget:
    root.mkdir(parents=True, exist_ok=True)
    source = root / f"{name}{'.onnx' if family == 'onnx' else '.pt'}"
    source.write_bytes(name.encode())
    manifest_path = root / f"{name}.yaml"
    manifest = VisionManifest(source=str(source), family=family, input={"shape": list(input_shape)})
    manifest.to_yaml(manifest_path)
    return ModelTarget(
        source=str(source),
        source_sha256=sha(source),
        manifest_path=str(manifest_path),
        manifest_sha256=sha(manifest_path),
        manifest_digest=digest(manifest.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )


def dataset(root: Path) -> DatasetManifest:
    image, annotations = root / "image.jpg", root / "annotations.json"
    Image.new("RGB", (10, 10)).save(image)
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image.name, "width": 10, "height": 10}],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}
                ],
            }
        )
    )
    manifest = VisionManifest(source="placeholder.onnx")
    body = {
        "split": "tiny",
        "annotations_path": str(annotations),
        "annotations_sha256": sha(annotations),
        "images": [{"id": 1, "path": str(image), "sha256": sha(image), "width": 10, "height": 10}],
        "categories": [1],
        "expected_image_count": 1,
        "expected_annotation_count": 1,
        "format": "coco",
        "cocoeval_iou_type": "bbox",
        "cocoeval_version": "pycocotools",
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
    }
    return DatasetManifest(**(body | {"digest": digest(body)}))


def request(
    root: Path, *, space: SearchSpace | None = None, validation_drop: float = 0.0
) -> tuple[SearchRequest, DatasetManifest]:
    reference = target(root, "reference")
    data = dataset(root)
    base = plan(source_model_digest=reference.source_sha256)
    ref_runtime = runtime_policy_contract(
        reference, backend="pytorch", provider="torch", device="cpu", precision="fp32", input_shape=(1, 3, 640, 640)
    )
    return (
        SearchRequest(
            baseline=base,
            space=space
            or SearchSpace(
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
            validation_policy=ValidationPolicy(max_map_drop=validation_drop),
            dataset_digest=data.digest,
            reference_target=reference,
            reference_runtime=ref_runtime,
            reference_source_model_digest=reference.source_sha256,
            build_template_digest="b" * 64,
            output=str(root / "search"),
        ),
        data,
    )


def validation(
    root: Path,
    data: DatasetManifest,
    reference: ModelTarget,
    candidate: ModelTarget,
    item: CandidatePlan,
    *,
    candidate_detects: bool,
    max_map_drop: float,
):
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

    class Model:
        def __init__(self, runtime, detects):
            self.runtime, self.detects = runtime, detects

        def runtime_identity(self):
            return self.runtime

        def __call__(self, _path):
            if not self.detects:
                return DetectionResult(metadata=self.runtime)
            return DetectionResult([Box(1, 1, 5, 5)], [0.9], [0], metadata=self.runtime)

    original = vision.load
    models = iter(
        (
            Model(ref_runtime.model_dump(exclude={"source_sha256", "manifest_digest", "digest"}), True),
            Model(
                candidate_runtime.model_dump(exclude={"source_sha256", "manifest_digest", "digest"}), candidate_detects
            ),
        )
    )
    vision.load = lambda *_args, **_kwargs: next(models)
    output = root / f"validation-{item.candidate_id}"
    try:
        return validate_dataset(
            reference,
            candidate,
            data,
            ValidationPolicy(
                max_map_drop=max_map_drop, reference_runtime=ref_runtime, candidate_runtime=candidate_runtime
            ),
            output,
        ), output
    finally:
        vision.load = original


def evaluation(
    root: Path,
    req: SearchRequest,
    data: DatasetManifest,
    item: CandidatePlan,
    *,
    latency: float,
    candidate_detects: bool = True,
) -> CandidateEvaluation:
    root.mkdir(parents=True, exist_ok=True)
    baseline = item.candidate_id == req.baseline.candidate_id
    candidate = (
        req.reference_target
        if baseline
        else target(
            root,
            f"artifact-{item.candidate_id}",
            "onnx" if item.backend == "onnxruntime" else "torch",
            (item.batch, 3, *item.resolution),
        )
    )
    upstream_path = None
    if not baseline:
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
            profiles=[value.model_dump(mode="json") for value in item.profiles],
        ).finalized()
        upstream_path.write_text(json.dumps(upstream.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
    record_path = root / f"build-{item.candidate_id}.json"
    raw = {
        "schema_version": 1,
        "kind": "original" if baseline else "exported",
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
        "profiles": [value.model_dump(mode="json") for value in item.profiles],
        "upstream_build_report_path": str(upstream_path) if upstream_path else None,
        "upstream_build_report_file_sha256": sha(upstream_path) if upstream_path else None,
        "upstream_build_report_self_digest": json.loads(upstream_path.read_text())["self_digest"]
        if upstream_path
        else None,
    }
    record = raw | {"self_digest": digest(raw)}
    record_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))
    resolved = ResolvedArtifactEvidence(
        candidate_id=item.candidate_id,
        artifact_digest=candidate.source_sha256,
        build_request_digest=raw["build_request_digest"],
        manifest_digest=candidate.manifest_digest,
        target_digest=raw["target_digest"],
        build_report_digest=record["self_digest"],
        artifact_path=candidate.source,
        artifact_sha256=sha(Path(candidate.source)),
        manifest_path=candidate.manifest_path,
        manifest_sha256=sha(Path(candidate.manifest_path)),
        build_report_path=str(record_path),
        build_report_sha256=sha(record_path),
    )
    report, root_path = validation(
        root,
        data,
        req.reference_target,
        candidate,
        item,
        candidate_detects=candidate_detects,
        max_map_drop=req.validation_policy.max_map_drop,
    )
    evidence = ValidationEvidence(
        report=report,
        root=str(root_path),
        expected_policy_digest=report.policy_digest,
        resolved=resolved,
        binding_digest=digest(
            {
                "report_digest": report.digest,
                "resolved": resolved.model_dump(mode="json"),
                "policy_digest": report.policy_digest,
            }
        ),
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
    benchmark = BenchmarkReport(
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
            StageEvidence(stage=stage)
            for stage in (
                Stage.capability,
                Stage.build,
                Stage.inspect,
                Stage.parity,
                Stage.dataset_map,
                Stage.benchmark,
            )
        ),
        validation=evidence,
        benchmark=benchmark,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"min_shape": (1, 3), "opt_shape": (1, 3, 2), "max_shape": (1, 3, 4)},
        {"min_shape": (0, 3, 1, 1)},
        {"min_shape": (2, 3, 320, 320), "opt_shape": (1, 3, 640, 640)},
    ],
)
def test_shape_profile_rejects_rank_domain_and_order(changes):
    with pytest.raises(ValueError):
        profile(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"dynamic": True},
        {"profiles": (profile(),)},
        {"dynamic": True, "profiles": (profile(opt_shape=(2, 3, 640, 640)),)},
        {"dynamic": True, "profiles": (profile(opt_shape=(1, 3, 320, 320)),)},
    ],
)
def test_candidate_plan_binds_dynamic_profiles_to_batch_and_resolution(changes):
    with pytest.raises(ValueError):
        plan(**changes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backends": ()},
        {"precisions": ()},
        {"devices": ()},
        {"providers": ()},
        {"resolutions": ()},
        {"batches": ()},
        {"dynamics": ()},
        {"batches": (0,)},
        {"resolutions": ((0, 640),)},
    ],
)
def test_search_space_rejects_empty_and_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        SearchSpace(max_candidates=1, **kwargs)


def test_search_space_rejects_profile_outside_matrix():
    with pytest.raises(ValueError, match="absent"):
        SearchSpace(
            max_candidates=2,
            batches=(1,),
            resolutions=((640, 640),),
            dynamics=(True,),
            profiles=(profile(opt_shape=(2, 3, 640, 640)),),
        )


def test_candidate_identity_is_only_canonical_plan_content():
    item = plan()
    assert item.candidate_id == digest(item.model_dump(mode="json"))
    with pytest.raises(ValueError):
        CandidatePlan.model_validate(item.model_dump() | {"build_request_digest": HEX})


def test_expand_candidates_dedupes_before_limit_and_rejects_actual_overflow(tmp_path):
    deduped, _ = request(
        tmp_path / "dedupe",
        space=SearchSpace(backends=("pytorch", "pytorch"), providers=("torch",), max_candidates=1),
    )
    assert len(expand_candidates(deduped)) == 1
    overflowing, _ = request(
        tmp_path / "overflow",
        space=SearchSpace(backends=("pytorch", "onnxruntime"), providers=("torch",), max_candidates=1),
    )
    with pytest.raises(ValueError, match="after dedupe"):
        expand_candidates(overflowing)


def test_faster_quality_degraded_candidate_is_rejected(tmp_path):
    req, data = request(tmp_path, validation_drop=1)
    outcome = optimize_search(
        req,
        lambda item: evaluation(
            tmp_path,
            req,
            data,
            item,
            latency=0.1 if item.backend == "onnxruntime" else 2.0,
            candidate_detects=item.backend == "pytorch",
        ),
    )
    degraded = next(
        item
        for item in outcome.evaluations
        if item.plan.backend == "onnxruntime" and item.plan.provider == "CPUExecutionProvider"
    )
    assert degraded.stages[-1].stage is Stage.rejected
    assert "quality gate" in degraded.stages[-1].payload["reason"]
    assert outcome.selected_candidate_id == req.baseline.candidate_id


def test_search_report_rejects_selected_rejected_nonwinner_and_none_with_accepted(tmp_path):
    req, data = request(tmp_path)
    report = optimize_search(
        req, lambda item: evaluation(tmp_path, req, data, item, latency=1.0 if item.backend == "onnxruntime" else 2.0)
    )
    selected = next(item for item in report.evaluations if item.plan.candidate_id == report.selected_candidate_id)
    rejected = selected.model_copy(update={"stages": selected.stages[:-1] + (StageEvidence(stage=Stage.rejected),)})
    values = tuple(rejected if item is selected else item for item in report.evaluations)
    with pytest.raises(ValueError, match="terminal accepted"):
        SearchReport(request=req, evaluations=values, selected_candidate_id=selected.plan.candidate_id)
    with pytest.raises(ValueError, match="deterministic"):
        SearchReport(request=req, evaluations=report.evaluations, selected_candidate_id=req.baseline.candidate_id)
    with pytest.raises(ValueError, match="selected candidate is required"):
        SearchReport(request=req, evaluations=report.evaluations, selected_candidate_id=None)


def test_deterministic_ties_include_candidate_id(tmp_path):
    req, data = request(tmp_path)
    report = optimize_search(req, lambda item: evaluation(tmp_path, req, data, item, latency=1.0))
    accepted = [item.plan.candidate_id for item in report.evaluations if item.stages[-1].stage is Stage.accepted]
    assert report.selected_candidate_id == min(accepted)


def altered_benchmark(baseline, *, fingerprint_updates=None, request_updates=None):
    request = baseline.request
    if fingerprint_updates:
        request = request.model_copy(update={"fingerprint": request.fingerprint.model_copy(update=fingerprint_updates)})
    if request_updates:
        request = request.model_copy(update=request_updates)
    return BenchmarkReport(
        request=request,
        samples_ms=baseline.samples_ms,
        artifact_size_bytes=baseline.artifact_size_bytes,
    )


@pytest.mark.parametrize("change", ["batch", "resolution", "static_dynamic"])
def test_comparability_rejects_workload_shape_changes(tmp_path, change):
    req, data = request(tmp_path)
    baseline = evaluation(tmp_path, req, data, req.baseline, latency=1.0)
    updates = {
        "batch": {"batch": 2},
        "resolution": {"resolution": (320, 320)},
        "static_dynamic": {"dynamic": True, "profiles": (profile(),)},
    }[change]
    assert not _comparable(
        baseline.benchmark, altered_benchmark(baseline.benchmark, request_updates=updates), req.policy
    )


def test_comparability_rejects_dynamic_profile_change_without_dynamic_change(tmp_path):
    req, data = request(tmp_path)
    baseline = evaluation(tmp_path, req, data, req.baseline, latency=1.0).benchmark
    first = altered_benchmark(baseline, request_updates={"dynamic": True, "profiles": (profile(),)})
    second_profile = profile(min_shape=(1, 3, 480, 480), max_shape=(2, 3, 960, 960))
    second = altered_benchmark(
        baseline,
        request_updates={"dynamic": True, "profiles": (second_profile,)},
    )
    assert not _comparable(first, second, SearchPolicy(max_map_drop=0, normalized_workload_semantics=True))


@pytest.mark.parametrize("field", ["workload_digest", "input_digest"])
def test_normalized_workload_semantics_relaxes_only_workload_and_input(tmp_path, field):
    req, data = request(tmp_path)
    baseline = evaluation(tmp_path, req, data, req.baseline, latency=1.0).benchmark
    changed = altered_benchmark(baseline, fingerprint_updates={field: "b" * 64})
    assert not _comparable(baseline, changed, SearchPolicy(max_map_drop=0))
    assert _comparable(baseline, changed, SearchPolicy(max_map_drop=0, normalized_workload_semantics=True))


@pytest.mark.parametrize(
    ("fingerprint_updates", "request_updates"),
    [
        ({"hardware_digest": "b" * 64}, None),
        ({"harness_digest": "b" * 64}, None),
        (None, {"batch": 2}),
        (None, {"resolution": (320, 320)}),
        (None, {"dynamic": True, "profiles": (profile(),)}),
    ],
)
def test_normalized_workload_semantics_rejects_immutable_hardware_and_shape_changes(
    tmp_path, fingerprint_updates, request_updates
):
    req, data = request(tmp_path)
    baseline = evaluation(tmp_path, req, data, req.baseline, latency=1.0).benchmark
    changed = altered_benchmark(baseline, fingerprint_updates=fingerprint_updates, request_updates=request_updates)
    assert not _comparable(baseline, changed, SearchPolicy(max_map_drop=0, normalized_workload_semantics=True))


@pytest.mark.parametrize("field", ["threads", "mode"])
def test_normalized_workload_semantics_rejects_threads_and_mode(tmp_path, field):
    req, data = request(tmp_path)
    baseline = evaluation(tmp_path, req, data, req.baseline, latency=1.0).benchmark
    changed = altered_benchmark(
        baseline,
        fingerprint_updates={field: 2 if field == "threads" else "inference-only"},
    )
    assert not _comparable(baseline, changed, SearchPolicy(max_map_drop=0, normalized_workload_semantics=True))


def test_normalized_workload_relaxes_only_workload_and_input_digests(tmp_path):
    strict, data = request(tmp_path / "strict")
    normalized, normalized_data = request(tmp_path / "normal")
    normalized = normalized.model_copy(
        update={"policy": SearchPolicy(max_map_drop=0, normalized_workload_semantics=True)}
    )

    def evaluator(req, input_data, root):
        def run(item):
            value = evaluation(root, req, input_data, item, latency=1.0)
            if item.backend == "onnxruntime":
                fingerprint = value.benchmark.request.fingerprint.model_copy(
                    update={"workload_digest": "c" * 64, "input_digest": "b" * 64}
                )
                benchmark = BenchmarkReport(
                    request=value.benchmark.request.model_copy(update={"fingerprint": fingerprint}),
                    samples_ms=value.benchmark.samples_ms,
                    artifact_size_bytes=10,
                )
                return value.model_copy(update={"benchmark": benchmark})
            return value

        return run

    strict_report = optimize_search(strict, evaluator(strict, data, tmp_path / "strict"))
    strict_alt = next(item for item in strict_report.evaluations if item.plan.backend == "onnxruntime")
    assert strict_alt.stages[-1].stage is Stage.rejected
    normal_report = optimize_search(normalized, evaluator(normalized, normalized_data, tmp_path / "normal"))
    assert normal_report.selected_candidate_id != normalized.baseline.candidate_id
