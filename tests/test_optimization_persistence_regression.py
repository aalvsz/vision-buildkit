"""Persistence and CLI regression boundaries for optimization evidence.

These deliberately use real, signed public schemas rather than dictionaries so
that reopening a saved search crosses the same trust boundaries as a capsule.
"""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

import vision
from vision.cli import app
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
    _persist,
    digest,
    expected_build_request_digest,
    load_search_report,
    optimize_search,
    recover_orphan_searches,
)
from vision.types import DetectionResult
from vision.validation import DatasetManifest, ModelTarget, ValidationPolicy, runtime_policy_contract, validate_dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(root: Path, name: str, family: str = "torch") -> ModelTarget:
    source = root / f"{name}{'.onnx' if family == 'onnx' else '.pt'}"
    source.write_bytes(name.encode())
    manifest_path = root / f"{name}.yaml"
    manifest = VisionManifest(source=str(source), family=family)
    manifest.to_yaml(manifest_path)
    return ModelTarget(
        source=str(source),
        source_sha256=_sha(source),
        manifest_path=str(manifest_path),
        manifest_sha256=_sha(manifest_path),
        manifest_digest=digest(manifest.model_dump(mode="json")),
        device="cpu",
        class_mapping={"0": 1},
    )


def _dataset(root: Path) -> DatasetManifest:
    image, annotations = root / "image.jpg", root / "annotations.json"
    Image.new("RGB", (10, 10)).save(image)
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": image.name, "width": 10, "height": 10}],
                "categories": [{"id": 1, "name": "x"}],
                "annotations": [],
            }
        )
    )
    manifest = VisionManifest(source="placeholder.onnx")
    body = {
        "split": "tiny",
        "annotations_path": str(annotations),
        "annotations_sha256": _sha(annotations),
        "images": [{"id": 1, "path": str(image), "sha256": _sha(image), "width": 10, "height": 10}],
        "categories": [1],
        "expected_image_count": 1,
        "expected_annotation_count": 0,
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


def _plan(source_digest: str, **changes) -> CandidatePlan:
    body = {
        "backend": "pytorch",
        "precision": "fp32",
        "device": "cpu",
        "provider": "torch",
        "resolution": (640, 640),
        "batch": 1,
        "source_model_digest": source_digest,
    }
    body.update(changes)
    return CandidatePlan(**body)


def _request(root: Path) -> tuple[SearchRequest, DatasetManifest]:
    reference = _target(root, "reference")
    data = _dataset(root)
    baseline = _plan(reference.source_sha256)
    reference_runtime = runtime_policy_contract(
        reference, backend="pytorch", provider="torch", device="cpu", precision="fp32", input_shape=(1, 3, 640, 640)
    )
    return SearchRequest(
        baseline=baseline,
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
        reference_runtime=reference_runtime,
        reference_source_model_digest=reference.source_sha256,
        build_template_digest="b" * 64,
        output=str(root / "search"),
    ), data


def _validation(root: Path, data: DatasetManifest, reference: ModelTarget, candidate: ModelTarget, item: CandidatePlan):
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
        def __init__(self, runtime):
            self.runtime = runtime

        def runtime_identity(self):
            return self.runtime

        def __call__(self, _path):
            return DetectionResult(metadata=self.runtime)

    original, models = (
        vision.load,
        iter(
            (
                Model(ref_runtime.model_dump(exclude={"source_sha256", "manifest_digest", "digest"})),
                Model(candidate_runtime.model_dump(exclude={"source_sha256", "manifest_digest", "digest"})),
            )
        ),
    )
    vision.load = lambda *_args, **_kwargs: next(models)
    output = root / f"validation-{item.candidate_id}"
    try:
        return validate_dataset(
            reference,
            candidate,
            data,
            ValidationPolicy(max_map_drop=0, reference_runtime=ref_runtime, candidate_runtime=candidate_runtime),
            output,
        ), output
    finally:
        vision.load = original


def _evaluation(
    root: Path, req: SearchRequest, data: DatasetManifest, item: CandidatePlan, *, exported: bool = False
) -> CandidateEvaluation:
    exported = item != req.baseline
    candidate = (
        req.reference_target
        if not exported
        else _target(root, f"artifact-{item.candidate_id}", "onnx" if item.backend == "onnxruntime" else "torch")
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
    record_path = root / f"build-{item.candidate_id}.json"
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
        "upstream_build_report_file_sha256": _sha(upstream_path) if upstream_path else None,
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
        artifact_sha256=_sha(Path(candidate.source)),
        manifest_path=candidate.manifest_path,
        manifest_sha256=_sha(Path(candidate.manifest_path)),
        build_report_path=str(record_path),
        build_report_sha256=_sha(record_path),
    )
    report, validation_root = _validation(root, data, req.reference_target, candidate, item)
    evidence = ValidationEvidence(
        report=report,
        root=str(validation_root),
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
        samples_ms=(1.0 if item.backend == "onnxruntime" else 2.0,) * 2,
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
        benchmark=benchmark,
    )


def _saved(root: Path, *, exported: bool = False):
    req, data = _request(root)
    report = optimize_search(req, lambda item: _evaluation(root, req, data, item, exported=exported))
    return report, Path(req.output) / "optimization_report.json"


@pytest.mark.parametrize("material", ("validation", "artifact", "manifest", "build"))
def test_load_rechecks_every_selected_material_boundary(tmp_path, material):
    report, path = _saved(tmp_path)
    selected = next(item for item in report.evaluations if item.plan.candidate_id == report.selected_candidate_id)
    if material == "validation":
        (Path(selected.validation.root) / "COMMITTED").unlink()
    elif material == "artifact":
        Path(selected.resolved.artifact_path).unlink()
    elif material == "manifest":
        Path(selected.resolved.manifest_path).write_text("task: detection\n")
    else:
        Path(selected.resolved.build_report_path).write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        load_search_report(path)


def test_load_rechecks_exported_upstream_build_report(tmp_path):
    report, path = _saved(tmp_path, exported=True)
    selected = next(item for item in report.evaluations if item.plan.candidate_id == report.selected_candidate_id)
    record = json.loads(Path(selected.resolved.build_report_path).read_text())
    upstream = Path(record["upstream_build_report_path"])
    upstream.write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        load_search_report(path)


def test_load_rejects_search_digest_tampering(tmp_path):
    _report, path = _saved(tmp_path)
    raw = json.loads(path.read_text())
    raw["digest"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest"):
        load_search_report(path)


def test_persist_and_optimize_never_clobber_existing_output(tmp_path):
    report, path = _saved(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _persist(report)
    assert path.read_bytes() == original
    with pytest.raises(FileExistsError):
        optimize_search(report.request, lambda item: _evaluation(tmp_path, report.request, _dataset(tmp_path), item))
    assert path.read_bytes() == original


def test_recovery_only_removes_matching_known_uncommitted_journals(tmp_path):
    postcommit = tmp_path / "postcommit"
    postcommit.mkdir()
    committed = postcommit / "optimization_report.json"
    committed.write_bytes(b"committed")
    committed_stage = postcommit / ".vision-optimization-postcommit.tmp"
    committed_stage.write_bytes(b"staged")
    postcommit_journal = postcommit / ".vision-optimization-postcommit.journal"
    postcommit_journal.write_text(json.dumps({"target": "optimization_report.json", "sha256": _sha(committed)}))
    uncommitted = tmp_path / "uncommitted"
    uncommitted.mkdir()
    stage = uncommitted / ".vision-optimization-uncommitted.tmp"
    stage.write_bytes(b"uncommitted")
    uncommitted_journal = uncommitted / ".vision-optimization-uncommitted.journal"
    uncommitted_journal.write_text(json.dumps({"target": "optimization_report.json", "sha256": "a" * 64}))
    foreign = uncommitted / ".vision-optimization-foreign.journal"
    foreign.write_text(json.dumps({"target": "other.json", "sha256": "a" * 64}))
    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    (mismatch_root / "optimization_report.json").write_bytes(b"different")
    mismatch = mismatch_root / ".vision-optimization-mismatch.journal"
    mismatch.write_text(json.dumps({"target": "optimization_report.json", "sha256": "b" * 64}))
    removed = (
        recover_orphan_searches(postcommit)
        + recover_orphan_searches(uncommitted)
        + recover_orphan_searches(mismatch_root)
    )
    assert not committed_stage.exists() and not stage.exists()
    assert not postcommit_journal.exists() and not uncommitted_journal.exists()
    assert foreign.exists() and mismatch.exists()
    assert len(removed) == 4


def test_cli_optimize_errors_have_nonzero_exit(tmp_path):
    runner = CliRunner()
    malformed = runner.invoke(
        app, ["optimize", "--search-config", str(tmp_path / "none.json"), "--evaluator-plugin", "not-a-plugin"]
    )
    assert malformed.exit_code != 0
    missing = runner.invoke(
        app, ["optimize", "--search-config", str(tmp_path / "none.json"), "--evaluator-plugin", "json:loads"]
    )
    assert missing.exit_code != 0 and '"accepted": false' in missing.stdout.lower()
    config = tmp_path / "search.json"
    req, _data = _request(tmp_path)
    config.write_text(req.model_dump_json())
    failing = runner.invoke(app, ["optimize", "--search-config", str(config), "--evaluator-plugin", "json:loads"])
    assert failing.exit_code != 0 and '"accepted": false' in failing.stdout.lower()


def test_baseline_failure_is_persisted_before_error(tmp_path):
    req, _data = _request(tmp_path)
    with pytest.raises(RuntimeError, match="baseline PyTorch fp32 failed"):
        optimize_search(req, lambda item: CandidateEvaluation(plan=item, stages=(), error="unavailable"))
    report = load_search_report(Path(req.output) / "optimization_report.json")
    assert report.selected_candidate_id is None
    assert all(item.stages[-1].stage is Stage.rejected for item in report.evaluations)
