"""Assemble a real two-candidate optimization search and release capsule.

This script intentionally consumes already-published validation/build evidence;
it does not accept booleans in place of those reports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import vision
from vision.benchmark import benchmark_model
from vision.capsule import publish_capsule, verify_capsule
from vision.export import BuildReport
from vision.optimization import (
    BenchmarkReport,
    BenchmarkRequest,
    CandidateEvaluation,
    CandidatePlan,
    ResolvedArtifactEvidence,
    ResolvedBuildRecord,
    RuntimeFingerprint,
    SearchPolicy,
    SearchRequest,
    SearchSpace,
    Stage,
    StageEvidence,
    ValidationEvidence,
    digest,
    expected_build_request_digest,
    optimize_search,
)
from vision.validation import ValidationPolicy, ValidationReport


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved(
    plan: CandidatePlan,
    target,
    build_template_digest: str,
    record_path: Path,
    upstream_path: Path | None,
) -> ResolvedArtifactEvidence:
    artifact = Path(target.source).absolute()
    manifest = Path(target.manifest_path).absolute()
    upstream = BuildReport.model_validate_json(upstream_path.read_text()) if upstream_path else None
    body = {
        "schema_version": 1,
        "kind": "exported" if upstream else "original",
        "candidate_id": plan.candidate_id,
        "plan_digest": plan.candidate_id,
        "source_model_digest": plan.source_model_digest,
        "build_template_digest": build_template_digest,
        "build_request_digest": expected_build_request_digest(plan, build_template_digest),
        "artifact_path": str(artifact),
        "artifact_digest": target.source_sha256,
        "manifest_path": str(manifest),
        "manifest_digest": target.manifest_digest,
        "target_digest": digest(target.model_dump(mode="json")),
        "backend": plan.backend,
        "precision": plan.precision,
        "device": plan.device,
        "provider": plan.provider,
        "resolution": plan.resolution,
        "batch": plan.batch,
        "dynamic": plan.dynamic,
        "profiles": plan.profiles,
        "upstream_build_report_path": str(upstream_path.absolute()) if upstream_path else None,
        "upstream_build_report_file_sha256": sha(upstream_path) if upstream_path else None,
        "upstream_build_report_self_digest": upstream.self_digest if upstream else None,
    }
    record = ResolvedBuildRecord(**(body | {"self_digest": digest(body)}))
    if record_path.exists():
        raise FileExistsError(record_path)
    record_path.write_text(record.model_dump_json(indent=2))
    return ResolvedArtifactEvidence(
        candidate_id=plan.candidate_id,
        artifact_digest=target.source_sha256,
        build_request_digest=record.build_request_digest,
        manifest_digest=target.manifest_digest,
        target_digest=record.target_digest,
        build_report_digest=record.self_digest,
        artifact_path=str(artifact),
        artifact_sha256=target.source_sha256,
        manifest_path=str(manifest),
        manifest_sha256=target.manifest_sha256,
        build_report_path=str(record_path.absolute()),
        build_report_sha256=sha(record_path),
    )


def validation_evidence(
    report: ValidationReport,
    root: Path,
    material: ResolvedArtifactEvidence,
) -> ValidationEvidence:
    binding = digest(
        {
            "report_digest": report.digest,
            "resolved": material.model_dump(mode="json"),
            "policy_digest": report.policy_digest,
        }
    )
    return ValidationEvidence(
        report=report,
        root=str(root.absolute()),
        expected_policy_digest=report.policy_digest,
        resolved=material,
        binding_digest=binding,
    )


def benchmark_evidence(plan: CandidatePlan, material: ResolvedArtifactEvidence, image: Path, repeats: int):
    model = vision.load(
        material.artifact_path,
        device=plan.device,
        manifest=vision.VisionManifest.from_yaml(material.manifest_path),
    )
    measured = benchmark_model(model, image, repeats=repeats, warmup=3)
    runtime = model.runtime_identity()
    try:
        runtime_version = importlib.metadata.version(
            {"pytorch": "torch", "openvino": "openvino", "onnxruntime": "onnxruntime"}[plan.backend]
        )
    except (KeyError, importlib.metadata.PackageNotFoundError):
        runtime_version = "unknown"
    fingerprint = RuntimeFingerprint(
        hardware_digest=measured["hardware_digest"],
        harness_digest=measured["harness_digest"],
        workload_digest=measured["workload_digest"],
        input_digest=measured["workload"]["input"]["sha256"],
        artifact_digest=material.artifact_digest,
        backend=runtime["backend"],
        precision=runtime["precision"],
        runtime_version=runtime_version,
        device=runtime["device"],
        provider=runtime["provider"],
        threads=os.cpu_count() or 1,
        mode="execution",
    )
    request = BenchmarkRequest(
        fingerprint=fingerprint,
        resolved=material,
        resolution=plan.resolution,
        dynamic=plan.dynamic,
        profiles=plan.profiles,
        warmup=3,
        repeats=repeats,
        batch=plan.batch,
    )
    artifact = Path(material.artifact_path)
    size = artifact.stat().st_size
    if artifact.suffix == ".xml":
        size += artifact.with_suffix(".bin").stat().st_size
    return BenchmarkReport(
        request=request,
        samples_ms=tuple(measured["samples_ms"]),
        artifact_size_bytes=size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-validation", type=Path, required=True)
    parser.add_argument("--candidate-validation", type=Path, required=True)
    parser.add_argument("--candidate-build-report", type=Path, required=True)
    parser.add_argument("--benchmark-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    baseline_root, candidate_root = args.baseline_validation, args.candidate_validation
    baseline_report = ValidationReport.model_validate_json((baseline_root / "validation_report.json").read_text())
    candidate_report = ValidationReport.model_validate_json((candidate_root / "validation_report.json").read_text())
    if baseline_report.dataset.digest != candidate_report.dataset.digest:
        raise ValueError("validation reports use different datasets")
    source_digest = baseline_report.reference.target.source_sha256
    baseline = CandidatePlan(
        backend="pytorch",
        precision="fp32",
        device=baseline_report.candidate.runtime.device,
        provider=baseline_report.candidate.runtime.provider,
        resolution=baseline_report.candidate.runtime.input_shape[-2:],
        batch=baseline_report.candidate.runtime.input_shape[0],
        source_model_digest=source_digest,
    )
    candidate = CandidatePlan(
        backend=candidate_report.candidate.runtime.backend,
        precision=candidate_report.candidate.runtime.precision,
        device=candidate_report.candidate.runtime.device,
        provider=candidate_report.candidate.runtime.provider,
        resolution=candidate_report.candidate.runtime.input_shape[-2:],
        batch=candidate_report.candidate.runtime.input_shape[0],
        source_model_digest=source_digest,
    )
    build_template_digest = digest(
        {
            "schema": 1,
            "pipeline": "pytorch-onnx-compiled",
            "quality": {"max_map_drop": 0.002, "score_delta": 0.1, "min_coverage": 0.98},
        }
    )
    search = SearchRequest(
        baseline=baseline,
        space=SearchSpace(
            backends=(candidate.backend,),
            precisions=(candidate.precision,),
            devices=(candidate.device,),
            providers=(candidate.provider,),
            resolutions=(candidate.resolution,),
            batches=(candidate.batch,),
            dynamics=(candidate.dynamic,),
            max_candidates=2,
        ),
        policy=SearchPolicy(max_map_drop=0.002),
        validation_policy=ValidationPolicy(max_map_drop=0.002, score_delta=0.1, min_coverage=0.98),
        dataset_digest=baseline_report.dataset.digest,
        reference_target=baseline_report.reference.target,
        reference_runtime=baseline_report.reference.runtime,
        reference_source_model_digest=source_digest,
        build_template_digest=build_template_digest,
        output=str(args.output.absolute()),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_material = resolved(
        baseline,
        baseline_report.candidate.target,
        build_template_digest,
        args.output / "baseline-resolved-build.json",
        None,
    )
    candidate_material = resolved(
        candidate,
        candidate_report.candidate.target,
        build_template_digest,
        args.output / "candidate-resolved-build.json",
        args.candidate_build_report,
    )
    progress = tuple(StageEvidence(stage=stage) for stage in list(Stage)[:6])
    evaluations = {
        baseline.candidate_id: CandidateEvaluation(
            plan=baseline,
            resolved=baseline_material,
            stages=progress,
            validation=validation_evidence(baseline_report, baseline_root, baseline_material),
            benchmark=benchmark_evidence(baseline, baseline_material, args.benchmark_image, args.repeats),
        ),
        candidate.candidate_id: CandidateEvaluation(
            plan=candidate,
            resolved=candidate_material,
            stages=progress,
            validation=validation_evidence(candidate_report, candidate_root, candidate_material),
            benchmark=benchmark_evidence(candidate, candidate_material, args.benchmark_image, args.repeats),
        ),
    }
    report = optimize_search(search, lambda plan: evaluations[plan.candidate_id])
    capsule = publish_capsule(args.capsule, report)
    expected_capsule_id = json.loads((capsule / "capsule.json").read_text())["capsule_id"]
    index = verify_capsule(capsule, expected_capsule_id=expected_capsule_id)
    loaded = vision.load(
        str(capsule),
        expected_capsule_id=index.capsule_id,
        device="auto",
    )
    result = loaded(args.benchmark_image)
    print(
        json.dumps(
            {
                "selected_candidate_id": report.selected_candidate_id,
                "capsule": str(capsule),
                "capsule_id": index.capsule_id,
                "backend": loaded.runtime_identity(),
                "detections": len(result.boxes),
                "benchmarks": {
                    key: {
                        "p50_ms": value.benchmark.p50_ms,
                        "p95_ms": value.benchmark.p95_ms,
                    }
                    for key, value in evaluations.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
