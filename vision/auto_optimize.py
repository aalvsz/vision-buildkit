from __future__ import annotations

import hashlib
import importlib.metadata
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import benchmark_model
from .capsule import publish_capsule, verify_capsule
from .export import BuildRequest, build
from .manifest import ArtifactSpec, CalibrationManifest, VisionManifest
from .optimization import (
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
    ShapeProfile,
    Stage,
    StageEvidence,
    ValidationEvidence,
    digest,
    expected_build_request_digest,
    optimize_search,
)
from .validation import (
    DatasetManifest,
    ModelTarget,
    ValidationPolicy,
    _digest,
    runtime_policy_contract,
    validate_dataset,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _processing_digests(manifest: VisionManifest, class_mapping: dict[str, int]) -> tuple[str, str]:
    return (
        _digest(
            {
                "input": manifest.input.model_dump(mode="json"),
                "preprocessing": manifest.preprocessing.model_dump(mode="json"),
            }
        ),
        _digest(
            {
                "output": manifest.output.model_dump(mode="json"),
                "decoder": manifest.decoder.model_dump(mode="json"),
                "postprocessing": manifest.postprocessing,
                "class_mapping": class_mapping,
            }
        ),
    )


def _map_drop_fraction(value: float) -> float:
    if value < 0:
        raise ValueError("max_map_drop must be non-negative")
    # The ergonomic API uses common mAP percentage points: 0.2 means 0.2 pp.
    # Advanced SearchPolicy callers continue to use normalized [0, 1] units.
    normalized = value / 100 if value > 0.01 else value
    if normalized > 1:
        raise ValueError("max_map_drop is outside the supported range")
    return normalized


def _default_class_mapping(manifest: VisionManifest, categories: list[int]) -> dict[str, int]:
    if manifest.decoder.kind == "rfdetr":
        return {str(category): category for category in categories}
    return {str(index): category for index, category in enumerate(categories)}


def _runtime_package(backend: str) -> str:
    return {"pytorch": "torch", "onnxruntime": "onnxruntime", "openvino": "openvino", "tensorrt": "tensorrt"}.get(
        backend, backend
    )


def _package_version(backend: str) -> str:
    try:
        return importlib.metadata.version(_runtime_package(backend))
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _target(
    source: Path,
    manifest_path: Path,
    device: str,
    class_mapping: dict[str, int],
) -> ModelTarget:
    parsed = VisionManifest.from_yaml(manifest_path)
    return ModelTarget(
        source=str(source.absolute()),
        source_sha256=_sha(source),
        manifest_path=str(manifest_path.absolute()),
        manifest_sha256=_sha(manifest_path),
        manifest_digest=_digest(parsed.model_dump(mode="json")),
        device=device,
        class_mapping=class_mapping,
    )


def _write_resolved(
    plan: CandidatePlan,
    target: ModelTarget,
    build_template_digest: str,
    path: Path,
    upstream_path: Path | None,
) -> ResolvedArtifactEvidence:
    upstream = None
    if upstream_path is not None:
        from .export import BuildReport

        upstream = BuildReport.model_validate_json(upstream_path.read_text())
    body = {
        "schema_version": 1,
        "kind": "exported" if upstream else "original",
        "candidate_id": plan.candidate_id,
        "plan_digest": plan.candidate_id,
        "source_model_digest": plan.source_model_digest,
        "build_template_digest": build_template_digest,
        "build_request_digest": expected_build_request_digest(plan, build_template_digest),
        "artifact_path": target.source,
        "artifact_digest": target.source_sha256,
        "manifest_path": target.manifest_path,
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
        "upstream_build_report_file_sha256": _sha(upstream_path) if upstream_path else None,
        "upstream_build_report_self_digest": upstream.self_digest if upstream else None,
    }
    record = ResolvedBuildRecord(**(body | {"self_digest": digest(body)}))
    path.write_text(record.model_dump_json(indent=2))
    return ResolvedArtifactEvidence(
        candidate_id=plan.candidate_id,
        artifact_digest=target.source_sha256,
        build_request_digest=record.build_request_digest,
        manifest_digest=target.manifest_digest,
        target_digest=record.target_digest,
        build_report_digest=record.self_digest,
        artifact_path=target.source,
        artifact_sha256=target.source_sha256,
        manifest_path=target.manifest_path,
        manifest_sha256=target.manifest_sha256,
        build_report_path=str(path.absolute()),
        build_report_sha256=_sha(path),
    )


def _validation_evidence(report, root: Path, material: ResolvedArtifactEvidence) -> ValidationEvidence:
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


def _benchmark_evidence(
    plan: CandidatePlan,
    material: ResolvedArtifactEvidence,
    image: Path,
    repeats: int,
    warmup: int,
):
    from .api import load

    model = load(
        material.artifact_path,
        device=plan.device,
        manifest=VisionManifest.from_yaml(material.manifest_path),
    )
    measured = benchmark_model(model, image, repeats=repeats, warmup=warmup, batch=plan.batch)
    runtime = model.runtime_identity()
    fingerprint = RuntimeFingerprint(
        hardware_digest=measured["hardware_digest"],
        harness_digest=measured["harness_digest"],
        workload_digest=measured["workload_digest"],
        input_digest=measured["workload"]["input"]["sha256"],
        artifact_digest=material.artifact_digest,
        backend=runtime["backend"],
        precision=runtime["precision"],
        runtime_version=_package_version(plan.backend),
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
        warmup=warmup,
        repeats=repeats,
        batch=plan.batch,
    )
    artifact = Path(material.artifact_path)
    size = artifact.stat().st_size
    if artifact.suffix.lower() == ".xml":
        size += artifact.with_suffix(".bin").stat().st_size
    return BenchmarkReport(request=request, samples_ms=tuple(measured["samples_ms"]), artifact_size_bytes=size)


def _derived_dataset(
    dataset: DatasetManifest,
    preprocessing: set[str],
    postprocessing: set[str],
    path: Path,
) -> DatasetManifest:
    body = dataset.model_dump(exclude={"digest"}, mode="json")
    body["allowed_preprocessing_digests"] = sorted(preprocessing - {dataset.preprocessing_digest})
    body["allowed_postprocessing_digests"] = sorted(postprocessing - {dataset.postprocessing_digest})
    if not body["allowed_preprocessing_digests"]:
        body.pop("allowed_preprocessing_digests")
    if not body["allowed_postprocessing_digests"]:
        body.pop("allowed_postprocessing_digests")
    derived = DatasetManifest.model_validate(body | {"digest": _digest(body)})
    path.write_text(derived.model_dump_json(indent=2))
    return derived


def _calibration(
    manifest: VisionManifest,
    dataset: DatasetManifest,
    output: Path,
    sample_limit: int = 8,
) -> Path:
    from PIL import Image

    from .adapters.processor import DetectionProcessor

    output.mkdir()
    processor = DetectionProcessor(manifest)
    batch = int(manifest.input.shape[0])
    images = dataset.images[:sample_limit]
    tensors = []
    for item in images:
        with Image.open(item.path) as image:
            tensors.append(processor(image.convert("RGB"))[0])
    groups = []
    for start in range(0, len(tensors), batch):
        group = tensors[start : start + batch]
        group += [group[-1]] * (batch - len(group))
        groups.append(np.concatenate(group, axis=0).astype(np.float32, copy=False))
    samples = []
    for index, array in enumerate(groups):
        path = output / f"{index:04d}.npy"
        np.save(path, array, allow_pickle=False)
        samples.append({"path": str(path.absolute()), "sha256": _sha(path)})
    preprocessing_digest, _ = _processing_digests(manifest, {})
    body = {
        "samples": samples,
        "sample_count": len(samples),
        "preprocessing_digest": preprocessing_digest,
        "input_name": manifest.input.input_name,
        "shape": list(groups[0].shape),
        "dtype": "float32",
    }
    value = CalibrationManifest.model_validate(body | {"digest": _digest(body)})
    path = output / "calibration-manifest.json"
    path.write_text(value.model_dump_json(indent=2))
    return path


@dataclass
class OptimizedVisionModel:
    model: Any
    report: Any
    capsule_path: Path
    capsule_id: str

    def __call__(self, image):
        return self.model(image)

    def predict_batch(self, images):
        return self.model.predict_batch(images)

    def runtime_identity(self):
        return self.model.runtime_identity()


def _candidate_backends(target: str, requested: tuple[str, ...] | None) -> list[tuple[str, str, str]]:
    if requested:
        names = requested
    elif target in {"cuda", "rtx-4090"}:
        names = ("onnxruntime", "tensorrt")
    else:
        names = ("onnxruntime", "openvino")
    result = []
    for backend in names:
        if backend == "onnxruntime":
            cuda = target in {"cuda", "rtx-4090"}
            provider = "CUDAExecutionProvider" if cuda else "CPUExecutionProvider"
            result.append((backend, provider, "cuda" if cuda else "cpu"))
        elif backend == "openvino":
            result.append((backend, "CPU", "cpu"))
        elif backend == "tensorrt":
            result.append((backend, "TensorRTExecutionProvider", "cuda"))
        else:
            raise ValueError(f"unsupported automatic backend {backend!r}")
    return result


def optimize_model(
    model,
    *,
    target: str = "auto",
    objective: str = "latency",
    max_map_drop: float = 0.2,
    dataset: str | Path | DatasetManifest | None = None,
    output: str | Path | None = None,
    class_mapping: dict[str, int] | None = None,
    backends: tuple[str, ...] | None = None,
    precisions: tuple[str, ...] = ("fp32", "fp16", "int8"),
    resolutions: tuple[tuple[int, int], ...] | None = None,
    batches: tuple[int, ...] = (1,),
    dynamics: tuple[bool, ...] = (False,),
    repeats: int = 20,
    warmup: int = 3,
    benchmark_image: str | Path | None = None,
) -> OptimizedVisionModel:
    if objective != "latency":
        raise ValueError("the MVP automatic optimizer currently supports objective='latency'")
    if dataset is None:
        configured = os.environ.get("VISION_DATASET_MANIFEST")
        if not configured:
            raise ValueError("automatic optimization requires a signed DatasetManifest or VISION_DATASET_MANIFEST")
        dataset = configured
    data = (
        dataset
        if isinstance(dataset, DatasetManifest)
        else DatasetManifest.model_validate_json(Path(dataset).read_text())
    )
    root = (
        Path(output).absolute()
        if output is not None
        else (Path.cwd() / "vision-runs" / f"optimize-{uuid.uuid4().hex[:12]}").absolute()
    )
    if root.exists():
        raise FileExistsError(f"optimization output already exists: {root}")
    root.mkdir(parents=True)
    materials = root / "materials"
    materials.mkdir()

    original_manifest = model.manifest
    source = Path(original_manifest.source)
    if not source.is_file():
        raise ValueError("automatic optimization requires a local PyTorch checkpoint source")
    source = source.absolute()
    if source.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
        raise ValueError("automatic optimization baseline must be a PyTorch checkpoint")
    baseline_device = "cpu" if target == "cpu" else "cuda" if target in {"cuda", "rtx-4090"} else model.device
    reference_manifest = original_manifest.model_copy(update={"source": str(source)})
    reference_manifest_path = materials / "reference-manifest.yaml"
    reference_manifest.to_yaml(reference_manifest_path)
    mapping = class_mapping or _default_class_mapping(reference_manifest, data.categories)
    reference_target = _target(source, reference_manifest_path, baseline_device, mapping)
    reference_target.assert_dataset_contract(data)

    from .api import load

    reference_model = load(str(source), device=baseline_device, manifest=reference_manifest)
    reference_runtime = runtime_policy_contract(reference_target, **reference_model.runtime_identity())
    source_digest = reference_target.source_sha256
    baseline = CandidatePlan(
        backend="pytorch",
        precision="fp32",
        device=reference_runtime.device,
        provider=reference_runtime.provider,
        resolution=reference_runtime.input_shape[-2:],
        batch=reference_runtime.input_shape[0],
        source_model_digest=source_digest,
    )
    selected_resolutions = resolutions or (baseline.resolution,)
    plans = []
    for backend, provider, device in _candidate_backends(target, backends):
        for precision in precisions:
            for resolution in selected_resolutions:
                for batch in batches:
                    for dynamic in dynamics:
                        profiles = ()
                        if dynamic:
                            profiles = (
                                ShapeProfile(
                                    input_name=reference_manifest.input.input_name,
                                    min_shape=(1, 3, *resolution),
                                    opt_shape=(batch, 3, *resolution),
                                    max_shape=(max(batch, 4), 3, *resolution),
                                ),
                            )
                        plans.append(
                            CandidatePlan(
                                backend=backend,
                                precision=precision,
                                device=device,
                                provider=provider,
                                resolution=resolution,
                                batch=batch,
                                dynamic=dynamic,
                                profiles=profiles,
                                source_model_digest=source_digest,
                            )
                        )
    manifest_templates: dict[str, tuple[VisionManifest, Path]] = {}
    pre_digests, post_digests = set(), set()
    baseline_pre, baseline_post = _processing_digests(reference_manifest, mapping)
    pre_digests.add(baseline_pre)
    post_digests.add(baseline_post)
    for plan in plans:
        source_contract = reference_manifest.model_copy(
            update={"input": reference_manifest.input.model_copy(update={"shape": [plan.batch, 3, *plan.resolution]})}
        )
        source_manifest_path = materials / f"{plan.candidate_id}-source.yaml"
        source_contract.to_yaml(source_manifest_path)
        candidate_contract = source_contract.model_copy(update={"family": "onnx", "source": "pending"})
        manifest_templates[plan.candidate_id] = (candidate_contract, source_manifest_path)
        pre, post = _processing_digests(candidate_contract, mapping)
        pre_digests.add(pre)
        post_digests.add(post)
    search_dataset = _derived_dataset(data, pre_digests, post_digests, materials / "dataset-manifest.json")

    build_template_digest = digest(
        {
            "schema": 1,
            "source": source_digest,
            "target": target,
            "objective": objective,
            "plans": [plan.model_dump(mode="json") for plan in plans],
            "dataset": search_dataset.digest,
        }
    )
    map_drop = _map_drop_fraction(max_map_drop)
    search = SearchRequest(
        baseline=baseline,
        space=SearchSpace(exact_plans=tuple(plans), max_candidates=len(plans) + 1),
        policy=SearchPolicy(
            max_map_drop=map_drop,
            normalized_workload_semantics=any(
                plan.resolution != baseline.resolution or plan.batch != baseline.batch or plan.dynamic for plan in plans
            ),
            allow_shape_search_comparison=any(
                plan.resolution != baseline.resolution or plan.batch != baseline.batch or plan.dynamic for plan in plans
            ),
        ),
        validation_policy=ValidationPolicy(max_map_drop=map_drop, score_delta=0.1, min_coverage=0.98),
        dataset_digest=search_dataset.digest,
        reference_target=reference_target,
        reference_runtime=reference_runtime,
        reference_source_model_digest=source_digest,
        build_template_digest=build_template_digest,
        output=str(root / "search"),
    )
    if benchmark_image is None:
        benchmark_path = Path(search_dataset.images[0].path)
    else:
        benchmark_path = Path(benchmark_image)
    (root / "validation").mkdir()
    progress = tuple(StageEvidence(stage=stage) for stage in list(Stage)[:6])

    def evaluate(plan: CandidatePlan) -> CandidateEvaluation:
        if plan == baseline:
            target_value = reference_target
            upstream_path = None
        else:
            contract, source_manifest_path = manifest_templates[plan.candidate_id]
            suffix = {"onnxruntime": ".onnx", "openvino": ".xml", "tensorrt": ".engine"}[plan.backend]
            artifact_path = (materials / f"{plan.candidate_id}{suffix}").absolute()
            calibration_path = None
            cache_path = None
            if plan.precision == "int8":
                if plan.backend == "tensorrt":
                    raise RuntimeError("automatic TensorRT INT8 requires an externally generated calibration cache")
                calibration_path = _calibration(
                    contract,
                    search_dataset,
                    materials / f"{plan.candidate_id}-calibration",
                )
            profiles = [
                {
                    "input_name": profile.input_name,
                    "min_shape": list(profile.min_shape),
                    "opt_shape": list(profile.opt_shape),
                    "max_shape": list(profile.max_shape),
                }
                for profile in plan.profiles
            ]
            built = build(
                BuildRequest(
                    source=str(source),
                    output=str(artifact_path),
                    target=plan.backend,
                    precision=plan.precision,
                    family=reference_manifest.family,
                    manifest_path=str(source_manifest_path.absolute()),
                    calibration_manifest_path=str(calibration_path) if calibration_path else None,
                    calibration_cache_path=cache_path,
                    dynamic=plan.dynamic,
                    profiles=profiles,
                )
            )
            artifact = ArtifactSpec(
                path=str(artifact_path),
                backend=plan.backend,
                precision=plan.precision,
                validated=True,
                checksum=built.output_digest,
                compatible={"device": plan.device, "provider": plan.provider},
            )
            contract = contract.model_copy(update={"source": str(artifact_path), "artifacts": [artifact]})
            candidate_manifest_path = materials / f"{plan.candidate_id}-manifest.yaml"
            contract.to_yaml(candidate_manifest_path)
            target_value = _target(artifact_path, candidate_manifest_path, plan.device, mapping)
            upstream_path = Path(f"{artifact_path}.build-report.json")
        record_path = materials / f"{plan.candidate_id}-resolved-build.json"
        material = _write_resolved(plan, target_value, build_template_digest, record_path, upstream_path)
        candidate_model = load(
            target_value.source,
            device=target_value.device,
            manifest=target_value.manifest(),
        )
        candidate_runtime = runtime_policy_contract(target_value, **candidate_model.runtime_identity())
        policy = search.validation_policy.model_copy(
            update={"reference_runtime": reference_runtime, "candidate_runtime": candidate_runtime}
        )
        validation_root = root / "validation" / plan.candidate_id
        report = validate_dataset(reference_target, target_value, search_dataset, policy, validation_root)
        evidence = _validation_evidence(report, validation_root, material)
        benchmark = _benchmark_evidence(plan, material, benchmark_path, repeats, warmup)
        return CandidateEvaluation(
            plan=plan,
            resolved=material,
            stages=progress,
            validation=evidence,
            benchmark=benchmark,
        )

    report = optimize_search(search, evaluate)
    if report.selected_candidate_id is None:
        raise RuntimeError("no candidate passed the quality and benchmark gates")
    capsule_path = publish_capsule(root / "model.vcap", report)
    index = verify_capsule(capsule_path)
    selected_model = load(str(capsule_path), device="auto", expected_capsule_id=index.capsule_id)
    return OptimizedVisionModel(selected_model, report, capsule_path, index.capsule_id)


__all__ = ["OptimizedVisionModel", "optimize_model"]
