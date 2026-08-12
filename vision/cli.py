from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import typer

from .api import load
from .backends import BACKENDS
from .benchmark import benchmark_model
from .buildkit import BuildPlanner
from .capsule import verify_capsule
from .contract import VisionContract
from .export import BuildRequest, build
from .manifest import VisionManifest
from .optimization import SearchRequest, optimize_search
from .plugins import DEFAULT_REGISTRY
from .validation import DatasetManifest, ModelTarget, ValidationPolicy, validate_dataset

app = typer.Typer(help="Honest detection portability tools")
contract_app = typer.Typer(help="Versioned semantic contract commands")
plugins_app = typer.Typer(help="BuildKit plugin commands")
app.add_typer(contract_app, name="contract")
app.add_typer(plugins_app, name="plugins")


def emit(x):
    typer.echo(json.dumps(x, sort_keys=True, default=str))


@app.command("support")
def support(path: str | None = typer.Option(None, help="Override the repository support matrix path")):
    """Print the evidence-bound backend support matrix."""
    from .support import support_report

    try:
        emit(support_report(path))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("doctor")
def doctor():
    """Report installed optional runtimes without claiming they are verified."""
    from .support import doctor_report

    emit(doctor_report())


@contract_app.command("validate")
def contract_validate(path: str):
    """Strictly validate a semantic contract and print its canonical digest."""
    contract = VisionContract.from_path(path)
    emit({"valid": True, "schema_version": contract.schema_version, "digest": contract.digest})


@plugins_app.command("list")
def plugins_list(discover: bool = typer.Option(False, help="Load installed entry-point plugins")):
    if discover:
        DEFAULT_REGISTRY.discover()

    def caps(plugin):
        value = plugin.capabilities
        return value.model_dump(mode="json") if hasattr(value, "model_dump") else sorted(value)

    emit(
        {
            "sources": [
                {"id": x.plugin_id, "version": x.plugin_version, "capabilities": caps(x)}
                for x in DEFAULT_REGISTRY.sources.values()
            ],
            "targets": [
                {"id": x.plugin_id, "version": x.plugin_version, "capabilities": caps(x)}
                for x in DEFAULT_REGISTRY.targets.values()
            ],
        }
    )


@app.command("build")
def buildkit_build(
    source_plugin: str,
    target_plugin: str,
    contract: str = typer.Option(..., help="VisionContract JSON/YAML"),
    source_digest: str = typer.Option(..., help="Verified SHA-256 of the source material"),
    precisions: str = "fp32",
    resolutions: str = "640x640",
    batches: str = "1",
    dynamics: str = "false",
    nms_owners: str = "source",
):
    """Plan builds only; target plugin execution is explicitly separate."""
    try:
        source, target = DEFAULT_REGISTRY.sources[source_plugin], DEFAULT_REGISTRY.targets[target_plugin]
        plan = BuildPlanner().plan(
            source,
            target,
            VisionContract.from_path(contract),
            source_digest,
            precisions=tuple(precisions.split(",")),
            resolutions=tuple(tuple(map(int, item.lower().split("x"))) for item in resolutions.split(",")),
            batches=tuple(map(int, batches.split(","))),
            dynamics=tuple(item.lower() == "true" for item in dynamics.split(",")),
            nms_owners=tuple(nms_owners.split(",")),
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit({"candidates": [x.__dict__ for x in plan.candidates], "rejections": plan.rejections})


@app.command("conform")
def conform_report(report: str):
    """Inspect a report; this is deliberately not a certification command."""
    from .conformance import CertificationReport

    value = CertificationReport.model_validate_json(Path(report).read_text())
    emit(value.model_dump(mode="json"))
    raise typer.Exit(1)


@app.command("certify")
def certify(capsule: str, expected_capsule_id: str = typer.Option(...)):
    """Certify only by fully verifying an already packaged capsule."""
    try:
        value = verify_capsule(capsule, expected_capsule_id=expected_capsule_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if value.mode != "certified" or value.certification_status != "PASS":
        emit({"self_attested_conformant": False, "mode": value.mode})
        raise typer.Exit(1)
    emit({"self_attested_conformant": True, "status": value.certification_status, "capsule_id": value.capsule_id})


@app.command()
def convert(
    source: str,
    output: str,
    family: str | None = typer.Option(None),
    manifest: str | None = typer.Option(None),
    target: str = typer.Option("onnxruntime"),
    precision: str = typer.Option("fp32"),
    calibration_manifest: str | None = typer.Option(None),
    dynamic: bool = typer.Option(False),
    min_shape: str | None = typer.Option(None),
    opt_shape: str | None = typer.Option(None),
    max_shape: str | None = typer.Option(None),
    input_name: str | None = typer.Option(None),
    calibration_cache_path: str | None = typer.Option(None),
    trt_timeout_s: int = typer.Option(300),
):
    def parse_shape(value: str | None) -> list[int] | None:
        return None if value is None else [int(part) for part in value.replace("x", ",").split(",") if part]

    shapes = [parse_shape(value) for value in (min_shape, opt_shape, max_shape)]
    profiles = []
    if any(shape is not None for shape in shapes):
        if not all(shape is not None for shape in shapes):
            raise typer.BadParameter("min/opt/max-shape must be supplied together")
        if not input_name:
            raise typer.BadParameter("--input-name is required with shape profiles")
        profiles = [{"input_name": input_name, "min_shape": shapes[0], "opt_shape": shapes[1], "max_shape": shapes[2]}]
    request = BuildRequest(
        source=source,
        output=output,
        target=target,
        precision=precision,
        family=family,
        manifest_path=manifest,
        calibration_manifest_path=calibration_manifest,
        dynamic=dynamic,
        profiles=profiles,
        calibration_cache_path=calibration_cache_path,
        trt_timeout_seconds=trt_timeout_s,
    )
    emit(build(request).model_dump(mode="json"))


@app.command()
def run(
    source: str,
    image: str,
    device: str = "auto",
    manifest: str | None = typer.Option(None),
    family: str | None = typer.Option(None),
    backend: str | None = typer.Option(None),
    provider: str | None = typer.Option(None),
    precision: str | None = typer.Option(None),
    expected_capsule_id: str | None = typer.Option(None),
):
    contract = VisionManifest.from_yaml(manifest) if manifest else None
    model = load(
        source,
        device,
        contract,
        family,
        expected_capsule_id,
        backend,
        provider,
        precision,
    )
    emit(model(image).to_dict())


@app.command()
def benchmark(
    source: str,
    image: str | None = None,
    repeats: int = 20,
    warmup: int = 3,
    device: str = "auto",
    manifest: str | None = typer.Option(None),
    family: str | None = typer.Option(None),
    backend: str | None = typer.Option(None),
    provider: str | None = typer.Option(None),
    precision: str | None = typer.Option(None),
    expected_capsule_id: str | None = typer.Option(None),
):
    if image is None:
        path = Path(source)
        if not path.is_dir() or not expected_capsule_id:
            raise typer.BadParameter(
                "benchmark requires IMAGE, or a capsule plus --expected-capsule-id to inspect its bound benchmarks"
            )
        index = verify_capsule(path, expected_capsule_id=expected_capsule_id)
        emit(
            {
                "capsule_id": index.capsule_id,
                "artifacts": [
                    {
                        "backend": artifact.backend,
                        "provider": artifact.provider,
                        "device": artifact.device,
                        "precision": artifact.precision,
                        "p50_ms": artifact.p50_ms,
                        "p95_ms": artifact.p95_ms,
                        "hardware_digest": artifact.hardware_digest,
                    }
                    for artifact in index.artifacts
                ],
            }
        )
        return
    contract = VisionManifest.from_yaml(manifest) if manifest else None
    model = load(
        source,
        device,
        contract,
        family,
        expected_capsule_id,
        backend,
        provider,
        precision,
    )
    emit(benchmark_model(model, image, repeats, warmup))


@app.command()
def optimize(
    source: str | None = typer.Argument(None),
    search_config: str | None = typer.Option(None, help="Advanced strict SearchRequest JSON"),
    evaluator_plugin: str | None = typer.Option(None, help="Advanced module:callable evaluator"),
    dataset: str | None = typer.Option(None, help="Signed DatasetManifest JSON"),
    output: str | None = typer.Option(None),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or a target such as rtx-4090"),
    objective: str = typer.Option("latency"),
    max_map_drop: float = typer.Option(0.2, help="mAP percentage points; 0.2 means 0.2 pp"),
    backends: str | None = typer.Option(None, help="Comma-separated backend list"),
    precisions: str = typer.Option("fp32,fp16,int8"),
    resolutions: str | None = typer.Option(None, help="Comma-separated HxW values"),
    batches: str = typer.Option("1"),
    dynamics: str = typer.Option("false"),
    repeats: int = typer.Option(20),
    warmup: int = typer.Option(3),
    benchmark_image: str | None = typer.Option(None),
    class_mapping: str | None = typer.Option(None, help="JSON model-class to dataset-category map"),
):
    """Optimize a model locally, or run the advanced signed evaluator API."""
    if source is not None:
        if search_config or evaluator_plugin:
            raise typer.BadParameter("SOURCE mode cannot be combined with advanced search options")

        def parse_resolutions(value: str | None):
            if value is None:
                return None
            parsed = []
            for item in value.split(","):
                height, width = item.lower().split("x", 1)
                parsed.append((int(height), int(width)))
            return tuple(parsed)

        from .auto_optimize import optimize_model

        native = load(source, device="auto" if device not in {"cpu", "cuda"} else device)
        optimized = optimize_model(
            native,
            target=device,
            objective=objective,
            max_map_drop=max_map_drop,
            dataset=dataset,
            output=output,
            class_mapping=json.loads(Path(class_mapping).read_text()) if class_mapping else None,
            backends=tuple(item for item in backends.split(",") if item) if backends else None,
            precisions=tuple(item for item in precisions.split(",") if item),
            resolutions=parse_resolutions(resolutions),
            batches=tuple(int(item) for item in batches.split(",") if item),
            dynamics=tuple(item.strip().lower() == "true" for item in dynamics.split(",") if item),
            repeats=repeats,
            warmup=warmup,
            benchmark_image=benchmark_image,
        )
        emit(
            {
                "capsule": str(optimized.capsule_path),
                "capsule_id": optimized.capsule_id,
                "selected_candidate_id": optimized.report.selected_candidate_id,
                "runtime": optimized.runtime_identity(),
            }
        )
        return
    if not search_config or not evaluator_plugin:
        raise typer.BadParameter("provide SOURCE, or both --search-config and --evaluator-plugin")
    module_name, separator, attr = evaluator_plugin.partition(":")
    if not separator or not module_name or not attr:
        raise typer.BadParameter("--evaluator-plugin must use module:callable")
    try:
        evaluator = getattr(importlib.import_module(module_name), attr)
        report = optimize_search(SearchRequest.model_validate_json(Path(search_config).read_text()), evaluator)
    except Exception as exc:
        emit({"accepted": False, "error": f"{type(exc).__name__}: {exc}"})
        raise typer.Exit(1) from exc
    emit(report.model_dump(mode="json"))
    if report.selected_candidate_id is None:
        raise typer.Exit(1)


@app.command("optimize-report")
def optimize_report(path: str):
    """Verify and print an immutable optimization report; never reinterprets dict evidence."""
    from .optimization import load_search_report

    report = load_search_report(path)
    emit(report.model_dump(mode="json"))


@app.command()
def validate(
    capsule: str,
    expected_capsule_id: str | None = typer.Option(None),
    allow_development: bool = typer.Option(False),
):
    emit(
        verify_capsule(
            capsule,
            expected_capsule_id=expected_capsule_id,
            allow_development=allow_development,
        ).model_dump()
    )


@app.command()
def parity(reference: str, candidate: str, iou_threshold: float = 0.5, score_delta: float = 0.05):
    """Compare two DetectionResult JSON documents produced by `vision run`."""
    from .types import Box, DetectionResult
    from .validation import compare_detections

    def read(path):
        raw = json.loads(Path(path).read_text())
        return DetectionResult([Box(*box) for box in raw["boxes"]], raw["scores"], raw["classes"])

    emit(
        compare_detections(
            read(reference),
            read(candidate),
            ValidationPolicy(max_map_drop=0, iou_threshold=iou_threshold, score_delta=score_delta),
        )
    )


@app.command("validate-dataset")
def validate_dataset_command(
    dataset: str,
    output_dir: str,
    max_map_drop: float,
    reference_source: str = typer.Option(...),
    reference_manifest: str = typer.Option(...),
    candidate_source: str = typer.Option(...),
    candidate_manifest: str = typer.Option(...),
    reference_device: str = typer.Option("auto"),
    candidate_device: str = typer.Option("auto"),
    iou_threshold: float = typer.Option(0.5),
    score_delta: float = typer.Option(0.05),
    high_confidence: float = typer.Option(0.5),
    min_coverage: float = typer.Option(1.0),
    max_class_mismatches: int = typer.Option(0),
    reference_class_mapping: str = typer.Option(..., help="JSON model-class to COCO-category map"),
    candidate_class_mapping: str = typer.Option(..., help="JSON model-class to COCO-category map"),
):
    """Run and validate two signed targets over a strict DatasetManifest JSON."""

    def target(source: str, manifest: str, device: str, class_mapping: str) -> ModelTarget:
        source_path, manifest_path = Path(source), Path(manifest)
        manifest_value = VisionManifest.from_yaml(manifest_path).model_dump(mode="json")
        return ModelTarget(
            source=str(source_path),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            manifest_path=str(manifest_path),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            manifest_digest=hashlib.sha256(
                json.dumps(manifest_value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            device=device,
            class_mapping=json.loads(class_mapping),
        )

    report = validate_dataset(
        target(reference_source, reference_manifest, reference_device, reference_class_mapping),
        target(candidate_source, candidate_manifest, candidate_device, candidate_class_mapping),
        DatasetManifest.model_validate_json(Path(dataset).read_text()),
        ValidationPolicy(
            max_map_drop=max_map_drop,
            iou_threshold=iou_threshold,
            score_delta=score_delta,
            high_confidence=high_confidence,
            min_coverage=min_coverage,
            max_class_mismatches=max_class_mismatches,
        ),
        output_dir,
    )
    emit(report.model_dump(mode="json"))
    if not report.passed:
        raise typer.Exit(1)


@app.command("validate-experiment-evidence")
def validate_experiment_evidence(report: str):
    """Strictly validate a canonical experiment evidence report."""
    from .experiment_evidence import load_experiment_report

    try:
        value = load_experiment_report(report)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit(
        {
            "schema_version": value.schema_version,
            "selected_configuration_id": value.selected_configuration_id,
            "status": value.status,
            "policy_digest": value.policy_digest,
            "report_digest": value.report_digest,
        }
    )


@app.command("verify-benchmark")
def verify_benchmark(path: str):
    """Verify benchmark samples, derived statistics, and provenance digests."""
    from .benchmark import verify_benchmark_report

    try:
        emit(verify_benchmark_report(json.loads(Path(path).read_text(encoding="utf-8"))))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def inspect(source: str):
    p = Path(source)
    if p.is_dir():
        emit(verify_capsule(p, allow_unvalidated=True).model_dump())
    else:
        emit(
            {
                "source": source,
                "suffix": p.suffix,
                "backends": {k: v.capabilities().__dict__ for k, v in BACKENDS.items()},
            }
        )


@app.command()
def serve(
    source: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    concurrency: int = 1,
    max_batch: int = 8,
    max_queue: int = 256,
    batch_window_ms: int = 5,
    max_upload_bytes: int = 10_000_000,
    max_image_pixels: int = 40_000_000,
    request_timeout_s: float | None = typer.Option(None, help="Per-request inference deadline in seconds"),
    api_key: str | None = typer.Option(None, envvar="VISION_API_KEY", help="Optional Bearer token for /v1 endpoints"),
    rate_limit_per_minute: int | None = typer.Option(None, help="Optional per-client /v1 request limit"),
    device: str = "auto",
    manifest: str | None = typer.Option(None),
    family: str | None = typer.Option(None),
    backend: str | None = typer.Option(None),
    provider: str | None = typer.Option(None),
    precision: str | None = typer.Option(None),
    expected_capsule_id: str | None = typer.Option(None),
):
    import uvicorn

    from .server import create_app

    contract = VisionManifest.from_yaml(manifest) if manifest else None
    model = load(
        source,
        device,
        contract,
        family,
        expected_capsule_id,
        backend,
        provider,
        precision,
    )
    application = create_app(
        model,
        concurrency=concurrency,
        max_batch=max_batch,
        max_queue=max_queue,
        batch_window_ms=batch_window_ms,
        max_upload_bytes=max_upload_bytes,
        max_image_pixels=max_image_pixels,
        request_timeout_s=request_timeout_s,
        api_key=api_key,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    uvicorn.run(application, host=host, port=port)
