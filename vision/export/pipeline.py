from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..manifest import CalibrationManifest, VisionManifest
from .quantization import calibration_arrays, quantize_ort_static


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


class ShapeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_name: str
    min_shape: list[int]
    opt_shape: list[int]
    max_shape: list[int]

    @model_validator(mode="after")
    def valid_ranks_and_order(self):
        if not (len(self.min_shape) == len(self.opt_shape) == len(self.max_shape)):
            raise ValueError("profile shapes must have the same rank")
        if any(
            a <= 0 or a > b or b > c for a, b, c in zip(self.min_shape, self.opt_shape, self.max_shape, strict=True)
        ):
            raise ValueError("profile must satisfy positive min <= opt <= max dimensions")
        return self


class Tolerances(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_abs: float = 1e-2
    mean_abs: float = 1e-3
    max_rel: float = 0.1
    min_cosine: float = 0.99


class BuildRequest(BaseModel):
    """The only public artifact-build request; unknown knobs fail closed."""

    model_config = ConfigDict(extra="forbid")
    source: str
    output: str
    target: Literal["onnxruntime", "openvino", "tensorrt"] = "onnxruntime"
    precision: Literal["fp32", "fp16", "int8"] = "fp32"
    family: Literal["yolo", "yolo26", "rfdetr", "rtdetr", "onnx", "torch"] | None = None
    manifest_path: str | None = None
    calibration_manifest_path: str | None = None
    dynamic: bool = False
    profiles: list[ShapeProfile] = Field(default_factory=list)
    tolerances: Tolerances = Field(default_factory=Tolerances)
    trt_timeout_seconds: int = Field(default=300, gt=0, le=3600)
    calibration_cache_path: str | None = None

    @model_validator(mode="after")
    def safe_paths(self):
        source, output = Path(self.source), Path(self.output)
        if source.is_symlink() or not source.is_file():
            raise ValueError("source must be an existing regular file")
        if source.resolve() == output.resolve():
            raise ValueError("source and output must differ")
        if output.is_symlink() or output.exists():
            raise ValueError("output must not exist or be a symlink (stale artifact refused)")
        if self.target == "tensorrt" and self.precision == "int8" and not self.calibration_cache_path:
            raise ValueError("TensorRT INT8 requires calibration_cache_path")
        return self


class BuildReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    intermediate: str | None
    output: str
    target: str
    requested_precision: str
    effective_precision: str
    source_digest: str
    intermediate_digest: str
    output_digest: str
    artifact_digests: dict[str, str] = Field(default_factory=dict)
    calibration_digest: str | None = None
    dynamic: bool
    profiles: list[dict[str, Any]] = Field(default_factory=list)
    model: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, str] = Field(default_factory=dict)
    target_fingerprint: dict[str, Any] = Field(default_factory=dict)
    command: list[str] | None = None
    logs_digest: str | None = None
    validation: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    self_digest: str = ""

    def finalized(self) -> BuildReport:
        raw = self.model_dump(exclude={"self_digest"}, mode="json")
        return self.model_copy(
            update={
                "self_digest": hashlib.sha256(
                    json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
            }
        )


def _assert_fresh(*paths: Path) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite concurrently-created artifact: {path}")


def _fsync_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def recover_orphan_builds(parent: Path) -> None:
    """Remove only digest-matching artifacts recorded by an interrupted publication."""
    for journal in parent.glob(".vision-build-pending-*.json"):
        try:
            data = json.loads(journal.read_text())
            report = Path(data["report"])
            if report.exists():
                journal.unlink(missing_ok=True)
                continue
            for raw, expected in data["artifacts"].items():
                path = Path(raw)
                if path.parent == parent and path.is_file() and not path.is_symlink() and digest(path) == expected:
                    path.unlink()
            journal.unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # A malformed journal is forensic evidence; fail closed rather than broad deletion.
            continue


def _publish_no_clobber(staged: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to overwrite concurrently-created artifact: {destination}")
    try:
        os.link(staged, destination)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite concurrently-created artifact: {destination}") from exc
    staged.unlink()


def _versions() -> dict[str, str]:
    result = {"python": os.sys.version.split()[0], "platform": platform.platform()}
    for package in ("onnx", "onnxruntime", "onnxconverter-common", "openvino", "nncf", "tensorrt"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _onnx_metadata(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)

    def spec(value):
        tensor = value.type.tensor_type
        return {
            "name": value.name,
            "shape": [d.dim_param or d.dim_value for d in tensor.shape.dim],
            "dtype": tensor.elem_type,
        }

    return {
        "opset": [f"{item.domain}:{item.version}" for item in model.opset_import],
        "inputs": [spec(v) for v in model.graph.input],
        "outputs": [spec(v) for v in model.graph.output],
        "custom_domains": sorted({n.domain for n in model.graph.node if n.domain}),
        "nodes": [n.op_type for n in model.graph.node],
    }


def _effective_onnx_precision(path: Path) -> str:
    metadata = _onnx_metadata(path)
    if "QuantizeLinear" in metadata["nodes"] and "DequantizeLinear" in metadata["nodes"]:
        return "int8"
    import onnx

    model = onnx.load(path)
    if any(t.data_type == onnx.TensorProto.FLOAT16 for t in model.graph.initializer):
        return "fp16"
    return "fp32"


def normalize_source(request: BuildRequest, tempdir: Path) -> Path:
    source = Path(request.source)
    if source.suffix.lower() == ".onnx":
        destination = tempdir / "source.onnx"
        shutil.copyfile(source, destination)
        _onnx_metadata(destination)
        return destination
    manifest = VisionManifest.from_yaml(request.manifest_path) if request.manifest_path else None
    from ..api import load

    isolated_source = tempdir / source.name
    try:
        os.link(source, isolated_source)
    except OSError:
        shutil.copyfile(source, isolated_source)
    adapter = load(str(isolated_source), manifest=manifest, family=request.family)
    destination = tempdir / "source.onnx"
    result = adapter.export_onnx(str(destination), dynamic=request.dynamic)
    exported = Path(result)
    if exported.resolve() != destination.resolve():
        shutil.copyfile(exported, destination)
    _onnx_metadata(destination)
    return destination


def _samples(metadata: dict[str, Any], calibration: CalibrationManifest | None) -> list[dict[str, np.ndarray]]:
    if len(metadata["inputs"]) != 1:
        raise ValueError("MVP export validation supports exactly one model input")
    if calibration:
        if metadata["inputs"][0]["name"] != calibration.input_name:
            raise ValueError("calibration input_name does not match model")
        expected = {1: "float32", 10: "float16"}.get(metadata["inputs"][0]["dtype"])
        if expected != calibration.dtype:
            raise ValueError("calibration sample dtype does not match model input dtype")
        return [{calibration.input_name: value} for value in calibration_arrays(calibration)]
    item = metadata["inputs"][0]
    shape = [int(v) if isinstance(v, int) and v > 0 else 1 for v in item["shape"]]
    dtype = {1: np.float32, 10: np.float16}.get(item["dtype"])
    if dtype is None:
        raise ValueError("MVP export validation supports float32/float16 model inputs")
    # An all-zero detector input creates large tied top-k sets in DETR heads,
    # where two numerically equivalent runtimes may return a different valid
    # order.  A fixed, non-degenerate sample makes the smoke gate reproducible;
    # publishable quality still requires the separate dataset validation gate.
    generator = np.random.default_rng(0)
    return [{item["name"]: generator.random(shape, dtype=np.float32).astype(dtype, copy=False)}]


def _compare(reference: list[np.ndarray], candidate: list[np.ndarray], tolerances: Tolerances) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise RuntimeError("candidate output count differs from FP32 reference")
    metrics: list[dict[str, float]] = []
    for ref, got in zip(reference, candidate, strict=True):
        if ref.shape != got.shape or not np.isfinite(ref).all() or not np.isfinite(got).all():
            raise RuntimeError("reference/candidate output shape differs or is non-finite")
        diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
        denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-8)
        ref_norm, got_norm = np.linalg.norm(ref.ravel()), np.linalg.norm(got.ravel())
        cosine = (
            1.0
            if ref_norm == 0 and got_norm == 0
            else float(np.dot(ref.ravel(), got.ravel()) / max(ref_norm * got_norm, 1e-12))
        )
        metrics.append(
            {
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "max_rel": float((diff / denom).max()),
                "cosine": cosine,
            }
        )
    merged = {
        name: max(item[name] for item in metrics) if name != "mean_abs" else max(item[name] for item in metrics)
        for name in metrics[0]
    }
    if (
        merged["max_abs"] > tolerances.max_abs
        or merged["mean_abs"] > tolerances.mean_abs
        or merged["max_rel"] > tolerances.max_rel
        or merged["cosine"] < tolerances.min_cosine
    ):
        raise RuntimeError(f"numerical validation gate failed: {merged}")
    return merged | {"samples": len(metrics), "passed": True}


def _compare_detection_sets(
    reference: list[np.ndarray],
    candidate: list[np.ndarray],
    tolerances: Tolerances,
    score_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compare end-to-end ``[..., 6]`` detections as sets, not row order."""

    if len(reference) != len(candidate):
        raise RuntimeError("candidate output count differs from FP32 reference")
    aligned_reference: list[np.ndarray] = []
    aligned_candidate: list[np.ndarray] = []
    overlaps: list[float] = []
    score_deltas: list[float] = []
    lost = added = 0

    def row_iou(left: np.ndarray, right: np.ndarray) -> float:
        x1, y1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
        x2, y2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
        right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
        union = left_area + right_area - intersection
        return intersection / union if union else 0.0

    for expected, actual in zip(reference, candidate, strict=True):
        if expected.shape != actual.shape or expected.ndim != 3 or expected.shape[-1] != 6:
            raise RuntimeError("detection-set comparison requires equal [batch, detections, 6] outputs")
        for batch_index in range(expected.shape[0]):
            left = expected[batch_index][expected[batch_index, :, 4] > score_threshold]
            right = actual[batch_index][actual[batch_index, :, 4] > score_threshold]
            candidates = set(range(len(right)))
            for row in left:
                same_class = [index for index in candidates if int(right[index, 5]) == int(row[5])]
                if not same_class:
                    lost += 1
                    continue
                index = min(
                    same_class,
                    key=lambda value: (
                        float(np.max(np.abs(row[:4] - right[value, :4]))),
                        float(abs(row[4] - right[value, 4])),
                        value,
                    ),
                )
                candidates.remove(index)
                aligned_reference.append(row[:5])
                aligned_candidate.append(right[index, :5])
                overlaps.append(row_iou(row, right[index]))
                score_deltas.append(float(abs(row[4] - right[index, 4])))
            added += len(candidates)
    if lost or added:
        raise RuntimeError(f"numerical validation detection set changed: lost={lost}, added={added}")
    if not aligned_reference:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_rel": 0.0,
            "cosine": 1.0,
            "samples": 0,
            "passed": True,
            "comparison": "detection_set",
            "score_threshold": score_threshold,
        }
    min_iou = min(overlaps)
    max_score_delta = max(score_deltas)
    if min_iou < max(0.5, 1 - tolerances.max_rel) or max_score_delta > tolerances.max_abs:
        raise RuntimeError(
            f"semantic detection validation gate failed: min_iou={min_iou}, max_score_delta={max_score_delta}"
        )
    metrics = {
        "max_abs": max_score_delta,
        "mean_abs": float(np.mean(score_deltas)),
        "max_rel": 1 - min_iou,
        "cosine": 1.0,
        "samples": len(overlaps),
        "passed": True,
    }
    return metrics | {
        "comparison": "detection_set",
        "score_threshold": score_threshold,
        "matched": len(aligned_reference),
        "lost": lost,
        "added": added,
    }


def _compare_manifest_detections(
    manifest: VisionManifest,
    output_names: list[str],
    reference: list[np.ndarray],
    candidate: list[np.ndarray],
    tolerances: Tolerances,
) -> dict[str, Any]:
    """Compare raw DETR query outputs after the manifest's set-valued decoder."""

    if manifest.decoder.kind != "rfdetr":
        raise RuntimeError(f"semantic export smoke is not implemented for decoder {manifest.decoder.kind!r}")
    boxes_name = manifest.output.boxes_name
    logits_name = manifest.output.logits_name
    if not boxes_name or not logits_name or boxes_name not in output_names or logits_name not in output_names:
        raise RuntimeError("RF-DETR semantic export smoke requires named boxes and logits outputs")
    from ..decoders import rfdetr_output
    from ..validation import ValidationPolicy, compare_detections

    def decoded(values: list[np.ndarray]):
        named = dict(zip(output_names, values, strict=True))
        return rfdetr_output(
            {"boxes": named[boxes_name], "logits": named[logits_name]},
            (int(manifest.input.shape[-1]), int(manifest.input.shape[-2])),
            manifest.decoder.top_k,
            manifest.decoder.score_threshold,
        )

    policy = ValidationPolicy(
        max_map_drop=1,
        iou_threshold=max(0.5, 1 - tolerances.max_rel),
        score_delta=tolerances.max_abs,
        high_confidence=manifest.decoder.score_threshold,
    )
    metrics = compare_detections(decoded(reference), decoded(candidate), policy)
    if (
        metrics["high_conf_lost"]
        or metrics["high_conf_added"]
        or metrics["class_mismatches"]
        or metrics["coverage"] < 1
        or metrics["max_score_delta"] > tolerances.max_abs
    ):
        raise RuntimeError(f"semantic detection validation gate failed: {metrics}")
    return {
        "max_abs": metrics["max_score_delta"],
        "mean_abs": metrics["mean_score_delta"],
        "max_rel": 1 - metrics["min_iou"] if metrics["matched"] else 0.0,
        "cosine": 1.0,
        "samples": metrics["matched"],
        "passed": True,
        "comparison": "manifest_decoded_detection_set",
        "parity": metrics,
    }


def _ort_run(path: Path, batches: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, batches[0])


def _validate_onnx(
    source: Path, candidate: Path, calibration: CalibrationManifest | None, tolerances: Tolerances
) -> dict[str, Any]:
    source_metadata = _onnx_metadata(source)
    batches = _samples(source_metadata, calibration)
    # Every golden sample is actually run; aggregate the worst metrics.
    all_metrics = [_compare(_ort_run(source, [batch]), _ort_run(candidate, [batch]), tolerances) for batch in batches]
    return {
        "backend": "onnxruntime",
        "samples": len(all_metrics),
        "passed": True,
        "worst": {
            k: max(m["worst"][k] if "worst" in m else m[k] for m in all_metrics)
            for k in ("max_abs", "mean_abs", "max_rel", "cosine")
        },
    }


def _dynamic_check(metadata: dict[str, Any], request: BuildRequest) -> None:
    if len(metadata["inputs"]) != 1:
        raise ValueError("MVP export supports exactly one model input")
    symbolic = any(isinstance(dim, str) and dim for item in metadata["inputs"] for dim in item["shape"])
    if request.dynamic != symbolic:
        raise ValueError("requested dynamic policy does not match artifact symbolic dimensions")
    if request.profiles and not request.dynamic:
        raise ValueError("shape profiles require --dynamic")
    names = {item["name"] for item in metadata["inputs"]}
    if any(profile.input_name not in names for profile in request.profiles):
        raise ValueError("profile references an unknown input")
    for profile in request.profiles:
        shape = metadata["inputs"][0]["shape"]
        if len(shape) != len(profile.opt_shape):
            raise ValueError("profile rank must match model input rank")
        for label, values in (("min", profile.min_shape), ("opt", profile.opt_shape), ("max", profile.max_shape)):
            if any(isinstance(dim, int) and dim > 0 and dim != value for dim, value in zip(shape, values, strict=True)):
                raise ValueError(f"{label} profile dimensions must preserve static model dimensions")


def _build_onnx(
    request: BuildRequest, source: Path, tempdir: Path, calibration: CalibrationManifest | None
) -> tuple[Path, dict[str, Any], str, dict[str, Any]]:
    import onnx

    candidate = tempdir / "candidate.onnx"
    if request.precision == "fp32":
        shutil.copyfile(source, candidate)
    elif request.precision == "fp16":
        from onnxconverter_common import float16

        converted = float16.convert_float_to_float16(onnx.load(source), keep_io_types=True)
        onnx.save(converted, candidate)
    else:
        if not calibration:
            raise ValueError("INT8 requires calibration_manifest_path")
        _samples(_onnx_metadata(source), calibration)
        quantize_ort_static(str(source), str(candidate), calibration)
    metadata = _onnx_metadata(candidate)
    effective = _effective_onnx_precision(candidate)
    if effective != request.precision:
        raise RuntimeError(f"requested {request.precision}, artifact inspection found {effective}")
    return candidate, metadata, effective, _validate_onnx(source, candidate, calibration, request.tolerances)


def _build_openvino(
    request: BuildRequest, source: Path, tempdir: Path, calibration: CalibrationManifest | None
) -> tuple[Path, dict[str, Any], str, dict[str, Any]]:
    from .openvino import build_openvino

    candidate, metadata, effective = build_openvino(source, tempdir / "candidate.xml", request.precision, calibration)
    import onnxruntime as ort
    import openvino as ov

    source_metadata = _onnx_metadata(source)
    batches = _samples(source_metadata, calibration)
    core = ov.Core()
    reference = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    reference_input = reference.get_inputs()[0].name
    reference_outputs = [item.name for item in reference.get_outputs()]
    got = core.compile_model(str(candidate), "CPU")
    manifest = VisionManifest.from_yaml(request.manifest_path) if request.manifest_path else None
    results = []
    for batch in batches:
        feed = batch if isinstance(batch, dict) else {reference_input: batch}
        expected = reference.run(reference_outputs, feed)
        inference = got(batch)
        named_output = {}
        for port, value in inference.items():
            names = set(port.get_names())
            if not names:
                names = {port.get_any_name()}
            for name in names:
                named_output[name] = value
        missing = [name for name in reference_outputs if name not in named_output]
        if missing:
            raise RuntimeError(f"OpenVINO outputs do not preserve ONNX names: {missing}")
        output = [named_output[name] for name in reference_outputs]
        output_shape = source_metadata["outputs"][0].get("shape", [])
        if manifest is not None and manifest.decoder.kind == "rfdetr":
            results.append(
                _compare_manifest_detections(manifest, reference_outputs, expected, output, request.tolerances)
            )
        else:
            compare = (
                _compare_detection_sets
                if len(source_metadata["outputs"]) == 1 and len(output_shape) == 3 and output_shape[-1] == 6
                else _compare
            )
            results.append(compare(expected, output, request.tolerances))
    return (
        candidate,
        metadata,
        effective,
        {
            "backend": "openvino-cpu",
            "samples": len(results),
            "passed": True,
            "worst": {k: max(item[k] for item in results) for k in ("max_abs", "mean_abs", "max_rel", "cosine")},
        },
    )


def build(request: BuildRequest) -> BuildReport:
    """Build exactly one portable artifact and atomically persist its report."""
    output = Path(request.output)
    if output.parent.is_symlink():
        raise ValueError("output parent must not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    recover_orphan_builds(output.parent)
    calibration = (
        CalibrationManifest.from_path(request.calibration_manifest_path) if request.calibration_manifest_path else None
    )
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.build-", dir=output.parent) as name:
        tempdir = Path(name)
        source = normalize_source(request, tempdir)
        source_meta = _onnx_metadata(source)
        _dynamic_check(source_meta, request)
        if request.target == "onnxruntime":
            candidate, metadata, effective, validation = _build_onnx(request, source, tempdir, calibration)
        elif request.target == "openvino":
            candidate, metadata, effective, validation = _build_openvino(request, source, tempdir, calibration)
        else:
            from ..backends import TensorRTBackend

            candidate, command, logs, fingerprint = TensorRTBackend.build_atomic(
                source, tempdir / "candidate.engine", request
            )
            metadata, effective, validation = (
                source_meta,
                request.precision,
                {"backend": "tensorrt", "build_only": True},
            )
        output_parts = [output]
        if request.target == "openvino":
            output_parts = [output.with_suffix(".bin"), output]
        report_path = Path(f"{output}.build-report.json")
        _assert_fresh(*output_parts, report_path)
        artifacts = {str(part.name): digest(candidate.with_suffix(part.suffix)) for part in output_parts}
        report = BuildReport(
            source=request.source,
            intermediate=None,
            output=str(output),
            target=request.target,
            requested_precision=request.precision,
            effective_precision=effective,
            source_digest=digest(request.source),
            intermediate_digest=digest(source),
            output_digest=digest(candidate),
            artifact_digests=artifacts,
            calibration_digest=calibration.digest if calibration else None,
            dynamic=request.dynamic,
            profiles=[p.model_dump() for p in request.profiles],
            model=metadata,
            versions=_versions(),
            target_fingerprint=locals().get("fingerprint", {}),
            command=locals().get("command"),
            logs_digest=hashlib.sha256(locals().get("logs", "").encode()).hexdigest()
            if request.target == "tensorrt"
            else None,
            validation=validation,
        ).finalized()
        staged_report = tempdir / "report.json"
        staged_report.write_text(json.dumps(report.model_dump(mode="json"), sort_keys=True, indent=2))
        journal = output.parent / f".vision-build-pending-{uuid.uuid4().hex}.json"
        _fsync_json(
            journal,
            {"report": str(report_path), "artifacts": {str(path): artifacts[path.name] for path in output_parts}},
        )
        owned_final_paths: list[tuple[Path, str]] = []
        try:
            for destination in output_parts:
                staged = candidate.with_suffix(destination.suffix)
                _publish_no_clobber(staged, destination)
                owned_final_paths.append((destination, artifacts[destination.name]))
            _publish_no_clobber(staged_report, report_path)
            journal.unlink(missing_ok=True)
            return report
        except Exception:
            for path, expected in owned_final_paths:
                if path.is_file() and not path.is_symlink() and digest(path) == expected:
                    path.unlink()
            if not report_path.exists():
                journal.unlink(missing_ok=True)
            raise


# Compatibility aliases deliberately funnel into the one public build contract.
def _build_onnx_compat(request: BuildRequest) -> BuildReport:
    if request.target != "onnxruntime":
        raise ValueError("build_onnx only accepts target='onnxruntime'")
    return build(request)


def _export_onnx(adapter, output, **kwargs):
    path = adapter.export_onnx(output, **kwargs)
    _onnx_metadata(Path(path))
    return {"onnx": str(path), "validated_with_onnx_checker": True}


def _compile_artifact(onnx_path, backend, output, **config):
    return build(
        BuildRequest(source=onnx_path, output=output, target=backend, precision=config.get("precision", "fp32"))
    ).output
