from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from .adapters import resolve_adapter
from .manifest import VisionManifest
from .optimization import SearchRequest, optimize_search
from .parity import compare
from .validation import compare_detections as compare_detections
from .validation import validate_dataset as _validate_dataset


def _known_native_source(source: str) -> tuple[str, str] | None:
    value = source.lower()
    match = re.fullmatch(r"(?:ultralytics/)?(yolo26[nslmx])(?:\.pt)?", value)
    if match:
        return f"{match.group(1)}.pt", "yolo26"
    match = re.fullmatch(r"(?:ultralytics/)?(rtdetr-[lx])(?:\.pt)?", value)
    if match:
        return f"{match.group(1)}.pt", "rtdetr"
    path = Path(source)
    if path.is_file() and re.fullmatch(r"yolo26[nslmx]\.pt", path.name.lower()):
        return source, "yolo26"
    if path.is_file() and re.fullmatch(r"rtdetr-[lx]\.pt", path.name.lower()):
        return source, "rtdetr"
    return None


def _ultralytics_onnx_manifest(source: str) -> VisionManifest | None:
    path = Path(source)
    if path.suffix.lower() != ".onnx" or not path.is_file():
        return None
    try:
        import onnx
    except ImportError:
        return None
    model = onnx.load(str(path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get("author") != "Ultralytics" or metadata.get("task") != "detect":
        return None
    inputs = [item.name for item in model.graph.input]
    outputs = [item.name for item in model.graph.output]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Ultralytics detection ONNX requires one named input and output")
    size = ast.literal_eval(metadata.get("imgsz", "[640, 640]"))
    if not isinstance(size, list) or len(size) != 2 or any(not isinstance(value, int) for value in size):
        raise ValueError("Ultralytics ONNX metadata has an invalid imgsz")
    if metadata.get("head") == "RTDETRDecoder":
        return VisionManifest(
            source=source,
            family="onnx",
            input={"shape": [1, 3, *size], "input_name": inputs[0]},
            preprocessing={"resize": "resize", "implementation": "opencv_linear_scale_fill"},
            output={"box_format": "cxcywh", "boxes_name": outputs[0]},
            decoder={"kind": "rtdetr_combined", "boxes_normalized": True},
        )
    if metadata.get("head") == "Detect" and metadata.get("end2end", "").lower() == "true":
        return VisionManifest(
            source=source,
            family="onnx",
            input={"shape": [1, 3, *size], "input_name": inputs[0]},
            output={"boxes_name": outputs[0]},
            decoder={"kind": "yolo_end2end"},
        )
    raise ValueError("unsupported Ultralytics ONNX detection head; supply an explicit manifest")


def load(
    source: str,
    device: str = "auto",
    manifest: VisionManifest | None = None,
    family: str | None = None,
    expected_capsule_id: str | None = None,
    backend: str | None = None,
    provider: str | None = None,
    precision: str | None = None,
):
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device must be auto/cpu/cuda/mps")
    if backend is not None and backend not in ("pytorch", "onnxruntime", "openvino", "tensorrt"):
        raise ValueError("backend must be pytorch/onnxruntime/openvino/tensorrt")
    if precision is not None and precision not in ("fp32", "fp16", "int8"):
        raise ValueError("precision must be fp32/fp16/int8")
    if manifest is None and family is None:
        native = _known_native_source(source)
        if native is not None:
            source, family = native
        else:
            manifest = _ultralytics_onnx_manifest(source)
            if manifest is not None:
                family = "onnx"
    rfdetr_variants = {
        "rfdetr-nano": ("nano", 384),
        "rf-detr-nano": ("nano", 384),
        "rfdetr-small": ("small", 512),
        "rf-detr-small": ("small", 512),
        "rfdetr-medium": ("medium", 576),
        "rf-detr-medium": ("medium", 576),
        "rfdetr-base": ("base", 560),
        "rf-detr-base": ("base", 560),
        "rfdetr-large": ("large", 704),
        "rf-detr-large": ("large", 704),
    }
    if manifest is None and family is None and source.lower() in rfdetr_variants:
        variant, resolution = rfdetr_variants[source.lower()]
        family = "rfdetr"
        manifest = VisionManifest(
            source=source,
            family="rfdetr",
            variant=variant,
            input={"shape": [1, 3, resolution, resolution]},
            preprocessing={
                "resize": "resize",
                "implementation": "torchvision_bilinear_antialias_false",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            output={"box_format": "cxcywh"},
            decoder={"kind": "rfdetr", "boxes_normalized": True},
        )
    p = Path(source)
    if p.is_dir() and (p / "manifest.yaml").exists():
        from .adapters import BackendAdapter
        from .capsule import RuntimeRequest, select_runtime

        if not expected_capsule_id:
            raise ValueError("loading a release capsule requires expected_capsule_id from a trusted registry")
        request = RuntimeRequest(
            backend=backend,
            provider=provider,
            device=None if device == "auto" else device,
            precision=precision,
            expected_capsule_id=expected_capsule_id,
        )
        selection = select_runtime(p, request)
        if selection.selected is None:
            detail = "; ".join(selection.reasons) or "no executable artifacts"
            raise RuntimeError(f"no capsule runtime satisfies the request: {detail}")
        artifact = selection.selected
        source = str(p / artifact.logical_path)
        manifest = VisionManifest.from_yaml(p / "manifest.yaml").model_copy(update={"source": source})
        if artifact.backend == "pytorch":
            return resolve_adapter(source, manifest, manifest.family).load(source, artifact.device)
        manifest = manifest.model_copy(update={"family": "onnx"})
        return BackendAdapter(
            manifest,
            artifact.backend,
            artifact.provider,
            artifact.precision,
        ).load(source, artifact.device)
    if manifest is None and family:
        if family in ("yolo", "yolo26"):
            decoder = {"kind": "yolo_end2end"}
            manifest = VisionManifest(
                source=source,
                family=family,
                output={"boxes_name": "output0"},
                decoder=decoder,
            )
        elif family == "rtdetr":
            decoder = {"kind": "rtdetr_combined", "boxes_normalized": True}
            manifest = VisionManifest(
                source=source,
                family=family,
                preprocessing={"resize": "resize", "implementation": "opencv_linear_scale_fill"},
                output={"box_format": "cxcywh", "boxes_name": "output0"},
                decoder=decoder,
            )
        elif family == "rfdetr":
            variant = "nano"
            decoder = {"kind": "rfdetr", "boxes_normalized": True}
            manifest = VisionManifest(
                source=source,
                family=family,
                variant=variant,
                input={"shape": [1, 3, 384, 384]},
                preprocessing={
                    "resize": "resize",
                    "implementation": "torchvision_bilinear_antialias_false",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                output={"box_format": "cxcywh"},
                decoder=decoder,
            )
        else:
            decoder = {"kind": "detr"}
        if manifest is None:
            manifest = VisionManifest(source=source, family=family, decoder=decoder)
    if backend is None and manifest is not None and p.is_file():
        matched = [artifact for artifact in manifest.artifacts if Path(artifact.path).resolve() == p.resolve()]
        if len(matched) > 1:
            raise ValueError("manifest declares the same source as multiple runtime artifacts")
        if matched:
            artifact = matched[0]
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if not artifact.validated or artifact.checksum != actual:
                raise ValueError("runtime artifact is not validated or its checksum changed")
            backend = artifact.backend
            precision = precision or artifact.precision
            provider = provider or artifact.compatible.get("provider")
        elif p.suffix.lower() in {".xml", ".engine"}:
            raise ValueError("compiled runtime artifacts require a validated manifest artifact declaration")
    if backend is not None and backend != "pytorch":
        from .adapters import BackendAdapter

        if manifest is None:
            raise ValueError("direct portable backend loading requires an explicit VisionManifest")
        return BackendAdapter(manifest, backend, provider, precision).load(source, device)
    return resolve_adapter(source, manifest, family).load(source, device)


def validate(
    reference,
    candidate,
    *,
    dataset=None,
    policy=None,
    output_dir=None,
    **kwargs,
):
    """Validate a pair of detection results or an auditable COCO dataset run.

    Dataset validation requires explicit identities for both *loaded* models;
    callers cannot silently attach predictions to an arbitrary label.
    """
    if dataset is None:
        return compare(reference, candidate, **kwargs).to_dict()
    if policy is None or output_dir is None or kwargs:
        raise ValueError("dataset validation requires only targets, dataset, policy and output_dir")
    return _validate_dataset(reference, candidate, dataset, policy, output_dir)


validate_dataset = _validate_dataset


def optimize(model_or_request, evaluator=None, **kwargs):
    """Optimize a loaded model locally, or run the advanced signed search API."""
    if isinstance(model_or_request, SearchRequest):
        if evaluator is None or kwargs:
            raise TypeError("advanced SearchRequest optimization requires exactly one evaluator")
        return optimize_search(model_or_request, evaluator)
    if evaluator is not None:
        raise TypeError("loaded-model optimization accepts keyword configuration, not an evaluator")
    from .auto_optimize import optimize_model

    return optimize_model(model_or_request, **kwargs)
