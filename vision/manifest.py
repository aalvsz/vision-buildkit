"""Strict, portable model contract.

The manifest is deliberately descriptive rather than a bag of backend options: an
artifact is only runnable when its tensor, geometry and decoder contracts agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputSpec(_Strict):
    shape: list[int | str] = Field(default_factory=lambda: [1, 3, 640, 640], min_length=4, max_length=4)
    color: Literal["RGB", "BGR"] = "RGB"
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    dtype: Literal["float32", "float16", "uint8"] = "float32"
    input_name: str = "images"


class PreprocessingSpec(_Strict):
    resize: Literal["letterbox", "resize"] = "letterbox"
    padding: int = Field(default=114, ge=0, le=255)
    interpolation: Literal["bilinear"] = "bilinear"
    implementation: Literal[
        "ultralytics_letterbox_cv2", "opencv_linear_scale_fill", "torchvision_bilinear_antialias_false"
    ] = "ultralytics_letterbox_cv2"
    normalization: Literal["1/255", "none"] = "1/255"
    mean: list[float] | None = None
    std: list[float] | None = None

    @model_validator(mode="after")
    def _normalization_pairs(self) -> PreprocessingSpec:
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must be supplied together")
        if self.mean is not None and (len(self.mean) != 3 or len(self.std or []) != 3):
            raise ValueError("mean and std must contain three RGB channels")
        if self.std is not None and any(value == 0 for value in self.std):
            raise ValueError("std values must not be zero")
        expected = {
            "letterbox": "ultralytics_letterbox_cv2",
            "resize": "opencv_linear_scale_fill" if self.mean is None else "torchvision_bilinear_antialias_false",
        }[self.resize]
        if self.implementation != expected:
            raise ValueError(f"resize {self.resize} requires implementation {expected}")
        return self


class OutputSpec(_Strict):
    box_format: Literal["xyxy", "cxcywh"] = "xyxy"
    boxes_name: str | None = None
    scores_name: str | None = None
    labels_name: str | None = None
    logits_name: str | None = None


class DecoderSpec(_Strict):
    kind: Literal[
        "yolo_end2end",
        "yolo_raw",
        "detr",
        "rtdetr_combined",
        "rtdetr_combined_v1",
        "rfdetr",
        "boxes_scores_classes",
    ] = "yolo_end2end"
    score_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    top_k: int = Field(default=300, gt=0)
    boxes_normalized: bool = False
    max_detections: int | None = Field(default=None, gt=0)
    nms_owner: Literal["none", "source", "target", "contract"] | None = None
    suppression: Literal["none", "greedy_iou"] | None = None
    class_aware_nms: bool | None = None


class CalibrationSample(_Strict):
    path: str
    sha256: str


class CalibrationManifest(_Strict):
    samples: list[CalibrationSample] = Field(min_length=1)
    sample_count: int = Field(gt=0)
    preprocessing_digest: str = Field(min_length=1)
    input_name: str = Field(min_length=1)
    shape: list[int] = Field(min_length=1)
    dtype: Literal["float32", "float16"] = "float32"
    digest: str = Field(min_length=64, max_length=64)

    def canonical_digest(self) -> str:
        payload = self.model_dump(exclude={"digest"}, mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @model_validator(mode="after")
    def verify_contract(self) -> CalibrationManifest:
        if self.sample_count != len(self.samples):
            raise ValueError("sample_count must equal the number of samples")
        if self.digest != self.canonical_digest():
            raise ValueError("calibration manifest digest does not match canonical content")
        return self

    @classmethod
    def from_path(cls, path: str | Path) -> CalibrationManifest:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("calibration manifest must be a regular file")
        raw = json.loads(source.read_text()) if source.suffix == ".json" else yaml.safe_load(source.read_text())
        return cls.model_validate(raw)


class ArtifactSpec(_Strict):
    path: str
    backend: str
    precision: str = "fp32"
    validated: bool = False
    checksum: str | None = None
    compatible: dict[str, str] = Field(default_factory=dict)


class ValidationSpec(_Strict):
    max_map_drop: float | None = None
    status: Literal["pending", "passed", "failed"] = "pending"


class VisionManifest(_Strict):
    schema_version: int = 1
    task: Literal["detection"] = "detection"
    source: str
    family: Literal["yolo", "yolo26", "rfdetr", "rtdetr", "onnx", "torch"] = "onnx"
    variant: str | None = None
    input: InputSpec = Field(default_factory=InputSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    postprocessing: dict[str, str] = Field(default_factory=dict)
    decoder: DecoderSpec = Field(default_factory=DecoderSpec)
    calibration: CalibrationManifest | None = None
    # Certified manifests carry the complete semantic source of truth.  The
    # legacy projection above remains for existing development capsules only.
    contract: dict | None = None

    @model_validator(mode="after")
    def _cross_validate(self) -> VisionManifest:
        if self.input.layout != "NCHW":
            raise ValueError("detection adapters currently require NCHW tensors")
        if self.family == "rfdetr" or self.decoder.kind == "rfdetr":
            if self.preprocessing.resize != "resize" or self.decoder.kind != "rfdetr":
                raise ValueError("RF-DETR requires resize preprocessing and the rfdetr decoder")
            if (
                any(not isinstance(value, int) or value <= 0 for value in self.input.shape[-2:])
                or self.input.shape[-2] != self.input.shape[-1]
            ):
                raise ValueError("RF-DETR requires a fixed positive square input")
            expected = {
                "nano": 384,
                "small": 512,
                "medium": 576,
                "base": 560,
                "large": 704,
            }.get(self.variant or "nano")
            if self.family == "rfdetr" and expected is None:
                raise ValueError("RF-DETR variant must be nano/small/medium/base/large")
            if self.family == "rfdetr" and self.input.shape[-2:] != [expected, expected]:
                raise ValueError(f"RF-DETR {self.variant or 'nano'} requires a {expected}x{expected} input")
        if self.decoder.kind == "rfdetr" and not self.decoder.boxes_normalized:
            raise ValueError("RF-DETR ONNX boxes must be normalized cxcywh")
        if self.decoder.kind in {"rtdetr_combined", "rtdetr_combined_v1"} and not self.decoder.boxes_normalized:
            raise ValueError("RT-DETR combined boxes must be normalized cxcywh")
        if (
            self.decoder.kind in {"rfdetr", "rtdetr_combined", "rtdetr_combined_v1"}
            and self.output.box_format != "cxcywh"
        ):
            raise ValueError("normalized DETR decoder outputs must declare cxcywh boxes")
        if self.decoder.kind == "rtdetr_combined_v1":
            from .contract import VisionContract

            if self.contract is None:
                raise ValueError("rtdetr_combined_v1 requires an embedded RT-DETR VisionContract")
            try:
                certified = VisionContract.model_validate(self.contract)
            except ValueError as error:
                raise ValueError("rtdetr_combined_v1 requires an exact valid VisionContract") from error
            if certified.raw_output.semantics != "rtdetr_combined":
                raise ValueError("rtdetr_combined_v1 requires an embedded RT-DETR VisionContract")
        if self.decoder.kind == "rtdetr_combined" and self.contract is not None:
            raise ValueError("legacy rtdetr_combined manifests cannot carry a certified VisionContract")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> VisionManifest:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))


def validate_contract_equivalence(manifest: VisionManifest, contract) -> None:
    """Fail closed when the legacy runtime manifest contradicts VisionContract."""
    from .contract import Letterbox, Resize

    canonical = contract.model_dump(mode="json")
    if manifest.contract != canonical:
        raise ValueError("manifest cannot exactly represent VisionContract")
    shape = tuple(manifest.input.shape)
    if shape != tuple(contract.input.shape):
        raise ValueError("manifest input shape contradicts VisionContract")
    if (manifest.input.dtype, manifest.input.layout, manifest.input.color) != (
        contract.input.dtype,
        contract.input.layout,
        contract.input.color,
    ):
        raise ValueError("manifest input semantics contradict VisionContract")
    if manifest.input.input_name != contract.input.input_name:
        raise ValueError("manifest input name contradicts VisionContract")
    geometry = next((item for item in contract.preprocessing if isinstance(item, (Resize, Letterbox))), None)
    if geometry is None:
        raise ValueError("VisionContract requires explicit geometry")
    if manifest.preprocessing.resize != geometry.kind:
        raise ValueError("manifest resize contradicts VisionContract")
    if isinstance(geometry, Letterbox) and manifest.preprocessing.padding != geometry.pad_value:
        raise ValueError("manifest letterbox padding contradicts VisionContract")
    expected_box_format = "cxcywh" if contract.raw_output.semantics == "rtdetr_combined" else contract.result.box_format
    if manifest.output.box_format != expected_box_format:
        raise ValueError("manifest box format contradicts VisionContract")
    roles = contract.raw_output.roles or {}
    output_names = (manifest.output.boxes_name, manifest.output.scores_name, manifest.output.labels_name)
    required_names = (
        (contract.raw_output.tensors[0].name, None, None)
        if contract.raw_output.semantics in {"yolo_end2end", "rtdetr_combined"}
        else (roles.get("boxes"), roles.get("scores"), roles.get("classes"))
    )
    if output_names != required_names or manifest.output.logits_name is not None:
        raise ValueError("manifest output roles contradict VisionContract")
    decoder = manifest.decoder
    expected_decoder = {
        "boxes_scores_classes": "boxes_scores_classes",
        "yolo_end2end": "yolo_end2end",
        "rtdetr_combined": "rtdetr_combined_v1",
    }.get(contract.raw_output.semantics)
    if expected_decoder is None:
        raise ValueError(f"VisionManifest cannot certify raw output semantics {contract.raw_output.semantics!r}")
    if decoder.kind != expected_decoder:
        raise ValueError("manifest decoder kind contradicts VisionContract raw output semantics")
    effective_limit = min(contract.result.top_k or contract.result.max_detections, contract.result.max_detections)
    if (
        decoder.score_threshold != contract.result.score_threshold
        or decoder.top_k != effective_limit
        or decoder.max_detections != contract.result.max_detections
        or decoder.nms_owner != contract.result.nms_owner
        or decoder.suppression != contract.result.suppression
        or decoder.class_aware_nms != contract.result.class_aware_nms
        or decoder.iou_threshold != (contract.result.nms_iou_threshold or 0.45)
    ):
        raise ValueError("manifest NMS contradicts VisionContract")


def manifest_from_contract(source: str, contract, *, family: str = "onnx") -> VisionManifest:
    """Produce a runtime manifest whose ``source`` names the runnable artifact.

    A certified capsule supplies its separately retained training checkpoint to
    ``create_certified_capsule``.  This field is intentionally the executable
    model path used by a runner (the artifact path before packaging, then its
    capsule-relative artifact path after packaging), never that checkpoint.
    """
    from .contract import Letterbox, Resize

    geometry = next(item for item in contract.preprocessing if isinstance(item, (Resize, Letterbox)))
    decoder_kind = {
        "boxes_scores_classes": "boxes_scores_classes",
        "yolo_end2end": "yolo_end2end",
        "rtdetr_combined": "rtdetr_combined_v1",
    }.get(contract.raw_output.semantics)
    if decoder_kind is None:
        raise ValueError(f"VisionManifest cannot certify raw output semantics {contract.raw_output.semantics!r}")
    roles = contract.raw_output.roles or {}
    preprocessing = {
        "resize": geometry.kind,
        "padding": geometry.pad_value if isinstance(geometry, Letterbox) else 114,
        "implementation": "ultralytics_letterbox_cv2"
        if isinstance(geometry, Letterbox)
        else "opencv_linear_scale_fill",
        "normalization": "1/255"
        if any(
            item.kind == "normalize" and item.mean == (0.0, 0.0, 0.0) and item.std == (255.0, 255.0, 255.0)
            for item in contract.preprocessing
        )
        else "none",
    }
    value = VisionManifest(
        source=source,
        family=family,
        input={
            "shape": list(contract.input.shape),
            "color": contract.input.color,
            "layout": contract.input.layout,
            "dtype": contract.input.dtype,
            "input_name": contract.input.input_name,
        },
        output={
            "box_format": (
                "cxcywh" if contract.raw_output.semantics == "rtdetr_combined" else contract.result.box_format
            ),
            "boxes_name": roles.get("boxes") or contract.raw_output.tensors[0].name,
            "scores_name": roles.get("scores"),
            "labels_name": roles.get("classes"),
        },
        preprocessing=preprocessing,
        decoder={
            "kind": decoder_kind,
            "boxes_normalized": contract.raw_output.semantics == "rtdetr_combined",
            "score_threshold": contract.result.score_threshold,
            "iou_threshold": contract.result.nms_iou_threshold or 0.45,
            "top_k": min(contract.result.top_k or contract.result.max_detections, contract.result.max_detections),
            "max_detections": contract.result.max_detections,
            "nms_owner": contract.result.nms_owner,
            "suppression": contract.result.suppression,
            "class_aware_nms": contract.result.class_aware_nms,
        },
        contract=contract.model_dump(mode="json"),
    )
    validate_contract_equivalence(value, contract)
    return value
