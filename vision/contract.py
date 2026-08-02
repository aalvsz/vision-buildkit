"""Strict, versioned semantic contract for portable detection builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Symbol = Annotated[str, StringConstraints(pattern=r"^(N|H|W)$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Resize(Strict):
    kind: Literal["resize"]
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    interpolation: Literal["nearest", "bilinear", "bicubic"]
    # Omitted values preserve the schema-v1 PIL behavior.  Certified provider
    # contracts must select their exact implementation explicitly.
    implementation: Literal["pil", "opencv_linear_scale_fill", "torchvision_bilinear_antialias_false"] = Field(
        default="pil", exclude_if=lambda value: value == "pil"
    )
    invertible: Literal[True] = True

    @model_validator(mode="after")
    def v(self):
        if self.implementation in {"opencv_linear_scale_fill", "torchvision_bilinear_antialias_false"}:
            if self.interpolation != "bilinear":
                raise ValueError(f"{self.implementation} requires bilinear interpolation")
        return self


class Letterbox(Strict):
    kind: Literal["letterbox"]
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    pad_value: int = Field(ge=0, le=255)
    interpolation: Literal["nearest", "bilinear", "bicubic"]
    placement: Literal["center", "top_left"] = "center"
    rounding: Literal["floor", "round_half_up"] = "round_half_up"
    scale: Literal["min_ratio", "exact"] = "min_ratio"
    implementation: Literal["pil", "ultralytics_letterbox_cv2", "libreyolo_yolo9_top_left_cv2"] = "pil"
    invertible: Literal[True] = True


class Normalize(Strict):
    kind: Literal["normalize"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @model_validator(mode="after")
    def v(self):
        if any(x == 0 for x in self.std):
            raise ValueError("normalization std must be non-zero")
        return self


class ColorConvert(Strict):
    kind: Literal["color_convert"]
    source: Literal["RGB", "BGR", "GRAY"]
    target: Literal["RGB", "BGR", "GRAY"]


class Transpose(Strict):
    kind: Literal["transpose"]
    order: tuple[Literal[0, 1, 2, 3], Literal[0, 1, 2, 3], Literal[0, 1, 2, 3], Literal[0, 1, 2, 3]]

    @model_validator(mode="after")
    def v(self):
        if set(self.order) != {0, 1, 2, 3}:
            raise ValueError("transpose order must be a permutation of 0,1,2,3")
        return self


Transform = Annotated[Resize | Letterbox | Normalize | ColorConvert | Transpose, Field(discriminator="kind")]


class SemanticInput(Strict):
    semantic_type: Literal["image"] = "image"
    color: Literal["RGB", "BGR", "GRAY"]
    layout: Literal["NCHW", "NHWC"]
    dtype: Literal["float32", "float16", "uint8"]
    input_name: str = Field(min_length=1)
    shape: tuple[int | Symbol, int | Symbol, int | Symbol, int | Symbol]

    @model_validator(mode="after")
    def v(self):
        if any(isinstance(x, int) and x <= 0 for x in self.shape):
            raise ValueError("input shape dimensions must be positive")
        return self


class TensorSpec(Strict):
    name: str = Field(min_length=1)
    dtype: Literal["float32", "float16", "int64", "uint8"]
    shape: tuple[int | Symbol, ...] = Field(min_length=1)


class RawOutput(Strict):
    semantics: Literal[
        "boxes_scores_classes",
        "yolo_end2end",
        "yolo_raw",
        "detr_logits_boxes",
        "rtdetr_combined",
        "custom",
    ]
    tensors: tuple[TensorSpec, ...] = Field(min_length=1)
    decoder_plugin: str | None = None
    roles: dict[Literal["boxes", "scores", "classes"], str] | None = None

    @model_validator(mode="after")
    def v(self):
        if len({x.name for x in self.tensors}) != len(self.tensors):
            raise ValueError("raw tensor names must be unique")
        if self.semantics == "custom" and not self.decoder_plugin:
            raise ValueError("custom output requires decoder_plugin")
        if self.semantics == "boxes_scores_classes":
            if self.roles is None or set(self.roles) != {"boxes", "scores", "classes"}:
                raise ValueError("boxes_scores_classes requires exactly boxes/scores/classes roles")
            by_name = {item.name: item for item in self.tensors}
            if set(self.roles.values()) != set(by_name) or len(self.tensors) != 3:
                raise ValueError("output roles must name exactly the declared tensors")
            boxes, scores, classes = (by_name[self.roles[key]] for key in ("boxes", "scores", "classes"))
            if boxes.dtype not in {"float32", "float16"} or len(boxes.shape) < 1 or boxes.shape[-1] != 4:
                raise ValueError("boxes must be floating tensors whose final dimension is 4")
            if scores.dtype not in {"float32", "float16"} or classes.dtype not in {"int64", "uint8"}:
                raise ValueError("scores must be floating and classes integral")
        if self.semantics == "yolo_end2end":
            if self.roles is not None or self.decoder_plugin is not None or len(self.tensors) != 1:
                raise ValueError("yolo_end2end requires one tensor and no output roles/plugin")
            item = self.tensors[0]
            if item.dtype not in {"float32", "float16"} or len(item.shape) not in {2, 3} or item.shape[-1] != 6:
                raise ValueError("yolo_end2end tensor must be float rank 2/3 with final dimension 6")
        if self.semantics == "rtdetr_combined":
            if self.roles is not None or self.decoder_plugin is not None or len(self.tensors) != 1:
                raise ValueError("rtdetr_combined requires one tensor and no output roles/plugin")
            item = self.tensors[0]
            if item.name != "output0" or item.dtype != "float32" or item.shape != (1, 300, 6):
                raise ValueError("rtdetr_combined requires exact output0 float32 [1,300,6] ABI")
        return self


class ResultSemantics(Strict):
    fields: tuple[Literal["boxes", "scores", "classes"], ...] = ("boxes", "scores", "classes")
    box_format: Literal["xyxy", "xywh", "cxcywh"]
    coordinate_space: Literal["original_image", "model_input", "normalized"]
    nms_owner: Literal["none", "source", "target", "contract"]
    nms_iou_threshold: float | None = Field(default=None, gt=0, le=1)
    score_threshold: float = Field(ge=0, le=1)
    class_aware_nms: bool
    clip_boxes: bool
    top_k: int | None = Field(default=None, gt=0)
    max_detections: int = Field(gt=0)
    suppression: Literal["none", "greedy_iou"]
    ordering: Literal["score_desc", "source_order"]
    tie_policy: Literal["stable", "class_then_box"]

    @model_validator(mode="after")
    def v(self):
        if len(set(self.fields)) != len(self.fields) or set(self.fields) != {"boxes", "scores", "classes"}:
            raise ValueError("result fields must be complete and unique")
        if self.nms_owner == "none":
            if self.suppression != "none" or self.nms_iou_threshold is not None or self.class_aware_nms:
                raise ValueError(
                    "NMS-free results require suppression none, no IoU threshold, and class_aware_nms false"
                )
        elif self.suppression != "greedy_iou" or self.nms_iou_threshold is None:
            raise ValueError("NMS results require greedy_iou and an IoU threshold")
        return self


class Tolerances(Strict):
    box_iou: float = Field(gt=0, le=1)
    score_delta: float = Field(ge=0)
    deterministic: bool = True


class Provenance(Strict):
    source: str = Field(min_length=1)
    source_sha256: Hex
    compatible_artifacts: tuple[str, ...] = ()


class VisionContract(Strict):
    schema_version: Literal[1] = 1
    task: Literal["detection"] = "detection"
    input: SemanticInput
    preprocessing: tuple[Transform, ...] = Field(min_length=1)
    raw_output: RawOutput
    result: ResultSemantics
    labels: tuple[str, ...] = Field(min_length=1)
    coco_category_ids: tuple[int, ...] = Field(min_length=1)
    tolerances: Tolerances
    provenance: Provenance

    @model_validator(mode="after")
    def v(self):
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if len(self.coco_category_ids) != len(self.labels) or (
            len(set(self.coco_category_ids)) != len(self.coco_category_ids)
        ):
            raise ValueError("coco_category_ids must be a unique one-to-one mapping for labels")
        geometries = [item for item in self.preprocessing if isinstance(item, (Resize, Letterbox))]
        if len(geometries) > 1:
            raise ValueError("preprocessing permits at most one geometry transform")
        if self.result.coordinate_space == "original_image" and len(geometries) != 1:
            raise ValueError("original-image results require exactly one invertible geometry transform")
        if geometries:
            geometry = geometries[0]
            spatial = self.input.shape[-2:] if self.input.layout == "NCHW" else self.input.shape[1:3]
            if spatial != (geometry.height, geometry.width):
                raise ValueError("geometry dimensions contradict declared input spatial axes")
            if geometry.kind == "letterbox" and geometry.implementation == "ultralytics_letterbox_cv2":
                if self.preprocessing[0] is not geometry:
                    raise ValueError("ultralytics cv2 letterbox must precede color and numeric transforms")
            if geometry.kind == "letterbox" and geometry.implementation == "libreyolo_yolo9_top_left_cv2":
                if self.preprocessing[0] is not geometry:
                    raise ValueError("LibreYOLO YOLO9 letterbox must precede color and numeric transforms")
                if (
                    geometry.placement != "top_left"
                    or geometry.rounding != "floor"
                    or geometry.scale != "min_ratio"
                    or geometry.interpolation != "bilinear"
                ):
                    raise ValueError("LibreYOLO YOLO9 letterbox requires top-left/int/min-ratio/linear semantics")
            if self.raw_output.semantics == "rtdetr_combined":
                if (
                    not isinstance(geometry, Resize)
                    or geometry.implementation != "opencv_linear_scale_fill"
                    or geometry.interpolation != "bilinear"
                    or self.preprocessing[0] is not geometry
                ):
                    raise ValueError("rtdetr_combined requires first-transform OpenCV INTER_LINEAR scale-fill resize")
        transposes = [item for item in self.preprocessing if isinstance(item, Transpose)]
        if len(transposes) > 1 or (transposes and self.preprocessing[-1] is not transposes[0]):
            raise ValueError("at most one transpose is supported and it must be final")
        if transposes:
            expected = (0, 3, 1, 2) if self.input.layout == "NCHW" else (0, 1, 2, 3)
            if transposes[0].order != expected:
                raise ValueError("transpose order contradicts declared input layout")
        return self

    def canonical_json(self):
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self):
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def to_json(self, path):
        Path(path).write_text(self.canonical_json() + "\n")

    def to_yaml(self, path):
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True))

    @classmethod
    def from_path(cls, path):
        p = Path(path)
        return cls.model_validate(json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text()))
