from __future__ import annotations

import numpy as np

from ..decoders import detr, rfdetr_output, rtdetr_combined, yolo_end2end, yolo_end2end_contract, yolo_raw
from ..types import Box, DetectionResult
from .base import ModelAdapter, need
from .processor import DetectionProcessor


class OnnxAdapter(ModelAdapter):
    def runtime_identity(self):
        provider = self.engaged_providers[0]
        return {
            "backend": "onnxruntime",
            "provider": provider,
            "device": {
                "CPUExecutionProvider": "cpu",
                "CUDAExecutionProvider": "cuda",
                "CoreMLExecutionProvider": "coreml",
            }[provider],
            "precision": {"float32": "fp32", "float16": "fp16", "uint8": "int8"}[self.manifest.input.dtype],
            "input_shape": list(self.manifest.input.shape),
        }

    def _preflight_outputs(self) -> None:
        decoder = self.manifest.decoder.kind
        output = self.manifest.output
        if decoder == "boxes_scores_classes":
            required = {output.boxes_name, output.scores_name, output.labels_name}
        elif decoder in {"yolo_end2end", "yolo_raw", "rtdetr_combined"}:
            required = {output.boxes_name}
        elif decoder == "rfdetr":
            required = {output.boxes_name, output.logits_name}
        else:
            required = {output.boxes_name, output.logits_name}
        if None in required or not required <= self.output_names:
            raise ValueError(
                f"ONNX outputs do not satisfy {decoder}: required={sorted(item for item in required if item)}, "
                f"available={sorted(self.output_names)}"
            )

    def load(self, source, device="auto"):
        try:
            import onnxruntime as ort
        except ImportError:
            need("onnxruntime", "adapters")
        explicit = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "coreml": "CoreMLExecutionProvider"}
        if device == "auto":
            provider = "CPUExecutionProvider"
        elif device in explicit:
            provider = explicit[device]
        else:
            raise ValueError("ONNX device must be auto/cpu/cuda/coreml; no provider fallback is permitted")
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(f"requested ORT provider {provider} unavailable; available={available}")
        self.session = ort.InferenceSession(source, providers=[provider])
        self.engaged_providers = self.session.get_providers()
        if self.engaged_providers != [provider]:
            raise RuntimeError(
                f"ORT provider fallback/reordering rejected: requested={provider}, engaged={self.engaged_providers}"
            )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("detection ONNX must expose exactly one image input")
        self.input_name = inputs[0].name
        if self.manifest.input.input_name != self.input_name:
            raise ValueError(
                f"manifest input_name {self.manifest.input.input_name!r} != ONNX input {self.input_name!r}"
            )
        self.output_names = {item.name for item in self.session.get_outputs()}
        expected_shape = self.manifest.input.shape
        actual_shape = inputs[0].shape
        if len(actual_shape) != 4 or any(
            isinstance(expected, int) and isinstance(actual, int) and expected != actual
            for expected, actual in zip(expected_shape, actual_shape, strict=True)
        ):
            raise ValueError(f"manifest input shape {expected_shape} != ONNX input shape {actual_shape}")
        expected_type = {"float32": "tensor(float)", "float16": "tensor(float16)", "uint8": "tensor(uint8)"}[
            self.manifest.input.dtype
        ]
        if inputs[0].type != expected_type:
            raise ValueError(f"manifest input dtype {expected_type} != ONNX input dtype {inputs[0].type}")
        self._preflight_contract_outputs()
        self._preflight_outputs()
        self.processor = DetectionProcessor(self.manifest)
        return self

    def _preflight_contract_outputs(self):
        """Bind certified tensor names/types to the actual ONNX graph."""
        contract = self.manifest.contract
        if contract is None:
            if self.manifest.decoder.kind == "boxes_scores_classes":
                raise ValueError("separate boxes/scores/classes runtime requires an embedded VisionContract")
            return
        raw = contract["raw_output"]
        actual = {item.name: item for item in self.session.get_outputs()}
        if set(actual) != {spec["name"] for spec in raw["tensors"]}:
            raise ValueError("ONNX output names contradict VisionContract")
        for spec in raw["tensors"]:
            observed = actual.get(spec["name"])
            expected_type = {
                "float32": "tensor(float)",
                "float16": "tensor(float16)",
                "int64": "tensor(int64)",
                "uint8": "tensor(uint8)",
            }[spec["dtype"]]
            if observed is None or observed.type != expected_type or len(observed.shape) != len(spec["shape"]):
                raise ValueError(f"ONNX output {spec['name']!r} contradicts VisionContract")
            if any(not isinstance(value, int) for value in observed.shape) or tuple(observed.shape) != tuple(
                spec["shape"]
            ):
                raise ValueError(f"ONNX output {spec['name']!r} concrete shape contradicts VisionContract")
        if self.manifest.decoder.kind != "boxes_scores_classes":
            return
        roles = raw.get("roles") or {}
        expected = {
            "boxes": self.manifest.output.boxes_name,
            "scores": self.manifest.output.scores_name,
            "classes": self.manifest.output.labels_name,
        }
        if roles != expected:
            raise ValueError("manifest separate-output roles contradict VisionContract")
        actual_types = {item.name: item.type for item in self.session.get_outputs()}
        for spec in raw["tensors"]:
            onnx_type = {
                "float32": "tensor(float)",
                "float16": "tensor(float16)",
                "int64": "tensor(int64)",
                "uint8": "tensor(uint8)",
            }[spec["dtype"]]
            if actual_types.get(spec["name"]) != onnx_type:
                raise ValueError(f"ONNX output {spec['name']!r} dtype contradicts VisionContract")

    def preprocess(self, image):
        return self.processor(image)

    def forward(self, context):
        tensor, trace = context
        names = [item.name for item in self.session.get_outputs()]
        values = self.session.run(names, {self.input_name: tensor})
        return dict(zip(names, values, strict=True)), trace

    def postprocess(self, raw, image_size):
        outputs, trace = raw
        decoder = self.manifest.decoder
        if decoder.kind == "boxes_scores_classes":
            result = self._separate_outputs(outputs, image_size)
        elif decoder.kind in {"yolo_end2end", "yolo_raw", "rtdetr_combined"}:
            output_name = self.manifest.output.boxes_name
            if output_name is None or output_name not in self.output_names:
                raise ValueError(f"{decoder.kind} requires an explicit valid output.boxes_name")
            array = outputs[output_name]
            if decoder.kind == "yolo_end2end":
                from ..contract import VisionContract

                result = (
                    yolo_end2end_contract(
                        array, trace, VisionContract.model_validate(self.manifest.contract), image_size=image_size
                    )
                    if self.manifest.contract is not None
                    else yolo_end2end(array, trace, decoder.score_threshold)
                )
            elif decoder.kind == "yolo_raw":
                result = yolo_raw(array, trace, decoder.score_threshold, decoder.iou_threshold)
            else:
                result = rtdetr_combined(array, trace, decoder.score_threshold)
        elif decoder.kind == "rfdetr":
            box_name = self.manifest.output.boxes_name or "dets"
            logits_name = self.manifest.output.logits_name or "labels"
            if {box_name, logits_name} - self.output_names:
                raise ValueError(
                    "RF-DETR named outputs unavailable: "
                    f"requested={box_name, logits_name}, got={sorted(self.output_names)}"
                )
            result = rfdetr_output(
                {"boxes": outputs[box_name], "logits": outputs[logits_name]},
                image_size,
                decoder.top_k,
                decoder.score_threshold,
            )
        else:
            required = {"boxes": self.manifest.output.boxes_name, "logits": self.manifest.output.logits_name}
            if any(value is None for value in required.values()) or any(
                value not in self.output_names for value in required.values()
            ):
                raise ValueError("DETR requires explicit manifest output tensor names")
            result = detr(
                outputs[required["boxes"]], outputs[required["logits"]], image_size=image_size, top_k=decoder.top_k
            )
        result.metadata.update(self.runtime_identity() | {"input_name": self.input_name})
        return result

    def _separate_outputs(self, outputs, image_size):
        output = self.manifest.output
        names = (output.boxes_name, output.scores_name, output.labels_name)
        if any(name is None for name in names) or any(name not in outputs for name in names):
            raise ValueError("separate outputs require explicit boxes/scores/labels names")
        boxes_raw = np.asarray(outputs[output.boxes_name])
        scores_raw = np.asarray(outputs[output.scores_name])
        labels_raw = np.asarray(outputs[output.labels_name])
        if boxes_raw.ndim < 2 or boxes_raw.shape[-1] != 4:
            raise ValueError("separate boxes output must end in coordinate dimension 4")
        if scores_raw.shape != boxes_raw.shape[:-1] or labels_raw.shape != boxes_raw.shape[:-1]:
            raise ValueError("separate output tensor shapes/cardinality mismatch")
        boxes = boxes_raw.reshape(-1, 4)
        scores = scores_raw.reshape(-1)
        labels = labels_raw.reshape(-1)
        if len(boxes) != len(scores) or len(boxes) != len(labels):
            raise ValueError("separate output cardinality mismatch")
        if (
            boxes.dtype not in (np.float32, np.float16)
            or scores.dtype not in (np.float32, np.float16)
            or labels.dtype not in (np.int64, np.uint8)
        ):
            raise ValueError("separate output dtypes contradict manifest semantics")
        contract = self.manifest.contract or {}
        result = contract.get("result", {})
        threshold = result.get("score_threshold", self.manifest.decoder.score_threshold)
        clipped = result.get("clip_boxes", False)
        converted = []
        for box, score, label in zip(boxes, scores, labels, strict=True):
            if float(score) < threshold:
                continue
            values = [float(value) for value in box]
            if clipped:
                values = [
                    max(0.0, min(values[0], image_size[0])),
                    max(0.0, min(values[1], image_size[1])),
                    max(0.0, min(values[2], image_size[0])),
                    max(0.0, min(values[3], image_size[1])),
                ]
            converted.append((Box(*values), float(score), int(label)))
        converted.sort(key=lambda item: item[1], reverse=True)
        return DetectionResult(
            [item[0] for item in converted],
            [item[1] for item in converted],
            [item[2] for item in converted],
            image_size=image_size,
        )

    def export_onnx(self, output, **kwargs):
        return str(output)
