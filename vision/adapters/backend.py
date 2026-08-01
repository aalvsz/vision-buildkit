from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..backends import OpenVINOBackend, OrtBackend, TensorRTBackend
from .onnx import OnnxAdapter
from .processor import DetectionProcessor


class BackendAdapter(OnnxAdapter):
    """Manifest-driven detection adapter over a named-output execution backend."""

    def __init__(
        self,
        manifest,
        backend: str,
        provider: str | None = None,
        precision: str | None = None,
    ):
        super().__init__(manifest)
        self.backend_name = backend
        self.requested_provider = provider
        self.precision = precision

    def load(self, source: str, device: str = "auto"):
        if self.manifest is None:
            raise ValueError("portable backend execution requires a VisionManifest")
        if self.backend_name == "onnxruntime":
            provider = self.requested_provider
            if provider is None:
                provider = {
                    "auto": "CPUExecutionProvider",
                    "cpu": "CPUExecutionProvider",
                    "cuda": "CUDAExecutionProvider",
                    "coreml": "CoreMLExecutionProvider",
                }.get(device)
            if provider is None:
                raise ValueError("ONNX Runtime device must be auto/cpu/cuda/coreml")
            self.backend = OrtBackend(source, [provider], input_name=self.manifest.input.input_name)
            self.provider = provider
            self.device = {
                "CPUExecutionProvider": "cpu",
                "CUDAExecutionProvider": "cuda",
                "CoreMLExecutionProvider": "coreml",
            }.get(provider, device)
            self.output_names = set(self.backend.outputs)
            self.input_name = self.backend.input
        elif self.backend_name == "openvino":
            selected = self.requested_provider or ("CPU" if device == "auto" else device.upper())
            self.backend = OpenVINOBackend(source, selected, input_name=self.manifest.input.input_name)
            self.provider = selected
            self.device = selected.lower()
            self.output_names = set(self.backend.outputs)
            self.input_name = self.backend.input_name
        elif self.backend_name == "tensorrt":
            if device not in ("auto", "cuda"):
                raise ValueError("TensorRT execution requires device auto/cuda")
            self.backend = TensorRTBackend(source, input_name=self.manifest.input.input_name)
            self.provider = self.requested_provider or "TensorRTExecutionProvider"
            self.device = "cuda"
            self.output_names = set(self.backend.output_names)
            self.input_name = self.backend.input_name
        else:
            raise ValueError(f"unsupported execution backend {self.backend_name!r}")
        self._preflight_outputs()
        self.processor = DetectionProcessor(self.manifest)
        return self

    def runtime_identity(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "provider": self.provider,
            "device": self.device,
            "precision": self.precision
            or {"float32": "fp32", "float16": "fp16", "uint8": "int8"}[self.manifest.input.dtype],
            "input_shape": list(self.manifest.input.shape),
        }

    def forward(self, context):
        tensor, trace = context
        return self.backend.run(tensor), trace

    def __call__(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        converted = [
            Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
            for image in images
        ]
        if not converted:
            return []
        declared = self.manifest.input.shape[0]
        fixed_batch = declared if isinstance(declared, int) else None
        capacity = fixed_batch or len(converted)
        if capacity <= 0:
            raise ValueError("manifest batch dimension must be positive or dynamic")
        results = []
        for start in range(0, len(converted), capacity):
            group = converted[start : start + capacity]
            actual = len(group)
            padded = group + [group[-1]] * ((fixed_batch or actual) - actual)
            contexts = [self.preprocess(image) for image in padded]
            tensor = np.concatenate([context[0] for context in contexts], axis=0)
            outputs = self.backend.run(tensor)
            for index, image in enumerate(group):
                sliced = {}
                for name, value in outputs.items():
                    array = np.asarray(value)
                    if array.ndim == 0 or array.shape[0] != len(padded):
                        raise RuntimeError(f"backend output {name!r} has no matching batch dimension: {array.shape}")
                    sliced[name] = array[index : index + 1]
                results.append(self.postprocess((sliced, contexts[index][1]), image.size))
        return results

    def export_onnx(self, output: str, **kwargs) -> str:
        if self.backend_name != "onnxruntime":
            raise RuntimeError("compiled backend artifacts cannot be reverse-exported to ONNX")
        return str(output)
