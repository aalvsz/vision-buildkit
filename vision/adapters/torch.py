from __future__ import annotations

from ..decoders import yolo_end2end, yolo_raw
from .base import ModelAdapter, need, resolve_torch_device
from .processor import DetectionProcessor


class TorchAdapter(ModelAdapter):
    def runtime_identity(self):
        return {
            "backend": "pytorch",
            "provider": "torch",
            "device": self.device,
            "precision": {"float32": "fp32", "float16": "fp16", "uint8": "int8"}[self.manifest.input.dtype],
            "input_shape": list(self.manifest.input.shape),
        }

    def load(self, source, device="auto"):
        try:
            import torch
        except ImportError:
            need("torch", "pytorch")
        self.torch = torch
        self.device = resolve_torch_device(device)
        self.model = torch.jit.load(source, map_location=self.device).eval()
        self.processor = DetectionProcessor(self.manifest)
        return self

    def preprocess(self, image):
        tensor, trace = self.processor(image)
        return self.torch.from_numpy(tensor).to(self.device), trace

    def forward(self, context):
        tensor, trace = context
        return self.model(tensor), trace

    def postprocess(self, raw, image_size):
        raw, trace = raw
        a = raw.detach().cpu().numpy() if hasattr(raw, "detach") else raw
        d = self.manifest.decoder
        result = (
            yolo_end2end(a, trace, d.score_threshold)
            if d.kind == "yolo_end2end"
            else yolo_raw(a, trace, d.score_threshold, d.iou_threshold)
        )
        result.metadata.update(
            {
                "backend": "pytorch",
                "provider": "torch",
                "device": self.device,
                "precision": {"float32": "fp32", "float16": "fp16", "uint8": "int8"}[self.manifest.input.dtype],
                "input_shape": list(self.manifest.input.shape),
            }
        )
        return result

    def export_onnx(self, output, **kwargs):
        self.torch.onnx.export(self.model, self.torch.zeros((1, 3, 640, 640)), output, input_names=["images"])
        return output
