from __future__ import annotations

from pathlib import Path

from ..types import Box, DetectionResult
from .base import ModelAdapter, need, resolve_torch_device


class UltralyticsAdapter(ModelAdapter):
    def runtime_identity(self):
        return {
            "backend": "pytorch",
            "provider": "ultralytics-torch",
            "device": self.device,
            "precision": "fp32",
            "input_shape": list(self.manifest.input.shape),
        }

    def load(self, source, device="auto"):
        try:
            from ultralytics import YOLO
        except ImportError:
            need("ultralytics", "yolo")
        self.model = YOLO(source)
        self.device = resolve_torch_device(device)
        return self

    def preprocess(self, image):
        return image

    def forward(self, tensor):
        height, width = (int(value) for value in self.manifest.input.shape[-2:])
        if height != width:
            raise ValueError("Ultralytics native adapter currently requires square manifest input")
        return self.model.predict(
            tensor,
            device=self.device,
            verbose=False,
            imgsz=height,
            conf=self.manifest.decoder.score_threshold,
            rect=False,
        )[0]

    def postprocess(self, raw, image_size):
        b = raw.boxes
        return DetectionResult(
            [Box(*map(float, x)) for x in b.xyxy.cpu().tolist()],
            b.conf.cpu().tolist(),
            [int(x) for x in b.cls.cpu().tolist()],
            dict(raw.names),
            image_size,
            metadata={
                "backend": "pytorch",
                "provider": "ultralytics-torch",
                "device": self.device,
                "precision": "fp32",
                "input_shape": list(self.manifest.input.shape),
                "requested_device": self.device,
                "effective_device": self.device,
            },
        )

    def export_onnx(self, output, **kwargs):
        height, width = (int(value) for value in self.manifest.input.shape[-2:])
        if height != width:
            raise ValueError("Ultralytics ONNX export currently requires square manifest input")
        options = {
            "imgsz": height,
            "batch": int(self.manifest.input.shape[0]),
            "dynamic": False,
            "verbose": False,
        }
        options.update(kwargs)
        p = Path(self.model.export(format="onnx", **options))
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(p, out)
        return str(out)
