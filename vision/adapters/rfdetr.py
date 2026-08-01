from __future__ import annotations

from pathlib import Path

from ..decoders import rfdetr_output
from .base import ModelAdapter, need, resolve_torch_device


class RfDetrAdapter(ModelAdapter):
    def runtime_identity(self):
        return {
            "backend": "pytorch",
            "provider": "rfdetr-torch",
            "device": self.device,
            "precision": "fp32",
            "input_shape": list(self.manifest.input.shape),
        }

    def load(self, source, device="cpu"):
        try:
            import rfdetr
        except ImportError:
            need("rfdetr", "rfdetr")
        variant = self.manifest.variant or "nano"
        factory = getattr(
            rfdetr,
            {
                "nano": "RFDETRNano",
                "small": "RFDETRSmall",
                "medium": "RFDETRMedium",
                "base": "RFDETRBase",
                "large": "RFDETRLarge",
            }.get(variant, ""),
            None,
        )
        if not factory:
            raise RuntimeError(f"official RF-DETR {variant} factory unavailable")
        if device not in ("cpu", "auto", "cuda", "mps"):
            raise ValueError("RF-DETR device must be cpu/auto/cuda/mps")
        selected = resolve_torch_device(device)
        official_aliases = {f"rfdetr-{variant}", f"rf-detr-{variant}"}
        if source.lower() in official_aliases:
            kw = {}
        elif Path(source).is_file():
            kw = {"pretrain_weights": source}
        else:
            raise FileNotFoundError(
                f"RF-DETR source must be an official {variant} alias or an explicit existing checkpoint: {source}"
            )
        self.model = factory(device=selected, **kw)
        self.device = selected
        return self

    def preprocess(self, image):
        return image

    def forward(self, image):
        height, width = (int(value) for value in self.manifest.input.shape[-2:])
        return self.model.predict(image, shape=(height, width), threshold=self.manifest.decoder.score_threshold)

    def postprocess(self, raw, image_size):
        result = rfdetr_output(raw, image_size, self.manifest.decoder.top_k, self.manifest.decoder.score_threshold)
        result.metadata.update(
            {
                "backend": "pytorch",
                "provider": "rfdetr-torch",
                "device": self.device,
                "precision": "fp32",
                "input_shape": list(self.manifest.input.shape),
            }
        )
        return result

    def export_onnx(self, output, **kwargs):
        height, width = (int(value) for value in self.manifest.input.shape[-2:])
        result = self.model.export(
            output_dir=str(Path(output).parent),
            format="onnx",
            shape=kwargs.get("shape", (height, width)),
            batch_size=1,
            opset_version=17,
            fp16=kwargs.get("fp16", False),
            verbose=bool(kwargs.get("verbose", False)),
        )
        p = Path(result) if result else next(Path(output).parent.glob("*.onnx"), None)
        if not p or not p.is_file():
            raise RuntimeError("RF-DETR export produced no ONNX artifact")
        import shutil

        shutil.copy2(p, output)
        return str(output)
