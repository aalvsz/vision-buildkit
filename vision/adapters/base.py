from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..manifest import VisionManifest
from ..types import DetectionResult


class OptionalDependencyError(RuntimeError):
    pass


def need(package: str, extra: str):
    raise OptionalDependencyError(f"{package} is required; install vision-buildkit[{extra}]")


def resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelAdapter(ABC):
    def __init__(self, manifest: VisionManifest | None = None):
        self.manifest = manifest

    @abstractmethod
    def load(self, source: str, device: str = "auto"): ...
    @abstractmethod
    def preprocess(self, image: Any): ...
    @abstractmethod
    def forward(self, tensor: Any): ...
    @abstractmethod
    def postprocess(self, raw: Any, image_size: tuple[int, int]) -> DetectionResult: ...
    @abstractmethod
    def export_onnx(self, output: str, **kwargs) -> str: ...
    def runtime_identity(self) -> dict[str, Any]:
        """Trusted session facts used to cross-check emitted result metadata."""
        raise RuntimeError("adapter does not expose a runtime session identity")

    def __call__(self, image):
        from PIL import Image

        im = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        return self.postprocess(self.forward(self.preprocess(im)), im.size)

    def predict_batch(self, images):
        """Correct fallback for native adapters that do not expose tensor batching."""
        return [self(image) for image in images]
