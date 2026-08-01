from ..manifest import VisionManifest
from .backend import BackendAdapter as BackendAdapter
from .base import ModelAdapter as ModelAdapter
from .base import OptionalDependencyError as OptionalDependencyError
from .onnx import OnnxAdapter
from .rfdetr import RfDetrAdapter
from .rtdetr import RTDetrAdapter
from .torch import TorchAdapter
from .ultralytics import UltralyticsAdapter


def resolve_adapter(source: str, manifest: VisionManifest | None = None, family: str | None = None):
    f = family or (manifest.family if manifest else None)
    if not f:
        raise ValueError("family or manifest is required; source is never silently classified")
    return {
        "onnx": OnnxAdapter,
        "torch": TorchAdapter,
        "yolo": UltralyticsAdapter,
        "yolo26": UltralyticsAdapter,
        "rtdetr": RTDetrAdapter,
        "rfdetr": RfDetrAdapter,
    }[f](manifest)
