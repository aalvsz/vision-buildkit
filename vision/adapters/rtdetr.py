from __future__ import annotations

from .base import need, resolve_torch_device
from .ultralytics import UltralyticsAdapter


class RTDetrAdapter(UltralyticsAdapter):
    """Ultralytics RT-DETR native adapter with the official loader."""

    def load(self, source, device="auto"):
        try:
            from ultralytics import RTDETR
        except ImportError:
            need("ultralytics", "yolo")
        self.model = RTDETR(source)
        self.device = resolve_torch_device(device)
        return self
