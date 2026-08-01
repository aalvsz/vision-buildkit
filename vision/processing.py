from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .types import Box


@dataclass(frozen=True)
class GeometryTrace:
    original: tuple[int, int]
    target: tuple[int, int]
    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0

    @property
    def scale(self) -> float:
        """Compatibility accessor; valid only for aspect-preserving letterbox."""
        return self.scale_x

    def inverse(self, box: Box) -> Box:
        width, height = self.original
        coords = (
            (box.x1 - self.pad_x) / self.scale_x,
            (box.y1 - self.pad_y) / self.scale_y,
            (box.x2 - self.pad_x) / self.scale_x,
            (box.y2 - self.pad_y) / self.scale_y,
        )
        return Box(
            max(0, min(width, coords[0])),
            max(0, min(height, coords[1])),
            max(0, min(width, coords[2])),
            max(0, min(height, coords[3])),
        )


def letterbox(image: Image.Image, target: tuple[int, int], padding: int = 114) -> tuple[np.ndarray, GeometryTrace]:
    """Ultralytics LetterBox(new_shape, auto=False, center=True) byte-exact geometry."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("ultralytics_letterbox_cv2 requires opencv-python") from error
    image = image.convert("RGB")
    original_width, original_height = image.size
    target_width, target_height = target
    scale = min(target_width / original_width, target_height / original_height)
    resized_width, resized_height = round(original_width * scale), round(original_height * scale)
    pad_x, pad_y = (target_width - resized_width) / 2, (target_height - resized_height) / 2
    resized = cv2.resize(np.asarray(image), (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    # Ultralytics rounds asymmetric odd padding with +/- 0.1, not Python round().
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(padding,) * 3)
    trace = GeometryTrace((original_width, original_height), target, scale, scale, left, top)
    return canvas, trace


def libreyolo_yolo9_letterbox(
    image: Image.Image, target: tuple[int, int], padding: int = 114
) -> tuple[np.ndarray, GeometryTrace]:
    """LibreYOLO 1.4.0 YOLO9 ``preprocess_numpy`` geometry, byte for byte.

    The provider uses ``int(original * ratio)`` (truncation) and writes the
    resized RGB image at ``[:new_h, :new_w]`` on a 114-filled canvas.  In
    particular, it does *not* center the residual padding.  Keep this separate
    from the Ultralytics-compatible centered letterbox above: the inverse map
    differs on every non-square image.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("libreyolo_yolo9_top_left_cv2 requires opencv-python") from error
    image = image.convert("RGB")
    original_width, original_height = image.size
    target_width, target_height = target
    ratio = min(target_height / original_height, target_width / original_width)
    resized_height, resized_width = max(int(original_height * ratio), 1), max(int(original_width * ratio), 1)
    resized = cv2.resize(np.asarray(image), (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), padding, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    return canvas, GeometryTrace((original_width, original_height), target, ratio, ratio, 0.0, 0.0)


def direct_resize(image: Image.Image, target: tuple[int, int]) -> tuple[np.ndarray, GeometryTrace]:
    image = image.convert("RGB")
    original = image.size
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv_linear_scale_fill requires opencv-python") from error
    result = cv2.resize(np.asarray(image), target, interpolation=cv2.INTER_LINEAR)
    return result, GeometryTrace(original, target, target[0] / original[0], target[1] / original[1])
