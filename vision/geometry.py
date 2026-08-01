from __future__ import annotations

from .types import Box


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Box:
    return Box(x, y, x + w, y + h)


def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Box:
    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a: Box, b: Box) -> float:
    intersection = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    union = a.area() + b.area() - intersection
    return intersection / union if union else 0.0


def letterbox_params(
    original: tuple[int, int],
    target: tuple[int, int],
) -> tuple[float, float, float]:
    original_width, original_height = original
    target_width, target_height = target
    scale = min(target_width / original_width, target_height / original_height)
    return (
        scale,
        (target_width - original_width * scale) / 2,
        (target_height - original_height * scale) / 2,
    )


def undo_letterbox(
    box: Box,
    original: tuple[int, int],
    target: tuple[int, int],
) -> Box:
    scale, pad_x, pad_y = letterbox_params(original, target)
    return Box(
        (box.x1 - pad_x) / scale,
        (box.y1 - pad_y) / scale,
        (box.x2 - pad_x) / scale,
        (box.y2 - pad_y) / scale,
    )


def nms(
    boxes: list[Box],
    scores: list[float],
    classes: list[int],
    threshold: float = 0.5,
) -> list[int]:
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError("NMS boxes, scores and classes must have equal length")
    if not 0 <= threshold <= 1:
        raise ValueError("NMS threshold must be between zero and one")
    kept: list[int] = []
    for index in sorted(range(len(boxes)), key=scores.__getitem__, reverse=True):
        if all(classes[index] != classes[other] or iou(boxes[index], boxes[other]) < threshold for other in kept):
            kept.append(index)
    return kept
