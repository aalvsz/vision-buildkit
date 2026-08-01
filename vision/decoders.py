from __future__ import annotations

import numpy as np

from .geometry import cxcywh_to_xyxy, nms
from .processing import GeometryTrace
from .types import Box, DetectionResult


def yolo_end2end(a, trace: GeometryTrace, threshold=0.25):
    a = np.asarray(a)
    a = a[0] if a.ndim == 3 else a
    rows = [r for r in a if len(r) >= 6 and r[4] > threshold]
    return DetectionResult(
        [trace.inverse(Box(*map(float, r[:4]))) for r in rows],
        [float(r[4]) for r in rows],
        [int(r[5]) for r in rows],
        image_size=trace.original,
    )


def yolo_end2end_contract(a, trace: GeometryTrace, contract, *, image_size=None):
    """Decode an already-NMS YOLO tensor using the certified result semantics."""
    a = np.asarray(a)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise ValueError("yolo_end2end requires a single-image output")
        a = a[0]
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError("yolo_end2end output must be [N,6] or [1,N,6]")
    if not np.isfinite(a).all():
        raise ValueError("yolo_end2end rows require finite boxes, scores, and classes")
    classes = a[:, 5]
    if not np.all(classes == np.floor(classes)) or np.any(classes < 0) or np.any(classes >= len(contract.labels)):
        raise ValueError("yolo_end2end classes must be integral contract label indices")
    # Python's stable sort retains source row order for equal scores.
    rows = [(index, row) for index, row in enumerate(a) if float(row[4]) > contract.result.score_threshold]
    rows.sort(key=lambda item: -float(item[1][4]))
    effective_limit = min(contract.result.top_k or contract.result.max_detections, contract.result.max_detections)
    rows = rows[:effective_limit]
    return DetectionResult(
        [trace.inverse(Box(*map(float, row[:4]))) for _, row in rows],
        [float(row[4]) for _, row in rows],
        [int(row[5]) for _, row in rows],
        {index: label for index, label in enumerate(contract.labels)},
        image_size=image_size or trace.original,
    )


def yolo_raw(a, trace: GeometryTrace, threshold=0.25, iou_threshold=0.45):
    a = np.asarray(a)
    a = a[0] if a.ndim == 3 else a
    if a.ndim != 2 or max(a.shape) < 5:
        raise ValueError("raw YOLO must be [4+C,N] or [N,4+C] with at least one class")
    # Feature axis is normally the small axis. A tiny [F,N] fixture is still unambiguous if N<5.
    if a.shape[0] < a.shape[1] or a.shape[1] < 5:
        a = a.T
    boxes = []
    scores = []
    classes = []
    for r in a:
        cls = int(np.argmax(r[4:]))
        score = float(r[4 + cls])
        if score > threshold:
            boxes.append(trace.inverse(cxcywh_to_xyxy(*map(float, r[:4]))))
            scores.append(score)
            classes.append(cls)
    keep = nms(boxes, scores, classes, iou_threshold)
    return DetectionResult(
        [boxes[i] for i in keep], [scores[i] for i in keep], [classes[i] for i in keep], image_size=trace.original
    )


def detr(boxes, logits=None, scores=None, labels=None, image_size=(1, 1), top_k=300):
    b = np.asarray(boxes)
    b = b[0] if b.ndim == 3 else b
    w, h = image_size
    if scores is None:
        logits = np.asarray(logits)
        logits = logits[0] if logits.ndim == 3 else logits
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        labels = probs[:, :-1].argmax(-1)
        scores = probs[:, :-1].max(-1)
    chosen = np.argsort(np.asarray(scores))[::-1][:top_k]
    out = [
        Box(*(v for v in ((r[0] - r[2] / 2) * w, (r[1] - r[3] / 2) * h, (r[0] + r[2] / 2) * w, (r[1] + r[3] / 2) * h)))
        for r in b[chosen]
    ]
    return DetectionResult(
        out,
        [float(np.asarray(scores)[i]) for i in chosen],
        [int(np.asarray(labels)[i]) for i in chosen],
        image_size=image_size,
    )


def rtdetr_combined(a, trace: GeometryTrace, threshold=0.25):
    a = np.asarray(a)
    a = a[0] if a.ndim == 3 else a
    if a.ndim != 2 or a.shape[1] < 6:
        raise ValueError("RT-DETR combined output must be [N,6]")
    tw, th = trace.target
    rows = [r for r in a if r[4] > threshold]
    boxes = [trace.inverse(cxcywh_to_xyxy(r[0] * tw, r[1] * th, r[2] * tw, r[3] * th)) for r in rows]
    return DetectionResult(
        boxes,
        [float(r[4]) for r in rows],
        [int(r[5]) for r in rows],
        image_size=trace.original,
    )


def rfdetr_output(raw, image_size, top_k=300, threshold=0.25):
    if hasattr(raw, "xyxy"):
        return DetectionResult(
            [Box(*map(float, x)) for x in raw.xyxy],
            list(map(float, raw.confidence)),
            list(map(int, raw.class_id)),
            image_size=image_size,
        )
    if isinstance(raw, dict):
        if "boxes" not in raw or "logits" not in raw:
            raise ValueError("RF-DETR output map requires named boxes and logits tensors")
        boxes = np.asarray(raw["boxes"])
        boxes = boxes[0] if boxes.ndim == 3 else boxes
        logits = np.asarray(raw["logits"])
        logits = logits[0] if logits.ndim == 3 else logits
        # Official exported RF-DETR uses independent sigmoid class logits.  Ranking
        # every query/class pair globally is intentionally not softmax DETR logic.
        probabilities = 1 / (1 + np.exp(-logits))
        flat = np.argsort(probabilities.ravel())[::-1][:top_k]
        width, height = image_size
        out_boxes = []
        scores = []
        classes = []
        for index in flat:
            query, label = np.unravel_index(index, probabilities.shape)
            score = float(probabilities[query, label])
            if score <= threshold:
                continue
            cx, cy, bw, bh = boxes[query]
            out_boxes.append(
                Box((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
            )
            scores.append(score)
            classes.append(int(label))
        return DetectionResult(out_boxes, scores, classes, image_size=image_size)
    raise ValueError("unsupported RF-DETR output; expected supervision Detections or output map")
