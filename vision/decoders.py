from __future__ import annotations

import numpy as np

from .geometry import cxcywh_to_xyxy, nms
from .processing import GeometryTrace
from .types import Box, DetectionResult

RTDETR_COMBINED_ABI = "rtdetr_combined.v1"


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
    """Legacy 0.1.0 convenience decoder; preserve its public behavior exactly."""
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


def rtdetr_combined_v1(
    a,
    trace: GeometryTrace,
    threshold=0.25,
    *,
    class_count=80,
    labels=None,
    clip_boxes=True,
    max_detections=300,
):
    """Decode the frozen Ultralytics RT-DETR combined-output ABI.

    This is deliberately stricter than the historical convenience decoder:
    certification accepts only the graph-proven ``float32 [1,300,6]`` tensor.
    The rows are normalized ``cx, cy, w, h, score, class`` and are NMS-free.
    """
    if not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != (1, 300, 6):
        raise ValueError("RT-DETR combined output must be exact np.float32 [1,300,6]")
    if not np.isfinite(a).all():
        raise ValueError("RT-DETR combined output must be finite")
    if not isinstance(class_count, int) or isinstance(class_count, bool) or class_count <= 0:
        raise ValueError("RT-DETR class_count must be a positive integer")
    if not isinstance(max_detections, int) or isinstance(max_detections, bool) or max_detections <= 0:
        raise ValueError("RT-DETR max_detections must be a positive integer")
    if not isinstance(clip_boxes, bool):
        raise ValueError("RT-DETR clip_boxes must be boolean")
    if labels is not None and (
        not isinstance(labels, (list, tuple))
        or len(labels) != class_count
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        raise ValueError("RT-DETR labels must exactly cover every class")
    a = a[0]
    normalized = a[:, :5]
    if np.any(normalized < 0) or np.any(normalized > 1):
        raise ValueError("RT-DETR normalized boxes and scores must be in [0,1]")
    classes = a[:, 5]
    if np.any(classes != np.floor(classes)) or np.any(classes < 0) or np.any(classes >= class_count):
        raise ValueError("RT-DETR classes must be integral in-range indices")
    tw, th = trace.target
    rows = [r for r in a if r[4] >= threshold]
    rows.sort(key=lambda r: (-float(r[4]), int(r[5]), *(float(value) for value in r[:4])))
    rows = rows[:max_detections]
    boxes = [
        trace.inverse(
            cxcywh_to_xyxy(
                float(row[0]) * tw,
                float(row[1]) * th,
                float(row[2]) * tw,
                float(row[3]) * th,
            ),
            clip=clip_boxes,
        )
        for row in rows
    ]
    return DetectionResult(
        boxes,
        [float(r[4]) for r in rows],
        [int(r[5]) for r in rows],
        {index: label for index, label in enumerate(labels or [str(index) for index in range(class_count)])},
        image_size=trace.original,
    )


def rtdetr_combined_v1_contract(a, trace: GeometryTrace, contract, *, image_size=None):
    """Decode the exact certified RT-DETR ABI using all result semantics."""
    tensor_specs = tuple((item.name, item.dtype, item.shape) for item in contract.raw_output.tensors)
    if contract.raw_output.semantics != "rtdetr_combined" or tensor_specs != (("output0", "float32", (1, 300, 6)),):
        raise ValueError("RT-DETR contract requires exact output0 float32 [1,300,6] ABI")
    result = contract.result
    if (
        result.box_format != "xyxy"
        or result.coordinate_space != "original_image"
        or result.nms_owner != "none"
        or result.suppression != "none"
        or result.ordering != "score_desc"
        or result.tie_policy != "class_then_box"
        or result.top_k != 300
        or result.max_detections != 300
    ):
        raise ValueError("RT-DETR result contract contradicts frozen NMS-free output semantics")
    decoded = rtdetr_combined_v1(
        a,
        trace,
        result.score_threshold,
        class_count=len(contract.labels),
        labels=contract.labels,
        clip_boxes=result.clip_boxes,
        max_detections=min(result.top_k, result.max_detections),
    )
    decoded.image_size = image_size or trace.original
    decoded.metadata["decoder_abi"] = RTDETR_COMBINED_ABI
    return decoded


RFDETR_COCO_CATEGORY_IDS = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
)
RFDETR_FORBIDDEN_SPARSE_SLOTS = frozenset({0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83})
RFDETR_COCO_LABELS = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def rfdetr_contract_output(raw, image_size, contract):
    """Decode the frozen RF-DETR ONNX ABI without provider-specific behavior."""
    if not isinstance(raw, dict) or set(raw) != {"dets", "labels"}:
        raise ValueError("RF-DETR raw output must be exactly named dets and labels tensors")
    boxes, logits = np.asarray(raw["dets"]), np.asarray(raw["labels"])
    if boxes.dtype != np.dtype("float32") or logits.dtype != np.dtype("float32"):
        raise ValueError("RF-DETR dets and labels must be float32 tensors")
    if boxes.shape != (1, 300, 4) or logits.shape != (1, 300, 91):
        raise ValueError("RF-DETR raw ABI requires dets [1,300,4] and labels [1,300,91]")
    if not np.isfinite(boxes).all() or not np.isfinite(logits).all():
        raise ValueError("RF-DETR raw tensors must be finite")
    if tuple(contract.coco_category_ids) != RFDETR_COCO_CATEGORY_IDS or tuple(contract.labels) != RFDETR_COCO_LABELS:
        raise ValueError("RF-DETR contract must declare the exact 80-class sparse COCO category mapping")
    if (
        contract.result.box_format != "xyxy"
        or contract.result.coordinate_space != "original_image"
        or contract.result.score_threshold != 0.30
        or contract.result.max_detections != 300
    ):
        raise ValueError("RF-DETR contract must pin original-image xyxy results, threshold 0.30, and max 300")
    probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    sparse_to_public = {category: index for index, category in enumerate(contract.coco_category_ids)}
    # Deterministic global ranking: score descending, then source query, then
    # sparse category id.  This makes ties independent of numpy sort details.
    selected = []
    for query in range(300):
        for sparse_id in range(91):
            if sparse_id in RFDETR_FORBIDDEN_SPARSE_SLOTS or sparse_id not in sparse_to_public:
                continue
            score = float(probabilities[0, query, sparse_id])
            if score > contract.result.score_threshold:
                selected.append((-score, query, sparse_id))
    selected.sort()
    width, height = image_size
    out_boxes, scores, classes = [], [], []
    for negative_score, query, sparse_id in selected[:300]:
        cx, cy, bw, bh = (float(value) for value in boxes[0, query])
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        out_boxes.append(
            Box(
                max(0.0, min(width, x1)), max(0.0, min(height, y1)), max(0.0, min(width, x2)), max(0.0, min(height, y2))
            )
        )
        scores.append(-negative_score)
        classes.append(sparse_to_public[sparse_id])
    return DetectionResult(
        out_boxes, scores, classes, {i: name for i, name in enumerate(contract.labels)}, image_size=image_size
    )


def rfdetr_output(raw, image_size, top_k=300, threshold=0.25, *, contract=None):
    if contract is not None:
        if isinstance(raw, dict) and set(raw) == {"boxes", "logits"}:
            raw = {"dets": raw["boxes"], "labels": raw["logits"]}
        return rfdetr_contract_output(raw, image_size, contract)
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
