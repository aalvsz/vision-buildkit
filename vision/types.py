from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Box:
    """Exclusive xyxy pixel box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class DetectionResult:
    boxes: list[Box] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    classes: list[int] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)
    image_size: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (len(self.boxes) == len(self.scores) == len(self.classes)):
            raise ValueError("boxes, scores, and classes must have equal length")

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": [b.as_xyxy() for b in self.boxes],
            "scores": self.scores,
            "classes": self.classes,
            "names": {str(k): v for k, v in self.names.items()},
            "image_size": self.image_size,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
