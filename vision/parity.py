"""Legacy single-image parity API backed by :mod:`vision.validation`."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator

from .types import DetectionResult
from .validation import ValidationPolicy, compare_detections


class ValidationReport(BaseModel):
    """Read-only compatibility parser for pre-dataset capsule reports.

    New reports use ``vision.validation.ValidationReport``; this intentionally
    small parser only protects old capsule documents from forged ``passed``.
    """

    model_config = ConfigDict(extra="forbid")
    reference_id: str
    candidate_id: str
    reference_digest: str
    candidate_digest: str
    parity: dict
    gates: dict[str, bool]
    passed: bool
    map_reference: float | None = None
    map_candidate: float | None = None
    max_map_drop: float | None = None

    @field_validator("map_reference", "map_candidate")
    @classmethod
    def finite_map(cls, value):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("mAP must be finite in [0,1]")
        return value

    def verify(self) -> bool:
        if self.passed:
            raise ValueError("legacy validation reports are not independently verifiable")
        if not all((self.reference_id, self.candidate_id, self.reference_digest, self.candidate_digest)):
            raise ValueError("report identities are required")
        if self.max_map_drop is not None:
            if self.map_reference is None or self.map_candidate is None:
                raise ValueError("mAP policy requires both measurements")
            expected = self.map_reference - self.map_candidate <= self.max_map_drop and bool(self.gates.get("parity"))
        else:
            expected = bool(self.gates.get("parity"))
        if self.passed != expected:
            raise ValueError("forged/inconsistent passed gate")
        return True


@dataclass(frozen=True)
class ParityReport:
    data: dict

    def to_dict(self) -> dict:
        return self.data

    def __getattr__(self, name):
        if name == "class_mismatches":
            return [(item["reference"], item["candidate"]) for item in self.data[name]]
        return self.data[name]


def compare(
    reference: DetectionResult,
    candidate: DetectionResult,
    iou_threshold: float = 0.5,
    score_tolerance: float = 0.05,
    high_confidence: float = 0.5,
) -> ParityReport:
    policy = ValidationPolicy(
        max_map_drop=1, iou_threshold=iou_threshold, score_delta=score_tolerance, high_confidence=high_confidence
    )
    data = compare_detections(reference, candidate, policy)
    data["passed"] = (
        not data["high_conf_lost"]
        and not data["high_conf_added"]
        and not data["class_mismatches"]
        and data["max_score_delta"] <= score_tolerance
    )
    return ParityReport(data)


def gate_map_drop(reference_map: float, candidate_map: float, max_drop: float) -> dict:
    if not 0 <= reference_map <= 1 or not 0 <= candidate_map <= 1 or max_drop < 0:
        raise ValueError("mAP values must be in [0,1] and max drop non-negative")
    return {
        "reference_map": reference_map,
        "candidate_map": candidate_map,
        "map_drop": reference_map - candidate_map,
        "max_map_drop": max_drop,
        "passed": reference_map - candidate_map <= max_drop,
    }
