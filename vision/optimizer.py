"""Compatibility boundary for the former dict-based optimizer API.

New callers must use :mod:`vision.optimization`; accepting arbitrary callback
dictionaries would permit a forged quality claim, so the legacy API is kept
only as an explicit failing migration aid.
"""

from __future__ import annotations

from dataclasses import dataclass

from .optimization import SearchRequest, optimize_search


@dataclass
class Candidate:
    backend: str
    precision: str = "fp32"
    resolution: tuple[int, int] = (640, 640)
    batch: int = 1
    dynamic: bool = False


@dataclass
class CandidateResult:
    candidate: Candidate
    accepted: bool
    reason: str
    benchmark: dict | None = None


def optimize_candidates(candidates: list[Candidate], evaluator, require_validation: bool = True):
    raise RuntimeError(
        "legacy optimize_candidates is fail-closed; use vision.optimize_search(SearchRequest, evaluator) with "
        "signed ValidationEvidence and BenchmarkReport"
    )


__all__ = ["Candidate", "CandidateResult", "SearchRequest", "optimize_candidates", "optimize_search"]
