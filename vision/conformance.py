# ruff: noqa: E501
"""File-backed, content-bound semantic conformance.

Reports are attestations over concrete evidence files; callers cannot turn a
collection of anonymous hashes into a PASS report.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .contract import Hex, Strict, VisionContract
from .plugins import PreparedArtifact, PreparedSource
from .types import DetectionResult

Gate = Literal["PASS", "FAIL", "UNTESTED"]
_OPERATIONAL_EXCEPTIONS = (ValueError, OSError, RuntimeError, ImportError)


def _operational_reason(exc: BaseException) -> str:
    """Retain a non-empty deterministic reason for ordinary operational errors."""
    return str(exc) or f"{type(exc).__name__} raised without a message"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _finite_json(value: Any) -> Any:
    """Reject non-finite numbers at every evidence JSON boundary.

    Python's decoder accepts ``NaN`` and infinities by default, and a large
    exponent can become ``inf`` without using one of those spellings.  Evidence
    is security-sensitive input, so accept only normal JSON scalar types and
    finite numeric values recursively.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evidence JSON contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)
    return value


def _load_evidence_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid evidence JSON: {path.name}") from exc
    return _finite_json(value)


def _canonical_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif isinstance(value, dict):
        value = {key: _jsonable(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_jsonable(item) for item in value]
    _finite_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class ConformancePolicy(Strict):
    """The complete, closed threshold policy for certificate evidence."""

    metric_name: str = Field(min_length=1)
    maximum_drop: float
    golden_parity_lost: int
    golden_parity_added: int
    mean_iou_min: float
    min_iou_min: float
    max_score_delta: float

    @field_validator("maximum_drop", "mean_iou_min", "min_iou_min", "max_score_delta", mode="before")
    @classmethod
    def _finite_threshold(cls, value: Any, info) -> float:
        return _finite_number(value, info.field_name)

    @field_validator("golden_parity_lost", "golden_parity_added", mode="before")
    @classmethod
    def _integer_threshold(cls, value: Any, info) -> int:
        return _nonnegative_int(value, info.field_name)

    @model_validator(mode="after")
    def _ranges(self):
        if self.maximum_drop < 0 or self.max_score_delta < 0:
            raise ValueError("drop and score-delta thresholds must be non-negative")
        if not 0 <= self.mean_iou_min <= 1 or not 0 <= self.min_iou_min <= 1:
            raise ValueError("IoU thresholds must be in [0, 1]")
        return self


def digest_path(path: str | Path) -> str:
    item = Path(path)
    if item.is_symlink() or not item.is_file():
        raise ValueError("evidence must be an existing regular non-symlink file")
    return hashlib.sha256(item.read_bytes()).hexdigest()


class EvidenceFile(Strict):
    path: str = Field(min_length=1)
    sha256: Hex

    def verify(self, base: str | Path | None = None) -> Path:
        declared = Path(self.path)
        if base is not None:
            # Packaged evidence is deliberately relocatable.  An absolute
            # pathname (or a traversal) would make a certified capsule depend
            # on the machine that produced it.
            root = Path(base).absolute()
            if declared.is_absolute() or "\\" in self.path or ".." in declared.parts:
                raise ValueError("packaged evidence path must be capsule-relative")
            path = root / declared
            try:
                path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError("packaged evidence escapes capsule root") from exc
        else:
            path = declared
        if path.is_symlink() or not path.is_file() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError("evidence must be a regular non-symlink file")
        if digest_path(path) != self.sha256:
            raise ValueError(f"evidence digest mismatch: {self.path}")
        return path


class DatasetImage(Strict):
    image_id: int
    file_name: str = Field(min_length=1)
    sha256: Hex


class DatasetManifest(Strict):
    """COCO manifest plus hashes for every byte executed by certification."""

    images: tuple[DatasetImage, ...] = Field(min_length=1)

    @classmethod
    def from_coco(cls, path: Path) -> DatasetManifest:
        raw = _load_evidence_json(path)
        if not isinstance(raw, dict):
            raise ValueError("dataset manifest must be a JSON object")
        root = path.parent
        images = []
        for image in raw.get("images", []):
            name = image["file_name"]
            # The hash lives in the declared COCO image record: no unnamed
            # directory walk can alter which material was evaluated.
            digest = image.get("sha256")
            if not isinstance(digest, str):
                raise ValueError("dataset image records require sha256")
            item = root / name
            if digest_path(item) != digest:
                raise ValueError(f"dataset image digest mismatch: {name}")
            images.append(DatasetImage(image_id=image["id"], file_name=name, sha256=digest))
        return cls(images=tuple(images))


class GoldenCase(Strict):
    case_id: str = Field(min_length=1)
    sha256: Hex


class GoldenCorpus(Strict):
    cases: tuple[GoldenCase, ...] = Field(min_length=1)
    manifest_digest: Hex


class AccuracyEvidence(Strict):
    dataset_digest: Hex
    prediction_digest: Hex
    report_digest: Hex
    metric_name: str = Field(min_length=1)
    metric_value: float
    evaluator_id: Literal["coco-map"] = "coco-map"
    evaluator_version: Literal["1"] = "1"
    evaluator_digest: Hex
    parity_schema_version: Literal[1, 2] = 1

    @field_validator("metric_value", mode="before")
    @classmethod
    def _metric_is_finite(cls, value: Any) -> float:
        return _finite_number(value, "metric_value")

    @field_validator("parity_schema_version", mode="before")
    @classmethod
    def _parity_schema_is_exact_int(cls, value: Any) -> int:
        if type(value) is not int or value not in (1, 2):
            raise ValueError("parity_schema_version must be exact int 1 or 2")
        return value

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Keep the serialized shape of schema-v1 accuracy attestations exact."""
        result = super().model_dump(*args, **kwargs)
        if self.parity_schema_version == 1:
            result.pop("parity_schema_version", None)
        return result


class RuntimeEvidence(Strict):
    command: tuple[str, ...] = Field(min_length=1)
    hardware: dict[str, str] = Field(min_length=1)
    environment: dict[str, str] = Field(min_length=1)
    source_digest: Hex
    artifact_digest: Hex
    source_plugin_version: str = Field(min_length=1)
    target_plugin_version: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    device: str = Field(min_length=1)
    precision: Literal["fp32", "fp16", "int8"]
    resolution: tuple[int, int]
    batch: int = Field(gt=0)
    dynamic: bool
    runtime_version: str = Field(min_length=1)
    p50_ms: float | None = Field(default=None, gt=0)
    p95_ms: float | None = Field(default=None, gt=0)

    @field_validator("p50_ms", "p95_ms", mode="before")
    @classmethod
    def _timing_is_finite(cls, value: Any, info) -> float | None:
        if value is None:
            return None
        return _finite_number(value, info.field_name)

    @model_validator(mode="after")
    def _timing_is_complete(self):
        if (self.p50_ms is None) != (self.p95_ms is None):
            raise ValueError("runtime timing requires both p50_ms and p95_ms")
        if self.p50_ms is not None and self.p95_ms < self.p50_ms:
            raise ValueError("runtime p95_ms must be at least p50_ms")
        return self


class ConformanceEvidenceBundle(Strict):
    golden_manifest: EvidenceFile
    case_inputs: tuple[EvidenceFile, ...] = Field(min_length=1)
    dataset_manifest: EvidenceFile
    policy: EvidenceFile
    predictions: EvidenceFile
    source_predictions: EvidenceFile
    accuracy_report: EvidenceFile
    runtime_record: EvidenceFile
    environment_record: EvidenceFile

    def verify(
        self,
        golden: GoldenCorpus,
        accuracy: AccuracyEvidence,
        runtime: RuntimeEvidence,
        *,
        source_plugin: str,
        target_plugin: str,
        base: str | Path | None = None,
    ) -> None:
        files = (
            self.golden_manifest,
            *self.case_inputs,
            self.dataset_manifest,
            self.policy,
            self.predictions,
            self.source_predictions,
            self.accuracy_report,
            self.runtime_record,
            self.environment_record,
        )
        for file in files:
            file.verify(base)
        if self.golden_manifest.sha256 != golden.manifest_digest or len(self.case_inputs) != len(golden.cases):
            raise ValueError("golden manifest or case cardinality mismatch")
        manifest = _load_evidence_json(self.golden_manifest.verify(base))
        if not isinstance(manifest, dict):
            raise ValueError("golden manifest must be a JSON object")
        declared = {x["case_id"]: x["sha256"] for x in manifest.get("cases", [])}
        if declared != {x.case_id: x.sha256 for x in golden.cases}:
            raise ValueError("golden manifest membership mismatch")
        if {x.sha256 for x in self.case_inputs} != set(declared.values()):
            raise ValueError("golden case input bytes mismatch")
        if (self.dataset_manifest.sha256, self.predictions.sha256, self.accuracy_report.sha256) != (
            accuracy.dataset_digest,
            accuracy.prediction_digest,
            accuracy.report_digest,
        ):
            raise ValueError("accuracy evidence digest mismatch")
        policy = ConformancePolicy.model_validate(_load_evidence_json(self.policy.verify(base)))
        metrics = _load_evidence_json(self.accuracy_report.verify(base))
        if not isinstance(metrics, dict):
            raise ValueError("accuracy report must be a JSON object")
        marker = metrics.get("parity_schema_version")
        if marker is None:
            if accuracy.parity_schema_version != 1:
                raise ValueError("v2 prediction parity schema marker is required")
            parity_schema_version = 1
        else:
            if type(marker) is not int or marker not in (1, 2):
                raise ValueError("parity_schema_version must be exact int 1 or 2")
            parity_schema_version = marker
        if parity_schema_version != accuracy.parity_schema_version:
            raise ValueError("accuracy evidence and prediction parity schema mismatch")
        dataset_payload = _load_evidence_json(self.dataset_manifest.verify(base))
        image_ids = _dataset_image_ids(dataset_payload)
        actual_metric = _coco_map(self.dataset_manifest.verify(base), self.predictions.verify(base))
        source_path = self.source_predictions.verify(base)
        _verify_source_prediction_binding(metrics, self.source_predictions, self.accuracy_report, base=base)
        actual_source_metric = _coco_map(self.dataset_manifest.verify(base), source_path)
        if (
            metrics.get("metric_name") != accuracy.metric_name
            or _finite_number(metrics.get("metric_value"), "metric_value") != accuracy.metric_value
            or abs(actual_metric - accuracy.metric_value) > 1e-12
            or accuracy.evaluator_digest != _evaluator_digest()
            or abs(_finite_number(metrics.get("source_metric_value"), "source_metric_value") - actual_source_metric)
            > 1e-12
        ):
            raise ValueError("accuracy report metric mismatch")
        if (
            policy.metric_name != accuracy.metric_name
            or accuracy.metric_value < actual_source_metric - policy.maximum_drop
        ):
            raise ValueError("accuracy policy threshold not met")
        parity = metrics.get("parity")
        expected_parity = _prediction_parity_report(
            source_path, self.predictions.verify(base), image_ids, schema_version=parity_schema_version
        )
        if _canonical_json(parity) != _canonical_json(expected_parity):
            raise ValueError("prediction parity report mismatch")
        diagnostics = metrics.get("parity_diagnostics")
        if parity_schema_version == 2 and diagnostics is None:
            raise ValueError("v2 prediction parity diagnostics are required")
        if diagnostics is not None and _canonical_json(diagnostics) != _canonical_json(
            _prediction_parity_diagnostics(
                source_path, self.predictions.verify(base), image_ids, schema_version=parity_schema_version
            )
        ):
            raise ValueError("prediction parity diagnostics mismatch")
        if not _parity_within_policy(expected_parity, policy):
            raise ValueError("prediction parity cardinality gate failed")
        record = _load_evidence_json(self.runtime_record.verify(base))
        if not isinstance(record, dict):
            raise ValueError("runtime record must be a JSON object")
        expected = {
            "source_digest": runtime.source_digest,
            "artifact_digest": runtime.artifact_digest,
            "source_plugin": source_plugin,
            "target_plugin": target_plugin,
            "source_plugin_version": runtime.source_plugin_version,
            "target_plugin_version": runtime.target_plugin_version,
            "command": list(runtime.command),
            "hardware": runtime.hardware,
            "environment": runtime.environment,
            "backend": runtime.backend,
            "provider": runtime.provider,
            "device": runtime.device,
            "precision": runtime.precision,
            "resolution": list(runtime.resolution),
            "batch": runtime.batch,
            "dynamic": runtime.dynamic,
            "runtime_version": runtime.runtime_version,
            "p50_ms": runtime.p50_ms,
            "p95_ms": runtime.p95_ms,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("runtime record cross-binding mismatch")
        if record.get("environment_digest") != self.environment_record.sha256:
            raise ValueError("runtime environment record mismatch")


def _verify_source_prediction_binding(
    metrics: dict[str, Any], source_predictions: EvidenceFile, accuracy_report: EvidenceFile, *, base: str | Path | None
) -> None:
    """Bind redundant accuracy-report source fields to the declared evidence.

    Before publication, a relative pathname is interpreted relative to the
    accuracy report so existing evidence bundles can be validated.  Inside a
    capsule it must be the exact capsule-relative evidence pathname: this
    prevents a report from silently referring back to the build directory.
    """
    path_value = metrics.get("source_predictions_path")
    digest_value = metrics.get("source_predictions_sha256")
    if not isinstance(path_value, str) or digest_value != source_predictions.sha256:
        raise ValueError("source prediction evidence binding mismatch")
    if base is not None:
        if path_value != source_predictions.path:
            raise ValueError("packaged source prediction path binding mismatch")
        return
    declared = Path(path_value)
    if declared.is_absolute():
        resolved = declared
    else:
        resolved = Path(accuracy_report.path).parent / declared
    try:
        same = resolved.resolve() == Path(source_predictions.path).resolve()
    except OSError:
        same = False
    if not same:
        raise ValueError("source prediction path does not name declared evidence")


def _evaluator_digest() -> str:
    return hashlib.sha256(b"vision.coco-map.v1.pycocotools").hexdigest()


def _coco_map(dataset_path: Path, predictions_path: Path) -> float:
    """Strict small COCO evaluator; empty predictions are always zero."""
    rows = _load_evidence_json(predictions_path)
    if not isinstance(rows, list) or not rows:
        return 0.0
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("coco-map evaluator requires pycocotools") from exc
    coco = COCO(str(dataset_path))
    result = coco.loadRes(rows)
    evaluation = COCOeval(coco, result, "bbox")
    dataset = _load_evidence_json(dataset_path)
    evaluation.params.imgIds = sorted(item["id"] for item in dataset.get("images", []))
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return float(evaluation.stats[0])


def _dataset_image_ids(dataset: Any) -> tuple[int, ...]:
    if not isinstance(dataset, dict) or not isinstance(dataset.get("images"), list):
        raise ValueError("dataset manifest must declare images")
    ids: list[int] = []
    for image in dataset["images"]:
        if not isinstance(image, dict):
            raise ValueError("dataset image record must be an object")
        image_id = image.get("id")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            raise ValueError("dataset image id must be an integer")
        ids.append(image_id)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("dataset image membership must be non-empty and unique")
    return tuple(sorted(ids))


def _prediction_rows(path: Path, image_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = _load_evidence_json(path)
    if not isinstance(rows, list):
        raise ValueError("prediction evidence must be a JSON list")
    expected = set(image_ids)
    checked: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"image_id", "category_id", "bbox", "score"}:
            raise ValueError("prediction rows require exactly image_id/category_id/bbox/score")
        image_id, category_id = row["image_id"], row["category_id"]
        if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id not in expected:
            raise ValueError("prediction image membership does not match dataset")
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise ValueError("prediction category id must be an integer")
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("prediction bbox must have four finite coordinates")
        numeric_bbox = [_finite_number(value, "prediction bbox") for value in bbox]
        if numeric_bbox[2] < 0 or numeric_bbox[3] < 0:
            raise ValueError("prediction bbox width and height must be non-negative")
        score = _finite_number(row["score"], "prediction score")
        checked.append({"image_id": image_id, "category_id": category_id, "bbox": numeric_bbox, "score": score})
    return checked


def _ranked_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministic score rank for every COCO prediction row."""
    return sorted(rows, key=lambda row: (-row["score"], row["category_id"], row["image_id"], row["bbox"]))


def _drift_diagnostic(rows: list[dict[str, Any]], unmatched: set[int]) -> list[dict[str, Any]]:
    """Describe unmatched rows without leaving their signed prediction bytes.

    ``rank`` is one-based within the report view (aggregate, image, or class).
    Keeping the score, class, image and geometry makes a top-k truncation or a
    full output-set substitution visible even when a COCO metric happens not to
    move on the evaluated images.
    """
    return [
        {
            "rank": rank,
            "image_id": row["image_id"],
            "category_id": row["category_id"],
            "score": row["score"],
            "bbox": row["bbox"],
        }
        for rank, row in enumerate(_ranked_rows(rows), start=1)
        if id(row) in unmatched
    ]


def _parity_rows(
    source: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    diagnostics: bool = False,
    zero_overlap_matches: bool = True,
) -> dict[str, Any]:
    """Greedy same-image/class matching with explicit empty-set semantics.

    If both sides are empty, parity is vacuously exact: lost/added are zero,
    IoUs are one, and score drift is zero.  If only one side is empty, IoUs
    are zero and the unmatched records are counted as lost or added.
    """
    used: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    matched_source: set[int] = set()
    ordered_source = sorted(source, key=lambda row: (row["image_id"], row["category_id"], -row["score"], row["bbox"]))
    ordered_candidate = sorted(
        candidate, key=lambda row: (row["image_id"], row["category_id"], -row["score"], row["bbox"])
    )
    for left in ordered_source:
        choices = [
            (index, right)
            for index, right in enumerate(ordered_candidate)
            if index not in used
            and right["image_id"] == left["image_id"]
            and right["category_id"] == left["category_id"]
            and (zero_overlap_matches or _bbox_iou(left["bbox"], right["bbox"]) > 0)
        ]
        if not choices:
            continue
        index, right = max(choices, key=lambda x: (_bbox_iou(left["bbox"], x[1]["bbox"]), -x[0]))
        used.add(index)
        matched_source.add(id(left))
        pairs.append((left, right))
    ious = [_bbox_iou(a["bbox"], b["bbox"]) for a, b in pairs]
    deltas = [abs(a["score"] - b["score"]) for a, b in pairs]
    both_empty = not source and not candidate
    result: dict[str, Any] = {
        "lost": len(source) - len(pairs),
        "added": len(candidate) - len(pairs),
        "mean_iou": sum(ious) / len(ious) if ious else 1.0 if both_empty else 0.0,
        "min_iou": min(ious, default=1.0 if both_empty else 0.0),
        "max_score_delta": max(deltas, default=0.0),
    }
    if diagnostics:
        result["lost_predictions"] = _drift_diagnostic(
            source, {id(row) for row in source if id(row) not in matched_source}
        )
        result["added_predictions"] = _drift_diagnostic(
            candidate, {id(row) for index, row in enumerate(ordered_candidate) if index not in used}
        )
    return result


def _prediction_parity(source_path: Path, candidate_path: Path) -> dict[str, float | int]:
    """Compatibility aggregate helper for callers without a dataset manifest."""
    source = _load_evidence_json(source_path)
    candidate = _load_evidence_json(candidate_path)
    if not isinstance(source, list) or not isinstance(candidate, list):
        raise ValueError("prediction evidence must be JSON lists")
    ids = tuple(sorted({row.get("image_id") for row in source + candidate if isinstance(row, dict)}))
    if not ids:
        return _parity_rows([], [])
    return _parity_rows(_prediction_rows(source_path, ids), _prediction_rows(candidate_path, ids))


def _prediction_parity_report(
    source_path: Path,
    candidate_path: Path,
    image_ids: tuple[int, ...],
    *,
    schema_version: Literal[1, 2] = 1,
) -> dict[str, Any]:
    """Recompute the schema-v1 parity object from the signed prediction bytes.

    Diagnostics intentionally live in ``parity_diagnostics`` rather than this
    signed v1 object, so historical Phase16/17 certificates retain their exact
    canonical parity bytes and digest.
    """
    source, candidate = _prediction_rows(source_path, image_ids), _prediction_rows(candidate_path, image_ids)
    per_image = []
    for image_id in image_ids:
        metrics = _parity_rows(
            [row for row in source if row["image_id"] == image_id],
            [row for row in candidate if row["image_id"] == image_id],
            zero_overlap_matches=schema_version == 1,
        )
        per_image.append({"image_id": image_id, **metrics})
    categories = sorted({row["category_id"] for row in source + candidate})
    per_class = []
    for category_id in categories:
        metrics = _parity_rows(
            [row for row in source if row["category_id"] == category_id],
            [row for row in candidate if row["category_id"] == category_id],
            zero_overlap_matches=schema_version == 1,
        )
        per_class.append({"category_id": category_id, **metrics})
    return {
        **_parity_rows(source, candidate, zero_overlap_matches=schema_version == 1),
        "per_image": per_image,
        "per_class": per_class,
    }


def _prediction_parity_diagnostics(
    source_path: Path,
    candidate_path: Path,
    image_ids: tuple[int, ...],
    *,
    schema_version: Literal[1, 2] = 2,
) -> dict[str, Any]:
    """Detailed deterministic drift view, additive to legacy v1 parity."""
    source, candidate = _prediction_rows(source_path, image_ids), _prediction_rows(candidate_path, image_ids)
    per_image = [
        {
            "image_id": image_id,
            **_parity_rows(
                [row for row in source if row["image_id"] == image_id],
                [row for row in candidate if row["image_id"] == image_id],
                diagnostics=True,
                zero_overlap_matches=schema_version == 1,
            ),
        }
        for image_id in image_ids
    ]
    categories = sorted({row["category_id"] for row in source + candidate})
    per_class = [
        {
            "category_id": category_id,
            **_parity_rows(
                [row for row in source if row["category_id"] == category_id],
                [row for row in candidate if row["category_id"] == category_id],
                diagnostics=True,
                zero_overlap_matches=schema_version == 1,
            ),
        }
        for category_id in categories
    ]
    aggregate = _parity_rows(source, candidate, diagnostics=True, zero_overlap_matches=schema_version == 1)
    return {
        "lost_predictions": aggregate["lost_predictions"],
        "added_predictions": aggregate["added_predictions"],
        "per_image": per_image,
        "per_class": per_class,
    }


def _parity_within_policy(parity: dict[str, Any], policy: ConformancePolicy) -> bool:
    """Every aggregate, image, and defined-class record must meet the policy."""
    records = [parity, *parity["per_image"], *parity["per_class"]]
    for record in records:
        if (
            record["lost"] > policy.golden_parity_lost
            or record["added"] > policy.golden_parity_added
            or record["mean_iou"] < policy.mean_iou_min
            or record["min_iou"] < policy.min_iou_min
            or record["max_score_delta"] > policy.max_score_delta
        ):
            return False
    return True


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix, iy = max(ax, bx), max(ay, by)
    ex, ey = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ex - ix) * max(0, ey - iy)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _canonical_predictions(target, artifact: PreparedArtifact, contract: VisionContract, dataset_path: Path) -> bytes:
    """Run the prepared artifact over the manifest, returning canonical COCO JSON.

    Greedy IoU only supports the contract's already-decoded, ranked result
    representation; custom decoder/NMS plugins must self-attest their decoding.
    """
    manifest = DatasetManifest.from_coco(dataset_path)
    rows = []
    for image in manifest.images:
        path = dataset_path.parent / image.file_name
        # Hash again directly before execution to close TOCTOU between manifest
        # validation and target decoding.
        if digest_path(path) != image.sha256:
            raise ValueError("dataset image changed before execution")
        result = target.run(artifact, target.prepare_input(artifact, path, contract))
        if not _valid_result(result, contract):
            raise ValueError("generated prediction violates result contract")
        for box, score, class_index in zip(result.boxes, result.scores, result.classes, strict=True):
            rows.append(
                {
                    "bbox": [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1],
                    "category_id": contract.coco_category_ids[class_index],
                    "image_id": image.image_id,
                    "score": score,
                }
            )
    rows.sort(key=lambda row: (row["image_id"], -row["score"], row["category_id"], row["bbox"]))
    return _canonical_json(rows).encode()


class GateDetail(Strict):
    """Machine-readable failure provenance for a single conformance phase."""

    gate: Literal["runtime", "accuracy_dataset", "determinism", "artifacts"]
    stage: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    exception_type: str = Field(min_length=1)


class CertificationReport(Strict):
    schema_version: Literal[1, 2] = 1
    status: Gate
    gates: dict[Literal["runtime", "accuracy_dataset", "determinism", "artifacts"], Gate]
    # Version 1 used free-form strings.  Version 2 makes a failure's owning
    # gate and stage machine-readable while preserving existing certificates.
    details: dict[str, str] | dict[str, GateDetail] = {}
    contract_digest: Hex
    source_plugin: str = Field(min_length=1)
    source_plugin_version: str = Field(min_length=1)
    target_plugin: str = Field(min_length=1)
    target_plugin_version: str = Field(min_length=1)
    source_digest: Hex
    artifact_digest: Hex
    golden: GoldenCorpus
    accuracy: AccuracyEvidence | None = None
    runtime: RuntimeEvidence | None = None
    evidence: ConformanceEvidenceBundle | None = None
    policy_digest: Hex
    environment: dict[str, str] = Field(min_length=1)
    digest: Hex

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Do not materialize v2-only nested fields into a v1 report digest."""
        result = super().model_dump(*args, **kwargs)
        if self.schema_version == 1 and isinstance(result.get("accuracy"), dict):
            result["accuracy"].pop("parity_schema_version", None)
        return result

    @model_validator(mode="after")
    def valid(self):
        if set(self.gates) != {"runtime", "accuracy_dataset", "determinism", "artifacts"}:
            raise ValueError("report requires exactly all gates")
        status = (
            "FAIL"
            if "FAIL" in self.gates.values()
            else "PASS"
            if all(x == "PASS" for x in self.gates.values())
            else "UNTESTED"
        )
        if self.status != status:
            raise ValueError("report status does not match gates")
        if self.schema_version == 2:
            failed = {name for name, state in self.gates.items() if state == "FAIL"}
            if set(self.details) != failed:
                raise ValueError("v2 report details must name exactly the failed gates")
            if not all(isinstance(detail, GateDetail) and detail.gate == name for name, detail in self.details.items()):
                raise ValueError("v2 report detail gate must match its key")
        elif not all(isinstance(detail, str) for detail in self.details.values()):
            raise ValueError("v1 report details must be strings")
        if self.gates["runtime"] == "PASS" and self.runtime is None:
            raise ValueError("runtime PASS requires RuntimeEvidence")
        if self.gates["accuracy_dataset"] == "PASS" and self.accuracy is None:
            raise ValueError("accuracy PASS requires AccuracyEvidence")
        if self.gates["determinism"] == "PASS" and not self.golden.cases:
            raise ValueError("determinism PASS requires executed cases")
        if self.status == "PASS" and (self.evidence is None or self.runtime is None or self.accuracy is None):
            raise ValueError("PASS requires file-backed evidence")
        if self.accuracy is not None and self.accuracy.parity_schema_version != self.schema_version:
            raise ValueError("report and prediction parity schema versions must match")
        if self.runtime and (
            self.runtime.source_digest != self.source_digest or self.runtime.artifact_digest != self.artifact_digest
        ):
            raise ValueError("runtime evidence binding mismatch")
        if self.runtime and (
            self.runtime.source_plugin_version != self.source_plugin_version
            or self.runtime.target_plugin_version != self.target_plugin_version
        ):
            raise ValueError("runtime plugin version binding mismatch")
        expected = _report_digest(self.model_dump(exclude={"digest"}, mode="json"))
        if self.digest != expected:
            raise ValueError("certification digest mismatch")
        return self

    @classmethod
    def trusted_create(cls, **values: Any) -> CertificationReport:
        values = dict(values)
        values.setdefault("schema_version", 2)
        values["digest"] = _report_digest(values)
        return cls.model_validate(values)


def _report_digest(value: dict[str, Any]) -> str:
    value = {key: item for key, item in value.items() if key != "digest"}
    return hashlib.sha256(
        _canonical_json(
            value,
        ).encode()
    ).hexdigest()


def _iou(a, b):
    x1, y1, x2, y2 = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area() + b.area() - inter
    return inter / union if union else 0


def _valid_result(value: Any, contract: VisionContract) -> bool:
    if not isinstance(value, DetectionResult) or not (len(value.boxes) == len(value.scores) == len(value.classes)):
        return False
    if len(value.boxes) > contract.result.max_detections or any(
        not contract.result.score_threshold <= x <= 1 for x in value.scores
    ):
        return False
    if any(not 0 <= x < len(contract.labels) for x in value.classes):
        return False
    if contract.result.ordering == "score_desc" and any(
        a < b for a, b in zip(value.scores, value.scores[1:], strict=False)
    ):
        return False
    # NMS invariant: after score threshold/top-k ordering, no later candidate
    # may remain when greedy IoU would have suppressed it.  This checks the
    # decoded output; it cannot prove a proprietary decoder's pre-NMS logits.
    if contract.result.suppression == "greedy_iou":
        seen: list[int] = []
        limit = contract.result.top_k or len(value.boxes)
        for index, box in enumerate(value.boxes[:limit]):
            for other in seen:
                same_class = value.classes[index] == value.classes[other]
                if (not contract.result.class_aware_nms or same_class) and _iou(
                    box, value.boxes[other]
                ) >= contract.result.nms_iou_threshold:
                    return False
            seen.append(index)
    for box in value.boxes:
        if box.x1 < 0 or box.y1 < 0 or box.x2 < box.x1 or box.y2 < box.y1:
            return False
        if contract.result.clip_boxes:
            if value.image_size is None or box.x2 > value.image_size[0] or box.y2 > value.image_size[1]:
                return False
    if contract.result.ordering == "score_desc" and contract.result.tie_policy == "class_then_box":
        tied = zip(
            value.scores, value.scores[1:], value.classes, value.classes[1:], value.boxes, value.boxes[1:], strict=False
        )
        if any(
            score == next_score
            and (cls, box.x1, box.y1, box.x2, box.y2) > (next_cls, next_box.x1, next_box.y1, next_box.x2, next_box.y2)
            for score, next_score, cls, next_cls, box, next_box in tied
        ):
            return False
    return True


def _semantic(a: Any, b: Any, c: VisionContract) -> bool:
    if not _valid_result(a, c) or not _valid_result(b, c) or len(a.boxes) != len(b.boxes):
        return False
    return all(
        a.classes[i] == b.classes[i]
        and abs(a.scores[i] - b.scores[i]) <= c.tolerances.score_delta
        and _iou(a.boxes[i], b.boxes[i]) >= c.tolerances.box_iou
        for i in range(len(a.boxes))
    )


def _repeatable(a: Any, b: Any, c: VisionContract) -> bool:
    """A deterministic contract is stricter than source/candidate parity.

    The public tolerance can be intentionally broad for a quantized candidate;
    it must never turn a materially different repeated execution into a
    determinism PASS.  Runtime outputs here are fp32/float16 decoded results,
    so require identical classes/cardinality and a tiny dtype-safe absolute
    coordinate/score drift.
    """
    if not _valid_result(a, c) or not _valid_result(b, c) or len(a.boxes) != len(b.boxes):
        return False
    epsilon = 1e-5
    return all(
        a.classes[i] == b.classes[i]
        and abs(a.scores[i] - b.scores[i]) <= epsilon
        and all(abs(x - y) <= epsilon for x, y in zip(a.boxes[i].as_xyxy(), b.boxes[i].as_xyxy(), strict=True))
        for i in range(len(a.boxes))
    )


def conform(
    source,
    target,
    captured: PreparedSource,
    artifact: PreparedArtifact,
    cases,
    contract: VisionContract,
    *,
    golden: GoldenCorpus,
    policy_digest: Hex,
    source_digest: Hex,
    artifact_digest: Hex,
    runtime: RuntimeEvidence | None = None,
    accuracy: AccuracyEvidence | None = None,
    evidence: ConformanceEvidenceBundle | None = None,
    report_schema_version: Literal[1, 2] = 2,
) -> CertificationReport:
    if (
        not cases
        or not isinstance(captured, PreparedSource)
        or not isinstance(artifact, PreparedArtifact)
        or len(cases) != len(golden.cases)
        or source_digest != contract.provenance.source_sha256
        or captured.sha256 != source_digest
        or artifact.sha256 != artifact_digest
        or captured.contract_digest != contract.digest
        or artifact.contract_digest != contract.digest
        or digest_path(captured.material_path) != captured.sha256
        or digest_path(artifact.material_path) != artifact.sha256
        or captured.plugin_id != source.plugin_id
        or artifact.plugin_id != target.plugin_id
        or captured.plugin_version != source.plugin_version
        or artifact.plugin_version != target.plugin_version
    ):
        raise ValueError("conformance cases and source contract binding are required")
    if accuracy is not None and accuracy.parity_schema_version != report_schema_version:
        raise ValueError("report and prediction parity schema versions must match")
    gates: dict[str, Gate] = {
        "runtime": "UNTESTED",
        "accuracy_dataset": "UNTESTED",
        "determinism": "UNTESTED",
        "artifacts": "UNTESTED",
    }
    details: dict[str, GateDetail] = {}
    # This is independent of file-backed accuracy evidence.  In particular, a
    # bad metrics file must not rewrite a real artifact verification attempt.
    try:
        if target.verify_artifact(artifact, contract):
            gates["artifacts"] = "PASS"
        else:
            gates["artifacts"] = "FAIL"
            details["artifacts"] = GateDetail(
                gate="artifacts",
                stage="artifact_verification",
                reason="target plugin rejected prepared artifact",
                exception_type="ArtifactVerificationError",
            )
    except _OPERATIONAL_EXCEPTIONS as exc:
        gates["artifacts"] = "FAIL"
        details["artifacts"] = GateDetail(
            gate="artifacts",
            stage="artifact_verification",
            reason=_operational_reason(exc),
            exception_type=type(exc).__name__,
        )

    verified_cases: list[bytes] | None = None
    if runtime is not None and accuracy is not None and evidence is not None:
        try:
            evidence.verify(golden, accuracy, runtime, source_plugin=source.plugin_id, target_plugin=target.plugin_id)
            candidate_cases = []
            for supplied, declared in zip(cases, evidence.case_inputs, strict=True):
                if isinstance(supplied, Path):
                    payload = supplied.read_bytes()
                elif isinstance(supplied, bytes):
                    payload = supplied
                else:
                    raise ValueError("cases must be verified Paths or bytes")
                if hashlib.sha256(payload).hexdigest() != declared.sha256:
                    raise ValueError("executed case bytes do not match golden evidence")
                candidate_cases.append(payload)
            if policy_digest != evidence.policy.sha256:
                raise ValueError("policy digest mismatch")
            verified_cases = candidate_cases
            gates["accuracy_dataset"] = "PASS"
        except _OPERATIONAL_EXCEPTIONS as exc:
            # Metrics, parity and file bindings are accuracy evidence.  They
            # must not be reported as a semantic runtime failure.
            gates["accuracy_dataset"] = "FAIL"
            details["accuracy_dataset"] = GateDetail(
                gate="accuracy_dataset",
                stage="evidence_binding",
                reason=_operational_reason(exc),
                exception_type=type(exc).__name__,
            )

    # A missing or rejected prerequisite means no execution took place, so the
    # runtime/determinism gates stay UNTESTED.  Terminal persistence and
    # source-device provenance remain explicit experiment-harness duties.
    if verified_cases is not None and gates["artifacts"] == "PASS":
        actual: list[DetectionResult] | None = None
        try:
            expected, actual = (
                [source.run(captured, source.prepare_input(captured, x, contract)) for x in verified_cases],
                [target.run(artifact, target.prepare_input(artifact, x, contract)) for x in verified_cases],
            )
            gates["runtime"] = (
                "PASS" if all(_semantic(a, b, contract) for a, b in zip(expected, actual, strict=True)) else "FAIL"
            )
            if gates["runtime"] == "FAIL":
                details["runtime"] = GateDetail(
                    gate="runtime",
                    stage="semantic_runtime",
                    reason="source and target results violate the semantic contract",
                    exception_type="SemanticMismatch",
                )
        except _OPERATIONAL_EXCEPTIONS as exc:
            gates["runtime"] = "FAIL"
            details["runtime"] = GateDetail(
                gate="runtime",
                stage="semantic_runtime",
                reason=_operational_reason(exc),
                exception_type=type(exc).__name__,
            )

        if actual is not None:
            try:
                repeat = [target.run(artifact, target.prepare_input(artifact, x, contract)) for x in verified_cases]
                gates["determinism"] = (
                    "PASS" if all(_repeatable(a, b, contract) for a, b in zip(actual, repeat, strict=True)) else "FAIL"
                )
                if gates["determinism"] == "FAIL":
                    details["determinism"] = GateDetail(
                        gate="determinism",
                        stage="repeatability",
                        reason="repeated target execution violates deterministic output tolerance",
                        exception_type="DeterminismMismatch",
                    )
            except _OPERATIONAL_EXCEPTIONS as exc:
                gates["determinism"] = "FAIL"
                details["determinism"] = GateDetail(
                    gate="determinism",
                    stage="repeatability",
                    reason=_operational_reason(exc),
                    exception_type=type(exc).__name__,
                )

        try:
            generated = _canonical_predictions(target, artifact, contract, evidence.dataset_manifest.verify())
            packaged = evidence.predictions.verify().read_bytes()
            if generated != packaged:
                raise ValueError("packaged predictions were not generated by the prepared artifact")
            generated_source = _canonical_predictions(source, captured, contract, evidence.dataset_manifest.verify())
            if generated_source != evidence.source_predictions.verify().read_bytes():
                raise ValueError("packaged source predictions were not generated by captured source material")
        except _OPERATIONAL_EXCEPTIONS as exc:
            gates["accuracy_dataset"] = "FAIL"
            details["accuracy_dataset"] = GateDetail(
                gate="accuracy_dataset",
                stage="generated_prediction_binding",
                reason=_operational_reason(exc),
                exception_type=type(exc).__name__,
            )
    status: Gate = (
        "FAIL" if "FAIL" in gates.values() else "PASS" if all(x == "PASS" for x in gates.values()) else "UNTESTED"
    )
    return CertificationReport.trusted_create(
        schema_version=report_schema_version,
        status=status,
        gates=gates,
        details=details,
        contract_digest=contract.digest,
        source_plugin=source.plugin_id,
        source_plugin_version=source.plugin_version,
        target_plugin=target.plugin_id,
        target_plugin_version=target.plugin_version,
        source_digest=source_digest,
        artifact_digest=artifact_digest,
        golden=golden,
        accuracy=accuracy,
        runtime=runtime,
        evidence=evidence,
        policy_digest=policy_digest,
        environment={"evidence_environment_digest": evidence.environment_record.sha256}
        if evidence
        else {"unverified": "true"},
    )
