"""Fail-closed, dataset-level object-detection validation.

The module deliberately keeps model identity, evaluated image inventory and
detection rows separate.  A JSON report is evidence, not a switch callers can
set to ``passed``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import stat
import time
import uuid
from ctypes import CDLL, c_char_p, c_int, get_errno
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..geometry import iou
from ..types import Box, DetectionResult


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(value: str) -> Path:
    """Return a portable, non-escaping report path."""
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError("report paths must be safe relative paths")
    return path


def _trusted_directory(path: Path) -> Path:
    """Reject a publication parent containing a symlink before writing."""
    absolute = path.absolute()
    for parent in (absolute, *absolute.parents):
        if parent.is_symlink():
            raise ValueError(f"untrusted symlink parent: {parent}")
    if not absolute.is_dir():
        raise ValueError(f"publication parent is not a directory: {absolute}")
    return absolute


def _relative_file(root: Path, value: str) -> Path:
    path = root / _safe_relative(value)
    if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
        raise ValueError("report file is missing, unsafe, or a symlink")
    return path


def _policy_digest(policy: ValidationPolicy) -> str:
    return _digest(policy.model_dump(mode="json"))


def _validation_hardware_digest() -> str:
    """Bind stable host facts without publishing a hostname or network identity."""

    return _digest(
        {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        }
    )


def _validation_harness_digest() -> str:
    return _digest(
        {
            "implementation": "vision.validation.validate_dataset",
            "module_sha256": _sha(Path(__file__)),
            "matching": "class-compatible-hungarian-max-iou",
            "metrics": "pycocotools-cocoeval-bbox",
        }
    )


_HEX64 = set("0123456789abcdef")


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _basename(name: str) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("unsafe publication basename")
    return name


def _write_bytes_at(directory_fd: int, basename: str, data: bytes) -> None:
    """Create a durable file relative to a held directory descriptor only."""
    name = _basename(basename)
    fd = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_bytes_at(directory_fd: int, basename: str) -> bytes:
    fd = os.open(_basename(basename), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    try:
        chunks = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _linkat(source_fd: int, source: str, destination_fd: int, destination: str) -> None:
    # Python's link exposes dir_fd and never resolves a stage path.
    os.link(
        _basename(source),
        _basename(destination),
        src_dir_fd=source_fd,
        dst_dir_fd=destination_fd,
        follow_symlinks=False,
    )


def _rmtree_at(parent_fd: int, basename: str) -> bool:
    """Remove an owned tree recursively without following a symlink."""
    name = _basename(basename)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    except FileNotFoundError:
        return True
    try:
        for entry in os.listdir(fd):
            info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                _rmtree_at(fd, entry)
            else:
                os.unlink(entry, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return True
    return True


def _fsync_fd(fd: int) -> None:
    os.fsync(fd)


def _verify_stage_fds(stage_fd: int, result_fd: int, report: ValidationReport, expected_policy_digest: str) -> None:
    """FD-only prepublication integrity check for the complete staged tree."""
    if expected_policy_digest != report.policy_digest:
        raise ValueError("unexpected external policy digest")
    exact = _read_bytes_at(result_fd, "validation_report.json")
    marker = _read_bytes_at(result_fd, "COMMITTED").decode()
    top_exact, top_marker = (
        _read_bytes_at(stage_fd, "validation_report.json"),
        _read_bytes_at(stage_fd, "COMMITTED").decode(),
    )
    if exact != top_exact or marker != hashlib.sha256(exact).hexdigest() or top_marker != marker:
        raise ValueError("staged validation markers do not bind exact report bytes")
    if _canonical(json.loads(exact)) != _canonical(report.model_dump(mode="json")):
        raise ValueError("staged report bytes do not match report object")
    for prediction in (report.reference, report.candidate):
        rows = json.loads(_read_bytes_at(result_fd, prediction.predictions_path))
        record = json.loads(_read_bytes_at(result_fd, prediction.record_path))
        if _sha_bytes(_canonical(rows).encode()) != prediction.predictions_sha256:
            raise ValueError("staged prediction bytes do not match report")
        if _canonical(record) != _canonical(prediction.model_dump(mode="json")):
            raise ValueError("staged prediction manifest bytes do not match report")
    report.__class__.model_validate(report.model_dump(mode="json"))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _noreplace_rename(
    parent_fd: int, source: str, destination: str, *, lock_path=None, held_fd=None, held_identity=None, token=None
) -> None:
    """Kernel-enforced non-clobber rename, never an exists-then-rename race."""
    if lock_path is not None:
        lock_fd = os.open(lock_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            current = os.fstat(lock_fd)
            payload = json.loads(os.read(lock_fd, 64 * 1024).decode())
            if (
                (current.st_dev, current.st_ino) != held_identity
                or (os.fstat(held_fd).st_dev, os.fstat(held_fd).st_ino) != held_identity
                or payload.get("token") != token
            ):
                raise ValueError("publication lock capability changed before commit")
        finally:
            os.close(lock_fd)
    libc = CDLL(None, use_errno=True)
    if platform.system() == "Darwin":
        fn = libc.renameatx_np
        fn.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]
        result = fn(parent_fd, source.encode(), parent_fd, destination.encode(), 0x0004)  # RENAME_EXCL
    elif platform.system() == "Linux":
        try:
            fn = libc.renameat2
        except AttributeError:
            numbers = {"x86_64": 316, "amd64": 316, "aarch64": 276, "arm64": 276}
            machine = platform.machine().lower()
            if machine not in numbers:
                raise RuntimeError(f"no verified renameat2 syscall number for {machine}") from None
            result = libc.syscall(numbers[machine], parent_fd, source.encode(), parent_fd, destination.encode(), 1)
        else:
            fn.argtypes = [c_int, c_char_p, c_int, c_char_p, c_int]
            result = fn(parent_fd, source.encode(), parent_fd, destination.encode(), 1)
    else:
        raise RuntimeError("platform lacks a verified no-clobber publication rename")
    if result != 0:
        error = get_errno()
        if error == 17:
            raise FileExistsError(f"validation output already exists: {destination}")
        raise OSError(error, os.strerror(error))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageEntry(Strict):
    id: int
    path: str
    sha256: str = Field(min_length=64, max_length=64)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DatasetManifest(Strict):
    format: Literal["coco"] = "coco"
    split: str
    annotations_path: str
    annotations_sha256: str = Field(min_length=64, max_length=64)
    images: list[ImageEntry] = Field(min_length=1)
    categories: list[int] = Field(min_length=1)
    preprocessing_digest: str = Field(min_length=64, max_length=64)
    postprocessing_digest: str = Field(min_length=64, max_length=64)
    allowed_preprocessing_digests: tuple[str, ...] = ()
    allowed_postprocessing_digests: tuple[str, ...] = ()
    expected_image_count: int = Field(gt=0)
    expected_annotation_count: int = Field(ge=0)
    cocoeval_iou_type: Literal["bbox"] = "bbox"
    cocoeval_version: str = "pycocotools"
    digest: str = Field(min_length=64, max_length=64)

    def canonical_digest(self) -> str:
        payload = self.model_dump(exclude={"digest"}, mode="json")
        full = _digest(payload)
        legacy = dict(payload)
        if not self.allowed_preprocessing_digests:
            legacy.pop("allowed_preprocessing_digests")
        if not self.allowed_postprocessing_digests:
            legacy.pop("allowed_postprocessing_digests")
        legacy_digest = _digest(legacy)
        return self.digest if self.digest in {full, legacy_digest} else legacy_digest

    @model_validator(mode="after")
    def valid(self):
        if self.expected_image_count != len(self.images) or len({item.id for item in self.images}) != len(self.images):
            raise ValueError("image inventory/count mismatch")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("duplicate dataset category IDs")
        for values in (self.allowed_preprocessing_digests, self.allowed_postprocessing_digests):
            invalid = any(len(value) != 64 or any(c not in _HEX64 for c in value) for value in values)
            if len(values) != len(set(values)) or invalid:
                raise ValueError("allowed processing digests must be unique lowercase SHA-256 values")
        if self.digest != self.canonical_digest():
            raise ValueError("dataset manifest digest mismatch")
        annotation = Path(self.annotations_path)
        if annotation.is_symlink() or not annotation.is_file() or _sha(annotation) != self.annotations_sha256:
            raise ValueError("annotation file identity mismatch")
        payload = json.loads(annotation.read_text())
        images = payload.get("images", [])
        categories = payload.get("categories", [])
        annotations = payload.get("annotations", [])
        if len(annotations) != self.expected_annotation_count:
            raise ValueError("annotation count mismatch")
        if len({item.get("id") for item in images}) != len(images) or len(
            {item.get("id") for item in categories}
        ) != len(categories):
            raise ValueError("duplicate COCO image/category IDs")
        if len({item.get("id") for item in annotations}) != len(annotations):
            raise ValueError("duplicate COCO annotation IDs")
        if {item["id"] for item in images} != {item.id for item in self.images}:
            raise ValueError("annotation image IDs mismatch")
        if {item["id"] for item in categories} != set(self.categories):
            raise ValueError("annotation categories mismatch")
        coco_images = {item["id"]: item for item in images}
        for annotation_item in annotations:
            bbox = annotation_item.get("bbox")
            image_id, category_id = annotation_item.get("image_id"), annotation_item.get("category_id")
            if image_id not in coco_images or category_id not in self.categories:
                raise ValueError("COCO annotation reference mismatch")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bbox)
            ):
                raise ValueError("invalid COCO annotation bbox")
            image = coco_images[image_id]
            if (
                bbox[2] <= 0
                or bbox[3] <= 0
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[0] + bbox[2] > image["width"]
                or bbox[1] + bbox[3] > image["height"]
            ):
                raise ValueError("COCO annotation bbox out of bounds")
        for item in self.images:
            path = Path(item.path)
            if path.is_symlink() or not path.is_file() or _sha(path) != item.sha256:
                raise ValueError("image identity mismatch")
            with Image.open(path) as image:
                if image.size != (item.width, item.height):
                    raise ValueError("image dimensions mismatch")
            coco = coco_images[item.id]
            if coco.get("width") != item.width or coco.get("height") != item.height:
                raise ValueError("COCO image dimensions mismatch")
            if coco.get("file_name") != path.name:
                raise ValueError("COCO image path mismatch")
        return self


class ModelTarget(Strict):
    """Identity of the loaded model used to generate a prediction file."""

    source: str
    source_sha256: str = Field(min_length=64, max_length=64)
    manifest_path: str
    manifest_sha256: str = Field(min_length=64, max_length=64)
    manifest_digest: str = Field(min_length=64, max_length=64)
    device: str = Field(min_length=1)
    class_mapping: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self):
        source, manifest = Path(self.source), Path(self.manifest_path)
        if source.is_symlink() or not source.is_file() or _sha(source) != self.source_sha256:
            raise ValueError("model source identity mismatch")
        if manifest.is_symlink() or not manifest.is_file() or _sha(manifest) != self.manifest_sha256:
            raise ValueError("model manifest identity mismatch")
        from ..manifest import VisionManifest

        parsed = VisionManifest.from_yaml(manifest)
        if _digest(parsed.model_dump(mode="json")) != self.manifest_digest:
            raise ValueError("model manifest canonical digest mismatch")
        declared_source = Path(parsed.source)
        bound_source = (
            declared_source.resolve()
            if declared_source.is_absolute()
            else (manifest.parent / declared_source).resolve()
        )
        if bound_source != source.resolve():
            raise ValueError("model manifest source is not bound to signed target")
        suffix = source.suffix.lower()
        allowed = {
            "onnx": {".onnx", ".xml", ".engine"},
            "torch": {".pt", ".pth", ".ckpt"},
            "yolo": {".pt", ".pth", ".ckpt"},
            "yolo26": {".pt", ".pth", ".ckpt"},
            "rfdetr": {".pt", ".pth", ".ckpt"},
            "rtdetr": {".pt", ".pth", ".ckpt"},
        }
        if suffix not in allowed.get(parsed.family, set()):
            raise ValueError("model source suffix is incompatible with manifest family")
        canonical_keys: list[int] = []
        for key in self.class_mapping:
            if not isinstance(key, str) or not key.isascii() or str(int(key)) != key or int(key) < 0:
                raise ValueError("class mapping keys must be canonical decimal model class IDs")
            canonical_keys.append(int(key))
        if len(set(canonical_keys)) != len(canonical_keys) or len(set(self.class_mapping.values())) != len(
            self.class_mapping
        ):
            raise ValueError("class mapping must be one-to-one")
        return self

    def manifest(self):
        from ..manifest import VisionManifest

        return VisionManifest.from_yaml(self.manifest_path)

    def assert_dataset_contract(self, dataset: DatasetManifest) -> None:
        manifest = self.manifest()
        if set(self.class_mapping.values()) != set(dataset.categories):
            raise ValueError("class mapping must cover exactly the dataset categories")
        effective_pre = _digest(
            {
                "input": manifest.input.model_dump(mode="json"),
                "preprocessing": manifest.preprocessing.model_dump(mode="json"),
            }
        )
        effective_post = _digest(
            {
                "output": manifest.output.model_dump(mode="json"),
                "decoder": manifest.decoder.model_dump(mode="json"),
                "postprocessing": manifest.postprocessing,
                "class_mapping": self.class_mapping,
            }
        )
        allowed_pre = {dataset.preprocessing_digest, *dataset.allowed_preprocessing_digests}
        allowed_post = {dataset.postprocessing_digest, *dataset.allowed_postprocessing_digests}
        if effective_pre not in allowed_pre:
            raise ValueError("preprocessing digest mismatch")
        if effective_post not in allowed_post:
            raise ValueError("postprocessing digest mismatch")


class PredictionManifest(Strict):
    target: ModelTarget
    dataset_digest: str = Field(min_length=64, max_length=64)
    predictions_path: str
    predictions_sha256: str = Field(min_length=64, max_length=64)
    input_preprocessing_digest: str = Field(min_length=64, max_length=64)
    output_postprocessing_digest: str = Field(min_length=64, max_length=64)
    runtime: RuntimeExecutionIdentity
    record_path: str = ""
    evaluated_image_ids: list[int]
    detection_image_ids: list[int]
    digest: str = Field(min_length=64, max_length=64)

    def canonical_digest(self) -> str:
        return _digest(self.model_dump(exclude={"digest"}, mode="json"))

    @model_validator(mode="after")
    def valid(self):
        if self.digest != self.canonical_digest():
            raise ValueError("prediction manifest identity mismatch")
        _safe_relative(self.predictions_path)
        if self.record_path:
            _safe_relative(self.record_path)
        if len(set(self.evaluated_image_ids)) != len(self.evaluated_image_ids):
            raise ValueError("duplicate evaluated image IDs")
        if len(set(self.detection_image_ids)) != len(self.detection_image_ids):
            raise ValueError("duplicate detection image IDs")
        return self


class RuntimeExecutionIdentity(Strict):
    """Observed runtime facts, bound to the signed model target by validation."""

    backend: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    device: str = Field(min_length=1)
    precision: str = Field(pattern="^(fp32|fp16|int8)$")
    input_shape: tuple[int, int, int, int]
    source_sha256: str = Field(min_length=64, max_length=64)
    manifest_digest: str = Field(min_length=64, max_length=64)
    digest: str

    def canonical_digest(self) -> str:
        return _digest(self.model_dump(exclude={"digest"}, mode="json"))

    @model_validator(mode="after")
    def valid(self):
        if any(value <= 0 for value in self.input_shape):
            raise ValueError("runtime input shape must be positive")
        if self.digest != self.canonical_digest():
            raise ValueError("runtime execution identity digest mismatch")
        return self


_RUNTIME_KEYS = frozenset({"backend", "provider", "device", "precision", "input_shape"})


def _verify_runtime_contract(target: ModelTarget, runtime: RuntimeExecutionIdentity) -> None:
    """Reject runtime claims that cannot be derived from the signed target.

    A report records *observed* session facts, but those facts are useful only
    when they agree with the manifest family/input contract and the device
    explicitly requested by the signed target.  This deliberately keeps a
    provider name from being used as a substitute for a device claim.
    """
    # ``_validate_materialized`` is a private test/ingestion helper and marks
    # its synthetic rows explicitly.  It is never produced by ``vision.load``
    # (which must provide a verifiable session identity below).
    if runtime.backend == runtime.provider == "materialized":
        if runtime.device != target.device:
            raise ValueError("materialized runtime device is not the signed target device")
        return
    manifest = target.manifest()
    allowed = {
        "onnx": {"onnxruntime", "openvino", "tensorrt"},
        "torch": {"pytorch"},
        "yolo": {"pytorch"},
        "yolo26": {"pytorch"},
        "rfdetr": {"pytorch"},
        "rtdetr": {"pytorch"},
    }
    providers = {
        "onnxruntime": {"CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"},
        "openvino": {"AUTO", "CPU", "GPU", "NPU"},
        "pytorch": {"torch", "ultralytics-torch", "rfdetr-torch"},
        "tensorrt": {"TensorRTExecutionProvider"},
    }
    if runtime.backend not in allowed[manifest.family]:
        raise ValueError("runtime backend is incompatible with manifest family")
    if runtime.provider not in providers.get(runtime.backend, set()):
        raise ValueError("runtime provider is incompatible with backend")
    declared_artifacts = []
    manifest_parent = Path(target.manifest_path).parent
    for artifact in manifest.artifacts:
        artifact_path = Path(artifact.path)
        resolved = artifact_path if artifact_path.is_absolute() else manifest_parent / artifact_path
        if resolved.resolve() == Path(target.source).resolve():
            declared_artifacts.append(artifact)
    if declared_artifacts:
        if not any(
            artifact.validated
            and artifact.backend == runtime.backend
            and artifact.precision == runtime.precision
            and artifact.checksum == target.source_sha256
            for artifact in declared_artifacts
        ):
            raise ValueError("runtime precision/backend is not bound to a validated manifest artifact")
    elif runtime.precision != {"float32": "fp32", "float16": "fp16", "uint8": "int8"}[manifest.input.dtype]:
        raise ValueError("runtime precision does not match manifest input dtype")
    if runtime.input_shape != tuple(manifest.input.shape):
        raise ValueError("runtime input shape does not match manifest")
    if target.device != "auto" and runtime.device != target.device:
        raise ValueError("runtime device is not the signed target device")


class ValidationPolicy(Strict):
    iou_threshold: float = Field(default=0.5, gt=0, le=1)
    score_delta: float = Field(default=0.05, ge=0, le=1)
    high_confidence: float = Field(default=0.5, ge=0, le=1)
    max_map_drop: float = Field(ge=0, le=1)
    min_coverage: float = Field(default=1, ge=0, le=1)
    max_class_mismatches: int = Field(default=0, ge=0)
    reference_runtime: RuntimeExecutionIdentity | None = None
    candidate_runtime: RuntimeExecutionIdentity | None = None


def compare_detections(
    reference: DetectionResult, candidate: DetectionResult, policy: ValidationPolicy
) -> dict[str, Any]:
    """Class-compatible maximum-IoU bipartite matching using SciPy Hungarian.

    Stable sort keys and a tiny lexicographic cost break make ties independent
    of incoming detection order.  Zero-cost dummies allow unmatched boxes.
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - validated by extra
        raise RuntimeError("install vision-buildkit[validation] for Hungarian matching") from exc
    refs = sorted(
        range(len(reference.boxes)),
        key=lambda k: (reference.classes[k], *reference.boxes[k].as_xyxy(), -reference.scores[k]),
    )
    cands = sorted(
        range(len(candidate.boxes)),
        key=lambda k: (candidate.classes[k], *candidate.boxes[k].as_xyxy(), -candidate.scores[k]),
    )
    n, m = len(refs), len(cands)
    size = n + m
    cost = np.zeros((size, size), dtype=float)
    valid: dict[tuple[int, int], float] = {}
    # Invalid real-real assignments lose to a dummy (positive cost); valid
    # assignments are negative and therefore selected when possible.
    if n and m:
        cost[:n, :m] = 1.0
        for ri, r in enumerate(refs):
            for ci, c in enumerate(cands):
                if reference.classes[r] == candidate.classes[c]:
                    overlap = iou(reference.boxes[r], candidate.boxes[c])
                    if overlap >= policy.iou_threshold:
                        valid[(ri, ci)] = overlap
                        cost[ri, ci] = -overlap + (ri * (m + 1) + ci) * 1e-12
    row, col = linear_sum_assignment(cost)
    pairs: list[dict[str, Any]] = []
    used_r, used_c = set(), set()
    for ri, ci in sorted(zip(row.tolist(), col.tolist(), strict=True)):
        if (ri, ci) not in valid:
            continue
        r, c = refs[ri], cands[ci]
        used_r.add(r)
        used_c.add(c)
        pairs.append(
            {
                "reference": r,
                "candidate": c,
                "class": reference.classes[r],
                "iou": valid[(ri, ci)],
                "score_delta": abs(reference.scores[r] - candidate.scores[c]),
            }
        )
    lost = sorted(set(range(n)) - used_r)
    added = sorted(set(range(m)) - used_c)
    mismatches: list[dict[str, int]] = []
    for r in lost:
        wrong = [
            c
            for c in added
            if candidate.classes[c] != reference.classes[r]
            and iou(reference.boxes[r], candidate.boxes[c]) >= policy.iou_threshold
        ]
        if wrong:
            mismatches.append({"reference": r, "candidate": min(wrong)})
    deltas, overlaps = [p["score_delta"] for p in pairs], [p["iou"] for p in pairs]
    high_lost = [x for x in lost if reference.scores[x] >= policy.high_confidence]
    high_added = [x for x in added if candidate.scores[x] >= policy.high_confidence]
    return {
        "matched": len(pairs),
        "reference_count": n,
        "candidate_count": m,
        "lost": lost,
        "added": added,
        "class_mismatches": mismatches,
        "high_conf_lost": high_lost,
        "high_conf_added": high_added,
        "coverage": len(pairs) / n if n else 1.0,
        "mean_iou": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "min_iou": min(overlaps) if overlaps else 0.0,
        "mean_score_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "max_score_delta": max(deltas) if deltas else 0.0,
        "pairs": pairs,
    }


def _rows_to_detections(rows: list[dict[str, Any]], image_id: int) -> DetectionResult:
    picked = [x for x in rows if x["image_id"] == image_id]
    return DetectionResult(
        [Box(x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]) for x in picked],
        [float(x["score"]) for x in picked],
        [int(x["category_id"]) for x in picked],
    )


def aggregate_parity(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    image_ids: list[int],
    policy: ValidationPolicy,
) -> dict[str, Any]:
    per_image = []
    for image_id in sorted(image_ids):
        item = compare_detections(
            _rows_to_detections(reference_rows, image_id), _rows_to_detections(candidate_rows, image_id), policy
        )
        item["image_id"] = image_id
        per_image.append(item)
    matches = sum(x["matched"] for x in per_image)
    refs = sum(x["reference_count"] for x in per_image)
    pairs = [pair | {"image_id": item["image_id"]} for item in per_image for pair in item["pairs"]]
    deltas, overlaps = [x["score_delta"] for x in pairs], [x["iou"] for x in pairs]
    return {
        "per_image": per_image,
        "matched": matches,
        "reference_count": refs,
        "candidate_count": sum(x["candidate_count"] for x in per_image),
        "lost": sum((x["lost"] for x in per_image), []),
        "added": sum((x["added"] for x in per_image), []),
        "class_mismatches": [
            x | {"image_id": item["image_id"]} for item in per_image for x in item["class_mismatches"]
        ],
        "high_conf_lost": [
            x | {"image_id": item["image_id"]}
            for item in per_image
            for x in [{"index": y} for y in item["high_conf_lost"]]
        ],
        "high_conf_added": [
            x | {"image_id": item["image_id"]}
            for item in per_image
            for x in [{"index": y} for y in item["high_conf_added"]]
        ],
        "coverage": matches / refs if refs else 1.0,
        "mean_iou": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "min_iou": min(overlaps) if overlaps else 0.0,
        "mean_score_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "max_score_delta": max(deltas) if deltas else 0.0,
        "pairs": pairs,
    }


class ValidationReport(Strict):
    dataset: DatasetManifest
    reference: PredictionManifest
    candidate: PredictionManifest
    policy: ValidationPolicy
    policy_digest: str = Field(min_length=64, max_length=64)
    evidence_kind: Literal["observed", "materialized"]
    publishable: bool
    hardware_digest: str | None = Field(
        default=None, min_length=64, max_length=64, exclude_if=lambda value: value is None
    )
    harness_digest: str | None = Field(
        default=None, min_length=64, max_length=64, exclude_if=lambda value: value is None
    )
    candidate_graph_digest: str | None = Field(
        default=None, min_length=64, max_length=64, exclude_if=lambda value: value is None
    )
    calibration_digest: str | None = Field(
        default=None, min_length=64, max_length=64, exclude_if=lambda value: value is None
    )
    parity: dict[str, Any]
    map_reference: float = Field(ge=0, le=1)
    map_candidate: float = Field(ge=0, le=1)
    ap50_reference: float = Field(ge=0, le=1)
    ap50_candidate: float = Field(ge=0, le=1)
    ap75_reference: float = Field(ge=0, le=1)
    ap75_candidate: float = Field(ge=0, le=1)
    gates: dict[str, bool]
    passed: bool
    digest: str = Field(min_length=64, max_length=64)

    def canonical_digest(self) -> str:
        return _digest(self.model_dump(exclude={"digest"}, mode="json"))

    def derived_gates(self) -> dict[str, bool]:
        expected = {x.id for x in self.dataset.images}
        inventory = (
            set(self.reference.evaluated_image_ids) == expected and set(self.candidate.evaluated_image_ids) == expected
        )
        parity = (
            not self.parity["high_conf_lost"]
            and not self.parity["high_conf_added"]
            and len(self.parity["class_mismatches"]) <= self.policy.max_class_mismatches
            and self.parity["max_score_delta"] <= self.policy.score_delta
        )
        coverage = inventory and self.parity["coverage"] >= self.policy.min_coverage
        return {
            "inventory": inventory,
            "coverage": coverage,
            "parity": parity,
            "map": self.map_reference - self.map_candidate <= self.policy.max_map_drop,
        }

    @model_validator(mode="after")
    def valid(self):
        expected = self.derived_gates()
        if (self.hardware_digest is None) != (self.harness_digest is None):
            raise ValueError("validation hardware and harness identities must be recorded together")
        if self.hardware_digest is not None and self.candidate_graph_digest is None:
            raise ValueError("observed validation identity requires a candidate graph digest")
        if (
            self.evidence_kind == "observed"
            and self.candidate.runtime.precision == "int8"
            and self.calibration_digest is None
        ):
            raise ValueError("observed INT8 validation requires a calibration digest")
        for value in (
            self.hardware_digest,
            self.harness_digest,
            self.candidate_graph_digest,
            self.calibration_digest,
        ):
            if value is not None and (len(value) != 64 or set(value) - _HEX64):
                raise ValueError("validation evidence digests must be lowercase SHA-256 values")
        if self.reference.dataset_digest != self.dataset.digest or self.candidate.dataset_digest != self.dataset.digest:
            raise ValueError("prediction dataset identity mismatch")
        if self.policy_digest != _policy_digest(self.policy):
            raise ValueError("validation policy digest mismatch")
        if self.evidence_kind == "materialized":
            if self.publishable:
                raise ValueError("materialized validation evidence is never publishable")
            if self.policy.reference_runtime is not None or self.policy.candidate_runtime is not None:
                raise ValueError("materialized validation evidence cannot carry runtime policy contracts")
            if any(
                prediction.runtime.backend != "materialized" or prediction.runtime.provider != "materialized"
                for prediction in (self.reference, self.candidate)
            ):
                raise ValueError("materialized validation evidence requires internal sentinels")
        else:
            expected_publishable = (
                self.policy.reference_runtime is not None and self.policy.candidate_runtime is not None
            )
            if self.publishable != expected_publishable:
                raise ValueError("observed validation publishability does not match runtime policy contracts")
            if self.policy.reference_runtime is not None and self.reference.runtime != self.policy.reference_runtime:
                raise ValueError("observed reference runtime does not match policy contract")
            if self.policy.candidate_runtime is not None and self.candidate.runtime != self.policy.candidate_runtime:
                raise ValueError("observed candidate runtime does not match policy contract")
            if any(
                prediction.runtime.backend == "materialized" or prediction.runtime.provider == "materialized"
                for prediction in (self.reference, self.candidate)
            ):
                raise ValueError("observed validation evidence cannot contain materialized sentinels")
        if self.gates != expected or self.passed != all(expected.values()):
            raise ValueError("forged validation gates or passed field")
        if self.digest != self.canonical_digest():
            raise ValueError("validation report digest mismatch")
        return self

    def verify(self, root: str | Path, expected_policy_digest: str, *, allow_materialized: bool = False) -> bool:
        """Recompute every quality gate from a committed, relative-path result.

        ``expected_policy_digest`` lets a capsule/optimizer bind a validation
        result to the policy it approved rather than accepting a substituted
        permissive policy stored inside the report.
        """
        if expected_policy_digest != self.policy_digest:
            raise ValueError("unexpected external policy digest")
        if self.evidence_kind == "materialized" and not allow_materialized:
            raise ValueError("materialized validation evidence requires explicit development authorization")
        outer = Path(root).resolve(strict=True)
        result = outer / "result" if (outer / "result").is_dir() else outer
        committed = result / "COMMITTED"
        report_path = result / "validation_report.json"
        if committed.is_symlink() or not committed.is_file() or not report_path.is_file() or report_path.is_symlink():
            raise ValueError("validation output is not committed")
        exact = report_path.read_bytes()
        if committed.read_text() != hashlib.sha256(exact).hexdigest():
            raise ValueError("validation commit marker does not bind exact report bytes")
        if outer != result:
            outer_marker = outer / "COMMITTED"
            if (
                outer_marker.is_symlink()
                or not outer_marker.is_file()
                or outer_marker.read_text() != hashlib.sha256(exact).hexdigest()
            ):
                raise ValueError("validation output is not published")
        if _canonical(json.loads(exact)) != _canonical(self.model_dump(mode="json")):
            raise ValueError("validation report bytes do not match report object")
        ref_path = _relative_file(result, self.reference.predictions_path)
        cand_path = _relative_file(result, self.candidate.predictions_path)
        if _sha(ref_path) != self.reference.predictions_sha256 or _sha(cand_path) != self.candidate.predictions_sha256:
            raise ValueError("prediction bytes do not match report")
        for prediction in (self.reference, self.candidate):
            if prediction.record_path:
                record = _relative_file(result, prediction.record_path)
                if _canonical(json.loads(record.read_text())) != _canonical(prediction.model_dump(mode="json")):
                    raise ValueError("prediction manifest bytes do not match report")
            runtime = prediction.runtime
            if (
                runtime.source_sha256 != prediction.target.source_sha256
                or runtime.manifest_digest != prediction.target.manifest_digest
            ):
                raise ValueError("runtime identity is not bound to signed target")
            _verify_runtime_contract(prediction.target, runtime)
        ref_rows = json.loads(ref_path.read_text())
        cand_rows = json.loads(cand_path.read_text())
        _, reference_detected = _validate_prediction(ref_rows, self.dataset)
        _, candidate_detected = _validate_prediction(cand_rows, self.dataset)
        if (
            reference_detected != self.reference.detection_image_ids
            or candidate_detected != self.candidate.detection_image_ids
        ):
            raise ValueError("detection image IDs were forged")
        if self.reference.input_preprocessing_digest != _inference_input_digest(self.reference.target, self.dataset):
            raise ValueError("reference inference input/preprocessing digest mismatch")
        if self.candidate.input_preprocessing_digest != _inference_input_digest(self.candidate.target, self.dataset):
            raise ValueError("candidate inference input/preprocessing digest mismatch")
        if self.reference.output_postprocessing_digest != _inference_output_digest(self.reference.target, ref_rows):
            raise ValueError("reference inference output/postprocessing digest mismatch")
        if self.candidate.output_postprocessing_digest != _inference_output_digest(self.candidate.target, cand_rows):
            raise ValueError("candidate inference output/postprocessing digest mismatch")
        parity = aggregate_parity(ref_rows, cand_rows, [x.id for x in self.dataset.images], self.policy)
        if _canonical(parity) != _canonical(self.parity):
            raise ValueError("validation report parity was forged")
        ids = sorted(item.id for item in self.dataset.images)
        metrics = (*_score(self.dataset, list(ref_rows), ids), *_score(self.dataset, list(cand_rows), ids))
        recorded = (
            self.map_reference,
            self.ap50_reference,
            self.ap75_reference,
            self.map_candidate,
            self.ap50_candidate,
            self.ap75_candidate,
        )
        if any(abs(left - right) > 1e-12 for left, right in zip(metrics, recorded, strict=True)):
            raise ValueError("validation report metrics were forged")
        experiment_path = result / "experiment_report.json"
        experiment_marker = result / "EXPERIMENT_COMMITTED"
        if self.hardware_digest is not None:
            if (
                experiment_path.is_symlink()
                or not experiment_path.is_file()
                or experiment_marker.is_symlink()
                or not experiment_marker.is_file()
            ):
                raise ValueError("canonical experiment evidence is not committed")
            experiment_bytes = experiment_path.read_bytes()
            if experiment_marker.read_text() != hashlib.sha256(experiment_bytes).hexdigest():
                raise ValueError("experiment commit marker does not bind exact report bytes")
            from ..experiment_evidence import ExperimentReport

            experiment = ExperimentReport.model_validate_json(experiment_bytes)
            experiment.verify_validation_binding(self)
        self.__class__.model_validate(self.model_dump(mode="json"))
        return True


def _validate_prediction(rows: list[dict[str, Any]], dataset: DatasetManifest) -> tuple[list[int], list[int]]:
    ids = {x.id: x for x in dataset.images}
    evaluated: list[int] = []
    for row in rows:
        if set(row) != {"image_id", "category_id", "bbox", "score"}:
            raise ValueError("prediction rows must have exactly COCO detection keys")
        image_id, category_id, bbox, score = row["image_id"], row["category_id"], row["bbox"], row["score"]
        if (
            image_id not in ids
            or category_id not in dataset.categories
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("invalid prediction ID/category/score")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            raise ValueError("invalid bbox")
        image = ids[image_id]
        if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > image.width or bbox[1] + bbox[3] > image.height:
            raise ValueError("bbox out of bounds")
        evaluated.append(image_id)
    return sorted(set(evaluated)), sorted(set(evaluated))


def _inference_input_digest(target: ModelTarget, dataset: DatasetManifest) -> str:
    """Bind the exact image bytes and preprocessing contract to a run."""
    manifest = target.manifest()
    return _digest(
        {
            "source_sha256": target.source_sha256,
            "manifest_digest": target.manifest_digest,
            "images": [{"id": item.id, "sha256": item.sha256} for item in dataset.images],
            "preprocessing": manifest.preprocessing.model_dump(mode="json"),
        }
    )


def _inference_output_digest(target: ModelTarget, rows: list[dict[str, Any]]) -> str:
    """Bind output rows to decoder and postprocessing, not just their JSON."""
    manifest = target.manifest()
    return _digest(
        {
            "rows": rows,
            "decoder": manifest.decoder.model_dump(mode="json"),
            "postprocessing": manifest.postprocessing,
            "class_mapping": target.class_mapping,
        }
    )


def _score(dataset: DatasetManifest, rows: list[dict[str, Any]], image_ids: list[int]) -> tuple[float, float, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("install vision-buildkit[validation]") from exc
    # pycocotools indexes ``anns[0]`` in ``loadRes``. Empty detections are
    # valid and have zero AP, rather than being an evaluator exception.
    if not rows:
        return 0.0, 0.0, 0.0
    coco = COCO(dataset.annotations_path)
    # COCO may decorate result annotations; never permit it to mutate the
    # signed prediction rows used by parity/digest verification.
    result = coco.loadRes(json.loads(_canonical(rows)))
    ev = COCOeval(coco, result, "bbox")
    ev.params.imgIds = sorted(image_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1]), float(ev.stats[2])


def _target(value: ModelTarget | dict[str, Any]) -> ModelTarget:
    return value if isinstance(value, ModelTarget) else ModelTarget.model_validate(value)


def runtime_policy_contract(
    target: ModelTarget | dict[str, Any],
    *,
    backend: str,
    provider: str,
    device: str,
    precision: str,
    input_shape: tuple[int, int, int, int] | list[int],
) -> RuntimeExecutionIdentity:
    """Build a target-bound expected runtime contract for ``ValidationPolicy``.

    Callers such as the optimizer must construct this from their independently
    selected candidate fields before validation.  The target hashes are added
    here, preventing a policy from treating a digest copied out of a report as
    an external approval.
    """
    signed_target = _target(target)
    body = {
        "backend": backend,
        "provider": provider,
        "device": device,
        "precision": precision,
        "input_shape": tuple(input_shape),
        "source_sha256": signed_target.source_sha256,
        "manifest_digest": signed_target.manifest_digest,
    }
    contract = RuntimeExecutionIdentity(**(body | {"digest": _digest(body)}))
    _verify_runtime_contract(signed_target, contract)
    return contract


def _validate_materialized(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    dataset: DatasetManifest,
    policy: ValidationPolicy,
    output_dir: str | Path,
    *,
    reference_target: ModelTarget | dict[str, Any],
    candidate_target: ModelTarget | dict[str, Any],
    reference_evaluated_image_ids: list[int],
    candidate_evaluated_image_ids: list[int],
    reference_runtime: RuntimeExecutionIdentity | None = None,
    candidate_runtime: RuntimeExecutionIdentity | None = None,
) -> ValidationReport:
    """Build and atomically publish synthetic/previously materialized rows."""
    if not isinstance(reference, list) or not isinstance(candidate, list):
        raise TypeError("reference/candidate must be COCO prediction lists")
    reference_target, candidate_target = _target(reference_target), _target(candidate_target)
    reference_target.assert_dataset_contract(dataset)
    candidate_target.assert_dataset_contract(dataset)
    _validate_prediction(reference, dataset)
    _validate_prediction(candidate, dataset)
    output = Path(output_dir)

    def synthetic_runtime(target: ModelTarget) -> RuntimeExecutionIdentity:
        body = {
            "backend": "materialized",
            "provider": "materialized",
            "device": target.device,
            "precision": "fp32",
            "input_shape": (1, 3, 1, 1),
            "source_sha256": target.source_sha256,
            "manifest_digest": target.manifest_digest,
        }
        return RuntimeExecutionIdentity(**(body | {"digest": _digest(body)}))

    # Never allow externally supplied runtime facts onto this synthetic path.
    # The sentinel is created here only so reports cannot be relabeled as a
    # public `vision.load` observation.
    if reference_runtime is not None or candidate_runtime is not None:
        raise ValueError("materialized validation cannot accept external runtime identities")
    reference_runtime = synthetic_runtime(reference_target)
    candidate_runtime = synthetic_runtime(candidate_target)
    reservation = _reserve_output(output, _policy_digest(policy))
    try:
        return _publish_materialized(
            reference,
            candidate,
            dataset,
            policy,
            output,
            reservation,
            reference_target,
            candidate_target,
            reference_evaluated_image_ids,
            candidate_evaluated_image_ids,
            reference_runtime,
            candidate_runtime,
            evidence_kind="materialized",
        )
    except Exception:
        _cleanup_owned(reservation)
        raise


def _target_build_evidence(target: ModelTarget, runtime: RuntimeExecutionIdentity) -> tuple[str, str | None]:
    """Recover graph and calibration identities from a verified adjacent build report."""

    graph_digest = target.source_sha256
    calibration_digest = None
    report_path = Path(f"{target.source}.build-report.json")
    if report_path.exists() or report_path.is_symlink():
        if report_path.is_symlink() or not report_path.is_file():
            raise ValueError("candidate build report must be a regular non-symlink file")
        from ..export import BuildReport

        build_report = BuildReport.model_validate_json(report_path.read_text())
        if build_report.self_digest != build_report.finalized().self_digest:
            raise ValueError("candidate build report self digest mismatch")
        if (
            Path(build_report.output).resolve() != Path(target.source).resolve()
            or build_report.output_digest != target.source_sha256
            or build_report.artifact_digests.get(Path(target.source).name) != target.source_sha256
            or build_report.target != runtime.backend
            or build_report.effective_precision != runtime.precision
        ):
            raise ValueError("candidate build report is not bound to the validated runtime artifact")
        graph_digest = build_report.intermediate_digest
        calibration_digest = build_report.calibration_digest
    if runtime.precision == "int8" and calibration_digest is None:
        raise ValueError("observed INT8 validation requires a bound calibration build report")
    return graph_digest, calibration_digest


def _reserve_output(output: Path, policy_digest: str) -> dict[str, Any]:
    """Acquire adjacent lock and stage via a trusted parent dirfd before work."""
    parent = _trusted_directory(output.parent)
    output_name = _basename(output.name)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    parent_stat = os.fstat(parent_fd)
    stage_name = lock_name = None
    stage_fd = -1
    try:
        try:
            os.stat(output_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"validation output already exists: {output_name}")
        token = uuid.uuid4().hex
        stage_name = f".{output_name}.validation-stage-{token}"
        lock_name = f".{output_name}.validation.lock"
        owner = {
            "token": token,
            "pid": os.getpid(),
            "created_at": time.time(),
            "policy_digest": policy_digest,
            "stage_name": stage_name,
            "lock_name": lock_name,
            "output_name": output_name,
        }
        lock_fd = os.open(
            lock_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd
        )
        with os.fdopen(lock_fd, "w", closefd=True) as lock:
            lock.write(_canonical(owner))
            lock.flush()
            os.fsync(lock.fileno())
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        stage_fd = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        _write_bytes_at(stage_fd, ".validation-owner.json", _canonical(owner).encode())
        _fsync_fd(stage_fd)
        return owner | {"parent": parent, "parent_fd": parent_fd, "parent_stat": parent_stat, "stage_fd": stage_fd}
    except Exception:
        if stage_fd >= 0:
            os.close(stage_fd)
        if stage_name:
            try:
                _rmtree_at(parent_fd, stage_name)
            except OSError:
                pass
        if lock_name:
            try:
                os.unlink(lock_name, dir_fd=parent_fd)
            except OSError:
                pass
        os.close(parent_fd)
        raise


def _strict_owner(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "token",
        "pid",
        "created_at",
        "policy_digest",
        "stage_name",
        "lock_name",
        "output_name",
    }:
        return False
    return (
        isinstance(value["token"], str)
        and len(value["token"]) == 32
        and set(value["token"]) <= _HEX64
        and isinstance(value["pid"], int)
        and value["pid"] > 0
        and isinstance(value["created_at"], (int, float))
        and math.isfinite(value["created_at"])
        and isinstance(value["policy_digest"], str)
        and len(value["policy_digest"]) == 64
        and set(value["policy_digest"]) <= _HEX64
        and isinstance(value["stage_name"], str)
        and isinstance(value["lock_name"], str)
        and isinstance(value["output_name"], str)
        and value["stage_name"] == f".{value['output_name']}.validation-stage-{value['token']}"
        and value["lock_name"] == f".{value['output_name']}.validation.lock"
    )


def _owned(reservation: dict[str, Any]) -> bool:
    try:
        stage_owner = json.loads(_read_bytes_at(reservation["stage_fd"], ".validation-owner.json"))
        lock_owner = json.loads(_read_bytes_at(reservation["parent_fd"], reservation["lock_name"]))
        return _strict_owner(stage_owner) and stage_owner == lock_owner == {k: reservation[k] for k in stage_owner}
    except (OSError, json.JSONDecodeError):
        return False


def _cleanup_owned(reservation: dict[str, Any], *, preserve_lock: bool = False) -> None:
    """Remove only a strict matching stage/lock; never touch final output."""
    stage_removed = False
    try:
        if _owned(reservation):
            if reservation.get("stage_fd", -1) >= 0:
                os.close(reservation["stage_fd"])
                reservation["stage_fd"] = -1
            try:
                stage_removed = _rmtree_at(reservation["parent_fd"], reservation["stage_name"])
            except OSError:
                stage_removed = False
            if stage_removed and not preserve_lock:
                try:
                    os.unlink(reservation["lock_name"], dir_fd=reservation["parent_fd"])
                except OSError:
                    pass
    except Exception:
        # Never replace a validation/build/rollback exception in a finally.
        pass
    finally:
        for key in ("result_fd", "stage_fd", "parent_fd"):
            fd = reservation.get(key, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                reservation[key] = -1


def _add_exception_note(exc: Exception, note: str) -> None:
    """Attach diagnostic context on Python 3.10 as well as 3.11+."""
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(note)
    else:
        exc.__notes__ = [*getattr(exc, "__notes__", ()), note]


def _publish_materialized(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    dataset: DatasetManifest,
    policy: ValidationPolicy,
    output: Path,
    reservation: dict[str, Any],
    reference_target: ModelTarget,
    candidate_target: ModelTarget,
    reference_evaluated_image_ids: list[int],
    candidate_evaluated_image_ids: list[int],
    reference_runtime: RuntimeExecutionIdentity,
    candidate_runtime: RuntimeExecutionIdentity,
    *,
    evidence_kind: Literal["observed", "materialized"],
) -> ValidationReport:
    if not _owned(reservation):
        raise RuntimeError("validation output reservation ownership was lost")
    stage_fd = reservation["stage_fd"]
    os.mkdir("result", 0o700, dir_fd=stage_fd)
    result_fd = os.open("result", os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=stage_fd)
    reservation["result_fd"] = result_fd
    published = False
    preserve_lock = False
    try:

        def write(name: str, value: Any) -> str:
            _safe_relative(name)
            raw = _canonical(value)
            _write_bytes_at(result_fd, name, raw.encode())
            return hashlib.sha256(raw.encode()).hexdigest()

        ref_sha = write("reference_predictions.json", reference)
        cand_sha = write("candidate_predictions.json", candidate)
        expected_ids = sorted(item.id for item in dataset.images)
        _, ref_detected = _validate_prediction(reference, dataset)
        _, cand_detected = _validate_prediction(candidate, dataset)
        allowed = set(expected_ids)
        ref_eval, cand_eval = sorted(set(reference_evaluated_image_ids)), sorted(set(candidate_evaluated_image_ids))
        if (
            set(ref_eval) - allowed
            or set(cand_eval) - allowed
            or len(ref_eval) != len(reference_evaluated_image_ids)
            or len(cand_eval) != len(candidate_evaluated_image_ids)
        ):
            raise ValueError("evaluated image IDs must be unique members of the dataset inventory")

        def prediction(
            name: str,
            row_digest: str,
            target: ModelTarget,
            evaluated: list[int],
            detected: list[int],
            runtime: RuntimeExecutionIdentity,
        ) -> PredictionManifest:
            payload = {
                "target": target.model_dump(mode="json"),
                "dataset_digest": dataset.digest,
                "predictions_path": name,
                "predictions_sha256": row_digest,
                "input_preprocessing_digest": _inference_input_digest(target, dataset),
                "output_postprocessing_digest": _inference_output_digest(
                    target, reference if name.startswith("reference") else candidate
                ),
                "runtime": runtime.model_dump(mode="json"),
                "record_path": name.replace("_predictions.json", "_prediction_manifest.json"),
                "evaluated_image_ids": evaluated,
                "detection_image_ids": detected,
            }
            return PredictionManifest(**(payload | {"digest": _digest(payload)}))

        ref = prediction(
            "reference_predictions.json", ref_sha, reference_target, ref_eval, ref_detected, reference_runtime
        )
        cand = prediction(
            "candidate_predictions.json", cand_sha, candidate_target, cand_eval, cand_detected, candidate_runtime
        )
        write(ref.record_path, ref.model_dump(mode="json"))
        write(cand.record_path, cand.model_dump(mode="json"))
        parity = aggregate_parity(reference, candidate, expected_ids, policy)
        mr, a50r, a75r = _score(dataset, reference, expected_ids)
        mc, a50c, a75c = _score(dataset, candidate, expected_ids)
        inventory = set(ref_eval) == allowed and set(cand_eval) == allowed
        gates = {
            "inventory": inventory,
            "coverage": inventory and parity["coverage"] >= policy.min_coverage,
            "parity": (
                not parity["high_conf_lost"]
                and not parity["high_conf_added"]
                and len(parity["class_mismatches"]) <= policy.max_class_mismatches
                and parity["max_score_delta"] <= policy.score_delta
            ),
            "map": mr - mc <= policy.max_map_drop,
        }
        observed_bindings: dict[str, str | None] = {}
        if evidence_kind == "observed":
            candidate_graph_digest, calibration_digest = _target_build_evidence(candidate_target, candidate_runtime)
            observed_bindings = {
                "hardware_digest": _validation_hardware_digest(),
                "harness_digest": _validation_harness_digest(),
                "candidate_graph_digest": candidate_graph_digest,
            }
            if calibration_digest is not None:
                observed_bindings["calibration_digest"] = calibration_digest
        body = {
            "dataset": dataset.model_dump(mode="json"),
            "reference": ref.model_dump(mode="json"),
            "candidate": cand.model_dump(mode="json"),
            "policy": policy.model_dump(mode="json"),
            "policy_digest": _policy_digest(policy),
            "evidence_kind": evidence_kind,
            "publishable": (
                evidence_kind == "observed"
                and policy.reference_runtime is not None
                and policy.candidate_runtime is not None
            ),
            "parity": parity,
            "map_reference": mr,
            "map_candidate": mc,
            "ap50_reference": a50r,
            "ap50_candidate": a50c,
            "ap75_reference": a75r,
            "ap75_candidate": a75c,
            "gates": gates,
            "passed": all(gates.values()),
        } | observed_bindings
        report = ValidationReport(**(body | {"digest": _digest(body)}))
        report_bytes = _canonical(report.model_dump(mode="json"))
        _write_bytes_at(result_fd, "validation_report.json", report_bytes.encode())
        if evidence_kind == "observed":
            from ..experiment_evidence import experiment_report_from_validation

            experiment = experiment_report_from_validation(report)
            experiment_bytes = _canonical(experiment.model_dump(mode="json")).encode()
            experiment_digest = hashlib.sha256(experiment_bytes).hexdigest()
            _write_bytes_at(result_fd, "experiment_report.json", experiment_bytes)
            _write_bytes_at(result_fd, "EXPERIMENT_COMMITTED", experiment_digest.encode())
        _write_bytes_at(result_fd, "COMMITTED", hashlib.sha256(report_bytes.encode()).hexdigest().encode())
        # Complete every public byte in the private stage before publication.
        _linkat(result_fd, "validation_report.json", stage_fd, "validation_report.json")
        if evidence_kind == "observed":
            _linkat(result_fd, "experiment_report.json", stage_fd, "experiment_report.json")
            _linkat(result_fd, "EXPERIMENT_COMMITTED", stage_fd, "EXPERIMENT_COMMITTED")
        _write_bytes_at(stage_fd, "COMMITTED", hashlib.sha256(report_bytes.encode()).hexdigest().encode())
        _fsync_fd(result_fd)
        _fsync_fd(stage_fd)
        _verify_stage_fds(stage_fd, result_fd, report, report.policy_digest)
        if not _owned(reservation):
            raise RuntimeError("validation output reservation ownership was lost")
        # Parent path must still resolve to the inode held by parent_fd.
        current = os.stat(reservation["parent"], follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (reservation["parent_stat"].st_dev, reservation["parent_stat"].st_ino):
            raise RuntimeError("trusted publication parent was replaced")
        _noreplace_rename(reservation["parent_fd"], reservation["stage_name"], reservation["output_name"])
        published = True
        _fsync_fd(reservation["parent_fd"])
        current = os.stat(reservation["parent"], follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (reservation["parent_stat"].st_dev, reservation["parent_stat"].st_ino):
            # Keep the requested output unavailable on a swapped pathname.
            _noreplace_rename(reservation["parent_fd"], reservation["output_name"], reservation["stage_name"])
            published = False
            raise RuntimeError("trusted publication parent was replaced after rename; rolled back")
        try:
            report.verify(
                reservation["parent"] / reservation["output_name"],
                report.policy_digest,
                allow_materialized=evidence_kind == "materialized",
            )
        except Exception:
            _noreplace_rename(reservation["parent_fd"], reservation["output_name"], reservation["stage_name"])
            published = False
            raise
        os.unlink(reservation["lock_name"], dir_fd=reservation["parent_fd"])
        _fsync_fd(reservation["parent_fd"])
        os.close(result_fd)
        os.close(reservation["stage_fd"])
        os.close(reservation["parent_fd"])
        reservation["stage_fd"] = -1
        reservation["parent_fd"] = -1
        return report
    except Exception as exc:
        if published and reservation.get("parent_fd", -1) >= 0:
            try:
                _noreplace_rename(reservation["parent_fd"], reservation["output_name"], reservation["stage_name"])
                published = False
            except Exception as rollback_error:
                preserve_lock = True
                _add_exception_note(exc, f"validation publication rollback failed: {rollback_error!r}")
        raise
    finally:
        if reservation.get("parent_fd", -1) >= 0:
            _cleanup_owned(reservation, preserve_lock=preserve_lock)


def _model_rows(
    model: Any, target: ModelTarget, dataset: DatasetManifest
) -> tuple[list[dict[str, Any]], list[int], RuntimeExecutionIdentity]:
    """Run one already-loaded public model over the complete signed inventory."""
    rows: list[dict[str, Any]] = []
    evaluated: list[int] = []
    observed: RuntimeExecutionIdentity | None = None
    mapping = {int(key): value for key, value in target.class_mapping.items()}
    for image in dataset.images:
        result = model(image.path)
        if not isinstance(result, DetectionResult):
            raise TypeError("vision.load target must return DetectionResult")
        metadata = result.metadata
        if not _RUNTIME_KEYS <= set(metadata):
            raise ValueError("inference result lacks complete observed runtime metadata")
        if metadata["backend"] == "materialized" or metadata["provider"] == "materialized":
            raise ValueError("public validation rejects materialized runtime sentinels")
        session_identity = getattr(model, "runtime_identity", None)
        if not callable(session_identity):
            raise ValueError("loaded model does not expose a runtime session identity")
        try:
            trusted = session_identity()
        except Exception as exc:
            raise ValueError("loaded model runtime session identity is unavailable") from exc
        if not isinstance(trusted, dict) or set(trusted) != _RUNTIME_KEYS:
            raise ValueError("loaded model runtime session identity is incomplete")
        if trusted["backend"] == "materialized" or trusted["provider"] == "materialized":
            raise ValueError("public validation rejects materialized runtime sentinels")
        if any(metadata[key] != trusted[key] for key in _RUNTIME_KEYS):
            raise ValueError("inference result runtime metadata disagrees with loaded session identity")
        identity_body = {
            "backend": metadata["backend"],
            "provider": metadata["provider"],
            "device": metadata["device"],
            "precision": metadata["precision"],
            "input_shape": tuple(metadata["input_shape"]),
            "source_sha256": target.source_sha256,
            "manifest_digest": target.manifest_digest,
        }
        current = RuntimeExecutionIdentity(**(identity_body | {"digest": _digest(identity_body)}))
        _verify_runtime_contract(target, current)
        if observed is not None and current != observed:
            raise ValueError("observed runtime identity changed within dataset validation")
        observed = current
        evaluated.append(image.id)
        for box, score, cls in zip(result.boxes, result.scores, result.classes, strict=True):
            if cls not in mapping:
                raise ValueError("detection class is absent from target class mapping")
            x1, y1, x2, y2 = box.as_xyxy()
            rows.append(
                {
                    "image_id": image.id,
                    "category_id": mapping[cls],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    if observed is None:
        raise ValueError("dataset validation requires at least one observed inference")
    return rows, evaluated, observed


def validate_dataset(
    reference_target: ModelTarget | dict[str, Any],
    candidate_target: ModelTarget | dict[str, Any],
    dataset: DatasetManifest,
    policy: ValidationPolicy,
    output_dir: str | Path,
) -> ValidationReport:
    """Validate two signed targets through the public ``vision.load`` API.

    Prediction-row validation remains private to prevent callers from claiming
    a model ran when they only supplied a JSON file.
    """
    reference, candidate = _target(reference_target), _target(candidate_target)
    reference.assert_dataset_contract(dataset)
    candidate.assert_dataset_contract(dataset)
    output = Path(output_dir)
    reservation = _reserve_output(output, _policy_digest(policy))
    import vision

    try:
        reference_model = vision.load(reference.source, device=reference.device, manifest=reference.manifest())
        candidate_model = vision.load(candidate.source, device=candidate.device, manifest=candidate.manifest())
        reference_rows, reference_ids, reference_runtime = _model_rows(reference_model, reference, dataset)
        candidate_rows, candidate_ids, candidate_runtime = _model_rows(candidate_model, candidate, dataset)
        return _publish_materialized(
            reference_rows,
            candidate_rows,
            dataset,
            policy,
            output,
            reservation,
            reference,
            candidate,
            reference_ids,
            candidate_ids,
            reference_runtime,
            candidate_runtime,
            evidence_kind="observed",
        )
    except Exception:
        _cleanup_owned(reservation)
        raise


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_orphan_validations(parent: str | Path, *, min_age_seconds: float = 3600) -> list[Path]:
    """Recover only strict stale stage+lock pairs, never a final output."""
    parent_path = _trusted_directory(Path(parent))
    parent_fd = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    now, removed = time.time(), []
    try:
        for lock_name in os.listdir(parent_fd):
            if not lock_name.endswith(".validation.lock"):
                continue
            try:
                owner = json.loads(_read_bytes_at(parent_fd, lock_name))
                if not _strict_owner(owner) or owner["lock_name"] != lock_name:
                    continue
                created, pid = float(owner["created_at"]), int(owner["pid"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
            if now - created < min_age_seconds or _pid_alive(pid):
                continue
            try:
                info = os.stat(owner["stage_name"], dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                info = None
            if info is not None and stat.S_ISDIR(info.st_mode):
                try:
                    _rmtree_at(parent_fd, owner["stage_name"])
                except OSError:
                    continue
                removed.append(parent_path / owner["stage_name"])
                os.unlink(lock_name, dir_fd=parent_fd)
                _fsync_fd(parent_fd)
                continue
            elif info is not None:
                continue
            # A post-publish crash leaves only an orphan lock.  Only remove it
            # if the completed final report is still policy-bound to this lock.
            try:
                output_fd = os.open(
                    owner["output_name"],
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
                try:
                    report = json.loads(_read_bytes_at(output_fd, "validation_report.json"))
                finally:
                    os.close(output_fd)
                if report.get("policy_digest") != owner["policy_digest"]:
                    continue
            except (OSError, json.JSONDecodeError):
                # The strict lock is stale and no stage/final output exists.
                try:
                    os.unlink(lock_name, dir_fd=parent_fd)
                    _fsync_fd(parent_fd)
                except OSError:
                    pass
                continue
            os.unlink(lock_name, dir_fd=parent_fd)
        _fsync_fd(parent_fd)
    finally:
        os.close(parent_fd)
    return removed
