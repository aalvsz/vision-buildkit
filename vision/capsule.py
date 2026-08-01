# ruff: noqa: E501
"""Reproducible, fail-closed Vision Capsule publication and selection.

Release capsules contain *path-free integrity snapshots*, not authentication or
a pointer to mutable optimizer state.  A release is selectable only when every material, evidence
snapshot and runtime constraint agrees with the content-addressed index.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .manifest import VisionManifest, manifest_from_contract, validate_contract_equivalence
from .optimization import SearchReport, _accepted, load_search_report, verify_resolved_material


def _canonical(value: Any) -> str:
    def finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("capsule JSON cannot contain non-finite numbers")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                finite(child)

    finite(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _path_free(value: Any) -> None:
    """Reject paths recursively: a snapshot must survive evidence-root moves."""
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            if name in {"path", "file", "filename", "filepath"} or name.endswith(("_path", "_file")):
                raise ValueError("integrity snapshot must be path-free")
            _path_free(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _path_free(item)
    elif isinstance(value, str) and (value.startswith("/") or value.startswith("~") or "\\" in value):
        raise ValueError("integrity snapshot must be path-free")


def _portable(value: Any) -> Any:
    """Remove root-local locations before deriving a semantic identity."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key).lower()
            if name in {
                "path",
                "file",
                "filename",
                "filepath",
                "digest",
                "self_digest",
                "identity",
                "contract",
            } or name.endswith(("_path", "_file", "_digest", "_identity", "_contract")):
                continue
            result[key] = _portable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, str) and (value.startswith("/") or value.startswith("~") or "\\" in value):
        return None
    return value


def _semantic_digest(value: Any) -> str:
    portable = _portable(value)
    _path_free(portable)
    return _digest(portable)


def _hex(value: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("expected lowercase SHA-256 digest")
    return value


def checksum(path: Path) -> str:
    item = _trusted(path)
    digest = hashlib.sha256()
    with item.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_tensorrt_build_receipt(receipt: dict[str, Any], report, contract_digest: str, artifact_digest: str) -> None:
    """Validate the portable TensorRT compiler attestation stored in a capsule."""
    expected = {
        "schema",
        "source_plugin",
        "source_plugin_version",
        "target_plugin",
        "target_plugin_version",
        "source_sha256",
        "onnx_sha256",
        "onnx_lineage",
        "onnx_lineage_sha256",
        "contract_json",
        "engine_sha256",
        "build_log_sha256",
        "build_log",
        "build_command",
        "requested_precision",
        "effective_precision",
        "dynamic",
        "interface",
        "build_host",
    }
    if set(receipt) != expected or receipt.get("schema") != 1:
        raise ValueError("TensorRT build receipt schema mismatch")
    if (
        receipt["target_plugin"] != "tensorrt"
        or receipt["target_plugin_version"] != report.target_plugin_version
        or receipt["source_plugin"] != report.source_plugin
        or receipt["source_plugin_version"] != report.source_plugin_version
        or receipt["source_sha256"] != report.source_digest
        or receipt["engine_sha256"] != artifact_digest
        or receipt["onnx_lineage"].get("contract_digest") != contract_digest
    ):
        raise ValueError("TensorRT build receipt cross-binding mismatch")
    from .contract import VisionContract

    if VisionContract.model_validate_json(receipt["contract_json"]).digest != contract_digest:
        raise ValueError("TensorRT receipt contract is not canonical")
    contract = VisionContract.model_validate_json(receipt["contract_json"])
    if contract.input.layout != "NCHW":
        raise ValueError("TensorRT receipt requires an NCHW VisionContract")
    expected_lineage = {
        "schema": 1,
        "source_sha256": report.source_digest,
        "onnx_sha256": receipt["onnx_sha256"],
        "source_plugin": report.source_plugin,
        "source_plugin_version": report.source_plugin_version,
        "contract_digest": contract_digest,
    }
    if (
        receipt["onnx_lineage"] != expected_lineage
        or hashlib.sha256(_canonical(receipt["onnx_lineage"]).encode()).hexdigest() != receipt["onnx_lineage_sha256"]
    ):
        raise ValueError("TensorRT receipt ONNX lineage binding mismatch")
    interface = receipt["interface"]
    if interface != {
        "input": {
            "name": contract.input.input_name,
            "shape": list(contract.input.shape),
            "dtype": contract.input.dtype,
        },
        "outputs": [item.model_dump(mode="json") for item in contract.raw_output.tensors],
    }:
        raise ValueError("TensorRT receipt interface contradicts VisionContract")
    if receipt["dynamic"] or report.runtime.dynamic:
        raise ValueError("TensorRT receipt/certification must be static")
    expected_effective = "fp32_tactics_no_tf32" if receipt["requested_precision"] == "fp32" else "mixed_fp16_fp32"
    if (
        receipt["requested_precision"] != report.runtime.precision
        or receipt["effective_precision"] != expected_effective
    ):
        raise ValueError("TensorRT receipt precision contradicts runtime evidence")
    expected_command = ["<trtexec>", "--onnx=<verified-onnx>", "--saveEngine=<artifact-engine>"]
    if receipt["requested_precision"] == "fp32":
        expected_command.append("--noTF32")
    elif receipt["requested_precision"] == "fp16":
        expected_command.append("--fp16")
    else:
        raise ValueError("TensorRT receipt precision is unsupported")
    expected_command.append("--skipInference")
    if receipt["build_command"] != expected_command:
        raise ValueError("TensorRT receipt command contradicts certified precision policy")
    if hashlib.sha256(receipt["build_log"].encode()).hexdigest() != receipt["build_log_sha256"]:
        raise ValueError("TensorRT receipt log digest mismatch")
    if any("/" in value or "\\" in value for value in receipt["build_command"]):
        raise ValueError("TensorRT receipt command contains a host path")
    host = receipt["build_host"]
    required_host = {
        "os",
        "architecture",
        "builder_target",
        "trtexec_sha256",
        "trtexec_version",
        "cuda_device",
        "cuda_compute_capability",
        "cuda_runtime",
        "tensorrt_python_version",
        "driver_or_jetpack",
    }
    if not required_host <= set(host) or any(not host[key] or host[key] == "unavailable" for key in required_host):
        raise ValueError("TensorRT receipt compiler identity is incomplete")
    if (
        host["builder_target"] != f"{host['os']}-{host['architecture']}"
        or len(host["trtexec_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in host["trtexec_sha256"].lower())
    ):
        raise ValueError("TensorRT receipt compiler identity is malformed")
    runtime = report.runtime
    expected_resolution = (contract.input.shape[2], contract.input.shape[3])
    expected_batch = contract.input.shape[0]
    if (
        runtime.backend != "tensorrt"
        or runtime.provider != "TensorRTExecutionProvider"
        or runtime.device != "cuda"
        or runtime.runtime_version != host["tensorrt_python_version"]
        or runtime.runtime_version != host["trtexec_version"]
        or runtime.resolution != expected_resolution
        or runtime.batch != expected_batch
        or not {"cuda_device", "cuda_compute_capability", "cuda_runtime", "driver_or_jetpack"} <= set(runtime.hardware)
        or any(runtime.hardware[key] != host[key] for key in runtime.hardware.keys() & host.keys())
    ):
        raise ValueError("TensorRT receipt build/runtime identity mismatch")


def _trusted(path: str | Path) -> Path:
    item = Path(path)
    if item.is_symlink() or not item.is_file():
        raise ValueError("capsule material must be a regular non-symlink file")
    if any(parent.is_symlink() for parent in item.parents):
        raise ValueError("capsule material has symlink ancestor")
    return item


def _safe(root: Path, rel: str) -> Path:
    value = PurePosixPath(rel)
    if not rel or value.is_absolute() or ".." in value.parts or "\\" in rel:
        raise ValueError(f"unsafe capsule path: {rel!r}")
    return root.joinpath(*value.parts)


def _inventory(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("symlinks are forbidden in capsule")
        rel = item.relative_to(root).as_posix()
        if item.is_file() and rel not in {"checksums.json", "capsule.json"}:
            entries[rel] = checksum(item)
    return dict(sorted(entries.items()))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Snapshot(Strict):
    digest: str | None = None

    def _set_digest(self) -> None:
        expected = _digest(self.model_dump(exclude={"digest"}, mode="json"))
        if self.digest is not None and self.digest != expected:
            raise ValueError("snapshot digest mismatch")
        object.__setattr__(self, "digest", expected)


class CapsuleReleaseSnapshot(_Snapshot):
    selected_candidate_id: str
    search_digest: str
    dataset_digest: str
    source_model_digest: str
    plan: dict[str, Any]

    @model_validator(mode="after")
    def _valid(self):
        for value in (self.search_digest, self.dataset_digest, self.source_model_digest):
            _hex(value)
        _path_free(self.plan)
        self._set_digest()
        return self


class ValidationSnapshot(_Snapshot):
    report_digest: str
    policy_digest: str
    dataset_digest: str
    candidate_source_sha256: str
    candidate_manifest_digest: str
    reference_source_sha256: str
    publishable: bool
    passed: bool
    evidence_kind: Literal["observed"]

    @model_validator(mode="after")
    def _valid(self):
        for value in (
            self.report_digest,
            self.policy_digest,
            self.dataset_digest,
            self.candidate_source_sha256,
            self.candidate_manifest_digest,
            self.reference_source_sha256,
        ):
            _hex(value)
        if not self.publishable or not self.passed:
            raise ValueError("release validation must be publishable and passed")
        self._set_digest()
        return self


class BenchmarkSnapshot(_Snapshot):
    report_digest: str
    artifact_digest: str
    fingerprint: dict[str, Any]
    resolution: tuple[int, int]
    batch: int = Field(gt=0)
    dynamic: bool
    profiles: tuple[dict[str, Any], ...]
    samples_ms: tuple[float, ...]
    repeats: int = Field(gt=0)
    p50_ms: float = Field(gt=0)
    p95_ms: float = Field(gt=0)
    mean_ms: float = Field(gt=0)
    throughput_items_per_s: float = Field(gt=0)
    mode: Literal["execution"]

    @model_validator(mode="after")
    def _valid(self):
        _hex(self.report_digest)
        _hex(self.artifact_digest)
        if self.fingerprint.get("mode") != "execution":
            raise ValueError("benchmark snapshot must be path-free execution evidence")
        _path_free(self.fingerprint)
        if len(self.samples_ms) != self.repeats or any(value <= 0 for value in self.samples_ms):
            raise ValueError("benchmark snapshot needs positive samples")
        ordered = sorted(self.samples_ms)

        def percentile(percent: float) -> float:
            position = (len(ordered) - 1) * percent / 100
            lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

        mean = sum(self.samples_ms) / len(self.samples_ms)
        if any(
            abs(actual - expected) > 1e-10
            for actual, expected in (
                (self.p50_ms, percentile(50)),
                (self.p95_ms, percentile(95)),
                (self.mean_ms, mean),
                (self.throughput_items_per_s, self.batch * 1000 / mean),
            )
        ):
            raise ValueError("benchmark snapshot metrics must be derived from raw samples")
        self._set_digest()
        return self


class BuildSnapshot(_Snapshot):
    resolved_digest: str
    build_digest: str
    artifact_digest: str
    manifest_digest: str
    target_digest: str
    candidate_id: str
    backend: str
    precision: str
    device: str
    provider: str
    resolution: tuple[int, int]
    batch: int = Field(gt=0)
    dynamic: bool
    profiles: tuple[dict[str, Any], ...]
    parts: tuple[CapsulePart, ...] = ()
    build_only: bool = False

    @model_validator(mode="after")
    def _valid(self):
        for value in (
            self.resolved_digest,
            self.build_digest,
            self.artifact_digest,
            self.manifest_digest,
            self.target_digest,
        ):
            _hex(value)
        if self.build_only:
            raise ValueError("compile-only builds cannot enter a release capsule")
        self._set_digest()
        return self


class CapsulePart(Strict):
    logical_path: str
    sha256: str
    logical_key_digest: str
    executable: bool = False

    @model_validator(mode="after")
    def _valid(self):
        _hex(self.sha256)
        _hex(self.logical_key_digest)
        if self.logical_path.startswith("/") or ".." in PurePosixPath(self.logical_path).parts:
            raise ValueError("unsafe capsule part")
        return self


class CapsuleArtifact(Strict):
    logical_path: str
    sha256: str = Field(min_length=64, max_length=64)
    backend: str
    provider: str
    device: str
    precision: Literal["fp32", "fp16", "int8"]
    resolution: tuple[int, int]
    batch: int = Field(gt=0)
    dynamic: bool
    executable: bool
    benchmark_digest: str | None = None
    hardware_digest: str | None = None
    p50_ms: float | None = Field(default=None, gt=0)
    p95_ms: float | None = Field(default=None, gt=0)
    runtime_version: str | None = None
    mode: Literal["execution", "compile_only"] = "execution"
    profiles: tuple[dict[str, Any], ...] = ()
    build_only: bool = False
    parts: tuple[CapsulePart, ...] = ()

    @model_validator(mode="after")
    def _valid(self):
        _hex(self.sha256)
        if self.executable and (self.mode != "execution" or self.build_only):
            raise ValueError("executable artifact has non-executable build state")
        return self


class CapsuleIndex(Strict):
    schema_version: int = 2
    capsule_id: str = Field(min_length=64, max_length=64)
    mode: Literal["development", "release", "certified"]
    files: dict[str, str]
    artifacts: tuple[CapsuleArtifact, ...]
    release_snapshot_digest: str | None = None
    validation_snapshot_digest: str | None = None
    benchmark_snapshot_digest: str | None = None
    build_snapshot_digest: str | None = None
    search_digest: str | None = None
    validation_digest: str | None = None
    dataset_digest: str | None = None
    policy_digest: str | None = None
    resolved_digest: str | None = None
    benchmark_digest: str | None = None
    build_digest: str | None = None
    contract_digest: str | None = None
    certification_digest: str | None = None
    certification_status: Literal["PASS", "FAIL", "UNTESTED"] | None = None
    source_plugin_id: str | None = None
    source_plugin_version: str | None = None
    target_plugin_id: str | None = None
    target_plugin_version: str | None = None
    runtime_environment_digest: str | None = None
    corpus_digest: str | None = None
    accuracy_digest: str | None = None
    manifest_digest: str | None = None

    @model_validator(mode="after")
    def _valid(self):
        if self.capsule_id != _digest(self.model_dump(exclude={"capsule_id"}, mode="json")):
            raise ValueError("capsule index digest mismatch")
        if tuple(sorted(self.files)) != tuple(self.files):
            raise ValueError("capsule inventory must be sorted")
        if self.mode == "release" and any(
            value is None
            for value in (
                self.release_snapshot_digest,
                self.validation_snapshot_digest,
                self.benchmark_snapshot_digest,
                self.build_snapshot_digest,
                self.search_digest,
                self.validation_digest,
                self.dataset_digest,
                self.policy_digest,
                self.resolved_digest,
                self.benchmark_digest,
                self.build_digest,
            )
        ):
            raise ValueError("release capsule has incomplete provenance")
        if self.mode == "certified" and any(
            value is None
            for value in (
                self.contract_digest,
                self.certification_digest,
                self.certification_status,
                self.source_plugin_id,
                self.source_plugin_version,
                self.target_plugin_id,
                self.target_plugin_version,
                self.runtime_environment_digest,
                self.corpus_digest,
                self.accuracy_digest,
                self.dataset_digest,
                self.policy_digest,
                self.manifest_digest,
            )
        ):
            raise ValueError("certified capsule has incomplete certification binding")
        return self


class RuntimeRequest(Strict):
    backend: str | None = None
    provider: str | None = None
    device: str | None = None
    precision: Literal["fp32", "fp16", "int8"] | None = None
    resolution: tuple[int, int] | None = None
    batch: int | None = Field(default=None, gt=0)
    dynamic: bool | None = None
    hardware_digest: str | None = None
    max_p95_ms: float | None = Field(default=None, gt=0)
    profiles: tuple[dict[str, Any], ...] | None = None
    expected_capsule_id: str | None = None
    allow_fallback: bool = False


class SelectionReport(Strict):
    request: RuntimeRequest
    selected: CapsuleArtifact | None
    reasons: tuple[str, ...]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical(value))
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _copy(source: str | Path, root: Path, logical: str) -> str:
    source = _trusted(source)
    target = _safe(root, logical)
    if target.exists() or target.is_symlink():
        raise ValueError("capsule logical path collision")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if checksum(target) != checksum(source):
        raise ValueError("capsule copy digest mismatch")
    return logical


def _finalize(
    stage: Path,
    output: Path,
    mode: str,
    artifacts: tuple[CapsuleArtifact, ...],
    *,
    capability=None,
    verifier=None,
    **provenance: Any,
) -> Path:
    (stage / ".capsule-stage.json").unlink(missing_ok=True)
    files = _inventory(stage)
    body = {
        "schema_version": 2,
        "mode": mode,
        "files": files,
        "artifacts": artifacts,
        **provenance,
    }
    draft = CapsuleIndex.model_construct(capsule_id="", **body)
    index = CapsuleIndex(capsule_id=_digest(draft.model_dump(exclude={"capsule_id"}, mode="json")), **body)
    _write_json(stage / "checksums.json", files)
    _write_json(stage / "capsule.json", index.model_dump(mode="json"))
    if verifier is not None:
        # Publication may check the just-created content address internally.
        # Callers never get this convenience: a certified capsule needs an
        # externally supplied pin before it can be accepted.
        if mode == "certified":
            verifier(stage, expected_capsule_id=index.capsule_id)
        else:
            verifier(stage)
    from .validation import _noreplace_rename

    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if capability is None:
            _noreplace_rename(fd, stage.name, output.name)
        else:
            held_fd, held_identity, token, lock = capability
            _noreplace_rename(
                fd, stage.name, output.name, lock_path=lock, held_fd=held_fd, held_identity=held_identity, token=token
            )
        os.fsync(fd)
    finally:
        os.close(fd)
    return output


def _publish(output: Path, build):
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("capsule output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(item.is_symlink() for item in (output.parent, *output.parent.parents)):
        raise ValueError("capsule publication parent has symlink ancestor")
    lock = output.parent / f".{output.name}.lock"
    token = uuid.uuid4().hex
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError("capsule publication already in progress") from exc
        owned_stat = os.fstat(fd)
        owned_identity = (owned_stat.st_dev, owned_stat.st_ino)
        lock_data = {"token": token, "pid": os.getpid(), "time": time.time(), "output": output.name, "stage": None}
        os.ftruncate(fd, 0)
        os.write(fd, _canonical(lock_data).encode())
        os.fsync(fd)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, _canonical(lock_data | {"stage": stage.name}).encode())
        os.fsync(fd)
        _write_json(stage / ".capsule-stage.json", {"token": token, "output": output.name})
        try:
            current = os.stat(lock, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != owned_identity:
                raise ValueError("capsule advisory lock pathname was replaced")
            return build(stage, output, (fd, owned_identity, token, lock))
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def recover_orphan_capsules(parent: str | Path, min_age: float = 300.0) -> tuple[str, ...]:
    """Only recover our stale, dead-owner stage/lock pair; never touch foreign data."""
    root, removed = Path(parent), []
    for lock in root.glob(".*.lock") if root.is_dir() else ():
        try:
            data = json.loads(lock.read_text())
            if set(data) != {"token", "pid", "time", "output", "stage"} or time.time() - float(data["time"]) < min_age:
                continue
            try:
                os.kill(int(data["pid"]), 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            for stage in root.glob(f".{data['output']}.stage-*"):
                marker = stage / ".capsule-stage.json"
                owned = {"token": data["token"], "output": data["output"]}
                if (
                    stage.is_dir()
                    and not stage.is_symlink()
                    and stage.name == data["stage"]
                    and (
                        not marker.exists()
                        or (marker.is_file() and not marker.is_symlink() and json.loads(marker.read_text()) == owned)
                    )
                ):
                    shutil.rmtree(stage)
                    removed.append(str(stage))
            # A dead owner may have published then crashed before its cleanup.
            # Compare the token immediately before unlinking a replacement lock.
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return tuple(removed)


def create_development_capsule(
    root: str | Path, manifest: VisionManifest, weights=(), artifacts=(), validation=None
) -> Path:
    def build(stage: Path, output: Path, capability) -> Path:
        manifest.to_yaml(stage / "manifest.yaml")
        for directory in ("original_weights", "artifacts", "evidence"):
            (stage / directory).mkdir()
        for prefix, values in (("original_weights", weights), ("artifacts", artifacts)):
            for value in values:
                item = _trusted(value)
                _copy(item, stage, f"{prefix}/{checksum(item)}{item.suffix.lower()}")
        _write_json(stage / "preprocessing.json", manifest.preprocessing.model_dump(mode="json"))
        _write_json(stage / "postprocessing.json", manifest.postprocessing)
        _write_json(stage / "validation_report.json", validation or {"mode": "development", "publishable": False})
        _write_json(stage / "benchmarks.json", {})
        return _finalize(stage, output, "development", (), capability=capability)

    return _publish(Path(root), build)


def create_capsule(
    root: str | Path, manifest: VisionManifest, weights=(), artifacts=(), validation=None, benchmarks=None
) -> Path:
    return create_development_capsule(root, manifest, weights, artifacts, validation=validation)


def create_certified_capsule(
    root: str | Path,
    manifest: VisionManifest,
    contract,
    report,
    *,
    source: str | Path,
    artifact: str | Path,
    build_receipt: dict[str, Any] | None = None,
    build_artifact: Any | None = None,
    target_plugin: Any | None = None,
    materialized_onnx: Any | None = None,
) -> Path:
    """Publish only a fully content-bound PASS certification using atomic capsule primitives."""
    from .conformance import CertificationReport, ConformanceEvidenceBundle

    report = CertificationReport.model_validate(report.model_dump(mode="json"))
    if report.status != "PASS":
        raise ValueError("only PASS certification reports can be packaged")
    if report.contract_digest != contract.digest:
        raise ValueError("certification report contract binding mismatch")
    source, artifact = _trusted(source), _trusted(artifact)
    if checksum(source) != report.source_digest:
        raise ValueError("certification report source checkpoint digest mismatch")
    if checksum(artifact) != report.artifact_digest:
        raise ValueError("certification report runnable artifact digest mismatch")
    manifest_source = _trusted(manifest.source)
    if checksum(manifest_source) != checksum(artifact):
        raise ValueError("manifest source does not name the certified runnable artifact bytes")

    if report.evidence is None:
        raise ValueError("certified capsule requires a file-backed evidence bundle")
    if report.target_plugin == "tensorrt":
        # A caller-supplied JSON receipt is forgeable.  The target must issue a
        # process-attested PreparedArtifact and reverify its bytes here.
        from .plugins import TensorRTTargetPlugin

        if (
            build_receipt is not None
            or build_artifact is None
            or materialized_onnx is None
            or type(target_plugin) is not TensorRTTargetPlugin
        ):
            raise ValueError("TensorRT certified capsule requires a target-attested PreparedArtifact")
        if (
            getattr(target_plugin, "plugin_id", None) != "tensorrt"
            or not target_plugin.verify_artifact(build_artifact, contract)
            or Path(build_artifact.material_path) != artifact
            or build_artifact.sha256 != report.artifact_digest
        ):
            raise ValueError("TensorRT build artifact attestation is invalid")
        try:
            build_receipt = json.loads(dict(build_artifact.descriptors)["build_receipt_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("TensorRT build artifact lacks canonical receipt") from exc
        onnx_path = Path(materialized_onnx.material_path)
        if onnx_path.is_symlink() or not onnx_path.is_file() or checksum(onnx_path) != build_receipt["onnx_sha256"]:
            raise ValueError("TensorRT materialized ONNX is not bound to build receipt")
        lineage_path = Path(materialized_onnx.lineage_path)
        if (
            lineage_path.is_symlink()
            or not lineage_path.is_file()
            or checksum(lineage_path) != build_receipt["onnx_lineage_sha256"]
        ):
            raise ValueError("TensorRT materialized ONNX lineage is not bound to build receipt")
        _verify_tensorrt_build_receipt(build_receipt, report, contract.digest, report.artifact_digest)
    elif build_receipt is not None:
        raise ValueError("build receipt is currently reserved for TensorRT certified capsules")

    def build(stage: Path, output: Path, capability) -> Path:
        # Recheck the mutable inputs immediately before any copy.  The report is
        # an attestation over bytes, not a permission token.
        report.evidence.verify(
            report.golden,
            report.accuracy,
            report.runtime,
            source_plugin=report.source_plugin,
            target_plugin=report.target_plugin,
        )
        if checksum(source) != report.source_digest or checksum(artifact) != report.artifact_digest:
            raise ValueError("certification material changed before publication")
        for directory in ("original_weights", "artifacts", "evidence"):
            (stage / directory).mkdir()
        source_logical = f"original_weights/{checksum(source)}{source.suffix.lower()}"
        artifact_logical = f"artifacts/{checksum(artifact)}{artifact.suffix.lower()}"
        # A capsule runs its packaged artifact.  Preserve all other manifest
        # semantics while making that material reference explicit.
        # The supplied manifest is only a publication hint.  Certified runtime
        # semantics are generated from VisionContract, so a stale 640 default
        # can never accompany a 32px certified artifact.
        packaged_manifest = manifest_from_contract(artifact_logical, contract, family=manifest.family)
        packaged_manifest.to_yaml(stage / "manifest.yaml")
        contract.to_json(stage / "vision-contract.json")
        # Build a self-contained evidence bundle before deriving *any* report
        # digest.  In particular, the accuracy report redundantly names the
        # source predictions, so copying it byte-for-byte would leave a hidden
        # dependency on the producer's directory.
        evidence_files = report.evidence.model_dump(mode="json")
        for name, item in evidence_files.items():
            if name == "accuracy_report":
                continue
            if isinstance(item, list):
                for number, value in enumerate(item):
                    logical = f"evidence/{name}-{number:04d}.bin"
                    _copy(value["path"], stage, logical)
                    value["path"] = logical
            else:
                logical = f"evidence/{name}.bin"
                _copy(item["path"], stage, logical)
                item["path"] = logical
        source_evidence = evidence_files["source_predictions"]
        original_metrics = json.loads(report.evidence.accuracy_report.verify().read_text())
        # These fields are intentionally not merely advisory.  The report must
        # say exactly which allowlisted packaged file was scored.
        original_metrics["source_predictions_path"] = source_evidence["path"]
        original_metrics["source_predictions_sha256"] = source_evidence["sha256"]
        accuracy_logical = "evidence/accuracy_report.bin"
        _write_json(stage / accuracy_logical, original_metrics)
        evidence_files["accuracy_report"] = {
            "path": accuracy_logical,
            "sha256": checksum(stage / accuracy_logical),
        }
        packaged_accuracy = report.accuracy.model_copy(
            update={"report_digest": evidence_files["accuracy_report"]["sha256"]}
        )
        # `model_copy(update=...)` deliberately skips validation in Pydantic;
        # reparse so every staged member remains an EvidenceFile rather than a
        # mutable dict supplied by the serializer.
        packaged_evidence = ConformanceEvidenceBundle.model_validate(evidence_files)
        # Verify the rebuilt report -> evidence chain in the staging directory
        # before it becomes visible at the destination path.
        packaged_evidence.verify(
            report.golden,
            packaged_accuracy,
            report.runtime,
            source_plugin=report.source_plugin,
            target_plugin=report.target_plugin,
            base=stage,
        )
        packaged = CertificationReport.trusted_create(
            **(
                report.model_dump(exclude={"digest"}, mode="json")
                | {"accuracy": packaged_accuracy, "evidence": packaged_evidence}
            )
        )
        _write_json(stage / "certification-report.json", packaged.model_dump(mode="json"))
        _copy(source, stage, source_logical)
        _copy(artifact, stage, artifact_logical)
        if build_receipt is not None:
            _write_json(stage / "evidence/tensorrt-build-receipt.json", build_receipt)
            _copy(onnx_path, stage, "evidence/tensorrt-materialized.onnx")
            _copy(lineage_path, stage, "evidence/tensorrt-materialized.onnx.lineage.json")
        # These sidecars are projections of the same canonical representation
        # embedded in the manifest, never caller supplied convenience fields.
        _write_json(
            stage / "preprocessing.json",
            {
                "contract_digest": contract.digest,
                "input": contract.input.model_dump(mode="json"),
                "transforms": [item.model_dump(mode="json") for item in contract.preprocessing],
            },
        )
        _write_json(
            stage / "postprocessing.json",
            {
                "contract_digest": contract.digest,
                "raw_output": contract.raw_output.model_dump(mode="json"),
                "result": contract.result.model_dump(mode="json"),
                "labels": list(contract.labels),
                "coco_category_ids": list(contract.coco_category_ids),
            },
        )
        _write_json(stage / "validation_report.json", {"certification_digest": packaged.digest})
        _write_json(stage / "benchmarks.json", {})
        environment_digest = _digest(packaged.environment)
        accuracy_digest = _digest(packaged.accuracy.model_dump(mode="json")) if packaged.accuracy else _digest({})
        certified_artifact = CapsuleArtifact(
            logical_path=artifact_logical,
            sha256=packaged.artifact_digest,
            backend=packaged.runtime.backend,
            provider=packaged.runtime.provider,
            device=packaged.runtime.device,
            precision=packaged.runtime.precision,
            resolution=packaged.runtime.resolution,
            batch=packaged.runtime.batch,
            dynamic=packaged.runtime.dynamic,
            executable=True,
            p50_ms=packaged.runtime.p50_ms,
            p95_ms=packaged.runtime.p95_ms,
            runtime_version=packaged.runtime.runtime_version,
        )
        return _finalize(
            stage,
            output,
            "certified",
            (certified_artifact,),
            capability=capability,
            contract_digest=contract.digest,
            certification_digest=packaged.digest,
            certification_status=packaged.status,
            source_plugin_id=packaged.source_plugin,
            source_plugin_version=packaged.source_plugin_version,
            target_plugin_id=packaged.target_plugin,
            target_plugin_version=packaged.target_plugin_version,
            runtime_environment_digest=environment_digest,
            corpus_digest=packaged.golden.manifest_digest,
            accuracy_digest=accuracy_digest,
            dataset_digest=packaged.accuracy.dataset_digest if packaged.accuracy else None,
            policy_digest=packaged.policy_digest,
            resolved_digest=packaged.artifact_digest,
            manifest_digest=_digest(packaged_manifest.model_dump(mode="json")),
            verifier=verify_capsule,
        )

    return _publish(Path(root), build)


def _snapshots(
    report: SearchReport, selected
) -> tuple[CapsuleReleaseSnapshot, ValidationSnapshot, BenchmarkSnapshot, BuildSnapshot]:
    validation = selected.validation.report
    benchmark = selected.benchmark
    resolved = selected.resolved
    record = verify_resolved_material(resolved, selected.plan, validation.candidate.target, report.request)
    raw_build = json.loads(Path(resolved.build_report_path).read_text())
    upstream = (
        json.loads(Path(record.upstream_build_report_path).read_text()) if record.upstream_build_report_path else {}
    )
    build_only = bool(raw_build.get("build_only", False))
    upstream_build_only = bool(upstream.get("validation", {}).get("build_only", False))
    if benchmark.request.fingerprint.mode != "execution" or build_only:
        raise ValueError("release requires measured execution material, never compile-only")
    if upstream_build_only and not _execution_closes_build_only(validation, benchmark, resolved, record):
        raise ValueError("compile-only build lacks matching observed runtime evidence")
    release = CapsuleReleaseSnapshot(
        selected_candidate_id=selected.plan.candidate_id,
        # The optimizer report itself is pathful.  This is the canonical
        # semantic digest of its selected, portable release facts.
        search_digest=_semantic_digest(
            {"candidate": selected.plan.model_dump(mode="json"), "dataset": validation.dataset.model_dump(mode="json")}
        ),
        dataset_digest=_semantic_digest(validation.dataset.model_dump(mode="json")),
        source_model_digest=report.request.reference_source_model_digest,
        plan=selected.plan.model_dump(mode="json"),
    )
    validation_snapshot = ValidationSnapshot(
        report_digest=_semantic_digest(
            {
                "dataset": validation.dataset.model_dump(mode="json"),
                "policy": validation.policy.model_dump(mode="json"),
                "passed": validation.passed,
                "map": validation.map_candidate,
            }
        ),
        policy_digest=_semantic_digest(validation.policy.model_dump(mode="json")),
        dataset_digest=_semantic_digest(validation.dataset.model_dump(mode="json")),
        candidate_source_sha256=validation.candidate.target.source_sha256,
        candidate_manifest_digest=_semantic_digest(
            {"source": validation.candidate.target.source_sha256, "mapping": validation.candidate.target.class_mapping}
        ),
        reference_source_sha256=validation.reference.target.source_sha256,
        publishable=validation.publishable,
        passed=validation.passed,
        evidence_kind=validation.evidence_kind,
    )
    fp = benchmark.request.fingerprint.model_dump(mode="json")
    fp["mode"] = "execution" if fp["mode"] == "execution" else fp["mode"]
    benchmark_snapshot = BenchmarkSnapshot(
        report_digest=_semantic_digest(
            {"fingerprint": fp, "samples": benchmark.samples_ms, "batch": benchmark.request.batch}
        ),
        artifact_digest=resolved.artifact_digest,
        fingerprint=fp,
        resolution=benchmark.request.resolution,
        batch=benchmark.request.batch,
        dynamic=benchmark.request.dynamic,
        profiles=tuple(x.model_dump(mode="json") for x in benchmark.request.profiles),
        samples_ms=benchmark.samples_ms,
        repeats=benchmark.request.repeats,
        p50_ms=benchmark.p50_ms,
        p95_ms=benchmark.p95_ms,
        mean_ms=benchmark.mean_ms,
        throughput_items_per_s=benchmark.throughput_items_per_s,
        mode="execution",
    )
    raw_parts = upstream.get("artifact_digests", {})
    parts = []
    for ordinal, (path, digest) in enumerate(sorted(raw_parts.items()), start=1):
        name = Path(str(path)).name
        key_digest = _digest({"name": name, "ordinal": ordinal})
        logical_path = _part_logical_path(
            resolved.artifact_path,
            resolved.artifact_digest,
            name,
            ordinal,
            key_digest,
        )
        parts.append(
            CapsulePart(
                logical_path=logical_path,
                sha256=digest,
                logical_key_digest=key_digest,
            )
        )
    build = BuildSnapshot(
        resolved_digest=_semantic_digest(
            {"artifact": resolved.artifact_digest, "plan": selected.plan.model_dump(mode="json")}
        ),
        build_digest=_semantic_digest(
            {"artifact": resolved.artifact_digest, "plan": selected.plan.model_dump(mode="json")}
        ),
        artifact_digest=resolved.artifact_digest,
        manifest_digest=_semantic_digest(
            {"artifact": resolved.artifact_digest, "plan": selected.plan.model_dump(mode="json")}
        ),
        target_digest=_semantic_digest(
            {"artifact": resolved.artifact_digest, "mapping": validation.candidate.target.class_mapping}
        ),
        candidate_id=resolved.candidate_id,
        backend=record.backend,
        precision=record.precision,
        device=record.device,
        provider=record.provider,
        resolution=record.resolution,
        batch=record.batch,
        dynamic=record.dynamic,
        profiles=tuple(x.model_dump(mode="json") for x in record.profiles),
        parts=tuple(parts),
        build_only=build_only,
    )
    return release, validation_snapshot, benchmark_snapshot, build


def _execution_closes_build_only(validation, benchmark, resolved, record) -> bool:
    """Accept a build-stage-only artifact only after exact runtime attestation."""
    runtime = validation.candidate.runtime
    fingerprint = benchmark.request.fingerprint
    return bool(
        validation.evidence_kind == "observed"
        and validation.publishable
        and validation.passed
        and runtime.backend == record.backend
        and runtime.source_sha256 == resolved.artifact_digest
        and fingerprint.mode == "execution"
        and fingerprint.backend == record.backend
        and fingerprint.artifact_digest == resolved.artifact_digest
    )


def _part_logical_path(
    primary_path: str | Path,
    primary_digest: str,
    part_name: str,
    ordinal: int,
    key_digest: str,
) -> str:
    """Place OpenVINO's required weights beside its renamed XML graph."""
    if Path(primary_path).suffix.lower() == ".xml" and Path(part_name).suffix.lower() == ".bin":
        return f"artifacts/{primary_digest}.bin"
    return f"artifacts/parts/{ordinal:04d}-{key_digest[:16]}-{part_name}"


def publish_capsule(root: str | Path, report: SearchReport | str | Path) -> Path:
    report = (
        load_search_report(report)
        if isinstance(report, (str, Path))
        else SearchReport.model_validate(report.model_dump(mode="json"))
    )
    if report.selected_candidate_id is None:
        raise ValueError("release requires selected accepted candidate")
    selected = next(item for item in report.evaluations if item.plan.candidate_id == report.selected_candidate_id)
    baseline = next(item for item in report.evaluations if item.plan == report.request.baseline)
    accepted, reason = _accepted(selected, baseline, report.request.policy, report.request)
    if not accepted or not selected.stages or selected.stages[-1].stage.value != "accepted":
        raise ValueError("release selected candidate is not accepted: " + reason)
    selected.validation.verify(report.request, selected.plan)
    release, validation, benchmark, build_snapshot = _snapshots(report, selected)
    manifest = VisionManifest.from_yaml(selected.validation.report.candidate.target.manifest_path)

    def build(stage: Path, output: Path, capability) -> Path:
        for directory in ("original_weights", "artifacts", "evidence"):
            (stage / directory).mkdir()
        artifact_name = f"{selected.resolved.artifact_digest}{Path(selected.resolved.artifact_path).suffix.lower()}"
        artifact_path = _copy(selected.resolved.artifact_path, stage, f"artifacts/{artifact_name}")
        upstream_record = json.loads(Path(selected.resolved.build_report_path).read_text())
        upstream_path = upstream_record.get("upstream_build_report_path")
        upstream = json.loads(Path(upstream_path).read_text()) if upstream_path else {}
        for part, (path, digest) in zip(
            build_snapshot.parts,
            sorted(upstream.get("artifact_digests", {}).items()),
            strict=True,
        ):
            source_part = Path(path)
            if not source_part.is_absolute():
                source_part = Path(upstream_path).parent / source_part
            if checksum(source_part) != digest:
                raise ValueError("upstream multipart digest mismatch")
            _copy(source_part, stage, part.logical_path)
        source = report.request.reference_target.source
        _copy(
            source,
            stage,
            f"original_weights/{report.request.reference_target.source_sha256}{Path(source).suffix.lower()}",
        )
        manifest.model_copy(update={"source": artifact_path}).to_yaml(stage / "manifest.yaml")
        _write_json(stage / "preprocessing.json", manifest.preprocessing.model_dump(mode="json"))
        _write_json(stage / "postprocessing.json", manifest.postprocessing)
        _write_json(stage / "validation_report.json", validation.model_dump(mode="json"))
        _write_json(stage / "benchmarks.json", benchmark.model_dump(mode="json"))
        _write_json(stage / "evidence/release_snapshot.json", release.model_dump(mode="json"))
        _write_json(stage / "evidence/validation_snapshot.json", validation.model_dump(mode="json"))
        _write_json(stage / "evidence/benchmark_snapshot.json", benchmark.model_dump(mode="json"))
        _write_json(stage / "evidence/build_snapshot.json", build_snapshot.model_dump(mode="json"))
        artifact = CapsuleArtifact(
            logical_path=artifact_path,
            sha256=selected.resolved.artifact_digest,
            backend=selected.plan.backend,
            provider=selected.plan.provider,
            device=selected.plan.device,
            precision=selected.plan.precision,
            resolution=selected.plan.resolution,
            batch=selected.plan.batch,
            dynamic=selected.plan.dynamic,
            executable=True,
            benchmark_digest=benchmark.report_digest,
            hardware_digest=benchmark.fingerprint.get("hardware_digest"),
            p50_ms=benchmark.p50_ms,
            p95_ms=benchmark.p95_ms,
            runtime_version=benchmark.fingerprint.get("runtime_version"),
            mode="execution",
            profiles=benchmark.profiles,
            build_only=False,
            parts=build_snapshot.parts,
        )
        return _finalize(
            stage,
            output,
            "release",
            (artifact,),
            capability=capability,
            release_snapshot_digest=release.digest,
            validation_snapshot_digest=validation.digest,
            benchmark_snapshot_digest=benchmark.digest,
            build_snapshot_digest=build_snapshot.digest,
            search_digest=release.search_digest,
            validation_digest=validation.report_digest,
            dataset_digest=release.dataset_digest,
            policy_digest=validation.policy_digest,
            resolved_digest=build_snapshot.resolved_digest,
            benchmark_digest=benchmark.report_digest,
            build_digest=build_snapshot.build_digest,
        )

    return _publish(Path(root), build)


def _layout(root: Path, mode: str) -> None:
    expected_dirs = {"original_weights", "artifacts", "evidence"}
    if mode == "release" and (root / "artifacts" / "parts").exists():
        expected_dirs.add("artifacts/parts")
    base_files = {
        "manifest.yaml",
        "preprocessing.json",
        "postprocessing.json",
        "validation_report.json",
        "benchmarks.json",
        "checksums.json",
        "capsule.json",
    }
    snapshots = {
        "evidence/release_snapshot.json",
        "evidence/validation_snapshot.json",
        "evidence/benchmark_snapshot.json",
        "evidence/build_snapshot.json",
    }
    certified = {"vision-contract.json", "certification-report.json"}
    expected_files = (
        base_files | (snapshots if mode == "release" else set()) | (certified if mode == "certified" else set())
    )
    actual_dirs = {item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_dir()}
    actual_files = {item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()}
    material_files = {
        name
        for name in actual_files
        if name.startswith("original_weights/")
        or name.startswith("artifacts/")
        or (mode == "certified" and name.startswith("evidence/"))
    }
    if any(name.count("/") != 1 and not name.startswith("artifacts/parts/") for name in material_files):
        raise ValueError("capsule materials must not have nested directories")
    if actual_dirs != expected_dirs or actual_files != expected_files | material_files:
        raise ValueError("invalid capsule exact layout")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("symlinks are forbidden in capsule")


def verify_capsule(
    root: str | Path,
    expected_capsule_id: str | None = None,
    allow_development: bool = False,
    **legacy: Any,
) -> CapsuleIndex:
    if isinstance(expected_capsule_id, bool):
        allow_development, expected_capsule_id = expected_capsule_id, None
    allow_development = allow_development or bool(legacy.get("allow_unvalidated", False))
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("capsule root must be a real directory")
    if not (root / "checksums.json").is_file():
        raise ValueError("capsule checksums.json is missing")
    index = CapsuleIndex.model_validate_json((root / "capsule.json").read_text())
    if index.mode == "certified" and not expected_capsule_id:
        raise ValueError("certified capsule verification requires expected_capsule_id from a trusted registry or pin")
    if expected_capsule_id and index.capsule_id != expected_capsule_id:
        raise ValueError("unexpected capsule id")
    sums = json.loads((root / "checksums.json").read_text())
    actual = _inventory(root)
    if sums != actual or index.files != actual:
        raise ValueError("capsule checksum inventory mismatch")
    _layout(root, index.mode)
    VisionManifest.from_yaml(root / "manifest.yaml")
    if index.mode == "certified":
        from .conformance import CertificationReport
        from .contract import VisionContract

        contract = VisionContract.from_path(root / "vision-contract.json")
        report = CertificationReport.model_validate_json((root / "certification-report.json").read_text())
        if report.status != "PASS" or index.certification_status != "PASS":
            raise ValueError("certified capsule report is not PASS")
        bindings = (
            index.contract_digest == contract.digest == report.contract_digest,
            index.certification_digest == report.digest,
            index.source_plugin_id == report.source_plugin,
            index.source_plugin_version == report.source_plugin_version,
            index.target_plugin_id == report.target_plugin,
            index.target_plugin_version == report.target_plugin_version,
            index.runtime_environment_digest == _digest(report.environment),
            index.corpus_digest == report.golden.manifest_digest,
            index.accuracy_digest == _digest(report.accuracy.model_dump(mode="json")),
            index.dataset_digest == report.accuracy.dataset_digest,
            index.policy_digest == report.policy_digest,
            index.resolved_digest == report.artifact_digest,
            index.manifest_digest == _digest(VisionManifest.from_yaml(root / "manifest.yaml").model_dump(mode="json")),
        )
        if not all(bindings):
            raise ValueError("certified capsule cross-binding mismatch")
        if report.evidence is None or report.runtime is None or report.accuracy is None:
            raise ValueError("certified capsule lacks evidence")
        report.evidence.verify(
            report.golden,
            report.accuracy,
            report.runtime,
            source_plugin=report.source_plugin,
            target_plugin=report.target_plugin,
            base=root,
        )
        if any(
            not item.path.startswith("evidence/") or actual.get(item.path) != item.sha256
            for item in (
                report.evidence.golden_manifest,
                *report.evidence.case_inputs,
                report.evidence.dataset_manifest,
                report.evidence.policy,
                report.evidence.predictions,
                report.evidence.source_predictions,
                report.evidence.accuracy_report,
                report.evidence.runtime_record,
                report.evidence.environment_record,
            )
        ):
            raise ValueError("certified evidence is not an indexed allowlisted capsule file")
        originals = {name: digest for name, digest in actual.items() if name.startswith("original_weights/")}
        artifacts = {name: digest for name, digest in actual.items() if name.startswith("artifacts/")}
        evidence_paths = {
            item.path
            for item in (
                report.evidence.golden_manifest,
                *report.evidence.case_inputs,
                report.evidence.dataset_manifest,
                report.evidence.policy,
                report.evidence.predictions,
                report.evidence.source_predictions,
                report.evidence.accuracy_report,
                report.evidence.runtime_record,
                report.evidence.environment_record,
            )
        }
        if report.target_plugin == "tensorrt":
            if contract.input.layout != "NCHW":
                raise ValueError("certified TensorRT capsule requires an NCHW VisionContract")
            receipt_path = root / "evidence/tensorrt-build-receipt.json"
            if not receipt_path.is_file() or receipt_path.is_symlink():
                raise ValueError("certified TensorRT capsule lacks indexed build receipt")
            _verify_tensorrt_build_receipt(
                json.loads(receipt_path.read_text()), report, contract.digest, report.artifact_digest
            )
            receipt = json.loads(receipt_path.read_text())
            onnx = root / "evidence/tensorrt-materialized.onnx"
            lineage = root / "evidence/tensorrt-materialized.onnx.lineage.json"
            if (
                not onnx.is_file()
                or not lineage.is_file()
                or checksum(onnx) != receipt["onnx_sha256"]
                or checksum(lineage) != receipt["onnx_lineage_sha256"]
                or json.loads(lineage.read_text()) != receipt["onnx_lineage"]
            ):
                raise ValueError("certified TensorRT ONNX evidence binding mismatch")
            evidence_paths.update(
                {
                    "evidence/tensorrt-build-receipt.json",
                    "evidence/tensorrt-materialized.onnx",
                    "evidence/tensorrt-materialized.onnx.lineage.json",
                }
            )
        elif (root / "evidence/tensorrt-build-receipt.json").exists():
            raise ValueError("non-TensorRT certified capsule must not contain a TensorRT build receipt")
        if len(index.artifacts) != 1 or index.artifacts[0].sha256 != report.artifact_digest:
            raise ValueError("certified capsule artifact inventory mismatch")
        if (
            len(originals) != 1
            or set(originals.values()) != {report.source_digest}
            or artifacts != {index.artifacts[0].logical_path: report.artifact_digest}
            or {name for name in actual if name.startswith("evidence/")} != evidence_paths
        ):
            raise ValueError("certified capsule material binding mismatch")
        artifact = index.artifacts[0]
        runtime_axes = {
            "backend": report.runtime.backend,
            "provider": report.runtime.provider,
            "device": report.runtime.device,
            "precision": report.runtime.precision,
            "resolution": report.runtime.resolution,
            "batch": report.runtime.batch,
            "dynamic": report.runtime.dynamic,
            "runtime_version": report.runtime.runtime_version,
            "p50_ms": report.runtime.p50_ms,
            "p95_ms": report.runtime.p95_ms,
        }
        if any(getattr(artifact, name) != expected for name, expected in runtime_axes.items()):
            raise ValueError("certified artifact axes are not bound to runtime evidence")
        packaged_manifest = VisionManifest.from_yaml(root / "manifest.yaml")
        if packaged_manifest.source != artifact.logical_path:
            raise ValueError("certified manifest does not point to packaged artifact")
        validate_contract_equivalence(packaged_manifest, contract)
        expected_preprocessing = {
            "contract_digest": contract.digest,
            "input": contract.input.model_dump(mode="json"),
            "transforms": [item.model_dump(mode="json") for item in contract.preprocessing],
        }
        expected_postprocessing = {
            "contract_digest": contract.digest,
            "raw_output": contract.raw_output.model_dump(mode="json"),
            "result": contract.result.model_dump(mode="json"),
            "labels": list(contract.labels),
            "coco_category_ids": list(contract.coco_category_ids),
        }
        if (
            json.loads((root / "preprocessing.json").read_text()) != expected_preprocessing
            or json.loads((root / "postprocessing.json").read_text()) != expected_postprocessing
        ):
            raise ValueError("certified sidecars contradict canonical VisionContract")
        return index
    if index.mode == "development":
        if not allow_development:
            raise ValueError("Capsule is unvalidated development material")
        return index
    release = CapsuleReleaseSnapshot.model_validate_json((root / "evidence/release_snapshot.json").read_text())
    validation = ValidationSnapshot.model_validate_json((root / "evidence/validation_snapshot.json").read_text())
    benchmark = BenchmarkSnapshot.model_validate_json((root / "evidence/benchmark_snapshot.json").read_text())
    build = BuildSnapshot.model_validate_json((root / "evidence/build_snapshot.json").read_text())
    indexed_snapshots = (
        index.release_snapshot_digest,
        index.validation_snapshot_digest,
        index.benchmark_snapshot_digest,
        index.build_snapshot_digest,
    )
    actual_snapshots = (release.digest, validation.digest, benchmark.digest, build.digest)
    if indexed_snapshots != actual_snapshots:
        raise ValueError("release snapshot digest binding mismatch")
    indexed_provenance = (
        index.search_digest,
        index.validation_digest,
        index.dataset_digest,
        index.policy_digest,
        index.resolved_digest,
        index.benchmark_digest,
        index.build_digest,
    )
    actual_provenance = (
        release.search_digest,
        validation.report_digest,
        release.dataset_digest,
        validation.policy_digest,
        build.resolved_digest,
        benchmark.report_digest,
        build.build_digest,
    )
    if indexed_provenance != actual_provenance:
        raise ValueError("release provenance binding mismatch")
    if not (
        release.selected_candidate_id == build.candidate_id
        and build.artifact_digest == benchmark.artifact_digest == validation.candidate_source_sha256
    ):
        raise ValueError("release snapshots bind different artifact material")
    expected_artifacts = {artifact.logical_path for artifact in index.artifacts}
    expected_parts = {part.logical_path for artifact in index.artifacts for part in artifact.parts}
    actual_artifacts = {name for name in actual if name.startswith("artifacts/")}
    if actual_artifacts != expected_artifacts | expected_parts:
        raise ValueError("release capsule has unreferenced artifact material")
    originals = {name: digest for name, digest in actual.items() if name.startswith("original_weights/")}
    if len(originals) != 1 or set(originals.values()) != {release.source_model_digest}:
        raise ValueError("release capsule original weight binding mismatch")
    for artifact in index.artifacts:
        if artifact.logical_path not in actual or actual[artifact.logical_path] != artifact.sha256:
            raise ValueError("capsule artifact binding mismatch")
        if not artifact.executable or artifact.mode != "execution" or artifact.build_only:
            raise ValueError("release capsule includes non-executable artifact")
        if artifact.sha256 != build.artifact_digest or artifact.benchmark_digest != benchmark.report_digest:
            raise ValueError("artifact is not bound to release evidence")
        fingerprint = benchmark.fingerprint
        axes = {
            "backend": benchmark.fingerprint.get("backend"),
            "provider": benchmark.fingerprint.get("provider"),
            "device": benchmark.fingerprint.get("device"),
            "precision": benchmark.fingerprint.get("precision"),
            "resolution": benchmark.resolution,
            "batch": benchmark.batch,
            "dynamic": benchmark.dynamic,
            "profiles": benchmark.profiles,
            "hardware_digest": fingerprint.get("hardware_digest"),
            "runtime_version": fingerprint.get("runtime_version"),
            "p50_ms": benchmark.p50_ms,
            "p95_ms": benchmark.p95_ms,
        }
        if any(getattr(artifact, name) != expected for name, expected in axes.items()):
            raise ValueError("artifact axes are not bound to benchmark evidence")
        build_axes = {
            "backend": build.backend,
            "provider": build.provider,
            "device": build.device,
            "precision": build.precision,
            "resolution": build.resolution,
            "batch": build.batch,
            "dynamic": build.dynamic,
            "profiles": build.profiles,
        }
        if any(getattr(artifact, name) != expected for name, expected in build_axes.items()):
            raise ValueError("artifact axes are not bound to build evidence")
        if artifact.parts != build.parts:
            raise ValueError("artifact multipart binding mismatch")
        for part in artifact.parts:
            if actual.get(part.logical_path) != part.sha256:
                raise ValueError("capsule multipart digest mismatch")
    return index


def _strict_mismatches(artifact: CapsuleArtifact, request: RuntimeRequest) -> list[str]:
    mismatches: list[str] = []
    for name in ("backend", "provider", "device", "precision", "resolution", "batch", "dynamic", "hardware_digest"):
        wanted = getattr(request, name)
        if wanted is not None and wanted != getattr(artifact, name):
            mismatches.append(name)
    if not artifact.executable or artifact.mode != "execution" or artifact.build_only:
        mismatches.append("execution")
    if request.max_p95_ms is not None and (artifact.p95_ms is None or artifact.p95_ms > request.max_p95_ms):
        mismatches.append("p95")
    if request.profiles is not None and artifact.profiles != request.profiles:
        mismatches.append("profiles")
    return mismatches


def select_runtime(root: str | Path, request: RuntimeRequest) -> SelectionReport:
    preliminary = verify_capsule(root, expected_capsule_id=request.expected_capsule_id, allow_development=True)
    if preliminary.mode not in {"release", "certified"}:
        raise ValueError("unvalidated development capsule is not selectable")
    if not request.expected_capsule_id:
        raise ValueError("runtime selection requires expected_capsule_id pin")
    index = verify_capsule(root, expected_capsule_id=request.expected_capsule_id)
    reasons, candidates = [], []
    for artifact in index.artifacts:
        mismatches = _strict_mismatches(artifact, request)
        if mismatches:
            reasons.append(f"{artifact.logical_path}: " + ",".join(mismatches))
        else:
            candidates.append(artifact)
    # Fallback is intentionally only a selection preference: all semantic,
    # executable, profile, hardware and SLO constraints remain hard above.
    if not candidates and request.allow_fallback:
        for artifact in index.artifacts:
            mismatches = _strict_mismatches(artifact, request)
            if not mismatches:
                candidates.append(artifact)
        reasons.append("fallback did not relax execution, hardware, profile, or SLO constraints")
    selected = (
        min(candidates, key=lambda item: (item.p50_ms or float("inf"), item.p95_ms or float("inf"), item.logical_path))
        if candidates
        else None
    )
    return SelectionReport(request=request, selected=selected, reasons=tuple(sorted(reasons)))


def select_artifact(root: str | Path, backend: str, expected_capsule_id: str, allow_fallback: bool = False) -> Path:
    result = select_runtime(
        root,
        RuntimeRequest(
            backend=backend,
            expected_capsule_id=expected_capsule_id,
            allow_fallback=allow_fallback,
        ),
    )
    if result.selected is None:
        raise ValueError(f"No release {backend} artifact in capsule")
    return _safe(Path(root), result.selected.logical_path)


__all__ = [
    "BenchmarkSnapshot",
    "BuildSnapshot",
    "CapsuleArtifact",
    "CapsuleIndex",
    "CapsuleReleaseSnapshot",
    "CapsuleMode",
    "RuntimeRequest",
    "SelectionReport",
    "ValidationSnapshot",
    "create_capsule",
    "create_development_capsule",
    "publish_capsule",
    "recover_orphan_capsules",
    "select_artifact",
    "select_runtime",
    "verify_capsule",
]
CapsuleMode = Literal["development", "release", "certified"]
