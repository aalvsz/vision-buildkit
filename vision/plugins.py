# ruff: noqa: E501
"""Provider-neutral plugin contracts and fail-closed registry."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from .contract import Strict, VisionContract
from .manifest import CalibrationManifest
from .preprocess import PreparedInput, preprocess
from .types import Box, DetectionResult


@dataclass(frozen=True)
class PreparedSource:
    material_path: Path
    sha256: str
    plugin_id: str
    plugin_version: str
    contract_digest: str
    descriptors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PreparedArtifact:
    material_path: Path
    sha256: str
    plugin_id: str
    plugin_version: str
    contract_digest: str
    descriptors: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class MaterializedOnnx:
    """A rehashable ONNX export plus its source-to-export lineage record.

    This is deliberately not a certification result.  Targets still validate the
    exported graph and conformance owns any certification decision.
    """

    material_path: Path
    sha256: str
    source_sha256: str
    plugin_id: str
    plugin_version: str
    contract_digest: str
    lineage_path: Path
    lineage_sha256: str
    receipt: str


@dataclass(frozen=True)
class NeutralOnnxRequest:
    """Centrally-attested ONNX bytes supplied to a target without provider code."""

    materialized: MaterializedOnnx
    contract: VisionContract
    _token: str = field(repr=False, compare=False)

    def __init_subclass__(cls, **kwargs):
        raise TypeError("NeutralOnnxRequest is sealed")


_NEUTRAL_RECEIPTS: dict[str, tuple[str, str, str, str]] = {}
_NEUTRAL_REQUESTS: dict[str, NeutralOnnxRequest] = {}


def _prepared(path: str | Path, plugin_id: str, plugin_version: str, contract: VisionContract) -> dict:
    """Create the only permitted material-handle fields.

    Handles deliberately contain no object graph, callback, tensor, or runtime
    session.  Runtime state must be derived again from verified material.
    """
    material = Path(path)
    if material.is_symlink() or not material.is_file():
        raise ValueError("material must be a regular file")
    return {
        "material_path": material,
        "sha256": hashlib.sha256(material.read_bytes()).hexdigest(),
        "plugin_id": plugin_id,
        "plugin_version": plugin_version,
        "contract_digest": contract.digest,
        "descriptors": (),
    }


def _verify_source_handle(captured: PreparedSource, contract: VisionContract, plugin: SourceAdapter) -> None:
    if (
        captured.plugin_id != plugin.plugin_id
        or captured.plugin_version != plugin.plugin_version
        or captured.contract_digest != contract.digest
        or captured.sha256 != contract.provenance.source_sha256
        or captured.material_path.is_symlink()
        or not captured.material_path.is_file()
        or hashlib.sha256(captured.material_path.read_bytes()).hexdigest() != captured.sha256
    ):
        raise ValueError("source handle is not bound to verified checkpoint material and contract")


def _materialized(
    path: Path,
    captured: PreparedSource,
    contract: VisionContract,
    plugin: SourceAdapter,
    *,
    lineage_path: Path | None = None,
) -> MaterializedOnnx:
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".onnx":
        raise ValueError("provider export did not produce a regular .onnx file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lineage = lineage_path or path.with_suffix(path.suffix + ".lineage.json")
    body = {
        "schema": 1,
        "source_sha256": captured.sha256,
        "onnx_sha256": digest,
        "source_plugin": plugin.plugin_id,
        "source_plugin_version": plugin.plugin_version,
        "contract_digest": contract.digest,
    }
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    if lineage.exists() or lineage.is_symlink():
        if lineage.is_symlink() or not lineage.is_file() or lineage.read_bytes() != payload:
            raise ValueError("materialized ONNX lineage publication is not coherent")
    else:
        # Legacy providers publish only a regular file; RF-DETR publishes its
        # sidecar before its visible ONNX replacement below.
        with lineage.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    materialized = MaterializedOnnx(
        path,
        digest,
        captured.sha256,
        plugin.plugin_id,
        plugin.plugin_version,
        contract.digest,
        lineage,
        hashlib.sha256(payload).hexdigest(),
        "",
    )
    return plugin._issue_materialization(materialized, captured, contract)


def _verify_materialized_onnx(materialized: MaterializedOnnx, contract: VisionContract) -> None:
    if (
        materialized.contract_digest != contract.digest
        or materialized.source_sha256 != contract.provenance.source_sha256
        or materialized.material_path.is_symlink()
        or not materialized.material_path.is_file()
        or hashlib.sha256(materialized.material_path.read_bytes()).hexdigest() != materialized.sha256
        or materialized.lineage_path.is_symlink()
        or not materialized.lineage_path.is_file()
        or hashlib.sha256(materialized.lineage_path.read_bytes()).hexdigest() != materialized.lineage_sha256
    ):
        raise ValueError("materialized ONNX is not bound to verified export and lineage bytes")
    try:
        lineage = json.loads(materialized.lineage_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("materialized ONNX lineage record is invalid") from exc
    if lineage != {
        "schema": 1,
        "source_sha256": materialized.source_sha256,
        "onnx_sha256": materialized.sha256,
        "source_plugin": materialized.plugin_id,
        "source_plugin_version": materialized.plugin_version,
        "contract_digest": materialized.contract_digest,
    }:
        raise ValueError("materialized ONNX lineage record does not bind the export")


def _verify_materialized_boundary(
    materialized: MaterializedOnnx, captured: PreparedSource, contract: VisionContract
) -> None:
    if (
        captured.contract_digest != contract.digest
        or captured.sha256 != contract.provenance.source_sha256
        or captured.material_path.is_symlink()
        or not captured.material_path.is_file()
        or hashlib.sha256(captured.material_path.read_bytes()).hexdigest() != captured.sha256
        or materialized.source_sha256 != captured.sha256
        or materialized.plugin_id != captured.plugin_id
        or materialized.plugin_version != captured.plugin_version
    ):
        raise ValueError("materialized ONNX is not bound to the captured source")
    _verify_materialized_onnx(materialized, contract)


def neutral_onnx_request(
    materialized: MaterializedOnnx, captured: PreparedSource, source: SourceAdapter, contract: VisionContract
) -> NeutralOnnxRequest:
    """Verify provider provenance once, before any target is invoked."""
    _verify_materialized_boundary(materialized, captured, contract)
    if not source.verify_materialization(materialized, captured, contract):
        raise ValueError("materialized ONNX lacks an in-process source receipt")
    _NEUTRAL_RECEIPTS[materialized.receipt] = (
        materialized.sha256,
        materialized.lineage_sha256,
        captured.sha256,
        contract.digest,
    )
    request = NeutralOnnxRequest(materialized, contract, secrets.token_urlsafe(32))
    _NEUTRAL_REQUESTS[request._token] = request
    return request


def _verify_neutral_request(request: NeutralOnnxRequest) -> MaterializedOnnx:
    if type(request) is not NeutralOnnxRequest or type(request.contract) is not VisionContract:
        raise ValueError("target requires a centrally verified neutral ONNX request")
    if _NEUTRAL_REQUESTS.get(request._token) is not request:
        raise ValueError("neutral ONNX request was not centrally issued")
    materialized, contract = request.materialized, request.contract
    receipt = _NEUTRAL_RECEIPTS.get(materialized.receipt)
    if receipt != (materialized.sha256, materialized.lineage_sha256, materialized.source_sha256, contract.digest):
        raise ValueError("neutral ONNX request lacks exact central attestation")
    _verify_materialized_onnx(materialized, contract)
    return materialized


def _validate_onnx_contract(path: Path, contract: VisionContract) -> None:
    """Check all observable input/output interface facts; not a decoder proof."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX graph validation requires vision-buildkit[onnx]") from exc
    graph = onnx.load(str(path)).graph
    types = {1: "float32", 10: "float16", 7: "int64", 2: "uint8"}

    def check(values, declared):
        actual = {item.name: item for item in values}
        if set(actual) != set(declared):
            raise ValueError(f"ONNX tensor names mismatch: declared={sorted(declared)}, actual={sorted(actual)}")
        for name, spec in declared.items():
            tensor = actual[name].type.tensor_type
            dims = tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim)
            if (
                types.get(tensor.elem_type) != spec[0]
                or len(dims) != len(spec[1])
                or any(
                    expected != observed
                    for expected, observed in zip(spec[1], dims, strict=True)
                    if isinstance(expected, int)
                )
            ):
                raise ValueError(f"ONNX tensor interface mismatch for {name}")

    check(graph.input, {contract.input.input_name: (contract.input.dtype, contract.input.shape)})
    check(graph.output, {item.name: (item.dtype, item.shape) for item in contract.raw_output.tensors})


class Capabilities(Strict):
    """Structured source requirements or target supports."""

    precisions: frozenset[str] = frozenset()
    modes: frozenset[Literal["static", "dynamic"]] = frozenset()
    nms_owners: frozenset[Literal["none", "source", "target", "contract"]] = frozenset()


class SourceAdapter(ABC):
    plugin_id: str
    plugin_version: str
    capabilities: Capabilities | frozenset[str] = frozenset()

    @abstractmethod
    def capture(self, source: Any, contract: VisionContract) -> Any: ...

    def prepare_input(self, captured: PreparedSource, sample: Any, contract: VisionContract) -> Any:
        """Decode a verified case.  Only this adapter may interpret its state."""
        return sample

    @abstractmethod
    def run(self, captured: PreparedSource, sample: Any) -> Any: ...

    def prepare(self, source: str | Path, contract: VisionContract) -> PreparedSource:
        self.capture(Path(source), contract)  # validate source-specific material only
        return PreparedSource(**_prepared(source, self.plugin_id, self.plugin_version, contract))

    def _issue_materialization(
        self, materialized: MaterializedOnnx, captured: PreparedSource, contract: VisionContract
    ) -> MaterializedOnnx:
        """Issue process-local integrity state; this is neither a signature nor authenticity proof."""
        receipt = secrets.token_urlsafe(24)
        receipts = getattr(self, "_materialization_receipts", None)
        if receipts is None:
            receipts = {}
            self._materialization_receipts = receipts
        receipts[receipt] = (
            captured.sha256,
            materialized.sha256,
            contract.digest,
            materialized.lineage_sha256,
            materialized.plugin_id,
            materialized.plugin_version,
        )
        return replace(materialized, receipt=receipt)

    def verify_materialization(
        self, materialized: MaterializedOnnx, captured: PreparedSource, contract: VisionContract
    ) -> bool:
        receipt = getattr(self, "_materialization_receipts", {}).get(materialized.receipt)
        return receipt == (
            captured.sha256,
            materialized.sha256,
            contract.digest,
            materialized.lineage_sha256,
            materialized.plugin_id,
            materialized.plugin_version,
        )

    def materialize_onnx(
        self, captured: PreparedSource, contract: VisionContract, output: str | Path
    ) -> MaterializedOnnx:
        """Export verified source bytes into ONNX; providers must override this.

        A target receives only the neutral ONNX material below, never a provider
        object, model, runner, or session.
        """
        raise RuntimeError(f"{self.plugin_id} cannot materialize ONNX")


class TargetPlugin(ABC):
    plugin_id: str
    plugin_version: str
    capabilities: Capabilities | frozenset[str] = frozenset()

    @abstractmethod
    def build(self, captured: Any, candidate: Any, output: Path | None = None) -> Any: ...

    def prepare_materialized(self, request: NeutralOnnxRequest, *legacy_args: Any) -> PreparedArtifact:
        """The sole provider-neutral target preparation boundary."""
        if legacy_args:
            raise TypeError(
                "vision-buildkit 0.2 migration: prepare_materialized accepts exactly one sealed "
                "NeutralOnnxRequest; the legacy four-argument provider call is unsupported"
            )
        materialized = _verify_neutral_request(request)
        contract = request.contract
        return self._prepare_onnx(materialized.material_path, materialized, contract)

    def _prepare_onnx(self, source: Path, materialized: MaterializedOnnx, contract: VisionContract) -> PreparedArtifact:
        raise RuntimeError(f"{self.plugin_id} does not prepare neutral ONNX material")

    def prepare_input(self, artifact: PreparedArtifact, sample: Any, contract: VisionContract) -> Any:
        return sample

    @abstractmethod
    def run(self, artifact: PreparedArtifact, sample: Any) -> Any: ...

    def verify_artifact(self, artifact: PreparedArtifact, contract: VisionContract) -> bool:
        """Return true only for an artifact this target can actually execute."""
        return False


class PythonObjectSource(SourceAdapter):
    plugin_id, plugin_version = "python-object", "1"
    capabilities = frozenset({"python", "detection"})

    def capture(self, source, contract):
        raise ValueError("python objects are not material build sources")

    def run(self, captured, sample):
        raise RuntimeError("python-object has no content-bound runner")


class ManualOnnxSource(SourceAdapter):
    plugin_id, plugin_version = "manual-onnx", "1"
    capabilities = Capabilities(
        precisions=frozenset({"fp32", "fp16"}),
        modes=frozenset({"static", "dynamic"}),
        nms_owners=frozenset({"none", "source", "target"}),
    )

    def capture(self, source, contract):
        path = Path(source)
        if path.suffix.lower() != ".onnx" or not path.is_file():
            raise ValueError("manual ONNX source must be an existing .onnx file")
        return path

    def run(self, captured, sample):
        raise RuntimeError("manual ONNX source has no generic runner; supply a runtime source adapter")

    def materialize_onnx(self, captured, contract, output):
        _verify_source_handle(captured, contract, self)
        path = captured.material_path
        if path != Path(output):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_bytes(path.read_bytes())
            path = Path(output)
        return _materialized(path, captured, contract, self)


class LegacyAdapterSource(SourceAdapter):
    """Bridge an existing ModelAdapter instance into the BuildKit source API."""

    plugin_id, plugin_version = "legacy-model-adapter", "1"
    capabilities = frozenset({"detection"})

    def capture(self, source, contract):
        raise ValueError("legacy object adapters are not material build sources")

    def run(self, captured, sample):
        raise RuntimeError("legacy object adapters have no content-bound runner")


class OnnxRuntimeTargetPlugin(TargetPlugin):
    """Lazy, real ONNX Runtime target for frozen detection output contracts."""

    plugin_id = "onnxruntime"
    capabilities = Capabilities(
        precisions=frozenset({"fp32", "fp16"}),
        modes=frozenset({"static", "dynamic"}),
        nms_owners=frozenset({"none", "source", "target"}),
    )

    @property
    def plugin_version(self):
        try:
            return metadata.version("onnxruntime")
        except metadata.PackageNotFoundError:
            return "unavailable"

    @staticmethod
    def _ort():
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime target unavailable; install vision-buildkit[onnx]") from exc
        return ort

    @staticmethod
    def _assert_cpu_execution_provider(session) -> dict[str, str]:
        """Fail closed unless ORT confirms the requested CPU-only execution path."""
        providers = session.get_providers()
        if providers != ["CPUExecutionProvider"]:
            raise ValueError(f"ONNX Runtime session must use exactly ['CPUExecutionProvider']; observed {providers!r}")
        return {"execution_provider": "CPUExecutionProvider", "execution_device": "cpu"}

    def build(self, captured, candidate, output=None):
        contract = getattr(captured, "contract", None) or (
            captured.get("contract") if isinstance(captured, dict) else None
        )
        source = getattr(captured, "path", None) or (captured.get("path") if isinstance(captured, dict) else captured)
        return self.prepare(source, contract)

    def prepare(self, source: str | Path, contract: VisionContract) -> PreparedArtifact:
        """Legacy manual-ONNX entry point; only contract-hashed material is accepted."""
        path = Path(source)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if digest != contract.provenance.source_sha256:
            raise ValueError("direct ONNX preparation requires contract-hashed source material")
        return self._prepare_onnx(path, None, contract)

    def _prepare_onnx(
        self, source: Path, materialized: MaterializedOnnx | None, contract: VisionContract
    ) -> PreparedArtifact:
        if contract is None:
            raise ValueError("ONNX Runtime target requires VisionContract")
        rfdetr_raw = (
            contract.raw_output.semantics == "custom"
            and contract.raw_output.decoder_plugin == "rfdetr.sigmoid_coco_v1"
            and contract.result.box_format == "xyxy"
        )
        rtdetr_raw = contract.raw_output.semantics == "rtdetr_combined"
        if rtdetr_raw and (
            contract.result.box_format != "xyxy"
            or contract.result.coordinate_space != "original_image"
            or contract.result.nms_owner != "none"
            or contract.result.suppression != "none"
            or contract.result.ordering != "score_desc"
            or contract.result.tie_policy != "class_then_box"
            or contract.result.top_k != 300
            or contract.result.max_detections != 300
        ):
            raise ValueError("ONNX RT-DETR result contract contradicts the frozen NMS-free semantics")
        if (
            (
                contract.raw_output.semantics not in {"boxes_scores_classes", "yolo_end2end"}
                or contract.result.box_format != "xyxy"
            )
            and not rfdetr_raw
            and not rtdetr_raw
        ):
            raise ValueError(
                "ONNX Runtime target supports declared xyxy separate outputs, yolo_end2end, "
                "or the frozen RF-DETR/RT-DETR raw ABIs"
            )
        if contract.result.coordinate_space != "original_image":
            raise ValueError("ONNX Runtime target requires original_image coordinate semantics")
        path = Path(source)
        if not path.is_file() or path.suffix.lower() != ".onnx":
            raise ValueError("ONNX Runtime target requires an ONNX file")
        import hashlib

        lineage_source = materialized.source_sha256 if materialized else hashlib.sha256(path.read_bytes()).hexdigest()
        if lineage_source != contract.provenance.source_sha256:
            raise ValueError("ONNX source lineage digest does not match contract provenance")
        _validate_onnx_contract(path, contract)
        if contract.raw_output.semantics == "yolo_end2end":
            self._validate_optional_ultralytics_metadata(path, contract)
        session = self._ort().InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._assert_cpu_execution_provider(session)
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1:
            raise ValueError("ONNX target requires exactly one input")
        if inputs[0].name != contract.input.input_name or tuple(inputs[0].shape) != tuple(contract.input.shape):
            raise ValueError("ONNX input name or concrete shape does not match contract")
        declared = {x.name: x for x in contract.raw_output.tensors}
        available = {x.name for x in outputs}
        if set(declared) != available:
            raise ValueError(f"ONNX output names mismatch: declared={sorted(declared)}, available={sorted(available)}")
        if contract.raw_output.semantics == "yolo_end2end":
            only = contract.raw_output.tensors[0]
            observed = outputs[0] if len(outputs) == 1 else None
            if (
                observed is None
                or observed.name != only.name
                or observed.type not in {"tensor(float)", "tensor(float16)"}
                or len(observed.shape) not in {2, 3}
                or observed.shape[-1] != 6
            ):
                raise ValueError("ONNX yolo_end2end output does not match contract")
        if rfdetr_raw and tuple((item.name, item.dtype, item.shape) for item in contract.raw_output.tensors) != (
            ("dets", "float32", (1, 300, 4)),
            ("labels", "float32", (1, 300, 91)),
        ):
            raise ValueError("ONNX RF-DETR output contract does not match the frozen raw ABI")
        if rtdetr_raw:
            expected = (("output0", "float32", (1, 300, 6)),)
            if tuple((item.name, item.dtype, item.shape) for item in contract.raw_output.tensors) != expected:
                raise ValueError("ONNX RT-DETR output contract does not match the frozen raw ABI")
            observed = outputs[0] if len(outputs) == 1 else None
            if (
                observed is None
                or observed.name != "output0"
                or observed.type != "tensor(float)"
                or tuple(observed.shape) != (1, 300, 6)
            ):
                raise ValueError("ONNX RT-DETR runtime output does not match exact float32 [1,300,6]")
        roles = contract.raw_output.roles or {}
        for role, expected in roles.items():
            observed = next((item for item in outputs if item.name == expected), None)
            if observed is None:
                raise ValueError(f"ONNX {role} role does not name an output")
            declared_spec = declared[expected]
            ort_type = {
                "tensor(float)": "float32",
                "tensor(float16)": "float16",
                "tensor(int64)": "int64",
                "tensor(uint8)": "uint8",
            }.get(observed.type)
            if ort_type != declared_spec.dtype:
                raise ValueError(f"ONNX {role} dtype does not match contract")
        handle = _prepared(path, self.plugin_id, self.plugin_version, contract)
        artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        descriptors = [
            ("origin", "provider_export" if materialized else "direct_source"),
            ("contract_json", contract.canonical_json()),
            ("source_sha256", lineage_source),
            ("artifact_sha256", artifact_sha),
        ]
        if rtdetr_raw:
            from .decoders import RTDETR_COMBINED_ABI

            descriptors.append(("decoder_abi", RTDETR_COMBINED_ABI))
        if materialized:
            descriptors.extend(
                [
                    ("source_plugin", materialized.plugin_id),
                    ("source_plugin_version", materialized.plugin_version),
                    ("lineage_sha256", materialized.lineage_sha256),
                    ("lineage_path", str(materialized.lineage_path)),
                ]
            )
        handle["descriptors"] = tuple(descriptors)
        return PreparedArtifact(**handle)

    @staticmethod
    def _validate_optional_ultralytics_metadata(path: Path, contract: VisionContract) -> None:
        try:
            import ast

            import onnx
        except ImportError:
            return
        meta = {item.key: item.value for item in onnx.load(str(path), load_external_data=False).metadata_props}
        if str(meta.get("author", "")).strip().lower() != "ultralytics":
            return
        args: dict[str, Any] = {}
        if "args" in meta:
            try:
                args = ast.literal_eval(meta["args"])
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Ultralytics metadata args is invalid") from exc
            if not isinstance(args, dict):
                raise ValueError("Ultralytics metadata args must be a dictionary")

        def declared(key: str) -> str:
            top, nested = meta.get(key), args.get(key)
            if top is not None and nested is not None and str(top).strip().lower() != str(nested).strip().lower():
                raise ValueError(f"Ultralytics metadata {key!r} contradicts args")
            return str(top if top is not None else nested).strip().lower()

        required = {"task": "detect", "end2end": "true", "nms": "false"}
        for key, value in required.items():
            if declared(key) != value:
                raise ValueError(f"Ultralytics metadata {key!r} contradicts yolo_end2end contract")
        head = str(meta.get("head", "")).strip().split(".")[-1].lower()
        if head != "detect":
            raise ValueError("Ultralytics metadata 'head' contradicts yolo_end2end contract")
        if "imgsz" in meta:
            try:
                size = ast.literal_eval(meta["imgsz"])
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Ultralytics metadata imgsz is invalid") from exc
            if tuple(size) != tuple(contract.input.shape[-2:]):
                raise ValueError("Ultralytics metadata imgsz contradicts contract input shape")
        if "names" in meta:
            try:
                names = ast.literal_eval(meta["names"])
            except (ValueError, SyntaxError) as exc:
                raise ValueError("Ultralytics metadata names is invalid") from exc
            if (
                not isinstance(names, dict)
                or list(names.keys()) != list(range(len(contract.labels)))
                or tuple(names.values()) != contract.labels
            ):
                raise ValueError("Ultralytics metadata names contradicts contract labels")

    def verify_artifact(self, artifact: PreparedArtifact, contract: VisionContract) -> bool:
        descriptors = dict(artifact.descriptors)
        if len(descriptors) != len(artifact.descriptors):
            return False
        origin = descriptors.get("origin")
        direct_keys = {"origin", "contract_json", "source_sha256", "artifact_sha256"}
        if contract.raw_output.semantics == "rtdetr_combined":
            from .decoders import RTDETR_COMBINED_ABI

            direct_keys.add("decoder_abi")
            if descriptors.get("decoder_abi") != RTDETR_COMBINED_ABI:
                return False
        export_keys = direct_keys | {"source_plugin", "source_plugin_version", "lineage_path", "lineage_sha256"}
        if (
            (origin == "direct_source" and set(descriptors) != direct_keys)
            or (origin == "provider_export" and set(descriptors) != export_keys)
            or origin not in {"direct_source", "provider_export"}
        ):
            return False
        material_sha = (
            hashlib.sha256(artifact.material_path.read_bytes()).hexdigest() if artifact.material_path.is_file() else ""
        )
        if (
            descriptors.get("contract_json") != contract.canonical_json()
            or descriptors.get("artifact_sha256") != material_sha
        ):
            return False
        if origin == "direct_source":
            return (
                artifact.sha256 == material_sha == contract.provenance.source_sha256
                and descriptors.get("source_sha256") == contract.provenance.source_sha256
                and artifact.plugin_id == self.plugin_id
                and artifact.plugin_version == self.plugin_version
                and artifact.contract_digest == contract.digest
                and artifact.material_path.is_file()
                and not artifact.material_path.is_symlink()
            )
        lineage_path = descriptors.get("lineage_path")
        lineage_sha256 = descriptors.get("lineage_sha256")
        lineage_ok = (lineage_path is None and lineage_sha256 is None) or (
            lineage_path is not None
            and lineage_sha256 is not None
            and Path(lineage_path).is_file()
            and not Path(lineage_path).is_symlink()
            and hashlib.sha256(Path(lineage_path).read_bytes()).hexdigest() == lineage_sha256
        )
        if not lineage_ok:
            return False
        try:
            lineage = json.loads(Path(lineage_path).read_text())
        except (OSError, json.JSONDecodeError, TypeError):
            return False
        return (
            artifact.plugin_id == self.plugin_id
            and artifact.plugin_version == self.plugin_version
            and descriptors.get("source_sha256") == contract.provenance.source_sha256
            and lineage
            == {
                "schema": 1,
                "source_sha256": descriptors["source_sha256"],
                "onnx_sha256": descriptors["artifact_sha256"],
                "source_plugin": descriptors["source_plugin"],
                "source_plugin_version": descriptors["source_plugin_version"],
                "contract_digest": contract.digest,
            }
            and artifact.contract_digest == contract.digest
            and artifact.material_path.is_file()
            and not artifact.material_path.is_symlink()
            and material_sha == artifact.sha256
        )

    def _verified_session(self, artifact: PreparedArtifact, contract: VisionContract):
        if not self.verify_artifact(artifact, contract):
            raise ValueError("ONNX artifact handle is not bound to verified material and contract")
        session = self._ort().InferenceSession(str(artifact.material_path), providers=["CPUExecutionProvider"])
        self._assert_cpu_execution_provider(session)
        if len(session.get_inputs()) != 1:
            raise ValueError("ONNX target requires exactly one input")
        return session

    def run(self, artifact, sample):
        import numpy as np

        if not isinstance(sample, PreparedInput):
            raise ValueError("ONNX Runtime target requires a contract PreparedInput")
        # The contract is supplied by `prepare_input`/certification verification;
        # run reconstructs a session only from rehashed ONNX material.  A caller
        # cannot substitute a prepared handle's session or runner.
        contract_json = dict(artifact.descriptors).get("contract_json")
        if not contract_json:
            raise ValueError("ONNX artifact lacks immutable contract descriptor")
        contract = VisionContract.model_validate_json(contract_json)
        if contract.raw_output.semantics == "rtdetr_combined" and sample.trace is None:
            raise ValueError("rtdetr_combined requires geometry trace")
        session = self._verified_session(artifact, contract)
        execution_metadata = self._assert_cpu_execution_provider(session)
        values = session.run(None, {session.get_inputs()[0].name: sample.tensor})
        tensors = {x.name: value for x, value in zip(session.get_outputs(), values, strict=True)}
        roles = contract.raw_output.roles or {}
        if contract.raw_output.semantics == "yolo_end2end":
            if sample.trace is None:
                raise ValueError("yolo_end2end requires geometry trace")
            from .decoders import yolo_end2end_contract

            result = yolo_end2end_contract(
                tensors[contract.raw_output.tensors[0].name], sample.trace, contract, image_size=sample.image_size
            )
            result.metadata.update(execution_metadata | {"input_sha256": sample.input_sha256})
            return result
        if contract.raw_output.semantics == "custom" and contract.raw_output.decoder_plugin == "rfdetr.sigmoid_coco_v1":
            from .decoders import rfdetr_contract_output

            result = rfdetr_contract_output(
                {"dets": tensors["dets"], "labels": tensors["labels"]}, sample.image_size, contract
            )
            result.metadata.update(execution_metadata | {"input_sha256": sample.input_sha256})
            return result
        if contract.raw_output.semantics == "rtdetr_combined":
            from .decoders import rtdetr_combined_v1_contract

            result = rtdetr_combined_v1_contract(
                tensors["output0"], sample.trace, contract, image_size=sample.image_size
            )
            result.metadata.update(execution_metadata | {"input_sha256": sample.input_sha256})
            return result
        boxes = np.asarray(tensors[roles["boxes"]]).reshape(-1, 4)
        scores = np.asarray(tensors[roles["scores"]]).reshape(-1)
        classes = np.asarray(tensors[roles["classes"]]).reshape(-1)
        if len(boxes) != len(scores) or len(boxes) != len(classes):
            raise ValueError("ONNX boxes/scores/classes output cardinality mismatch")
        return DetectionResult(
            [Box(*map(float, box)) for box in boxes],
            [float(x) for x in scores],
            [int(x) for x in classes],
            image_size=sample.image_size,
            metadata=execution_metadata | {"input_sha256": sample.input_sha256},
        )

    def prepare_input(self, artifact: PreparedArtifact, sample: Any, contract: VisionContract) -> PreparedInput:
        """Decode image material and apply the exact contract preprocessing.

        This deliberately accepts only a regular image file or its verified bytes;
        callers cannot smuggle a ready-made tensor past the contract.
        """
        if not self.verify_artifact(artifact, contract):
            raise ValueError("ONNX artifact handle is not bound to verified material and contract")
        return preprocess(contract, sample)


class OpenVinoTargetPlugin(TargetPlugin):
    """Neutral ONNX -> OpenVINO CPU target with sealed IR publication."""

    plugin_id = "openvino"
    capabilities = Capabilities(
        precisions=frozenset({"fp32", "fp16", "int8"}),
        modes=frozenset({"static"}),
        nms_owners=frozenset({"none", "source"}),
    )
    _configuration_fields = frozenset({"output", "precision", "calibration"})

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_configuration_sealed", False) and name in self._configuration_fields:
            raise AttributeError(f"OpenVINO target configuration is immutable: {name}")
        super().__setattr__(name, value)

    def __init__(self, output: str | Path, precision: str = "fp32", calibration: CalibrationManifest | None = None):
        self.output = Path(output)
        self.precision = precision
        # Own a deep, revalidated, immutable snapshot.  Receipt binding and
        # quantization must never observe caller mutations after construction.
        self.calibration = (
            CalibrationManifest.model_validate(calibration.model_dump(mode="json")) if calibration is not None else None
        )
        if self.output.suffix.lower() != ".xml":
            raise ValueError("OpenVINO target requires an explicit .xml output path")
        if precision not in self.capabilities.precisions:
            raise ValueError("OpenVINO precision must be fp32, fp16, or int8")
        if precision == "int8" and calibration is None:
            raise ValueError("OpenVINO INT8 requires CalibrationManifest")
        self._artifact_receipts: dict[str, tuple[str, ...]] = {}
        self._configuration_sealed = True

    @property
    def plugin_version(self) -> str:
        try:
            return metadata.version("openvino")
        except metadata.PackageNotFoundError:
            return "unavailable"

    @staticmethod
    def _ov():
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("openvino target unavailable; install vision-buildkit[openvino]") from exc
        return ov

    def build(self, captured: Any, candidate: Any, output: Path | None = None) -> PreparedArtifact:
        if not isinstance(candidate, NeutralOnnxRequest):
            raise TypeError("OpenVINO target requires exactly one sealed NeutralOnnxRequest")
        return self.prepare_materialized(candidate)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _identity_at(directory_fd: int, name: str) -> tuple[int, int]:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"OpenVINO staged member is not a regular file: {name}")
        return observed.st_dev, observed.st_ino

    @staticmethod
    def _sha_at(directory_fd: int, name: str) -> str:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            digest = hashlib.sha256()
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_at(directory_fd: int, name: str) -> None:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _directory_identity(path: Path, *, message: str) -> tuple[int, int]:
        observed = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(observed.st_mode) or path.is_symlink():
            raise RuntimeError(message)
        return observed.st_dev, observed.st_ino

    @staticmethod
    def _directory_identity_at(directory_fd: int, name: str) -> tuple[int, int]:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("OpenVINO private stage identity changed")
        return observed.st_dev, observed.st_ino

    @staticmethod
    def _entry_exists_at(directory_fd: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    @classmethod
    def _assert_parent_path(cls, path: Path, expected: tuple[int, int]) -> None:
        try:
            observed = cls._directory_identity(path, message="OpenVINO output parent identity changed")
        except (FileNotFoundError, NotADirectoryError, OSError) as exc:
            raise RuntimeError("OpenVINO output parent identity changed") from exc
        if observed != expected:
            raise RuntimeError("OpenVINO output parent identity changed")

    @classmethod
    def _fsync_public_at(cls, parent_fd: int, name: str) -> None:
        cls._fsync_at(parent_fd, name)

    @staticmethod
    def _fsync_parent(parent_fd: int) -> None:
        os.fsync(parent_fd)

    @staticmethod
    def _fsync_stage(stage_fd: int) -> None:
        os.fsync(stage_fd)

    @staticmethod
    def _validate_detection_semantics(contract: VisionContract) -> tuple[bool, bool]:
        rfdetr_raw = (
            contract.raw_output.semantics == "custom"
            and contract.raw_output.decoder_plugin == "rfdetr.sigmoid_coco_v1"
            and contract.result.box_format == "xyxy"
        )
        rtdetr_raw = contract.raw_output.semantics == "rtdetr_combined"
        if rtdetr_raw and (
            contract.result.box_format != "xyxy"
            or contract.result.coordinate_space != "original_image"
            or contract.result.nms_owner != "none"
            or contract.result.suppression != "none"
            or contract.result.ordering != "score_desc"
            or contract.result.tie_policy != "class_then_box"
            or contract.result.top_k != 300
            or contract.result.max_detections != 300
        ):
            raise ValueError("OpenVINO RT-DETR result contract contradicts the frozen NMS-free semantics")
        if (
            (
                contract.raw_output.semantics not in {"boxes_scores_classes", "yolo_end2end"}
                or contract.result.box_format != "xyxy"
            )
            and not rfdetr_raw
            and not rtdetr_raw
        ):
            raise ValueError("OpenVINO target supports only the frozen detection decoder contracts")
        if contract.result.coordinate_space != "original_image":
            raise ValueError("OpenVINO target requires original_image coordinate semantics")
        if contract.result.nms_owner not in OpenVinoTargetPlugin.capabilities.nms_owners:
            raise ValueError("OpenVINO target does not execute target/contract NMS")
        if rfdetr_raw and tuple((item.name, item.dtype, item.shape) for item in contract.raw_output.tensors) != (
            ("dets", "float32", (1, 300, 4)),
            ("labels", "float32", (1, 300, 91)),
        ):
            raise ValueError("OpenVINO RF-DETR output contract does not match the frozen raw ABI")
        if rtdetr_raw and tuple((item.name, item.dtype, item.shape) for item in contract.raw_output.tensors) != (
            ("output0", "float32", (1, 300, 6)),
        ):
            raise ValueError("OpenVINO RT-DETR output contract does not match the frozen raw ABI")
        return rfdetr_raw, rtdetr_raw

    @staticmethod
    def _preprocessing_digest(contract: VisionContract) -> str:
        preprocessing = contract.model_dump(mode="json")["preprocessing"]
        return hashlib.sha256(json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def _calibration_binding_for(
        cls,
        contract: VisionContract,
        precision: str,
        calibration: CalibrationManifest | None,
    ) -> dict[str, str] | None:
        if precision != "int8":
            if calibration is not None:
                raise ValueError("OpenVINO calibration is accepted only for INT8")
            return None
        if calibration is None:
            raise ValueError("OpenVINO INT8 requires CalibrationManifest")
        if calibration.digest != calibration.canonical_digest():
            raise ValueError("OpenVINO calibration manifest digest changed after validation")
        preprocessing_digest = cls._preprocessing_digest(contract)
        if (
            calibration.input_name != contract.input.input_name
            or tuple(calibration.shape) != tuple(contract.input.shape)
            or calibration.dtype != contract.input.dtype
            or calibration.preprocessing_digest != preprocessing_digest
        ):
            raise ValueError("OpenVINO INT8 calibration manifest contradicts contract input/preprocessing")
        return {"digest": calibration.digest, "preprocessing_digest": preprocessing_digest}

    def _calibration_binding(self, contract: VisionContract) -> dict[str, str] | None:
        return self._calibration_binding_for(contract, self.precision, self.calibration)

    def _prepare_onnx(self, source: Path, materialized: MaterializedOnnx, contract: VisionContract) -> PreparedArtifact:
        if self.output.suffix.lower() != ".xml":
            raise ValueError("OpenVINO target requires an explicit .xml output path")
        if any(
            path.exists() or path.is_symlink()
            for path in (self.output, self.output.with_suffix(".bin"), self.output.with_suffix(".receipt.json"))
        ):
            raise FileExistsError("OpenVINO output XML/BIN/receipt paths must be absent")
        _validate_onnx_contract(source, contract)
        self._validate_detection_semantics(contract)
        if contract.raw_output.semantics == "yolo_end2end":
            OnnxRuntimeTargetPlugin._validate_optional_ultralytics_metadata(source, contract)
        _verify_materialized_onnx(materialized, contract)
        # Freeze one configuration view for the complete transaction.  Do not
        # dereference mutable target state again during conversion.
        precision = self.precision
        calibration = self.calibration
        calibration_binding = self._calibration_binding_for(contract, precision, calibration)
        parent = self.output.parent
        try:
            parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise ValueError("OpenVINO output parent must be an existing regular directory") from exc
        parent_stat = os.fstat(parent_fd)
        parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        stage_name = f".{self.output.stem}.openvino-{secrets.token_hex(8)}"
        stage = parent / stage_name
        owned_stage_members: dict[str, tuple[int, int]] = {}
        created: list[tuple[str, tuple[int, int]]] = []
        snapshot_cleanup_fd: int | None = None
        stage_fd: int | None = None
        publication: tuple[dict[str, Any], str, str] | None = None
        artifact_token: str | None = None
        source_snapshot_name = ".verified-source.onnx"

        def stage_is_owned() -> bool:
            try:
                return self._directory_identity_at(parent_fd, stage_name) == stage_identity
            except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError):
                return False

        def rollback_publication() -> None:
            for name, identity in reversed(created):
                try:
                    if self._identity_at(parent_fd, name) == identity:
                        os.unlink(name, dir_fd=parent_fd)
                except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError):
                    pass
            try:
                self._fsync_parent(parent_fd)
            except OSError:
                pass

        try:
            self._assert_parent_path(parent, parent_identity)
            os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
            stage_identity = self._directory_identity_at(parent_fd, stage_name)
            stage_fd = os.open(stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            source_bytes = materialized.material_path.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != materialized.sha256:
                raise ValueError("materialized ONNX changed before OpenVINO snapshot")
            snapshot_fd = os.open(
                source_snapshot_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=stage_fd,
            )
            try:
                snapshot_stat = os.fstat(snapshot_fd)
                if not stat.S_ISREG(snapshot_stat.st_mode):
                    raise RuntimeError("OpenVINO source snapshot is not a regular file")
                owned_stage_members[source_snapshot_name] = (snapshot_stat.st_dev, snapshot_stat.st_ino)
                snapshot_cleanup_fd = os.dup(snapshot_fd)
                with os.fdopen(os.dup(snapshot_fd), "wb") as handle:
                    handle.write(source_bytes)
                    handle.flush()
                os.fsync(snapshot_fd)
            finally:
                os.close(snapshot_fd)
            source_snapshot = stage / source_snapshot_name
            staged_xml = stage / self.output.name
            from .export.openvino import build_openvino

            if self._identity_at(stage_fd, source_snapshot_name) != owned_stage_members[source_snapshot_name]:
                raise RuntimeError("OpenVINO source snapshot identity changed before conversion")
            xml, metadata_value, effective = build_openvino(source_snapshot, staged_xml, precision, calibration)
            staged_bin = xml.with_suffix(".bin")
            for name in (self.output.name, self.output.with_suffix(".bin").name):
                owned_stage_members[name] = self._identity_at(stage_fd, name)
            compatibility_name = "openvino-compat.onnx"
            try:
                owned_stage_members[compatibility_name] = self._identity_at(stage_fd, compatibility_name)
            except FileNotFoundError:
                pass
            if set(os.listdir(stage_fd)) != set(owned_stage_members):
                raise RuntimeError("OpenVINO conversion created unexpected private stage members")
            if not stage_is_owned():
                raise RuntimeError("OpenVINO private stage was replaced during conversion")
            xml_name, bin_name = xml.name, staged_bin.name
            if (
                xml_name != self.output.name
                or bin_name != self.output.with_suffix(".bin").name
                or self._sha_at(stage_fd, source_snapshot_name) != materialized.sha256
            ):
                raise RuntimeError("OpenVINO conversion did not produce regular XML and BIN")
            if (
                effective != precision
                or not isinstance(metadata_value, dict)
                or {item["name"] for item in metadata_value.get("inputs", ())} != {contract.input.input_name}
                or set(metadata_value.get("outputs", ())) != {item.name for item in contract.raw_output.tensors}
            ):
                raise RuntimeError("OpenVINO conversion metadata contradicts contract or requested precision")
            receipt = {
                "schema": 1,
                "target": self.plugin_id,
                "openvino_version": self.plugin_version,
                "device": "CPU",
                "precision": effective,
                "contract_json": contract.canonical_json(),
                "contract_sha256": contract.digest,
                "onnx_sha256": materialized.sha256,
                "source_sha256": materialized.source_sha256,
                "lineage_sha256": materialized.lineage_sha256,
                "lineage_path": str(materialized.lineage_path),
                "xml_sha256": self._sha_at(stage_fd, xml_name),
                "bin_sha256": self._sha_at(stage_fd, bin_name),
                "metadata": metadata_value,
                "calibration": calibration_binding,
            }
            receipt_name = self.output.stem + ".receipt.json"
            receipt_fd = os.open(
                receipt_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=stage_fd,
            )
            try:
                payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
                with os.fdopen(os.dup(receipt_fd), "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                os.fsync(receipt_fd)
            finally:
                os.close(receipt_fd)
            owned_stage_members[receipt_name] = self._identity_at(stage_fd, receipt_name)
            if set(os.listdir(stage_fd)) != set(owned_stage_members):
                raise RuntimeError("OpenVINO private stage gained an unowned member")
            for staged_name in (xml_name, bin_name, receipt_name):
                self._fsync_at(stage_fd, staged_name)
                if self._identity_at(stage_fd, staged_name) != owned_stage_members[staged_name]:
                    raise RuntimeError("OpenVINO staged member identity changed during fsync")
                if not stage_is_owned():
                    raise RuntimeError("OpenVINO private stage was replaced during fsync")
            self._fsync_stage(stage_fd)
            if not stage_is_owned():
                raise RuntimeError("OpenVINO private stage was replaced during directory fsync")
            final_bin_name = self.output.with_suffix(".bin").name
            final_receipt_name = self.output.with_suffix(".receipt.json").name
            public_names = (self.output.name, final_bin_name, final_receipt_name)
            if any(self._entry_exists_at(parent_fd, name) for name in public_names):
                raise FileExistsError("OpenVINO publication collision")
            try:
                # Receipt is the final commit marker.  Link never replaces an existing path.
                for staged_name, final_name in (
                    (xml_name, self.output.name),
                    (bin_name, final_bin_name),
                    (receipt_name, final_receipt_name),
                ):
                    self._assert_parent_path(parent, parent_identity)
                    os.link(
                        staged_name,
                        final_name,
                        src_dir_fd=stage_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    staged_identity = self._identity_at(stage_fd, staged_name)
                    identity = self._identity_at(parent_fd, final_name)
                    if identity != staged_identity:
                        raise RuntimeError("OpenVINO publication identity changed during no-clobber link")
                    created.append((final_name, identity))
                    self._assert_parent_path(parent, parent_identity)
                    self._fsync_public_at(parent_fd, final_name)
                    self._fsync_parent(parent_fd)
                if (
                    self._sha_at(parent_fd, self.output.name) != receipt["xml_sha256"]
                    or self._sha_at(parent_fd, final_bin_name) != receipt["bin_sha256"]
                    or self._sha_at(parent_fd, final_receipt_name) != self._sha_at(stage_fd, receipt_name)
                    or self._sha_at(stage_fd, source_snapshot_name) != materialized.sha256
                ):
                    raise RuntimeError("OpenVINO publication changed before commit")
            except Exception:
                rollback_publication()
                raise
            publication = (receipt, effective, self._sha_at(parent_fd, final_receipt_name))
        finally:
            had_error = sys.exc_info()[0] is not None
            cleanup_error: Exception | None = None
            if snapshot_cleanup_fd is not None:
                try:
                    os.ftruncate(snapshot_cleanup_fd, 0)
                    os.fsync(snapshot_cleanup_fd)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
                os.close(snapshot_cleanup_fd)
            if stage_fd is not None:
                for name, identity in reversed(tuple(owned_stage_members.items())):
                    try:
                        if self._identity_at(stage_fd, name) == identity:
                            os.unlink(name, dir_fd=stage_fd)
                    except (OSError, RuntimeError) as exc:
                        cleanup_error = cleanup_error or exc
                try:
                    self._fsync_stage(stage_fd)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
                finally:
                    os.close(stage_fd)
                try:
                    if not stage_is_owned():
                        raise RuntimeError("OpenVINO private stage identity changed during cleanup")
                    os.rmdir(stage_name, dir_fd=parent_fd)
                    self._fsync_parent(parent_fd)
                except (OSError, RuntimeError) as exc:
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None and not had_error:
                rollback_publication()
                os.close(parent_fd)
                raise cleanup_error
            if had_error:
                os.close(parent_fd)
        try:
            self._assert_parent_path(parent, parent_identity)
            if publication is None:
                raise RuntimeError("OpenVINO publication did not complete")
            receipt, effective, receipt_sha = publication
            artifact_token = secrets.token_urlsafe(32)
            final_receipt = self.output.with_suffix(".receipt.json")
            descriptors = (
                ("contract_json", contract.canonical_json()),
                ("receipt_path", str(final_receipt)),
                ("onnx_sha256", materialized.sha256),
                ("lineage_sha256", materialized.lineage_sha256),
                ("effective_precision", effective),
                ("device", "CPU"),
                ("receipt_sha256", receipt_sha),
                ("artifact_token", artifact_token),
            )
            artifact = PreparedArtifact(
                self.output, receipt["xml_sha256"], self.plugin_id, self.plugin_version, contract.digest, descriptors
            )
            self._artifact_receipts[artifact_token] = (
                artifact.sha256,
                receipt["bin_sha256"],
                receipt_sha,
                contract.digest,
                materialized.sha256,
                materialized.lineage_sha256,
                effective,
                self.plugin_version,
                str(materialized.lineage_path),
            )
            self._assert_parent_path(parent, parent_identity)
            return artifact
        except Exception:
            if artifact_token is not None:
                self._artifact_receipts.pop(artifact_token, None)
            rollback_publication()
            raise
        finally:
            os.close(parent_fd)

    def verify_artifact(self, artifact: PreparedArtifact, contract: VisionContract) -> bool:
        try:
            values = dict(artifact.descriptors)
            expected_keys = {
                "contract_json",
                "receipt_path",
                "receipt_sha256",
                "onnx_sha256",
                "lineage_sha256",
                "effective_precision",
                "device",
                "artifact_token",
            }
            if len(values) != len(artifact.descriptors) or set(values) != expected_keys:
                return False
            receipt_path = Path(values["receipt_path"])
            if (
                artifact.plugin_id != self.plugin_id
                or artifact.contract_digest != contract.digest
                or artifact.plugin_version != self.plugin_version
                or artifact.material_path.is_symlink()
                or receipt_path.is_symlink()
                or not receipt_path.is_file()
                or artifact.material_path != self.output
                or receipt_path != self.output.with_suffix(".receipt.json")
            ):
                return False
            receipt = json.loads(receipt_path.read_text())
            xml, binary = artifact.material_path, artifact.material_path.with_suffix(".bin")
            lineage_path = Path(receipt["lineage_path"])
            lineage = json.loads(lineage_path.read_text())
            receipt_keys = {
                "schema",
                "target",
                "openvino_version",
                "device",
                "precision",
                "contract_json",
                "contract_sha256",
                "onnx_sha256",
                "source_sha256",
                "lineage_sha256",
                "lineage_path",
                "xml_sha256",
                "bin_sha256",
                "metadata",
                "calibration",
            }
            runtime_receipt = self._artifact_receipts.get(values["artifact_token"])
            return (
                set(receipt) == receipt_keys
                and receipt["schema"] == 1
                and xml.is_file()
                and binary.is_file()
                and not binary.is_symlink()
                and receipt["target"] == self.plugin_id
                and receipt["openvino_version"] == self.plugin_version
                and receipt["device"] == values["device"] == "CPU"
                and receipt["precision"] == values["effective_precision"]
                and receipt["precision"] == self.precision
                and receipt["contract_json"] == values["contract_json"] == contract.canonical_json()
                and receipt["contract_sha256"] == contract.digest
                and hashlib.sha256(receipt_path.read_bytes()).hexdigest() == values["receipt_sha256"]
                and receipt["xml_sha256"] == hashlib.sha256(xml.read_bytes()).hexdigest() == artifact.sha256
                and receipt["bin_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
                and receipt["onnx_sha256"] == values["onnx_sha256"]
                and receipt["lineage_sha256"] == values["lineage_sha256"]
                and receipt["source_sha256"] == contract.provenance.source_sha256
                and receipt["calibration"] == self._calibration_binding(contract)
                and isinstance(receipt["metadata"], dict)
                and {item["name"] for item in receipt["metadata"].get("inputs", ())} == {contract.input.input_name}
                and set(receipt["metadata"].get("outputs", ())) == {item.name for item in contract.raw_output.tensors}
                and lineage_path.is_file()
                and not lineage_path.is_symlink()
                and hashlib.sha256(lineage_path.read_bytes()).hexdigest() == receipt["lineage_sha256"]
                and lineage
                == {
                    "schema": 1,
                    "source_sha256": receipt["source_sha256"],
                    "onnx_sha256": receipt["onnx_sha256"],
                    "source_plugin": lineage["source_plugin"],
                    "source_plugin_version": lineage["source_plugin_version"],
                    "contract_digest": contract.digest,
                }
                and runtime_receipt
                == (
                    artifact.sha256,
                    receipt["bin_sha256"],
                    values["receipt_sha256"],
                    contract.digest,
                    receipt["onnx_sha256"],
                    receipt["lineage_sha256"],
                    receipt["precision"],
                    self.plugin_version,
                    str(lineage_path),
                )
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def prepare_input(self, artifact: PreparedArtifact, sample: Any, contract: VisionContract) -> PreparedInput:
        if not self.verify_artifact(artifact, contract):
            raise ValueError("OpenVINO artifact is not sealed")
        return preprocess(contract, sample)

    def run(self, artifact: PreparedArtifact, sample: Any) -> DetectionResult:
        if not isinstance(sample, PreparedInput):
            raise ValueError("OpenVINO target requires a contract PreparedInput")
        descriptor_values = dict(artifact.descriptors)
        contract = VisionContract.model_validate_json(descriptor_values["contract_json"])
        if not self.verify_artifact(artifact, contract):
            raise ValueError("OpenVINO artifact verification failed")
        receipt_bytes = Path(descriptor_values["receipt_path"]).read_bytes()
        if hashlib.sha256(receipt_bytes).hexdigest() != descriptor_values["receipt_sha256"]:
            raise ValueError("OpenVINO receipt changed before in-memory compilation")
        receipt = json.loads(receipt_bytes)
        xml_bytes = artifact.material_path.read_bytes()
        binary_bytes = artifact.material_path.with_suffix(".bin").read_bytes()
        runtime_receipt = self._artifact_receipts.get(descriptor_values["artifact_token"])
        if (
            hashlib.sha256(xml_bytes).hexdigest() != receipt["xml_sha256"]
            or hashlib.sha256(binary_bytes).hexdigest() != receipt["bin_sha256"]
            or receipt["xml_sha256"] != artifact.sha256
            or receipt["target"] != self.plugin_id
            or receipt["openvino_version"] != self.plugin_version
            or receipt["device"] != descriptor_values["device"]
            or descriptor_values["device"] != "CPU"
            or receipt["precision"] != descriptor_values["effective_precision"]
            or descriptor_values["effective_precision"] != self.precision
            or receipt["contract_json"] != contract.canonical_json()
            or receipt["contract_sha256"] != contract.digest
            or receipt["onnx_sha256"] != descriptor_values["onnx_sha256"]
            or receipt["lineage_sha256"] != descriptor_values["lineage_sha256"]
            or receipt["source_sha256"] != contract.provenance.source_sha256
            or receipt["calibration"] != self._calibration_binding(contract)
            or runtime_receipt
            != (
                artifact.sha256,
                receipt["bin_sha256"],
                descriptor_values["receipt_sha256"],
                contract.digest,
                receipt["onnx_sha256"],
                receipt["lineage_sha256"],
                receipt["precision"],
                self.plugin_version,
                receipt["lineage_path"],
            )
        ):
            raise ValueError("OpenVINO artifact changed before in-memory compilation")
        core = self._ov().Core()
        model = core.compile_model(core.read_model(xml_bytes, binary_bytes), "CPU")
        inputs, outputs = model.inputs, model.outputs
        if (
            len(inputs) != 1
            or inputs[0].get_any_name() != contract.input.input_name
            or inputs[0].get_element_type().to_dtype().name != contract.input.dtype
            or tuple(inputs[0].get_shape()) != tuple(contract.input.shape)
        ):
            raise ValueError("OpenVINO input ABI mismatch")
        import numpy as np

        if np.asarray(sample.tensor).dtype.name != contract.input.dtype or tuple(sample.tensor.shape) != tuple(
            contract.input.shape
        ):
            raise ValueError("OpenVINO PreparedInput dtype or shape contradicts contract")
        declared = {item.name: item for item in contract.raw_output.tensors}
        if {port.get_any_name() for port in outputs} != set(declared):
            raise ValueError("OpenVINO output names contradict contract")
        values = model({inputs[0].get_any_name(): sample.tensor})
        tensors = {port.get_any_name(): values[port] for port in outputs}
        for name, spec in declared.items():
            value = np.asarray(tensors[name])
            if (
                value.dtype.name != spec.dtype
                or len(value.shape) != len(spec.shape)
                or any(isinstance(want, int) and want != got for want, got in zip(spec.shape, value.shape, strict=True))
            ):
                raise ValueError(f"OpenVINO output ABI mismatch for {name}")
        if list(model.get_property("EXECUTION_DEVICES")) != ["CPU"]:
            raise ValueError("OpenVINO execution device is not exactly CPU")
        meta = {"execution_device": "CPU", "input_sha256": sample.input_sha256}
        if contract.raw_output.semantics == "yolo_end2end":
            from .decoders import yolo_end2end_contract

            result = yolo_end2end_contract(
                tensors[contract.raw_output.tensors[0].name], sample.trace, contract, image_size=sample.image_size
            )
        elif contract.raw_output.semantics == "custom":
            from .decoders import rfdetr_contract_output

            result = rfdetr_contract_output(
                {"dets": tensors["dets"], "labels": tensors["labels"]}, sample.image_size, contract
            )
        elif contract.raw_output.semantics == "rtdetr_combined":
            from .decoders import rtdetr_combined_v1_contract

            result = rtdetr_combined_v1_contract(
                tensors["output0"], sample.trace, contract, image_size=sample.image_size
            )
        else:
            roles = contract.raw_output.roles or {}
            boxes = np.asarray(tensors[roles["boxes"]]).reshape(-1, 4)
            scores = np.asarray(tensors[roles["scores"]]).reshape(-1)
            classes = np.asarray(tensors[roles["classes"]]).reshape(-1)
            result = DetectionResult(
                [Box(*map(float, x)) for x in boxes],
                [float(x) for x in scores],
                [int(x) for x in classes],
                image_size=sample.image_size,
            )
        result.metadata.update(meta)
        return result


class TensorRTTargetPlugin(TargetPlugin):
    """Pinned TensorRT engine target with no ORT or CPU execution fallback.

    A plugin instance represents one explicit build request.  This avoids a
    mutable global target configuration becoming detached from the engine it
    produced, which matters for a certificate rather more than convenience.
    """

    plugin_id = "tensorrt"
    plugin_version = "1"
    capabilities = Capabilities(
        precisions=frozenset({"fp32", "fp16"}),
        modes=frozenset({"static"}),
        nms_owners=frozenset({"none", "source"}),
    )

    def __init__(
        self,
        *,
        engine_output: str | Path | None = None,
        precision: Literal["fp32", "fp16"] = "fp32",
        trtexec_path: str | None = None,
        target_platform: str | None = None,
        timeout_seconds: int = 300,
    ):
        if precision not in self.capabilities.precisions:
            raise ValueError("TensorRT BuildKit target currently supports fp32 or fp16 only")
        if timeout_seconds <= 0:
            raise ValueError("TensorRT timeout_seconds must be positive")
        if target_platform is not None:
            raise ValueError("TensorRT target platform is observed from the build host, not user asserted")
        self.engine_output = Path(engine_output) if engine_output is not None else None
        self.precision = precision
        self.trtexec_path = trtexec_path
        self.timeout_seconds = timeout_seconds
        self._artifact_receipts: dict[str, tuple[str, str, str]] = {}

    @staticmethod
    def _sha(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise ValueError("TensorRT artifact material must be a regular non-symlink file")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _redact_command(command: list[str]) -> list[str]:
        """Keep flags/roles, never persist local paths or executable locations."""
        redacted = ["<trtexec>"]
        for item in command[1:]:
            if item.startswith("--onnx="):
                redacted.append("--onnx=<verified-onnx>")
            elif item.startswith("--saveEngine="):
                redacted.append("--saveEngine=<artifact-engine>")
            elif item.startswith("--calib="):
                redacted.append("--calib=<redacted>")
            else:
                redacted.append(item)
        return redacted

    @staticmethod
    def _expected_command(precision: str) -> tuple[str, ...]:
        """The sole portable compiler policy accepted for certified engines."""
        command = ["<trtexec>", "--onnx=<verified-onnx>", "--saveEngine=<artifact-engine>"]
        if precision == "fp16":
            command.append("--fp16")
        elif precision == "fp32":
            command.append("--noTF32")
        else:
            raise ValueError("TensorRT certified precision is unsupported")
        command.append("--skipInference")
        return tuple(command)

    @staticmethod
    def _redact_log(logs: str) -> str:
        # trtexec diagnostics can repeat absolute path arguments.  Preserve a
        # bounded semantic diagnostic, never a build host path or command line.
        cleaned = re.sub(r"(?:[A-Za-z]:)?/[^\s'\"]+", "<path>", logs)
        return cleaned[-8000:]

    @staticmethod
    def _build_host_metadata(executable: str, trtexec_version: str) -> dict[str, str]:
        result = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "builder_target": f"{platform.system()}-{platform.machine()}",
            "trtexec_sha256": TensorRTTargetPlugin._sha(Path(executable)),
            "trtexec_version": trtexec_version,
            "cuda_device": "unavailable",
            "cuda_compute_capability": "unavailable",
            "cuda_runtime": "unavailable",
            "tensorrt_python_version": "unavailable",
            "driver_or_jetpack": "unavailable",
        }
        try:
            import tensorrt as trt

            result["tensorrt_python_version"] = str(trt.__version__)
        except ImportError:
            pass
        try:
            version = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                capture_output=True,
                timeout=5,
            ).stdout.strip()
            if version:
                result["driver_or_jetpack"] = version
            elif Path("/etc/nv_tegra_release").is_file():
                result["driver_or_jetpack"] = Path("/etc/nv_tegra_release").read_text().strip()
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                result.update(
                    {
                        "cuda_device": props.name,
                        "cuda_compute_capability": f"{props.major}.{props.minor}",
                        "cuda_runtime": str(torch.version.cuda),
                    }
                )
        except (ImportError, RuntimeError):
            pass
        return result

    @staticmethod
    def _contract_from_artifact(artifact: PreparedArtifact) -> VisionContract:
        value = dict(artifact.descriptors).get("contract_json")
        if not value:
            raise ValueError("TensorRT artifact lacks immutable contract descriptor")
        return VisionContract.model_validate_json(value)

    def build(self, captured, candidate, output=None):
        raise RuntimeError("use BuildPlanner.materialize with a verified source, ONNX receipt, and VisionContract")

    def prepare_materialized(self, request: NeutralOnnxRequest, *legacy_args: Any):
        if legacy_args:
            raise TypeError(
                "vision-buildkit 0.2 migration: prepare_materialized accepts exactly one sealed "
                "NeutralOnnxRequest; the legacy four-argument provider call is unsupported"
            )
        # The planner has already verified source provenance.  Do not retain or
        # invoke a provider here: this target receives neutral, attested bytes.
        materialized = _verify_neutral_request(request)
        return self._prepare_from_materialized(materialized, request.contract)

    def _prepare_from_materialized(self, materialized: MaterializedOnnx, contract: VisionContract) -> PreparedArtifact:
        if self.engine_output is None:
            raise ValueError("TensorRT target requires an explicit engine_output")
        if contract.input.layout != "NCHW":
            raise ValueError("TensorRT target requires an NCHW VisionContract")
        if (
            contract.raw_output.semantics not in {"boxes_scores_classes", "yolo_end2end"}
            or contract.result.box_format != "xyxy"
            or contract.result.coordinate_space != "original_image"
        ):
            raise ValueError("TensorRT target supports original-image xyxy boxes/scores/classes or yolo_end2end only")
        if contract.result.nms_owner not in self.capabilities.nms_owners:
            raise ValueError("TensorRT target does not execute target/contract NMS")
        declared_shapes = (contract.input.shape,) + tuple(item.shape for item in contract.raw_output.tensors)
        if any(any(not isinstance(dim, int) for dim in shape) for shape in declared_shapes):
            raise ValueError("TensorRT static target rejects symbolic/dynamic contract shapes")
        output = self.engine_output
        if output.exists() or output.is_symlink():
            raise ValueError("TensorRT engine output must not already exist or be a symlink")
        if output.suffix.lower() not in {".engine", ".plan"}:
            raise ValueError("TensorRT engine_output must use .engine or .plan")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot verified ONNX bytes into a private, owned path before the
        # builder subprocess; never hand it a mutable provider path.
        if self._sha(materialized.lineage_path) != materialized.lineage_sha256:
            raise ValueError("materialized ONNX lineage changed before TensorRT build")
        lineage_body = json.loads(materialized.lineage_path.read_text())
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".vision-buildkit-", suffix=".onnx", delete=False
        ) as handle:
            snapshot = Path(handle.name)
            handle.write(materialized.material_path.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        if self._sha(snapshot) != materialized.sha256:
            snapshot.unlink(missing_ok=True)
            raise ValueError("materialized ONNX changed before TensorRT build")
        from .backends import TensorRTBackend

        try:
            executable = TensorRTBackend.resolve_trtexec(self.trtexec_path)
            executable_sha = self._sha(Path(executable))
        except Exception:
            snapshot.unlink(missing_ok=True)
            raise
        staged = output.parent / f".{output.name}.{secrets.token_hex(12)}.tmp"
        log_path = output.with_suffix(output.suffix + ".build.log")
        published_identity: tuple[int, int] | None = None

        def remove_owned_publication() -> None:
            if published_identity is None:
                return
            try:
                current = output.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == published_identity:
                    output.unlink()
            except FileNotFoundError:
                pass

        if log_path.exists() or log_path.is_symlink():
            snapshot.unlink(missing_ok=True)
            raise ValueError("TensorRT build log output must not already exist or be a symlink")
        request = type(
            "TensorRTRequest",
            (),
            {
                "precision": self.precision,
                "calibration_cache_path": None,
                "profiles": [],
                "dynamic": False,
                "trt_timeout_seconds": self.timeout_seconds,
                "trtexec_path": executable,
                "fp32_tactics_no_tf32": self.precision == "fp32",
            },
        )()
        try:
            _validate_onnx_contract(snapshot, contract)
            built, command, logs, fingerprint = TensorRTBackend.build_atomic(snapshot, staged, request)
            if built != staged or staged.is_symlink() or not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError("TensorRT builder returned an invalid staged engine")
            # Only a redacted diagnostic digest is retained in the receipt.
            with log_path.open("x", encoding="utf-8") as handle:
                handle.write(self._redact_log(logs))
                handle.flush()
                os.fsync(handle.fileno())
            staged_sha = self._sha(staged)
            # link() is no-clobber publication, unlike os.replace().  It also
            # makes the final name refer to the exact bytes we just hashed.
            os.link(staged, output)
            staged.unlink()
            current = output.stat(follow_symlinks=False)
            published_identity = (current.st_dev, current.st_ino)
            if self._sha(output) != staged_sha:
                raise RuntimeError("published TensorRT engine bytes changed during build")
            if self._sha(Path(executable)) != executable_sha:
                raise RuntimeError("trtexec executable changed during TensorRT build")
        except Exception:
            staged.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)
            remove_owned_publication()
            raise
        finally:
            snapshot.unlink(missing_ok=True)
        try:
            engine_sha = self._sha(output)
            log_sha = self._sha(log_path)
            clean_log = log_path.read_text()
            # The digest is self-contained in the descriptor; do not leave a
            # relocatability-breaking sidecar with build-host diagnostics.
            log_path.unlink()
            trt_version = str(fingerprint.get("trtexec", ""))
            if not trt_version:
                raise RuntimeError("TensorRT builder did not report a version")
            # Persist no user filesystem paths.  The canonical receipt is portable
            # with an engine when copied into a capsule or registry.
            redacted_command = self._redact_command(command)
            if tuple(redacted_command) != self._expected_command(self.precision):
                raise RuntimeError("TensorRT builder command contradicts certified static precision policy")
            build_receipt = {
                "schema": 1,
                "source_plugin": materialized.plugin_id,
                "source_plugin_version": materialized.plugin_version,
                "target_plugin": self.plugin_id,
                "target_plugin_version": self.plugin_version,
                "source_sha256": materialized.source_sha256,
                "onnx_sha256": materialized.sha256,
                "onnx_lineage": lineage_body,
                "onnx_lineage_sha256": materialized.lineage_sha256,
                "contract_json": contract.canonical_json(),
                "engine_sha256": engine_sha,
                "build_log_sha256": log_sha,
                "build_log": clean_log,
                "build_command": redacted_command,
                "requested_precision": self.precision,
                "effective_precision": "fp32_tactics_no_tf32" if self.precision == "fp32" else "mixed_fp16_fp32",
                "dynamic": False,
                "interface": {
                    "input": {
                        "name": contract.input.input_name,
                        "shape": contract.input.shape,
                        "dtype": contract.input.dtype,
                    },
                    "outputs": [item.model_dump(mode="json") for item in contract.raw_output.tensors],
                },
                "build_host": self._build_host_metadata(executable, trt_version),
            }
            token = secrets.token_urlsafe(32)
            receipt_json = json.dumps(build_receipt, sort_keys=True, separators=(",", ":"))
            self._artifact_receipts[token] = (
                engine_sha,
                contract.digest,
                hashlib.sha256(receipt_json.encode()).hexdigest(),
            )
            descriptors = (
                ("contract_json", contract.canonical_json()),
                ("source_sha256", materialized.source_sha256),
                ("onnx_sha256", materialized.sha256),
                ("engine_sha256", engine_sha),
                ("build_log_sha256", log_sha),
                ("build_receipt_json", receipt_json),
                ("target_receipt", token),
                ("requested_precision", self.precision),
                ("effective_precision", "fp32_tactics_no_tf32" if self.precision == "fp32" else "mixed_fp16_fp32"),
                ("dynamic", "false"),
            )
            return PreparedArtifact(
                output, engine_sha, self.plugin_id, self.plugin_version, contract.digest, descriptors
            )
        except Exception:
            remove_owned_publication()
            log_path.unlink(missing_ok=True)
            raise

    def _prepare_onnx(self, source, materialized, contract):
        raise RuntimeError("TensorRT target requires source receipt via prepare_materialized")

    def verify_artifact(self, artifact: PreparedArtifact, contract: VisionContract) -> bool:
        try:
            if contract.input.layout != "NCHW":
                return False
            items = artifact.descriptors
            values = dict(items)
            required = {
                "contract_json",
                "source_sha256",
                "onnx_sha256",
                "engine_sha256",
                "build_log_sha256",
                "build_receipt_json",
                "target_receipt",
                "requested_precision",
                "effective_precision",
                "dynamic",
            }
            if len(values) != len(items) or set(values) != required:
                return False
            if artifact.material_path.is_symlink() or not artifact.material_path.is_file():
                return False
            receipt = json.loads(values["build_receipt_json"])
            issued = self._artifact_receipts.get(values["target_receipt"])
            if (
                artifact.plugin_id != self.plugin_id
                or artifact.plugin_version != self.plugin_version
                or artifact.contract_digest != contract.digest
                or values["contract_json"] != contract.canonical_json()
                or values["source_sha256"] != contract.provenance.source_sha256
                or not (self._sha(artifact.material_path) == artifact.sha256 == values["engine_sha256"])
                or values["requested_precision"] not in {"fp32", "fp16"}
                or values["effective_precision"]
                != ("fp32_tactics_no_tf32" if values["requested_precision"] == "fp32" else "mixed_fp16_fp32")
                or values["dynamic"] != "false"
                or issued
                != (
                    values["engine_sha256"],
                    contract.digest,
                    hashlib.sha256(values["build_receipt_json"].encode()).hexdigest(),
                )
            ):
                return False
            return (
                set(receipt)
                == {
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
                and receipt["schema"] == 1
                and receipt["source_sha256"] == values["source_sha256"]
                and receipt["source_plugin"] == receipt["onnx_lineage"]["source_plugin"]
                and receipt["source_plugin_version"] == receipt["onnx_lineage"]["source_plugin_version"]
                and receipt["target_plugin"] == self.plugin_id
                and receipt["target_plugin_version"] == self.plugin_version
                and receipt["onnx_sha256"] == values["onnx_sha256"]
                and receipt["engine_sha256"] == values["engine_sha256"]
                and receipt["build_log_sha256"] == values["build_log_sha256"]
                and hashlib.sha256(receipt["build_log"].encode()).hexdigest() == values["build_log_sha256"]
                and receipt["contract_json"] == contract.canonical_json()
                and receipt["requested_precision"] == values["requested_precision"]
                and receipt["effective_precision"] == values["effective_precision"]
                and receipt["dynamic"] is False
                and receipt["onnx_lineage"]
                == {
                    "schema": 1,
                    "source_sha256": values["source_sha256"],
                    "onnx_sha256": values["onnx_sha256"],
                    "source_plugin": receipt["onnx_lineage"]["source_plugin"],
                    "source_plugin_version": receipt["onnx_lineage"]["source_plugin_version"],
                    "contract_digest": contract.digest,
                }
                and hashlib.sha256(
                    json.dumps(receipt["onnx_lineage"], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                == receipt["onnx_lineage_sha256"]
                and receipt["interface"]["input"]
                == {
                    "name": contract.input.input_name,
                    "shape": list(contract.input.shape),
                    "dtype": contract.input.dtype,
                }
                and receipt["interface"]["outputs"]
                == [item.model_dump(mode="json") for item in contract.raw_output.tensors]
                and tuple(receipt["build_command"]) == self._expected_command(values["requested_precision"])
                and set(receipt["build_host"])
                >= {"os", "architecture", "builder_target", "trtexec_sha256", "trtexec_version"}
                and all(receipt["build_host"][key] for key in receipt["build_host"])
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            return False

    def prepare_input(self, artifact, sample, contract):
        if not self.verify_artifact(artifact, contract):
            raise ValueError("TensorRT artifact handle is not bound to verified material and contract")
        return preprocess(contract, sample)

    @staticmethod
    def _runtime_provenance(runtime) -> dict[str, str]:
        trt = getattr(runtime, "trt", None)
        version = str(getattr(trt, "__version__", "unknown"))
        metadata = {
            "execution_provider": "TensorRTExecutionProvider",
            "execution_device": "cuda",
            "tensorrt_python_version": version,
        }
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            metadata.update({"cuda_device": props.name, "cuda_compute_capability": f"{props.major}.{props.minor}"})
        except Exception as exc:
            raise RuntimeError("TensorRT target refuses non-CUDA execution") from exc
        return metadata

    def run(self, artifact, sample):
        import numpy as np

        if not isinstance(sample, PreparedInput):
            raise ValueError("TensorRT target requires a contract PreparedInput")
        contract = self._contract_from_artifact(artifact)
        if not self.verify_artifact(artifact, contract):
            raise ValueError("TensorRT artifact handle is not bound to verified material and contract")
        if tuple(sample.tensor.shape) != tuple(contract.input.shape):
            raise ValueError("TensorRT input shape contradicts the contract")
        from .backends import TensorRTBackend

        runtime = TensorRTBackend(str(artifact.material_path), input_name=contract.input.input_name)
        if tuple(runtime.input_names) != (contract.input.input_name,) or tuple(runtime.output_names) != tuple(
            item.name for item in contract.raw_output.tensors
        ):
            raise ValueError("TensorRT engine interface contradicts the contract")
        expected_input = (tuple(contract.input.shape), contract.input.dtype)
        if getattr(runtime, "input_specs", {}).get(contract.input.input_name) != expected_input:
            raise ValueError("TensorRT engine input shape/dtype contradicts the contract")
        expected_outputs = {item.name: (tuple(item.shape), item.dtype) for item in contract.raw_output.tensors}
        if getattr(runtime, "output_specs", {}) != expected_outputs:
            raise ValueError("TensorRT engine output shape/dtype contradicts the contract")
        tensors = runtime.run(sample.tensor)
        if set(tensors) != {item.name for item in contract.raw_output.tensors}:
            raise ValueError("TensorRT runtime outputs contradict the contract")
        for spec in contract.raw_output.tensors:
            actual = np.asarray(tensors[spec.name])
            if actual.ndim != len(spec.shape) or any(
                isinstance(expected, int) and expected != observed
                for expected, observed in zip(spec.shape, actual.shape, strict=True)
            ):
                raise ValueError(f"TensorRT output {spec.name!r} shape contradicts the contract")
        metadata = self._runtime_provenance(runtime) | {"input_sha256": sample.input_sha256}
        if contract.raw_output.semantics == "yolo_end2end":
            if sample.trace is None:
                raise ValueError("yolo_end2end requires geometry trace")
            from .decoders import yolo_end2end_contract

            result = yolo_end2end_contract(
                tensors[contract.raw_output.tensors[0].name], sample.trace, contract, image_size=sample.image_size
            )
            result.metadata.update(metadata)
            return result
        roles = contract.raw_output.roles or {}
        boxes = np.asarray(tensors[roles["boxes"]]).reshape(-1, 4)
        scores = np.asarray(tensors[roles["scores"]]).reshape(-1)
        classes = np.asarray(tensors[roles["classes"]]).reshape(-1)
        if len(boxes) != len(scores) or len(boxes) != len(classes):
            raise ValueError("TensorRT boxes/scores/classes output cardinality mismatch")
        return DetectionResult(
            [Box(*map(float, row)) for row in boxes],
            [float(x) for x in scores],
            [int(x) for x in classes],
            {index: label for index, label in enumerate(contract.labels)},
            sample.image_size,
            metadata,
        )


class PluginRegistry:
    def __init__(self):
        self.sources: dict[str, SourceAdapter] = {}
        self.targets: dict[str, TargetPlugin] = {}

    def register_source(self, plugin: SourceAdapter) -> None:
        self._register(self.sources, plugin, SourceAdapter)

    def register_target(self, plugin: TargetPlugin) -> None:
        self._register(self.targets, plugin, TargetPlugin)

    @staticmethod
    def _register(bucket, plugin, base):
        if (
            not isinstance(plugin, base)
            or not getattr(plugin, "plugin_id", "")
            or not getattr(plugin, "plugin_version", "")
        ):
            raise TypeError("plugin must implement stable plugin_id and plugin_version")
        if plugin.plugin_id in bucket:
            raise ValueError(f"plugin collision: {plugin.plugin_id}")
        bucket[plugin.plugin_id] = plugin

    def discover(self, group: str = "vision.buildkit.plugins") -> None:
        entries = metadata.entry_points()
        selected = entries.select(group=group) if hasattr(entries, "select") else entries.get(group, ())
        sources, targets = dict(self.sources), dict(self.targets)
        try:
            for entry in sorted(selected, key=lambda e: e.name):
                plugin = entry.load()()
                if isinstance(plugin, SourceAdapter):
                    self.register_source(plugin)
                elif isinstance(plugin, TargetPlugin):
                    self.register_target(plugin)
                else:
                    raise TypeError("entry point must create SourceAdapter or TargetPlugin")
        except Exception:
            self.sources, self.targets = sources, targets
            raise


def make_tensorrt_target(**config: Any) -> TensorRTTargetPlugin:
    """Create a configured TensorRT target; there is deliberately no default."""
    return TensorRTTargetPlugin(**config)


def make_openvino_target(**config: Any) -> OpenVinoTargetPlugin:
    """Create an explicitly configured OpenVINO CPU target."""
    return OpenVinoTargetPlugin(**config)


DEFAULT_REGISTRY = PluginRegistry()
DEFAULT_REGISTRY.register_source(PythonObjectSource())
DEFAULT_REGISTRY.register_source(ManualOnnxSource())
from .providers import LibreYoloSourceAdapter, RFDetrSourceAdapter, UltralyticsSourceAdapter  # noqa: E402

DEFAULT_REGISTRY.register_source(UltralyticsSourceAdapter())
DEFAULT_REGISTRY.register_source(LibreYoloSourceAdapter())
DEFAULT_REGISTRY.register_source(RFDetrSourceAdapter())
try:
    metadata.version("onnxruntime")
    DEFAULT_REGISTRY.register_target(OnnxRuntimeTargetPlugin())
except metadata.PackageNotFoundError:
    pass
