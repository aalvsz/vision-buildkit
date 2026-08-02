"""Built-in checkpoint providers; imports and runtime objects are strictly lazy."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .contract import VisionContract
from .decoders import RFDETR_COCO_CATEGORY_IDS, RFDETR_COCO_LABELS
from .plugins import (
    Capabilities,
    MaterializedOnnx,
    PreparedSource,
    SourceAdapter,
    _materialized,
    _validate_onnx_contract,
    _verify_source_handle,
)
from .types import Box, DetectionResult


@dataclass(frozen=True)
class ProviderInput:
    data: bytes
    sha256: str
    image_size: tuple[int, int]


class _CheckpointProvider(SourceAdapter):
    """Common, stateless checkpoint-provider protocol.

    The only supported provider calls are ``Loader(path)``,
    ``model.predict(image, verbose=False, imgsz=H, conf=threshold, rect=False)``
    and ``model.export(format='onnx', imgsz=H, batch=B, dynamic=False,
    verbose=False)``.  A provider with a different API fails explicitly.
    """

    package: str
    loader_name: str = "YOLO"
    extra: str
    capabilities = Capabilities(
        precisions=frozenset({"fp32"}),
        modes=frozenset({"static"}),
        nms_owners=frozenset({"source"}),
    )

    @property
    def plugin_version(self) -> str:
        try:
            return metadata.version(self.package)
        except metadata.PackageNotFoundError:
            return "unavailable"

    def _load(self, source: Path):
        try:
            module = importlib.import_module(self.package)
            loader = getattr(module, self.loader_name)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"{self.plugin_id} unavailable; install vision-buildkit[{self.extra}] "
                f"with {self.package}.{self.loader_name} support"
            ) from exc
        try:
            return loader(str(source))
        except Exception as exc:
            raise ValueError(f"{self.plugin_id} could not load detection checkpoint {source}") from exc

    def capture(self, source: Any, contract: VisionContract) -> Path:
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError("provider checkpoint must be a regular existing file")
        model = self._load(path)
        task = getattr(model, "task", None) or getattr(getattr(model, "model", None), "task", None)
        if task is not None and str(task).lower() != "detect":
            raise ValueError(f"{self.plugin_id} only supports detection checkpoints (got {task!r})")
        return path

    def _result(self, raw: Any, image_size: tuple[int, int] | None) -> DetectionResult:
        boxes = getattr(raw, "boxes", None)
        if boxes is None or not all(hasattr(boxes, x) for x in ("xyxy", "conf", "cls")):
            raise ValueError(f"{self.plugin_id} native prediction did not expose detection boxes")

        def values(value):
            return value.cpu().tolist() if hasattr(value, "cpu") else value.tolist()

        # Do not infer a device from host configuration or requested options:
        # record it only when the provider's actual result/tensors expose one.
        # Some providers materialize CPU views for serialization, hence inspect
        # the result tensors before calling ``.cpu()`` above.
        execution_device = None
        for value in (
            getattr(boxes, "xyxy", None),
            getattr(boxes, "conf", None),
            getattr(boxes, "cls", None),
            boxes,
            raw,
        ):
            device = getattr(value, "device", None)
            if device is not None and str(device):
                execution_device = str(device)
                break
        metadata = {"provider": self.plugin_id, "provider_version": self.plugin_version}
        if execution_device is not None:
            metadata["execution_device"] = execution_device

        return DetectionResult(
            [Box(*map(float, row)) for row in values(boxes.xyxy)],
            [float(x) for x in values(boxes.conf)],
            [int(x) for x in values(boxes.cls)],
            dict(getattr(raw, "names", {})),
            image_size,
            metadata=metadata,
        )

    def prepare_input(self, captured: PreparedSource, sample: Any, contract: VisionContract) -> ProviderInput:
        _verify_source_handle(captured, contract, self)
        if isinstance(sample, Path):
            if sample.is_symlink() or not sample.is_file():
                raise ValueError("provider input must be a regular image file")
            data = sample.read_bytes()
        elif isinstance(sample, bytes):
            data = sample
        else:
            raise ValueError("provider input must be a regular Path or verified bytes")
        try:
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(data)) as image:
                size = image.size
        except Exception as exc:
            raise ValueError("provider input is not a decodable image") from exc
        return ProviderInput(data, hashlib.sha256(data).hexdigest(), size)

    @staticmethod
    def _image(sample: ProviderInput):
        if hashlib.sha256(sample.data).hexdigest() != sample.sha256:
            raise ValueError("provider input is not bound to verified image bytes")
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(sample.data)) as image:
            return image.convert("RGB").copy()

    def run(self, captured: PreparedSource, sample: ProviderInput) -> DetectionResult:
        contract = VisionContract.model_validate_json(dict(captured.descriptors)["contract_json"])
        _verify_source_handle(captured, contract, self)
        height, width = (int(x) for x in contract.input.shape[-2:])
        if height != width:
            raise ValueError(f"{self.plugin_id} native reference inference requires square contract input")
        if not isinstance(sample, ProviderInput):
            raise ValueError("provider run requires a prepared content-bound image")
        image = self._image(sample)
        model = self._load(captured.material_path)
        try:
            raw = model.predict(image, verbose=False, imgsz=height, conf=contract.result.score_threshold, rect=False)[0]
        except Exception as exc:
            raise RuntimeError(f"{self.plugin_id} native detection inference failed") from exc
        result = self._result(raw, sample.image_size)
        result.metadata["input_sha256"] = sample.sha256
        return result

    def prepare(self, source: str | Path, contract: VisionContract) -> PreparedSource:
        prepared = super().prepare(source, contract)
        return PreparedSource(**(prepared.__dict__ | {"descriptors": (("contract_json", contract.canonical_json()),)}))

    def materialize_onnx(
        self, captured: PreparedSource, contract: VisionContract, output: str | Path
    ) -> MaterializedOnnx:
        _verify_source_handle(captured, contract, self)
        if (
            contract.raw_output.semantics not in {"boxes_scores_classes", "yolo_end2end"}
            or contract.result.box_format != "xyxy"
        ):
            raise ValueError("provider ONNX materialization only accepts declared xyxy supported contracts")
        height, width = (int(x) for x in contract.input.shape[-2:])
        if height != width:
            raise ValueError(f"{self.plugin_id} ONNX export requires square contract input")
        model = self._load(captured.material_path)
        try:
            options = {
                "format": "onnx",
                "imgsz": height,
                "batch": int(contract.input.shape[0]),
                "dynamic": False,
                "verbose": False,
            }
            if contract.raw_output.semantics == "yolo_end2end":
                options |= {"half": False, "nms": False}
            exported = Path(model.export(**options))
        except Exception as exc:
            raise RuntimeError(f"{self.plugin_id} official ONNX export failed") from exc
        if exported.is_symlink() or not exported.is_file():
            raise ValueError(f"{self.plugin_id} official export returned no regular ONNX artifact")
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != destination.resolve():
            shutil.copy2(exported, destination)
        _verify_source_handle(captured, contract, self)
        _validate_onnx_contract(destination, contract)
        return _materialized(destination, captured, contract, self)


class UltralyticsSourceAdapter(_CheckpointProvider):
    plugin_id = "ultralytics"
    package, extra = "ultralytics", "yolo"
    capabilities = Capabilities(
        precisions=frozenset({"fp32"}), modes=frozenset({"static"}), nms_owners=frozenset({"none", "source"})
    )


class LibreYoloSourceAdapter(_CheckpointProvider):
    """Lazy adapter limited to the current ``libreyolo.LibreYOLO`` API."""

    plugin_id = "libreyolo"
    package, extra = "libreyolo", "libreyolo"
    loader_name = "LibreYOLO"

    def run(self, captured: PreparedSource, sample: ProviderInput) -> DetectionResult:
        contract = VisionContract.model_validate_json(dict(captured.descriptors)["contract_json"])
        _verify_source_handle(captured, contract, self)
        height, width = (int(x) for x in contract.input.shape[-2:])
        if height != width:
            raise ValueError("libreyolo native reference inference requires square contract input")
        if not isinstance(sample, ProviderInput):
            raise ValueError("provider run requires a prepared content-bound image")
        image = self._image(sample)
        try:
            raw = self._load(captured.material_path)(
                image,
                conf=contract.result.score_threshold,
                iou=contract.result.nms_iou_threshold,
                imgsz=height,
                max_det=contract.result.max_detections,
            )
        except Exception as exc:
            raise RuntimeError("libreyolo native detection inference failed") from exc
        result = self._result(raw, sample.image_size)
        result.metadata["input_sha256"] = sample.sha256
        return result

    def materialize_onnx(
        self, captured: PreparedSource, contract: VisionContract, output: str | Path
    ) -> MaterializedOnnx:
        _verify_source_handle(captured, contract, self)
        if (
            contract.raw_output.semantics not in {"boxes_scores_classes", "yolo_end2end"}
            or contract.result.box_format != "xyxy"
        ):
            raise ValueError("libreyolo ONNX materialization only accepts declared xyxy supported contracts")
        if int(contract.input.shape[0]) != 1:
            raise ValueError("libreyolo ONNX materialization currently requires batch=1")
        if contract.raw_output.semantics == "yolo_end2end" and contract.input.dtype != "float32":
            raise ValueError("libreyolo YOLO end-to-end ONNX materialization requires fp32 input")
        height, width = (int(x) for x in contract.input.shape[-2:])
        if height != width:
            raise ValueError("libreyolo ONNX export requires square contract input")
        destination = Path(output)
        if destination.suffix.lower() != ".onnx":
            raise ValueError("libreyolo ONNX materialization requires a .onnx destination")
        if destination.resolve() == captured.material_path.resolve():
            raise ValueError("libreyolo ONNX destination must differ from the captured checkpoint")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if contract.raw_output.semantics == "yolo_end2end":
            if (
                contract.result.nms_owner != "source"
                or not contract.result.class_aware_nms
                or contract.result.nms_iou_threshold is None
            ):
                raise ValueError(
                    "libreyolo YOLO end-to-end export requires source-owned class-aware NMS with an IoU threshold"
                )
            self._materialize_end2end_onnx(captured, contract, destination, height)
            return _materialized(destination, captured, contract, self)

        options = {
            "format": "onnx",
            "output_path": str(destination),
            "imgsz": height,
            "batch": 1,
            "simplify": False,
            "dynamic": False,
            "half": False,
        }
        try:
            exported = self._load(captured.material_path).export(**options)
        except Exception as exc:
            raise RuntimeError("libreyolo official ONNX export failed") from exc
        actual = Path(exported) if isinstance(exported, (str, Path)) else destination
        if actual.is_symlink() or not actual.is_file():
            raise ValueError("libreyolo official export returned no regular ONNX artifact")
        if actual.resolve() != destination.resolve():
            shutil.copy2(actual, destination)
        _verify_source_handle(captured, contract, self)
        _validate_onnx_contract(destination, contract)
        return _materialized(destination, captured, contract, self)

    def _materialize_end2end_onnx(
        self, captured: PreparedSource, contract: VisionContract, destination: Path, height: int
    ) -> None:
        """Export, verify, and publish an end-to-end artifact without touching destination early."""
        fd, staging_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".onnx", dir=destination.parent)
        os.close(fd)
        staging = Path(staging_name)
        staging.unlink()
        try:
            options = {
                "format": "onnx",
                "output_path": str(staging),
                "imgsz": height,
                "batch": 1,
                "simplify": False,
                "dynamic": False,
                "half": False,
                "nms": True,
                "conf": contract.result.score_threshold,
                "iou": contract.result.nms_iou_threshold,
                "max_det": contract.result.max_detections,
            }
            try:
                exported = self._load(captured.material_path).export(**options)
            except Exception as exc:
                raise RuntimeError("libreyolo official ONNX export failed") from exc
            actual = Path(exported) if isinstance(exported, (str, Path)) else staging
            if actual.is_symlink() or not actual.is_file():
                raise ValueError("libreyolo official export returned no regular ONNX artifact")
            if actual.resolve() != staging.resolve():
                shutil.copy2(actual, staging)
            self._canonicalize_end2end_export(staging, contract)
            _verify_source_handle(captured, contract, self)
            _validate_onnx_contract(staging, contract)
            os.replace(staging, destination)
        finally:
            if staging.exists():
                staging.unlink()

    @staticmethod
    def _canonicalize_end2end_export(staging: Path, contract: VisionContract) -> None:
        """Publish LibreYOLO's NMS tensor while retaining its graph implementation.

        LibreYOLO v1.4 exports the public NMS result as ``output`` and also
        exposes its internal pre-NMS ``raw`` tensor.  The latter must remain in
        the graph (it can be consumed by internal nodes), but is not part of the
        declared end-to-end interface.
        """
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("LibreYOLO end-to-end ONNX canonicalization requires vision-buildkit[onnx]") from exc

        model = onnx.load(str(staging), load_external_data=False)
        outputs = list(model.graph.output)
        names = [item.name for item in outputs]
        if len(outputs) != 2 or len(set(names)) != len(names) or set(names) != {"output", "raw"}:
            raise ValueError(
                "libreyolo end-to-end export must expose exactly the 'output' and auxiliary 'raw' graph outputs"
            )
        graph_outputs = {item.name: item for item in outputs}
        metadata = {item.key: item.value for item in model.metadata_props}
        LibreYoloSourceAdapter._validate_end2end_metadata_labels(model.metadata_props, contract)
        if str(metadata.get("nms", "")).strip().lower() != "true":
            raise ValueError("libreyolo end-to-end metadata must prove nms=true")
        if str(metadata.get("nms_raw_output", "")).strip().lower() != "true":
            raise ValueError("libreyolo end-to-end metadata must prove nms_raw_output=true")
        try:
            nms_conf, nms_iou = float(metadata["nms_conf"]), float(metadata["nms_iou"])
            max_det = int(metadata["max_det"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("libreyolo end-to-end metadata must bind nms_conf, nms_iou, and max_det") from exc
        if (
            nms_conf != contract.result.score_threshold
            or nms_iou != contract.result.nms_iou_threshold
            or max_det != contract.result.max_detections
        ):
            raise ValueError("libreyolo end-to-end metadata contradicts contract NMS thresholds")
        output_tensor = graph_outputs["output"].type.tensor_type
        output_shape = tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in output_tensor.shape.dim)
        if output_tensor.elem_type != onnx.TensorProto.FLOAT or output_shape != (1, contract.result.max_detections, 6):
            raise ValueError("libreyolo end-to-end 'output' must be fp32 with shape [1, max_det, 6]")

        # Remove only the public graph-output declaration.  Do not remove the
        # raw value, producer, consumers, initializers, or any other graph data.
        del model.graph.output[:]
        model.graph.output.extend([graph_outputs["output"]])

        metadata["nms_raw_output"] = "false"
        del model.metadata_props[:]
        for key in sorted(metadata):
            item = model.metadata_props.add()
            item.key, item.value = key, metadata[key]

        # Re-serialize atomically within staging before the caller publishes it.
        fd, temporary_name = tempfile.mkstemp(prefix=f".{staging.name}.", suffix=".tmp", dir=staging.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            onnx.save(model, str(temporary))
            _validate_onnx_contract(temporary, contract)
            os.replace(temporary, staging)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _validate_end2end_metadata_labels(metadata_props: Any, contract: VisionContract) -> None:
        """Require the provider's declared class ABI to equal the contract.

        Metadata properties are a repeated ONNX field, so converting directly to
        a dictionary would silently accept duplicate keys.  `names` is required
        to be canonical JSON with stringified, contiguous class IDs: this avoids
        treating Python literals, reordered maps, or lossy coercions as an ABI.
        """
        required = ("names", "nc", "nb_classes")
        grouped: dict[str, list[str]] = {key: [] for key in required}
        for item in metadata_props:
            if item.key in grouped:
                grouped[item.key].append(item.value)
        if any(len(grouped[key]) != 1 for key in required):
            raise ValueError("libreyolo end-to-end metadata must contain exactly one names, nc, and nb_classes")

        expected_count = len(contract.labels)
        for key in ("nc", "nb_classes"):
            try:
                count = int(grouped[key][0])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"libreyolo end-to-end metadata {key} must be an integer") from exc
            if str(count) != grouped[key][0] or count != expected_count:
                raise ValueError(f"libreyolo end-to-end metadata {key} contradicts contract labels")

        try:
            pairs = json.loads(grouped["names"][0], object_pairs_hook=list)
        except json.JSONDecodeError as exc:
            raise ValueError("libreyolo end-to-end metadata names must be canonical JSON") from exc
        if not isinstance(pairs, list) or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
            raise ValueError("libreyolo end-to-end metadata names must be a JSON object")
        expected = [(str(index), label) for index, label in enumerate(contract.labels)]
        if pairs != expected:
            raise ValueError("libreyolo end-to-end metadata names contradict contract labels")


class RFDetrSourceAdapter(SourceAdapter):
    """Lazy RF-DETR Nano source plugin with a deliberately narrow frozen ABI.

    Native prediction is accepted only when the vendor returns the same raw
    tensors as the exported ABI.  A postprocessed vendor detection object has
    no evidence binding it to this decoder, so it fails closed.
    """

    plugin_id = "rfdetr"
    package = "rfdetr"
    extra = "rfdetr"
    validated_version = "1.9.0"
    capabilities = Capabilities(
        precisions=frozenset({"fp32"}), modes=frozenset({"static"}), nms_owners=frozenset({"none"})
    )

    @property
    def plugin_version(self) -> str:
        try:
            return metadata.version(self.package)
        except metadata.PackageNotFoundError:
            return "unavailable"

    @staticmethod
    def _validate_contract(contract: VisionContract) -> None:
        expected_tensors = (("dets", "float32", (1, 300, 4)), ("labels", "float32", (1, 300, 91)))
        observed_tensors = tuple((item.name, item.dtype, item.shape) for item in contract.raw_output.tensors)
        if (
            contract.task != "detection"
            or contract.input.color != "RGB"
            or contract.input.layout != "NCHW"
            or contract.input.dtype != "float32"
            or contract.input.input_name != "input"
            or contract.input.shape != (1, 3, 384, 384)
            or contract.raw_output.semantics != "custom"
            or contract.raw_output.decoder_plugin != "rfdetr.sigmoid_coco_v1"
            or observed_tensors != expected_tensors
            or contract.result.box_format != "cxcywh"
            or contract.result.coordinate_space != "original_image"
            or contract.result.nms_owner != "none"
            or contract.result.suppression != "none"
            or contract.result.score_threshold != 0.30
            or contract.result.top_k != 300
            or contract.result.max_detections != 300
            or contract.result.ordering != "score_desc"
            or contract.result.tie_policy != "stable"
            or not contract.result.clip_boxes
            or tuple(contract.coco_category_ids) != RFDETR_COCO_CATEGORY_IDS
            or tuple(contract.labels) != RFDETR_COCO_LABELS
            or len(contract.preprocessing) != 3
            or contract.preprocessing[0].kind != "resize"
            or contract.preprocessing[0].interpolation != "bilinear"
            or contract.preprocessing[1].kind != "normalize"
            or contract.preprocessing[1].mean != (0.485, 0.456, 0.406)
            or contract.preprocessing[1].std != (0.229, 0.224, 0.225)
            or contract.preprocessing[2].kind != "transpose"
        ):
            raise ValueError("RF-DETR Nano requires its exact fp32 static 384px raw ABI contract")

    def _load(self, source: Path):
        try:
            if metadata.version(self.package) != self.validated_version:
                raise RuntimeError(f"rfdetr must be exactly {self.validated_version}")
            module = importlib.import_module(self.package)
            factory = module.RFDETRNano
        except (ImportError, AttributeError, metadata.PackageNotFoundError) as exc:
            raise RuntimeError(
                "rfdetr unavailable; install vision-buildkit[rfdetr] with rfdetr.RFDETRNano support"
            ) from exc
        try:
            return factory(pretrain_weights=str(source))
        except Exception as exc:
            raise ValueError(f"rfdetr could not load RF-DETR Nano checkpoint {source}") from exc

    def capture(self, source: Any, contract: VisionContract) -> Path:
        self._validate_contract(contract)
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError("RF-DETR checkpoint must be a regular existing file")
        # Validate the official factory interface at capture time, but retain no
        # model object in the content-bound source handle.
        self._load(path)
        return path

    def prepare(self, source: str | Path, contract: VisionContract) -> PreparedSource:
        prepared = super().prepare(source, contract)
        return PreparedSource(**(prepared.__dict__ | {"descriptors": (("contract_json", contract.canonical_json()),)}))

    def prepare_input(self, captured: PreparedSource, sample: Any, contract: VisionContract) -> ProviderInput:
        self._validate_contract(contract)
        return _CheckpointProvider.prepare_input(self, captured, sample, contract)

    def run(self, captured: PreparedSource, sample: ProviderInput) -> DetectionResult:
        contract = VisionContract.model_validate_json(dict(captured.descriptors)["contract_json"])
        self._validate_contract(contract)
        _verify_source_handle(captured, contract, self)
        if not isinstance(sample, ProviderInput):
            raise ValueError("rfdetr run requires a prepared content-bound image")
        image = _CheckpointProvider._image(sample)
        try:
            raw = self._load(captured.material_path).predict(image, shape=(384, 384), threshold=0.30)
        except Exception as exc:
            raise RuntimeError("rfdetr native prediction failed") from exc
        try:
            boxes = np.asarray(raw.xyxy)
            scores = np.asarray(raw.confidence)
            classes = np.asarray(raw.class_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("rfdetr native predict did not expose the official Detections ABI") from exc
        if (
            boxes.dtype != np.dtype("float32")
            or scores.dtype != np.dtype("float32")
            or classes.dtype != np.dtype("int64")
            or boxes.ndim != 2
            or boxes.shape[1:] != (4,)
            or scores.shape != (boxes.shape[0],)
            or classes.shape != (boxes.shape[0],)
            or not np.isfinite(boxes).all()
            or not np.isfinite(scores).all()
            or any(int(category) not in RFDETR_COCO_CATEGORY_IDS for category in classes)
        ):
            raise ValueError("rfdetr native Detections ABI contradicts the pinned RF-DETR contract")
        rows = sorted(
            (
                (-float(score), index)
                for index, score in enumerate(scores)
                if float(score) > contract.result.score_threshold
            )
        )[:300]
        width, height = sample.image_size
        sparse_to_dense = {category: index for index, category in enumerate(RFDETR_COCO_CATEGORY_IDS)}

        def clipped(index: int) -> Box:
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            return Box(
                max(0.0, min(width, x1)),
                max(0.0, min(height, y1)),
                max(0.0, min(width, x2)),
                max(0.0, min(height, y2)),
            )

        result = DetectionResult(
            [clipped(index) for _, index in rows],
            [-score for score, _ in rows],
            [sparse_to_dense[int(classes[index])] for _, index in rows],
            {index: label for index, label in enumerate(RFDETR_COCO_LABELS)},
            image_size=sample.image_size,
        )
        result.metadata.update(
            {"provider": self.plugin_id, "provider_version": self.plugin_version, "input_sha256": sample.sha256}
        )
        return result

    def materialize_onnx(
        self, captured: PreparedSource, contract: VisionContract, output: str | Path
    ) -> MaterializedOnnx:
        self._validate_contract(contract)
        _verify_source_handle(captured, contract, self)
        destination = Path(output)
        if (
            destination.suffix.lower() != ".onnx"
            or destination.is_symlink()
            or destination.resolve() == captured.material_path.resolve()
        ):
            raise ValueError("rfdetr ONNX destination must be a distinct .onnx path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ValueError("rfdetr ONNX destination must not already exist or be a symlink")
        export_stage = Path(tempfile.mkdtemp(prefix=".rfdetr-export-", dir=destination.parent))
        try:
            exported = self._load(captured.material_path).export(
                output_dir=str(export_stage),
                format="onnx",
                shape=(384, 384),
                batch_size=1,
                opset_version=17,
                fp16=False,
                verbose=False,
            )
        except Exception as exc:
            shutil.rmtree(export_stage, ignore_errors=True)
            raise RuntimeError("rfdetr official ONNX export failed") from exc
        try:
            actual = Path(exported) if isinstance(exported, (str, Path)) else None
            expected = export_stage / "rfdetr-nano.onnx"
            if (
                actual is None
                or actual.resolve() != expected.resolve()
                or expected.is_symlink()
                or not expected.is_file()
            ):
                raise ValueError("rfdetr official export must return the staged rfdetr-nano.onnx artifact")
            _verify_source_handle(captured, contract, self)
            _validate_onnx_contract(expected, contract)
            digest = hashlib.sha256(expected.read_bytes()).hexdigest()
            stage = Path(tempfile.mkdtemp(prefix=".rfdetr-publish-", dir=destination.parent))
            lineage_body = {
                "schema": 1,
                "source_sha256": captured.sha256,
                "onnx_sha256": digest,
                "source_plugin": self.plugin_id,
                "source_plugin_version": self.plugin_version,
                "contract_digest": contract.digest,
            }
            staged_lineage = stage / "lineage.json"
            staged_model = stage / "model.onnx"
            shutil.copy2(expected, staged_model)
            with staged_lineage.open("xb") as handle:
                handle.write(json.dumps(lineage_body, sort_keys=True, separators=(",", ":")).encode())
                handle.flush()
                os.fsync(handle.fileno())
            entries = {item.name: item for item in stage.iterdir()}
            if set(entries) != {"model.onnx", "lineage.json"} or any(
                item.is_symlink() or not stat.S_ISREG(item.stat(follow_symlinks=False).st_mode)
                for item in entries.values()
            ):
                raise ValueError("rfdetr publication staging must contain exactly regular model.onnx and lineage.json")
            bundle_digest = hashlib.sha256(staged_model.read_bytes() + staged_lineage.read_bytes()).hexdigest()
            bundle = destination.parent / f"{destination.stem}-{bundle_digest}.rfdetr-bundle"
            for child in (staged_model, staged_lineage):
                with child.open("rb") as handle:
                    os.fsync(handle.fileno())
            stage_fd = os.open(stage, os.O_RDONLY)
            try:
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            if bundle.is_symlink() or (bundle.exists() and not bundle.is_dir()):
                raise ValueError("rfdetr content-addressed bundle target is not a real directory")
            if bundle.exists():
                if (
                    {child.name for child in bundle.iterdir()} != {"model.onnx", "lineage.json"}
                    or (bundle / "model.onnx").is_symlink()
                    or (bundle / "lineage.json").is_symlink()
                    or (bundle / "model.onnx").read_bytes() != staged_model.read_bytes()
                    or (bundle / "lineage.json").read_bytes() != staged_lineage.read_bytes()
                ):
                    raise ValueError("rfdetr content-addressed bundle conflicts with staged export")
                shutil.rmtree(stage)
            else:
                os.rename(stage, bundle)
            parent_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            shutil.rmtree(export_stage, ignore_errors=True)
            if "stage" in locals():
                shutil.rmtree(stage, ignore_errors=True)
        return _materialized(bundle / "model.onnx", captured, contract, self, lineage_path=bundle / "lineage.json")
