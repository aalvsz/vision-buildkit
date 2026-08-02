"""Canonical, contract-bound image preprocessing shared by certified runtimes."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .contract import VisionContract
from .processing import GeometryTrace, direct_resize, letterbox, libreyolo_yolo9_letterbox


@dataclass(frozen=True)
class PreparedInput:
    """A single immutable execution input and the source-image identity."""

    tensor: NDArray[np.generic]
    image_size: tuple[int, int]
    input_sha256: str
    trace: GeometryTrace | None = None


def image_payload(sample: Path | bytes) -> bytes:
    if isinstance(sample, Path):
        if sample.is_symlink() or not sample.is_file():
            raise ValueError("case image must be a regular file")
        return sample.read_bytes()
    if isinstance(sample, bytes):
        return sample
    raise ValueError("runtime target requires verified image bytes or Path")


def preprocess(contract: VisionContract, sample: Path | bytes | Image.Image) -> PreparedInput:
    """Execute every currently expressible transform in declared order."""
    if isinstance(sample, Image.Image):
        image = sample.copy()
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        payload = encoded.getvalue()
    else:
        payload = image_payload(sample)
        image = Image.open(io.BytesIO(payload))
        image.load()
    image_size = image.size
    array = np.asarray(image.convert("RGB"))
    transposed = False
    trace = GeometryTrace(image_size, image_size, 1.0, 1.0)
    for number, transform in enumerate(contract.preprocessing):
        if transform.kind == "color_convert":
            if transform.source == transform.target:
                continue
            if transform.target == "BGR":
                array = array[..., ::-1]
            elif transform.target == "GRAY":
                array = np.asarray(Image.fromarray(array).convert("L"))[..., None]
            elif transform.source == "GRAY":
                array = np.asarray(Image.fromarray(array[..., 0]).convert("RGB"))
            else:
                array = array[..., ::-1]
        elif transform.kind in {"resize", "letterbox"}:
            if transform.kind == "resize" and transform.implementation == "opencv_linear_scale_fill":
                array, trace = direct_resize(Image.fromarray(array), (transform.width, transform.height))
                continue
            if transform.kind == "letterbox" and transform.implementation == "ultralytics_letterbox_cv2":
                if number != 0:
                    raise ValueError("ultralytics letterbox must be the first transform")
                if (
                    transform.placement != "center"
                    or transform.rounding != "round_half_up"
                    or transform.scale != "min_ratio"
                    or transform.interpolation != "bilinear"
                ):
                    raise ValueError("ultralytics letterbox requires its declared center/round/inter-linear semantics")
                array, trace = letterbox(image, (transform.width, transform.height), transform.pad_value)
                continue
            if transform.kind == "letterbox" and transform.implementation == "libreyolo_yolo9_top_left_cv2":
                if number != 0:
                    raise ValueError("LibreYOLO YOLO9 letterbox must be the first transform")
                if (
                    transform.placement != "top_left"
                    or transform.rounding != "floor"
                    or transform.scale != "min_ratio"
                    or transform.interpolation != "bilinear"
                ):
                    raise ValueError("LibreYOLO YOLO9 letterbox requires its declared top-left/int/linear semantics")
                array, trace = libreyolo_yolo9_letterbox(
                    image, (transform.width, transform.height), transform.pad_value
                )
                continue
            if transform.kind == "resize" and transform.implementation == "torchvision_bilinear_antialias_false":
                if transform.interpolation != "bilinear":
                    raise ValueError("torch tensor resize requires bilinear interpolation")
                try:
                    import torch
                    import torch.nn.functional as functional
                except ImportError as exc:
                    raise RuntimeError("torch_bilinear_antialias_false requires torch") from exc
                height, width = transform.height, transform.width
                tensor = torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
                tensor = functional.interpolate(
                    tensor, size=(height, width), mode="bilinear", align_corners=False, antialias=False
                )
                array = tensor.squeeze(0).permute(1, 2, 0).numpy()
                trace = GeometryTrace(
                    image_size,
                    (width, height),
                    width / image_size[0],
                    height / image_size[1],
                )
                continue
            resample = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
            }[transform.interpolation]
            height, width = transform.height, transform.width
            if transform.kind == "resize" or transform.scale == "exact":
                array = np.asarray(Image.fromarray(array).resize((width, height), resample))
                trace = GeometryTrace(
                    image_size,
                    (width, height),
                    width / image_size[0],
                    height / image_size[1],
                )
            else:
                scale = min(width / array.shape[1], height / array.shape[0])
                rounder = int if transform.rounding == "floor" else lambda value: int(value + 0.5)
                new_width, new_height = rounder(array.shape[1] * scale), rounder(array.shape[0] * scale)
                resized = np.asarray(Image.fromarray(array).resize((new_width, new_height), resample))
                canvas = np.full((height, width, resized.shape[2]), transform.pad_value, dtype=resized.dtype)
                left, top = (
                    ((width - new_width) // 2, (height - new_height) // 2)
                    if transform.placement == "center"
                    else (0, 0)
                )
                canvas[top : top + new_height, left : left + new_width] = resized
                array = canvas
        elif transform.kind == "normalize":
            array = array.astype(np.float32)
            array = (array - np.asarray(transform.mean, dtype=np.float32)) / np.asarray(transform.std, dtype=np.float32)
        elif transform.kind == "transpose":
            if number != len(contract.preprocessing) - 1:
                raise ValueError("transpose must be the final preprocessing transform")
            array = np.expand_dims(array, 0).transpose(transform.order)
            transposed = True
    if array.dtype != np.float32 and contract.input.dtype != "uint8":
        array = array.astype(np.float32)
    if not transposed:
        array = array.transpose(2, 0, 1) if contract.input.layout == "NCHW" else array
        array = array[None]
    dtype = {"float32": np.float32, "float16": np.float16, "uint8": np.uint8}[contract.input.dtype]
    tensor = array.astype(dtype, copy=False)
    if tensor.ndim != 4 or tuple(tensor.shape) != tuple(contract.input.shape):
        raise ValueError("preprocessing tensor contradicts declared input shape")
    tensor.setflags(write=False)
    return PreparedInput(
        tensor=tensor, image_size=image_size, input_sha256=hashlib.sha256(payload).hexdigest(), trace=trace
    )
