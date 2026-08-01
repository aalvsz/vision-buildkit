from __future__ import annotations

import numpy as np
from PIL import Image

from ..manifest import VisionManifest
from ..processing import GeometryTrace, direct_resize, letterbox


class DetectionProcessor:
    """The manifest is the single source of preprocessing truth for ONNX/Torch."""

    def __init__(self, manifest: VisionManifest):
        self.manifest = manifest

    def __call__(self, image: Image.Image):
        if self.manifest.contract is not None:
            return self._contract_process(image)
        height, width = (int(v) for v in self.manifest.input.shape[-2:])
        spec = self.manifest.preprocessing
        if spec.implementation == "ultralytics_letterbox_cv2":
            array, trace = letterbox(image, (width, height), spec.padding)
            tensor = array.astype(np.float32)
            if spec.normalization == "1/255":
                tensor /= 255.0
        elif spec.implementation == "torchvision_bilinear_antialias_false":
            # RF-DETR's published transform converts RGB bytes to a tensor first,
            # then invokes torchvision Resize(..., antialias=False). PIL resizing
            # is not numerically equivalent on natural images.
            try:
                import torch
                from torchvision.transforms import InterpolationMode
                from torchvision.transforms.functional import resize
            except ImportError as error:
                raise RuntimeError("torchvision_bilinear_antialias_false requires torch and torchvision") from error
            array = np.asarray(image.convert("RGB"))
            original_width, original_height = image.size
            trace = GeometryTrace(
                (original_width, original_height), (width, height), width / original_width, height / original_height
            )
            tensor_t = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
            tensor_t = resize(tensor_t, [height, width], InterpolationMode.BILINEAR, antialias=False)
            tensor = tensor_t.permute(1, 2, 0).numpy()
        elif spec.implementation == "opencv_linear_scale_fill":
            array, trace = direct_resize(image, (width, height))
            tensor = array.astype(np.float32)
            if spec.normalization == "1/255":
                tensor /= 255.0
        else:
            raise ValueError(f"unsupported preprocessing implementation: {spec.implementation}")
        if self.manifest.input.color == "BGR":
            tensor = tensor[..., ::-1]
        if spec.mean is not None:
            tensor = (tensor - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)
        tensor = tensor.transpose(2, 0, 1)[None]
        dtype = np.float16 if self.manifest.input.dtype == "float16" else np.float32
        return tensor.astype(dtype), trace

    def _contract_process(self, image: Image.Image):
        """Execute certified capsules from their embedded canonical contract."""
        from ..contract import VisionContract
        from ..preprocess import preprocess

        contract = VisionContract.model_validate(self.manifest.contract)
        prepared = preprocess(contract, image)
        if prepared.trace is None:
            raise ValueError("certified contract preprocessing lacks geometry trace")
        return prepared.tensor, prepared.trace
