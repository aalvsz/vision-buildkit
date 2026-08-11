#!/usr/bin/env python3
"""Deterministic, head-only fake-QAT primitives for the YOLO26 raw-output pilot.

This module intentionally does not invoke a trainer.  ``one_batch_step`` is the
only optimisation primitive: callers must supply a normal Ultralytics batch and
must opt in to any multi-batch schedule outside this bounded pilot.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TARGETS = tuple(
    [f"model.23.cv2.{index}.2" for index in range(3)]
    + [f"model.23.cv3.{index}.2" for index in range(3)]
)
INT8_QMIN, INT8_QMAX = -127, 127  # symmetric narrow-range INT8, no unused -128 bin
CHECKPOINT_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"


def seed_everything(seed: int = 260811) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)


def symmetric_int8_ste(value: torch.Tensor) -> torch.Tensor:
    """Native-MPS fake quantization with a straight-through gradient.

    torch.fake_quantize_per_tensor_affine dispatches an unsupported MPS cachemask
    kernel under fallback-disabled execution.  This is the equivalent auditable
    symmetric per-tensor operation: scale=max(abs(x))/127, round/clamp to int8,
    dequantize, then use the identity derivative.
    """
    max_abs = value.detach().abs().amax()
    scale = torch.clamp(max_abs / INT8_QMAX, min=torch.finfo(value.dtype).eps)
    quantized = torch.round(value / scale).clamp(INT8_QMIN, INT8_QMAX) * scale
    return value + (quantized - value).detach()


class STEQATConv2d(nn.Conv2d):
    """Conv2d whose weight and input activation are symmetric INT8 fake-quantized."""

    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__(
            source.in_channels, source.out_channels, source.kernel_size, source.stride,
            source.padding, source.dilation, source.groups, source.bias is not None,
            source.padding_mode,
        )
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if source.bias is not None and self.bias is not None:
                self.bias.copy_(source.bias)
        self.register_buffer("fake_quant_calls", torch.zeros((), dtype=torch.int64), persistent=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.fake_quant_calls.add_(1)
        return F.conv2d(
            symmetric_int8_ste(value), symmetric_int8_ste(self.weight), self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def _parent_and_leaf(model: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def get_target_modules(model: nn.Module) -> dict[str, nn.Module]:
    found = dict(model.named_modules())
    missing = [name for name in TARGETS if name not in found]
    if missing:
        raise RuntimeError(f"YOLO26 Detect-head identities changed; refusing QAT: {missing}")
    if len(found) != len(set(found)):
        raise RuntimeError("duplicate module names unexpectedly reported")
    return {name: found[name] for name in TARGETS}


def install_head_fake_qat(model: nn.Module) -> None:
    """Replace exactly the six approved raw-head terminal Conv2d modules."""
    targets = get_target_modules(model)
    if any(type(module) is not nn.Conv2d for module in targets.values()):
        raise RuntimeError("target set must be stock torch.nn.Conv2d before installation")
    for name, source in targets.items():
        parent, leaf = _parent_and_leaf(model, name)
        replacement = STEQATConv2d(source).to(device=source.weight.device, dtype=source.weight.dtype)
        if leaf.isdigit():
            parent[int(leaf)] = replacement  # type: ignore[index]
        else:
            setattr(parent, leaf, replacement)


def strip_head_fake_qat(model: nn.Module) -> None:
    """Restore stock Conv2d modules while retaining the trained target tensors."""
    for name, module in get_target_modules(model).items():
        if type(module) is not STEQATConv2d:
            raise RuntimeError(f"cannot strip non-QAT target {name}: {type(module).__name__}")
        plain = nn.Conv2d(
            module.in_channels, module.out_channels, module.kernel_size, module.stride,
            module.padding, module.dilation, module.groups, module.bias is not None,
            module.padding_mode,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            plain.weight.copy_(module.weight)
            if plain.bias is not None and module.bias is not None:
                plain.bias.copy_(module.bias)
        parent, leaf = _parent_and_leaf(model, name)
        if leaf.isdigit():
            parent[int(leaf)] = plain  # type: ignore[index]
        else:
            setattr(parent, leaf, plain)


def freeze_except_targets(model: nn.Module) -> tuple[str, ...]:
    allowed = set(TARGETS)
    for name, parameter in model.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        parameter.requires_grad = module_name in allowed
    trainable = tuple(name for name, p in model.named_parameters() if p.requires_grad)
    expected = tuple(f"{target}.{field}" for target in TARGETS for field in ("weight", "bias"))
    if trainable != expected:
        raise RuntimeError(f"freeze gate failed: {trainable}")
    return trainable


def parameter_hashes(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, parameter in model.named_parameters()
    }


def state_dict_hashes(model: nn.Module) -> dict[str, str]:
    """Hash parameters and buffers, so BatchNorm drift cannot hide behind freezing."""
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def assert_state_only_targets_changed(before: dict[str, str], model: nn.Module) -> set[str]:
    after = state_dict_hashes(model)
    changed = {name for name in before if before[name] != after[name]}
    allowed = {f"{target}.{field}" for target in TARGETS for field in ("weight", "bias", "fake_quant_calls")}
    unexpected = sorted(changed - allowed)
    if unexpected:
        raise RuntimeError(f"non-target state mutation: {unexpected}")
    return changed


def _training_head_without_buffer_updates(model: nn.Module) -> None:
    """Expose Detect training branch while all BatchNorm modules remain eval/frozen."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _teacher_training_head_eval_backbone(teacher: nn.Module) -> None:
    teacher.eval()
    teacher.model[-1].train()  # only Detect needs its training forward branch
    for module in teacher.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def changed_parameter_names(before: dict[str, str], model: nn.Module) -> set[str]:
    after = parameter_hashes(model)
    if before.keys() != after.keys():
        raise RuntimeError("parameter identities changed after QAT installation")
    return {name for name in before if before[name] != after[name]}


def assert_only_targets_changed(changed: Iterable[str]) -> None:
    unexpected = sorted(name for name in changed if name.rsplit(".", 1)[0] not in TARGETS)
    if unexpected:
        raise RuntimeError(f"non-head parameter mutation: {unexpected}")


def head_logit_consistency(student_outputs: dict[str, torch.Tensor], teacher_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """MSE over the pre-Sigmoid score logits from the ordinary training forward."""
    return F.mse_loss(student_outputs["scores"], teacher_outputs["scores"].detach())


def one_batch_step(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    consistency_weight: float = 0.01,
) -> dict[str, float]:
    """One standard Ultralytics detection-loss update plus score-logit consistency."""
    # YOLO('checkpoint.pt').model exposes saved args as a dict, but v8.4.112's
    # DetectionLoss consumes attribute-style hyperparameters.  The regular
    # Trainer performs this setup; this bounded standalone primitive must do it
    # explicitly before criterion initialization.
    if isinstance(getattr(student, "args", None), dict):
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.utils import IterableSimpleNamespace

        student.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **student.args})
    if batch["img"].device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("FAIL_CLOSED: MPS requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    _training_head_without_buffer_updates(student)
    _teacher_training_head_eval_backbone(teacher)
    optimizer.zero_grad(set_to_none=True)
    student_preds = student(batch["img"])
    detection_loss, _ = student.loss(batch, preds=student_preds)
    with torch.no_grad():
        teacher_preds = teacher(batch["img"])
    consistency = head_logit_consistency(student_preds["one2many"], teacher_preds["one2many"])
    # Ultralytics returns the three component losses for this checkpoint; reduce
    # them explicitly before backward instead of relying on Trainer-side scaling.
    loss = detection_loss.sum() + consistency_weight * consistency
    try:
        loss.backward()
    except RuntimeError as error:
        if "does not have a deterministic implementation" in str(error):
            raise RuntimeError(
                "FAIL_CLOSED: ordinary Ultralytics loss backward is nondeterministic on MPS; "
                "do not enable fallback or warn_only"
            ) from error
        raise
    optimizer.step()
    return {"detection_loss": float(detection_loss.detach().sum().cpu()), "consistency_loss": float(consistency.detach().cpu())}


def build_dataset_manifest(dataset_root: Path, calibration_names: Path, regression_dir: Path, output: Path) -> dict:
    image_dir, label_dir = dataset_root / "images" / "train2017", dataset_root / "labels" / "train2017"
    images, labels = sorted(image_dir.glob("*.jpg")), sorted(label_dir.glob("*.txt"))
    if {p.stem for p in images} != {p.stem for p in labels}:
        raise RuntimeError("image/label stem mismatch")
    files = sorted([*images, *labels])
    if not files:
        raise RuntimeError("empty QAT corpus")
    calibration = {line.strip() for line in calibration_names.read_text().splitlines() if line.strip()}
    regression_files = sorted(regression_dir.glob("*.jpg"))
    regression = {path.name for path in regression_files}
    records, overlaps = [], []
    for path in files:
        record = {"path": str(path.relative_to(dataset_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(record)
        if path.name in calibration or path.name in regression:
            overlaps.append(path.name)
    if overlaps:
        raise RuntimeError(f"train corpus overlap forbidden: {sorted(set(overlaps))}")
    manifest = {"schema_version": 2, "dataset_root": str(dataset_root), "files": records, "file_count": len(records),
                "image_label_stems_equal": True, "calibration_names_sha256": hashlib.sha256(calibration_names.read_bytes()).hexdigest(),
                "calibration_names": sorted(calibration), "regression_files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in regression_files], "overlap": []}
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def export_plain_raw_onnx(model: nn.Module, output: Path, sample: torch.Tensor) -> None:
    """Export a stock, raw-output ONNX. Call only after ``strip_head_fake_qat``."""
    if any(type(module) is STEQATConv2d for module in get_target_modules(model).values()):
        raise RuntimeError("refusing ONNX export with fake-QAT wrappers present")
    # Export and ORT parity use the same CPU numerical path; the trained source
    # remains on MPS for the caller's subsequent update steps.
    model = copy.deepcopy(model).cpu().eval()
    sample = sample.detach().cpu()
    model.model[-1].end2end = False
    # The pinned environment has no onnxscript; force the stock legacy exporter.
    torch.onnx.export(model, sample, output, opset_version=17, input_names=["images"], output_names=["output0"], dynamo=False)
