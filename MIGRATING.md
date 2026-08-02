# Migrating Vision BuildKit

## 0.2.0 to 0.2.1

`Resize` transforms now carry an explicit `implementation` discriminator. An
omitted value still means `pil`, preserving schema-v1 behavior for existing
general contracts. Provider-certified contracts must be explicit:

- RT-DETR combined v1 requires `opencv_linear_scale_fill` with bilinear
  interpolation. This matches Ultralytics RT-DETR `LetterBox(...,
  scale_fill=True)` and executes with OpenCV `INTER_LINEAR`.
- RF-DETR requires `torchvision_bilinear_antialias_false`, which performs the
  historical float `/255` conversion before `torch.nn.functional.interpolate`
  with `align_corners=False` and `antialias=False`.

The frozen RF-DETR contract now distinguishes its raw and public geometry:
`dets` remains normalized `cxcywh` in the generated manifest, while
`result.box_format` is `xyxy` in original-image coordinates after decoding.
Only the exact `rfdetr.sigmoid_coco_v1` Nano ABI is certifiable; other `custom`
decoder contracts remain rejected.

Old RT-DETR contracts that omitted the discriminator used PIL resizing and are
not certifiable as combined v1. Recreate the contract and rerun parity gates;
do not edit an existing capsule or relax its thresholds.

Certified 0.2.0 capsules with a general `Resize` contract remain verifiable.
Those capsules omitted `implementation` from the embedded contract (meaning
PIL at runtime) while their redundant top-level manifest projection named
`opencv_linear_scale_fill`. The verifier accepts only that exact historical
projection for `boxes_scores_classes` and `yolo_end2end`; execution continues
to use embedded-contract PIL semantics. This exception does not apply to the
strict RT-DETR combined or RF-DETR ABIs. New manifests project the declared
implementation truthfully.

The minimum supported Pydantic version is now 2.12.5 because canonical
serialization relies on `Field(exclude_if=...)` to omit the default PIL
discriminator while retaining nondefault implementations in contract digests.
Earlier 2.12 patch releases emit nested-union serializer warnings for this
otherwise valid omitted-field representation.

## 0.1.x to 0.2.0

Version 0.2.0 changes the source-to-target plugin boundary so target plugins no
longer receive a provider object or checkpoint handle. This is an intentional
pre-1.0 API break: it enforces the N-source-plus-M-target architecture rather
than allowing provider-specific target integrations.

### Target plugins

The old call passed provider material directly:

```python
artifact = target.prepare_materialized(materialized, captured, source, contract)
```

Use `BuildPlanner.materialize` in 0.2.0. The planner validates the provider
receipt, creates a sealed `NeutralOnnxRequest`, and gives only that neutral
request to the target:

```python
from vision.buildkit import BuildPlanner

captured, materialized, artifact = BuildPlanner().materialize(
    source=source_plugin,
    target=target_plugin,
    checkpoint="weights.pt",
    contract=contract,
    output="build/model.onnx",
)
```

Custom target plugins should implement `_prepare_onnx` and must not import or
invoke a source provider. Calling `prepare_materialized` with the old
four-argument shape fails before provider code or material is accessed.

### RT-DETR manifests and decoders

The public `vision.decoders.rtdetr_combined` callable and the legacy manifest
kind `rtdetr_combined` retain their 0.1.x behavior for existing callers. They
are not accepted as the strict certifiable ABI.

New RT-DETR contracts project to `rtdetr_combined_v1`. That decoder requires
the frozen `float32 [1, 300, 6]` output, validates the complete result contract,
and records the ABI identifier `rtdetr_combined.v1`. Do not relabel an old
manifest: recapture the model into a new Vision Contract and rerun parity
validation before creating a certified capsule.

### RF-DETR installation

The RF-DETR adapter is frozen to the upstream export ABI in
`rfdetr[onnx]==1.9.0`. Install it with:

```bash
uv sync --extra rfdetr
```

This makes the adapter version and its ONNX export dependencies reproducible;
it does not by itself establish model parity or hardware support.
