# Migrating Vision BuildKit

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
