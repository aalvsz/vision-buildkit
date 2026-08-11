# Axelera INT8 accuracy recovery (2026-08-11)

This experiment is intentionally separate from product code and from the
immutable historical evidence under `docs/evidence/`.

## Objective

Improve YOLO26n INT8 accuracy on the Axelera Metis AIPU without weakening the
frozen semantic gates or disrupting the active translation service that owns
the device.

## Frozen baseline

- Source ONNX SHA-256: `82849fd17775d3584c7d917f73464b43a54bb2001d30ff213efc95047e7c0e7c`
- Current SDK opset-17 source SHA-256: `d502815b0ebd8a97d8ea48ddf23278118555f08d66022abb5a804deb4158d960`.
  On the frozen ten images its raw `(1, 300, 6)` outputs are numerically
  identical to the historical opset-20 graph (maximum absolute output delta
  0.0; 57/57 detections, IoU 1.0, score delta 0.0). The current SDK rejects the
  opset-20 graph at `Split` before calibration, so new candidates use the
  verified-equivalent opset-17 graph.
- Existing compiled kernel SHA-256: `29a4ff3816864925374acbf801cddc9174fcb2d4d6d6ed2e20ccf834716cb8d0`
- Dataset: the frozen ten-image COCO validation subset used by the historical
  Axelera evidence.
- Historical AIPU result: mAP50-95 0.200382, detection coverage 0.403509,
  maximum score delta 0.373610.
- Gates remain: mAP drop <= 0.002, coverage >= 0.98, maximum score delta <=
  0.10, zero class mismatches.

## First hypothesis

The SDK's histogram and hybrid PTQ schemes previously failed because the
default `both_symmetric` MatMul strategy could not satisfy a pinned-zero input
constraint. Changing only the MatMul strategy to `single_asymmetric` may keep
one pinned-zero operand while permitting a better activation calibration
scheme. This is a hypothesis, not a vendor-guaranteed behavior.

## Execution gates

1. Reproduce the existing min-max failure using the vendor quantized-model CPU
   simulator and an accepted, frozen source ONNX profile.
2. Quantize one candidate at a time. First try histogram plus
   `single_asymmetric`; try hybrid plus `single_asymmetric` only after the first
   candidate has a terminal result.
3. A CPU-simulator improvement is only a screening result. It does not become
   an AIPU result until the exact exported artifact is compiled and executed on
   the device.
4. Do not stop or alter the active translation process. AIPU execution remains
   blocked while that process owns `/dev/metis-0:3:0`.

## Terminal screening result

Three min-max compiler-flag ablations (`single_asymmetric` MatMul,
channel-wise depthwise weights, and floating residuals) produced byte-identical
predictions to the baseline. Histogram calibration completed on the accepted
opset-17 source but still failed the SDK 1.7.0 pinned-zero MatMul exporter
constraint. These branches are discarded.

The materially different raw-output graph (`end2end=false`, output
`(1, 84, 8400)`) was then screened with deterministic external class-aware
NMS. It improved the CPU-simulator candidate mAP from `0.295937` to `0.324403`
and coverage from `0.859649` to `0.965517`. It still failed: one class
mismatch, coverage below `0.98`, and maximum score delta `0.448225`. This is a
promising research direction, not an accepted artifact and not an AIPU result.
The 64 candidate detections contain only 19 unique scores (nine are exactly
`0.5`), so the mAP increase may reflect score ties/reordering and added
detections rather than faithful score recovery.

Do not spend an AIPU compile/run window on the current artifact. First repeat
the raw-output result with a distinct calibration seed and a disjoint held-out
set, then perform confidence-head sensitivity analysis. Only a reproducible
candidate that improves the frozen and held-out gates should be compiled and
run after the translation service releases the device.
