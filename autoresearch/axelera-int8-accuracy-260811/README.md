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
