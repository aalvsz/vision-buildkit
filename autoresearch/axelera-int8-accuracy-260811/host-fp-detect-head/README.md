# Host-FP YOLO26n detection-head prototype

This directory contains an offline, structural ONNX rewrite prototype only.
It does not invoke the Axelera SDK, quantization, an AIPU, or a remote host.

`rewrite_host_fp_detect_head.py` accepts only the SHA-256-pinned local source
`yolo26n-raw.onnx` (`db42840c9ae5f1490c3d748f0bc72253d1509ed2c0f5d20b3f3847102596e9a9`).
It replaces these exact six terminal 1x1 FP32 Conv nodes with six model-local
functions in `custom_host_onnx_functions`:

- `/model.23/cv2.0/cv2.0.2/Conv`
- `/model.23/cv3.0/cv3.0.2/Conv`
- `/model.23/cv2.1/cv2.1.2/Conv`
- `/model.23/cv3.1/cv3.1.2/Conv`
- `/model.23/cv2.2/cv2.2.2/Conv`
- `/model.23/cv3.2/cv3.2.2/Conv`

Every function receives the original activation, weight initializer, and bias
initializer and has an equivalent standard-domain FP32 1x1 `Conv` body. Its
body copies and verifies all source attributes exactly: `dilations=[1,1]`,
`group=1`, `kernel_shape=[1,1]`, `pads=[0,0,0,0]`, and `strides=[1,1]`. The
script refuses unexpected node attributes or tensor identities, preserves the
single `[1,84,8400]` source-output signature, requires an ONNX checker pass,
and rejects any introduced Q/DQ node. It serializes deterministically and
writes `rewrite-proof.json` beside the generated model.

## Voyager SDK 1.7.0 preflight result

The reviewed graph-cleaning-only preflight on `axe-ubuntu-125` failed closed.
Voyager correctly split the six pre-Conv tensors out of `aipu_0` and retained
the six custom-domain calls in `host_0`, but it stripped all six model-local
`FunctionProto` definitions. The resulting host graph cannot be instantiated,
so this route must not proceed to quantization or AIPU compilation without an
SDK-supported fix. See `graph-cleaner-preflight.json` for the exact evidence.

The supported fallback for a future experiment phase is the lower-confidence
one-variable probe `smoothquant_coefficient = 0.3`, subject to its own reviewed
manifest, artifact-change gate, and frozen simulator prediction-change gate.

Run locally:

```bash
python3 autoresearch/axelera-int8-accuracy-260811/host-fp-detect-head/rewrite_host_fp_detect_head.py
```

ONNX Runtime parity is attempted only when `onnxruntime` is importable in the
same Python process. If it cannot execute model-local functions, the proof
records the exact blocker; structural proof remains valid but is not runtime
evidence. The output is a prototype for a future host-float partitioning
investigation, not an Axelera-supported deployment artifact or an accuracy
claim.
