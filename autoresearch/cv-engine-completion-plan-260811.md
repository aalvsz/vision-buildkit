# CV Engine completion plan (2026-08-11)

## Verdict

The V1 core is substantially implemented: contracts and geometry, source
locking, detection parity, ORT support, YOLO26 TensorRT/OpenVINO FP32/FP16,
capsules, evidence binding, a deterministic optimization report, and an E2E
benchmark harness are present. The project is not yet at the original V1 exit
condition because several claims remain limited to ten images, the optimized
DETR path and production distribution/security pieces are unfinished, and the
`<5%` wrapper-overhead criterion has not been measured.

INT8 is valuable but was a P2 exploratory item in the original plan. MPS INT8
was not an original V1 requirement; the Apple INT8 work here is OpenVINO on
CPU. Axelera was also an additional target. QAT/training was explicitly outside
the MVP, so adopting QAT is a deliberate scope expansion.

## Priority 0: close the quality evidence

1. **OpenVINO INT8, lowest-cost remaining PTQ test.** Quantize the raw-output
   `(1,84,8400)` ONNX rather than the end-to-end graph. Start with NNCF `MIXED`,
   ignore confidence-path Sigmoid/Softmax operations, set
   `fast_bias_correction=False`, and save with `compress_to_fp16=False`.
   Evaluate with the exact external decoder and unchanged gates. Repeat on a
   held-out set. If max score delta remains above `0.12`, stop PTQ and move to
   explicit Q/DQ QAT.
2. **TensorRT INT8.** The 20-config PTQ sweep is exhausted. Retain the raw graph
   and external NMS, but train/export an explicit Q/DQ model. Keep the sensitive
   confidence head out of INT8 initially, then reduce the high-precision scope
   only if gates remain satisfied. Build as a strongly typed TensorRT network
   and verify coverage `>=0.98` and max score delta `<=0.12` before benchmarking.
3. **Axelera INT8.** Keep the raw-output research branch and retain the
   compiler-matched head-QAT negative result. The candidate improved CPU-screen
   mAP (`0.296` to `0.324`) but still failed coverage, class parity, and score
   drift. A direct public-schema audit of Voyager SDK 1.7 confirms that the
   graph cleaner can split only on MatMul, Gemm, or Clip; it cannot exclude the
   six terminal Detect Convs or redefine their outputs. Do not fund another
   blind QAT sweep or an AIPU window under SDK 1.7. Re-open this path only when
   a vendor-supported terminal-head exclusion feature or a newer SDK is
   available, then require a frozen and held-out CPU screen before device use.
4. **Evidence promotion.** Any passing ten-image result must then run full COCO
   val2017, bind the exact artifact/runtime/harness, and receive independent
   review. Ten-image results remain parity screens, not accuracy claims.

## Priority 1: finish the original product gates

1. Add one real optimized backend for RF-DETR or RT-DETR (OpenVINO first on
   Apple/x86, TensorRT second if NVIDIA deployment is required), then run the
   same parity, accuracy, and benchmark evidence flow used for YOLO26.
2. Expose the optimizer's already-recorded acceptance/rejection reasons through
   a user-facing `vision optimize --explain` or equivalent report command.
3. Measure the Vision layer itself around a no-op/direct-runtime control and
   enforce the original `<5%` overhead criterion in CI on fixed hardware.
4. Run full end-to-end benchmarks for the verified CPU, Jetson, and eventual
   Axelera artifacts. Report latency distributions and throughput bound to the
   exact hardware; do not infer speedups from static or simulator results.
5. Reproduce builds in a clean CI environment and promote evidence only after
   artifact digests and tolerances match.

## Priority 2: enterprise and distribution work

1. Generate a CycloneDX/SPDX SBOM for each capsule and bind its digest into the
   capsule evidence.
2. Add Sigstore/cosign signing and verification. Current checksums prove
   internal integrity but not publisher authenticity.
3. Implement a local registry first, then OCI push/pull with digest pinning and
   tamper tests.
4. Add a Triton model-repository export for supported ONNX/TensorRT artifacts,
   including preprocessing/postprocessing and configuration provenance.

## External exit criteria

The original design-partner, paid-pilot, adapter-by-independent-engineer, and
unit-economics criteria require product/business work outside this repository.
Track them separately; code tests cannot satisfy them.

## Recommended execution order

Run the raw OpenVINO PTQ control first because it is cheap and still genuinely
untested. In parallel planning, define one shared QAT/export contract for
TensorRT, OpenVINO, and Axelera, but begin training only if the project accepts
the V1 scope expansion. Then close full-COCO/performance evidence, optimized
DETR support, explainability, and finally SBOM/signing/OCI/Triton.
