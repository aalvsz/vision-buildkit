# Security Policy

Vision BuildKit is research and evaluation software, not a hardened production
inference service. Do not expose `vision serve` directly to an untrusted network
without an authenticated reverse proxy, TLS termination, rate limiting, and
resource isolation.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting for this repository:

1. Open the repository's **Security** tab.
2. Select **Report a vulnerability**.
3. Include affected versions, impact, reproduction steps, and any suggested
   mitigation.

You should receive an initial acknowledgement within seven days. No fixed
remediation timeline is promised; timing depends on severity and the scope of
the affected runtime.

## Supported versions

Until the project reaches a stable release, security fixes apply to the latest
commit on `main`. Older commits and generated model artifacts are not
independently supported.

## Model artifacts

Model checkpoints, ONNX graphs, TensorRT engines, OpenVINO artifacts, manifests,
and Vision Capsules should be treated as untrusted input unless their expected
content digest comes from a trusted source.

## Deployment boundaries

- Treat uploaded images and videos as untrusted input.
- Keep upload, pixel, queue, batch, and duration limits enabled.
- Run optional decoders and model runtimes in a restricted process or container.
- Keep capsules, calibration data, model weights, and runtime logs out of public
  issue reports unless their licenses and contents permit disclosure.
