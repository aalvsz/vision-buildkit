# Operations boundary

Vision BuildKit's Python API and `vision serve` command are useful for
evaluation and controlled internal experiments. They are not, by themselves, a
hardened production inference service.

## Readiness contract

- `GET /healthz` means that the process is responsive.
- `GET /readyz` means that the configured model completed warmup and the bounded
  inference queue is accepting work.
- `GET /metrics` exposes bounded in-process counters and latency observations;
  it is not a durable monitoring system.
- Every response includes an `x-request-id` header. Clients may provide one for
  log correlation; otherwise the service generates one.
- Set `request_timeout_s` (or `vision serve --request-timeout-s`) to bound work
  on untrusted or overloaded requests; timed-out requests return HTTP 504.
- Set `VISION_API_KEY` or pass `api_key` to require `Authorization: Bearer ...`
  on `/v1/*`; health and readiness remain probeable without the token.
- Set `rate_limit_per_minute` to enable a bounded per-client in-process limit.
  For multi-worker or distributed deployments, enforce the authoritative limit
  at the reverse proxy instead.

Put an authenticated, TLS-terminating reverse proxy in front of the service.
Keep upload, pixel, queue, batch, and video-duration limits enabled, and run
optional decoders and model runtimes with process/container isolation.

## Release readiness checklist

Before describing a backend as supported, verify all of the following:

1. The combination exists in `docs/support-matrix.yml`.
2. The evidence names the exact model, artifact, runtime, device, dataset, and
   harness.
3. The paired validation gates pass on the declared scope.
4. A reproducible capsule or canonical experiment report verifies successfully.
5. The limitations in the evidence are reproduced in user-facing documentation.

An installed optional dependency, successful import, engine build, or smoke test
does not satisfy this checklist.
