import asyncio
import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from vision.cli import app
from vision.server import DeadlineBatcher, Metrics, create_app
from vision.types import DetectionResult


class FakeRunner:
    def __call__(self, image):
        return DetectionResult(image_size=image.size)

    def predict_batch(self, images):
        return [self(image) for image in images]


def png(size=(4, 4)):
    buffer = io.BytesIO()
    Image.new("RGB", size).save(buffer, "PNG")
    return buffer.getvalue()


def test_cli_help():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "convert" in result.stdout


def test_server_image_and_bounded_metrics():
    with TestClient(create_app(FakeRunner(), warmup=False, batch_window_ms=0, metrics_history=1)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}
        for _ in range(2):
            response = client.post("/v1/detect", files={"image": ("a.png", png(), "image/png")})
            assert response.status_code == 200 and response.json()["boxes"] == []
            assert response.headers["x-request-id"]
        forwarded = client.get("/healthz", headers={"x-request-id": "trace-123"})
        assert forwarded.headers["x-request-id"] == "trace-123"
        metrics = client.get("/metrics").json()
        assert metrics["requests"] == 2
        assert metrics["observations"] == 1


def test_server_rejects_media_byte_and_pixel_limits():
    with TestClient(create_app(FakeRunner(), warmup=False, max_upload_bytes=32)) as client:
        assert client.post("/v1/detect", files={"image": ("x.txt", b"x", "text/plain")}).status_code == 415
        assert client.post("/v1/detect", files={"image": ("x.png", png(), "image/png")}).status_code == 413
        assert client.post("/v1/video", files={"video": ("x.mp4", b"x", "image/png")}).status_code == 415

    with TestClient(create_app(FakeRunner(), warmup=False, max_image_pixels=8)) as client:
        assert client.post("/v1/detect", files={"image": ("x.png", png(), "image/png")}).status_code == 413


def test_invalid_server_configuration_fails_closed():
    with pytest.raises(ValueError):
        create_app(FakeRunner(), concurrency=0)
    with pytest.raises(ValueError):
        create_app(FakeRunner(), max_queue=1, max_batch=2)
    with pytest.raises(ValueError):
        Metrics(history_size=0)
    with pytest.raises(ValueError):
        create_app(FakeRunner(), request_timeout_s=0)
    with pytest.raises(ValueError):
        create_app(FakeRunner(), api_key="")
    with pytest.raises(ValueError):
        create_app(FakeRunner(), rate_limit_per_minute=0)


def test_optional_auth_and_rate_limit_are_fail_closed():
    with TestClient(create_app(FakeRunner(), warmup=False, api_key="secret", rate_limit_per_minute=1)) as client:
        assert client.get("/healthz").status_code == 200
        assert client.post("/v1/detect", files={"image": ("a.png", png(), "image/png")}).status_code == 401
        headers = {"authorization": "Bearer secret"}
        assert (
            client.post("/v1/detect", headers=headers, files={"image": ("a.png", png(), "image/png")}).status_code
            == 200
        )
        assert (
            client.post("/v1/detect", headers=headers, files={"image": ("a.png", png(), "image/png")}).status_code
            == 429
        )


def test_queue_backpressure_and_result_cardinality():
    entered = threading.Event()
    release = threading.Event()

    class BlockingRunner(FakeRunner):
        def predict_batch(self, images):
            entered.set()
            release.wait(timeout=2)
            return super().predict_batch(images)

    async def exercise():
        batcher = DeadlineBatcher(BlockingRunner(), window_ms=0, max_batch=1, max_queue=1)
        first = asyncio.create_task(batcher.submit(Image.new("RGB", (2, 2))))
        assert await asyncio.to_thread(entered.wait, 1)
        second = asyncio.create_task(batcher.submit(Image.new("RGB", (2, 2))))
        await asyncio.sleep(0)
        with pytest.raises(Exception) as rejected:
            await batcher.submit(Image.new("RGB", (2, 2)))
        assert getattr(rejected.value, "status_code", None) == 503
        release.set()
        await asyncio.gather(first, second)
        await batcher.close()

        class WrongCardinality(FakeRunner):
            def predict_batch(self, images):
                return []

        bad = DeadlineBatcher(WrongCardinality(), window_ms=0, max_batch=1, max_queue=1)
        with pytest.raises(RuntimeError, match="0 results for 1 inputs"):
            await bad.submit(Image.new("RGB", (2, 2)))
        await bad.close()

    asyncio.run(exercise())


def test_configured_concurrency_runs_batches_in_parallel():
    lock = threading.Lock()
    active = 0
    maximum = 0

    class ConcurrentRunner(FakeRunner):
        def predict_batch(self, images):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return super().predict_batch(images)

    async def exercise():
        batcher = DeadlineBatcher(ConcurrentRunner(), window_ms=0, max_batch=1, concurrency=2, max_queue=4)
        await asyncio.gather(
            batcher.submit(Image.new("RGB", (2, 2))),
            batcher.submit(Image.new("RGB", (2, 2))),
        )
        assert maximum == 2
        await batcher.close()

    asyncio.run(exercise())
