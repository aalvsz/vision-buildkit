from __future__ import annotations

import asyncio
import hmac
import logging
import os
import statistics
import tempfile
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from io import BytesIO
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger("vision.server")


def _percentile(values: deque[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


@dataclass
class Metrics:
    """Bounded in-process latency metrics.

    The counters are lifetime totals; latency summaries cover only the most
    recent ``history_size`` observations so a long-running server cannot grow
    memory without bound.
    """

    history_size: int = 4096
    requests: int = 0
    batches: int = 0
    errors: int = 0
    rejected: int = 0
    queue_ms: deque[float] = field(init=False)
    run_ms: deque[float] = field(init=False)
    batch_sizes: deque[int] = field(init=False)

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ValueError("metrics history_size must be positive")
        self.queue_ms = deque(maxlen=self.history_size)
        self.run_ms = deque(maxlen=self.history_size)
        self.batch_sizes = deque(maxlen=self.history_size)

    def view(self) -> dict[str, int | float]:
        mean_batch = statistics.fmean(self.batch_sizes) if self.batch_sizes else 0.0
        return {
            "requests": self.requests,
            "batches": self.batches,
            "errors": self.errors,
            "rejected": self.rejected,
            "observations": len(self.run_ms),
            "mean_batch_size": float(mean_batch),
            "queue_p50_ms": _percentile(self.queue_ms, 50),
            "queue_p95_ms": _percentile(self.queue_ms, 95),
            "run_p50_ms": _percentile(self.run_ms, 50),
            "run_p95_ms": _percentile(self.run_ms, 95),
            "run_p99_ms": _percentile(self.run_ms, 99),
        }


@dataclass
class _Pending:
    image: Image.Image
    future: asyncio.Future
    queued_at: float


class DeadlineBatcher:
    """A bounded deadline batcher with real concurrent batch execution."""

    def __init__(
        self,
        runner: Any,
        window_ms: int = 5,
        max_batch: int = 8,
        concurrency: int = 1,
        max_queue: int = 256,
        metrics_history: int = 4096,
        request_timeout_s: float | None = None,
        api_key: str | None = None,
        rate_limit_per_minute: int | None = None,
    ):
        if window_ms < 0:
            raise ValueError("batch window must be non-negative")
        if max_batch <= 0 or concurrency <= 0 or max_queue <= 0:
            raise ValueError("max_batch, concurrency and max_queue must be positive")
        if max_queue < max_batch:
            raise ValueError("max_queue must be at least max_batch")
        if request_timeout_s is not None and request_timeout_s <= 0:
            raise ValueError("request timeout must be positive when configured")
        if api_key is not None and not api_key:
            raise ValueError("api key must not be empty when configured")
        if rate_limit_per_minute is not None and rate_limit_per_minute <= 0:
            raise ValueError("rate limit must be positive when configured")
        self.runner = runner
        self.window = window_ms / 1000
        self.max_batch = max_batch
        self.max_queue = max_queue
        self.queue: deque[_Pending] = deque()
        self.lock = asyncio.Lock()
        self._timer: asyncio.TimerHandle | None = None
        self._flush_tasks: set[asyncio.Task] = set()
        self._closed = False
        self.metrics = Metrics(metrics_history)
        self.sem = asyncio.Semaphore(concurrency)
        self.request_timeout_s = request_timeout_s
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute

    def _start_flush(self) -> None:
        task = asyncio.create_task(self.flush())
        self._flush_tasks.add(task)
        task.add_done_callback(self._flush_tasks.discard)

    def _schedule_locked(self) -> None:
        if self._timer is None:
            loop = asyncio.get_running_loop()
            self._timer = loop.call_later(self.window, self._start_flush)

    async def submit(self, image: Image.Image) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        queued_at = time.perf_counter()
        async with self.lock:
            if self._closed:
                raise RuntimeError("batcher is closed")
            if len(self.queue) >= self.max_queue:
                self.metrics.rejected += 1
                raise HTTPException(503, "inference queue is full")
            self.queue.append(_Pending(image, future, queued_at))
            if len(self.queue) >= self.max_batch:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                self._start_flush()
            else:
                self._schedule_locked()
        try:
            if self.request_timeout_s is None:
                return await future
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.request_timeout_s)
        except asyncio.TimeoutError as exc:
            future.cancel()
            self.metrics.errors += 1
            raise HTTPException(504, "inference deadline exceeded") from exc
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def flush(self) -> None:
        async with self.lock:
            self._timer = None
            items = [self.queue.popleft() for _ in range(min(self.max_batch, len(self.queue)))]
            if self.queue:
                self._schedule_locked()
        if not items:
            return
        active = [item for item in items if not item.future.cancelled()]
        if not active:
            return
        now = time.perf_counter()
        self.metrics.queue_ms.extend((now - item.queued_at) * 1000 for item in active)
        try:
            async with self.sem:
                started = time.perf_counter()
                images = [item.image for item in active]
                if hasattr(self.runner, "predict_batch"):
                    results = await asyncio.to_thread(self.runner.predict_batch, images)
                else:
                    results = await asyncio.gather(*(asyncio.to_thread(self.runner, image) for image in images))
                if not isinstance(results, (list, tuple)) or len(results) != len(active):
                    size = len(results) if hasattr(results, "__len__") else "unknown"
                    raise RuntimeError(f"runner returned {size} results for {len(active)} inputs")
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.metrics.run_ms.append(elapsed_ms)
                self.metrics.batch_sizes.append(len(active))
            for item, result in zip(active, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)
            self.metrics.requests += len(active)
            self.metrics.batches += 1
        except Exception as exc:
            self.metrics.errors += len(active)
            for item in active:
                if not item.future.done():
                    item.future.set_exception(exc)

    async def close(self) -> None:
        async with self.lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            queued = list(self.queue)
            self.queue.clear()
        for item in queued:
            if not item.future.done():
                item.future.set_exception(RuntimeError("server shutdown"))
        if self._flush_tasks:
            await asyncio.gather(*tuple(self._flush_tasks), return_exceptions=True)


def _jsonable(result: Any) -> Any:
    return result.to_dict() if hasattr(result, "to_dict") else result


def create_app(
    runner: Any,
    *,
    max_upload_bytes: int = 10_000_000,
    max_image_pixels: int = 40_000_000,
    concurrency: int = 1,
    max_batch: int = 8,
    max_queue: int = 256,
    batch_window_ms: int = 5,
    metrics_history: int = 4096,
    request_timeout_s: float | None = None,
    api_key: str | None = None,
    rate_limit_per_minute: int | None = None,
    warmup: bool = True,
    warmup_size: tuple[int, int] = (640, 640),
    fail_on_warmup: bool = True,
) -> FastAPI:
    if max_upload_bytes <= 0 or max_image_pixels <= 0:
        raise ValueError("upload and pixel limits must be positive")
    if any(value <= 0 for value in warmup_size):
        raise ValueError("warmup dimensions must be positive")

    batcher = DeadlineBatcher(
        runner,
        batch_window_ms,
        max_batch,
        concurrency,
        max_queue,
        metrics_history,
        request_timeout_s,
        api_key,
        rate_limit_per_minute,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if warmup:
            try:
                await asyncio.to_thread(runner, Image.new("RGB", warmup_size))
            except Exception:
                if fail_on_warmup:
                    raise
        try:
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await batcher.close()

    app = FastAPI(
        title="Vision BuildKit",
        version="0.2.2",
        description="Research-only object detection service; runtime claims remain evidence-bound.",
        lifespan=lifespan,
    )
    app.state.batcher = batcher
    app.state.ready = False

    @app.middleware("http")
    async def request_id(request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        if batcher.api_key is not None and request.url.path.startswith("/v1/"):
            supplied = request.headers.get("authorization", "")
            expected = f"Bearer {batcher.api_key}"
            if not hmac.compare_digest(supplied, expected):
                response = JSONResponse({"detail": "authentication required"}, status_code=401)
                response.headers["x-request-id"] = request_id
                return response
        if batcher.rate_limit_per_minute is not None and request.url.path.startswith("/v1/"):
            now = time.monotonic()
            window = getattr(app.state, "rate_window", {})
            client = request.client.host if request.client else "unknown"
            start, count = window.get(client, (now, 0))
            if now - start >= 60:
                start, count = now, 0
            if count >= batcher.rate_limit_per_minute:
                response = JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
                response.headers["retry-after"] = str(max(1, int(60 - (now - start))))
                response.headers["x-request-id"] = request_id
                return response
            window[client] = (start, count + 1)
            app.state.rate_window = window
        LOGGER.info("request start id=%s method=%s path=%s", request_id, request.method, request.url.path)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        LOGGER.info("request end id=%s status=%s", request_id, response.status_code)
        return response

    async def read_body(upload: UploadFile) -> bytes:
        raw = await upload.read(max_upload_bytes + 1)
        if len(raw) > max_upload_bytes:
            raise HTTPException(413, "upload exceeds configured limit")
        if not raw:
            raise HTTPException(415, "empty upload")
        return raw

    async def read_image(upload: UploadFile) -> Image.Image:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(415, "image media type required")
        try:
            image = Image.open(BytesIO(await read_body(upload)))
            image.load()
            if image.width * image.height > max_image_pixels:
                raise HTTPException(413, "decoded image exceeds configured pixel limit")
            return image.convert("RGB")
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(415, "valid image required") from exc

    @app.post("/v1/detect")
    async def detect(image: Annotated[UploadFile, File()]) -> JSONResponse:
        result = await batcher.submit(await read_image(image))
        return JSONResponse(_jsonable(result))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness endpoint: the process is responsive."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness endpoint: model warmup completed and queue accepts work."""
        if not app.state.ready:
            return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    @app.post("/v1/video")
    async def video(
        video: Annotated[UploadFile, File()],
        sample_every: int = 1,
        max_frames: int = 300,
        max_duration_s: float = 300.0,
    ) -> dict[str, Any]:
        if video.content_type and not video.content_type.startswith("video/"):
            raise HTTPException(415, "video media type required")
        if sample_every < 1 or max_frames < 1 or max_frames > 10_000 or not 0 < max_duration_s <= 3600:
            raise HTTPException(422, "invalid video limits")
        raw = await read_body(video)
        try:
            import cv2
        except ImportError as exc:
            raise HTTPException(501, "video decoding requires optional opencv-python") from exc

        suffix = os.path.splitext(video.filename or "")[1][:10] or ".video"
        descriptor, path = tempfile.mkstemp(suffix=suffix)
        cap = None
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise HTTPException(415, "video cannot be decoded")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            declared_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0 and declared_frames > 0 and declared_frames / fps > max_duration_s:
                raise HTTPException(413, "video exceeds configured duration limit")
            frames: list[Image.Image] = []
            indices: list[int] = []
            index = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if fps > 0 and index / fps > max_duration_s:
                    raise HTTPException(413, "decoded video exceeds configured duration limit")
                if index % sample_every == 0:
                    height, width = frame.shape[:2]
                    if width * height > max_image_pixels:
                        raise HTTPException(413, "decoded video frame exceeds configured pixel limit")
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    indices.append(index)
                index += 1
            if not frames:
                raise HTTPException(415, "video contains no decodable frames")
            results = await asyncio.gather(*(batcher.submit(frame) for frame in frames))
            return {
                "frames": [
                    {"index": frame_index, "result": _jsonable(result)}
                    for frame_index, result in zip(indices, results, strict=True)
                ],
                "sampled": len(frames),
                "decoded": index,
                "truncated": len(frames) == max_frames,
            }
        finally:
            if cap is not None:
                cap.release()
            with suppress(FileNotFoundError):
                os.unlink(path)

    @app.get("/metrics")
    async def metrics() -> dict[str, int | float]:
        return batcher.metrics.view()

    return app
