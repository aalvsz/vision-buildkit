from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Capabilities:
    available: bool
    precisions: tuple[str, ...] = ("fp32",)
    dynamic_shapes: bool = False
    reason: str | None = None
    compile: bool = False
    execution: bool = True


class Backend(ABC):
    name: str

    @classmethod
    @abstractmethod
    def capabilities(cls) -> Capabilities: ...
    @abstractmethod
    def compile(self, source: str, output: str, **config) -> str: ...
    @abstractmethod
    def run(self, inputs: Any) -> Any: ...
    def benchmark(self, inputs: Any, repeats: int = 20) -> dict[str, float]:
        for _ in range(3):
            self.run(inputs)
        times = []
        for _ in range(repeats):
            t = perf_counter()
            self.run(inputs)
            times.append((perf_counter() - t) * 1000)
        return {
            "samples": len(times),
            "mean_ms": float(np.mean(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
        }


class PyTorchBackend(Backend):
    name = "pytorch"

    @classmethod
    def capabilities(cls):
        try:
            import torch

            return Capabilities(True, ("fp32", "fp16" if torch.cuda.is_available() else "fp32"), True)
        except ImportError:
            return Capabilities(False, reason="Install vision-buildkit[pytorch]")

    def __init__(self, model):
        self.model = model.eval() if hasattr(model, "eval") else model

    def compile(self, source, output, **config):
        raise RuntimeError("PyTorch compile is in-memory; export ONNX for a portable artifact")

    def run(self, inputs):
        try:
            import torch
        except ImportError:
            return self.model(inputs)
        with torch.inference_mode():
            return self.model(inputs)


class OrtBackend(Backend):
    name = "onnxruntime"

    @classmethod
    def capabilities(cls):
        try:
            import onnxruntime as ort

            del ort
            return Capabilities(True, ("fp32",), True)
        except ImportError:
            return Capabilities(False, reason="Install vision-buildkit[onnx]")

    def __init__(self, path, providers=None, *, input_name: str | None = None):
        import onnxruntime as ort

        requested = list(providers or ["CPUExecutionProvider"])
        available = ort.get_available_providers()
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeError(f"requested ORT providers unavailable: {missing}; available={available}")
        self.session = ort.InferenceSession(str(path), providers=requested)
        engaged = self.session.get_providers()
        if engaged != requested:
            raise RuntimeError(f"ORT provider fallback/reordering rejected: requested={requested}, engaged={engaged}")
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("detection runtime requires exactly one image input")
        self.input = inputs[0].name
        if input_name is not None and self.input != input_name:
            raise ValueError(f"runtime input {self.input!r} != manifest input {input_name!r}")
        self.outputs = tuple(item.name for item in self.session.get_outputs())
        self.providers = tuple(engaged)

    def compile(self, source, output, **config):
        return source

    def run(self, inputs):
        values = self.session.run(list(self.outputs), {self.input: inputs})
        return dict(zip(self.outputs, values, strict=True))


class OpenVINOBackend(Backend):
    name = "openvino"

    @classmethod
    def capabilities(cls):
        try:
            import openvino as ov

            del ov
            return Capabilities(True, ("fp32", "fp16", "int8"), True)
        except ImportError:
            return Capabilities(False, reason="Install vision-buildkit[openvino]")

    def __init__(self, path, device="CPU", *, input_name: str | None = None):
        from openvino import Core

        self.core = Core()
        self.compiled = self.core.compile_model(str(path), device)
        self.device = device
        self.input = self.compiled.input(0)
        self.input_name = self.input.get_any_name()
        if input_name is not None and self.input_name != input_name:
            raise ValueError(f"runtime input {self.input_name!r} != manifest input {input_name!r}")
        self.outputs = tuple(output.get_any_name() for output in self.compiled.outputs)

    def compile(self, source, output, **config):
        # Core.compile_model directly accepts ONNX; saving compiled blobs is device specific.
        return source

    def run(self, inputs):
        result = self.compiled({self.input: inputs})
        return {
            name: np.asarray(result[output]) for name, output in zip(self.outputs, self.compiled.outputs, strict=True)
        }


class TensorRTBackend(Backend):
    name = "tensorrt"

    @staticmethod
    def _sanitize_diagnostic(value: str) -> str:
        """Never surface build-host paths in CLI diagnostics."""
        import re

        return re.sub(r"(?:[A-Za-z]:)?/[^\s'\"]+", "<path>", value)[-2000:]

    @classmethod
    def resolve_trtexec(cls, executable: str | None = None) -> str:
        """Resolve a real trtexec binary without silently choosing a wrapper.

        JetPack installs it outside PATH on a number of Jetson images, hence the
        explicit final fallback.  The caller may always pin a binary path.
        """
        requested = executable or os.environ.get("VISION_BUILDKIT_TRTEXEC")
        candidates = [requested] if requested else [shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"]
        for candidate in candidates:
            if candidate:
                path = Path(candidate)
                if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
                    continue
                return str(path)
        raise RuntimeError("trtexec/TensorRT not found; pin trtexec_path or install TensorRT on the target host")

    @classmethod
    def capabilities(cls, executable: str | None = None):
        try:
            cls.resolve_trtexec(executable)
        except RuntimeError:
            exe = None
        else:
            exe = True
        if not exe:
            return Capabilities(
                False, reason="trtexec/TensorRT not found; engines must be built on a matching GPU family"
            )
        return Capabilities(
            True,
            ("fp32", "fp16", "int8"),
            True,
            "Build-only: engine runtime requires Python TensorRT and CUDA bindings",
            True,
            False,
        )

    def __init__(self, engine: str | None = None, *, input_name: str | None = None):
        self.engine = engine
        try:
            import tensorrt as trt  # lazy: builder is not a runtime dependency

            self.trt = trt
        except ImportError:
            self.trt = None
        self.context = None
        self.input_name = input_name
        self.input_names: tuple[str, ...] = ()
        self.output_names: tuple[str, ...] = ()
        self.input_specs: dict[str, tuple[tuple[int, ...], str]] = {}
        self.output_specs: dict[str, tuple[tuple[int, ...], str]] = {}
        if engine is not None:
            self._load_engine(Path(engine))

    def _load_engine(self, engine_path: Path) -> None:
        if self.trt is None:
            raise RuntimeError("TensorRT Python bindings are required to execute an engine")
        if engine_path.is_symlink() or not engine_path.is_file():
            raise ValueError("TensorRT engine must be a regular non-symlink file")
        logger = self.trt.Logger(self.trt.Logger.ERROR)
        runtime = self.trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.runtime = runtime
        self._engine_object = engine
        self.context = context
        if hasattr(engine, "num_io_tensors"):
            names = tuple(engine.get_tensor_name(index) for index in range(engine.num_io_tensors))
            inputs = tuple(name for name in names if engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT)
        else:
            names = tuple(engine.get_binding_name(index) for index in range(engine.num_bindings))
            inputs = tuple(name for index, name in enumerate(names) if engine.binding_is_input(index))
        outputs = tuple(name for name in names if name not in inputs)
        if len(inputs) != 1:
            raise ValueError("detection TensorRT engine must expose exactly one image input")
        if self.input_name is not None and inputs[0] != self.input_name:
            raise ValueError(f"runtime input {inputs[0]!r} != manifest input {self.input_name!r}")
        self.input_name = inputs[0]
        self.input_names = inputs
        self.output_names = outputs
        if hasattr(engine, "num_io_tensors"):
            self.input_specs = {
                name: (
                    tuple(engine.get_tensor_shape(name)),
                    np.dtype(self.trt.nptype(engine.get_tensor_dtype(name))).name,
                )
                for name in inputs
            }
            self.output_specs = {
                name: (
                    tuple(engine.get_tensor_shape(name)),
                    np.dtype(self.trt.nptype(engine.get_tensor_dtype(name))).name,
                )
                for name in outputs
            }
        else:
            self.input_specs = {
                name: (
                    tuple(engine.get_binding_shape(engine.get_binding_index(name))),
                    np.dtype(self.trt.nptype(engine.get_binding_dtype(engine.get_binding_index(name)))).name,
                )
                for name in inputs
            }
            self.output_specs = {
                name: (
                    tuple(engine.get_binding_shape(engine.get_binding_index(name))),
                    np.dtype(self.trt.nptype(engine.get_binding_dtype(engine.get_binding_index(name)))).name,
                )
                for name in outputs
            }

    @staticmethod
    def build_command(
        source: str,
        output: str,
        *,
        executable: str | None = None,
        fp16: bool = False,
        int8: bool = False,
        calibration_cache: str | None = None,
        min_shape: str | None = None,
        opt_shape: str | None = None,
        max_shape: str | None = None,
        fp32_tactics_no_tf32: bool = False,
        skip_inference: bool = True,
    ) -> list[str]:
        if int8 and not calibration_cache:
            raise ValueError("TensorRT INT8 requires calibration_cache; refusing uncalibrated INT8")
        # Keep the legacy CLI surface stable; BuildKit passes a pinned resolved
        # path.  Generic callers retain PATH lookup after capabilities checks.
        cmd = [executable or "trtexec", f"--onnx={source}", f"--saveEngine={output}"]
        if fp16:
            cmd.append("--fp16")
        if fp32_tactics_no_tf32:
            if fp16 or int8:
                raise ValueError("fp32_tactics_no_tf32 cannot be combined with fp16 or int8")
            cmd.append("--noTF32")
        if int8:
            cmd += ["--int8", f"--calib={calibration_cache}"]
        for flag, value in (("--minShapes", min_shape), ("--optShapes", opt_shape), ("--maxShapes", max_shape)):
            if value:
                cmd.append(f"{flag}={value}")
        if skip_inference:
            cmd.append("--skipInference")
        return cmd

    @classmethod
    def compile_file(cls, source, output, **config):
        executable = config.get("trtexec_path")
        if executable:
            executable = cls.resolve_trtexec(executable)
        else:
            cap = cls.capabilities()
            if not cap.available:
                raise RuntimeError(cap.reason)
        cmd = cls.build_command(
            source,
            output,
            executable=executable,
            fp16=config.get("fp16", False),
            int8=config.get("int8", False),
            calibration_cache=config.get("calibration_cache"),
            min_shape=config.get("min_shape"),
            opt_shape=config.get("opt_shape"),
            max_shape=config.get("max_shape"),
            fp32_tactics_no_tf32=config.get("fp32_tactics_no_tf32", False),
        )
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"trtexec failed: {cls._sanitize_diagnostic(result.stderr or '')}")
        if not Path(output).is_file():
            raise RuntimeError("trtexec reported success but engine output was not created")
        return str(output)

    @classmethod
    def build_atomic(cls, source: Path, output: Path, request) -> tuple[Path, list[str], str, dict[str, Any]]:
        """Build into the caller's temporary directory; never leave partial engines."""
        executable = getattr(request, "trtexec_path", None)
        if executable:
            executable = cls.resolve_trtexec(executable)
        else:
            cap = cls.capabilities()
            if not cap.available:
                raise RuntimeError(cap.reason)
        cache = request.calibration_cache_path
        if request.precision == "int8":
            cache_path = Path(cache)
            if cache_path.is_symlink() or not cache_path.is_file():
                raise ValueError("TensorRT calibration cache must be a regular file")
            cache_digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        else:
            cache_digest = None
        profiles = request.profiles
        if request.dynamic and not profiles:
            raise ValueError("TensorRT dynamic build requires at least one shape profile")
        if len(profiles) > 1:
            raise ValueError("trtexec supports one consolidated profile per invocation")
        profile = profiles[0] if profiles else None

        def encode(dims):
            return f"{profile.input_name}:{'x'.join(map(str, dims))}" if profile else None

        command = cls.build_command(
            str(source),
            str(output),
            executable=executable,
            fp16=request.precision == "fp16",
            int8=request.precision == "int8",
            calibration_cache=cache,
            min_shape=encode(profile.min_shape) if profile else None,
            opt_shape=encode(profile.opt_shape) if profile else None,
            max_shape=encode(profile.max_shape) if profile else None,
            fp32_tactics_no_tf32=getattr(request, "fp32_tactics_no_tf32", False),
        )
        output.unlink(missing_ok=True)
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=request.trt_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("trtexec timed out; see sanitized build receipt diagnostics") from exc
        logs = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            output.unlink(missing_ok=True)
            raise RuntimeError(f"trtexec failed (exit {result.returncode}): {cls._sanitize_diagnostic(logs)}")
        version = subprocess.run([command[0], "--version"], text=True, capture_output=True).stdout.strip()
        fingerprint = {"platform": platform.system(), "trtexec": version, "calibration_cache_sha256": cache_digest}
        return (
            output,
            [arg if not arg.startswith("--calib=") else "--calib=<redacted-path>" for arg in command],
            logs,
            fingerprint,
        )

    def compile(self, source, output, **config):
        return self.compile_file(source, output, **config)

    def run(self, inputs):
        if not self.trt or not self.engine or self.context is None:
            raise RuntimeError(
                "TensorRT runtime execution unavailable: install Python TensorRT/CUDA bindings and provide an engine"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TensorRT execution requires torch CUDA tensors for safe buffer ownership") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engine execution requires an available CUDA device")
        tensor = torch.as_tensor(inputs, device="cuda").contiguous()
        stream = torch.cuda.current_stream()
        engine = self._engine_object
        context = self.context
        output_tensors: dict[str, Any] = {}

        if hasattr(engine, "num_io_tensors"):
            if not context.set_input_shape(self.input_name, tuple(tensor.shape)):
                raise ValueError(f"TensorRT rejected input shape {tuple(tensor.shape)}")
            if not context.set_tensor_address(self.input_name, tensor.data_ptr()):
                raise RuntimeError(f"TensorRT rejected input address for {self.input_name!r}")
            for name in self.output_names:
                shape = tuple(context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"TensorRT left output {name!r} shape unresolved: {shape}")
                dtype = torch.from_numpy(np.empty((), dtype=self.trt.nptype(engine.get_tensor_dtype(name)))).dtype
                output = torch.empty(shape, dtype=dtype, device="cuda")
                if not context.set_tensor_address(name, output.data_ptr()):
                    raise RuntimeError(f"TensorRT rejected output address for {name!r}")
                output_tensors[name] = output
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
        else:
            input_index = engine.get_binding_index(self.input_name)
            if not context.set_binding_shape(input_index, tuple(tensor.shape)):
                raise ValueError(f"TensorRT rejected input shape {tuple(tensor.shape)}")
            bindings = [0] * engine.num_bindings
            bindings[input_index] = tensor.data_ptr()
            for name in self.output_names:
                index = engine.get_binding_index(name)
                shape = tuple(context.get_binding_shape(index))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"TensorRT left output {name!r} shape unresolved: {shape}")
                dtype = torch.from_numpy(np.empty((), dtype=self.trt.nptype(engine.get_binding_dtype(index)))).dtype
                output = torch.empty(shape, dtype=dtype, device="cuda")
                bindings[index] = output.data_ptr()
                output_tensors[name] = output
            if not context.execute_async_v2(bindings, stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v2 failed")
        stream.synchronize()
        return {name: value.cpu().numpy() for name, value in output_tensors.items()}


BACKENDS = {b.name: b for b in (PyTorchBackend, OrtBackend, OpenVINOBackend, TensorRTBackend)}
