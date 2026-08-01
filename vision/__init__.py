"""Detection portability API; optional runtime imports occur only on use."""

from .api import compare_detections, load, optimize, validate, validate_dataset
from .buildkit import BuildPlanner
from .capsule import CapsuleArtifact, CapsuleMode, select_runtime
from .conformance import CertificationReport, ConformanceEvidenceBundle
from .contract import VisionContract
from .manifest import VisionManifest
from .optimization import optimize_search
from .plugins import (
    MaterializedOnnx,
    PluginRegistry,
    SourceAdapter,
    TargetPlugin,
    TensorRTTargetPlugin,
    make_tensorrt_target,
)
from .providers import LibreYoloSourceAdapter, UltralyticsSourceAdapter
from .types import Box, DetectionResult

__all__ = [
    "load",
    "validate",
    "validate_dataset",
    "compare_detections",
    "optimize",
    "optimize_search",
    "DetectionResult",
    "Box",
    "VisionManifest",
    "VisionContract",
    "BuildPlanner",
    "PluginRegistry",
    "SourceAdapter",
    "MaterializedOnnx",
    "TargetPlugin",
    "TensorRTTargetPlugin",
    "make_tensorrt_target",
    "UltralyticsSourceAdapter",
    "LibreYoloSourceAdapter",
    "CapsuleArtifact",
    "CapsuleMode",
    "select_runtime",
    "CertificationReport",
    "ConformanceEvidenceBundle",
]
