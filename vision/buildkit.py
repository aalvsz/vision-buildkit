"""Deterministic capability negotiation and candidate planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product

from .contract import VisionContract
from .plugins import Capabilities, MaterializedOnnx, PreparedArtifact, PreparedSource, SourceAdapter, TargetPlugin


@dataclass(frozen=True)
class BuildCandidate:
    candidate_id: str
    precision: str
    resolution: tuple[int, int]
    batch: int
    dynamic: bool
    nms_owner: str
    pareto: bool = False
    selected: bool = False


@dataclass(frozen=True)
class Plan:
    candidates: tuple[BuildCandidate, ...]
    rejections: tuple[str, ...]


class BuildPlanner:
    def materialize(
        self,
        source: SourceAdapter,
        target: TargetPlugin,
        checkpoint: str,
        contract: VisionContract,
        output: str,
    ) -> tuple[PreparedSource, MaterializedOnnx, PreparedArtifact]:
        """Provider-neutral source export followed by target preparation.

        This is intentionally the only orchestration bridge: target plugins see
        an ONNX path and source lineage, not a provider id or provider runtime.
        """
        captured = source.prepare(checkpoint, contract)
        materialized = source.materialize_onnx(captured, contract, output)
        artifact = target.prepare_materialized(materialized, captured, source, contract)
        return captured, materialized, artifact

    def plan(
        self,
        source: SourceAdapter,
        target: TargetPlugin,
        contract: VisionContract,
        source_digest: str,
        *,
        precisions=("fp32",),
        resolutions=((640, 640),),
        batches=(1,),
        dynamics=(False,),
        nms_owners=("source",),
    ) -> Plan:
        if source_digest != contract.provenance.source_sha256:
            raise ValueError("source digest must match contract provenance")
        if set(nms_owners) != {contract.result.nms_owner}:
            raise ValueError("build candidates must use the contract nms_owner")
        source_caps, target_caps = source.capabilities, target.capabilities
        candidates, rejects = [], []
        for precision, resolution, batch, dynamic, nms in product(
            sorted(precisions), sorted(resolutions), sorted(batches), sorted(dynamics), sorted(nms_owners)
        ):
            if resolution[0] <= 0 or resolution[1] <= 0 or batch <= 0:
                raise ValueError("resolution dimensions and batch must be positive")
            required = {precision, "dynamic" if dynamic else "static", f"nms:{nms}"}
            if isinstance(source_caps, Capabilities) and isinstance(target_caps, Capabilities):
                missing = sorted(
                    ({precision} - target_caps.precisions)
                    | ({"dynamic" if dynamic else "static"} - target_caps.modes)
                    | ({nms} - target_caps.nms_owners)
                    | ({precision} - source_caps.precisions if source_caps.precisions else set())
                )
            else:
                missing = sorted(required - (source_caps & target_caps))
            if missing:
                rejects.append(
                    f"precision={precision},resolution={resolution},batch={batch},"
                    f"dynamic={dynamic},nms={nms}: missing {','.join(missing)}"
                )
                continue
            body = {
                "source": source.plugin_id,
                "source_version": source.plugin_version,
                "target": target.plugin_id,
                "target_version": target.plugin_version,
                "precision": precision,
                "resolution": resolution,
                "batch": batch,
                "dynamic": dynamic,
                "nms": nms,
                "contract_digest": contract.digest,
                "source_digest": source_digest,
            }
            candidates.append(
                BuildCandidate(
                    hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    precision,
                    resolution,
                    batch,
                    dynamic,
                    nms,
                )
            )
        if not candidates:
            raise ValueError("no supported build combinations: " + "; ".join(rejects))
        return Plan(tuple(sorted(candidates, key=lambda x: x.candidate_id)), tuple(sorted(rejects)))
