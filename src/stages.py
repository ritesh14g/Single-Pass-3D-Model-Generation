"""Stage registry — the single source of truth for what has been built.

The build spec divides the system into stages (section headings §4-§8). This
registry records each one, its status, and which manifest stages and budget
entries it owns. Three things read it:

  * ``src.pipeline.run_pipeline`` — runs only stages marked BUILT (or ones
    explicitly requested), so half-built work never contaminates a run;
  * the Stage Lab UI (``ui/``) — one test panel per stage, and a "pipeline so
    far" view that extends automatically to the last BUILT stage;
  * ``DEVLOG.md`` — the status board there must match this file.

When a stage is finished: flip its status here, add its panel under
``ui/stages/``, and log the change in DEVLOG.md.

Note on numbering vs execution order: the spec numbers the occlusion engine
(§6) as Stage 3 and the reconstruction tracks (§7) as Stage 4, but fusion
consumes reconstruction output, so Stage 4 must *execute* before Stage 3.
``execution_order`` captures that; stage numbers follow the spec's headings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StageStatus(str, Enum):
    BUILT = "built"              # implemented, tested, has a Stage Lab panel
    IN_PROGRESS = "in_progress"  # code exists but not validated; not run by default
    PLANNED = "planned"          # nothing implemented yet


@dataclass(frozen=True)
class StageSpec:
    number: int
    key: str
    title: str
    spec_ref: str
    summary: str
    status: StageStatus
    execution_order: int
    manifest_stages: tuple[str, ...]     # names used in RunManifest / src.core.manifest
    budget_keys: tuple[str, ...]         # keys under budget.stages in config

    @property
    def is_built(self) -> bool:
        return self.status is StageStatus.BUILT


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        number=1, key="ingest", title="Ingest", spec_ref="§4",
        summary="Streaming video decode, telemetry parsing (SRT/CSV/EXIF), adaptive overlap-based frame selection with blur gate.",
        status=StageStatus.BUILT, execution_order=1,
        manifest_stages=("ingest",), budget_keys=("ingest",),
    ),
    StageSpec(
        number=2, key="condition", title="Conditioning", spec_ref="§5",
        summary="Blur correction, compression artifacts, exposure/shadows/low light, dynamic objects, GPS noise.",
        status=StageStatus.IN_PROGRESS, execution_order=2,
        manifest_stages=("condition",), budget_keys=("condition",),
    ),
    StageSpec(
        number=3, key="occlusion", title="Occluded surface reconstruction", spec_ref="§6",
        summary="Three-zone classification, anchored monocular fusion via TSDF, honest gap reporting.",
        status=StageStatus.PLANNED, execution_order=4,
        manifest_stages=("fusion",), budget_keys=("fusion",),
    ),
    StageSpec(
        number=4, key="recon", title="Reconstruction tracks", spec_ref="§7",
        summary="Track A (OpenDroneMap), Track B (VGGT-Ω chunked), refinement BA with regression gate.",
        status=StageStatus.PLANNED, execution_order=3,
        manifest_stages=("track_b", "refine_ba", "track_a"), budget_keys=("track_b", "refine_ba", "track_a_mvs"),
    ),
    StageSpec(
        number=5, key="geo_export", title="Georeferencing & export", spec_ref="§8.1-8.3",
        summary="GPS similarity/scale, CRS + geoid, OBJ/PLY/LAS/GeoTIFF/glTF/FBX export.",
        status=StageStatus.PLANNED, execution_order=5,
        manifest_stages=("geo", "export"), budget_keys=("export",),
    ),
    StageSpec(
        number=6, key="viewer_qa", title="Viewer & QA", spec_ref="§8.4-8.5",
        summary="Web viewer with confidence overlay, degradation harness, single-pass benchmark.",
        status=StageStatus.PLANNED, execution_order=6,
        manifest_stages=("qa",), budget_keys=(),
    ),
)


def get_stage(key: str) -> StageSpec:
    for stage in STAGES:
        if stage.key == key:
            return stage
    raise KeyError(f"unknown stage {key!r}; known: {[s.key for s in STAGES]}")


def built_stages() -> list[StageSpec]:
    """Built stages in the order they execute."""
    return sorted((s for s in STAGES if s.is_built), key=lambda s: s.execution_order)


def built_manifest_stages() -> list[str]:
    return [name for stage in built_stages() for name in stage.manifest_stages]


def last_built_stage() -> StageSpec | None:
    stages = built_stages()
    return stages[-1] if stages else None
