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
Stage 3 also works in metres, so it needs Stage 5's ``geo`` (it runs between
``geo`` and ``export``): ``needs`` lists manifest stages a stage requires
beyond those that execute before it.
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
    needs: tuple[str, ...] = ()          # manifest stages owned by later stages that must run first

    @property
    def is_built(self) -> bool:
        return self.status is StageStatus.BUILT


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        number=0, key="input_check", title="Input check", spec_ref="§1.3, §4",
        summary="Accepts any drone video container/codec and any telemetry format; checks resolution, decoding, "
                "GPS presence and sync, physical consistency and feasibility before the budget is spent.",
        status=StageStatus.BUILT, execution_order=0,
        manifest_stages=("preflight",), budget_keys=("preflight",),
    ),
    StageSpec(
        number=1, key="ingest", title="Ingest", spec_ref="§4",
        summary="Streaming video decode, telemetry parsing (SRT/CSV/EXIF), adaptive overlap-based frame selection with blur gate.",
        status=StageStatus.BUILT, execution_order=1,
        manifest_stages=("ingest",), budget_keys=("ingest",),
    ),
    StageSpec(
        number=2, key="condition", title="Conditioning", spec_ref="§5",
        summary="Blur correction, compression artifacts, exposure/shadows/low light, dynamic objects, GPS noise.",
        status=StageStatus.BUILT, execution_order=2,
        manifest_stages=("condition",), budget_keys=("condition",),
    ),
    StageSpec(
        number=3, key="occlusion", title="Occluded surface reconstruction", spec_ref="§6",
        summary="Three-zone voxel classification (views, triangulation angle, visibility), Zone 2 filled from "
                "monocular depth anchored to Zone 1 (RANSAC scale/shift, residual gate, never overriding Zone 1), "
                "mesh faces without support flagged as inferred, gaps.geojson + coverage percent.",
        status=StageStatus.BUILT, execution_order=4,
        manifest_stages=("fusion",), budget_keys=("fusion",), needs=("geo",),
    ),
    StageSpec(
        number=4, key="recon", title="Reconstruction tracks", spec_ref="§7",
        summary="Track A (pycolmap SfM + dense on GPU, OpenMVS mesh + texture) and Track B as the "
                "§7.4 hybrid (VGGT depth on Track A cameras). Refinement BA with regression gate: planned.",
        status=StageStatus.BUILT, execution_order=3,
        manifest_stages=("track_b", "refine_ba", "track_a"), budget_keys=("track_b", "refine_ba", "track_a_mvs"),
    ),
    StageSpec(
        number=5, key="geo_export", title="Georeferencing & export", spec_ref="§8.1-8.3",
        summary="RANSAC GPS similarity (straight-path safe), UTM + EGM96 orthometric heights, "
                "OBJ/PLY/LAS/GeoTIFF/glb/FBX export with confidence, coverage percent, metadata sidecar.",
        status=StageStatus.BUILT, execution_order=5,
        manifest_stages=("geo", "export"), budget_keys=("geo", "export"),
    ),
    StageSpec(
        number=6, key="viewer_qa", title="Viewer & QA", spec_ref="§8.4-8.5",
        summary="Web viewer (three.js: photo / confidence / zone overlays, Zone 3 gap outlines, point-to-point "
                "measurement with zone warnings, run stats) and the QA report (every stage's scorecard, time vs "
                "budget, accuracy per zone against a reference surface, degradation and single-pass benchmarks).",
        status=StageStatus.BUILT, execution_order=6,
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


def manifest_stages_through(stage: StageSpec) -> list[str]:
    """Manifest stages needed to produce ``stage``'s output, in execution order.

    A stage's ``manifest_stages`` is what it *owns*, not what it *needs*: Stage 2
    owns ``condition`` alone, but conditioning reads the frames Stage 1 selected,
    so running ``condition`` into an empty directory raises "conditioning needs a
    completed ingest stage". The Stage Lab therefore runs every BUILT stage that
    executes at or before the requested one, then the stage itself.

    The stage is included whatever its status, so a panel for an IN_PROGRESS
    stage can still be exercised before it is flipped to BUILT.
    """
    upstream = [s for s in built_stages() if s.execution_order < stage.execution_order]
    chain = upstream + [stage] if stage not in upstream else upstream
    names = [name for s in chain for name in s.manifest_stages]
    return names[:-len(stage.manifest_stages)] + [n for n in stage.needs if n not in names] + names[-len(stage.manifest_stages):]
