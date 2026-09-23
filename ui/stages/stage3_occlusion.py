"""Stage Lab panel — Stage 3: Occluded surface reconstruction (spec §6).

Contract every stage panel implements (see ui/stages/__init__.py):
    KEY, render_params(cfg) -> overrides, evaluate(run_dir, truth) -> StageEvaluation,
    headline(run_dir) -> dict, render_results(run_dir, truth, evaluation)

Optional (this panel): ``existing_runs()`` + ``rerun(run_dir, cfg)`` — Stage 3 on a run that
already has Track A and georeferencing, so it can be tuned without rebuilding Stage 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.core.config import Config
from src.core.manifest import RunManifest
from src.qa.stage3_eval import FusionOutputs, StageEvaluation, evaluate_fusion

KEY = "occlusion"
ROOT = Path(__file__).resolve().parents[2]
RUN_ROOTS = [ROOT / "data" / "interim", ROOT / "data" / "lab" / "runs", ROOT / "data" / "box" / "runs"]
ZONE_RGB = {0: (240, 240, 240), 1: (31, 136, 61), 2: (191, 135, 0), 3: (207, 34, 46)}
ZONE_NAMES = {1: "Zone 1 · well observed", 2: "Zone 2 · thinly observed", 3: "Zone 3 · never observed (gap)"}


def render_params(cfg: Config) -> dict[str, Any]:
    values: dict[str, Any] = {}
    z = cfg.get_path("fusion.zones")
    with st.sidebar.expander("Zones (§6.1–6.2)", expanded=True):
        values["fusion.zones.zone1_min_views"] = st.slider(
            "Zone 1: minimum confirming views", 2, 10, int(z["zone1_min_views"]))
        values["fusion.zones.zone1_min_triangulation_deg"] = st.slider(
            "Zone 1: minimum triangulation angle (°)", 1.0, 20.0, float(z["zone1_min_triangulation_deg"]), 0.5)
        values["fusion.zones.voxel_gsd_multiple"] = st.slider(
            "Voxel size (x GSD)", 1.0, 6.0, float(z["voxel_gsd_multiple"]), 0.5)
        values["fusion.zones.camera_clearance_fraction"] = st.slider(
            "Reject points nearer a camera than (x flight height)", 0.0, 0.3,
            float(z["camera_clearance_fraction"]), 0.01,
            help="DJI_0047: 26% of the OpenMVS cloud sat 2.4 m ahead of a hovering camera.")
    m = cfg.get_path("fusion.mono_depth")
    with st.sidebar.expander("Zone 2 fill (§6.3)", expanded=True):
        sources = ["auto", "cached", "predictor", "none"]
        values["fusion.mono_depth.source"] = st.radio(
            "Monocular depth", sources, index=sources.index(str(m["source"])), horizontal=True,
            help="auto: Track B's saved VGGT maps, else the predictor. none: gaps reported, not filled.")
        values["fusion.mono_depth.max_frames"] = st.slider("Frames (GPU / cached)", 1, 120, int(m["max_frames"]))
        values["fusion.mono_depth.cpu_max_frames"] = st.slider("Frames on CPU", 0, 10, int(m["cpu_max_frames"]))
        values["fusion.mono_depth.boundary_band_px"] = st.slider("Boundary band (px)", 4, 64, int(m["boundary_band_px"]))
        values["fusion.mono_depth.min_inlier_ratio"] = st.slider("Minimum inlier share", 0.1, 0.9,
                                                                 float(m["min_inlier_ratio"]), 0.05)
        values["fusion.mono_depth.max_residual_rel"] = st.slider(
            "Refuse if fit RMS > (x depth)", 0.005, 0.05, float(m["max_residual_rel"]), 0.005)
    with st.sidebar.expander("Gaps (§6.4)", expanded=False):
        values["fusion.gaps.min_area_m2"] = st.number_input("List gaps from (m²)", 1.0, 1000.0,
                                                            float(cfg.get_path("fusion.gaps.min_area_m2")), 1.0)
        values["fusion.ground.cell_m"] = st.select_slider("Ground cell (m)", [0.5, 1.0, 2.0, 5.0],
                                                          float(cfg.get_path("fusion.ground.cell_m")))
    return {k: v for k, v in values.items() if v != cfg.get_path(k)}


# -- re-run on an existing run ------------------------------------------------------------
def existing_runs() -> list[Path]:
    """Run directories that already have Track A and georeferencing."""
    runs = []
    for root in RUN_ROOTS:
        if root.is_dir():
            runs += [d for d in sorted(root.iterdir()) if (d / "manifest.json").is_file() and (d / "geo" / "georef.json").is_file()]
    return runs


def rerun(run_dir: Path, cfg: Config) -> Path:
    """Stage 3 on ``run_dir`` in place (its ``fusion/`` folder is replaced)."""
    from src.fusion.stage import run_fusion

    manifest = RunManifest.load(Path(run_dir))
    record = manifest.stages["track_a"]
    track_a = {k: manifest.resolve(v) for k, v in record.artifacts.items() if manifest.resolve(v).exists()}
    run_fusion(track_a, Path(run_dir) / "geo" / "georef.json", Path(run_dir) / "fusion", cfg)
    return Path(run_dir)


# -- scorecard + headline ----------------------------------------------------------------------
def _load(run_dir: Path) -> tuple[FusionOutputs, Config]:
    outputs = FusionOutputs.load(run_dir)
    return outputs, Config(outputs.config) if outputs.config else Config({})


def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    outputs, cfg = _load(Path(run_dir))
    return evaluate_fusion(outputs, cfg)


def headline(run_dir: Path) -> dict[str, Any]:
    r = FusionOutputs.load(run_dir).report
    return {"coverage_pct": r.get("coverage_pct"), "zone1_pct": (r.get("ground") or {}).get("zone1_pct"),
            "gaps": (r.get("gaps") or {}).get("count"), "fill_points": (r.get("fill") or {}).get("fill_points")}


# -- diagnostics ---------------------------------------------------------------------------------
def _zone_image(zmap: dict[str, np.ndarray]) -> np.ndarray:
    zone = zmap["zone"]
    step = max(1, max(zone.shape) // 900)
    zone = zone[::step, ::step]
    img = np.zeros(zone.shape + (3,), np.uint8)
    for z, rgb in ZONE_RGB.items():
        img[zone == z] = rgb
    return img


def render_results(run_dir: Path, truth: dict | None, evaluation: StageEvaluation) -> None:
    outputs, _ = _load(Path(run_dir))
    r = outputs.report
    if not r:
        st.info("No Stage 3 output in this run directory.")
        return
    ground, gaps, fill = r.get("ground") or {}, r.get("gaps") or {}, r.get("fill") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage", f"{r.get('coverage_pct')}%", help="Zone 1 + 2 of the ground the cameras saw.")
    c2.metric("Zone 1 (well observed)", f"{ground.get('zone1_pct')}%")
    c3.metric("Gaps (Zone 3)", f"{gaps.get('count', 0)}", f"{gaps.get('total_m2', 0):,.0f} m²", delta_color="off")
    c4.metric("Anchored fill points", f"{fill.get('fill_points', 0):,}",
              f"{fill.get('frames_anchored', 0)}/{fill.get('frames_tried', 0)} frames", delta_color="off")

    zmap = outputs.zone_map()
    left, right = st.columns([3, 2])
    if zmap is not None:
        left.markdown("**Zone map** (ground cells, north up)")
        left.image(_zone_image(zmap), use_container_width=True)
        left.caption("🟩 Zone 1 well observed · 🟧 Zone 2 thinly observed / anchored fill · "
                     "🟥 Zone 3 never observed (listed in gaps.geojson) · ⬜ not in any camera's view")
    shares = pd.DataFrame([{"zone": ZONE_NAMES[z], "percent": float(ground.get(f"zone{z}_pct", 0.0))} for z in (1, 2, 3)])
    right.markdown("**Ground in view, by zone**")
    right.altair_chart(alt.Chart(shares).mark_bar().encode(
        x=alt.X("percent:Q", title="% of visible ground", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("zone:N", title=None, sort=None),
        color=alt.Color("zone:N", legend=None, scale=alt.Scale(domain=list(ZONE_NAMES.values()),
                                                               range=["#1f883d", "#bf8700", "#cf222e"])),
        tooltip=["zone", "percent"]).properties(height=150), use_container_width=True)
    if r.get("rejected_near_camera"):
        right.warning(f"{r['rejected_near_camera']:,} dense points ({r.get('rejected_near_camera_pct')}%) rejected: "
                      "they sit next to the flight path, so they are stereo failures, not surface.")

    voxels = outputs.zones_frame()
    if not voxels.empty:
        sample = voxels.sample(min(len(voxels), 40000), random_state=0)
        sample["zone"] = sample["zone"].map(lambda z: ZONE_NAMES.get(int(z), str(z)))
        st.markdown("**Per voxel: confirming views and triangulation angle** (the §6.2 classification)")
        colour = alt.Color("zone:N", scale=alt.Scale(domain=list(ZONE_NAMES.values()),
                                                     range=["#1f883d", "#bf8700", "#cf222e"]))
        a, b = st.columns(2)
        a.altair_chart(alt.Chart(sample).mark_bar().encode(
            x=alt.X("views:Q", bin=alt.Bin(maxbins=30), title="Confirming views"), y=alt.Y("count():Q", title="Voxels"),
            color=colour).properties(height=200), use_container_width=True)
        b.altair_chart(alt.Chart(sample).mark_bar().encode(
            x=alt.X("tri_deg:Q", bin=alt.Bin(maxbins=40), title="Widest triangulation angle (°)"),
            y=alt.Y("count():Q", title="Voxels"), color=colour).properties(height=200), use_container_width=True)

    frames = outputs.frames_frame()
    st.markdown("**Zone 2 anchoring, per frame** (§6.3: fit on the Zone 1 band, refuse on a bad fit)")
    if fill.get("reason"):
        st.caption(f"Fill: {fill['reason']}")
    if not frames.empty:
        cols = [c for c in ("frame", "status", "reason", "scale", "shift", "residual_m", "residual_limit_m",
                            "inlier_ratio", "holdout_error_m", "holdout_error_pct", "band_px", "region_px",
                            "points_in_targets") if c in frames.columns]
        st.dataframe(frames[cols], hide_index=True, use_container_width=True)
    else:
        st.caption("No frame was chosen for the fill.")

    gap_path = outputs.folder / "gaps.geojson"
    if gap_path.is_file():
        import json

        feats = json.loads(gap_path.read_text(encoding="utf-8")).get("features", [])
        if feats:
            st.markdown("**Largest gaps** (gaps.geojson)")
            st.dataframe(pd.DataFrame([f["properties"] for f in feats[:15]])[
                ["id", "area_m2", "kind", "centroid_map"]], hide_index=True, use_container_width=True)
    for note in r.get("downgrades", []):
        st.warning(note)
    with st.expander("fusion_report.json"):
        st.json({k: v for k, v in r.items() if k != "fill_frames"})
