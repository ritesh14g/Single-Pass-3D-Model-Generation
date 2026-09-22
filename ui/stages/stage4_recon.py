"""Stage Lab panel — Stage 4: Reconstruction tracks (spec §7). Track A only for now.

Contract every stage panel implements (see ui/stages/__init__.py):
    KEY, render_params(cfg) -> overrides, evaluate(run_dir, truth) -> StageEvaluation,
    headline(run_dir) -> dict, render_results(run_dir, truth, evaluation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import streamlit as st

from src.core.config import Config
from src.core.manifest import RunManifest
from src.qa.stage4_eval import TrackAOutputs, StageEvaluation, camera_track, evaluate_track_a, timings_frame

KEY = "recon"


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
def render_params(cfg: Config) -> dict[str, Any]:
    """Widgets for the Track A tunables measured on the box (DEVLOG S4-8)."""
    t = cfg.get_path("recon.track_a")
    values: dict[str, Any] = {}
    with st.sidebar.expander("Sparse (SfM)", expanded=True):
        values["recon.track_a.focal_from_telemetry"] = st.checkbox(
            "Focal from telemetry, held fixed", bool(t["focal_from_telemetry"]),
            help="Off = self-calibrated focal, which drifted to -17% height on Esri.")
        values["recon.track_a.use_dynamic_masks"] = st.checkbox(
            "Use Stage 2 dynamic-object masks", bool(t["use_dynamic_masks"]))
        values["recon.track_a.sift.max_image_size"] = st.select_slider(
            "SIFT image size (px)", [960, 1280, 1600, 2400, 3200], int(t["sift"]["max_image_size"]))
        values["recon.track_a.matching.sequential_overlap"] = st.slider(
            "Sequential overlap (neighbours matched)", 3, 30, int(t["matching"]["sequential_overlap"]))
    with st.sidebar.expander("Mode and Track B (VGGT depth)", expanded=True):
        modes = ["auto", "hybrid", "A"]
        current = str(cfg.get_path("run.mode"))
        values["run.mode"] = st.radio(
            "Mode", modes, index=modes.index(current) if current in modes else 0, horizontal=True,
            help="auto/hybrid: VGGT depth on Track A cameras (~0.1 s/frame, ~37 cm/px on Esri), falling "
                 "back to Track A dense on failure. A: Track A dense only (slow, full detail).")
        b = cfg.get_path("recon.track_b")
        values["recon.track_b.window_frames"] = st.slider(
            "VGGT window (frames)", 4, 16, int(b["window_frames"]),
            help="8 matched COLMAP to 0.7 m on Esri; 32 broke (90 m).")
        values["recon.track_b.drop_low_conf"] = st.slider(
            "Drop least-confident pixels", 0.0, 0.8, float(b["drop_low_conf"]), 0.05)
        values["recon.track_b.consistency.rel_tolerance"] = st.slider(
            "Multi-view agreement tolerance", 0.01, 0.10, float(b["consistency"]["rel_tolerance"]), 0.005)
    with st.sidebar.expander("Dense (MVS)", expanded=True):
        d = t["dense"]
        values["recon.track_a.dense.max_image_size"] = st.select_slider(
            "Dense image size (px)", [640, 960, 1280, 1920], int(d["max_image_size"]),
            help="Sets detail: ~20 / 15 / 10 cm per pixel on Esri at 960 / 1280 / 1920.")
        values["recon.track_a.dense.src_images"] = st.slider(
            "Source views per frame", 4, 20, int(d["src_images"]),
            help="Not a speed lever at full resolution (S4-8): 20 -> 8 saved 9%.")
        values["recon.track_a.dense.iterations"] = st.slider("PatchMatch iterations", 1, 5, int(d["iterations"]))
        values["recon.track_a.dense.geom_consistency"] = st.checkbox(
            "Geometric-consistency pass", bool(d["geom_consistency"]),
            help="The filter that removes wrong depths. Off is faster and noisier.")
        values["recon.track_a.dense.fusion_min_num_pixels"] = st.slider(
            "Fusion: min agreeing views", 2, 8, int(d["fusion_min_num_pixels"]))
    with st.sidebar.expander("Mesh and texture"):
        values["recon.track_a.mesh.mesher"] = st.radio(
            "Mesher", ["openmvs", "poisson"], index=["openmvs", "poisson"].index(str(t["mesh"]["mesher"])),
            help="Poisson shattered clouds on the Linux GPU build (S4-6).")
        values["recon.track_a.texture.enabled"] = st.checkbox("Texture the mesh", bool(t["texture"]["enabled"]))
    return {k: v for k, v in values.items() if v != cfg.get_path(k)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _load(run_dir: Path) -> tuple[TrackAOutputs, Config, Path | None]:
    run_dir = Path(run_dir)
    manifest = RunManifest.load(run_dir)
    record = manifest.stages.get("track_a")
    degradations = list(record.degradations) if record else []
    geo = manifest.artifact("condition", "geo") if manifest.has_artifact("condition", "geo") else None
    return TrackAOutputs.load(run_dir / "track_a", degradations), Config(manifest.config), geo


def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    outputs, cfg, _ = _load(run_dir)
    return evaluate_track_a(outputs, cfg)


def headline(run_dir: Path) -> dict[str, Any]:
    outputs, _, _ = _load(run_dir)
    r = outputs.report
    return {
        "registered": f"{r.get('registered', 0)}/{r.get('frames_in', 0)}",
        "gps_rms_m": r.get("cam_vs_gps_rms_m"),
        "dense_points": (r.get("dense") or {}).get("points"),
        "textured": bool(r.get("textured")),
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_results(run_dir: Path, truth: dict | None, evaluation: StageEvaluation) -> None:
    outputs, _, geo = _load(run_dir)
    r = outputs.report
    if not r:
        st.info("No Track A report in this run directory.")
        return

    st.subheader("Reconstruction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered", f"{r.get('registered')}/{r.get('frames_in')}")
    c2.metric("Camera vs GPS", f"{r['cam_vs_gps_rms_m']} m" if "cam_vs_gps_rms_m" in r else "—")
    c3.metric("Height vs telemetry", f"{r['height_error_pct']:+.1f}%" if "height_error_pct" in r else "—")
    c4.metric("Dense points", f"{(r.get('dense') or {}).get('points', 0):,}")

    track = camera_track(outputs, geo)
    if not track.empty:
        st.markdown("**Camera path: SfM (GPS-aligned) vs GPS**")
        st.caption("Offsets between the two are the camera-vs-GPS residual. A constant lag along the path "
                   "points at telemetry timing, not at the reconstruction (S4-1).")
        chart = (alt.Chart(track).mark_circle(size=36)
                 .encode(x=alt.X("east:Q", title="East (m)"), y=alt.Y("north:Q", title="North (m)"),
                         color=alt.Color("source:N", title=None), tooltip=["frame", "source", "east", "north"])
                 .properties(height=320))
        st.altair_chart(chart, use_container_width=True)

    timings = timings_frame(outputs)
    if not timings.empty:
        st.markdown("**Time per step**")
        chart = (alt.Chart(timings).mark_bar(color="#0969da")
                 .encode(x=alt.X("seconds:Q", title="Seconds"), y=alt.Y("step:N", sort="-x", title=None),
                         tooltip=["step", "seconds"])
                 .properties(height=max(120, 26 * len(timings))))
        st.altair_chart(chart, use_container_width=True)

    for downgrade in r.get("downgrades") or []:
        st.warning(f"Downgrade: {downgrade}")

    stage_dir = Path(run_dir) / "track_a"
    files = [(label, stage_dir / name) for label, name in
             [("Textured mesh (OBJ)", "textured.obj"), ("Mesh (PLY)", "mesh.ply"), ("Dense cloud (PLY)", "dense/fused.ply")]]
    present = [(label, path) for label, path in files if path.exists()]
    if present:
        st.markdown("**Outputs** (open in MeshLab, CloudCompare or Blender)")
        for label, path in present:
            st.code(f"{label}: {path}", language=None)
