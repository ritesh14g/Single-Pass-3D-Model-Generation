"""Stage Lab panel — Stage 0: Input check (spec §1.3 mandatory inputs, §4 ingest).

Contract every stage panel implements (see ui/stages/__init__.py):
    KEY, render_params(cfg) -> overrides, evaluate(run_dir, truth) -> StageEvaluation,
    headline(run_dir) -> dict, render_results(run_dir, truth, evaluation)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.core.config import Config
from src.qa.stage0_eval import StageEvaluation, checks_frame, evaluate_input, load_report

KEY = "input_check"
BADGE = {"READY": ("✅", "#1f883d"), "READY_WITH_WARNINGS": ("⚠️", "#bf8700"), "BLOCKED": ("⛔", "#cf222e")}


def render_params(cfg: Config) -> dict[str, Any]:
    values: dict[str, Any] = {}
    p = cfg.get_path("preflight")
    with st.sidebar.expander("Input check (§1.3)", expanded=True):
        values["preflight.enabled"] = st.checkbox("Check the input before running", bool(p["enabled"]))
        values["preflight.block_on_fail"] = st.checkbox(
            "Refuse a blocking input", bool(p["block_on_fail"]),
            help="Off: problems are reported and the run continues anyway (same as --accept-input).")
        values["preflight.apply_time_offset"] = st.checkbox(
            "Correct a measured telemetry time offset", bool(p["apply_time_offset"]),
            help="Shifts telemetry onto the video clock when image motion and GPS speed agree at an offset.")
        values["preflight.telemetry.require_gps"] = st.checkbox(
            "GPS is mandatory", bool(p["telemetry"]["require_gps"]),
            help="Off: a clip without GPS is accepted and reconstructed scale-free.")
    with st.sidebar.expander("Thresholds", expanded=False):
        values["preflight.video.min_short_side_px"] = st.select_slider(
            "Minimum resolution (short side)", [480, 720, 1080, 1440, 2160], int(p["video"]["min_short_side_px"]))
        values["preflight.sampling.max_samples"] = st.slider("Frames sampled", 40, 400,
                                                             int(p["sampling"]["max_samples"]), 20)
        values["preflight.sync.max_search_s"] = st.slider("Time-offset search (± s)", 2.0, 60.0,
                                                          float(p["sync"]["max_search_s"]), 1.0)
        values["preflight.authenticity.rigid_min"] = st.slider(
            "Minimum rigid-scene share", 0.0, 0.9, float(p["authenticity"]["rigid_min"]), 0.05,
            help="Below this, the picture does not behave like one rigid 3-D scene: water, crowds, heavy warping "
                 "or synthetic footage.")
    return {k: v for k, v in values.items() if v != cfg.get_path(k)}


def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    return evaluate_input(load_report(run_dir))


def headline(run_dir: Path) -> dict[str, Any]:
    report = load_report(run_dir)
    if report is None:
        return {}
    counts = {s: sum(1 for c in report.checks if c.status == s) for s in ("pass", "warn", "block")}
    return {"verdict": report.verdict.replace("_", " ").lower(), "blocking": counts["block"],
            "warnings": counts["warn"], "sync_lag_s": (report.sync or {}).get("lag_s")}


def render_results(run_dir: Path, truth: dict | None, evaluation: StageEvaluation) -> None:
    report = load_report(run_dir)
    if report is None:
        st.info("No input check in this run directory. Run the pipeline, or use "
                "`python -m src.cli check <video>`.")
        return
    icon, colour = BADGE.get(report.verdict, ("", "#57606a"))
    counts = {s: sum(1 for c in report.checks if c.status == s) for s in ("pass", "warn", "block", "info")}
    st.markdown(f"<h3 style='color:{colour}'>{icon} {report.verdict.replace('_', ' ').title()}</h3>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Blocking", counts["block"])
    c2.metric("Warnings", counts["warn"])
    c3.metric("Passed", counts["pass"])
    c4.metric("Checked in", f"{(report.timing_s or {}).get('total', 0):.0f} s")

    problems = [c for c in report.checks if c.status in ("block", "warn")]
    if problems:
        st.markdown("**What needs attention**")
        for c in problems:
            (st.error if c.status == "block" else st.warning)(f"**{c.label}** — {c.detail}"
                                                              + (f"\n\n*Fix:* {c.fix}" if c.fix else ""))
    else:
        st.success("Every check passed. " + report.note)

    with st.expander("All checks", expanded=not problems):
        st.dataframe(checks_frame(report), use_container_width=True, hide_index=True)

    tel = report.telemetry or {}
    cand = pd.DataFrame(tel.get("candidates", []))
    if not cand.empty:
        st.markdown("**Telemetry sources found** (content-sniffed, not by file extension)")
        st.dataframe(cand, use_container_width=True, hide_index=True)
    cam = report.camera or {}
    if cam.get("model") or cam.get("hfov_deg"):
        st.caption(f"Camera: {cam.get('model') or 'from telemetry'} — horizontal FOV "
                   f"{cam.get('hfov_deg')}° ({cam.get('source')})")

    series = report.series
    if not series.empty and "gps_speed" in series:
        st.markdown("**Video motion vs GPS speed** (the sync check)")
        sync = report.sync or {}
        st.caption(f"Best match r={sync.get('r_best')} at {sync.get('lag_s')} s "
                   f"(r={sync.get('r_at_zero')} with no shift). Both curves are scaled to their own maximum.")
        frame = series[["t", "shift_px_s", "gps_speed"]].copy()
        for col in ("shift_px_s", "gps_speed"):
            top = frame[col].abs().max()
            frame[col] = frame[col] / top if top else frame[col]
        melted = frame.melt("t", var_name="signal", value_name="normalised").replace(
            {"shift_px_s": "image motion", "gps_speed": "GPS speed"})
        st.altair_chart(alt.Chart(melted.dropna()).mark_line().encode(
            x=alt.X("t:Q", title="Video time (s)"), y=alt.Y("normalised:Q", title="Relative"),
            color=alt.Color("signal:N", title=None)).properties(height=200), use_container_width=True)

    if not series.empty:
        cols = [c for c in ("sharpness", "keypoints", "rigid_ratio", "sky_frac", "brightness") if c in series]
        st.markdown("**Sampled frames across the clip**")
        melted = series[["t", *cols]].melt("t", var_name="measure", value_name="value")
        st.altair_chart(alt.Chart(melted.dropna()).mark_line().encode(
            x=alt.X("t:Q", title="Video time (s)"), y=alt.Y("value:Q", title=None),
            color=alt.Color("measure:N", title=None),
            facet=alt.Facet("measure:N", columns=2, title=None)).resolve_scale(y="independent")
            .properties(height=110, width=300), use_container_width=False)

    gps = report.gps_track
    if not gps.empty:
        st.markdown("**GPS track during the video**")
        st.altair_chart(alt.Chart(gps).mark_circle(size=12).encode(
            x=alt.X("e:Q", title="East (m)", scale=alt.Scale(zero=False)),
            y=alt.Y("n:Q", title="North (m)", scale=alt.Scale(zero=False)),
            color=alt.Color("t:Q", title="t (s)", scale=alt.Scale(scheme="viridis")),
            tooltip=["t", "lat", "lon"]).properties(height=280), use_container_width=True)
    st.caption(report.note)
