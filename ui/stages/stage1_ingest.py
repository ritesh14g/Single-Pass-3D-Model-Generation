"""Stage Lab panel — Stage 1: Ingest (spec §4).

Contract every stage panel implements (see ui/stages/__init__.py):
    KEY, render_params(cfg) -> overrides, evaluate(run_dir, truth) -> StageEvaluation,
    render_results(run_dir, truth, evaluation)
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
from src.ingest.video_reader import VideoReader
from src.ingest.telemetry import TELEMETRY_SOURCES
from src.qa.stage1_eval import (
    OVERLAP_BAND,
    IngestOutputs,
    StageEvaluation,
    blur_timeline,
    evaluate_ingest,
    overlap_series,
    track_metres,
)

KEY = "ingest"
VERDICT_COLORS = alt.Scale(domain=["clean", "correct", "reject"], range=["#1f883d", "#bf8700", "#cf222e"])


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def render_params(cfg: Config) -> dict[str, Any]:
    """Widgets for every Stage 1 tunable; returns only values changed from the preset."""
    sel = cfg.get_path("ingest.frame_selection")
    blur = cfg.get_path("condition.blur")
    video = cfg.get_path("ingest.video")
    values: dict[str, Any] = {}

    with st.sidebar.expander("Frame selection (§4.3)", expanded=True):
        values["ingest.frame_selection.target_overlap"] = st.slider(
            "Target overlap", 0.40, 0.95, float(sel["target_overlap"]), 0.01,
            help="Overlap the selector aims for between consecutive kept frames. Spec: 0.70–0.80.")
        values["ingest.frame_selection.overlap_tolerance"] = st.slider(
            "Overlap tolerance", 0.01, 0.20, float(sel["overlap_tolerance"]), 0.01)
        c1, c2 = st.columns(2)
        values["ingest.frame_selection.min_stride_frames"] = c1.number_input(
            "Min stride", 1, 30, int(sel["min_stride_frames"]))
        values["ingest.frame_selection.max_stride_frames"] = c2.number_input(
            "Max stride", 2, 600, int(sel["max_stride_frames"]))
        max_seconds = sel.get("max_stride_seconds")
        use_seconds = st.checkbox(
            "Max stride in seconds (overrides frames)", max_seconds is not None,
            help="Frame counts mean different durations at different fps: 90 frames is 1.5 s at 60 fps "
                 "but 3 s at 30 fps. Seconds are converted with the clip's own fps.")
        values["ingest.frame_selection.max_stride_seconds"] = (
            st.number_input("Max stride (s)", 0.1, 60.0, float(max_seconds or 10.0), step=0.5)
            if use_seconds else None)
        c3, c4 = st.columns(2)
        values["ingest.frame_selection.min_frames"] = c3.number_input("Min frames", 2, 1000, int(sel["min_frames"]))
        values["ingest.frame_selection.max_frames"] = c4.number_input("Max frames", 4, 10000, int(sel["max_frames"]))
        values["ingest.frame_selection.prefer_keyframes"] = st.checkbox(
            "Prefer I-frames", bool(sel["prefer_keyframes"]), help="Needs PyAV; otherwise ignored.")
        values["ingest.frame_selection.keyframe_search_radius"] = st.number_input(
            "Replacement search radius (frames)", 0, 15, int(sel["keyframe_search_radius"]))

    with st.sidebar.expander("Optical flow (overlap estimator)"):
        flow = sel["flow"]
        values["ingest.frame_selection.flow.pyramid_levels"] = st.slider(
            "Pyramid levels", 1, 7, int(flow["pyramid_levels"]),
            help="Too few levels and large motions silently mistrack.")
        values["ingest.frame_selection.flow.win_size"] = st.select_slider(
            "Window size", [9, 13, 15, 21, 31, 41], int(flow["win_size"]))
        values["ingest.frame_selection.flow.max_corners"] = st.number_input(
            "Max corners", 20, 3000, int(flow["max_corners"]), step=50)

    with st.sidebar.expander("Blur gate (§5.2 via §4.3)"):
        values["condition.blur.robust_sigma"] = st.slider(
            "Outlier sigma", 0.5, 6.0, float(blur["robust_sigma"]), 0.25,
            help="Reject frames this many robust sigmas below the video's median sharpness.")
        values["condition.blur.absolute_floor"] = st.number_input(
            "Absolute floor (var. of Laplacian)", 0.0, 1000.0, float(blur["absolute_floor"]), step=5.0)
        values["condition.blur.max_reject_fraction"] = st.slider(
            "Coverage cap (max reject fraction)", 0.0, 0.9, float(blur["max_reject_fraction"]), 0.05)
        values["condition.blur.directional.fft_anisotropy_reject"] = st.slider(
            "Directional anisotropy limit", 1.2, 6.0, float(blur["directional"]["fft_anisotropy_reject"]), 0.1)
        values["condition.blur.directional.strict_percentile"] = st.slider(
            "Directional frames must beat percentile", 5, 95, int(blur["directional"]["strict_percentile"]), 5)
        values["condition.blur.max_consecutive_rejects"] = st.number_input(
            "Coverage warning after N consecutive rejects", 1, 50, int(blur["max_consecutive_rejects"]))
        rolling = blur["rolling_baseline"]
        values["condition.blur.rolling_baseline.enabled"] = st.checkbox(
            "Rolling baseline (judge each frame against its neighbours)", bool(rolling["enabled"]),
            help="Sharpness depends on the scene. Off = one whole-video threshold, which rejected 450 "
                 "in-focus pasture frames on a real clip (S1-7). Motion-smear detection is unaffected.")
        c5, c6 = st.columns(2)
        values["condition.blur.rolling_baseline.window"] = c5.number_input(
            "Window (frames)", 5, 300, int(rolling["window"]))
        values["condition.blur.rolling_baseline.min_samples"] = c6.number_input(
            "Warm-up frames", 1, 100, int(rolling["min_samples"]))
        values["condition.blur.rolling_baseline.max_local_drop"] = st.slider(
            "Reject if softer than neighbours by", 0.1, 0.9, float(rolling["max_local_drop"]), 0.05,
            help="0.4 = a frame 40% below the median of its window is rejected.")
        values["condition.blur.rolling_baseline.hard_floor_fraction"] = st.slider(
            "Hard floor (fraction of whole-video median)", 0.0, 0.9, float(rolling["hard_floor_fraction"]), 0.05,
            help="Below this a frame is rejected however its neighbours score — stops a long blurred run "
                 "from lowering its own bar.")

    with st.sidebar.expander("Video & telemetry (§4.1–4.2)"):
        widths = [None, 3840, 1920, 1280, 960]
        current = video["max_width"] if video["max_width"] in widths else None
        values["ingest.video.max_width"] = st.selectbox(
            "Working max width", widths, index=widths.index(current),
            format_func=lambda w: "native" if w is None else f"{w}px")
        values["ingest.video.hardware_decode"] = st.checkbox("Try hardware decode", bool(video["hardware_decode"]))
        sources = list(cfg.get_path("ingest.telemetry.sources"))
        # Options come from the registry, plus anything a preset configured that
        # the registry does not list — a default outside the options would make
        # Streamlit raise and take the whole panel down with it.
        options = list(dict.fromkeys(list(TELEMETRY_SOURCES) + sources))
        chosen = st.multiselect("Telemetry sources (priority order)", options, sources,
                                help="klv = MISB ST 0601 / STANAG 4609 embedded metadata.")
        values["ingest.telemetry.sources"] = chosen
        record = cfg.get_path("ingest.telemetry.flight_record")
        segment = st.text_input(
            "DJI flight record: recording", str(record["segment"]),
            help="'auto' picks the recording whose duration matches the video; a number picks one (0 = first).")
        segment = segment.strip() or "auto"
        values["ingest.telemetry.flight_record.segment"] = int(segment) if segment.isdigit() else segment
        manual = st.checkbox("DJI flight record: set video start manually", record["offset_s"] is not None,
                             help="For clips trimmed after recording, where no recording matches the duration.")
        values["ingest.telemetry.flight_record.offset_s"] = (
            st.number_input("Video starts at flight time (s)", 0.0, 7200.0, float(record["offset_s"] or 0.0), step=0.1)
            if manual else None)

    # Keep only what differs, so history shows what was actually changed.
    return {k: v for k, v in values.items() if cfg.get_path(k, None) != v}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def _load(run_dir: Path) -> tuple[RunManifest, IngestOutputs, Config]:
    manifest = RunManifest.load(run_dir)
    outputs = IngestOutputs.load(run_dir / "ingest", manifest.stages["ingest"].metrics)
    return manifest, outputs, Config(manifest.config)


def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    _, outputs, cfg = _load(run_dir)
    return evaluate_ingest(outputs, cfg, truth)


def headline(run_dir: Path) -> dict[str, Any]:
    """One-line outputs for the pipeline view."""
    manifest, outputs, _ = _load(run_dir)
    m = manifest.stages["ingest"].metrics
    return {
        "frames selected": len(outputs.selection),
        "video frames": m.get("video", {}).get("frame_count"),
        "median overlap": m.get("selection", {}).get("overlap", {}).get("median"),
        "telemetry": m.get("telemetry", {}).get("source"),
    }


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def render_results(run_dir: Path, truth: dict | None, evaluation: StageEvaluation) -> None:
    manifest, outputs, cfg = _load(run_dir)
    metrics = manifest.stages["ingest"].metrics
    tabs = st.tabs(["Frame selection", "Blur gate", "Telemetry", "Selected frames", "Timing", "Raw metrics"])
    with tabs[0]:
        _selection_tab(outputs, metrics, truth)
    with tabs[1]:
        _blur_tab(outputs, metrics, truth)
    with tabs[2]:
        _telemetry_tab(outputs, metrics)
    with tabs[3]:
        _frames_tab(manifest, outputs)
    with tabs[4]:
        _timing_tab(manifest, metrics, cfg)
    with tabs[5]:
        st.json(metrics, expanded=False)
        warnings = manifest.stages["ingest"].warnings
        if warnings:
            st.warning("\n".join(warnings))


def _selection_tab(outputs: IngestOutputs, metrics: dict, truth: dict | None) -> None:
    series = overlap_series(outputs, truth)
    if series.empty:
        st.error("Fewer than two frames selected — nothing to plot.")
        return
    total = metrics.get("video", {}).get("frame_count") or int(series["frame_index"].max())
    st.caption("How much of the previous kept frame each new frame still sees. The green band is the "
               "§4.3 target; points below 0.5 are likely breaks in the reconstruction's match graph.")
    band = alt.Chart(pd.DataFrame({"lo": [OVERLAP_BAND[0]], "hi": [OVERLAP_BAND[1]]})).mark_rect(
        opacity=0.15, color="#1f883d").encode(y="lo:Q", y2="hi:Q")
    long = series.melt(id_vars=["pair", "frame_index", "stride"], value_vars=[c for c in ("measured", "true") if c in series],
                       var_name="series", value_name="overlap")
    line = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("frame_index:Q", title="Frame index", scale=alt.Scale(domain=[0, total])),
        y=alt.Y("overlap:Q", title="Overlap with previous kept frame", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("series:N", scale=alt.Scale(domain=["measured", "true"], range=["#0969da", "#8250df"])),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["measured", "true"], range=[[1, 0], [4, 3]])),
        tooltip=["pair", "frame_index", "stride", "series", alt.Tooltip("overlap:Q", format=".3f")],
    )
    st.altair_chart((band + line).properties(height=300), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Overlap distribution**")
        hist = alt.Chart(series).mark_bar().encode(
            x=alt.X("measured:Q", bin=alt.Bin(step=0.025), title="Measured overlap"),
            y=alt.Y("count():Q", title="Pairs"))
        st.altair_chart((band.encode(x=alt.X("lo:Q"), x2="hi:Q", y=alt.value(0), y2=alt.value(220)) + hist)
                        .properties(height=220), use_container_width=True)
    with c2:
        st.markdown("**Adaptive stride** (frames skipped between kept frames)")
        stride = alt.Chart(series).mark_bar(color="#6e7781").encode(
            x=alt.X("frame_index:Q", title="Frame index"), y=alt.Y("stride:Q", title="Stride"),
            tooltip=["frame_index", "stride"])
        st.altair_chart(stride.properties(height=220), use_container_width=True)

    notes = metrics.get("selection", {}).get("notes", [])
    for note in notes:
        st.warning(note)


def _blur_tab(outputs: IngestOutputs, metrics: dict, truth: dict | None) -> None:
    timeline = blur_timeline(outputs)
    if timeline.empty:
        st.info("No blur evaluations recorded.")
        return
    profile = metrics.get("selection", {}).get("blur_profile", {})
    if truth:
        timeline["truly_blurred"] = timeline["index"].isin(truth.get("blurred_indices", []))
    thresholds = pd.DataFrame([
        {"name": "reject", "value": profile.get("reject_threshold")},
        {"name": "directional reject", "value": profile.get("directional_reject_threshold")},
        {"name": "clean", "value": profile.get("clean_threshold")},
    ]).dropna()

    st.caption("Every frame the gate evaluated. Only frames the stride landed on are scored — the gate "
               "never decodes the whole video. Horizontal rules are the thresholds derived from this video's "
               "own sharpness distribution.")
    tooltip = ["index", "verdict", alt.Tooltip("blur_score:Q", format=".1f"),
               alt.Tooltip("anisotropy:Q", format=".2f"), "selected"] + (["truly_blurred"] if truth else [])
    points = alt.Chart(timeline).mark_point(filled=True, size=70).encode(
        x=alt.X("index:Q", title="Frame index"),
        y=alt.Y("blur_score:Q", title="Sharpness (variance of Laplacian)"),
        color=alt.Color("verdict:N", scale=VERDICT_COLORS),
        shape=alt.Shape("selected:N", scale=alt.Scale(domain=[True, False], range=["diamond", "circle"])),
        tooltip=tooltip,
    )
    rules = alt.Chart(thresholds).mark_rule(strokeDash=[5, 4]).encode(
        y="value:Q", color=alt.Color("name:N", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="threshold")))
    layers = points + rules
    if truth:
        rings = alt.Chart(timeline[timeline["truly_blurred"]]).mark_point(size=260, color="#cf222e", strokeWidth=1.5).encode(
            x="index:Q", y="blur_score:Q")
        layers = layers + rings
        st.caption("Red rings mark frames that are truly blurred in the synthetic ground truth.")
    st.altair_chart(layers.resolve_scale(color="independent").properties(height=320), use_container_width=True)

    st.markdown("**Decision plane** — the gate uses two measurements, not one")
    st.caption("Sharp and smeared frames can overlap in sharpness score while separating cleanly on spectral "
               "anisotropy. Frames right of the vertical line are treated as directional (drone-jerk) blur "
               "and must beat the stricter threshold.")
    limit = profile.get("anisotropy_limit")
    plane = alt.Chart(timeline).mark_point(filled=True, size=70).encode(
        x=alt.X("anisotropy:Q", title="Spectral anisotropy (1 = isotropic)"),
        y=alt.Y("blur_score:Q", title="Sharpness"),
        color=alt.Color("verdict:N", scale=VERDICT_COLORS), tooltip=tooltip)
    layers = plane
    if limit:
        layers = layers + alt.Chart(pd.DataFrame({"x": [limit]})).mark_rule(strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart(layers.properties(height=300), use_container_width=True)

    runs = metrics.get("selection", {}).get("coverage_warnings", [])
    if runs:
        st.warning(f"{len(runs)} coverage warning(s): consecutive frames rejected")
        st.dataframe(pd.DataFrame(runs), hide_index=True)


def _telemetry_tab(outputs: IngestOutputs, metrics: dict) -> None:
    tel = metrics.get("telemetry", {})
    cols = st.columns(6)
    cols[0].metric("Source", tel.get("source", "none"))
    for col, (label, key) in zip(cols[1:], [("GPS", "has_gps"), ("Baro", "has_baro"), ("Attitude", "has_attitude"),
                                            ("Focal", "has_focal"), ("RTK", "has_rtk")]):
        col.metric(label, "yes" if tel.get(key) else "no")
    if tel.get("scale_free"):
        st.error("No usable GPS: the reconstruction would be scale-free and not georeferenced (§4.2 path 4).")
    for note in tel.get("notes", []):
        st.caption(f"note: {note}")

    track = track_metres(outputs.telemetry)
    if track.empty:
        return
    selected = track_metres(outputs.frame_telemetry)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Flight track** (local metres from first fix; dots = selected frames)")
        path = alt.Chart(track).mark_line(color="#6e7781").encode(
            x=alt.X("east_m:Q", title="East (m)"), y=alt.Y("north_m:Q", title="North (m)"), order="t:Q")
        dots = alt.Chart(selected).mark_point(filled=True, color="#0969da", size=50).encode(
            x="east_m:Q", y="north_m:Q", tooltip=[alt.Tooltip("t:Q", format=".2f")])
        st.altair_chart((path + dots).properties(height=300), use_container_width=True)
    with c2:
        st.markdown("**Altitude sources**")
        alt_cols = [c for c in ("alt_gps", "alt_baro") if c in track and track[c].notna().any()]
        if alt_cols:
            long = track.melt(id_vars=["t"], value_vars=alt_cols, var_name="source", value_name="altitude_m")
            chart = alt.Chart(long).mark_line().encode(
                x=alt.X("t:Q", title="Time (s)"), y=alt.Y("altitude_m:Q", title="Altitude (m)", scale=alt.Scale(zero=False)),
                color="source:N")
            st.altair_chart(chart.properties(height=300), use_container_width=True)
            st.caption("GPS altitude is absolute (ellipsoidal); barometric is relative to takeoff. "
                       "Fusing them is Stage 2 (§5.6).")
        else:
            st.info("No altitude in telemetry.")


def _frames_tab(manifest: RunManifest, outputs: IngestOutputs) -> None:
    sel = outputs.selection
    if sel.empty:
        return
    count = st.slider("Thumbnails", 4, min(48, len(sel)) if len(sel) > 4 else 4, min(12, len(sel)))
    picks = sel.iloc[np.linspace(0, len(sel) - 1, min(count, len(sel))).astype(int)]
    video = Path(manifest.inputs["video"])
    images = []
    with VideoReader(video, hardware_decode=False, max_width=320) as reader:
        for frame in reader.read_indices(picks["index"].tolist()):
            images.append((frame.index, frame.image[:, :, ::-1]))
    rows = {int(r["index"]): r for r in picks.to_dict("records")}
    cols = st.columns(4)
    for i, (index, image) in enumerate(images):
        r = rows.get(index, {})
        cols[i % 4].image(image, caption=f"#{index} · ov {r.get('overlap_prev', 0):.2f} · {r.get('blur_verdict', '')}",
                          use_container_width=True)


def _timing_tab(manifest: RunManifest, metrics: dict, cfg: Config) -> None:
    timing = metrics.get("timing", {})
    total = manifest.stages["ingest"].duration_s or 0.0
    budget = float(cfg.get_path("budget.stages.ingest"))
    duration = metrics.get("video", {}).get("duration_s") or 0.0
    cols = st.columns(3)
    cols[0].metric("Stage wall time", f"{total:.2f} s")
    cols[1].metric("§9 allotment", f"{budget:.0f} s")
    if duration:
        cols[2].metric("Projected for 10-min clip (same resolution)", f"{total * 600 / duration:.0f} s")
    if timing:
        frame = pd.DataFrame([{"part": k.replace("_s", ""), "seconds": v} for k, v in timing.items()])
        st.altair_chart(alt.Chart(frame).mark_bar().encode(
            x=alt.X("seconds:Q"), y=alt.Y("part:N", sort="-x"), tooltip=["part", "seconds"]).properties(height=160),
            use_container_width=True)
