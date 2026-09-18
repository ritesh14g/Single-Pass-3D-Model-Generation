"""Stage Lab panel — Stage 2: Conditioning (spec §5).

Contract every stage panel implements (see ui/stages/__init__.py):
    KEY, render_params(cfg) -> overrides, evaluate(run_dir, truth) -> StageEvaluation,
    headline(run_dir) -> dict, render_results(run_dir, truth, evaluation)
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
from src.qa.stage2_eval import (
    ConditionOutputs,
    StageEvaluation,
    artifact_timeline,
    evaluate_condition,
    illumination_timeline,
)

KEY = "condition"

STATUS_COLORS = alt.Scale(
    domain=["pass", "warn", "fail", "info"],
    range=["#1f883d", "#bf8700", "#cf222e", "#636e7b"],
)


# ---------------------------------------------------------------------------
# Parameters — sidebar widgets for every Stage 2 tunable
# ---------------------------------------------------------------------------
def render_params(cfg: Config) -> dict[str, Any]:
    """Widgets for Stage 2 tunables; returns only values changed from the preset."""
    values: dict[str, Any] = {}

    with st.sidebar.expander("Artifact suppression (§5.3)", expanded=True):
        art = cfg.get_path("condition.artifacts")
        values["condition.artifacts.enabled"] = st.checkbox(
            "Enable artifact suppression", bool(art["enabled"]))
        values["condition.artifacts.blockiness_threshold"] = st.slider(
            "Blockiness threshold", 1.0, 3.0, float(art["blockiness_threshold"]), 0.05,
            help="Score above this triggers correction + keypoint masking.")
        values["condition.artifacts.keypoint_exclusion_px"] = st.number_input(
            "Keypoint exclusion radius (px)", 0, 10, int(art["keypoint_exclusion_px"]))
        with st.sidebar.expander("Bilateral filter", expanded=False):
            bil = art["bilateral"]
            values["condition.artifacts.bilateral.diameter"] = st.number_input(
                "Filter diameter", 3, 15, int(bil["diameter"]))
            values["condition.artifacts.bilateral.sigma_color"] = st.slider(
                "Sigma color", 5.0, 200.0, float(bil["sigma_color"]), 5.0)
            values["condition.artifacts.bilateral.sigma_space"] = st.slider(
                "Sigma space", 1.0, 30.0, float(bil["sigma_space"]), 1.0)

    with st.sidebar.expander("Illumination (§5.4)"):
        illum = cfg.get_path("condition.illumination")
        values["condition.illumination.enabled"] = st.checkbox(
            "Enable illumination conditioning", bool(illum["enabled"]))
        shadow = illum["shadow"]
        st.markdown("**Shadow detection**")
        values["condition.illumination.shadow.enabled"] = st.checkbox(
            "Detect shadows", bool(shadow["enabled"]))
        values["condition.illumination.shadow.blue_ratio_delta"] = st.slider(
            "Blue ratio delta above median (sky-light gate)", 0.05, 0.5,
            float(shadow.get("blue_ratio_delta", 0.15)), 0.01)
        values["condition.illumination.shadow.luminance_ceiling_factor"] = st.slider(
            "Luminance ceiling factor", 0.8, 1.5,
            float(shadow.get("luminance_ceiling_factor", 1.05)), 0.05)
        values["condition.illumination.shadow.mvs_weight"] = st.slider(
            "MVS weight for shadow pixels", 0.0, 1.0, float(shadow["mvs_weight"]), 0.05,
            help="Photo-consistency weight during reconstruction (1.0 = no down-weighting).")
        st.markdown("**Exposure chain**")
        chain = illum["exposure_chain"]
        values["condition.illumination.exposure_chain.enabled"] = st.checkbox(
            "Chain exposure normalisation", bool(chain["enabled"]))
        c1, c2 = st.columns(2)
        values["condition.illumination.exposure_chain.min_gain"] = c1.number_input(
            "Min gain", 0.1, 1.0, float(chain["min_gain"]), step=0.05)
        values["condition.illumination.exposure_chain.max_gain"] = c2.number_input(
            "Max gain", 1.0, 5.0, float(chain["max_gain"]), step=0.1)
        st.markdown("**Low light**")
        low = illum["low_light"]
        values["condition.illumination.low_light.mean_luma_threshold"] = st.slider(
            "Low-light luma threshold", 0.05, 0.5, float(low["mean_luma_threshold"]), 0.01)
        values["condition.illumination.low_light.fusion_weight"] = st.slider(
            "Low-light MVS weight", 0.0, 1.0, float(low["fusion_weight"]), 0.05)

    with st.sidebar.expander("Dynamic masking (§5.5)"):
        dyn = cfg.get_path("condition.dynamic")
        values["condition.dynamic.enabled"] = st.checkbox(
            "Enable dynamic masking", bool(dyn["enabled"]))
        values["condition.dynamic.conf_threshold"] = st.slider(
            "YOLO confidence threshold", 0.05, 0.9, float(dyn["conf_threshold"]), 0.05)
        values["condition.dynamic.dilate_px"] = st.number_input(
            "Mask dilation (px)", 0, 50, int(dyn["dilate_px"]))
        geo = dyn["geometric"]
        values["condition.dynamic.geometric.enabled"] = st.checkbox(
            "Geometric consistency fallback", bool(geo["enabled"]),
            help="Used when the semantic model is unavailable.")
        values["condition.dynamic.geometric.depth_disagreement"] = st.slider(
            "Depth disagreement threshold (relative)", 0.05, 0.5,
            float(geo["depth_disagreement"]), 0.01)

    with st.sidebar.expander("GPS conditioning (§5.6)"):
        gps = cfg.get_path("condition.gps")
        values["condition.gps.enabled"] = st.checkbox(
            "Enable GPS conditioning", bool(gps["enabled"]))
        values["condition.gps.max_speed_mps"] = st.slider(
            "Platform max speed (m/s)", 5.0, 100.0, float(gps["max_speed_mps"]), 1.0)
        values["condition.gps.max_accel_mps2"] = st.slider(
            "Platform max acceleration (m/s²)", 1.0, 30.0, float(gps["max_accel_mps2"]), 0.5)
        values["condition.gps.median_window"] = st.number_input(
            "Median-filter window (fixes)", 3, 21, int(gps["median_window"]))
        kalman = gps["kalman"]
        st.markdown("**Kalman smoother**")
        values["condition.gps.kalman.process_noise"] = st.slider(
            "Process noise (m/s²)", 0.01, 5.0, float(kalman["process_noise"]), 0.05)
        values["condition.gps.kalman.measurement_noise_horizontal"] = st.slider(
            "Horizontal GPS noise (m)", 0.1, 20.0, float(kalman["measurement_noise_horizontal"]), 0.1)
        values["condition.gps.kalman.measurement_noise_vertical"] = st.slider(
            "Vertical GPS noise (m)", 0.5, 50.0, float(kalman["measurement_noise_vertical"]), 0.5)

    return values


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _load(run_dir: Path) -> tuple[ConditionOutputs, Config]:
    """Stage 2 artifacts plus the config the run actually used.

    ``run_dir`` is the run root; conditioning writes under ``condition/``. The
    config comes from the manifest, not ``load_config()``, so a Lab run scores
    against the parameters it was given rather than the defaults.
    """
    manifest = RunManifest.load(run_dir)
    return ConditionOutputs.load(run_dir / "condition"), Config(manifest.config)


def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    outputs, cfg = _load(run_dir)
    return evaluate_condition(outputs, cfg, truth)


# ---------------------------------------------------------------------------
# Headline numbers (shown in the pipeline overview)
# ---------------------------------------------------------------------------
def headline(run_dir: Path) -> dict[str, Any]:
    outputs, _ = _load(run_dir)
    illum = outputs.illumination_report
    art = outputs.artifacts_report
    gps = outputs.gps_report
    return {
        "shadow_pct": round(float(illum.get("shadow_fraction_mean", 0.0)) * 100, 1),
        "blocked_frames": int(art.get("frames_corrected", 0)),
        "gps_outliers": int(gps.get("envelope_outliers", 0)) + int(gps.get("median_outliers", 0)),
        "low_light_frames": int(illum.get("low_light_frames", 0)),
    }


# ---------------------------------------------------------------------------
# Results — charts and diagnostic tables
# ---------------------------------------------------------------------------
def render_results(
    run_dir: Path, truth: dict | None, evaluation: StageEvaluation
) -> None:
    """Render the Stage 2 diagnostics panel."""
    outputs, _ = _load(run_dir)

    # ── Scorecard ─────────────────────────────────────────────────────────
    st.subheader("Scorecard")
    score = evaluation.score
    counts = evaluation.counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score", f"{score:.1f}%" if score is not None else "—")
    c2.metric("Pass", counts.get("pass", 0))
    c3.metric("Warn", counts.get("warn", 0))
    c4.metric("Fail", counts.get("fail", 0))
    c5.metric("Info", counts.get("info", 0))

    df = evaluation.table()
    if not df.empty:
        styled = df[["group", "label", "value", "target", "status", "detail"]].copy()
        styled["status"] = styled["status"].str.upper()
        # KPI values are deliberately heterogeneous (floats, counts, bools and
        # strings like "baro+gps_complementary"). Arrow infers a numeric column
        # from the leading rows and then fails on the first string, so render
        # them all as text.
        styled["value"] = styled["value"].map(lambda v: "—" if v is None else str(v))
        def _row_color(row):
            # Translucent tints, not opaque pastels: the fill sits over whatever
            # ground the viewer's theme paints, so the text keeps its contrast in
            # both. Opaque light fills made every row unreadable in dark mode.
            tints = {
                "pass": "rgba(31, 136, 61, 0.22)",
                "warn": "rgba(191, 135, 0, 0.26)",
                "fail": "rgba(207, 34, 46, 0.24)",
                "info": "rgba(110, 118, 129, 0.16)",
            }
            return ["background-color: " + tints.get(row["status"].lower(), "")] * len(row)
        st.dataframe(
            styled.style.apply(_row_color, axis=1),
            use_container_width=True, hide_index=True,
        )

    # ── Illumination ──────────────────────────────────────────────────────
    st.subheader("Illumination")
    illum = outputs.illumination_report
    if illum:
        chain = illum.get("exposure_chain", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Shadow coverage (mean)", str(round(float(illum.get("shadow_fraction_mean", 0)) * 100, 1)) + "%")
        col2.metric("Low-light frames", str(illum.get("low_light_frames", 0)))
        col3.metric("Exposure gain span", str(round(float(chain.get("gain_span", 0)), 3)) if chain else "—")

        frame_il = illumination_timeline(outputs)
        if not frame_il.empty and "shadow_fraction" in frame_il.columns and "index" in frame_il.columns:
            st.markdown("**Shadow fraction per frame**")
            chart = (
                alt.Chart(frame_il)
                .mark_area(opacity=0.6, color="#d73a4a")
                .encode(
                    x=alt.X("index:Q", title="Frame index"),
                    y=alt.Y("shadow_fraction:Q", title="Shadow fraction", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["index", "shadow_fraction"],
                )
                .properties(height=160)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No illumination report found in this run directory.")

    # ── Artifact suppression ──────────────────────────────────────────────
    st.subheader("Artifact suppression")
    art = outputs.artifacts_report
    if art:
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames corrected", str(art.get("frames_corrected", 0)))
        col2.metric("Keypoints vetoed", str(art.get("keypoints_vetoed", 0)))
        col3.metric("Evaluated", str(art.get("frames_evaluated", 0)))

        frame_art = artifact_timeline(outputs)
        if not frame_art.empty and "blockiness" in frame_art.columns and "index" in frame_art.columns:
            st.markdown("**Blockiness score per frame** (> 1.35 = visible blocking)")
            threshold_df = pd.DataFrame({"index": frame_art["index"], "threshold": 1.35})
            base = alt.Chart(frame_art)
            line = base.mark_line(color="#0969da").encode(
                x=alt.X("index:Q", title="Frame index"),
                y=alt.Y("blockiness:Q", title="Blockiness score"),
                tooltip=["index", "blockiness"],
            )
            tline = alt.Chart(threshold_df).mark_rule(color="#cf222e", strokeDash=[4, 2]).encode(
                y="threshold:Q"
            )
            st.altair_chart((line + tline).properties(height=160), use_container_width=True)
    else:
        st.info("No artifact report found in this run directory.")

    # ── Dynamic masking ───────────────────────────────────────────────────
    st.subheader("Dynamic masking")
    dyn = outputs.dynamic_report
    if dyn:
        col1, col2, col3 = st.columns(3)
        col1.metric("Frames with movers", str(dyn.get("frames_with_movers", 0)))
        col2.metric("Peak masked area", str(round(float(dyn.get("masked_fraction_max", 0)) * 100, 1)) + "%")
        col3.metric("Mean masked area", str(round(float(dyn.get("masked_fraction_mean", 0)) * 100, 1)) + "%")
        methods = dyn.get("methods", [])
        classes = dyn.get("detections_by_class", {})
        if methods:
            st.caption("Methods used: " + ", ".join(methods))
        if classes:
            cls_df = pd.DataFrame(
                [{"class": k, "detections": v} for k, v in sorted(classes.items(), key=lambda x: -x[1])]
            )
            st.markdown("**Detections by class**")
            chart = (
                alt.Chart(cls_df)
                .mark_bar(color="#8250df")
                .encode(
                    x=alt.X("detections:Q"),
                    y=alt.Y("class:N", sort="-x"),
                    tooltip=["class", "detections"],
                )
                .properties(height=max(100, len(classes) * 28))
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No dynamic masking report found (model unavailable or masking skipped).")

    # ── GPS conditioning ──────────────────────────────────────────────────
    st.subheader("GPS conditioning")
    gps = outputs.gps_report
    if gps:
        notes = gps.get("notes", [])
        if "scale-free" in " ".join(notes).lower():
            st.warning("Scale-free reconstruction: no GPS in telemetry.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Input fixes", str(gps.get("input_fixes", "—")))
            col2.metric("Envelope outliers", str(gps.get("envelope_outliers", 0)))
            col3.metric("Median outliers", str(gps.get("median_outliers", 0)))
            col4.metric("Smoothed fixes", str(gps.get("smoothed_fixes", "—")))
            col1b, col2b = st.columns(2)
            col1b.metric("Altitude source", str(gps.get("altitude_source", "—")))
            rtk_label = "Yes" if gps.get("rtk_detected") else "No"
            col2b.metric("RTK detected", rtk_label)
            if notes:
                for note in notes:
                    if note:
                        st.warning(note)
    else:
        st.info("No GPS conditioning report found.")

    # ── Ground truth ──────────────────────────────────────────────────────
    if truth is not None and evaluation.has_ground_truth:
        st.subheader("Ground-truth comparison")
        gt_kpis = [k for k in evaluation.kpis if "ground truth" in k.group.lower()]
        if gt_kpis:
            gt_df = pd.DataFrame([k.to_dict() for k in gt_kpis])
            st.dataframe(gt_df[["label", "value", "target", "status", "detail"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No ground-truth KPIs available (truth dict did not include shadow/blockiness fields).")
