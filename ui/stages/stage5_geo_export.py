"""Stage Lab panel — Stage 5: Georeferencing & export (spec §8.1–8.3).

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
from src.qa.stage5_eval import ExportOutputs, StageEvaluation, evaluate_export, residuals_frame

KEY = "geo_export"
ALL_FORMATS = ["obj", "ply", "las", "geotiff", "glb", "fbx"]


def render_params(cfg: Config) -> dict[str, Any]:
    values: dict[str, Any] = {}
    with st.sidebar.expander("Georeferencing (§8.1)", expanded=True):
        g = cfg.get_path("geo")
        values["geo.vertical.output_datum"] = st.radio(
            "Output heights", ["orthometric", "ellipsoidal"], horizontal=True,
            index=["orthometric", "ellipsoidal"].index(str(g["vertical"]["output_datum"])),
            help="Orthometric = EGM96 geoid heights (sea-level based); needs the geoid grid (fetched by PROJ).")
        values["geo.vertical.gps_altitude_datum"] = st.selectbox(
            "Telemetry altitude datum", ["auto", "ellipsoidal", "orthometric"],
            index=["auto", "ellipsoidal", "orthometric"].index(str(g["vertical"].get("gps_altitude_datum", "auto"))),
            help="auto: KLV = orthometric (MSL, ST 0601), DJI = ellipsoidal (assumed).")
        values["geo.similarity.inlier_threshold_m"] = st.slider(
            "RANSAC inlier threshold (m)", 1.0, 30.0, float(g["similarity"]["inlier_threshold_m"]), 0.5,
            help="Rejects gross GPS outliers only; tighter just flatters the RMS on systematic errors (S4-1).")
    with st.sidebar.expander("Export (§8.2)", expanded=True):
        e = cfg.get_path("export")
        values["export.formats"] = st.multiselect("Formats", ALL_FORMATS, default=list(e["formats"]))
        auto = st.checkbox("Raster resolution: auto", str(e["raster"]["dsm_resolution_m"]) == "auto",
                           help="Spacing of unique surface samples (~0.5 m on Esri).")
        if not auto:
            values["export.raster.dsm_resolution_m"] = st.number_input("DSM / ortho cell (m)", 0.05, 5.0, 0.5, 0.05)
        values["export.raster.fill_max_cells"] = st.slider("Fill raster gaps up to (cells)", 0, 10,
                                                           int(e["raster"]["fill_max_cells"]))
        values["export.confidence_full_views"] = st.slider("Views for full confidence", 2, 10,
                                                           int(e["confidence_full_views"]))
    return {k: v for k, v in values.items() if v != cfg.get_path(k)}


def _load(run_dir: Path):
    manifest = RunManifest.load(Path(run_dir))
    return ExportOutputs.load(run_dir), Config(manifest.config)


def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    outputs, cfg = _load(run_dir)
    return evaluate_export(outputs, cfg)


def headline(run_dir: Path) -> dict[str, Any]:
    outputs, _ = _load(run_dir)
    meta = outputs.metadata
    return {"crs": meta.get("crs"), "gps_rms_m": outputs.georef.get("rms_all_m"),
            "formats": len(meta.get("formats_produced", [])),
            "coverage_pct": (meta.get("coverage") or {}).get("coverage_pct")}


def _preview(path: Path, rgb: bool) -> np.ndarray | None:
    try:
        import rasterio

        with rasterio.open(path) as src:
            step = max(1, max(src.width, src.height) // 900)
            data = src.read(out_shape=(src.count, src.height // step, src.width // step))
            nodata = src.nodata
    except Exception:  # noqa: BLE001
        return None
    if rgb:
        return np.moveaxis(data[:3], 0, -1).astype(np.uint8)
    dsm = data[0].astype(np.float64)
    valid = dsm != nodata if nodata is not None else np.isfinite(dsm)
    if not valid.any():
        return None
    lo, hi = np.percentile(dsm[valid], [2, 98])
    norm = np.clip((dsm - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (np.stack([norm, 0.5 + 0.5 * norm, 1 - norm], -1) * 255).astype(np.uint8)
    img[~valid] = 0
    return img


def render_results(run_dir: Path, truth: dict | None, evaluation: StageEvaluation) -> None:
    outputs, _ = _load(run_dir)
    g, meta = outputs.georef, outputs.metadata
    if not meta:
        st.info("No export metadata in this run directory.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRS", meta.get("crs") or "—")
    c2.metric("Camera vs GPS", f"{g.get('rms_all_m')} m" if g.get("rms_all_m") is not None else "—")
    c3.metric("Formats written", f"{len(meta.get('formats_produced', []))} / 6")
    cov = meta.get("coverage") or {}
    c4.metric("Coverage", f"{cov.get('coverage_pct')}%" if cov else "—")

    res = residuals_frame(outputs)
    if not res.empty:
        st.markdown("**Camera-vs-GPS residual per frame** (after the similarity fit)")
        st.caption("A smooth wave along the flight suggests telemetry timing (S4-1); isolated spikes are GPS outliers.")
        chart = (alt.Chart(res.reset_index()).mark_bar()
                 .encode(x=alt.X("index:Q", title="Frame (in name order)"), y=alt.Y("residual_m:Q", title="Residual (m)"),
                         color=alt.Color("inlier:N", scale=alt.Scale(domain=[True, False], range=["#0969da", "#cf222e"])),
                         tooltip=["frame", "residual_m", "inlier"])
                 .properties(height=180))
        st.altair_chart(chart, use_container_width=True)

    col_a, col_b = st.columns(2)
    for column, name, rgb, title in ((col_a, "dsm.tif", False, "DSM (height, low blue → high yellow)"),
                                     (col_b, "orthophoto.tif", True, "Orthophoto")):
        img = _preview(outputs.export_dir / name, rgb)
        if img is not None:
            column.markdown(f"**{title}**")
            column.image(img, use_container_width=True)

    files = pd.DataFrame(meta.get("files", []))
    if not files.empty:
        files["MB"] = (files["bytes"] / 1e6).round(1)
        st.markdown("**Files**")
        st.dataframe(files[["format", "MB", "path"]], hide_index=True, use_container_width=True)
    for fmt, reason in (meta.get("formats_failed") or {}).items():
        st.warning(f"{fmt}: {reason}")
    with st.expander("metadata.json"):
        st.json(meta)
