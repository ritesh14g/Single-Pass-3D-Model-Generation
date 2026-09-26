"""Stage Lab panel — Stage 6: Viewer & QA (spec §8.4–8.5).

Contract every stage panel implements (see ui/stages/__init__.py):
    KEY, render_params(cfg) -> overrides, evaluate(run_dir, truth) -> StageEvaluation,
    headline(run_dir) -> dict, render_results(run_dir, truth, evaluation)

Optional (this panel): ``existing_runs()`` + ``rerun(run_dir, cfg)`` — Stage 6 on any run with
an export (including runs copied back from the GPU box), without re-running anything else.
The viewer is served from a local background thread and embedded below the scorecard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.core.config import Config, load_config
from src.qa.stage6_eval import QaOutputs, StageEvaluation, evaluate_qa

KEY = "viewer_qa"
ROOT = Path(__file__).resolve().parents[2]
RUN_ROOTS = [ROOT / "data" / "interim", ROOT / "data" / "lab" / "runs", ROOT / "data" / "box" / "runs",
             ROOT / "data" / "outputs"]


def render_params(cfg: Config) -> dict[str, Any]:
    values: dict[str, Any] = {}
    v = cfg.get_path("qa.viewer")
    with st.sidebar.expander("Viewer (§8.4)", expanded=True):
        values["qa.viewer.max_texture_px"] = st.select_slider(
            "Texture size limit (px)", [2048, 4096, 8192, 16384], int(v["max_texture_px"]),
            help="WebGL guarantees 4096; 8192 works on desktop GPUs. Larger = sharper, heavier.")
        values["qa.viewer.layer_max_distance_m"] = st.slider(
            "Layer reach (m)", 0.5, 10.0, float(v["layer_max_distance_m"]), 0.5,
            help="A mesh vertex takes the confidence of the nearest exported point within this distance.")
    with st.sidebar.expander("Accuracy reference (§8.5)", expanded=True):
        ref = st.text_input("Reference surface (DSM .tif or .las/.laz)",
                            str(cfg.get_path("qa.reference.file") or ""),
                            help="Lidar or survey surface in any CRS. Empty = agreement with GPS only.")
        values["qa.reference.file"] = ref.strip() or None
    return {k: val for k, val in values.items() if val != cfg.get_path(k)}


# -- re-run on an existing run ------------------------------------------------------------
def existing_runs() -> list[Path]:
    runs = []
    for root in RUN_ROOTS:
        if root.is_dir():
            runs += [d for d in sorted(root.iterdir()) if (d / "export" / "metadata.json").is_file()]
    return runs


def rerun(run_dir: Path, cfg: Config) -> Path:
    from src.qa.stage import run_qa

    ref = cfg.get_path("qa.reference.file", None)
    run_qa(Path(run_dir), Path(run_dir) / "qa", cfg, reference=Path(ref) if ref else None)
    return Path(run_dir)


# -- scorecard + headline ------------------------------------------------------------------
def evaluate(run_dir: Path, truth: dict | None) -> StageEvaluation:
    manifest = Path(run_dir) / "manifest.json"
    cfg = Config(json.loads(manifest.read_text(encoding="utf-8")).get("config", {})) if manifest.is_file() \
        else load_config()
    return evaluate_qa(QaOutputs.load(run_dir), cfg)


def headline(run_dir: Path) -> dict[str, Any]:
    o = QaOutputs.load(run_dir)
    layers = o.scene.get("layers") or {}
    scores = [v.get("score") for v in (o.report.get("scorecards") or {}).values() if v.get("score") is not None]
    return {"viewer": "built" if o.scene else "—",
            "overlays": ", ".join(k for k, on in layers.items() if on) or "—",
            "mean stage score": round(sum(scores) / len(scores), 1) if scores else None,
            "zone 1 vs ref (m)": ((o.accuracy.get("points_vs_reference_dsm") or {}).get("zone1_measured") or {}).get("rms_m")
            if o.accuracy else None}


# -- diagnostics -------------------------------------------------------------------------------
def render_results(run_dir: Path, truth: dict | None, evaluation: StageEvaluation) -> None:
    o = QaOutputs.load(run_dir)
    viewer = o.qa_dir / "viewer"
    if (viewer / "scene.glb").is_file():
        st.markdown("**Web viewer** — the page judges see (photo / confidence / zones, gaps, measurement).")
        c1, c2 = st.columns([1, 3])
        show = c1.toggle("Show the viewer here", value=False, key="viewer_qa_embed",
                         help="Starts a local server for this run's qa/viewer folder.")
        c2.code(f".venv\\Scripts\\python -m src.cli view {run_dir}", language=None)
        if show:
            from src.viewer.serve import serve_background

            url = serve_background(viewer, 0)
            st.link_button("Open in a new tab", url)
            st.components.v1.iframe(url, height=640)
    else:
        st.info("No viewer in this run yet: re-run Stage 6 here, or `python -m src.cli qa <run>`.")

    r = o.report
    if r:
        cards = pd.DataFrame([{"stage": f"{v['number']}: {v['title']}", "score": v.get("score"),
                               "pass": (v.get("counts") or {}).get("pass"), "warn": (v.get("counts") or {}).get("warn"),
                               "fail": (v.get("counts") or {}).get("fail"), "note": v.get("skipped", "")}
                              for v in (r.get("scorecards") or {}).values()])
        st.markdown("**Every stage's scorecard on this run**")
        st.dataframe(cards, hide_index=True, use_container_width=True)
        t = r.get("time") or {}
        if t.get("stage_seconds"):
            st.markdown("**Time per stage (s)**")
            st.bar_chart(pd.Series(t["stage_seconds"], name="seconds"), horizontal=True)
        if r.get("limitations"):
            st.markdown("**Limitations (stated first in the report)**")
            st.markdown("\n".join(f"- {x}" for x in r["limitations"]))
        html_path = o.qa_dir / "report.html"
        if html_path.is_file():
            st.download_button("Download report.html", html_path.read_bytes(), file_name=f"{Path(run_dir).name}_qa.html",
                               mime="text/html")
    if o.accuracy:
        rows = [{"points": k, **(v or {})} for k, v in (o.accuracy.get("points_vs_reference_dsm") or {}).items()]
        st.markdown("**Heights against the reference surface, per zone (m)**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.json(o.accuracy.get("horizontal_shift") or {})
