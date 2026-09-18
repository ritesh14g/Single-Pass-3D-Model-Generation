"""Shared Stage Lab plumbing: inputs, runs, config overrides, scorecards, history.

Stage panels (``ui/stages/``) supply parameters, evaluation and charts; this
module supplies everything that is the same for every stage, so a new stage's
panel is only the part that is actually about that stage.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.core.config import Config, load_config
from src.pipeline import RunInputs, run_pipeline
from src.qa.stage1_eval import StageEvaluation
from src.qa.synthetic import SRT_DIALECTS, generate_synthetic_flight, load_ground_truth

ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "data" / "lab"
RUNS_DIR = LAB_DIR / "runs"
HISTORY_PATH = LAB_DIR / "history.jsonl"
RAW_DIR = ROOT / "data" / "raw"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}

STATUS_COLORS = {"pass": "#1f883d", "warn": "#bf8700", "fail": "#cf222e", "info": "#6e7781"}
STATUS_ICONS = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}


@dataclass
class LabInput:
    video: Path
    srt: Path | None
    csv: Path | None
    truth: dict | None
    label: str


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def input_picker(key: str) -> LabInput | None:
    """Choose what to test on: a generated flight with ground truth, or real footage."""
    mode = st.radio(
        "Test input",
        ["Synthetic flight (ground truth)", "Video in data/raw", "Upload a video"],
        horizontal=True, key=f"{key}_input_mode",
        help="Synthetic flights come with exact ground truth, which unlocks accuracy KPIs. "
             "Real footage is scored on internal consistency only.",
    )
    if mode.startswith("Synthetic"):
        return _synthetic_picker(key)
    if mode.startswith("Video in"):
        return _raw_picker(key)
    return _upload_picker(key)


def _synthetic_picker(key: str) -> LabInput:
    c1, c2, c3, c4 = st.columns(4)
    frames = c1.number_input("Frames", 30, 2000, 150, step=30, key=f"{key}_syn_frames")
    width = c2.selectbox("Width (px)", [640, 960, 1280, 1920], key=f"{key}_syn_width")
    overlap = c3.slider("True overlap per frame", 0.80, 0.99, 0.92, 0.01, key=f"{key}_syn_overlap",
                        help="Overlap between *adjacent* video frames; sets camera speed.")
    telemetry = c4.selectbox("Telemetry", SRT_DIALECTS, key=f"{key}_syn_tel",
                             help="DJI SRT dialect, CSV flight log, or none (scale-free path).")
    c5, c6, c7, _ = st.columns(4)
    blur_every = c5.number_input("Blur every Nth frame (0 = none)", 0, 50, 11, key=f"{key}_syn_blur")
    blur_length = c6.number_input("Blur kernel length (px)", 3, 61, 25, step=2, key=f"{key}_syn_blen")
    seed = c7.number_input("Seed", 0, 9999, 5, key=f"{key}_syn_seed")

    params = dict(frames=int(frames), width=int(width), height=int(width * 3 // 4), overlap=float(overlap),
                  blur_every=int(blur_every) or None, blur_length=int(blur_length), seed=int(seed),
                  telemetry=telemetry)
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    name = f"syn_{digest}"
    video = LAB_DIR / "synthetic" / f"{name}.mp4"
    if not video.is_file():
        with st.spinner("Rendering synthetic flight…"):
            generate_synthetic_flight(LAB_DIR / "synthetic", name=name, **params)
    truth = load_ground_truth(video)
    label = f"synthetic {params['frames']}f {params['width']}px ov{params['overlap']:.2f} blur{params['blur_every']} {telemetry}"
    return LabInput(video=video, srt=None, csv=None, truth=truth, label=label)


def _raw_picker(key: str) -> LabInput | None:
    videos = sorted(p for p in RAW_DIR.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        st.info(f"No videos in {RAW_DIR}. Drop a clip (and its .SRT) there, or use a synthetic flight.")
        return None
    video = st.selectbox("Video", videos, format_func=lambda p: p.name, key=f"{key}_raw_video")
    st.caption("Telemetry sidecars (.SRT / .csv / .txt) next to the video are found automatically.")
    return LabInput(video=video, srt=None, csv=None, truth=load_ground_truth(video), label=video.name)


def _upload_picker(key: str) -> LabInput | None:
    upload = st.file_uploader("Video", type=[s.strip(".") for s in VIDEO_SUFFIXES], key=f"{key}_up_video")
    sidecar = st.file_uploader(
        "Telemetry sidecar (optional)", type=["srt", "csv", "txt"], key=f"{key}_up_side",
        help="DJI .SRT, a flight-log CSV/TXT, or a binary DJIFlightRecord_*.txt from DJI GO/Pilot "
             "(log format v12 or older; v13+ is encrypted by DJI).")
    if upload is None:
        return None
    target = LAB_DIR / "uploads"
    target.mkdir(parents=True, exist_ok=True)
    video = target / upload.name
    video.write_bytes(upload.getbuffer())
    srt = csv = None
    if sidecar is not None:
        side = target / sidecar.name
        side.write_bytes(sidecar.getbuffer())
        srt, csv = (side, None) if side.suffix.lower() == ".srt" else (None, side)
    return LabInput(video=video, srt=srt, csv=csv, truth=None, label=upload.name)


# --------------------------------------------------------------------------
# Config and runs
# --------------------------------------------------------------------------
def build_config(preset: str, overrides: dict[str, Any]) -> Config:
    items = [f"{k}={json.dumps(v)}" for k, v in overrides.items()]
    items += ["run.resume=false", "logging.console_color=false", "logging.level=WARNING"]
    return load_config(preset=None if preset == "default" else preset, overrides=items)


def preset_picker(key: str) -> str:
    return st.sidebar.selectbox("Config preset", ["default", "fast", "accurate"], key=f"{key}_preset",
                                help="Parameters below start from this preset (configs/*.yaml).")


def execute(lab_input: LabInput, cfg: Config, manifest_stages: list[str], tag: str) -> Path:
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    run_pipeline(
        RunInputs(video=lab_input.video, srt=lab_input.srt, csv=lab_input.csv),
        cfg, run_dir=run_dir, stages=manifest_stages, force=True,
    )
    return run_dir


# --------------------------------------------------------------------------
# Optional inputs
# --------------------------------------------------------------------------
INPUT_ICONS = {"present": "✅", "absent": "➖", "unknown": "❔"}


def render_optional_inputs(run_dir: "Path | str") -> None:
    """What this run had to work with, and what ran in place of what it lacked.

    Absent is not an error — every one of these has a working fallback. It is
    shown next to the scorecard because a KPI that looks wrong is very often a
    missing input rather than a broken stage, and that connection is invisible
    when the fallback is only a warning line in run.jsonl.
    """
    path = Path(run_dir) / "optional_inputs.json"
    if not path.is_file():
        return
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not ledger:
        return

    counts = {s: sum(1 for r in ledger if r["status"] == s) for s in INPUT_ICONS}
    label = f"Optional inputs — {counts['present']} present, {counts['absent']} absent"
    if counts["unknown"]:
        label += f", {counts['unknown']} not determined"

    with st.expander(label, expanded=bool(counts["absent"])):
        st.caption(
            "Absent inputs are not failures: each one has a fallback and the run "
            "continues. They are listed because a surprising KPI is often a missing "
            "input rather than a broken stage."
        )
        frame = pd.DataFrame([
            {
                "": INPUT_ICONS.get(row["status"], ""),
                "input": row["label"],
                "spec": row["spec_ref"],
                "status": row["status"],
                "detail": row["detail"],
                "running instead": row["fallback"] if row["status"] == "absent" else "—",
                "cost": row["impact"] if row["status"] == "absent" else "—",
            }
            for row in ledger
        ])
        st.dataframe(
            frame, hide_index=True, use_container_width=True,
            column_config={
                "": st.column_config.TextColumn(width="small"),
                "detail": st.column_config.TextColumn(width="medium"),
                "cost": st.column_config.TextColumn(width="large"),
            },
        )


# --------------------------------------------------------------------------
# Scorecards
# --------------------------------------------------------------------------
def render_scorecard(evaluation: StageEvaluation, title: str | None = None) -> None:
    counts = evaluation.counts()
    cols = st.columns(5)
    cols[0].metric(title or "Stage score", f"{evaluation.score:.0f} / 100" if evaluation.score is not None else "—")
    cols[1].metric("Pass", counts["pass"])
    cols[2].metric("Warn", counts["warn"])
    cols[3].metric("Fail", counts["fail"])
    cols[4].metric("Ground truth", "yes" if evaluation.has_ground_truth else "no")

    table = evaluation.table()
    if table.empty:
        return
    table["status"] = table["status"].map(lambda s: f"{STATUS_ICONS.get(s, '')} {s}")
    table["value"] = table.apply(lambda r: f"{r['value']} {r['unit']}".strip() if r["value"] is not None else "—",
                                 axis=1)
    for group, rows in table.groupby("group", sort=False):
        st.markdown(f"**{group}**")
        st.dataframe(
            rows[["status", "label", "value", "target", "detail"]],
            hide_index=True, use_container_width=True,
            column_config={"detail": st.column_config.TextColumn(width="large")},
        )


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------
def record_history(stage: str, lab_input: LabInput, preset: str, overrides: dict[str, Any],
                   evaluation: StageEvaluation, run_dir: Path, elapsed_s: float | None) -> None:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "input": lab_input.label,
        "preset": preset,
        "overrides": overrides,
        "score": evaluation.score,
        "counts": evaluation.counts(),
        "elapsed_s": elapsed_s,
        "kpis": {k.key: k.value for k in evaluation.kpis},
        "run_dir": str(run_dir),
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def load_history(stage: str) -> pd.DataFrame:
    if not HISTORY_PATH.is_file():
        return pd.DataFrame()
    rows = [json.loads(line) for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("stage") == stage]
    return pd.DataFrame(rows)


def render_history(stage: str) -> None:
    history = load_history(stage)
    if history.empty:
        st.info("No runs recorded yet. Every run from this page is logged to data/lab/history.jsonl.")
        return
    history = history.reset_index(drop=True)
    history["run"] = history.index + 1
    kpi_frame = pd.json_normalize(history["kpis"]).add_prefix("kpi.")
    wide = pd.concat([history.drop(columns=["kpis"]), kpi_frame], axis=1)

    numeric = [c for c in kpi_frame.columns if pd.api.types.is_numeric_dtype(kpi_frame[c])]
    choice = st.selectbox("Plot across runs", ["score"] + numeric, key=f"{stage}_hist_metric")
    chart = alt.Chart(wide).mark_line(point=True).encode(
        x=alt.X("run:O", title="Run"),
        y=alt.Y(f"{choice}:Q", title=choice),
        tooltip=["run", "ts", "input", "preset", choice],
    ).properties(height=260)
    st.altair_chart(chart, use_container_width=True)
    wide["overrides"] = wide["overrides"].map(lambda o: json.dumps(o) if o else "")
    st.dataframe(wide[["run", "ts", "input", "preset", "score", "elapsed_s", "overrides"]],
                 hide_index=True, use_container_width=True)
