"""Stage Lab — interactive test bench for each pipeline stage.

    streamlit run ui/app.py

Pages:
  * Pipeline so far — runs every BUILT stage in execution order on one input
    and scores the whole chain against the §9 time budget. It extends
    automatically when a stage is flipped to BUILT in src/stages.py.
  * One page per stage — its parameters, a KPI scorecard against the spec's
    acceptance criteria, diagnostic charts, and run history.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.core.config import load_config  # noqa: E402
from src.core.manifest import RunManifest  # noqa: E402
from src.stages import (  # noqa: E402
    STAGES, StageStatus, built_stages, get_stage, manifest_stages_through,
)
from ui import lab  # noqa: E402
from ui.stages import PANELS  # noqa: E402

st.set_page_config(page_title="PS-17 Stage Lab", page_icon="🛩️", layout="wide")

STATUS_BADGE = {StageStatus.BUILT: "🟢", StageStatus.IN_PROGRESS: "🟡", StageStatus.PLANNED: "⚪"}


def stage_page(key: str) -> None:
    spec = get_stage(key)
    st.title(f"Stage {spec.number} — {spec.title}")
    st.caption(f"Spec {spec.spec_ref} · status: {spec.status.value} · {spec.summary}")

    panel = PANELS.get(key)
    if panel is None:
        st.info("This stage has no Stage Lab panel yet. It gets one when it is built "
                "(see ui/stages/__init__.py for the contract).")
        return

    preset = lab.preset_picker(key)
    base_cfg = load_config(None if preset == "default" else preset)
    overrides = panel.render_params(base_cfg)
    lab_input = lab.input_picker(key)

    state_key = f"{key}_last_run"
    if lab_input is not None and st.button(f"Run Stage {spec.number}", type="primary", key=f"{key}_run"):
        cfg = lab.build_config(preset, overrides)
        started = time.monotonic()
        with st.spinner(f"Running Stage {spec.number}…"):
            run_dir = lab.execute(lab_input, cfg, manifest_stages_through(spec), tag=key)
        elapsed = time.monotonic() - started
        evaluation = panel.evaluate(run_dir, lab_input.truth)
        lab.record_history(key, lab_input, preset, overrides, evaluation, run_dir, round(elapsed, 2))
        st.session_state[state_key] = (str(run_dir), lab_input.truth)

    # Stages that can re-run alone on a finished run (tuning without rebuilding what feeds them).
    if hasattr(panel, "existing_runs"):
        with st.expander(f"…or re-run Stage {spec.number} alone on an existing run"):
            runs = panel.existing_runs()
            if not runs:
                st.caption("No run directory with the inputs this stage needs yet.")
            else:
                chosen = st.selectbox("Run directory", runs, key=f"{key}_existing",
                                      format_func=lambda p: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
                st.caption("Writes this stage's outputs into that run directory (replacing earlier ones).")
                if st.button(f"Re-run Stage {spec.number} here", key=f"{key}_rerun"):
                    cfg = lab.build_config(preset, overrides)
                    started = time.monotonic()
                    with st.spinner(f"Running Stage {spec.number}…"):
                        run_dir = panel.rerun(chosen, cfg)
                    evaluation = panel.evaluate(run_dir, None)
                    rerun_input = lab.LabInput(video=Path(run_dir), srt=None, csv=None, truth=None,
                                               label=f"existing run {Path(run_dir).name}")
                    lab.record_history(key, rerun_input, preset, overrides, evaluation, run_dir,
                                       round(time.monotonic() - started, 2))
                    st.session_state[state_key] = (str(run_dir), None)

    if state_key not in st.session_state:
        st.info("Pick an input and parameters, then run the stage.")
        st.divider()
        st.subheader("Run history")
        lab.render_history(key)
        return

    run_dir, truth = st.session_state[state_key]
    run_dir = Path(run_dir)
    evaluation = panel.evaluate(run_dir, truth)
    st.divider()
    st.subheader("Scorecard")
    st.caption(f"Run directory: `{run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir}`")
    lab.render_scorecard(evaluation)
    lab.render_optional_inputs(run_dir)
    st.divider()
    st.subheader("Diagnostics")
    panel.render_results(run_dir, truth, evaluation)
    st.divider()
    st.subheader("Run history")
    lab.render_history(key)


def pipeline_page() -> None:
    stages = [s for s in built_stages() if s.key in PANELS]
    st.title("Pipeline so far")
    if not stages:
        st.warning("No stage is BUILT yet.")
        return
    chain = " → ".join(f"Stage {s.number} ({s.title})" for s in stages)
    st.caption(f"Runs every built stage in execution order: {chain}. "
               "This page extends automatically as stages are completed.")

    preset = lab.preset_picker("pipeline")
    cfg = lab.build_config(preset, {})
    lab_input = lab.input_picker("pipeline")
    manifest_stages = [name for s in stages for name in s.manifest_stages]

    if lab_input is not None and st.button("Run pipeline", type="primary"):
        started = time.monotonic()
        with st.spinner("Running built stages…"):
            run_dir = lab.execute(lab_input, cfg, manifest_stages, tag="pipeline")
        elapsed = time.monotonic() - started
        for s in stages:
            evaluation = PANELS[s.key].evaluate(run_dir, lab_input.truth)
            lab.record_history(f"pipeline:{s.key}", lab_input, preset, {}, evaluation, run_dir, round(elapsed, 2))
        st.session_state["pipeline_last_run"] = (str(run_dir), lab_input.truth)

    if "pipeline_last_run" not in st.session_state:
        st.info("Pick an input and run the pipeline.")
        return

    run_dir, truth = st.session_state["pipeline_last_run"]
    run_dir = Path(run_dir)
    manifest = RunManifest.load(run_dir)
    run_cfg = manifest.config

    rows, evaluations = [], {}
    for s in stages:
        evaluation = PANELS[s.key].evaluate(run_dir, truth)
        evaluations[s.key] = evaluation
        seconds = sum(manifest.stages[n].duration_s or 0.0 for n in s.manifest_stages)
        allotment = sum(float(run_cfg["budget"]["stages"].get(k, 0)) for k in s.budget_keys)
        rows.append({"stage": f"{s.number}. {s.title}", "score": evaluation.score, "seconds": round(seconds, 2),
                     "allotment_s": allotment, **{f"fail": evaluation.counts()["fail"]}})
    table = pd.DataFrame(rows)

    video_duration = manifest.stages["ingest"].metrics.get("video", {}).get("duration_s") or 0.0
    total_s = float(table["seconds"].sum())
    scored = table["score"].dropna()
    cols = st.columns(4)
    cols[0].metric("Pipeline score (mean of stages)", f"{scored.mean():.0f} / 100" if len(scored) else "—")
    cols[1].metric("Wall time, built stages", f"{total_s:.1f} s")
    if video_duration:
        projected = total_s * 600 / video_duration
        cols[2].metric("Projected for a 10-min video", f"{projected:.0f} s",
                       help="Linear in video length at this input's resolution.")
    budget_total = float(run_cfg["budget"]["total_s"])
    allotted = float(table["allotment_s"].sum())
    cols[3].metric("§9 allotment for these stages", f"{allotted:.0f} s of {budget_total:.0f} s")

    lab.render_optional_inputs(run_dir)

    st.subheader("Stage scores and time vs. budget")
    st.dataframe(table, hide_index=True, use_container_width=True)
    if video_duration:
        table["projected_10min_s"] = table["seconds"] * 600 / video_duration
        long = table.melt(id_vars=["stage"], value_vars=["projected_10min_s", "allotment_s"],
                          var_name="measure", value_name="seconds")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y("stage:N", title=None), x=alt.X("seconds:Q", title="Seconds for a 10-minute video"),
            color=alt.Color("measure:N", scale=alt.Scale(range=["#0969da", "#d0d7de"])),
            yOffset="measure:N", tooltip=["stage", "measure", alt.Tooltip("seconds:Q", format=".1f")],
        ).properties(height=70 * len(stages)), use_container_width=True)

    last = stages[-1]
    st.subheader(f"Output at the end of the pipeline — Stage {last.number} ({last.title})")
    for s in stages:
        with st.expander(f"Stage {s.number} scorecard", expanded=(s is last)):
            lab.render_scorecard(evaluations[s.key], title=f"Stage {s.number} score")
    PANELS[last.key].render_results(run_dir, truth, evaluations[last.key])

    st.subheader("Pipeline history")
    lab.render_history(f"pipeline:{last.key}")


def main() -> None:
    st.sidebar.title("PS-17 Stage Lab")
    pages = ["Pipeline so far"] + [f"{STATUS_BADGE[s.status]} Stage {s.number}: {s.title}" for s in STAGES]
    choice = st.sidebar.radio("Page", pages, key="page")
    st.sidebar.caption("🟢 built · 🟡 in progress · ⚪ planned")
    st.sidebar.divider()
    if choice == "Pipeline so far":
        pipeline_page()
    else:
        number = int(choice.split("Stage ")[1].split(":")[0])
        stage_page(next(s.key for s in STAGES if s.number == number))


main()
