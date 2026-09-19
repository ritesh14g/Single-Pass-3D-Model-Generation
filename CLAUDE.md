# CLAUDE.md — instructions for AI agents working in this repo

1. **Read `DEVLOG.md` first.** It records the stage status board, every file created,
   modified and deleted, and the dead ends that must not be retried.
   Don't re-derive or redo work that's already logged there.
2. **Authoritative spec:** `SIH26158_PS17_BUILD_SPEC.md`. Stages follow its section
   headings (§4 = Stage 1 … §8 = Stages 5–6).
3. **Build stage by stage.** `src/stages.py` is the source of truth for stage status.
   The pipeline runs only BUILT stages by default.
4. **Every built stage has a Stage Lab panel** (`ui/stages/`) with parameters,
   a scorecard (logic in `src/qa/stageN_eval.py`), and diagnostic charts.
   Never delete a panel. `tests/test_ui_smoke.py` must stay green.
5. **After any change, append a DEVLOG session entry** (Created / Modified / Deleted /
   Decisions / Dead ends / Open issues / Next) and update the status board.
6. No hard-coded parameters: everything tunable lives in `configs/default.yaml`.
7. Environment: Windows, `.venv` (Python 3.10). Run tests with
   `.venv\Scripts\python -m pytest tests -q`. Launch the UI with
   `.venv\Scripts\python -m streamlit run ui/app.py`.
8. GPU work (Track A/B, NVDEC, speed measurements) happens on the institute's cloud
   notebook: a **20 GB H100 MIG slice, 3 CPU cores, 56 GB RAM** — follow `CLOUD_GPU_GUIDE.md`
   for setup, data transfer, remote UI and what to record.
9. **GPU first, CPU fallback, never a crash.** Get the device from
   `src/core/device.py::resolve_device`; never hard-code `"cuda"`. Any GPU path must log a
   downgrade and continue on CPU on failure. Size memory for 20 GB and threads for 3 cores
   (`CLOUD_GPU_GUIDE.md` §0).
