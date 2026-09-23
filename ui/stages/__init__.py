"""Stage Lab panels, keyed by ``src.stages`` stage key.

To add a stage panel: create ``ui/stages/stageN_<key>.py`` implementing

    KEY: str
    render_params(cfg) -> dict[str, Any]          # sidebar widgets -> config overrides
    evaluate(run_dir, truth) -> StageEvaluation   # scorecard (logic lives in src/qa/)
    headline(run_dir) -> dict[str, Any]           # 3-4 numbers for the pipeline view
    render_results(run_dir, truth, evaluation)    # charts / diagnostics

register it in PANELS below, and flip the stage to BUILT in src/stages.py.

Optional: ``existing_runs() -> list[Path]`` and ``rerun(run_dir, cfg) -> Path`` let the page
re-run just this stage on a run that already has its inputs (Stage 3 on a finished Stage 4/5).
Never delete an existing panel: panels are how earlier stages keep getting
re-tested as later stages change shared code.
"""

from __future__ import annotations

from types import ModuleType

from ui.stages import (stage0_input_check, stage1_ingest, stage2_condition, stage3_occlusion, stage4_recon,
                       stage5_geo_export)

PANELS: dict[str, ModuleType] = {
    stage0_input_check.KEY: stage0_input_check,
    stage1_ingest.KEY: stage1_ingest,
    stage2_condition.KEY: stage2_condition,
    stage3_occlusion.KEY: stage3_occlusion,
    stage4_recon.KEY: stage4_recon,
    stage5_geo_export.KEY: stage5_geo_export,
}
