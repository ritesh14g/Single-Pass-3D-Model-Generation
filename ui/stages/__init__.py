"""Stage Lab panels, keyed by ``src.stages`` stage key.

To add a stage panel: create ``ui/stages/stageN_<key>.py`` implementing

    KEY: str
    render_params(cfg) -> dict[str, Any]          # sidebar widgets -> config overrides
    evaluate(run_dir, truth) -> StageEvaluation   # scorecard (logic lives in src/qa/)
    headline(run_dir) -> dict[str, Any]           # 3-4 numbers for the pipeline view
    render_results(run_dir, truth, evaluation)    # charts / diagnostics

register it in PANELS below, and flip the stage to BUILT in src/stages.py.
Never delete an existing panel: panels are how earlier stages keep getting
re-tested as later stages change shared code.
"""

from __future__ import annotations

from types import ModuleType

from ui.stages import stage1_ingest, stage2_condition

PANELS: dict[str, ModuleType] = {
    stage1_ingest.KEY: stage1_ingest,
    stage2_condition.KEY: stage2_condition,
}
