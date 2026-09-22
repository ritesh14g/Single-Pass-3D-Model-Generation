"""Stage 0 (Input check) scorecard — the analyzer's checks as KPIs.

The analyzer already decides PASS / WARN / BLOCK / INFO per check with a fix; this turns
that into the same ``StageEvaluation`` every other stage panel renders. A blocking check
scores as a fail, so the stage score is "how much of this input is fit for the job".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.qa.stage1_eval import FAIL, INFO, PASS, WARN, Kpi, StageEvaluation

STATUS = {"pass": PASS, "warn": WARN, "block": FAIL, "info": INFO}


def evaluate_input(report: Any) -> StageEvaluation:
    """``report`` is an ``InputReport`` or the dict stored in the manifest."""
    checks = report.checks if hasattr(report, "checks") else [_obj(c) for c in (report or {}).get("checks", [])]
    ev = StageEvaluation(stage="input_check", title="Stage 0 — Input check")
    for c in checks:
        detail = c.detail + (f" Fix: {c.fix}" if c.fix and c.status in ("warn", "block") else "")
        ev.kpis.append(Kpi(c.id, c.group, c.label, c.value, "", STATUS.get(c.status, INFO), detail))
    return ev


class _obj:
    def __init__(self, d: dict):
        self.__dict__.update({"id": "", "group": "", "label": "", "status": "info", "value": None,
                              "detail": "", "fix": "", **d})


def checks_frame(report: Any) -> pd.DataFrame:
    checks = report.checks if hasattr(report, "checks") else [_obj(c) for c in (report or {}).get("checks", [])]
    return pd.DataFrame([{"group": c.group, "check": c.label, "status": c.status, "value": str(c.value),
                          "detail": c.detail, "fix": c.fix} for c in checks])


def load_report(run_dir: Path | str):
    """The input report saved by a run, or None."""
    from src.preflight.analyzer import InputReport

    for candidate in (Path(run_dir) / "preflight", Path(run_dir)):
        if (candidate / "input_report.json").exists():
            return InputReport.load(candidate)
    return None
