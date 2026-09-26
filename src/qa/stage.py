"""Stage 6 runner (manifest stage ``qa``): the web viewer package and the QA report.

    qa/viewer/        the §8.4 viewer (src/viewer/package.py); serve with `python -m src.cli view <run>`
    qa/accuracy.json  accuracy against a reference surface, per zone (only with --reference)
    qa/report.json    every stage's scorecard, times, residuals, accuracy, benchmarks
    qa/report.html    the same, as a self-contained page

Nothing here changes a model; each part is attempted separately, so a missing mesh still
leaves a report and a missing reference still leaves a viewer.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.core.logging import get_logger, log_downgrade, log_event

log = get_logger(__name__)


def load_benchmarks(dirs: list[Path | str]) -> dict[str, Any]:
    """Benchmark results written by `src.cli bench` (degradation.json / single_pass.json)."""
    found: dict[str, Any] = {}
    for d in dirs or []:
        for name, key in (("degradation.json", "degradation"), ("single_pass.json", "single_pass")):
            path = Path(d) / name
            if path.is_file() and key not in found:
                found[key] = json.loads(path.read_text(encoding="utf-8"))
                found[key]["folder"] = str(Path(d))
    return found


def run_qa(run_dir: Path, out_dir: Path, cfg: Any, *, reference: Path | None = None,
           benchmark_dirs: list[Path] | None = None) -> dict[str, Any]:
    from src.qa.metrics import accuracy_vs_reference
    from src.qa.report import build_report
    from src.viewer.package import build_viewer

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qcfg = cfg.get_path("qa")
    export_dir = run_dir / "export"
    artifacts: dict[str, Path] = {}
    failures: dict[str, str] = {}
    timings: dict[str, float] = {}

    def attempt(name: str, fn):
        started = time.perf_counter()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a missing part is reported, never fatal
            failures[name] = f"{type(exc).__name__}: {exc}"
            log_downgrade(log, f"qa {name}", "the rest of the QA stage", failures[name])
            return None
        finally:
            timings[name] = round(time.perf_counter() - started, 1)

    viewer = None
    if bool(qcfg.viewer.enabled):
        summary = {"budget": {"total_s": cfg.get_path("budget.total_s", None)}}
        viewer = attempt("viewer", lambda: build_viewer(export_dir, out_dir / "viewer", qcfg.viewer,
                                                        run_summary=summary))
        if viewer:
            artifacts.update({"viewer": viewer["artifacts"]["viewer"], "viewer_index": viewer["artifacts"]["index"]})

    accuracy = None
    ref = reference or qcfg.get("reference", {}).get("file")
    if ref:
        accuracy = attempt("accuracy", lambda: accuracy_vs_reference(export_dir, Path(str(ref)), qcfg.metrics))
        if accuracy:
            path = out_dir / "accuracy.json"
            path.write_text(json.dumps(accuracy, indent=2, default=str), encoding="utf-8")
            artifacts["accuracy"] = path

    bench_dirs = list(benchmark_dirs or []) + [Path(p) for p in (qcfg.report.get("benchmark_dirs") or [])]
    benchmarks = load_benchmarks(bench_dirs)
    report = attempt("report", lambda: build_report(run_dir, out_dir, cfg, accuracy=accuracy, benchmarks=benchmarks,
                                                    viewer=(viewer or {}).get("metrics")))
    if report:
        artifacts.update({"report_json": out_dir / "report.json", "report_html": out_dir / "report.html"})
    metrics = {
        "viewer": (viewer or {}).get("metrics"), "accuracy": _accuracy_summary(accuracy),
        "benchmarks": sorted(benchmarks), "failures": failures, "timings_s": timings,
        "stage_scores": {k: v.get("score") for k, v in ((report or {}).get("scorecards") or {}).items()},
        "limitations": (report or {}).get("limitations"),
    }
    log_event(log, logging.INFO, "qa finished", parts=sorted(artifacts), failed=sorted(failures))
    return {"artifacts": artifacts, "metrics": metrics}


def _accuracy_summary(acc: dict | None) -> dict | None:
    if not acc:
        return None
    pts = acc.get("points_vs_reference_dsm") or {}
    return {"horizontal_shift_m": (acc.get("horizontal_shift") or {}).get("horizontal_m"),
            **{f"{k}_rms_m": (v or {}).get("rms_m") for k, v in pts.items()}}
