"""The consolidated QA report (spec §8.5): one JSON + one self-contained HTML page per run.

It gathers what every stage already measured rather than measuring anything new:

  * the scorecard of every built stage (the same evaluators the Stage Lab panels use),
  * wall-clock time per stage against the §9 budget,
  * georeferencing residuals, coverage and the zone split,
  * accuracy against a reference surface, per zone (``src.qa.metrics``), when one was given,
  * the degradation table and the single-pass result, when those benchmarks were run.

The HTML is what goes in front of judges and the team: no external assets, readable offline.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Any, Callable

from src.core.logging import get_logger

log = get_logger(__name__)


# -- scorecards --------------------------------------------------------------------------------
def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _scorers() -> dict[str, Callable[[Path], Any]]:
    """Stage key -> run_dir -> StageEvaluation (mirrors ui/stages/*.evaluate without Streamlit)."""
    from src.core.config import Config
    from src.core.manifest import RunManifest

    def s0(run_dir):
        from src.qa.stage0_eval import evaluate_input, load_report

        report = load_report(run_dir)
        return evaluate_input(report) if report is not None else None

    def s1(run_dir):
        from src.qa.stage1_eval import IngestOutputs, evaluate_ingest

        m = RunManifest.load(run_dir)
        return evaluate_ingest(IngestOutputs.load(run_dir / "ingest", m.stages["ingest"].metrics), Config(m.config))

    def s2(run_dir):
        from src.qa.stage2_eval import ConditionOutputs, evaluate_condition

        m = RunManifest.load(run_dir)
        return evaluate_condition(ConditionOutputs.load(run_dir / "condition"), Config(m.config), None)

    def s3(run_dir):
        from src.qa.stage3_eval import FusionOutputs, evaluate_fusion

        outputs = FusionOutputs.load(run_dir)
        return evaluate_fusion(outputs, Config(outputs.config)) if outputs.report else None

    def s4(run_dir):
        from src.qa.stage4_eval import TrackAOutputs, evaluate_track_a

        m = RunManifest.load(run_dir)
        record = m.stages.get("track_a")
        return evaluate_track_a(TrackAOutputs.load(run_dir / "track_a", list(record.degradations) if record else []),
                                Config(m.config))

    def s5(run_dir):
        from src.qa.stage5_eval import ExportOutputs, evaluate_export

        m = RunManifest.load(run_dir)
        outputs = ExportOutputs.load(run_dir)
        return evaluate_export(outputs, Config(m.config)) if outputs.metadata else None

    return {"input_check": s0, "ingest": s1, "condition": s2, "occlusion": s3, "recon": s4, "geo_export": s5}


STAGE_DIRS = {"input_check": "preflight", "ingest": "ingest", "condition": "condition", "occlusion": "fusion",
              "recon": "track_a", "geo_export": "export"}


def stage_scorecards(run_dir: Path) -> dict[str, dict[str, Any]]:
    """{stage key: evaluation dict | {"skipped": why}} for every stage before Stage 6."""
    from src.stages import STAGES

    run_dir = Path(run_dir)
    out: dict[str, dict[str, Any]] = {}
    scorers = _scorers()
    for spec in STAGES:
        if spec.key not in scorers:
            continue
        head = {"number": spec.number, "title": spec.title, "spec_ref": spec.spec_ref}
        if not (run_dir / STAGE_DIRS[spec.key]).is_dir():
            out[spec.key] = {**head, "skipped": "did not run in this run"}
            continue
        try:
            ev = scorers[spec.key](run_dir)
            out[spec.key] = {**head, **ev.to_dict()} if ev is not None else {**head, "skipped": "no outputs"}
        except Exception as exc:  # noqa: BLE001 - one broken scorecard never hides the others
            out[spec.key] = {**head, "skipped": f"scorecard failed: {type(exc).__name__}: {exc}"}
    return out


# -- the report --------------------------------------------------------------------------------
def build_report(run_dir: Path, qa_dir: Path, cfg: Any, *, accuracy: dict | None = None,
                 benchmarks: dict[str, Any] | None = None, viewer: dict | None = None) -> dict[str, Any]:
    run_dir, qa_dir = Path(run_dir), Path(qa_dir)
    qa_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(run_dir)
    meta_path = run_dir / "export" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    stages = manifest.get("stages", {})
    seconds = {name: round(float(rec.get("duration_s") or 0.0), 1) for name, rec in stages.items()
               if rec.get("status") == "done" and name != "qa"}
    budget = cfg.get_path("budget", {})
    g = meta.get("georeferencing") or {}
    report = {
        "run": run_dir.name, "written": dt.datetime.now().isoformat(timespec="seconds"),
        "video": (manifest.get("inputs") or {}).get("video"),
        "preset": (manifest.get("config") or {}).get("preset"),
        "device": (manifest.get("environment") or {}).get("device"),
        "time": {"stage_seconds": seconds, "total_s": round(sum(seconds.values()), 1),
                 "video_s": ((stages.get("ingest") or {}).get("metrics") or {}).get("video", {}).get("duration_s"),
                 "budget_s": budget.get("total_s") if budget else None,
                 "stage_budgets_s": dict((budget or {}).get("stages", {}))},
        "georeferencing": {k: g.get(k) for k in ("rms_all_m", "rms_inliers_m", "horizontal_rms_m", "vertical_rms_m",
                                                 "inliers", "cameras")} | {"crs": meta.get("crs"),
                                                                           "gcp": (g.get("gcp") or None)},
        "coverage": meta.get("coverage"), "zones": meta.get("zones"),
        "formats": {"produced": meta.get("formats_produced"), "failed": meta.get("formats_failed")},
        "scorecards": stage_scorecards(run_dir),
        "accuracy_vs_reference": accuracy, "benchmarks": benchmarks or {}, "viewer": viewer,
        "warnings": {name: rec.get("warnings") for name, rec in stages.items() if rec.get("warnings")},
    }
    report["limitations"] = limitations(report)
    (qa_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (qa_dir / "report.html").write_text(render_html(report), encoding="utf-8")
    return report


def limitations(report: dict[str, Any]) -> list[str]:
    """Honest limitations (§12) that this run's numbers support; stated first, not hidden."""
    out = []
    t = report["time"]
    if t.get("budget_s") and t.get("video_s"):
        projected = t["total_s"] * 600.0 / float(t["video_s"])
        if projected > float(t["budget_s"]):
            out.append(f"Speed: {t['total_s'] / 60:.1f} min for {float(t['video_s']) / 60:.1f} min of video projects to "
                       f"{projected / 60:.0f} min for a 10-minute video against the §9 target of "
                       f"{float(t['budget_s']) / 60:.0f} min (device {report.get('device')}; S4-8).")
    g = report["georeferencing"]
    if g.get("rms_all_m") is not None and float(g["rms_all_m"]) > 1.0:
        out.append(f"Camera centres agree with GPS to {g['rms_all_m']} m RMS: absolute position is limited by the "
                   "telemetry; relative distances within Zone 1 are more reliable.")
    z = report.get("zones") or {}
    if z.get("zone3_pct"):
        out.append(f"{z['zone3_pct']}% of the ground the cameras saw was never observed (Zone 3): flagged in "
                   "gaps.geojson and the viewer, never filled, excluded from accuracy.")
    if not report.get("accuracy_vs_reference"):
        out.append("No independent reference surface was given for this run: accuracy is agreement with the "
                   "telemetry, not with ground truth (pass --reference with a lidar DSM or LAS).")
    if not (report.get("benchmarks") or {}).get("degradation"):
        out.append("The degradation benchmark was not run for this report (python -m src.cli bench degrade).")
    if not (report.get("benchmarks") or {}).get("single_pass"):
        out.append("The single-pass simulation needs a multi-strip survey; not run for this report.")
    return out


# -- HTML ----------------------------------------------------------------------------------------
_CSS = """
:root{--bg:#ffffff;--text:#1f2328;--muted:#59636e;--border:#d1d9e0;--soft:#f6f8fa;--pass:#1a7f37;--warn:#9a6700;
--fail:#cf222e;--info:#59636e;--accent:#0969da}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0d1117;--text:#e6edf3;--muted:#9198a1;
--border:#3d444d;--soft:#151b23;--pass:#3fb950;--warn:#d29922;--fail:#f85149;--info:#9198a1;--accent:#4493f8}}
:root[data-theme="dark"]{--bg:#0d1117;--text:#e6edf3;--muted:#9198a1;--border:#3d444d;--soft:#151b23;--pass:#3fb950;
--warn:#d29922;--fail:#f85149;--info:#9198a1;--accent:#4493f8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,
"Segoe UI",Roboto,sans-serif}main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--border)}
h3{font-size:15px;margin:18px 0 6px}.muted{color:var(--muted)}.cards{display:grid;
grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:16px}.card{border:1px solid var(--border);
border-radius:8px;padding:10px 12px;background:var(--soft)}.card .k{color:var(--muted);font-size:13px}.card .v{font-size:22px;
font-weight:600;font-variant-numeric:tabular-nums}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--border);vertical-align:top}th{color:var(--muted);
font-weight:600}th.num,td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.st{font-weight:600;
text-transform:uppercase;font-size:12px}.pass{color:var(--pass)}.warn{color:var(--warn)}.fail{color:var(--fail)}
.info{color:var(--info)}.bar{height:10px;background:var(--accent);border-radius:3px;min-width:2px}.over{background:var(--fail)}
ul.lim li{margin:4px 0}details summary{cursor:pointer;padding:4px 0}code{background:var(--soft);padding:1px 4px;border-radius:4px}
"""


def _e(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _num(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}{unit}"
    except (TypeError, ValueError):
        return _e(value)


def _stats_row(label: str, s: dict | None) -> str:
    s = s or {}
    if not s.get("n"):
        return f"<tr><td>{_e(label)}</td><td class='num' colspan='6'>no points</td></tr>"
    return (f"<tr><td>{_e(label)}</td><td class='num'>{s['n']:,}</td><td class='num'>{_num(s['mean_m'])}</td>"
            f"<td class='num'>{_num(s['median_m'])}</td><td class='num'>{_num(s['rms_m'])}</td>"
            f"<td class='num'>{_num(s['nmad_m'])}</td><td class='num'>{_num(s['p95_abs_m'])}</td></tr>")


_STATS_HEAD = ("<tr><th>Points</th><th class='num'>n</th><th class='num'>bias (m)</th><th class='num'>median</th>"
               "<th class='num'>RMS</th><th class='num'>NMAD</th><th class='num'>95% |err|</th></tr>")
_GROUP_LABELS = {"zone1_measured": "Zone 1 · measured (MVS)", "zone2_measured": "Zone 2 · measured (thin)",
                 "zone2_fill": "Zone 2 · anchored monocular fill", "all_points": "All points"}


def render_html(r: dict[str, Any]) -> str:
    g, cov, zones, t = r["georeferencing"], r.get("coverage") or {}, r.get("zones") or {}, r["time"]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
             f"content='width=device-width,initial-scale=1'><title>QA Report</title><style>{_CSS}</style></head><body><main>",
             f"<h1>QA report · {_e(Path(str(r.get('video') or r['run'])).name)}</h1>",
             f"<div class='muted'>Run <code>{_e(r['run'])}</code> · preset {_e(r.get('preset'))} · device "
             f"{_e(r.get('device'))} · written {_e(r['written'])}</div>"]
    acc = r.get("accuracy_vs_reference") or {}
    z1 = ((acc.get("points_vs_reference_dsm") or {}).get("zone1_measured") or {})
    cards = [("Processing time", f"{t['total_s'] / 60:.1f} min" if t.get("total_s") else "—"),
             ("Cameras vs GPS (RMS)", _num(g.get("rms_all_m"), 2, " m")),
             ("Coverage of visible ground", _num(cov.get("coverage_pct"), 1, "%")),
             ("Zone 1 / 2 / 3", " / ".join(_num(zones.get(f"zone{k}_pct"), 0, "%") for k in (1, 2, 3)) if zones else "—"),
             ("Formats written", f"{len((r.get('formats') or {}).get('produced') or [])} / 6")]
    if z1.get("n"):
        cards.append(("Zone 1 vs reference (RMS)", _num(z1.get("rms_m"), 2, " m")))
    if (g.get("gcp") or {}).get("check_rms_m") is not None:
        cards.append(("GCP check points (RMS)", _num(g["gcp"]["check_rms_m"], 2, " m")))
    parts.append("<div class='cards'>" + "".join(f"<div class='card'><div class='k'>{_e(k)}</div><div class='v'>{v}</div></div>"
                                                 for k, v in cards) + "</div>")

    parts.append("<h2>Limitations (stated first)</h2><ul class='lim'>" +
                 "".join(f"<li>{_e(x)}</li>" for x in r.get("limitations") or ["None found in this run."]) + "</ul>")

    # scorecards
    parts.append("<h2>Stage scorecards</h2><div class='table-wrap'><table><tr><th>Stage</th><th class='num'>Score</th>"
                 "<th class='num'>pass</th><th class='num'>warn</th><th class='num'>fail</th><th>Note</th></tr>")
    for key, sc in r["scorecards"].items():
        c = sc.get("counts") or {}
        parts.append(f"<tr><td>Stage {sc['number']}: {_e(sc['title'])} <span class='muted'>({_e(sc['spec_ref'])})</span></td>"
                     f"<td class='num'>{_num(sc.get('score'), 1)}</td><td class='num pass'>{c.get('pass', '')}</td>"
                     f"<td class='num warn'>{c.get('warn', '')}</td><td class='num fail'>{c.get('fail', '')}</td>"
                     f"<td class='muted'>{_e(sc.get('skipped', ''))}</td></tr>")
    parts.append("</table></div>")
    for key, sc in r["scorecards"].items():
        if not sc.get("kpis"):
            continue
        rows = "".join(f"<tr><td>{_e(k['label'])}</td><td class='num'>{_e(k['value'])}{_e(' ' + k['unit']) if k.get('unit') else ''}</td>"
                       f"<td class='st {k['status']}'>{_e(k['status'])}</td><td class='muted'>{_e(k.get('target'))}</td>"
                       f"<td class='muted'>{_e(k.get('detail'))}</td></tr>" for k in sc["kpis"])
        parts.append(f"<details><summary>Stage {sc['number']}: {_e(sc['title'])} — {len(sc['kpis'])} KPIs</summary>"
                     f"<div class='table-wrap'><table><tr><th>KPI</th><th class='num'>Value</th><th>Status</th><th>Target</th>"
                     f"<th>Detail</th></tr>{rows}</table></div></details>")

    # time
    budgets = t.get("stage_budgets_s") or {}
    peak = max(list(t["stage_seconds"].values()) + [1.0])
    parts.append("<h2>Time per stage</h2><div class='table-wrap'><table><tr><th>Stage</th><th class='num'>Seconds</th>"
                 "<th class='num'>Budget</th><th style='width:45%'></th></tr>")
    for name, sec in t["stage_seconds"].items():
        allot = budgets.get(name) or budgets.get({"track_a": "track_a_mvs"}.get(name, name))
        over = " over" if allot and sec > float(allot) else ""
        parts.append(f"<tr><td>{_e(name)}</td><td class='num'>{sec:,.1f}</td><td class='num'>{_e(allot)}</td>"
                     f"<td><div class='bar{over}' style='width:{100 * sec / peak:.1f}%'></div></td></tr>")
    parts.append(f"<tr><th>Total</th><th class='num'>{t['total_s']:,.1f}</th><th class='num'>{_e(t.get('budget_s'))}</th><th></th></tr>"
                 "</table></div>")

    # accuracy
    parts.append("<h2>Accuracy</h2>")
    parts.append(f"<p>Camera centres vs GPS after the similarity fit: <b>{_num(g.get('rms_all_m'))} m</b> RMS "
                 f"(horizontal {_num(g.get('horizontal_rms_m'))} m, vertical {_num(g.get('vertical_rms_m'))} m; "
                 f"{_e(g.get('inliers'))} of {_e(g.get('cameras'))} cameras inliers). This measures agreement with the "
                 f"telemetry, not truth.</p>")
    if acc:
        ref = acc.get("reference") or {}
        sh = acc.get("horizontal_shift") or {}
        parts.append(f"<h3>Against the reference surface</h3><p class='muted'>{_e(ref.get('file'))} · {_e(ref.get('kind'))} · "
                     f"{_e(ref.get('vertical'))}</p>")
        if sh.get("estimated"):
            weak = " — weak peak, one shift does not fit this model: see the per-tile placement below"                 if float(sh.get("peak") or 0) < 0.05 else ""
            parts.append(f"<p>Horizontal placement, one shift for the whole model: {_num(sh['horizontal_m'])} m "
                         f"(east {_num(sh['east_m'])}, north {_num(sh['north_m'])}; correlation peak "
                         f"{_num(sh['peak'], 3)}{weak}).</p>")
        parts.append("<div class='table-wrap'><table>" + _STATS_HEAD + "".join(
            _stats_row(_GROUP_LABELS.get(k, k), v) for k, v in (acc.get("points_vs_reference_dsm") or {}).items())
                     + "</table></div>")
        loc = acc.get("local")
        if loc:
            pl = loc["placement"]
            parts.append(f"<h3>Local accuracy: each {loc['tile_m']:g} m tile's own offset removed ({loc['tiles']} tiles)</h3>"
                         f"<p>Placement per tile: median {_num(pl['horizontal_median_m'])} m horizontally "
                         f"(90th percentile {_num(pl['horizontal_p90_m'])} m), heights {_num(pl['vertical_median_m'])} m "
                         f"(spread {_num(pl['vertical_spread_m'])} m between tiles). What remains below is shape error "
                         f"at that scale, which a distance measured inside a tile depends on.</p>"
                         "<div class='table-wrap'><table>" + _STATS_HEAD + "".join(
                             _stats_row(_GROUP_LABELS.get(k, k), v) for k, v in loc["points_after_tile_offset"].items())
                         + "</table></div>")
        gp = acc.get("ground_points")
        if gp:
            parts.append(f"<h3>Ground points only, absolute (against {_e(gp.get('against'))})</h3><div class='table-wrap'><table>"
                         + _STATS_HEAD + "".join(_stats_row(_GROUP_LABELS.get(k, k), v) for k, v in gp.items()
                                                 if isinstance(v, dict)) + "</table></div>")
        parts.append(f"<p class='muted'>{_e(acc.get('note'))}</p>")
    else:
        parts.append("<p class='muted'>No reference surface given for this run.</p>")

    # benchmarks
    b = r.get("benchmarks") or {}
    deg = b.get("degradation")
    if deg:
        parts.append("<h2>Degradation benchmark</h2><p class='muted'>Each degradation applied to a clean clip and "
                     "reconstructed with and without Stage 2 conditioning. Error = surface deviation from the clean "
                     "reconstruction (RMS of DSM differences) and camera-centre error against the clean GPS.</p>")
        parts.append(deg.get("table_html", ""))
    sp = b.get("single_pass")
    if sp:
        parts.append("<h2>Single-pass simulation</h2>")
        parts.append(sp.get("summary_html", ""))
    parts.append("</main></body></html>")
    return "".join(parts)
