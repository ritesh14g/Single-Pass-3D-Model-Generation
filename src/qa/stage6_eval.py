"""Stage 6 (Viewer & QA, §8.4–8.5) scorecard.

Three groups:

  * Viewer: the package exists, scene.glb re-opens with the layers the page needs, and the
    overlays are available (confidence is the §11 requirement; zones need Stage 3).
  * QA report: written, every earlier stage scored, accuracy against a reference per zone
    (Zone 1 and Zone 2 separately, §8.5), benchmarks present.
  * §11 definition of done, measured from the run rather than ticked by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.qa.stage1_eval import FAIL, INFO, PASS, WARN, Kpi, StageEvaluation

DEFAULTS = {"scene_mb_warn": 150.0, "zone1_rms_pass_m": 1.0, "zone1_rms_warn_m": 3.0,
            "shift_pass_m": 1.0, "shift_warn_m": 3.0, "projected_budget_s": 900.0}


@dataclass
class QaOutputs:
    run_dir: Path
    qa_dir: Path
    manifest: dict[str, Any]
    report: dict[str, Any]
    scene: dict[str, Any]
    accuracy: dict[str, Any]

    @classmethod
    def load(cls, run_dir: Path | str) -> "QaOutputs":
        run_dir = Path(run_dir)
        qa = run_dir / "qa"

        def rj(path: Path) -> dict:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

        return cls(run_dir, qa, rj(run_dir / "manifest.json"), rj(qa / "report.json"), rj(qa / "viewer" / "scene.json"),
                   rj(qa / "accuracy.json"))


def _band(cfg: Any, key: str) -> float:
    try:
        value = cfg.get_path(f"qa.stage6.{key}")
    except Exception:  # noqa: BLE001
        value = None
    return float(DEFAULTS[key] if value is None else value)


def _lower(value: float, pass_at: float, warn_at: float) -> str:
    return PASS if value <= pass_at else WARN if value <= warn_at else FAIL


def check_scene_glb(path: Path) -> tuple[bool, str]:
    """Re-open scene.glb: the attributes the page reads must be there."""
    if not path.is_file():
        return False, "missing"
    try:
        import pygltflib

        gltf = pygltflib.GLTF2().load(str(path))
        attrs = gltf.meshes[0].primitives[0].attributes
        missing = [a for a in ("POSITION", "_CONFIDENCE", "_ZONE") if getattr(attrs, a, None) is None]
        if missing:
            return False, f"attributes missing: {missing}"
        count = gltf.accessors[attrs.POSITION].count
        faces = gltf.accessors[gltf.meshes[0].primitives[0].indices].count // 3
        return True, f"{count:,} vertices, {faces:,} faces, texture {'yes' if gltf.images else 'no'}"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {type(exc).__name__}: {exc}"


def evaluate_qa(outputs: QaOutputs, cfg: Any) -> StageEvaluation:
    ev = StageEvaluation(stage="viewer_qa")
    _viewer_kpis(ev, outputs, cfg)
    _report_kpis(ev, outputs, cfg)
    _done_kpis(ev, outputs, cfg)
    return ev


def _viewer_kpis(ev: StageEvaluation, o: QaOutputs, cfg: Any) -> None:
    grp = "Viewer (§8.4)"
    folder = o.qa_dir / "viewer"
    built = (folder / "index.html").is_file() and (folder / "vendor" / "three.module.min.js").is_file()
    ev.kpis.append(Kpi("viewer_built", grp, "Viewer package", built, "index.html + scene.glb + three.js",
                       PASS if built else FAIL, str(folder) if built else "not written (see qa failures)"))
    if not built:
        return
    ok, detail = check_scene_glb(folder / "scene.glb")
    ev.kpis.append(Kpi("scene_glb", grp, "scene.glb re-opened", ok, "position + confidence + zone layers",
                       PASS if ok else FAIL, detail))
    layers = o.scene.get("layers") or {}
    ev.kpis.append(Kpi("layer_confidence", grp, "Confidence overlay", bool(layers.get("confidence")), "available",
                       PASS if layers.get("confidence") else FAIL, "§11: the viewer toggles the confidence overlay"))
    ev.kpis.append(Kpi("layer_zones", grp, "Zone overlay + gaps", bool(layers.get("zones")), "available",
                       PASS if layers.get("zones") else WARN,
                       f"{len(o.scene.get('gaps') or [])} gap outlines" if layers.get("zones")
                       else "Stage 3 did not run for this export"))
    ev.kpis.append(Kpi("layer_texture", grp, "Photo texture", bool(layers.get("texture")), "available",
                       PASS if layers.get("texture") else WARN,
                       "" if layers.get("texture") else "untextured mesh (OpenMVS texturing did not run)"))
    size = (folder / "scene.glb").stat().st_size / 1e6 if (folder / "scene.glb").is_file() else 0.0
    ev.kpis.append(Kpi("scene_mb", grp, "Download size", round(size, 1), f"<= {_band(cfg, 'scene_mb_warn'):g} MB",
                       PASS if size <= _band(cfg, "scene_mb_warn") else WARN,
                       f"{(o.scene.get('mesh') or {}).get('faces', 0):,} faces", unit="MB"))


def _report_kpis(ev: StageEvaluation, o: QaOutputs, cfg: Any) -> None:
    grp = "QA report (§8.5)"
    r = o.report
    ev.kpis.append(Kpi("report", grp, "QA report", bool(r), "report.json + report.html", PASS if r else FAIL,
                       str(o.qa_dir / "report.html") if r else "not written"))
    if not r:
        return
    cards = r.get("scorecards") or {}
    scored = [k for k, v in cards.items() if v.get("score") is not None]
    ran = [k for k, v in cards.items() if "did not run" not in str(v.get("skipped", ""))]
    ev.kpis.append(Kpi("stages_scored", grp, "Earlier stages scored", f"{len(scored)}/{len(ran)}", "all that ran",
                       PASS if len(scored) == len(ran) else WARN,
                       "; ".join(f"{k}: {v['skipped']}" for k, v in cards.items() if v.get("skipped"))))
    acc = o.accuracy
    if not acc:
        ev.kpis.append(Kpi("reference", grp, "Accuracy vs reference", None, "per zone", INFO,
                           "no reference surface given (--reference lidar DSM/LAS)"))
    else:
        pts = acc.get("points_vs_reference_dsm") or {}
        z1, z2 = pts.get("zone1_measured") or {}, pts.get("zone2_measured") or {}
        if z1.get("n"):
            ev.kpis.append(Kpi("zone1_rms_m", grp, "Zone 1 height error vs reference (absolute, RMS)", z1["rms_m"],
                               f"<= {_band(cfg, 'zone1_rms_pass_m'):g} m",
                               _lower(z1["rms_m"], _band(cfg, "zone1_rms_pass_m"), _band(cfg, "zone1_rms_warn_m")),
                               f"bias {z1['mean_m']} m, NMAD {z1['nmad_m']} m, n={z1['n']:,}", unit="m"))
        local = (acc.get("local") or {}).get("points_after_tile_offset") or {}
        l1, l2 = local.get("zone1_measured") or {}, local.get("zone2_measured") or {}
        if l1.get("n"):
            tile = (acc.get("local") or {}).get("tile_m")
            ev.kpis.append(Kpi("zone1_local_rms_m", grp, f"Zone 1 height error within {tile:g} m tiles (RMS)",
                               l1["rms_m"], f"<= {_band(cfg, 'zone1_rms_pass_m'):g} m",
                               _lower(l1["rms_m"], _band(cfg, "zone1_rms_pass_m"), _band(cfg, "zone1_rms_warn_m")),
                               f"NMAD {l1['nmad_m']} m after removing each tile's 3D offset (placement "
                               f"{(acc['local']['placement'] or {}).get('horizontal_median_m')} m median)", unit="m"))
            if l2.get("n"):
                ranked = l1["nmad_m"] <= l2["nmad_m"]
                ev.kpis.append(Kpi("zones_rank_accuracy", grp, "Zone 1 more accurate than Zone 2 (vs reference)",
                                   ranked, "yes", PASS if ranked else WARN,
                                   f"NMAD zone 1 {l1['nmad_m']} m vs zone 2 {l2['nmad_m']} m: the zones "
                                   f"{'do' if ranked else 'do NOT'} rank trustworthiness correctly"))
        ev.kpis.append(Kpi("zone2_reported", grp, "Zone 2 accuracy reported separately", bool(z2.get("n")), "yes",
                           PASS if z2.get("n") else INFO,
                           f"RMS {z2.get('rms_m')} m (fill: {(pts.get('zone2_fill') or {}).get('rms_m')} m)"
                           if z2.get("n") else "no Zone 2 points in this export"))
        sh = acc.get("horizontal_shift") or {}
        placement = ((acc.get("local") or {}).get("placement")) or {}
        if placement.get("horizontal_median_m") is not None:
            value = placement["horizontal_median_m"]
            ev.kpis.append(Kpi("horizontal_shift_m", grp, "Horizontal placement vs reference (median of tiles)", value,
                               f"<= {_band(cfg, 'shift_pass_m'):g} m",
                               _lower(value, _band(cfg, "shift_pass_m"), _band(cfg, "shift_warn_m")),
                               f"{acc['local']['tiles']} tiles of {acc['local']['tile_m']:g} m, 90th percentile "
                               f"{placement.get('horizontal_p90_m')} m; tile heights {placement.get('vertical_median_m')} m "
                               f"(spread {placement.get('vertical_spread_m')} m)", unit="m"))
        elif sh.get("estimated"):
            ev.kpis.append(Kpi("horizontal_shift_m", grp, "Horizontal placement vs reference", sh["horizontal_m"],
                               f"<= {_band(cfg, 'shift_pass_m'):g} m",
                               _lower(sh["horizontal_m"], _band(cfg, "shift_pass_m"), _band(cfg, "shift_warn_m")),
                               f"east {sh['east_m']} m, north {sh['north_m']} m, peak {sh['peak']}", unit="m"))
    bench = r.get("benchmarks") or {}
    ev.kpis.append(Kpi("degradation", grp, "Degradation table", "degradation" in bench, "present", INFO,
                       bench.get("degradation", {}).get("folder", "not run (python -m src.cli bench degrade)")))
    ev.kpis.append(Kpi("single_pass", grp, "Single-pass simulation", "single_pass" in bench, "present", INFO,
                       bench.get("single_pass", {}).get("folder", "not run (needs a multi-strip survey)")))


def _done_kpis(ev: StageEvaluation, o: QaOutputs, cfg: Any) -> None:
    from src.stages import built_manifest_stages

    grp = "§11 definition of done"
    stages = o.manifest.get("stages") or {}
    # Skipped by design (Track B runs inside track_a; refine_ba not built) is not a failure.
    needed = [s for s in built_manifest_stages() if s != "qa"]
    missing = [s for s in needed if (stages.get(s) or {}).get("status") not in ("done", "skipped")]
    done = [s for s in needed if (stages.get(s) or {}).get("status") == "done"]
    ev.kpis.append(Kpi("end_to_end", grp, "Every built stage completed", not missing, "all",
                       PASS if not missing else FAIL,
                       f"not done: {missing}" if missing else f"{len(done)} ran, {len(needed) - len(done)} skipped by design"))
    video = ((stages.get("ingest") or {}).get("metrics") or {}).get("video") or {}
    total = sum(float(rec.get("duration_s") or 0.0) for name, rec in stages.items()
                if rec.get("status") == "done" and name != "qa")
    duration = float(video.get("duration_s") or 0.0)
    if duration > 0:
        projected = total * 600.0 / duration
        budget = _band(cfg, "projected_budget_s")
        ev.kpis.append(Kpi("projected_10min_s", grp, "Projected time for a 10-min video", round(projected / 60, 1),
                           f"<= {budget / 60:g} min", PASS if projected <= budget else
                           WARN if projected <= 2 * budget else FAIL,
                           f"{total:,.0f} s for {duration:,.0f} s of video (linear projection; device "
                           f"{(o.manifest.get('environment') or {}).get('device')})", unit="min"))
    r = o.report
    produced = ((r.get("formats") or {}).get("produced")) or []
    ev.kpis.append(Kpi("six_formats", grp, "All six formats", f"{len(produced)}/6", "6/6",
                       PASS if len(produced) >= 6 else WARN, ", ".join(produced)))
    g = r.get("georeferencing") or {}
    ev.kpis.append(Kpi("rms_reported", grp, "Camera-centre RMS vs GPS reported", g.get("rms_all_m") is not None,
                       "reported", PASS if g.get("rms_all_m") is not None else FAIL, f"{g.get('rms_all_m')} m"))
    gaps = (o.run_dir / "export" / "gaps.geojson").is_file()
    cov = (r.get("coverage") or {}).get("coverage_pct")
    ev.kpis.append(Kpi("coverage_and_gaps", grp, "Coverage % + unobserved regions flagged", bool(cov is not None and gaps),
                       "both", PASS if cov is not None and gaps else WARN,
                       f"coverage {cov}%, gaps.geojson {'yes' if gaps else 'no (Stage 3 did not run)'}"))
    acc = (o.accuracy.get("points_vs_reference_dsm") or {}) if o.accuracy else {}
    both = bool((acc.get("zone1_measured") or {}).get("n")) and bool((acc.get("zone2_measured") or {}).get("n"))
    ev.kpis.append(Kpi("zone_accuracy", grp, "Separate accuracy for well / thinly observed zones", both, "both",
                       PASS if both else WARN, "" if both else "needs a reference surface (--reference)"))
