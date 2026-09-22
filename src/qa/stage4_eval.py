"""Stage 4 (Reconstruction, spec §7) evaluation — scorecard behind the Stage Lab panel.

Track A only for now (Track B is not built). KPI groups:

  * Sparse (SfM): registered fraction, model count (did the flight split?),
    reprojection error, mean track length.
  * Metric: focal source, camera-centre RMS against GPS after a similarity fit (the
    PS's <= 1 m target), height above ground against telemetry (the scale check).
  * Dense and mesh: dense engine, points, footprint, mesher, faces per vertex
    (fragmentation, S4-6), texture.
  * Runtime: GPU actually used, downgrades, per-step time.

Status per KPI: pass / warn / fail against ``qa.stage4`` bands, or info. Stage score =
share of scored KPIs passed (warn counts half), the same rule as Stages 1-2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.qa.stage1_eval import FAIL, INFO, PASS, WARN, Kpi, StageEvaluation

DEFAULTS = {
    "registered_fraction_pass": 0.90, "registered_fraction_warn": 0.70, "models_warn": 2,
    "reproj_px_pass": 1.5, "reproj_px_warn": 2.5, "track_length_pass": 3.0, "track_length_warn": 2.0,
    "cam_vs_gps_rms_pass_m": 1.0, "cam_vs_gps_rms_warn_m": 5.0,
    "height_error_pass_pct": 5.0, "height_error_warn_pct": 15.0,
    "faces_per_vertex_pass": 1.8, "faces_per_vertex_warn": 1.4,
    "anchor_spread_pass_pct": 2.0, "anchor_spread_warn_pct": 5.0,
    "frames_anchored_pass": 0.90, "frames_anchored_warn": 0.70,
}


@dataclass
class TrackAOutputs:
    """Track A's report plus the stage folder it came from."""

    report: dict[str, Any]
    stage_dir: Path | None = None
    degradations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, stage_dir: Path | str, degradations: list[dict[str, Any]] | None = None) -> "TrackAOutputs":
        stage_dir = Path(stage_dir)
        path = stage_dir / "track_a_report.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return cls(report=report, stage_dir=stage_dir, degradations=list(degradations or []))


def _band(cfg: Any, key: str) -> float:
    try:
        value = cfg.get_path(f"qa.stage4.{key}")
    except Exception:
        value = None
    return float(DEFAULTS[key] if value is None else value)


def _higher_better(value: float, pass_at: float, warn_at: float) -> str:
    return PASS if value >= pass_at else WARN if value >= warn_at else FAIL


def _lower_better(value: float, pass_at: float, warn_at: float) -> str:
    return PASS if value <= pass_at else WARN if value <= warn_at else FAIL


def evaluate_track_a(outputs: TrackAOutputs, cfg: Any) -> StageEvaluation:
    ev = StageEvaluation(stage="recon")
    r = outputs.report
    if not r:
        ev.kpis.append(Kpi("report", "Track A", "Track A report", None, "track_a_report.json present", FAIL,
                           "no report: Track A did not run or failed before writing it"))
        return ev
    _sparse_kpis(ev, r, cfg)
    _metric_kpis(ev, r, cfg)
    _track_b_kpis(ev, r, cfg)
    _dense_mesh_kpis(ev, r, cfg)
    _runtime_kpis(ev, r, outputs.degradations)
    return ev


def _sparse_kpis(ev: StageEvaluation, r: dict, cfg: Any) -> None:
    g = "Sparse (SfM)"
    frac = float(r.get("registered_fraction", 0.0))
    ev.kpis.append(Kpi("registered_fraction", g, "Frames registered", round(frac, 3),
                       f">= {_band(cfg, 'registered_fraction_pass'):.2f}",
                       _higher_better(frac, _band(cfg, "registered_fraction_pass"), _band(cfg, "registered_fraction_warn")),
                       f"{r.get('registered')} of {r.get('frames_in')} conditioned frames"))
    models = int(r.get("models", 1))
    merged = int(r.get("models_merged", 0))
    unplaced = models - merged  # pieces still separate after the GPS merge
    status = PASS if unplaced <= 1 else WARN if unplaced <= _band(cfg, "models_warn") else FAIL
    detail = ""
    if merged:
        detail = (f"SfM split the flight into {models} pieces; {merged} placed through GPS "
                  f"(+{r.get('frames_added_by_merge', 0)} frames, piece GPS fit {r.get('merge_piece_gps_rms_m')} m)")
    elif models > 1:
        detail = "the flight split into disconnected pieces; only the largest is used"
    ev.kpis.append(Kpi("models", g, "Separate models", unplaced, "1", status, detail))
    reproj = float(r.get("reproj_px", float("nan")))
    ev.kpis.append(Kpi("reproj_px", g, "Mean reprojection error", reproj, f"<= {_band(cfg, 'reproj_px_pass')} px",
                       _lower_better(reproj, _band(cfg, "reproj_px_pass"), _band(cfg, "reproj_px_warn")), unit="px"))
    track = float(r.get("track_length", 0.0))
    ev.kpis.append(Kpi("track_length", g, "Mean track length", track, f">= {_band(cfg, 'track_length_pass')}",
                       _higher_better(track, _band(cfg, "track_length_pass"), _band(cfg, "track_length_warn")),
                       "views per sparse point: more views, better triangulated"))


def _metric_kpis(ev: StageEvaluation, r: dict, cfg: Any) -> None:
    g = "Metric (vs telemetry)"
    source = r.get("focal_source", "self_calibrated")
    ev.kpis.append(Kpi("focal_source", g, "Focal length source", source, "telemetry",
                       PASS if str(source).startswith("telemetry") else WARN,
                       f"{r.get('focal_px')} px" + ("" if str(source).startswith("telemetry")
                                                    else "; self-calibrated focal drifts on nadir flights")))
    if "cam_vs_gps_rms_m" in r:
        rms = float(r["cam_vs_gps_rms_m"])
        ev.kpis.append(Kpi("cam_vs_gps_rms_m", g, "Camera centres vs GPS (RMS)", rms,
                           f"<= {_band(cfg, 'cam_vs_gps_rms_pass_m')} m (PS spatial accuracy)",
                           _lower_better(rms, _band(cfg, "cam_vs_gps_rms_pass_m"), _band(cfg, "cam_vs_gps_rms_warn_m")),
                           f"similarity fit over {r.get('gps_matched_frames')} frames; includes GPS noise and "
                           "telemetry timing error, not only reconstruction error", unit="m"))
    else:
        ev.kpis.append(Kpi("cam_vs_gps_rms_m", g, "Camera centres vs GPS (RMS)", None, "<= 1 m", INFO,
                           "no GPS for the registered frames: the model is scale-free"))
    if "height_error_pct" in r:
        err = float(r["height_error_pct"])
        ev.kpis.append(Kpi("height_error_pct", g, "Height above ground vs telemetry", err,
                           f"within ±{_band(cfg, 'height_error_pass_pct'):.0f}%",
                           _lower_better(abs(err), _band(cfg, "height_error_pass_pct"), _band(cfg, "height_error_warn_pct")),
                           f"{r.get('height_above_ground_m')} m measured vs {r.get('expected_agl_m')} m from telemetry",
                           unit="%"))
    else:
        ev.kpis.append(Kpi("height_error_pct", g, "Height above ground vs telemetry", None, "within ±5%", INFO,
                           "telemetry carries no ground elevation to check scale against"))


def _track_b_kpis(ev: StageEvaluation, r: dict, cfg: Any) -> None:
    """The hybrid: VGGT depth on Track A's cameras (§7.4). Absent in mode A."""
    g = "Track B (VGGT depth)"
    dense = r.get("dense") or {}
    mode = dense.get("mode", "auto")
    if mode == "A":
        ev.kpis.append(Kpi("track_b_used", g, "Track B", "off (mode A)", "-", INFO,
                           "Track A dense only: the accurate preset"))
        return
    used = dense.get("engine") == "vggt_hybrid"
    fell_back = [d for d in (r.get("downgrades") or []) if d.startswith("Track B")]
    ev.kpis.append(Kpi("track_b_used", g, "Track B depth used", used, "yes", PASS if used else WARN,
                       f"{dense.get('model')}: {dense.get('vggt_seconds')} s for {dense.get('frames')} frames"
                       + (f" (VGGT-Omega unavailable: {dense['model_fallback']})" if dense.get("model_fallback") else "")
                       if used
                       else ("fell back to Track A dense: " + fell_back[0].split(": ", 1)[-1] if fell_back else "")))
    if not used:
        return
    anchored = dense.get("frames_anchored", 0) / max(dense.get("frames") or 1, 1)
    ev.kpis.append(Kpi("frames_anchored", g, "Frames with anchored depth", round(anchored, 3),
                       f">= {_band(cfg, 'frames_anchored_pass'):.2f}",
                       _higher_better(anchored, _band(cfg, "frames_anchored_pass"), _band(cfg, "frames_anchored_warn")),
                       f"rejected: {dense.get('frames_rejected') or 'none'}"))
    spread = dense.get("anchor_spread_median_pct")
    if spread is not None:
        ev.kpis.append(Kpi("anchor_spread_pct", g, "Anchor disagreement (median)", spread,
                           f"<= {_band(cfg, 'anchor_spread_pass_pct')}%",
                           _lower_better(float(spread), _band(cfg, "anchor_spread_pass_pct"),
                                         _band(cfg, "anchor_spread_warn_pct")),
                           f"how far the Track A points in a frame disagree with its scaled VGGT depth; "
                           f"{dense.get('anchors_median'):.0f} anchors per frame", unit="%"))
    ev.kpis.append(Kpi("views_per_point", g, "Views confirming each point", dense.get("views_per_point_median"),
                       ">= 2", INFO, f"{dense.get('points_before_consistency', 0):,} back-projected, "
                       f"{dense.get('points', 0):,} kept after the multi-view check"))


def _dense_mesh_kpis(ev: StageEvaluation, r: dict, cfg: Any) -> None:
    g = "Dense and mesh"
    dense = r.get("dense") or {}
    engine = dense.get("engine")
    detail = "sparse model only"
    if engine == "vggt_hybrid":
        detail = f"VGGT depth (518 px) on Track A cameras, window {dense.get('window')}, {dense.get('frames')} frames"
    elif engine:
        detail = f"{dense.get('size')} px, {dense.get('src_images')} source views, {dense.get('frames')} frames"
    ev.kpis.append(Kpi("dense_engine", g, "Dense reconstruction", engine or "none", "any dense engine",
                       PASS if engine else FAIL, detail))
    if engine:
        ev.kpis.append(Kpi("dense_points", g, "Dense points", dense.get("points"), "-", INFO,
                           f"{(dense.get('points') or 0) / max(dense.get('frames') or 1, 1):,.0f} per frame"))
        ev.kpis.append(Kpi("footprint_m2", g, "Ground covered", dense.get("footprint_m2"), "-", INFO,
                           "occupied 1 m cells after GPS alignment" if dense.get("footprint_m2") is not None
                           else "no GPS alignment, so no metric footprint", unit="m²"))
    mesh = r.get("mesh") or {}
    mesher = mesh.get("mesher")
    ev.kpis.append(Kpi("mesher", g, "Mesher", mesher or "none", "openmvs_delaunay",
                       PASS if mesher == "openmvs_delaunay" else WARN if mesher else FAIL,
                       f"{mesh.get('faces', 0):,} faces" if mesher else "no mesh"))
    if mesh.get("faces_per_vertex") is not None:
        fpv = float(mesh["faces_per_vertex"])
        ev.kpis.append(Kpi("faces_per_vertex", g, "Faces per vertex", fpv, f">= {_band(cfg, 'faces_per_vertex_pass')}",
                           _higher_better(fpv, _band(cfg, "faces_per_vertex_pass"), _band(cfg, "faces_per_vertex_warn")),
                           "~2 on a clean surface; ~1 means the mesh is fragments"))
    ev.kpis.append(Kpi("textured", g, "Textured mesh", bool(r.get("textured")), "yes",
                       PASS if r.get("textured") else WARN))


def _runtime_kpis(ev: StageEvaluation, r: dict, degradations: list[dict]) -> None:
    g = "Runtime"
    ev.kpis.append(Kpi("using_cuda", g, "GPU used", bool(r.get("using_cuda")), "yes",
                       PASS if r.get("using_cuda") else WARN, "GPU first, CPU fallback"))
    downgrades = r.get("downgrades") or []
    ev.kpis.append(Kpi("downgrades", g, "Downgrades", len(downgrades), "0", INFO, "; ".join(downgrades)))
    timings = r.get("timings_s") or {}
    total = round(sum(timings.values()), 1)
    frames = max(int(r.get("frames_in") or 1), 1)
    ev.kpis.append(Kpi("track_a_seconds", g, "Track A time", total, "-", INFO,
                       f"{total / frames:.1f} s per frame; slowest: "
                       + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(timings.items(), key=lambda kv: -kv[1])[:3]),
                       unit="s"))
    if degradations:
        ev.kpis.append(Kpi("budget_degradations", g, "Budget degradations", len(degradations), "0", INFO,
                           "; ".join(f"{d.get('action')} ({d.get('details', {}).get('from_px')}→"
                                     f"{d.get('details', {}).get('to_px')} px)" for d in degradations)))


# -- chart data ---------------------------------------------------------------
def timings_frame(outputs: TrackAOutputs) -> pd.DataFrame:
    timings = outputs.report.get("timings_s") or {}
    return pd.DataFrame({"step": list(timings), "seconds": list(timings.values())})


def camera_track(outputs: TrackAOutputs, geo_path: Path | None) -> pd.DataFrame:
    """Registered camera centres (GPS-aligned) next to the GPS fixes, east/north in metres."""
    from src.recon import alignment

    sparse = outputs.stage_dir / "sparse" if outputs.stage_dir else None
    if sparse is None or not sparse.is_dir() or geo_path is None or not Path(geo_path).exists():
        return pd.DataFrame(columns=["east", "north", "source", "frame"])
    import pycolmap

    models = [pycolmap.Reconstruction(str(d)) for d in sorted(sparse.iterdir()) if (d / "images.bin").exists()]
    if not models:
        return pd.DataFrame(columns=["east", "north", "source", "frame"])
    rec = max(models, key=lambda m: m.num_reg_images())
    gps = alignment.read_geo_enu(geo_path)
    names = sorted(im.name for im in rec.images.values() if im.has_pose)
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    centres = np.array([by_name[n].projection_center() for n in names])
    _, transform = alignment.metric_check(names, centres, np.empty((0, 3)), gps)
    rows = [{"east": g[0], "north": g[1], "source": "GPS", "frame": n} for n, g in gps.items()]
    if transform is not None:
        aligned = alignment.apply(transform, centres)
        rows += [{"east": a[0], "north": a[1], "source": "SfM (aligned)", "frame": n} for n, a in zip(names, aligned)]
    return pd.DataFrame(rows)


def save_evaluation(evaluation: StageEvaluation, path: Path | str) -> Path:
    path = Path(path)
    path.write_text(json.dumps(evaluation.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
