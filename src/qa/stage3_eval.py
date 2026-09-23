"""Stage 3 (Occluded surface reconstruction, §6) scorecard.

Files are re-opened, not trusted from the report:

  * Zones: the voxel map exists; Zone 1 share of the ground the cameras saw (the accuracy
    backbone); points rejected as not scene surface (a Stage 4 defect Stage 3 has to clean up).
  * Coverage and gaps (§6.4): coverage percent; ``gaps.geojson`` parses, has one feature per
    listed gap and the same total area; share of the mesh with no measured support (inferred).
  * Zone 2 anchoring (§6.3): frames anchored vs refused; **the critical rule re-checked from the
    files** — no filled point lies in a Zone 1 voxel — and every fill weight is below Zone 1's;
    the held-out Zone 2 error (anchored depth vs Zone 1 pixels the fit never saw).
  * Time against the §9 fusion allotment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.qa.stage1_eval import FAIL, INFO, PASS, WARN, Kpi, StageEvaluation

DEFAULTS = {"zone1_pass_pct": 50.0, "zone1_warn_pct": 25.0, "coverage_pass_pct": 90.0, "coverage_warn_pct": 70.0,
            "rejected_pass_pct": 2.0, "rejected_warn_pct": 10.0, "anchored_pass": 0.7, "anchored_warn": 0.4,
            "holdout_pass_pct": 2.0, "holdout_warn_pct": 5.0, "inferred_mesh_warn_pct": 30.0}


@dataclass
class FusionOutputs:
    report: dict[str, Any]
    folder: Path
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: Path | str) -> "FusionOutputs":
        run_dir = Path(run_dir)
        folder = run_dir / "fusion"
        path = folder / "fusion_report.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        manifest = run_dir / "manifest.json"
        config = json.loads(manifest.read_text(encoding="utf-8")).get("config", {}) if manifest.is_file() else {}
        return cls(report, folder, config)

    def zone_map(self) -> dict[str, np.ndarray] | None:
        path = self.folder / "zone_map.npz"
        return dict(np.load(path)) if path.is_file() else None

    def zones_frame(self):
        import pandas as pd

        path = self.folder / "zones.parquet"
        return pd.read_parquet(path) if path.is_file() else pd.DataFrame()

    def frames_frame(self):
        import pandas as pd

        return pd.DataFrame(self.report.get("fill_frames") or [])


def _band(cfg: Any, key: str) -> float:
    try:
        value = cfg.get_path(f"qa.stage3.{key}")
    except Exception:  # noqa: BLE001
        value = None
    return float(DEFAULTS[key] if value is None else value)


def fill_in_zone1(outputs: FusionOutputs) -> tuple[int, int]:
    """(filled points inside a Zone 1 voxel, filled points), recomputed from the files."""
    from src.fusion.stage import read_fill_ply
    from src.fusion.zones import ZONE1, VoxelGrid

    fill_path = outputs.folder / "fill.ply"
    voxels = outputs.zones_frame()
    grid = outputs.report.get("voxel") or {}
    if not fill_path.is_file() or voxels.empty or not grid:
        return 0, 0
    pts = read_fill_ply(fill_path)[0]
    if not len(pts):
        return 0, 0
    vg = VoxelGrid(float(grid["size"]), np.asarray(grid["origin"]), np.asarray(grid["dims"], np.int64))
    zone1 = voxels.loc[voxels["zone"] == ZONE1, "key"].to_numpy()
    return int(np.isin(vg.keys(pts), zone1).sum()), int(len(pts))


def verify_gaps(outputs: FusionOutputs) -> tuple[bool, str]:
    path = outputs.folder / "gaps.geojson"
    if not path.is_file():
        return False, "gaps.geojson missing"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {type(exc).__name__}: {exc}"
    gaps = outputs.report.get("gaps") or {}
    feats = doc.get("features", [])
    area = round(sum(f["properties"]["area_m2"] for f in feats), 1)
    ok = doc.get("type") == "FeatureCollection" and len(feats) == gaps.get("count") and abs(area - gaps.get("total_m2", 0)) < 1
    if ok and feats and "4326" in str((doc.get("properties") or {}).get("crs")):
        ring = np.asarray(feats[0]["geometry"]["coordinates"][0] if feats[0]["geometry"]["type"] == "Polygon"
                          else feats[0]["geometry"]["coordinates"][0][0])
        ok = bool(np.all(np.abs(ring[:, 0]) <= 180) and np.all(np.abs(ring[:, 1]) <= 90))
    return ok, f"{len(feats)} features, {area:,.0f} m², {(doc.get('properties') or {}).get('crs')}"


def evaluate_fusion(outputs: FusionOutputs, cfg: Any) -> StageEvaluation:
    ev = StageEvaluation(stage="occlusion")
    r = outputs.report
    grp = "Zones (§6.1–6.2)"
    if not r:
        ev.kpis.append(Kpi("report", grp, "fusion_report.json", None, "present", FAIL, "Stage 3 did not run"))
        return ev
    units = r.get("units", "m")
    voxel = r.get("voxel") or {}
    ev.kpis.append(Kpi("zone_map", grp, "Zone map persisted", int(voxel.get("voxels", 0)), "> 0 voxels + ground map",
                       PASS if voxel.get("voxels") and outputs.zone_map() is not None else FAIL,
                       f"voxel {voxel.get('size', 0):.3f} {units} (GSD {voxel.get('gsd')}); angles from "
                       f"{voxel.get('angle_source')}; {r.get('cameras')} cameras"))
    ground = r.get("ground") or {}
    z1 = float(ground.get("zone1_pct", 0.0))
    ev.kpis.append(Kpi("zone1_pct", grp, "Well observed (Zone 1) share of visible ground", z1,
                       f">= {_band(cfg, 'zone1_pass_pct'):.0f}%",
                       PASS if z1 >= _band(cfg, "zone1_pass_pct") else WARN if z1 >= _band(cfg, "zone1_warn_pct") else FAIL,
                       f"Zone 2 {ground.get('zone2_pct')}%, Zone 3 {ground.get('zone3_pct')}% of "
                       f"{ground.get('visible_m2', 0):,} m² in view", unit="%"))
    thr = r.get("thresholds") or {}
    ev.kpis.append(Kpi("observation", grp, "Views / triangulation angle (median per voxel)",
                       f"{r.get('views_median')} / {r.get('tri_deg_median')}°",
                       f">= {thr.get('zone1_min_views')} / {thr.get('zone1_min_triangulation_deg')}° for Zone 1", INFO,
                       f"could-see views (z-buffer) median {r.get('views_geom_median')}; photometric consistency "
                       f"{r.get('photometric_median')}"))
    rej = float(r.get("rejected_near_camera_pct", 0.0))
    ev.kpis.append(Kpi("rejected_pct", grp, "Dense points rejected as not surface", rej,
                       f"<= {_band(cfg, 'rejected_pass_pct'):.0f}%",
                       PASS if rej <= _band(cfg, "rejected_pass_pct") else
                       WARN if rej <= _band(cfg, "rejected_warn_pct") else FAIL,
                       f"{r.get('rejected_near_camera', 0):,} points next to the flight path (stereo failures); "
                       "removed from zones and from every export", unit="%"))

    grp = "Coverage and gaps (§6.4)"
    cov = float(r.get("coverage_pct", 0.0))
    ev.kpis.append(Kpi("coverage_pct", grp, "Coverage (Zone 1 + 2 of the ground in view)", cov,
                       f">= {_band(cfg, 'coverage_pass_pct'):.0f}%",
                       PASS if cov >= _band(cfg, "coverage_pass_pct") else
                       WARN if cov >= _band(cfg, "coverage_warn_pct") else FAIL,
                       f"measured only: {r.get('measured_coverage_pct')}%; the rest is flagged, not filled", unit="%"))
    ok, detail = verify_gaps(outputs)
    gaps = r.get("gaps") or {}
    ev.kpis.append(Kpi("gaps_geojson", grp, "gaps.geojson (re-opened)", ok, "valid, matches the report",
                       PASS if ok else FAIL, f"{detail}; largest {gaps.get('largest_m2', 0):,} m², "
                                             f"{gaps.get('interior', 0)} interior"))
    mesh = r.get("mesh")
    if mesh:
        inferred = float(mesh.get("zone3_area_pct", 0.0))
        ev.kpis.append(Kpi("inferred_mesh_pct", grp, "Mesh area without measured support", inferred,
                           f"<= {_band(cfg, 'inferred_mesh_warn_pct'):.0f}% (flagged, shown as inferred)",
                           PASS if inferred <= _band(cfg, "inferred_mesh_warn_pct") else WARN,
                           f"{mesh.get('inferred_faces', 0):,} of {mesh.get('faces', 0):,} faces further than "
                           f"{mesh.get('support_radius')} {units} from any measured voxel", unit="%"))
    else:
        ev.kpis.append(Kpi("inferred_mesh_pct", grp, "Mesh area without measured support", None, "-", INFO,
                           "Stage 4 produced no mesh"))

    grp = "Zone 2 anchoring (§6.3)"
    fill = r.get("fill") or {}
    tried = int(fill.get("frames_tried", 0))
    anchored = int(fill.get("frames_anchored", 0))
    if tried and fill.get("source") not in (None, "none"):
        share = anchored / tried
        ev.kpis.append(Kpi("frames_anchored", grp, "Frames anchored to Zone 1", round(share, 3),
                           f">= {_band(cfg, 'anchored_pass'):.2f}",
                           PASS if share >= _band(cfg, "anchored_pass") else
                           WARN if share >= _band(cfg, "anchored_warn") else FAIL,
                           f"{anchored} of {tried} from {fill.get('source')}; refused: "
                           f"{', '.join(fill.get('refusal_reasons') or []) or 'none'}"
                           + (f"; stopped: {fill['reason']}" if fill.get("reason") else "")))
    else:
        ev.kpis.append(Kpi("frames_anchored", grp, "Frames anchored to Zone 1", 0, "a monocular source", WARN,
                           f"gaps reported, not filled: {fill.get('reason') or 'no source'}"))
    inside, total = fill_in_zone1(outputs)
    ev.kpis.append(Kpi("zone1_untouched", grp, "Monocular depth inside Zone 1 (critical rule)", inside, "0",
                       PASS if inside == 0 else FAIL,
                       f"{total:,} filled points re-checked against the Zone 1 voxels; "
                       f"{fill.get('blocked_by_zone1', 0):,} blocked during fusion"))
    wmax = fill.get("weight_max")
    ev.kpis.append(Kpi("fill_weight", grp, "Largest fill weight (Zone 1 = 1.0)", wmax, "< 1.0",
                       PASS if wmax is None or wmax < 1.0 else FAIL,
                       f"{fill.get('fill_points', 0):,} filled points; {fill.get('below_weight', 0):,} voxels below "
                       "the surface threshold"))
    held = fill.get("holdout_error_median_pct")
    if held is not None:
        ev.kpis.append(Kpi("holdout_error_pct", grp, "Zone 2 error on held-out Zone 1 pixels", held,
                           f"<= {_band(cfg, 'holdout_pass_pct'):.1f}% of depth",
                           PASS if held <= _band(cfg, "holdout_pass_pct") else
                           WARN if held <= _band(cfg, "holdout_warn_pct") else FAIL,
                           f"median {fill.get('holdout_error_median_m')} {units}; anchor fit residual "
                           f"{fill.get('anchor_residual_median_m')} {units}", unit="%"))
    else:
        ev.kpis.append(Kpi("holdout_error_pct", grp, "Zone 2 error on held-out Zone 1 pixels", None, "-", INFO,
                           "no frame was anchored, so there is nothing to hold out"))

    grp = "Time (§9)"
    seconds = float((r.get("timings_s") or {}).get("total", 0.0))
    allot = float(((outputs.config.get("budget") or {}).get("stages") or {}).get("fusion", 120))
    ev.kpis.append(Kpi("seconds", grp, "Stage 3 wall time", seconds, f"<= {allot:.0f} s (fusion allotment)",
                       PASS if seconds <= allot else WARN if seconds <= 2 * allot else FAIL,
                       ", ".join(f"{k} {v}s" for k, v in (r.get("timings_s") or {}).items() if k != "total"), unit="s"))
    return ev
