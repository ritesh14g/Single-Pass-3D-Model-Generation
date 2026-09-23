"""A one-screen summary of box reports, to paste back instead of the full JSON.

    .venv/bin/python scripts/box_summary.py after_fix_report.json dji47_after.json dji47_diag_after.json

Accepts stage4_report.py output ({run: {...}}) and box_recon_diag.py output ({"run": ..., "paths": ...}).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _run(name: str, r: dict) -> list[str]:
    out = [f"## {name}  total {r.get('total_seconds')} s  stage s {r.get('stage_seconds')}"]
    out.append(f"S4 {r.get('stage4_score')}  registered {r.get('registered')}  models {r.get('models')}  "
               f"cam-vs-GPS {r.get('cam_vs_gps_rms_m')} m  height err {r.get('height_error_pct')}%  "
               f"GPS refine: {(r.get('gps_refinement') or {}).get('kept')}")
    d = r.get("dense") or {}
    out.append(f"dense {d.get('engine')} {d.get('points')} pts  footprint {d.get('footprint_m2')} m2  "
               f"anchor spread {d.get('anchor_spread_median_pct')}%  rejected {d.get('frames_rejected')}")
    for g in r.get("downgrades") or []:
        out.append(f"  downgrade: {g[:150]}")
    s3 = r.get("stage3") or {}
    if s3:
        g, f = s3.get("ground") or {}, s3.get("fill") or {}
        out.append(f"S3 {s3.get('score')}  zones 1/2/3 {g.get('zone1_pct')}/{g.get('zone2_pct')}/{g.get('zone3_pct')}%  "
                   f"coverage {s3.get('measured_coverage_pct')} -> {g.get('coverage_pct')}%  "
                   f"gaps {(s3.get('gaps') or {}).get('count')} / {(s3.get('gaps') or {}).get('total_m2')} m2")
        out.append(f"fill {f.get('source')}: {f.get('frames_anchored')}/{f.get('frames_tried')} anchored, "
                   f"{f.get('fill_points')} pts, held-out {f.get('holdout_error_median_pct')}% "
                   f"({f.get('holdout_error_median_m')} m), resid {f.get('anchor_residual_median_m')} m, "
                   f"{f.get('seconds')} s, reason {f.get('reason')}")
        for fr in s3.get("fill_frames") or []:
            odd = fr.get("status") != "anchored" or not (0.5 < (fr.get("scale") or 1) < 2) \
                or (fr.get("extrapolation_dropped_px") or 0) > 0.3 * (fr.get("region_px") or 1)
            if odd:
                out.append(f"  {fr.get('frame')} {fr.get('status')}: {fr.get('reason')}  s={fr.get('scale')} "
                           f"t={fr.get('shift')} band={fr.get('band_px')} region={fr.get('region_px')} "
                           f"inl={fr.get('inlier_ratio')} extrap_drop={fr.get('extrapolation_dropped_px')} "
                           f"pts={fr.get('points_in_targets')}")
        dropped = sum(fr.get("extrapolation_dropped_px") or 0 for fr in s3.get("fill_frames") or [])
        out.append(f"extrapolation-dropped px, all frames: {dropped}")
        out += [f"  kpi: {k[:160]}" for k in s3.get("kpis") or [] if "(fail)" in k or "(warn)" in k]
    s5 = r.get("stage5") or {}
    if s5:
        out.append(f"S5 {s5.get('score')}  formats {s5.get('formats_produced')}  failed {s5.get('formats_failed')}")
    return out


def _diag(d: dict) -> list[str]:
    ta, sm, p = d.get("track_a") or {}, d.get("sfm_model") or {}, d.get("paths") or {}
    out = [f"## diag {d.get('run')}",
           f"focal {ta.get('focal_px')} px ({ta.get('focal_source')}), prior {sm.get('focal_prior_px') if isinstance(sm, dict) else None}, "
           f"vs prior {sm.get('focal_vs_prior_pct') if isinstance(sm, dict) else None}%  model {sm.get('folder') if isinstance(sm, dict) else sm}",
           f"GPS refine: {(ta.get('gps_refinement') or {}).get('kept')}  priors {(ta.get('gps_refinement') or {}).get('priors')}  "
           f"cam-vs-GPS {ta.get('cam_vs_gps_rms_m')} m  height above ground {ta.get('height_above_ground_m')} m"]
    if isinstance(p, dict):
        for k in ("sfm_model_units", "gps_m"):
            q = p.get(k) or {}
            out.append(f"  path {k}: length {q.get('length')} chord {q.get('chord')} collinearity {q.get('collinearity')}")
        out.append(f"  m per model unit: by length {p.get('gps_per_model_unit_by_length')}, by chord {p.get('gps_per_model_unit_by_chord')}")
    out.append(f"residual along flight (m): {d.get('residual_along_flight_m')}")
    return out


def main(paths: list[str]) -> None:
    for path in paths:
        try:
            text = Path(path).read_text()
            data = json.loads(text[text.find("{"):])   # stage4_report.py prints a banner first
        except Exception as exc:  # noqa: BLE001
            print(f"## {path}: unreadable ({type(exc).__name__}: {exc})")
            continue
        lines = _diag(data) if "paths" in data or "sfm_model" in data else \
            [ln for name, r in data.items() if isinstance(r, dict) for ln in _run(name, r)]
        print(f"# {path}")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
