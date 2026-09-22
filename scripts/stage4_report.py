"""Print a compact Stage 1-4 report for one or more run folders (and optional probe JSON files).

    python scripts/stage4_report.py data/interim/esri_omega data/interim/esri_vggt probe_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.manifest import RunManifest  # noqa: E402


def run_report(run_dir: Path) -> dict:
    from ui.stages import stage4_recon

    manifest = RunManifest.load(run_dir)
    report = json.loads((run_dir / "track_a" / "track_a_report.json").read_text())
    ev = stage4_recon.evaluate(run_dir, None)
    durations = {name: round(rec.duration_s or 0, 1) for name, rec in manifest.stages.items()
                 if rec.status.value == "done"}
    dense, mesh = report.get("dense") or {}, report.get("mesh") or {}
    return {
        "run": run_dir.name,
        "stage_seconds": durations,
        "total_seconds": round(sum(durations.values()), 1),
        "stage4_score": ev.score, "counts": ev.counts(),
        "not_passing": [f"{k.label}: {k.value} ({k.status}) {k.detail}".strip() for k in ev.kpis
                        if k.status in ("warn", "fail")],
        "registered": f"{report.get('registered')}/{report.get('frames_in')}",
        "models": report.get("models"), "models_merged": report.get("models_merged"),
        "frames_added_by_merge": report.get("frames_added_by_merge"),
        "cam_vs_gps_rms_m": report.get("cam_vs_gps_rms_m"), "height_error_pct": report.get("height_error_pct"),
        "gps_refinement": {k: (report.get("gps_refinement") or {}).get(k) for k in (
            "kept", "skipped", "time_offset_s", "gps_rms_before_m", "gps_rms_after_m", "reproj_before_px",
            "reproj_after_px", "registered_before", "registered_after")},
        "dense": {k: dense.get(k) for k in ("engine", "model", "points", "footprint_m2", "anchor_spread_median_pct",
                                             "views_per_point_median", "vggt_seconds", "frames_rejected")},
        "mesh": {k: mesh.get(k) for k in ("mesher", "target_faces", "faces", "faces_per_vertex")},
        "textured": report.get("textured"), "timings_s": report.get("timings_s"),
        "downgrades": report.get("downgrades"),
        **_stage5(run_dir),
    }


def _stage5(run_dir: Path) -> dict:
    if not (run_dir / "export" / "metadata.json").exists():
        return {}
    from ui.stages import stage5_geo_export

    ev = stage5_geo_export.evaluate(run_dir, None)
    meta = json.loads((run_dir / "export" / "metadata.json").read_text())
    return {"stage5": {
        "score": ev.score, "counts": ev.counts(),
        "kpis": [f"{k.label}: {k.value} ({k.status}) {k.detail}".strip() for k in ev.kpis],
        "crs": meta.get("crs"), "coverage": meta.get("coverage"),
        "formats_produced": meta.get("formats_produced"), "formats_failed": meta.get("formats_failed"),
    }}


def main() -> int:
    out = {}
    for arg in sys.argv[1:]:
        path = Path(arg)
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text())
                out[path.parent.name] = {w: {k: v.get(k) for k in ("model", "input_hw", "peak_gpu_gb", "vggt_seconds",
                                                                    "cloud_points", "footprint_m2", "median",
                                                                    "worst_frame_rel_err_pct")}
                                         for w, v in data.get("widths", {}).items()}
            else:
                out[path.name] = run_report(path)
        except Exception as exc:  # noqa: BLE001 - report what failed, keep going
            out[str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
    print("\n===== PASTE EVERYTHING BELOW BACK TO CLAUDE =====")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
