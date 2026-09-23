"""Why did a run's Track A reconstruction not fit its GPS? Read-only; prints one JSON report.

    .venv/bin/python scripts/box_recon_diag.py data/interim/dji47_s3 > dji47_diag.json

Written for DJI_0047 on the box (2026-09-23): camera centres vs GPS 256 m RMS on a straight
414 m pass, GPS-prior mapping produced no model, VGGT anchor spread 42%. It answers:
  * focal: what BA ended on vs the camera-table prior (a drifting focal bends straight passes);
  * shape: SfM camera path vs GPS path, both straightness and length (a bent/folded model);
  * where: the residual along the flight (constant = offset, growing = bend, jump = broken piece);
  * log: the Track A lines about focal, GPS priors, models and merging.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np


def _collinearity(xy: np.ndarray) -> float | None:
    """Second / first principal spread of a 2-D path: 0 = a straight line."""
    if len(xy) < 3:
        return None
    ev = np.sort(np.linalg.eigvalsh(np.cov((xy - xy.mean(0)).T)))[::-1]
    return round(float(math.sqrt(max(ev[1], 0) / max(ev[0], 1e-12))), 4)


def _path(xyz: np.ndarray) -> dict:
    if len(xyz) < 2:
        return {}
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return {"n": len(xyz), "length": round(float(steps.sum()), 2), "chord": round(float(np.linalg.norm(xyz[-1] - xyz[0])), 2),
            "collinearity": _collinearity(xyz[:, :2]), "z_range": round(float(np.ptp(xyz[:, 2])), 2),
            "largest_step": round(float(steps.max()), 2), "median_step": round(float(np.median(steps)), 3)}


def main(run: Path) -> dict:
    out: dict = {"run": run.name}
    m = json.loads((run / "manifest.json").read_text())["stages"]
    pre = m.get("preflight", {}).get("metrics", {})
    out["camera_prior"] = pre.get("camera")
    out["input_check_offset_s"] = (pre.get("recommended") or {}).get("time_offset_s")
    ta = m.get("track_a", {}).get("metrics", {})
    out["track_a"] = {k: ta.get(k) for k in ("registered", "models", "models_merged", "sparse_points", "reproj_px",
                                             "focal_px", "focal_source", "height_above_ground_m", "expected_agl_m",
                                             "height_error_pct", "cam_vs_gps_rms_m", "gps_refinement", "downgrades")}
    geo = m.get("geo", {}).get("metrics", {})
    out["geo"] = {k: geo.get(k) for k in ("scale_model_to_m", "track_collinearity", "ground_constraint", "rms_all_m",
                                          "horizontal_rms_m", "vertical_rms_m", "inliers", "gps_altitude_datum")}

    # SfM camera path (model units) from the sparse model Track A kept.
    sfm = {}
    try:
        import pycolmap

        for sub in ("gps_refined", "merged", "0"):
            d = run / "track_a" / "sparse" / sub
            if (d / "images.bin").is_file() or (d / "images.txt").is_file():
                rec = pycolmap.Reconstruction(str(d))
                imgs = sorted(rec.images.values(), key=lambda im: im.name)
                sfm = {im.name: np.asarray(im.projection_center()) for im in imgs}
                cam = next(iter(rec.cameras.values()))
                out["sfm_model"] = {"folder": sub, "images": len(imgs), "points": len(rec.points3D),
                                    "camera_model": str(cam.model), "width": cam.width, "height": cam.height,
                                    "params": [round(float(p), 4) for p in cam.params]}
                hfov = (out["camera_prior"] or {}).get("hfov_deg")
                if hfov:
                    prior = cam.width / 2 / math.tan(math.radians(hfov / 2))
                    out["sfm_model"]["focal_prior_px"] = round(prior, 1)
                    out["sfm_model"]["focal_vs_prior_pct"] = round(100 * (float(cam.params[0]) / prior - 1), 1)
                break
    except Exception as exc:  # noqa: BLE001
        out["sfm_model"] = f"unreadable: {type(exc).__name__}: {exc}"

    # GPS path (metres, local tangent plane) for the same frames.
    try:
        import pandas as pd

        tel = pd.read_parquet(run / "ingest" / "frame_telemetry.parquet")
        tel["name"] = [f"frame_{int(i):06d}.jpg" for i in tel["frame_index"]]
        tel = tel.dropna(subset=["lat", "lon"])
        lat0, lon0 = float(tel["lat"].mean()), float(tel["lon"].mean())
        e = (tel["lon"] - lon0).to_numpy() * 111320.0 * math.cos(math.radians(lat0))
        n = (tel["lat"] - lat0).to_numpy() * 110540.0
        gps = {nm: np.array([x, y, z]) for nm, x, y, z in zip(tel["name"], e, n, tel["alt_gps"].fillna(0).to_numpy())}
        common = [k for k in sorted(sfm) if k in gps]
        s_xyz, g_xyz = np.array([sfm[k] for k in common]), np.array([gps[k] for k in common])
        out["paths"] = {"frames_in_both": len(common), "sfm_model_units": _path(s_xyz), "gps_m": _path(g_xyz)}
        if len(common) >= 3:
            ratio = out["paths"]["gps_m"]["length"] / max(out["paths"]["sfm_model_units"]["length"], 1e-9)
            out["paths"]["gps_per_model_unit_by_length"] = round(ratio, 4)
            out["paths"]["gps_per_model_unit_by_chord"] = round(
                out["paths"]["gps_m"]["chord"] / max(out["paths"]["sfm_model_units"]["chord"], 1e-9), 4)
    except Exception as exc:  # noqa: BLE001
        out["paths"] = f"unreadable: {type(exc).__name__}: {exc}"

    # Residual along the flight, in tenths of the frame list.
    try:
        import pandas as pd

        res = pd.read_csv(run / "geo" / "camera_residuals.csv").sort_values("frame")
        r = res["residual_m"].to_numpy()
        parts = np.array_split(r, min(10, len(r)))
        out["residual_along_flight_m"] = [round(float(np.median(p)), 1) for p in parts if len(p)]
    except Exception as exc:  # noqa: BLE001
        out["residual_along_flight_m"] = f"unreadable: {type(exc).__name__}: {exc}"

    # Track A's own log lines about focal, priors, models, merging.
    lines = []
    pat = re.compile(r"focal|GPS-prior|gps prior|models?\b|merge|registered|degrad|downgrade", re.I)
    for log in [run / "run.jsonl", Path(f"{run.name}.log")]:
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                if pat.search(line) and "feature_extraction.cc" not in line:
                    lines.append(line[:220])
    out["log"] = lines[-40:]
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/interim/dji47_s3")), indent=2, default=str))
