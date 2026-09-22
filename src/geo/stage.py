"""Stage 5a runner: georeference Track A's sparse model against the filtered GPS."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from src.core.logging import get_logger, log_downgrade, log_event
from src.geo.georef import georeference

log = get_logger(__name__)


def run_geo(sparse_model: Path, geo_path: Path | None, out_dir: Path, cfg: Any, *,
            telemetry_source: str = "") -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    georef = georeference(Path(sparse_model), geo_path, cfg, telemetry_source=telemetry_source)
    path = georef.save(out_dir / "georef.json")
    artifacts: dict[str, Path] = {"georef": path}
    downgrades: list[str] = []
    if not georef.referenced:
        reason = georef.stats.get("reason", "no GPS")
        log_downgrade(log, "georeferencing", "model frame (scale-free, unreferenced)", reason)
        downgrades.append(f"georeferencing -> model frame: {reason}")
        return {"artifacts": artifacts, "metrics": {"referenced": False, "reason": reason,
                                                     "downgrades": downgrades}}
    frame = georef.frame
    for note in frame.notes:
        log_downgrade(log, "orthometric heights", frame.vertical_datum, note)
        downgrades.append(f"orthometric heights -> {frame.vertical_datum}: {note}")
    residuals = out_dir / "camera_residuals.csv"
    with residuals.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "residual_m", "inlier"])
        writer.writeheader()
        writer.writerows(georef.stats["per_camera"])
    artifacts["camera_residuals"] = residuals
    metrics = {k: v for k, v in georef.stats.items() if k != "per_camera"}
    metrics.update(crs=frame.crs_string, vertical_datum=frame.vertical_datum, geoid_applied=frame.geoid_applied,
                   gps_altitude_datum=frame.gps_altitude_datum,
                   gps_altitude_datum_assumed=frame.gps_altitude_datum_assumed, offset=list(frame.offset),
                   scale_model_to_m=round(georef.scale, 6), downgrades=downgrades)
    log_event(log, logging.INFO, "georeferenced", crs=frame.crs_string, rms_all_m=metrics["rms_all_m"],
              inliers=metrics["inliers"], cameras=metrics["cameras"])
    return {"artifacts": artifacts, "metrics": metrics}
