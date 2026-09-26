"""GPS alignment and metric checks shared by the reconstruction tracks.

Everything here measures a reconstruction against telemetry; nothing changes it.
Georeferencing proper (CRS, geoid, the Sim(3) that is written out) is Stage 5.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6378137.0
# 35 mm-equivalent focal lengths are defined against a 36 mm-wide frame.
FULL_FRAME_WIDTH_MM = 36.0


def read_geo_enu(path: Path) -> dict[str, np.ndarray]:
    """``geo.txt`` (``EPSG:4326`` header, then ``name lon lat alt``) as local east/north/up metres.

    Equirectangular about the first fix: over a flight of a few kilometres the error
    is centimetres, far below GPS noise, and it keeps this module free of a CRS stack.
    """
    rows = [line.split() for line in Path(path).read_text().splitlines()[1:] if line.strip()]
    if not rows:
        return {}
    lon = np.array([float(r[1]) for r in rows])
    lat = np.array([float(r[2]) for r in rows])
    alt = np.array([float(r[3]) for r in rows])
    east = np.radians(lon - lon[0]) * EARTH_RADIUS_M * math.cos(math.radians(lat[0]))
    north = np.radians(lat - lat[0]) * EARTH_RADIUS_M
    return {r[0]: np.array([east[i], north[i], alt[i] - alt[0]]) for i, r in enumerate(rows)}


def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale, rotation, translation) mapping ``src`` onto ``dst``;
    ``weights`` per pair (GCPs count more than GPS fixes, S5-3)."""
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    cs, cd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd((cd * w[:, None]).T @ cs)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[2, 2] = -1
    rot = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (w @ (cs ** 2).sum(1)))
    return scale, rot, mu_d - scale * rot @ mu_s


def apply(transform: tuple[float, np.ndarray, np.ndarray], points: np.ndarray) -> np.ndarray:
    scale, rot, trans = transform
    return (scale * (rot @ np.asarray(points, dtype=np.float64).T)).T + trans


def focal_from_telemetry(hints: dict[str, float], image_width: int) -> tuple[float | None, str]:
    """Focal length in pixels and where it came from, or ``(None, "self_calibrated")``."""
    if hints.get("hfov_deg"):
        return (image_width / 2) / math.tan(math.radians(hints["hfov_deg"] / 2)), "telemetry_hfov"
    if hints.get("focal_mm_35"):
        return hints["focal_mm_35"] / FULL_FRAME_WIDTH_MM * image_width, "telemetry_focal_35mm"
    return None, "self_calibrated"


def telemetry_hints(telemetry_path: Path | None) -> dict[str, float]:
    """Intrinsics and height hints the telemetry carries, when it carries them.

    ``focal_mm`` from DJI SRT is the 35 mm-equivalent value the OSD shows; KLV has no
    focal length but has the sensor field of view (DEVLOG S1-11).
    """
    import pandas as pd

    hints: dict[str, float] = {}
    if telemetry_path is None or not Path(telemetry_path).exists():
        return hints
    tel = pd.read_parquet(telemetry_path)

    def median(column: str) -> float | None:
        if column in tel and tel[column].notna().any():
            value = float(tel[column].median())
            return value if math.isfinite(value) and value > 0 else None
        return None

    if (hfov := median("hfov_deg")) is not None:
        hints["hfov_deg"] = hfov
        if "hfov_source" in tel and tel["hfov_source"].notna().any():
            hints["hfov_source"] = str(tel["hfov_source"].dropna().iloc[0])
    if (focal := median("focal_mm")) is not None:
        hints["focal_mm_35"] = focal
    if {"alt_gps", "frame_center_alt"} <= set(tel.columns) and tel["frame_center_alt"].notna().any():
        agl = float((tel["alt_gps"] - tel["frame_center_alt"]).median())
        if math.isfinite(agl) and agl > 0:
            hints["expected_agl_m"] = agl
    return hints


def metric_check(names: list[str], centres: np.ndarray, points: np.ndarray,
                 gps: dict[str, np.ndarray], height_radius_m: float = 30.0):
    """Fit camera centres to GPS and measure the model in metres.

    Returns ``(stats, transform)``; ``transform`` is ``None`` when fewer than three
    frames have a GPS fix (a similarity needs three non-collinear positions).
    """
    matched = [i for i, name in enumerate(names) if name in gps]
    if len(matched) < 3:
        return {"gps_matched_frames": len(matched)}, None
    ref = np.array([gps[names[i]] for i in matched])
    transform = umeyama(centres[matched], ref)
    aligned = apply(transform, centres[matched])
    stats: dict[str, float] = {
        "gps_matched_frames": len(matched),
        "cam_vs_gps_rms_m": round(float(np.sqrt(((aligned - ref) ** 2).sum(1).mean())), 2),
    }
    if len(points):
        pts = apply(transform, points)
        heights = []
        for c in aligned:
            near = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < height_radius_m
            if near.sum() > 20:
                heights.append(c[2] - np.median(pts[near, 2]))
        if heights:
            stats["height_above_ground_m"] = round(float(np.median(heights)), 1)
    return stats, transform


def footprint_m2(points: np.ndarray, transform, cell_m: float = 1.0) -> float | None:
    """Ground area covered by ``points``: occupied east/north cells after GPS alignment."""
    if transform is None or not len(points):
        return None
    east_north = apply(transform, points)[:, :2]
    cells = np.unique(np.floor(east_north / cell_m).astype(np.int64), axis=0)
    return float(len(cells)) * cell_m * cell_m
