"""GPS-to-video time offset and GPS-prior refinement inputs (§7.3 / §8.1 step 4; DEVLOG S4-1).

On Esri the camera-vs-GPS residual was 7.6 m. Measured causes:
  * a constant metadata-to-video lag: the SfM-vs-GPS horizontal RMS has one clean minimum at
    about -1.7 s, stable before and after refinement;
  * SfM drift: cameras and ground bent together by ~20 m in height over 800 m of level flight,
    in steps at weak links between frames.
Estimating the lag and re-mapping with GPS priors (loose horizontally, tight vertically) took
the residual to 3.4 m; what remains matches this file's whole-second KLV timestamps.

Frame names are ``frame_<index:06d>.jpg`` (Stage 2); their video times come from Stage 1's
``frames.parquet``; the GPS track is Stage 2's filtered telemetry.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.recon.alignment import EARTH_RADIUS_M, apply, umeyama


class GpsTrack:
    """A GPS track (time, lat, lon, alt) interpolated at arbitrary times."""

    def __init__(self, t: np.ndarray, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray):
        order = np.argsort(t)
        self.t, self.lat, self.lon, self.alt = (np.asarray(v, dtype=np.float64)[order] for v in (t, lat, lon, alt))
        self._lat0, self._lon0, self._alt0 = self.lat[0], self.lon[0], self.alt[0]

    @classmethod
    def load(cls, telemetry_path: Path) -> "GpsTrack | None":
        import pandas as pd

        tel = pd.read_parquet(telemetry_path)
        tel = tel.dropna(subset=["t", "lat", "lon", "alt_gps"])
        return cls(tel["t"], tel["lat"], tel["lon"], tel["alt_gps"]) if len(tel) >= 2 else None

    def lonlatalt(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.float64)
        return np.c_[np.interp(times, self.t, self.lon), np.interp(times, self.t, self.lat),
                     np.interp(times, self.t, self.alt)]

    def enu(self, times: np.ndarray) -> np.ndarray:
        """Local east/north/up metres about the first fix (equirectangular: cm-level over km)."""
        lla = self.lonlatalt(times)
        east = np.radians(lla[:, 0] - self._lon0) * EARTH_RADIUS_M * math.cos(math.radians(self._lat0))
        north = np.radians(lla[:, 1] - self._lat0) * EARTH_RADIUS_M
        return np.c_[east, north, lla[:, 2] - self._alt0]


def frame_times(frames_path: Path) -> dict[str, float]:
    import pandas as pd

    frames = pd.read_parquet(frames_path)
    return {f"frame_{int(i):06d}.jpg": float(t) for i, t in zip(frames["index"], frames["timestamp_s"])}


def horizontal_rms(centres: np.ndarray, gps: np.ndarray) -> float:
    """Horizontal RMS after a similarity fit (GPS heights ignored: they are the least trusted axis)."""
    flat = np.c_[gps[:, :2], np.zeros(len(gps))]
    res = apply(umeyama(centres, flat), centres) - flat
    return float(np.sqrt((res[:, :2] ** 2).sum(1).mean()))


def estimate_offset(centres: np.ndarray, times: np.ndarray, track: GpsTrack, search_s: float,
                    step_s: float) -> dict[str, Any]:
    """Time shift of the GPS track that best matches the SfM camera path."""
    shifts = np.arange(-search_s, search_s + step_s / 2, step_s)
    rms = np.array([horizontal_rms(centres, track.enu(times + dt)) for dt in shifts])
    best = int(np.argmin(rms))
    at_zero = float(rms[int(np.argmin(np.abs(shifts)))])
    return {"best_s": round(float(shifts[best]), 3), "rms_at_best_m": round(float(rms[best]), 3),
            "rms_at_zero_m": round(at_zero, 3), "at_search_edge": best in (0, len(shifts) - 1),
            "improvement_pct": round(100 * (1 - rms[best] / max(at_zero, 1e-9)), 1)}


def write_geo(path: Path, names: list[str], times: dict[str, float], track: GpsTrack, offset_s: float) -> Path:
    """``geo.txt`` (EPSG:4326, name lon lat alt) for these frames at their times + ``offset_s``."""
    keep = [n for n in names if n in times]
    lla = track.lonlatalt(np.array([times[n] for n in keep]) + offset_s)
    lines = ["EPSG:4326"] + [f"{n} {lon:.9f} {lat:.9f} {alt:.3f}" for n, (lon, lat, alt) in zip(keep, lla)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def write_pose_priors(database: Path, times: dict[str, float], track: GpsTrack, offset_s: float,
                      sigma_h: float, sigma_v: float) -> int:
    """GPS positions as COLMAP pose priors: anisotropic, because horizontal GPS carries the timing
    error while the altitude of a level flight is known to its quantisation (KLV: 0.30 m)."""
    import pycolmap

    handle = pycolmap.Database.open(str(database))
    try:
        handle.clear_pose_priors()
        cov = np.diag([sigma_h ** 2, sigma_h ** 2, sigma_v ** 2])
        count = 0
        for image in handle.read_all_images():
            if image.name not in times:
                continue
            prior = pycolmap.PosePrior()
            prior.position = track.enu(np.array([times[image.name] + offset_s]))[0]
            prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            prior.position_covariance = cov
            prior.corr_data_id = image.data_id
            handle.write_pose_prior(prior)
            count += 1
        return count
    finally:
        handle.close()
