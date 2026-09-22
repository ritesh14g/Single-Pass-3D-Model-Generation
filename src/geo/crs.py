"""Projected CRS, vertical datum and GPS-to-map transformation (§8.1).

Horizontal: the UTM zone of the flight's mean position (EPSG 326xx north, 327xx south).
Vertical: orthometric heights on EGM96 (EPSG:5773) when the geoid grid is available.

pyproj silently falls back to a ballpark transformation (no geoid) when the grid file is
missing, which would leave ellipsoidal heights labelled as orthometric. ``only_best=True``
makes that an error instead, and the caller records a downgrade.

Input altitude datum: MISB ST 0601 (KLV) defines sensor altitude above mean sea level, so
KLV heights are already orthometric. DJI SRT/CSV heights are treated as WGS84-ellipsoidal,
the spec's assumption (§8.1), and flagged as assumed in the metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GEOID_EPSG = {"EGM96": 5773, "EGM2008": 3855}


@dataclass
class MapFrame:
    """Where exported coordinates live: a UTM zone, a vertical datum, and a local offset."""

    horizontal_epsg: int
    vertical_epsg: int | None           # None = heights as given (ellipsoidal or input datum)
    vertical_datum: str                 # "EGM96 orthometric", "WGS84 ellipsoidal", ...
    geoid_applied: bool
    gps_altitude_datum: str             # "orthometric" | "ellipsoidal"
    gps_altitude_datum_assumed: bool
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: list[str] = field(default_factory=list)

    @property
    def crs_string(self) -> str:
        return f"EPSG:{self.horizontal_epsg}" + (f"+{self.vertical_epsg}" if self.vertical_epsg else "")

    def to_dict(self) -> dict:
        return {
            "crs": self.crs_string, "horizontal_epsg": self.horizontal_epsg, "vertical_epsg": self.vertical_epsg,
            "vertical_datum": self.vertical_datum, "geoid_applied": self.geoid_applied,
            "gps_altitude_datum": self.gps_altitude_datum,
            "gps_altitude_datum_assumed": self.gps_altitude_datum_assumed,
            "offset": list(self.offset), "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapFrame":
        return cls(d["horizontal_epsg"], d.get("vertical_epsg"), d["vertical_datum"], d["geoid_applied"],
                   d["gps_altitude_datum"], d["gps_altitude_datum_assumed"], tuple(d.get("offset", (0, 0, 0))),
                   list(d.get("notes", [])))


def utm_epsg(lon: float, lat: float) -> int:
    """UTM zone EPSG code for a position (Norway/Svalbard exceptions do not matter here)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def gps_altitude_datum(telemetry_source: str, configured: str) -> tuple[str, bool]:
    """(datum, assumed) for the telemetry's altitude."""
    if configured in ("ellipsoidal", "orthometric"):
        return configured, False
    if str(telemetry_source).lower().startswith("klv"):
        return "orthometric", False  # ST 0601 tag 15: sensor true altitude, MSL
    return "ellipsoidal", True


def gps_to_map(lon: np.ndarray, lat: np.ndarray, alt: np.ndarray, *, horizontal_epsg: int,
               geoid_model: str, want_orthometric: bool, gps_datum: str, network: bool = True):
    """Project GPS fixes to (E, N, H). Returns (enh [n,3], vertical_epsg or None, geoid_applied, note)."""
    import pyproj
    from pyproj import Transformer

    if network:
        pyproj.network.set_network_enabled(True)
    vertical = GEOID_EPSG.get(str(geoid_model).upper())
    lon, lat, alt = (np.asarray(v, dtype=np.float64) for v in (lon, lat, alt))
    if want_orthometric and vertical:
        # KLV heights are already orthometric: carry them through on the geoid's vertical CRS.
        source = f"EPSG:4326+{vertical}" if gps_datum == "orthometric" else "EPSG:4979"
        try:
            t = Transformer.from_crs(source, f"EPSG:{horizontal_epsg}+{vertical}", always_xy=True, only_best=True)
            e, n, h = t.transform(lon, lat, alt)
            if np.all(np.isfinite(h)):
                return np.c_[e, n, h], vertical, gps_datum == "ellipsoidal", ""
            note = "geoid transformation returned non-finite heights"
        except Exception as exc:  # noqa: BLE001 - missing grid, no network: fall back, say so
            note = f"{geoid_model} geoid unavailable ({type(exc).__name__}: {str(exc)[:120]})"
    else:
        note = "" if not want_orthometric else f"unknown geoid model {geoid_model!r}"
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{horizontal_epsg}", always_xy=True)
    e, n = t.transform(lon, lat)
    return np.c_[e, n, alt], None, False, note


def map_to_lonlat(e: np.ndarray, n: np.ndarray, horizontal_epsg: int) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    t = Transformer.from_crs(f"EPSG:{horizontal_epsg}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64))
    return np.asarray(lon), np.asarray(lat)
