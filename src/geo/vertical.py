"""Which vertical datum is the telemetry altitude in? Measured, not assumed (S5-1).

GPS receivers report altitude either above the WGS84 ellipsoid or above mean sea level
(the geoid), and drone logs rarely say which. The two differ by the geoid undulation N:
-26 m in Florida, -86 m at Bengaluru. Assuming wrong puts the whole model that far up or down.

The test: the telemetry's altitude of the *takeoff ground* (a DJI log's home record, or
absolute minus height-above-takeoff) is compared with a terrain model's height there (a local
DEM, or public terrain tiles). Orthometric: altitude = terrain. Ellipsoidal: altitude = terrain + N.
The test decides when one hypothesis fits within ``tolerance_m`` and |N| is large enough to tell
them apart; otherwise it says it could not, and the configured assumption stands (flagged).

It also repairs a log whose "altitude" is really height above takeoff (DJI_0047's AirData CSV:
``OSD.altitude`` = ``OSD.height``): absolute = home altitude + height, or terrain at takeoff +
height when the log has no home altitude.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging import get_logger

log = get_logger(__name__)
GEOID_EPSG = {"EGM96": 5773, "EGM2008": 3855}


def geoid_undulation(lon: float, lat: float, model: str = "EGM96") -> float | None:
    """N (m) = ellipsoidal height - orthometric height; None when the geoid grid is unavailable."""
    import pyproj
    from pyproj import Transformer

    vertical = GEOID_EPSG.get(str(model).upper())
    if vertical is None:
        return None
    try:
        pyproj.network.set_network_enabled(True)
        t = Transformer.from_crs("EPSG:4979", f"EPSG:4326+{vertical}", always_xy=True, only_best=True)
        h = float(t.transform(lon, lat, 0.0)[2])
    except Exception:  # noqa: BLE001 - no grid, no network
        return None
    # Without the grid pyproj silently passes heights through: N would read as exactly 0.
    return -h if math.isfinite(h) and abs(h) > 1e-6 else None


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _fetch_tile(url: str, cache: Path, timeout_s: float) -> bytes | None:
    """One terrain tile, from the cache or the network. None when neither has it."""
    if cache.is_file():
        return cache.read_bytes()
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310 - fixed https URL template
            data = response.read()
    except Exception as exc:  # noqa: BLE001 - offline is a normal operating mode
        log.info("terrain tile unavailable (%s): %s", url, exc)
        return None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    except OSError:
        pass
    return data


def _sample(dataset, lon: float, lat: float) -> float | None:
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True).transform(lon, lat)
    left, bottom, right, top = dataset.bounds
    if not (left <= x <= right and bottom <= y <= top):
        return None
    value = float(next(dataset.sample([(x, y)]))[0])
    nodata = dataset.nodata
    if not math.isfinite(value) or (nodata is not None and value == nodata) or value < -500:
        return None
    return value


def terrain_height(lon: float, lat: float, dcfg: Any) -> tuple[float | None, str]:
    """Orthometric terrain height at a point: a local DEM file first, then public terrain tiles."""
    import rasterio

    dem_file = dcfg.get("dem_file")
    if dem_file and Path(str(dem_file)).is_file():
        try:
            with rasterio.open(str(dem_file)) as ds:
                value = _sample(ds, lon, lat)
            if value is not None:
                return value, f"DEM {Path(str(dem_file)).name}"
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read DEM %s: %s", dem_file, exc)
    if not bool(dcfg.get("online", False)):
        return None, "no terrain model (datum_check.dem_file unset, online off)"
    zoom = int(dcfg.get("zoom", 12))
    x, y = _tile_xy(lon, lat, zoom)
    url = str(dcfg.get("tile_url")).format(z=zoom, x=x, y=y)
    data = _fetch_tile(url, Path(str(dcfg.get("cache_dir", "data/cache/dem"))) / f"{zoom}_{x}_{y}.tif",
                       float(dcfg.get("timeout_s", 10)))
    if data is None:
        return None, "terrain tiles unreachable (offline?)"
    from rasterio.io import MemoryFile

    try:
        with MemoryFile(data) as mem, mem.open() as ds:
            value = _sample(ds, lon, lat)
    except Exception as exc:  # noqa: BLE001
        return None, f"terrain tile unreadable ({type(exc).__name__})"
    return value, f"terrain tiles z{zoom} ({url.split('/')[2]})"


def decide_datum(ground_alt: float, terrain: float, undulation: float, tolerance_m: float) -> tuple[str | None, str]:
    """(datum or None, explanation) from the takeoff ground's altitude in the telemetry."""
    d_ortho = abs(ground_alt - terrain)
    d_ell = abs(ground_alt - (terrain + undulation))
    numbers = (f"takeoff ground {ground_alt:.1f} m in the telemetry; terrain {terrain:.1f} m (orthometric), "
               f"{terrain + undulation:.1f} m (ellipsoidal, N {undulation:+.1f} m)")
    if abs(undulation) < 2 * tolerance_m:
        return None, f"{numbers}: |N| too small to tell the datums apart at ±{tolerance_m:.0f} m"
    if d_ell <= tolerance_m and d_ell < d_ortho:
        return "ellipsoidal", f"{numbers}: matches ellipsoidal to {d_ell:.1f} m (orthometric off by {d_ortho:.1f} m)"
    if d_ortho <= tolerance_m and d_ortho < d_ell:
        return "orthometric", f"{numbers}: matches orthometric to {d_ortho:.1f} m (ellipsoidal off by {d_ell:.1f} m)"
    return None, f"{numbers}: neither fits within {tolerance_m:.0f} m (off by {d_ortho:.1f} / {d_ell:.1f} m)"


def resolve_altitude(table: Any, cfg: Any) -> dict[str, Any]:
    """Make ``alt_gps`` absolute where the log allows and measure its datum. Mutates ``table.frame``.

    Returns the record the geo stage uses: {"datum": "ellipsoidal" | "orthometric" | None,
    "method": ..., "detail": ..., numbers}. Never raises: a failed check leaves the assumption.
    """
    vcfg = cfg.get_path("geo.vertical")
    dcfg = vcfg.get("datum_check") or {}
    record: dict[str, Any] = {"datum": None, "method": "not checked"}
    frame = getattr(table, "frame", None)
    if frame is None or len(frame) == 0 or not frame["lat"].notna().any():
        return record
    home = dict(getattr(table, "home", None) or {})
    alt, rel = frame["alt_gps"].to_numpy(float), frame["alt_baro"].to_numpy(float)
    both = np.isfinite(alt) & np.isfinite(rel)
    relative = (not np.isfinite(alt).any() and np.isfinite(rel).any()) or (
        both.sum() >= 3 and float(np.median(np.abs(alt[both] - rel[both]))) < float(dcfg.get("relative_match_m", 1.0)))
    first = frame.dropna(subset=["lat", "lon"]).iloc[0]
    lon, lat = float(home.get("lon", first["lon"])), float(home.get("lat", first["lat"]))
    record.update(home=home or None, where={"lon": round(lon, 7), "lat": round(lat, 7),
                                            "from": "home record" if "lat" in home else "first GPS fix"})

    enabled = bool(dcfg.get("enabled", True))
    terrain, terrain_source = terrain_height(lon, lat, dcfg) if enabled else (None, "datum check disabled")
    record.update(terrain_m=None if terrain is None else round(terrain, 2), terrain_source=terrain_source)

    ground_alt = None
    if relative:
        if "alt" in home and math.isfinite(float(home["alt"])):
            ground_alt = float(home["alt"])
            frame["alt_gps"] = ground_alt + frame["alt_baro"]
            record["relative_fixed"] = f"altitude was height above takeoff: home altitude {ground_alt:.2f} m added"
        elif terrain is not None:
            frame["alt_gps"] = terrain + frame["alt_baro"]
            record.update(datum="orthometric", method="terrain at takeoff",
                          relative_fixed=f"altitude was height above takeoff: terrain {terrain:.1f} m "
                                         f"({terrain_source}) added at the takeoff point")
            record["detail"] = record["relative_fixed"]
            return record
        else:
            record.update(method="relative only",
                          detail="altitude is height above takeoff and neither a home altitude nor a terrain "
                                 "model is available: heights are offset by the takeoff elevation")
            return record
    elif both.any():
        ground_alt = float(np.median(alt[both] - rel[both]))       # absolute minus above-takeoff
    elif "alt" in home and math.isfinite(float(home["alt"])):
        ground_alt = float(home["alt"])
    record["ground_alt_m"] = None if ground_alt is None else round(ground_alt, 2)
    if ground_alt is None or terrain is None:
        record.update(method="assumed", detail=terrain_source if terrain is None
                      else "the log has no takeoff altitude to compare with the terrain")
        return record
    undulation = geoid_undulation(lon, lat, str(vcfg.get("geoid_model", "EGM96")))
    if undulation is None:
        record.update(method="assumed", detail="geoid grid unavailable: cannot compare the datums")
        return record
    datum, detail = decide_datum(ground_alt, terrain, undulation, float(dcfg.get("tolerance_m", 8.0)))
    record.update(datum=datum, method="terrain check" if datum else "undecided", detail=detail,
                  undulation_m=round(undulation, 2))
    return record
