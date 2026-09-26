"""Accuracy against an independent reference surface (spec §8.5 "Metrics to compute").

The reference is a DSM GeoTIFF or a LAS/LAZ point cloud (lidar, a survey, or the all-strips
reconstruction in the single-pass simulation) in any CRS. Two measurements, both learned the
hard way on Esri vs USGS 3DEP lidar (DEVLOG 2026-09-22):

  * **Surface-to-surface, not cloud-to-cloud.** Nearest-neighbour distance to a lidar cloud
    did not see an injected 3.8 m shift (vegetation is a volume of points; some point is
    always near). Heights are compared against the reference *surface* under each point.
  * **Horizontal placement by correlation of height maps.** The high-passed DSMs are
    phase-correlated; the peak is the horizontal shift of our model against the reference.

Vertical errors are reported per Stage 3 zone and source (Zone 1 MVS, Zone 2 MVS, Zone 2
anchored fill) and on ground points (our LAS class 2), which is the fairest figure where the
reference has vegetation: canopy heights differ between a lidar first return and photos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging import get_logger

log = get_logger(__name__)


def error_stats(d: np.ndarray) -> dict[str, Any]:
    """Signed errors -> mean (bias), median, RMS, 95th percentile of |d|, NMAD (robust sigma)."""
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"n": 0}
    med = float(np.median(d))
    return {"n": int(len(d)), "mean_m": round(float(d.mean()), 3), "median_m": round(med, 3),
            "rms_m": round(float(np.sqrt(np.mean(d ** 2))), 3),
            "p95_abs_m": round(float(np.percentile(np.abs(d), 95)), 3),
            "nmad_m": round(float(1.4826 * np.median(np.abs(d - med))), 3)}


def cloud_to_cloud(a: np.ndarray, b: np.ndarray, max_dist: float) -> dict[str, Any]:
    """Nearest-neighbour distance from each point of ``a`` to ``b`` (§8.5). Valid between two
    reconstructions of the same surface; against lidar with vegetation it hides shifts (see
    the module docstring), so the report uses it only for reconstruction-vs-reconstruction."""
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(b).query(a, k=1, distance_upper_bound=max_dist)
    found = np.isfinite(dist)
    out = error_stats(dist[found])
    out["matched_pct"] = round(100.0 * float(found.mean()), 1) if len(a) else 0.0
    return out


# -- rasters ---------------------------------------------------------------------------------
class Surface:
    """A height grid in a known CRS: z[row, col], NaN = no data; row 0 is the north edge."""

    def __init__(self, z: np.ndarray, transform, crs, source: str = ""):
        self.z, self.transform, self.crs, self.source = z, transform, crs, source

    @classmethod
    def read(cls, path: Path) -> "Surface":
        import rasterio

        with rasterio.open(path) as src:
            z = src.read(1).astype(np.float64)
            if src.nodata is not None:
                z[z == src.nodata] = np.nan
            return cls(z, src.transform, src.crs, str(path))

    @property
    def res(self) -> float:
        return float(abs(self.transform.a))

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear height at map (x, y); NaN outside or next to no-data."""
        inv = ~self.transform
        col, row = inv * (np.asarray(x, np.float64), np.asarray(y, np.float64))
        col, row = np.asarray(col) - 0.5, np.asarray(row) - 0.5
        c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        fc, fr = col - c0, row - r0
        h, w = self.z.shape
        ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < w) & (r0 + 1 < h)
        out = np.full(len(col), np.nan)
        c, r, a, b = c0[ok], r0[ok], fc[ok], fr[ok]
        z = self.z
        out[ok] = ((1 - a) * (1 - b) * z[r, c] + a * (1 - b) * z[r, c + 1]
                   + (1 - a) * b * z[r + 1, c] + a * b * z[r + 1, c + 1])
        return out


def _vertical_shift(src_crs, dst_crs, x: float, y: float, z: float = 0.0) -> tuple[float, str]:
    """Height offset (dst - src) at one point when both CRSs define heights; 0 otherwise."""
    from pyproj import CRS, Transformer

    s, d = CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs)
    if len(s.sub_crs_list) < 2 and not s.is_vertical:
        return 0.0, f"reference heights taken as-is ({s.name} has no vertical datum)"
    t = Transformer.from_crs(s, d, always_xy=True)
    x2, y2, z2 = t.transform(x, y, z)
    if not np.isfinite(z2):
        return 0.0, "vertical transformation unavailable (PROJ grid missing?): heights taken as-is"
    return float(z2 - z), f"reference heights converted {s.name} -> {d.name}: {z2 - z:+.3f} m here"


def reference_on_grid(path: Path, target: Surface, *, ground_class: int = 2,
                      exclude_classes: tuple[int, ...] = (7, 18)) -> tuple[Surface, Surface | None, dict]:
    """The reference as surfaces on ``target``'s grid and CRS: (DSM, ground DTM or None, info).

    GeoTIFF: resampled (bilinear). LAS/LAZ: highest point per cell (DSM) and lowest class-2
    point per cell (DTM, when the file is classified). Points in ``exclude_classes`` (ASPRS 7 / 18,
    low and high noise) enter neither surface: one bird return would become the cell's "surface".
    """
    path = Path(path)
    info: dict[str, Any] = {"file": str(path)}
    h, w = target.z.shape
    t = target.transform
    cx, cy = t * (w / 2, h / 2)
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio
        from rasterio.warp import Resampling, reproject

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
            src_crs = src.crs
            if nodata is not None:
                data[data == nodata] = np.nan
            out = np.full((h, w), np.nan)
            reproject(data, out, src_transform=src.transform, src_crs=src_crs, src_nodata=np.nan,
                      dst_transform=t, dst_crs=target.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        shift, note = _vertical_shift(src_crs.to_wkt(), target.crs.to_wkt(), *(
            _to_crs(target.crs, src_crs, cx, cy)))
        info.update(kind="raster", crs=str(src_crs), vertical=note)
        return Surface(out + shift, t, target.crs, str(path)), None, info

    import laspy
    from pyproj import CRS, Transformer

    las = laspy.read(str(path))
    src_crs = las.header.parse_crs()
    if src_crs is None:
        raise ValueError(f"{path} has no CRS in its header")
    tr = Transformer.from_crs(src_crs, CRS.from_user_input(target.crs.to_wkt()), always_xy=True)
    x, y = tr.transform(np.asarray(las.x), np.asarray(las.y))
    shift, note = _vertical_shift(src_crs, target.crs.to_wkt(), *_to_crs(target.crs, src_crs, cx, cy))
    z = np.asarray(las.z, np.float64) + shift
    inv = ~t
    col, row = inv * (np.asarray(x), np.asarray(y))
    col, row = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
    cls = np.asarray(las.classification)
    ok = (col >= 0) & (row >= 0) & (col < w) & (row < h) & ~np.isin(cls, list(exclude_classes))
    flat = row[ok] * w + col[ok]
    dsm = np.full(h * w, -np.inf)
    np.maximum.at(dsm, flat, z[ok])
    dsm[~np.isfinite(dsm)] = np.nan
    dtm = None
    ground = ok & (cls == ground_class)
    if ground.any():
        gflat = row[ground] * w + col[ground]
        dtm_flat = np.full(h * w, np.inf)
        np.minimum.at(dtm_flat, gflat, z[ground])
        dtm_flat[~np.isfinite(dtm_flat)] = np.nan
        dtm = Surface(dtm_flat.reshape(h, w), t, target.crs, str(path))
    info.update(kind="point cloud", crs=src_crs.name, vertical=note, points=int(ok.sum()),
                ground_points=int(ground.sum()))
    return Surface(dsm.reshape(h, w), t, target.crs, str(path)), dtm, info


def _to_crs(dst_crs, src_crs, x: float, y: float) -> tuple[float, float]:
    """(x, y) given in ``dst_crs`` expressed in ``src_crs`` (for the vertical-shift probe)."""
    from pyproj import CRS, Transformer

    t = Transformer.from_crs(CRS.from_user_input(dst_crs.to_wkt() if hasattr(dst_crs, "to_wkt") else dst_crs),
                             CRS.from_user_input(src_crs.to_wkt() if hasattr(src_crs, "to_wkt") else src_crs),
                             always_xy=True)
    x2, y2 = t.transform(x, y)
    return float(x2), float(y2)


def surface_shift(ours: Surface, ref: Surface, highpass_m: float = 20.0) -> dict[str, Any]:
    """Horizontal offset of ``ours`` relative to ``ref`` (same grid) by phase correlation of
    high-passed heights: a model placed 3 m east of the truth gives east_m = +3."""
    import cv2

    both = np.isfinite(ours.z) & np.isfinite(ref.z)
    if both.sum() < 400:
        return {"estimated": False, "reason": f"only {int(both.sum())} cells overlap"}
    sigma = max(highpass_m / ours.res, 1.0)

    def prep(z):
        filled = np.where(both, z, np.nanmedian(z[both]))
        hp = filled - cv2.GaussianBlur(filled, (0, 0), sigma)
        return np.where(both, hp, 0.0).astype(np.float64)

    a, b = prep(ours.z), prep(ref.z)
    # Normalised cross-power spectrum; its inverse peaks at the shift of ``a`` against ``b``.
    # (cv2.phaseCorrelate's sub-pixel centroid read a 3 m shift as 2.4 m: integer peak +
    # parabola instead.)
    window = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    cross = np.fft.fft2(a * window) * np.conj(np.fft.fft2(b * window))
    surface = np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-12)).real
    r, c = np.unravel_index(int(np.argmax(surface)), surface.shape)
    h, w = surface.shape

    def refine(m1, m0, p1):
        denom = m1 - 2 * m0 + p1
        return 0.5 * (m1 - p1) / denom if abs(denom) > 1e-12 else 0.0

    dy = r + refine(surface[(r - 1) % h, c], surface[r, c], surface[(r + 1) % h, c])
    dx = c + refine(surface[r, (c - 1) % w], surface[r, c], surface[r, (c + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    response = float(surface[r, c])
    # image x = east, image y = south (row 0 is north)
    return {"estimated": True, "east_m": round(float(dx * ours.res), 3), "north_m": round(float(-dy * ours.res), 3),
            "horizontal_m": round(float(np.hypot(dx, dy) * ours.res), 3), "peak": round(float(response), 3),
            "overlap_cells": int(both.sum()), "note": "peak < 0.05: weak texture, shift unreliable"}


def tile_offsets(ours: Surface, ref: Surface, tile_m: float, max_shift_m: float, highpass_m: float,
                 min_valid: float = 0.5, min_score: float = 0.3) -> list[dict[str, Any]]:
    """Per-tile horizontal offset of ``ours`` against ``ref`` by normalised cross-correlation of
    high-passed heights (the method that recovered an injected 3.6 m shift on Esri vs lidar,
    DEVLOG 2026-09-22), plus the tile's median height offset at that alignment."""
    import cv2

    res = ours.res
    t, pad = max(int(round(tile_m / res)), 8), max(int(round(max_shift_m / res)), 1)
    sigma = max(highpass_m / res, 1.0)

    def highpass(z):
        valid = np.isfinite(z)
        filled = np.where(valid, z, np.nanmedian(z) if valid.any() else 0.0)
        hp = filled - cv2.GaussianBlur(filled, (0, 0), sigma)
        return np.where(valid, hp, 0.0).astype(np.float32), valid

    a_hp, a_ok = highpass(ours.z)
    b_hp, b_ok = highpass(ref.z)
    h, w = ours.z.shape
    out = []
    for r0 in range(pad, h - t - pad + 1, t):
        for c0 in range(pad, w - t - pad + 1, t):
            if a_ok[r0:r0 + t, c0:c0 + t].mean() < min_valid or b_ok[r0 - pad:r0 + t + pad, c0 - pad:c0 + t + pad].mean() < min_valid:
                continue
            score = cv2.matchTemplate(b_hp[r0 - pad:r0 + t + pad, c0 - pad:c0 + t + pad],
                                      a_hp[r0:r0 + t, c0:c0 + t], cv2.TM_CCOEFF_NORMED)
            _, best, _, (mx, my) = cv2.minMaxLoc(score)
            if best < min_score:
                continue

            def sub(m1, m0, p1):
                d = m1 - 2 * m0 + p1
                return 0.5 * (m1 - p1) / d if abs(d) > 1e-9 else 0.0

            fx = sub(score[my, mx - 1], score[my, mx], score[my, mx + 1]) if 0 < mx < score.shape[1] - 1 else 0.0
            fy = sub(score[my - 1, mx], score[my, mx], score[my + 1, mx]) if 0 < my < score.shape[0] - 1 else 0.0
            # ours content found (mx - pad, my - pad) cells away in the reference: ours sits the opposite way
            dx_cells, dy_cells = -(mx + fx - pad), -(my + fy - pad)
            x0, y0 = ours.transform * (c0, r0)
            out.append({"x0": x0, "y1": y0, "x1": x0 + t * res, "y0": y0 - t * res,
                        "east_m": float(dx_cells * res), "north_m": float(-dy_cells * res), "score": float(best)})
    return out


def accuracy_vs_reference(export_dir: Path, reference: Path, qcfg: Any) -> dict[str, Any]:
    """Our export (dsm.tif + cloud.las) against a reference surface; per-zone vertical errors."""
    import laspy

    export_dir = Path(export_dir)
    dsm_path, las_path = export_dir / "dsm.tif", export_dir / "cloud.las"
    if not dsm_path.is_file() or not las_path.is_file():
        raise FileNotFoundError(f"{export_dir} needs dsm.tif and cloud.las")
    ours = Surface.read(dsm_path)
    ref, ref_dtm, info = reference_on_grid(Path(reference), ours,
                                           exclude_classes=tuple(qcfg.get("reference_exclude_classes", [7, 18])))
    result: dict[str, Any] = {"reference": info,
                              "horizontal_shift": surface_shift(ours, ref, float(qcfg.get("highpass_m", 20.0)))}
    result["dsm_vs_reference"] = error_stats(ours.z - ref.z)

    las = laspy.read(str(las_path))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    names = set(las.point_format.dimension_names)
    zone = np.asarray(las.zone) if "zone" in names else np.ones(len(x), np.uint8)
    source = np.asarray(las.source) if "source" in names else np.zeros(len(x), np.uint8)
    classes = np.asarray(las.classification)
    cap = int(qcfg.get("max_points", 2_000_000))
    pick = np.arange(len(x)) if len(x) <= cap else np.random.default_rng(0).choice(len(x), cap, replace=False)
    dz = z[pick] - ref.sample(x[pick], y[pick])
    groups = {"zone1_measured": (zone[pick] == 1) & (source[pick] == 0),
              "zone2_measured": (zone[pick] == 2) & (source[pick] == 0),
              "zone2_fill": source[pick] == 1, "all_points": np.ones(len(pick), bool)}
    result["points_vs_reference_dsm"] = {k: error_stats(dz[m]) for k, m in groups.items()}
    ground = classes[pick] == 2
    if ground.any():
        surface = ref_dtm if ref_dtm is not None else ref
        gz = z[pick][ground] - surface.sample(x[pick][ground], y[pick][ground])
        result["ground_points"] = {"against": "reference ground (class 2)" if ref_dtm is not None else "reference DSM",
                                   **{k: error_stats(gz[m[ground]]) for k, m in groups.items() if k != "zone2_fill"}}
    # Local (relative) accuracy: each tile's own 3D offset removed, so what remains is shape error at
    # the tile's scale -- what a distance measured inside the tile depends on.
    tile_m = float(qcfg.get("local_tile_m", 100.0))
    tiles = tile_offsets(ours, ref, tile_m, float(qcfg.get("local_max_shift_m", 20.0)),
                         float(qcfg.get("local_highpass_m", 10.0)))
    if tiles:
        px, py = x[pick], y[pick]
        resid = np.full(len(pick), np.nan)
        for tile in tiles:
            inside = (px >= tile["x0"]) & (px < tile["x1"]) & (py >= tile["y0"]) & (py < tile["y1"])
            if inside.sum() < 20:
                continue
            dz_t = z[pick][inside] - ref.sample(px[inside] - tile["east_m"], py[inside] - tile["north_m"])
            tile["up_m"] = float(np.nanmedian(dz_t)) if np.isfinite(dz_t).any() else None
            if tile["up_m"] is not None:
                resid[inside] = dz_t - tile["up_m"]
        shifts = np.array([[t_["east_m"], t_["north_m"]] for t_ in tiles])
        horiz = np.hypot(shifts[:, 0], shifts[:, 1])
        ups = np.array([t_["up_m"] for t_ in tiles if t_.get("up_m") is not None])
        result["local"] = {
            "tile_m": tile_m, "tiles": len(tiles),
            "placement": {"horizontal_median_m": round(float(np.median(horiz)), 3),
                          "horizontal_p90_m": round(float(np.percentile(horiz, 90)), 3),
                          "vertical_median_m": round(float(np.median(ups)), 3) if len(ups) else None,
                          "vertical_spread_m": round(float(np.percentile(ups, 90) - np.percentile(ups, 10)), 3)
                          if len(ups) else None},
            "points_after_tile_offset": {k: error_stats(resid[m]) for k, m in groups.items()},
            "tile_offsets": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in t_.items()} for t_ in tiles],
        }
        if ground.any():
            result["local"]["ground_after_tile_offset"] = {k: error_stats(resid[m & ground]) for k, m in groups.items()
                                                          if k != "zone2_fill"}
    result["note"] = ("vertical errors = our point height minus the reference surface under it; zone 3 is never "
                      "measured (no points), so it has no accuracy figure by construction. 'local' removes each "
                      f"{tile_m:g} m tile's own 3D offset: what remains is shape error at that scale")
    return result
