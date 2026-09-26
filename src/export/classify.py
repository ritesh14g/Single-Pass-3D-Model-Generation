"""Ground / above-ground classification of the point cloud for LAS (spec §8.2 "classified"; S5-3).

Progressive morphological filter (Zhang et al. 2003, IEEE TGRS 41(4)): the lowest point per
grid cell is opened (erosion then dilation) with growing windows; a cell that sits above the
opened surface by more than a slope-dependent threshold is an object (building, tree, vehicle),
not terrain. The terrain model (DTM) is interpolated from the cells that survive every window,
and each point is classed by its height above it, with the ASPRS LAS codes:

    2 = ground        within ``ground_threshold_m`` of the DTM
    1 = unclassified  above it (vegetation, buildings, structures: not separated here)
    7 = low noise     more than ``noise_below_m`` under it (stereo outliers below the terrain)
"""

from __future__ import annotations

from typing import Any

import numpy as np

GROUND, UNCLASSIFIED, LOW_NOISE = 2, 1, 7


def ground_classes(xyz: np.ndarray, ccfg: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """(uint8 ASPRS class per point, summary) for points in a metric, z-up frame."""
    from scipy import interpolate, ndimage

    n = len(xyz)
    if n < 10:
        return np.full(n, UNCLASSIFIED, np.uint8), {"method": "too few points"}
    cell = float(ccfg.get("cell_m", 1.0))
    lo = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - lo) / cell).astype(np.int64)
    w, h = int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1
    flat = ij[:, 1] * w + ij[:, 0]
    # A low percentile per cell, not the minimum: stereo on water and reflections leaves clusters
    # of points tens of metres under the surface (Esri: 1% below -26 m on ~10 m terrain).
    order = np.lexsort((xyz[:, 2], flat))
    keys, zs = flat[order], xyz[order, 2]
    first = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    count = np.diff(np.r_[first, len(keys)])
    q = float(ccfg.get("low_percentile", 10.0)) / 100.0
    zmin = np.full(w * h, np.inf)
    zmin[keys[first]] = zs[first + np.floor(q * (count - 1)).astype(np.int64)]
    zmin = zmin.reshape(h, w)
    empty = ~np.isfinite(zmin)
    if empty.all():
        return np.full(n, UNCLASSIFIED, np.uint8), {"method": "no cells"}
    # Empty cells take their nearest neighbour's height so the openings see a continuous surface.
    _, (ri, ci) = ndimage.distance_transform_edt(empty, return_indices=True)
    zmin = zmin[ri, ci]
    # A low outlier is its cell's minimum, and erosion would spread it over the whole window:
    # cells far below their neighbourhood's median are pulled up to it first.
    median = ndimage.median_filter(zmin, size=int(ccfg.get("despike_cells", 11)))
    spikes = zmin < median - float(ccfg.get("noise_below_m", 2.0))
    zmin = np.where(spikes, median, zmin)
    surface = zmin.copy()

    slope, dh0, dh_max = float(ccfg.get("slope", 0.3)), float(ccfg.get("dh0_m", 0.3)), float(ccfg.get("dh_max_m", 3.0))
    max_cells = max(int(round(float(ccfg.get("max_window_m", 24.0)) / cell)), 3)
    objects = np.zeros_like(empty)
    previous, k = 1, 0
    while True:
        size = 2 * (2 ** k) + 1                               # 3, 5, 9, 17, 33, ... cells
        if size > max_cells:
            break
        opened = ndimage.grey_opening(surface, size=(size, size))
        dh = min(dh0 + slope * (size - previous) * cell, dh_max)
        objects |= (surface - opened) > dh
        surface, previous, k = opened, size, k + 1

    ground_cells = ~objects & ~empty
    if ground_cells.sum() < 3:
        return np.full(n, UNCLASSIFIED, np.uint8), {"method": "no ground cells"}
    rows, cols = np.nonzero(ground_cells)
    centres = np.c_[cols + 0.5, rows + 0.5]
    grid_r, grid_c = np.mgrid[0:h, 0:w]
    query = np.c_[grid_c.ravel() + 0.5, grid_r.ravel() + 0.5]
    dtm = interpolate.griddata(centres, zmin[rows, cols], query, method="linear")
    gaps = ~np.isfinite(dtm)                                  # outside the ground cells' hull
    if gaps.any():
        dtm[gaps] = interpolate.griddata(centres, zmin[rows, cols], query[gaps], method="nearest")
    dz = xyz[:, 2] - dtm.reshape(h, w).ravel()[flat]

    classes = np.full(n, UNCLASSIFIED, np.uint8)
    classes[np.abs(dz) <= float(ccfg.get("ground_threshold_m", 0.5))] = GROUND
    classes[dz < -float(ccfg.get("noise_below_m", 2.0))] = LOW_NOISE
    counts = {name: int((classes == code).sum()) for name, code in
              (("ground", GROUND), ("above_ground", UNCLASSIFIED), ("low_noise", LOW_NOISE))}
    return classes, {"method": "progressive morphological filter", "cell_m": cell, "windows": k,
                     "despiked_cells": int(spikes.sum()),
                     "counts": counts, "ground_pct": round(100.0 * counts["ground"] / n, 1)}
