"""DSM + orthophoto rasters and the coverage measure (§8.2 GeoTIFF; §11 "reports coverage percent").

Rasters are binned from the dense cloud in the local map frame: DSM = per-cell median height
(robust to stray points), orthophoto = per-cell mean colour. Gaps up to ``fill_max_cells``
are filled from the nearest valid cell; larger gaps stay nodata — never invented.

Coverage = share of the ground the cameras actually saw (each frame's footprint projected
onto the ground plane) that the dense cloud covers. It is the "entire visible scene" check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Grid:
    x0: float      # west edge
    y1: float      # north edge
    res: float
    width: int
    height: int

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] - self.x0) / self.res).astype(np.int64)
        row = np.floor((self.y1 - xy[:, 1]) / self.res).astype(np.int64)
        return row, col

    @classmethod
    def around(cls, xy: np.ndarray, res: float, pad: float = 0.0) -> "Grid":
        lo, hi = xy.min(0) - pad, xy.max(0) + pad
        x0, y1 = np.floor(lo[0] / res) * res, np.ceil(hi[1] / res) * res
        return cls(float(x0), float(y1), float(res), int(np.ceil((hi[0] - x0) / res)) + 1,
                   int(np.ceil((y1 - lo[1]) / res)) + 1)


def auto_resolution(local_xyz: np.ndarray, views: np.ndarray, floor_m: float = 0.05) -> float:
    """Raster cell ≈ spacing of unique surface samples: sqrt(occupied area / (points / views))."""
    occupied = len(np.unique(np.floor(local_xyz[:, :2]).astype(np.int64), axis=0))  # 1 m cells
    unique = max(len(local_xyz) / max(float(np.median(views)), 1.0), 1.0)
    return float(max(floor_m, round(np.sqrt(occupied / unique) / 0.05) * 0.05))


def rasterize(local_xyz: np.ndarray, rgb: np.ndarray, res: float, fill_max_cells: int, nodata: float):
    """(grid, dsm float32 [h,w], ortho uint8 [h,w,4] RGBA, filled cell count)."""
    import pandas as pd
    from scipy import ndimage

    grid = Grid.around(local_xyz[:, :2], res)
    row, col = grid.cells(local_xyz[:, :2])
    frame = pd.DataFrame({"cell": row * grid.width + col, "z": local_xyz[:, 2],
                          "r": rgb[:, 0], "g": rgb[:, 1], "b": rgb[:, 2]})
    stats = frame.groupby("cell").agg(z=("z", "median"), r=("r", "mean"), g=("g", "mean"), b=("b", "mean"))
    dsm = np.full(grid.width * grid.height, nodata, np.float32)
    ortho = np.zeros((grid.width * grid.height, 4), np.uint8)
    idx = stats.index.to_numpy()
    dsm[idx] = stats["z"].to_numpy(np.float32)
    ortho[idx, :3] = np.clip(stats[["r", "g", "b"]].to_numpy(), 0, 255).astype(np.uint8)
    ortho[idx, 3] = 255
    dsm, ortho = dsm.reshape(grid.height, grid.width), ortho.reshape(grid.height, grid.width, 4)

    filled = 0
    if fill_max_cells > 0:
        empty = dsm == nodata
        dist, (ri, ci) = ndimage.distance_transform_edt(empty, return_indices=True)
        fill = empty & (dist <= fill_max_cells)
        dsm[fill] = dsm[ri[fill], ci[fill]]
        ortho[fill] = ortho[ri[fill], ci[fill]]
        filled = int(fill.sum())
    return grid, dsm, ortho, filled


def write_geotiffs(folder: Path, grid: Grid, dsm: np.ndarray, ortho: np.ndarray, offset, epsg: int | None,
                   vertical_datum: str, nodata: float) -> tuple[Path, Path]:
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(grid.x0 + offset[0], grid.y1 + offset[1], grid.res, grid.res)
    crs = f"EPSG:{epsg}" if epsg else None
    dsm_path, ortho_path = Path(folder) / "dsm.tif", Path(folder) / "orthophoto.tif"
    with rasterio.open(dsm_path, "w", driver="GTiff", width=grid.width, height=grid.height, count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=nodata, compress="deflate",
                       tiled=True) as dst:
        dst.write(dsm, 1)
        dst.update_tags(VERTICAL_DATUM=vertical_datum, UNITS="metre", CONTENT="digital surface model")
    with rasterio.open(ortho_path, "w", driver="GTiff", width=grid.width, height=grid.height, count=4,
                       dtype="uint8", crs=crs, transform=transform, compress="deflate", tiled=True,
                       photometric="RGB") as dst:
        for band in range(4):
            dst.write(ortho[:, :, band], band + 1)
        dst.update_tags(CONTENT="orthophoto from the dense cloud colours; band 4 = alpha")
    return dsm_path, ortho_path


def camera_footprint(georef, rec, ground_z: float, grid: Grid, max_range_factor: float = 3.0) -> np.ndarray:
    """Boolean raster of ground cells inside at least one camera's view."""
    import cv2

    from src.recon.merge import posed

    mask = np.zeros((grid.height, grid.width), np.uint8)
    for im in posed(rec):
        cam = im.camera
        k_inv = np.linalg.inv(cam.calibration_matrix())
        rot_model = im.cam_from_world().rotation.matrix().T            # world_from_cam, model frame
        centre = georef.to_local(np.asarray(im.projection_center())[None])[0]
        height = centre[2] - ground_z
        if height <= 0:
            continue
        corners = []
        for u, v in ((0, 0), (cam.width, 0), (cam.width, cam.height), (0, cam.height)):
            ray = georef.rotation @ (rot_model @ (k_inv @ np.array([u, v, 1.0])))
            t = (ground_z - centre[2]) / ray[2] if ray[2] < -1e-9 else np.inf
            t = min(t, max_range_factor * height / max(np.linalg.norm(ray), 1e-9))
            corners.append(centre[:2] + t * ray[:2])
        row, col = grid.cells(np.asarray(corners))
        cv2.fillPoly(mask, [np.c_[col, row].astype(np.int32)], 1)
    return mask.astype(bool)


def coverage_percent(local_xyz: np.ndarray, georef, rec, cell_m: float = 1.0) -> dict:
    ground_z = float(np.median(local_xyz[:, 2]))
    grid = Grid.around(local_xyz[:, :2], cell_m, pad=200.0)
    seen = camera_footprint(georef, rec, ground_z, grid)
    covered = np.zeros_like(seen)
    row, col = grid.cells(local_xyz[:, :2])
    covered[row, col] = True
    visible = int(seen.sum())
    return {"coverage_pct": round(100.0 * float((covered & seen).sum()) / max(visible, 1), 1),
            "visible_m2": round(visible * cell_m ** 2), "covered_m2": round(float((covered & seen).sum()) * cell_m ** 2),
            "outside_camera_view_m2": round(float((covered & ~seen).sum()) * cell_m ** 2),
            "ground_plane_z_m": round(ground_z, 2), "cell_m": cell_m}
