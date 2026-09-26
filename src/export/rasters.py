"""DSM + orthophoto rasters and the coverage measure (§8.2 GeoTIFF; §11 "reports coverage percent").

Rasters are binned from the dense cloud in the local map frame: DSM = per-cell median height
(robust to stray points), orthophoto = per-cell mean colour. Gaps up to ``fill_max_cells``
are filled from the nearest valid cell; larger gaps stay nodata — never invented.

When Stage 4 made a textured mesh, the orthophoto is instead *rendered* from it (S5-2): a
top-down orthographic z-buffer over the mesh's triangles, colour from the texture at each
cell's interpolated UV, at the texture's own ground resolution. Cells no triangle covers stay
transparent.

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


def texture_image(mesh) -> np.ndarray | None:
    """The mesh's texture as an RGB uint8 array, or None when it has no UV texture."""
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    if uv is None or image is None or len(uv) != len(mesh.vertices):
        return None
    return np.asarray(image.convert("RGB"))


def texel_size(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, tex_shape, sample: int = 20000,
               seed: int = 0) -> float | None:
    """Median ground metres per texel: sqrt(horizontal face area / texture area it uses)."""
    if not len(faces):
        return None
    pick = np.random.default_rng(seed).choice(len(faces), min(len(faces), sample), replace=False)
    tri, tuv = vertices[faces[pick]], uv[faces[pick]] * np.array([tex_shape[1], tex_shape[0]])

    def area2(a):
        return 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1])
                            - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))

    ground, texels = area2(tri[:, :, :2]), area2(tuv)
    ok = (ground > 0) & (texels > 0)
    return float(np.sqrt(np.median(ground[ok] / texels[ok]))) if ok.any() else None


def render_ortho(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, texture: np.ndarray, grid: Grid,
                 max_triangle_px: int = 4096, chunk: int = 200_000, empty_color=None,
                 empty_tolerance: int = 24) -> tuple[np.ndarray, float]:
    """Top-down orthographic render of a textured mesh: (RGBA uint8 [h,w,4], share of cells hit).

    Per cell centre, the highest triangle over it wins (what a camera looking straight down sees);
    its colour is the texture at the barycentric-interpolated UV. Triangles spanning more than
    ``max_triangle_px`` cells are skipped: on a Delaunay mesh those are the skirts at the edges,
    not surface. Texels in ``empty_color`` (faces OpenMVS had no photo for) stay transparent.
    """
    w, h = grid.width, grid.height
    px = (vertices[:, 0] - grid.x0) / grid.res - 0.5          # cell centres at integer coordinates
    py = (grid.y1 - vertices[:, 1]) / grid.res - 0.5
    zbuf = np.full(w * h, -np.inf)
    face = np.full(w * h, -1, np.int64)
    b1 = np.zeros(w * h, np.float32)
    b2 = np.zeros(w * h, np.float32)
    for start in range(0, len(faces), chunk):
        f = faces[start:start + chunk]
        fid = np.arange(start, start + len(f))
        x, y, z = px[f], py[f], vertices[f][:, :, 2]
        c0, c1 = np.ceil(x.min(1)).astype(np.int64), np.floor(x.max(1)).astype(np.int64)
        r0, r1 = np.ceil(y.min(1)).astype(np.int64), np.floor(y.max(1)).astype(np.int64)
        c0, c1, r0, r1 = np.maximum(c0, 0), np.minimum(c1, w - 1), np.maximum(r0, 0), np.minimum(r1, h - 1)
        cw, ch = c1 - c0 + 1, r1 - r0 + 1
        det = (y[:, 1] - y[:, 2]) * (x[:, 0] - x[:, 2]) + (x[:, 2] - x[:, 1]) * (y[:, 0] - y[:, 2])
        keep = (cw > 0) & (ch > 0) & (cw * ch <= max_triangle_px) & (np.abs(det) > 1e-12)
        if not keep.any():
            continue
        k = np.flatnonzero(keep)
        counts = (cw * ch)[k]
        tri = np.repeat(k, counts)
        offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        col = c0[tri] + offset % cw[tri]
        row = r0[tri] + offset // cw[tri]
        dx, dy = col - x[tri, 2], row - y[tri, 2]
        l0 = ((y[tri, 1] - y[tri, 2]) * dx + (x[tri, 2] - x[tri, 1]) * dy) / det[tri]
        l1 = ((y[tri, 2] - y[tri, 0]) * dx + (x[tri, 0] - x[tri, 2]) * dy) / det[tri]
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        tri, col, row, l0, l1, l2 = tri[inside], col[inside], row[inside], l0[inside], l1[inside], l2[inside]
        zc = l0 * z[tri, 0] + l1 * z[tri, 1] + l2 * z[tri, 2]
        cell = row * w + col
        # Highest candidate per cell in this chunk, then against what earlier chunks left.
        order = np.lexsort((-zc, cell))
        first = np.r_[True, np.diff(cell[order]) != 0]
        best = order[first]
        cell, zc = cell[best], zc[best]
        better = zc > zbuf[cell]
        cell, best = cell[better], best[better]
        zbuf[cell] = zc[better]
        face[cell] = fid[tri[best]]
        b1[cell], b2[cell] = l1[best], l2[best]
    hit = face >= 0
    ortho = np.zeros((w * h, 4), np.uint8)
    if hit.any():
        f = faces[face[hit]]
        l1, l2 = b1[hit], b2[hit]
        tuv = (1.0 - l1 - l2)[:, None] * uv[f[:, 0]] + l1[:, None] * uv[f[:, 1]] + l2[:, None] * uv[f[:, 2]]
        th, tw = texture.shape[:2]
        tc = np.clip(np.round(tuv[:, 0] * (tw - 1)).astype(np.int64), 0, tw - 1)
        tr = np.clip(np.round((1.0 - tuv[:, 1]) * (th - 1)).astype(np.int64), 0, th - 1)   # OBJ v runs up
        colour = texture[tr, tc, :3]
        opaque = np.full(len(colour), 255, np.uint8)
        if empty_color is not None:
            empty = np.abs(colour.astype(np.int16) - np.asarray(empty_color, np.int16)).max(axis=1) <= empty_tolerance
            opaque[empty] = 0
            colour[empty] = 0
        ortho[hit, :3] = colour
        ortho[hit, 3] = opaque
    return ortho.reshape(h, w, 4), float((ortho[:, 3] > 0).mean())


def support_mask(dsm: np.ndarray, nodata: float, dsm_grid: Grid, grid: Grid, dilate_cells: int) -> np.ndarray:
    """Ortho cells within ``dilate_cells`` DSM cells of measured surface (the dense cloud). A Delaunay
    mesh spans the convex hull; outside the cloud its texture is stretched guesswork."""
    import cv2

    valid = (dsm != nodata).astype(np.uint8)
    if dilate_cells > 0:
        valid = cv2.dilate(valid, np.ones((2 * dilate_cells + 1, 2 * dilate_cells + 1), np.uint8))
    cols = ((grid.x0 + (np.arange(grid.width) + 0.5) * grid.res - dsm_grid.x0) / dsm_grid.res).astype(np.int64)
    rows = ((dsm_grid.y1 - (grid.y1 - (np.arange(grid.height) + 0.5) * grid.res)) / dsm_grid.res).astype(np.int64)
    ok_c = (cols >= 0) & (cols < dsm_grid.width)
    ok_r = (rows >= 0) & (rows < dsm_grid.height)
    out = np.zeros((grid.height, grid.width), bool)
    out[np.ix_(ok_r, ok_c)] = valid[np.ix_(rows[ok_r], cols[ok_c])] > 0
    return out


def write_ortho(path: Path, grid: Grid, ortho: np.ndarray, offset, epsg: int | None, content: str) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", width=grid.width, height=grid.height, count=4, dtype="uint8",
                       crs=f"EPSG:{epsg}" if epsg else None,
                       transform=from_origin(grid.x0 + offset[0], grid.y1 + offset[1], grid.res, grid.res),
                       compress="deflate", tiled=True, photometric="RGB") as dst:
        for band in range(4):
            dst.write(ortho[:, :, band], band + 1)
        dst.update_tags(CONTENT=content)
    return Path(path)


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
    write_ortho(ortho_path, grid, ortho, offset, epsg, "orthophoto from the dense cloud colours; band 4 = alpha")
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
