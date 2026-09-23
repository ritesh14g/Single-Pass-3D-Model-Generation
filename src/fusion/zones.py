"""Three-zone classification (spec §6.1-6.2) and the ground coverage / gap map (§6.4).

Voxel grid: sparse, over the surface the dense cloud occupies; voxel = ``voxel_gsd_multiple``
x the ground sample distance of the images the cloud was built from. Per voxel:

  * ``views``       distinct frames that confirmed a depth in it (the dense cloud's visibility
                    lists: "valid observations"); the largest per-point count when only counts exist
  * ``views_geom``  frames that could see it: in the frustum and not behind nearer surface
                    (a z-buffer per camera — the "ray cast" of §6.2)
  * ``tri_deg``     widest angle between two observing rays (a 2-pass diameter estimate:
                    the ray farthest from the mean, then the ray farthest from that one)
  * ``photo``       photometric consistency: 1 - (std of the observing pixels' grey level / scale)
  * ``conf``        mean per-point confidence (views / ``export.confidence_full_views``)

Zone 1 = views >= ``zone1_min_views`` and tri_deg >= ``zone1_min_triangulation_deg``;
Zone 2 = at least ``zone2_min_views`` but not Zone 1 (few views, or many at a poor angle).

Zone 3 lives on the ground grid: a cell the cameras' footprints cover but no measured surface
occupies. It is never "filled in" here — it becomes a gap in ``gaps.geojson``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.fusion.scene import Scene

OUTSIDE, ZONE1, ZONE2, ZONE3 = 0, 1, 2, 3


@dataclass
class VoxelGrid:
    size: float
    origin: np.ndarray          # local-frame corner of voxel (0, 0, 0)
    dims: np.ndarray            # voxels per axis (int64), for linear keys

    def ijk(self, pts: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(pts) - self.origin) / self.size).astype(np.int64)

    def keys(self, pts: np.ndarray) -> np.ndarray:
        """Linear key per point; -1 outside the grid."""
        ijk = self.ijk(pts)
        ok = np.all((ijk >= 0) & (ijk < self.dims), axis=1)
        key = (ijk[:, 0] * self.dims[1] + ijk[:, 1]) * self.dims[2] + ijk[:, 2]
        return np.where(ok, key, -1)

    def to_dict(self) -> dict[str, Any]:
        return {"size": self.size, "origin": self.origin.tolist(), "dims": self.dims.tolist()}


@dataclass
class ZoneResult:
    grid: VoxelGrid
    keys: np.ndarray            # [V] sorted voxel keys
    centroid: np.ndarray        # [V,3]
    n_points: np.ndarray
    views: np.ndarray
    views_geom: np.ndarray
    tri_deg: np.ndarray
    photo: np.ndarray           # NaN where no images were available
    conf: np.ndarray
    zone: np.ndarray            # [V] uint8
    point_voxel: np.ndarray     # [N] row into the voxel arrays
    gsd: float
    angle_source: str           # "confirming views" | "geometric visibility"
    spacing: float = 0.0        # median distance between neighbouring surface samples
    timings: dict[str, float] = field(default_factory=dict)   # seconds per classification step

    @property
    def point_zone(self) -> np.ndarray:
        return self.zone[self.point_voxel]

    def frame(self):
        import pandas as pd

        return pd.DataFrame({
            "key": self.keys, "x": self.centroid[:, 0], "y": self.centroid[:, 1], "z": self.centroid[:, 2],
            "n_points": self.n_points, "views": self.views, "views_geom": self.views_geom,
            "tri_deg": self.tri_deg.astype(np.float32), "photo": self.photo.astype(np.float32),
            "conf": self.conf.astype(np.float32), "zone": self.zone,
        })


# -- voxel size ------------------------------------------------------------------------
def ground_sample_distance(scene: Scene) -> float:
    """Median metres (local units) per pixel: depth of the scene below each camera / focal."""
    ground = scene.ground_z()
    per_cam = []
    for cam in scene.cameras:
        height = cam.centre[2] - ground
        if height > 0:
            per_cam.append(height / float(cam.k[0, 0]))
    return float(np.median(per_cam)) if per_cam else 1.0


def _occupied(points: np.ndarray, size: float) -> int:
    """Occupied voxels at ``size``: 1-D linear keys (``np.unique(axis=0)`` on rows took 18.7 s
    over the retries on a DJI_0047-sized cloud, S3-8)."""
    if not len(points):
        return 0
    ijk = np.floor(points / size).astype(np.int64)
    ijk -= ijk.min(0)
    dims = ijk.max(0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) >= 2.0 ** 62:
        return len(np.unique(ijk, axis=0))
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    key.sort()
    return int(1 + np.count_nonzero(np.diff(key)))


def voxel_size(scene: Scene, zcfg: Any, max_voxels: int) -> tuple[float, float, list[str]]:
    gsd = ground_sample_distance(scene)
    size = float(zcfg.voxel_gsd_multiple) * gsd
    notes = []
    if scene.metric:
        size = float(np.clip(size, float(zcfg.voxel_size_min_m), float(zcfg.voxel_size_max_m)))
    # Never more occupied voxels than the budget allows: coarsen until the surface fits.
    while True:
        occupied = _occupied(scene.points, size)
        if occupied <= max_voxels:
            break
        notes.append(f"voxel {size:.3f} -> {size * 1.5:.3f}: {occupied:,} occupied voxels > {max_voxels:,}")
        size *= 1.5
    return size, gsd, notes


# -- per-voxel accumulation ----------------------------------------------------------------
def _pair_angles(vox: np.ndarray, cam: np.ndarray, centroid: np.ndarray, centres: np.ndarray, n_vox: int):
    """Widest angle between the rays of each voxel's (voxel, camera) pairs, degrees."""
    out = np.zeros(n_vox)
    if not len(vox):
        return out
    rays = centres[cam] - centroid[vox]
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    mean = np.zeros((n_vox, 3))
    for axis in range(3):
        mean[:, axis] = np.bincount(vox, rays[:, axis], minlength=n_vox)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    # Per voxel, the ray farthest from the mean (a per-voxel minimum, not a sort of every pair:
    # 10 M pairs on DJI_0047).
    to_mean = np.einsum("ij,ij->i", rays, mean[vox])
    lowest = np.full(n_vox, np.inf)
    np.minimum.at(lowest, vox, to_mean)
    pick = np.flatnonzero(to_mean == lowest[vox])
    far = np.zeros((n_vox, 3))
    far[vox[pick]] = rays[pick]
    dots = np.einsum("ij,ij->i", rays, far[vox])
    min_dot = np.ones(n_vox)
    np.minimum.at(min_dot, vox, dots)
    return np.degrees(np.arccos(np.clip(min_dot, -1.0, 1.0)))


def sample_spacing(centroid: np.ndarray, size: float, rng_seed: int = 0) -> float:
    """Median nearest-neighbour distance between voxel centroids (>= the voxel size).

    Z-buffers close the holes between projected samples with this, not the voxel size: a
    cloud sampled more sparsely than its voxels would otherwise let hidden surface show through."""
    from scipy.spatial import cKDTree

    if len(centroid) < 2:
        return size
    rng = np.random.default_rng(rng_seed)
    probe = centroid[rng.choice(len(centroid), min(len(centroid), 20000), replace=False)]
    dist, _ = cKDTree(centroid).query(probe, k=2)
    return float(max(size, np.median(dist[:, 1])))


class _XYIndex:
    """Voxels bucketed on a coarse ground grid, for "which voxels can this camera see" queries."""

    def __init__(self, xy: np.ndarray, cell: float):
        self.cell = float(cell)
        self.lo = xy.min(0) if len(xy) else np.zeros(2)
        c = np.floor((xy - self.lo) / self.cell).astype(np.int64)
        self.nx, self.ny = (int(c[:, 0].max()) + 1, int(c[:, 1].max()) + 1) if len(xy) else (1, 1)
        key = c[:, 0] * self.ny + c[:, 1]
        self.order = np.argsort(key, kind="stable")
        self.key = key[self.order]

    def query(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        cx0, cx1 = max(int((x0 - self.lo[0]) // self.cell), 0), min(int((x1 - self.lo[0]) // self.cell), self.nx - 1)
        cy0, cy1 = max(int((y0 - self.lo[1]) // self.cell), 0), min(int((y1 - self.lo[1]) // self.cell), self.ny - 1)
        if cx0 > cx1 or cy0 > cy1:
            return np.zeros(0, np.int64)
        cols = np.arange(cx0, cx1 + 1) * self.ny
        starts = np.searchsorted(self.key, cols + cy0, "left")
        ends = np.searchsorted(self.key, cols + cy1, "right")
        return np.concatenate([self.order[a:b] for a, b in zip(starts, ends)])


def _frustum_box(cam, z_lo: float, z_hi: float, margin: float):
    """Ground bounding box of the camera's frustum between two heights, or None when the
    frustum does not cross both planes (camera inside the slab, or a view up to the horizon):
    the frustum between the planes is then not the hull of the corner-ray hits."""
    corners = np.array([[0.0, 0.0], [cam.width, 0.0], [0.0, cam.height], [cam.width, cam.height]])
    d = cam.rays(corners)
    hits = []
    for plane in (z_lo, z_hi):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (plane - cam.centre[2]) / d[:, 2]
        if not np.all(np.isfinite(t) & (t > 0)):
            return None
        hits.append(cam.centre[:2] + t[:, None] * d[:, :2])
    hits = np.concatenate(hits)
    return hits[:, 0].min() - margin, hits[:, 0].max() + margin, hits[:, 1].min() - margin, hits[:, 1].max() + margin


def _visibility(scene: Scene, centroid: np.ndarray, size: float, zcfg: Any):
    """Per camera: which voxels it sees (in the frustum, not behind a nearer surface). Packed bits.

    Only voxels under the camera's footprint are projected: a nadir camera sees ~a tenth of a
    strip's voxels, and projecting all of them for 136 cameras was most of DJI_0047's 115 s (S3-8)."""
    import cv2

    width_cap = int(zcfg.zbuffer_width)
    bits = []
    everything = np.arange(len(centroid))
    z_lo, z_hi = (float(centroid[:, 2].min()), float(centroid[:, 2].max())) if len(centroid) else (0.0, 0.0)
    extent = np.ptp(centroid[:, :2], axis=0).max() if len(centroid) else 1.0
    index = _XYIndex(centroid[:, :2], max(float(extent) / 256.0, size))
    for cam in scene.cameras:
        box = _frustum_box(cam, z_lo, z_hi, 2 * size)
        cand = everything if box is None else index.query(*box)
        uv, z = cam.project(centroid[cand])
        inside = cam.inside(uv, z)
        f = min(1.0, width_cap / cam.width)
        w, h = max(int(cam.width * f), 1), max(int(cam.height * f), 1)
        col = np.clip((uv[inside, 0] * f).astype(np.int64), 0, w - 1)
        row = np.clip((uv[inside, 1] * f).astype(np.int64), 0, h - 1)
        zbuf = np.full(h * w, np.inf, np.float32)
        np.minimum.at(zbuf, row * w + col, z[inside].astype(np.float32))
        # Voxels are points in the buffer: close the holes between them with a min filter the
        # size of one projected voxel, so surface behind a gap in the sampling stays hidden.
        radius = int(np.clip(np.round(np.median(size * cam.k[0, 0] * f / z[inside])) if inside.any() else 1, 1, 7))
        zbuf = cv2.erode(zbuf.reshape(h, w), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)).ravel()
        zbuf[zbuf >= 1e30] = np.inf  # cv2.erode turns inf into FLT_MAX
        seen = np.zeros(len(centroid), bool)
        tol = np.maximum(float(zcfg.occlusion_tolerance_voxels) * size, 0.01 * z[inside])
        seen[cand[inside]] = z[inside] <= zbuf[row * w + col] + tol
        bits.append(np.packbits(seen))
    return bits


def _photometric(scene: Scene, vox: np.ndarray, cam: np.ndarray, centroid: np.ndarray, n_vox: int,
                 scale: float, width_cap: int) -> np.ndarray:
    """1 - std(grey level across observing views) / scale, per voxel; NaN without images."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    from src.core.device import cpu_thread_budget

    def load(c: int):
        view = scene.cameras[int(c)]
        if view.image_path is None:
            return None
        # Decode at 1/2, 1/4 or 1/8 scale straight from the JPEG when that stays at or above the
        # working width, instead of a full 4K decode per camera (S3-8).
        flag = cv2.IMREAD_GRAYSCALE
        for factor, reduced in ((8, cv2.IMREAD_REDUCED_GRAYSCALE_8), (4, cv2.IMREAD_REDUCED_GRAYSCALE_4),
                                (2, cv2.IMREAD_REDUCED_GRAYSCALE_2)):
            if view.width / factor >= width_cap:
                flag = reduced
                break
        img = cv2.imread(str(view.image_path), flag)
        if img is None:
            return None
        f = min(1.0, width_cap / img.shape[1])
        if f < 1.0:
            img = cv2.resize(img, (max(int(img.shape[1] * f), 1), max(int(img.shape[0] * f), 1)),
                             interpolation=cv2.INTER_AREA)
        return img

    # Pairs grouped by camera once (a `cam == c` mask per camera scanned every pair each time).
    order = np.argsort(cam, kind="stable")
    cams_sorted = cam[order]
    used = np.unique(cam)
    # The images (small: z-buffer width) decode in parallel; OpenCV releases the GIL.
    with ThreadPoolExecutor(max_workers=max(1, min(cpu_thread_budget(), 4))) as pool:
        images = list(pool.map(load, used)) if len(used) else []
    samples_v, samples_g = [], []
    for c, img in zip(used, images):
        if img is None:
            continue
        view = scene.cameras[int(c)]
        sel = order[np.searchsorted(cams_sorted, c, "left"):np.searchsorted(cams_sorted, c, "right")]
        uv, z = view.project(centroid[vox[sel]])
        ok = view.inside(uv, z)
        sx, sy = img.shape[1] / view.width, img.shape[0] / view.height
        grey = img[np.clip((uv[ok, 1] * sy).astype(int), 0, img.shape[0] - 1),
                   np.clip((uv[ok, 0] * sx).astype(int), 0, img.shape[1] - 1)].astype(np.float64)
        samples_v.append(vox[sel][ok])
        samples_g.append(grey)
    v = np.concatenate(samples_v) if samples_v else np.zeros(0, np.int64)
    g = np.concatenate(samples_g) if samples_g else np.zeros(0)
    total = np.bincount(v, g, minlength=n_vox)
    total_sq = np.bincount(v, g ** 2, minlength=n_vox)
    count = np.bincount(v, minlength=n_vox).astype(np.float64)
    out = np.full(n_vox, np.nan)
    many = count >= 2
    std = np.sqrt(np.maximum(total_sq[many] / count[many] - (total[many] / count[many]) ** 2, 0))
    out[many] = np.clip(1.0 - std / scale, 0.0, 1.0)
    return out


def classify(scene: Scene, cfg: Any) -> tuple[ZoneResult, list[str]]:
    zcfg = cfg.get_path("fusion.zones")
    timings: dict[str, float] = {}
    clock = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        timings[name] = round(now - clock[0], 2)
        clock[0] = now

    size, gsd, notes = voxel_size(scene, zcfg, int(zcfg.max_voxels))
    lap("voxel_size")
    pts = scene.points
    origin = np.floor(pts.min(0) / size) * size - size
    dims = (np.ceil((pts.max(0) - origin) / size).astype(np.int64) + 2)
    grid = VoxelGrid(size, origin, dims)
    keys, point_voxel = np.unique(grid.keys(pts), return_inverse=True)
    n_vox = len(keys)
    n_points = np.bincount(point_voxel, minlength=n_vox)
    centroid = np.stack([np.bincount(point_voxel, pts[:, a], minlength=n_vox) for a in range(3)], 1) / n_points[:, None]
    full = max(int(cfg.get_path("export.confidence_full_views", 5)), 1)
    conf = np.bincount(point_voxel, np.clip(scene.views / full, 0, 1), minlength=n_vox) / n_points
    lap("voxelise")

    centres = np.array([c.centre for c in scene.cameras]) if scene.cameras else np.zeros((0, 3))
    spacing = sample_spacing(centroid, size)
    bits = _visibility(scene, centroid, spacing, zcfg) if scene.cameras else []
    views_geom = np.zeros(n_vox, np.int32)
    for packed in bits:
        views_geom += np.unpackbits(packed, count=n_vox).astype(np.int32)
    lap("visibility")

    if scene.view_idx is not None:
        per_point = np.diff(scene.view_ptr)
        pair_vox = np.repeat(point_voxel, per_point)
        pair_cam = scene.view_idx
        ok = pair_cam >= 0
        pair = np.unique(pair_vox[ok] * max(len(scene.cameras), 1) + pair_cam[ok])
        vox, cam = pair // max(len(scene.cameras), 1), pair % max(len(scene.cameras), 1)
        views = np.bincount(vox, minlength=n_vox).astype(np.int32)
        angle_source = "confirming views"
    else:
        views = np.zeros(n_vox, np.int32)
        np.maximum.at(views, point_voxel, scene.views.astype(np.int32))
        vox_list, cam_list = [], []
        for c, packed in enumerate(bits):
            seen = np.flatnonzero(np.unpackbits(packed, count=n_vox))
            vox_list.append(seen)
            cam_list.append(np.full(len(seen), c, np.int64))
        vox = np.concatenate(vox_list) if vox_list else np.zeros(0, np.int64)
        cam = np.concatenate(cam_list) if cam_list else np.zeros(0, np.int64)
        angle_source = "geometric visibility"
    lap("views")
    tri = _pair_angles(vox, cam, centroid, centres, n_vox)
    lap("angles")
    photo = _photometric(scene, vox, cam, centroid, n_vox, float(zcfg.photometric_std_scale),
                         int(zcfg.zbuffer_width)) if scene.images_dir is not None else np.full(n_vox, np.nan)
    lap("photometric")

    zone = np.full(n_vox, ZONE3, np.uint8)
    zone[views >= int(zcfg.zone2_min_views)] = ZONE2
    zone[(views >= int(zcfg.zone1_min_views)) & (tri >= float(zcfg.zone1_min_triangulation_deg))] = ZONE1
    return ZoneResult(grid, keys, centroid, n_points, views, views_geom, tri, photo, conf, zone,
                      point_voxel, gsd, angle_source, spacing, timings), notes


# -- the ground: coverage and gaps (§6.4) --------------------------------------------------
@dataclass
class GroundMap:
    x0: float
    y1: float
    cell: float
    zone: np.ndarray            # [h,w] uint8: OUTSIDE, ZONE1, ZONE2, ZONE3
    seen: np.ndarray            # [h,w] bool: inside at least one camera footprint
    ground_z: float

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        row = np.floor((self.y1 - xy[:, 1]) / self.cell).astype(np.int64)
        return row, col

    def centres(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.c_[self.x0 + (cols + 0.5) * self.cell, self.y1 - (rows + 0.5) * self.cell]

    def inside(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        return (row >= 0) & (col >= 0) & (row < self.zone.shape[0]) & (col < self.zone.shape[1])

    def stats(self) -> dict[str, Any]:
        area = self.cell ** 2
        visible = int(self.seen.sum())
        counts = {z: int(((self.zone == z) & self.seen).sum()) for z in (ZONE1, ZONE2, ZONE3)}
        pct = {z: round(100.0 * n / max(visible, 1), 1) for z, n in counts.items()}
        return {"cell_m": self.cell, "visible_m2": round(visible * area), "zone1_m2": round(counts[ZONE1] * area),
                "zone2_m2": round(counts[ZONE2] * area), "zone3_m2": round(counts[ZONE3] * area),
                "zone1_pct": pct[ZONE1], "zone2_pct": pct[ZONE2], "zone3_pct": pct[ZONE3],
                "coverage_pct": round(pct[ZONE1] + pct[ZONE2], 1),
                "outside_view_m2": round(int(((self.zone != OUTSIDE) & ~self.seen).sum()) * area),
                "ground_plane_z": round(self.ground_z, 2)}


def footprint(scene: Scene, ground_z: float, x0: float, y1: float, cell: float, shape, max_range_factor: float):
    """Cells entirely inside at least one camera's view of the ground plane.

    The polygon is rasterised with cell centres at +0.5 and then eroded by one cell, so a cell the
    view only clips is not "seen": otherwise every footprint grows a one-cell fringe of false gap
    (synthetic check: 217 m² of fringe around a 100 m² hole)."""
    import cv2

    mask = np.zeros(shape, np.uint8)
    for cam in scene.cameras:
        height = cam.centre[2] - ground_z
        if height <= 0:
            continue
        corners = np.array([[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]], np.float64)
        rays = cam.rays(corners)
        pts = []
        for ray in rays:
            t = (ground_z - cam.centre[2]) / ray[2] if ray[2] < -1e-9 else np.inf
            t = min(t, max_range_factor * height / max(np.linalg.norm(ray), 1e-9))
            pts.append(cam.centre[:2] + t * ray[:2])
        pts = np.asarray(pts)
        col = (pts[:, 0] - x0) / cell - 0.5
        row = (y1 - pts[:, 1]) / cell - 0.5
        single = np.zeros(shape, np.uint8)
        cv2.fillPoly(single, [np.c_[col, row].round().astype(np.int32)], 1)
        mask |= cv2.erode(single, np.ones((3, 3), np.uint8))
    return mask.astype(bool)


def ground_map(scene: Scene, zones: ZoneResult, cell: float, max_range_factor: float,
               extra_xy: np.ndarray | None = None) -> GroundMap:
    """Best zone per ground cell over the camera footprints; Zone 3 = seen but no surface."""
    ground_z = scene.ground_z()
    heights = [c.centre[2] - ground_z for c in scene.cameras if c.centre[2] > ground_z]
    pad = max_range_factor * (float(np.median(heights)) if heights else 0.0)
    xy = np.r_[scene.points[:, :2], np.array([c.centre[:2] for c in scene.cameras]).reshape(-1, 2)]
    lo, hi = xy.min(0) - pad, xy.max(0) + pad
    x0, y1 = float(np.floor(lo[0] / cell) * cell), float(np.ceil(hi[1] / cell) * cell)
    shape = (int(np.ceil((y1 - lo[1]) / cell)) + 1, int(np.ceil((hi[0] - x0) / cell)) + 1)
    seen = footprint(scene, ground_z, x0, y1, cell, shape, max_range_factor)
    gm = GroundMap(x0, y1, cell, np.zeros(shape, np.uint8), seen, ground_z)
    gm.zone[seen] = ZONE3
    # Best (lowest-numbered measured) zone per column: Zone 2 first, then Zone 1 overwrites.
    for z in (ZONE2, ZONE1):
        sel = zones.zone == z
        row, col = gm.cells(zones.centroid[sel, :2])
        ok = gm.inside(row, col)
        gm.zone[row[ok], col[ok]] = z
    if extra_xy is not None and len(extra_xy):
        mark_filled(gm, extra_xy)
    return crop(gm)


def mark_filled(gm: GroundMap, xy: np.ndarray) -> int:
    """Cells an anchored monocular fill reached: Zone 3 -> Zone 2 (thinly observed, one view)."""
    row, col = gm.cells(xy)
    ok = gm.inside(row, col)
    row, col = row[ok], col[ok]
    gap = gm.zone[row, col] == ZONE3
    before = int((gm.zone == ZONE3).sum())
    gm.zone[row[gap], col[gap]] = ZONE2
    return before - int((gm.zone == ZONE3).sum())


def crop(gm: GroundMap) -> GroundMap:
    rows, cols = np.nonzero(gm.zone != OUTSIDE)
    if not len(rows):
        return gm
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return GroundMap(gm.x0 + c0 * gm.cell, gm.y1 - r0 * gm.cell, gm.cell, gm.zone[r0:r1, c0:c1].copy(),
                     gm.seen[r0:r1, c0:c1].copy(), gm.ground_z)


def gap_regions(gm: GroundMap, min_area_m2: float, edge_band_cells: int = 3) -> list[dict[str, Any]]:
    """Connected Zone 3 regions (8-connected) as local-frame polygons with their areas.

    A region within ``edge_band_cells`` of the edge of every view is an "edge of view" gap: the
    footprint is traced on a flat ground plane, so where terrain rises the cameras saw a little
    less than the plane says (synthetic, +-2.4 m relief at 50 m: 1-2 cells)."""
    import cv2
    from scipy import ndimage

    gaps = gm.zone == ZONE3
    labels, n = ndimage.label(gaps, structure=np.ones((3, 3), int))
    if not n:
        return []
    area = gm.cell ** 2
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    outside = ~gm.seen
    edge = ndimage.binary_dilation(outside, structure=np.ones((3, 3), bool), iterations=max(int(edge_band_cells), 1))
    touches = np.bincount(labels[edge & gaps], minlength=n + 1) > 0
    out = []
    for lab in range(1, n + 1):
        if sizes[lab] * area < min_area_m2:
            continue
        mask = (labels == lab).astype(np.uint8)
        # Trace on a 2x grid so the ring follows cell edges to within a quarter cell.
        big = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        contours, hierarchy = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        rings = []
        for i, contour in enumerate(contours):
            # Simplify to within half a cell: a stair-stepped ring per cell edge is megabytes of noise.
            pts = cv2.approxPolyDP(contour, 1.0, True)[:, 0, :].astype(np.float64)
            if len(pts) < 3:
                continue
            xy = np.c_[gm.x0 + (pts[:, 0] + 0.5) * gm.cell / 2, gm.y1 - (pts[:, 1] + 0.5) * gm.cell / 2]
            outer = hierarchy[0][i][3] < 0
            rings.append((outer, np.vstack([xy, xy[:1]])))
        if not rings:
            continue
        rows, cols = np.nonzero(mask)
        centre = gm.centres(rows, cols).mean(0)
        out.append({"id": len(out) + 1, "area_m2": round(float(sizes[lab] * area), 1), "cells": int(sizes[lab]),
                    "touches_view_edge": bool(touches[lab]), "centroid_local": centre.round(2).tolist(),
                    "rings": rings})
    out.sort(key=lambda g: -g["area_m2"])
    for i, g in enumerate(out, 1):
        g["id"] = i
    return out
