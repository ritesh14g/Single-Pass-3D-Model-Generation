"""What Stage 3 works on: Track A's cameras and dense cloud, moved into the local map frame.

Stage 3 runs after georeferencing so that voxels, thresholds and gap areas are in metres
(``local`` = map minus the offset in ``georef.json``). Without GPS the georef is the identity
and everything stays in model units; the report says so.

Cameras come from Track A's undistorted workspace (``dense/sparse`` + ``dense/images``,
PINHOLE) when it exists, which is also the image space every dense point's visibility list
refers to: ``view_indices`` (OpenMVS) and ``fused.ply.vis`` (COLMAP, hybrid) index the
model's images in image-id order, unposed ones included (verified on DJI_0047: 100% of the
listed views see their point). Without a dense cloud the sparse points and their tracks
stand in, so zones still exist, only coarser.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.geo.georef import Georef


@dataclass
class CameraView:
    """One posed frame: pinhole intrinsics (or a pycolmap camera for distorted models)."""

    name: str
    width: int
    height: int
    k: np.ndarray                      # 3x3, pixels
    rot_local: np.ndarray              # cam_from_local rotation (orthonormal)
    centre: np.ndarray                 # camera centre, local frame
    image_path: Path | None = None
    colmap_camera: object | None = None  # set when the model is not a pinhole

    def to_cam(self, pts: np.ndarray) -> np.ndarray:
        """Local points -> camera coordinates in local units (z = depth along the optical axis)."""
        return (self.rot_local @ (np.asarray(pts, np.float64) - self.centre).T).T

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pixels [n,2], depth [n]); depth <= 0 means behind the camera."""
        cam = self.to_cam(pts)
        z = cam[:, 2]
        safe = np.where(z > 1e-9, z, 1e-9)
        if self.colmap_camera is not None:
            uv = np.asarray(self.colmap_camera.img_from_cam(cam / safe[:, None]))
        else:
            uv = np.c_[self.k[0, 0] * cam[:, 0] / safe + self.k[0, 2], self.k[1, 1] * cam[:, 1] / safe + self.k[1, 2]]
        return uv, z

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Local-frame ray directions with unit optical-axis component: point = centre + depth * ray."""
        uv = np.asarray(uv, np.float64)
        if self.colmap_camera is not None:
            xy = np.asarray(self.colmap_camera.cam_from_img(uv))
        else:
            xy = np.c_[(uv[:, 0] - self.k[0, 2]) / self.k[0, 0], (uv[:, 1] - self.k[1, 2]) / self.k[1, 1]]
        return (self.rot_local.T @ np.c_[xy, np.ones(len(xy))].T).T

    def inside(self, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < self.width) & (uv[:, 1] < self.height)


@dataclass
class Scene:
    points: np.ndarray                 # [n,3] local frame
    rgb: np.ndarray                    # [n,3] uint8
    views: np.ndarray                  # [n] confirming views per point
    view_ptr: np.ndarray | None        # CSR over view_idx, [n+1]; None when only counts are known
    view_idx: np.ndarray | None        # camera row per (point, view); -1 = unposed/unknown image
    cameras: list[CameraView]
    georef: Georef
    source: str                        # "dense" | "sparse"
    images_dir: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def metric(self) -> bool:
        return self.georef.referenced

    def ground_z(self) -> float:
        return float(np.median(self.points[:, 2]))

    def subset(self, keep: np.ndarray) -> "Scene":
        """The scene without the dropped points (visibility lists kept aligned)."""
        ptr = idx = None
        if self.view_ptr is not None:
            counts = np.diff(self.view_ptr)[keep]
            entry = np.repeat(keep, np.diff(self.view_ptr))
            idx = self.view_idx[entry]
            ptr = np.r_[0, np.cumsum(counts)]
        return Scene(self.points[keep], self.rgb[keep], self.views[keep], ptr, idx, self.cameras, self.georef,
                     self.source, self.images_dir, list(self.notes))


def near_camera(scene: Scene, clearance_fraction: float) -> np.ndarray:
    """Points closer to a camera centre than ``clearance_fraction`` x the flight height above ground.

    Nothing a survey flight images sits within a few metres of the aircraft: such points are stereo
    failures (DJI_0047: 26% of the OpenMVS cloud, 2.4 m in front of a hovering camera, spread over the
    far field) or the aircraft itself. They would also occlude everything in the visibility z-buffers.
    """
    from scipy.spatial import cKDTree

    if not scene.cameras or clearance_fraction <= 0 or not len(scene.points):
        return np.zeros(len(scene.points), bool)
    centres = np.array([c.centre for c in scene.cameras])
    height = float(np.median(centres[:, 2]) - np.median(scene.points[:, 2]))
    if height <= 0:
        return np.zeros(len(scene.points), bool)
    dist, _ = cKDTree(centres).query(scene.points, k=1)
    return dist < clearance_fraction * height


# -- readers -----------------------------------------------------------------------
def read_vis(path: Path, expected: int) -> tuple[np.ndarray, np.ndarray] | None:
    """COLMAP ``fused.ply.vis``: uint64 n, then per point uint32 count + that many image indices."""
    raw = np.fromfile(path, dtype=np.uint32, offset=8)
    n = struct.unpack("<Q", Path(path).open("rb").read(8))[0]
    if n != expected:
        return None
    ptr = np.empty(n + 1, np.int64)
    ptr[0] = 0
    starts = np.empty(n, np.int64)
    pos = 0
    for i in range(n):  # sequential by construction: each count says how far to skip
        count = int(raw[pos])
        starts[i] = pos + 1
        ptr[i + 1] = ptr[i] + count
        pos += 1 + count
    idx = np.empty(ptr[-1], np.int64)
    counts = np.diff(ptr)
    # Gather every visibility entry with one fancy index: entry j of point i sits at starts[i] + j.
    point_of_entry = np.repeat(np.arange(n), counts)
    idx[:] = raw[starts[point_of_entry] + (np.arange(ptr[-1]) - ptr[point_of_entry])]
    return ptr, idx


def read_dense_with_views(fused: Path):
    """(xyz, rgb, views, ptr | None, image indices | None) from a COLMAP or OpenMVS cloud."""
    from plyfile import PlyData

    vertex = PlyData.read(str(fused))["vertex"]
    xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)
    names = vertex.data.dtype.names
    rgb = (np.c_[vertex["red"], vertex["green"], vertex["blue"]].astype(np.uint8) if "red" in names
           else np.full((len(xyz), 3), 180, np.uint8))
    ptr = idx = None
    if "view_indices" in names:
        lists = vertex["view_indices"]
        counts = np.fromiter((len(v) for v in lists), dtype=np.int64, count=len(xyz))
        ptr = np.r_[0, np.cumsum(counts)]
        idx = np.concatenate([np.asarray(v, np.int64) for v in lists]) if ptr[-1] else np.zeros(0, np.int64)
    elif Path(str(fused) + ".vis").exists():
        got = read_vis(Path(str(fused) + ".vis"), len(xyz))
        if got is not None:
            ptr, idx = got
    if ptr is not None:
        views = np.diff(ptr).astype(np.int32)
    elif "views" in names:
        views = np.asarray(vertex["views"], np.int32)
    else:
        views = np.ones(len(xyz), np.int32)
    return xyz, rgb, views, ptr, idx


# -- scene assembly ------------------------------------------------------------------
def cameras_from_model(rec, georef: Georef, images_dir: Path | None) -> tuple[list[CameraView], np.ndarray]:
    """Posed cameras in the local frame, and a map from image-id order to camera row (-1 = unposed)."""
    images = [rec.images[i] for i in sorted(rec.images)]
    row_of = np.full(len(images), -1, np.int64)
    cams: list[CameraView] = []
    rot_g = np.asarray(georef.rotation, np.float64)
    for order, im in enumerate(images):
        if not im.has_pose:
            continue
        cam = im.camera
        pose = im.cam_from_world()
        rot_cw = np.asarray(pose.rotation.matrix())
        centre = georef.to_local(np.asarray(im.projection_center())[None])[0]
        pinhole = str(cam.model).split(".")[-1] in ("PINHOLE", "SIMPLE_PINHOLE")
        path = (Path(images_dir) / im.name) if images_dir is not None else None
        row_of[order] = len(cams)
        cams.append(CameraView(
            name=im.name, width=int(cam.width), height=int(cam.height),
            k=np.asarray(cam.calibration_matrix(), np.float64),
            # local = s R_g model + t, so cam_from_local = R_cw R_g^T (the scale moves into depth).
            rot_local=rot_cw @ rot_g.T, centre=centre,
            image_path=path if path is not None and path.is_file() else None,
            colmap_camera=None if pinhole else cam,
        ))
    return cams, row_of


def load_scene(track_a: dict[str, Path], georef: Georef) -> Scene:
    """Dense cloud + undistorted cameras when Track A made them; sparse points + SfM cameras otherwise."""
    import pycolmap

    notes: list[str] = []
    dense = track_a.get("dense")
    workspace = Path(dense).parent if dense else None
    if dense and Path(dense).is_file() and (workspace / "sparse").is_dir():
        rec = pycolmap.Reconstruction(str(workspace / "sparse"))
        images_dir = workspace / "images" if (workspace / "images").is_dir() else None
        cams, row_of = cameras_from_model(rec, georef, images_dir)
        xyz, rgb, views, ptr, idx = read_dense_with_views(Path(dense))
        if idx is not None:
            idx = np.where((idx >= 0) & (idx < len(row_of)), row_of[np.clip(idx, 0, len(row_of) - 1)], -1)
        else:
            notes.append("dense cloud has view counts but no view lists: triangulation angles use "
                         "geometric visibility instead of the confirming views")
        return Scene(georef.to_local(xyz), rgb, views, ptr, idx, cams, georef, "dense", images_dir, notes)

    rec = pycolmap.Reconstruction(str(track_a["sparse"]))
    cams, row_of = cameras_from_model(rec, georef, None)
    order_of_id = {image_id: order for order, image_id in enumerate(sorted(rec.images))}
    pts, rgb, lists = [], [], []
    for p in rec.points3D.values():
        pts.append(p.xyz)
        rgb.append(p.color)
        lists.append(sorted({row_of[order_of_id[el.image_id]] for el in p.track.elements}))
    counts = np.fromiter((len(v) for v in lists), dtype=np.int64, count=len(lists))
    ptr = np.r_[0, np.cumsum(counts)]
    idx = np.concatenate([np.asarray(v, np.int64) for v in lists]) if len(lists) else np.zeros(0, np.int64)
    notes.append("no dense cloud: zones are classified on the sparse SfM points (coarse)")
    return Scene(georef.to_local(np.asarray(pts, np.float64).reshape(-1, 3)),
                 np.asarray(rgb, np.uint8).reshape(-1, 3), counts.astype(np.int32), ptr, idx, cams, georef,
                 "sparse", None, notes)
