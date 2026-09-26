"""Ground control points (spec §8.1: "without *extensive* GCPs" — a handful is a legitimate
accuracy feature, as long as the system works with zero). S5-3.

Input: an OpenDroneMap ``gcp_list.txt`` — the format survey teams already produce:

    EPSG:32643                                   <- CRS line (EPSG code, proj string, or "WGS84 UTM 43N")
    781234.12 1435678.90 912.34 1520.5 866.0 frame_000120.jpg GCP1
    781234.12 1435678.90 912.34 1301.0 902.5 frame_000136.jpg GCP1
    ...                                          <- geo_x geo_y geo_z im_x im_y image [name]

Each point is triangulated from its image observations with the SfM cameras (≥ 2 views), then
joins the camera-to-GPS similarity fit weighted by inverse variance, (gps_sigma / gcp_sigma)². Points
whose name starts with one of ``geo.gcp.check_prefixes`` are *check points*: triangulated and
reported, never fitted, so their residual is an independent accuracy measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ControlPoint:
    name: str
    world: np.ndarray                       # in the file's CRS
    observations: list[tuple[str, float, float]] = field(default_factory=list)
    check: bool = False


def _crs_from_header(line: str) -> str:
    """The CRS line as something pyproj accepts ("WGS84 UTM 43N" is ODM's own shorthand)."""
    text = line.strip()
    m = re.fullmatch(r"WGS84\s+UTM\s+(\d{1,2})\s*([NS])", text, re.I)
    if m:
        return f"EPSG:{(32600 if m.group(2).upper() == 'N' else 32700) + int(m.group(1))}"
    return text


def read_gcp_file(path: Path, check_prefixes: list[str]) -> tuple[str, list[ControlPoint]]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"{path} is empty")
    crs = _crs_from_header(lines[0])
    points: dict[str, ControlPoint] = {}
    for n, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"{path}:{n}: expected 'geo_x geo_y geo_z im_x im_y image [name]', got {line!r}")
        gx, gy, gz, ix, iy = (float(v) for v in parts[:5])
        name = parts[6] if len(parts) > 6 else f"{gx:.3f}_{gy:.3f}_{gz:.3f}"
        point = points.setdefault(name, ControlPoint(name, np.array([gx, gy, gz]),
                                                     check=any(name.lower().startswith(p.lower())
                                                               for p in check_prefixes)))
        point.observations.append((parts[5], ix, iy))
    return crs, list(points.values())


def triangulate(rec, observations: list[tuple[str, float, float]]) -> tuple[np.ndarray | None, float, int]:
    """(model-frame point, mean reprojection px, views used) by linear triangulation (DLT)."""
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    rows, used = [], []
    for image_name, x, y in observations:
        image = by_name.get(image_name)
        if image is None:
            continue
        camera = rec.cameras[image.camera_id]
        pose = image.cam_from_world()
        p = pose.matrix()                                     # 3x4, normalized camera coordinates
        u, v = camera.cam_from_img(np.array([[x, y]], float))[0]
        rows += [u * p[2] - p[0], v * p[2] - p[1]]
        used.append((camera, pose, x, y))
    if len(used) < 2:
        return None, float("nan"), len(used)
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1]
    if abs(h[3]) < 1e-12:
        return None, float("nan"), len(used)
    point = h[:3] / h[3]
    errors = []
    for camera, pose, x, y in used:
        in_cam = pose.matrix() @ np.r_[point, 1.0]
        if in_cam[2] <= 0:
            return None, float("nan"), len(used)
        errors.append(float(np.hypot(*(camera.img_from_cam(in_cam[None, :])[0] - (x, y)))))
    return point, float(np.mean(errors)), len(used)


def to_map(world: np.ndarray, crs: str, horizontal_epsg: int, vertical_epsg: int | None,
           height_datum: str) -> np.ndarray:
    """GCP coordinates into the model's map frame (E, N, H). Heights are taken as orthometric
    (what surveys deliver) unless ``height_datum`` says ellipsoidal or the CRS carries its own."""
    from pyproj import CRS, Transformer

    src = CRS.from_user_input(crs)
    if vertical_epsg is not None and not src.is_vertical and len(src.sub_crs_list) < 2:
        if height_datum == "ellipsoidal":
            src = CRS.from_user_input("EPSG:4979") if src.is_geographic else src.to_3d()
        else:
            base = src.to_epsg()
            src = CRS.from_user_input(f"EPSG:{base}+{vertical_epsg}") if base else src
    dst = f"EPSG:{horizontal_epsg}+{vertical_epsg}" if vertical_epsg else f"EPSG:{horizontal_epsg}"
    t = Transformer.from_crs(src, dst, always_xy=True, only_best=vertical_epsg is not None)
    e, n, h = t.transform(world[:, 0], world[:, 1], world[:, 2])
    return np.c_[e, n, h]


def load_control(rec, path: Path, gcfg: Any, horizontal_epsg: int, vertical_epsg: int | None,
                 offset) -> tuple[list[dict[str, Any]], list[str]]:
    """Triangulated GCPs as dicts {name, check, model, local, reproj_px, views}; plus notes on the rest."""
    crs, points = read_gcp_file(path, list(gcfg.get("check_prefixes", ["chk", "check"])))
    notes: list[str] = []
    max_px = float(gcfg.get("max_reprojection_px", 5.0))
    kept: list[dict[str, Any]] = []
    world = to_map(np.array([p.world for p in points]), crs, horizontal_epsg, vertical_epsg,
                   str(gcfg.get("height_datum", "orthometric")))
    for point, xyz in zip(points, world):
        model, reproj, views = triangulate(rec, point.observations)
        if model is None:
            notes.append(f"{point.name}: {views} registered view(s), needs 2")
            continue
        if reproj > max_px:
            notes.append(f"{point.name}: reprojection {reproj:.1f} px > {max_px:g} (mis-clicked?)")
            continue
        kept.append({"name": point.name, "check": point.check, "model": model,
                     "local": xyz - np.asarray(offset), "reproj_px": round(reproj, 2), "views": views})
    return kept, notes
