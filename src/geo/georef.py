"""Fit the reconstruction to GPS and describe the map frame every exporter writes into (§8.1).

model  --(similarity: scale, rotation, translation)-->  local  --(+ offset)-->  map (UTM E, N, H)

``local`` is map minus a rounded offset near the flight, so exported vertices stay small
enough for float32 viewers; the offset is in every sidecar. The similarity is fitted from
SfM camera centres to the filtered GPS fixes (Stage 2 ``geo.txt``), RANSAC-robust against
gross GPS outliers. The headline number is the RMS over *all* cameras: the inlier RMS is
reported too, but a tight threshold on systematic residuals (S4-1) would only flatter it.

A straight flight line leaves the roll about the path undetermined by camera centres alone.
When the GPS track is near-collinear the fit adds one virtual ground point per camera: the
point below it on the reconstruction's ground plane must map to the point the same distance
below the GPS fix, i.e. the ground plane is made horizontal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.geo.crs import MapFrame, gps_altitude_datum, gps_to_map, utm_epsg
from src.recon.alignment import apply, umeyama


def read_geo_lonlat(path: Path) -> dict[str, tuple[float, float, float]]:
    """``geo.txt`` rows as ``name -> (lon, lat, alt)``."""
    rows = [line.split() for line in Path(path).read_text().splitlines()[1:] if line.strip()]
    return {r[0]: (float(r[1]), float(r[2]), float(r[3])) for r in rows}


@dataclass
class Georef:
    frame: MapFrame | None
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    stats: dict[str, Any]

    @property
    def referenced(self) -> bool:
        return self.frame is not None

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return apply((self.scale, self.rotation, self.translation), points)

    def to_map(self, points: np.ndarray) -> np.ndarray:
        offset = np.asarray(self.frame.offset if self.frame else (0.0, 0.0, 0.0))
        return self.to_local(points) + offset

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame.to_dict() if self.frame else None, "scale": self.scale,
                "rotation": self.rotation.tolist(), "translation": self.translation.tolist(), **self.stats}

    def save(self, path: Path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=float), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "Georef":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        frame = MapFrame.from_dict(d["frame"]) if d.get("frame") else None
        stats = {k: v for k, v in d.items() if k not in ("frame", "scale", "rotation", "translation")}
        return cls(frame, float(d["scale"]), np.asarray(d["rotation"]), np.asarray(d["translation"]), stats)


def _ground_normal(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Normal of the best-fit ground plane, pointing towards the cameras."""
    centred = points - points.mean(0)
    normal = np.linalg.svd(centred[:: max(len(points) // 20000, 1)], full_matrices=False)[2][2]
    return normal if float((centres.mean(0) - points.mean(0)) @ normal) > 0 else -normal


def fit_similarity(model: np.ndarray, world: np.ndarray, *, iterations: int, threshold_m: float,
                   min_inliers: int, ground_points: np.ndarray | None = None, seed: int = 0):
    """RANSAC similarity from camera centres (``model``) to GPS (``world``).

    Returns ``(transform, inlier mask, info)``. With ``ground_points`` (the reconstruction's
    sparse points) and a near-collinear track, virtual ground pairs fix the roll about the path.
    """
    n = len(model)
    info: dict[str, Any] = {"cameras": n}
    horiz = world[:, :2] - world[:, :2].mean(0)
    sv = np.linalg.svd(horiz, compute_uv=False) if n >= 2 else np.array([1.0, 0.0])
    collinearity = float(sv[1] / max(sv[0], 1e-9)) if len(sv) > 1 else 0.0
    info["track_collinearity"] = round(collinearity, 4)
    extra_m, extra_w = np.empty((0, 3)), np.empty((0, 3))
    if ground_points is not None and len(ground_points) >= 10 and collinearity < 0.1:
        s0 = umeyama(model, world)[0]
        normal = _ground_normal(ground_points, model)
        heights = (model - ground_points.mean(0)) @ normal  # camera heights above the plane, model units
        extra_m = model - heights[:, None] * normal[None, :]
        extra_w = world - (s0 * heights)[:, None] * np.array([0.0, 0.0, 1.0])[None, :]
        info["ground_constraint"] = True
    else:
        info["ground_constraint"] = False

    def fit(idx):
        return umeyama(np.vstack([model[idx], extra_m]), np.vstack([world[idx], extra_w]))

    rng = np.random.default_rng(seed)
    best, best_key = np.ones(n, bool), (-1, np.inf)
    sample = 3 if info["ground_constraint"] or collinearity >= 0.1 else min(n, 4)
    for _ in range(max(int(iterations), 1) if n > sample else 1):
        idx = rng.choice(n, sample, replace=False) if n > sample else np.arange(n)
        try:
            transform = fit(idx)
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(apply(transform, model) - world, axis=1)
        inliers = residual < threshold_m
        key = (int(inliers.sum()), -float(np.median(residual)))
        if key > best_key:
            best, best_key = inliers, key
    if best.sum() >= max(min_inliers, 3):
        transform, info["fit_on"] = fit(np.flatnonzero(best)), "inliers"
    else:
        transform, info["fit_on"] = fit(np.arange(n)), "all (too few inliers)"
        best = np.ones(n, bool)
    residual = np.linalg.norm(apply(transform, model) - world, axis=1)
    info.update(
        rms_all_m=round(float(np.sqrt((residual ** 2).mean())), 3),
        rms_inliers_m=round(float(np.sqrt((residual[best] ** 2).mean())), 3),
        inliers=int(best.sum()), inlier_threshold_m=threshold_m,
        horizontal_rms_m=round(float(np.sqrt(((apply(transform, model) - world)[:, :2] ** 2).sum(1).mean())), 3),
        vertical_rms_m=round(float(np.sqrt(((apply(transform, model) - world)[:, 2] ** 2).mean())), 3),
    )
    return transform, best, info, residual


def georeference(sparse_model: Path, geo_path: Path | None, cfg: Any, *, telemetry_source: str = "") -> Georef:
    """Fit ``sparse_model`` to ``geo_path``; an identity, unreferenced result when there is no GPS."""
    import pycolmap

    from src.recon.merge import posed

    gcfg = cfg.get_path("geo")
    rec = pycolmap.Reconstruction(str(sparse_model))
    images = sorted(posed(rec), key=lambda im: im.name)
    fixes = read_geo_lonlat(geo_path) if geo_path and Path(geo_path).exists() else {}
    matched = [im for im in images if im.name in fixes]
    if len(matched) < 3:
        return Georef(None, 1.0, np.eye(3), np.zeros(3),
                      {"referenced": False, "reason": f"{len(matched)} registered frames with a GPS fix",
                       "cameras": len(images)})

    lon, lat, alt = (np.array([fixes[im.name][k] for im in matched]) for k in range(3))
    horizontal = utm_epsg(float(np.mean(lon)), float(np.mean(lat)))
    if str(gcfg.crs.target) not in ("auto", "", "None"):
        horizontal = int(str(gcfg.crs.target).split(":")[-1])
    datum, assumed = gps_altitude_datum(telemetry_source, str(gcfg.vertical.get("gps_altitude_datum", "auto")))
    want_ortho = str(gcfg.vertical.output_datum) == "orthometric"
    enh, vertical, geoid_applied, note = gps_to_map(lon, lat, alt, horizontal_epsg=horizontal,
                                                    geoid_model=str(gcfg.vertical.geoid_model),
                                                    want_orthometric=want_ortho, gps_datum=datum)
    offset = (float(np.round(enh[:, 0].mean(), -2)), float(np.round(enh[:, 1].mean(), -2)), 0.0)
    if vertical:
        vdatum = f"{gcfg.vertical.geoid_model} orthometric"
    else:
        vdatum = "orthometric (as given by KLV)" if datum == "orthometric" else "WGS84 ellipsoidal"
    frame = MapFrame(horizontal, vertical, vdatum, geoid_applied, datum, assumed, offset,
                     [note] if note else [])

    model = np.array([im.projection_center() for im in matched])
    local = enh - np.asarray(offset)
    ground = np.array([p.xyz for p in rec.points3D.values()]) if rec.num_points3D() else None
    scfg = gcfg.similarity
    transform, inliers, info, residual = fit_similarity(
        model, local, iterations=int(scfg.ransac_iterations), threshold_m=float(scfg.inlier_threshold_m),
        min_inliers=int(scfg.min_cameras), ground_points=ground)
    info.update(referenced=True, gps_frames=len(matched), registered=len(images),
                per_camera=[{"frame": im.name, "residual_m": round(float(r), 3), "inlier": bool(ok)}
                            for im, r, ok in zip(matched, residual, inliers)])
    scale, rot, trans = transform
    return Georef(frame, float(scale), rot, trans, info)
