"""Stage 3 runner (spec §6): zones, anchored Zone 2 fill, mesh support, gaps and coverage.

Runs after georeferencing (so it works in metres) and before export (which carries the zones
into PLY/LAS/glb). Outputs, all in ``fusion/``:

  zones.parquet        one row per surface voxel: views, angle, photometric, confidence, zone
  point_zones.npy      zone per dense point, in the dense cloud's own order (export reads it)
  fill.ply             anchored monocular points (local frame), weight < Zone 1's, zone 2
  zone_map.npz / .tif  best zone per ground cell over the camera footprints (0 = not in view)
  gaps.geojson         Zone 3 regions: WGS84 polygons with area (RFC 7946), map-frame centroid
  mesh_face_zones.npy  zone per Track A mesh face; 3 = no measured support (inferred surface)
  fusion_report.json   everything the scorecard and the QA report read
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.device import resolve_device
from src.core.logging import get_logger, log_downgrade, log_event
from src.fusion import anchor, zones as zmod
from src.fusion.scene import Scene, load_scene, near_camera
from src.fusion.zones import ZONE1, ZONE2, ZONE3
from src.geo.georef import Georef

log = get_logger(__name__)


# -- monocular depth: cached Track B maps, else the predictor, GPU first ----------------------
class MonoSource:
    """Callable depth source with the §2.4 rule: GPU first, logged CPU fallback, never a crash."""

    def __init__(self, cfg: Any, device: str, cached: anchor.MonoDepth | None, mcfg: Any):
        self.cfg, self.device, self.cached, self.mcfg = cfg, device, cached, mcfg
        self.predictor: anchor.MonoDepth | None = None
        self.downgrades: list[str] = []
        self.name = "track_b_cache" if cached else f"{cfg.get_path('recon.track_b.model')}@{device}"

    @property
    def frame_cap_cpu(self) -> int:
        return int(self.mcfg.cpu_max_frames)

    @property
    def frame_cap(self) -> int:
        on_cpu = self.cached is None and self.device != "cuda"
        return int(self.mcfg.cpu_max_frames if on_cpu else self.mcfg.max_frames)

    def _predict(self, cam):
        if self.predictor is None:
            try:
                self.predictor = anchor.predictor_depth(self.cfg, self.device)
            except Exception as exc:  # noqa: BLE001 - no model/weights here: the source is unavailable
                raise anchor.MonoUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return self.predictor(cam)

    def __call__(self, cam):
        if self.cached is not None:
            try:
                return self.cached(cam)
            except FileNotFoundError:
                pass  # this frame was not anchored by Track B; predict it
        try:
            return self._predict(cam)
        except Exception as exc:  # noqa: BLE001
            if self.device != "cuda" or self.frame_cap_cpu <= 0:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            log_downgrade(log, "Zone 2 monocular depth", "CPU", reason)
            self.downgrades.append(f"monocular depth GPU -> CPU: {reason}")
            self.device, self.predictor = "cpu", None
            self.name = f"{self.cfg.get_path('recon.track_b.model')}@cpu"
            return self._predict(cam)


def mono_source(cfg: Any, depth_dir: Path | None) -> tuple[MonoSource | None, str | None]:
    mcfg = cfg.get_path("fusion.mono_depth")
    if not bool(mcfg.enabled) or str(mcfg.source) == "none":
        return None, "fusion.mono_depth disabled"
    cached = anchor.cached_depth(depth_dir) if str(mcfg.source) in ("auto", "cached") else None
    if str(mcfg.source) == "cached" and cached is None:
        return None, "no Track B depth maps in this run (fusion.mono_depth.source = cached)"
    device = resolve_device(str(cfg.get_path("device.prefer", "auto")))
    source = MonoSource(cfg, device, cached, mcfg)
    if cached is None and source.frame_cap <= 0:
        return None, f"no Track B depth maps and the predictor is off on {device} (fusion.mono_depth.cpu_max_frames)"
    return source, None


# -- helpers ------------------------------------------------------------------------------------
def levelled_identity(sparse_model: Path) -> Georef:
    """Without GPS: model units, but rotated so the ground plane is horizontal (z up)."""
    import pycolmap

    from src.geo.georef import _ground_normal
    from src.recon.merge import posed

    rec = pycolmap.Reconstruction(str(sparse_model))
    pts = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    centres = np.array([im.projection_center() for im in posed(rec)]).reshape(-1, 3)
    up = _ground_normal(pts, centres) if len(pts) >= 3 else np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, [0.0, 0.0, 1.0])
    sin, cos = np.linalg.norm(axis), float(np.clip(up @ [0.0, 0.0, 1.0], -1, 1))
    if sin < 1e-12:
        rot = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / sin
        rot = np.eye(3) + sin * k + (1 - cos) * k @ k
    return Georef(None, 1.0, rot, np.zeros(3), {"referenced": False, "levelled": True})


def mesh_support(mesh_path: Path, georef: Georef, zones: zmod.ZoneResult, fill_pts: np.ndarray,
                 radius: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Zone per mesh face: nearest measured voxel's zone, 2 near filled points, 3 = unsupported."""
    from scipy.spatial import cKDTree

    from src.export.writers import load_mesh

    mesh = load_mesh(Path(mesh_path))
    vertices = georef.to_local(np.asarray(mesh.vertices))
    faces = np.asarray(mesh.faces)
    tri = vertices[faces]
    centres = tri.mean(1)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    measured = zones.zone != ZONE3
    dist, idx = cKDTree(zones.centroid[measured]).query(centres, k=1)
    face_zone = np.where(dist <= radius, zones.zone[measured][idx], ZONE3).astype(np.uint8)
    if len(fill_pts):
        fdist, _ = cKDTree(fill_pts).query(centres, k=1)
        face_zone[(face_zone == ZONE3) & (fdist <= radius)] = ZONE2
    total = max(float(area.sum()), 1e-12)
    stats = {"faces": int(len(faces)), "support_radius": round(radius, 3),
             **{f"zone{z}_area_pct": round(100.0 * float(area[face_zone == z].sum()) / total, 1) for z in (1, 2, 3)},
             "inferred_faces": int((face_zone == ZONE3).sum())}
    return face_zone, stats


def write_fill_ply(path: Path, fill: anchor.FillResult) -> Path:
    from plyfile import PlyData, PlyElement

    vertex = np.zeros(len(fill.points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"),
                                               ("blue", "u1"), ("weight", "f4"), ("frames", "u1"), ("zone", "u1")])
    for i, axis in enumerate("xyz"):
        vertex[axis] = fill.points[:, i] if len(fill.points) else []
    for i, c in enumerate(("red", "green", "blue")):
        vertex[c] = fill.rgb[:, i] if len(fill.rgb) else []
    vertex["weight"] = fill.weight
    vertex["frames"] = np.clip(fill.frames_per_point, 0, 255)
    vertex["zone"] = ZONE2
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))
    return Path(path)


def read_fill_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(local xyz, rgb, weight, frames) of ``fill.ply``."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    return (np.c_[v["x"], v["y"], v["z"]].astype(np.float64), np.c_[v["red"], v["green"], v["blue"]].astype(np.uint8),
            np.asarray(v["weight"], np.float32), np.asarray(v["frames"], np.int32))


def save_zone_map(out_dir: Path, gm: zmod.GroundMap, georef: Georef) -> dict[str, Path]:
    paths = {"zone_map": Path(out_dir) / "zone_map.npz"}
    np.savez_compressed(paths["zone_map"], zone=gm.zone, seen=gm.seen, x0=gm.x0, y1=gm.y1, cell=gm.cell,
                        ground_z=gm.ground_z)
    if georef.referenced:
        import rasterio
        from rasterio.transform import from_origin

        off = georef.frame.offset
        tif = Path(out_dir) / "zone_map.tif"
        with rasterio.open(tif, "w", driver="GTiff", width=gm.zone.shape[1], height=gm.zone.shape[0], count=1,
                           dtype="uint8", crs=f"EPSG:{georef.frame.horizontal_epsg}", nodata=0, compress="deflate",
                           transform=from_origin(gm.x0 + off[0], gm.y1 + off[1], gm.cell, gm.cell)) as dst:
            dst.write(gm.zone, 1)
            dst.write_colormap(1, {0: (0, 0, 0, 0), 1: (31, 136, 61, 255), 2: (191, 135, 0, 255),
                                   3: (207, 34, 46, 255)})
            dst.update_tags(CONTENT="Stage 3 zone per ground cell: 1 well observed, 2 thinly observed, "
                                    "3 never observed (gap), 0 not in any camera view")
        paths["zone_map_tif"] = tif
    return paths


def write_gaps(path: Path, gaps: list[dict[str, Any]], georef: Georef, gm: zmod.GroundMap) -> Path:
    """RFC 7946 GeoJSON (WGS84 lon/lat) when georeferenced; local coordinates, flagged, otherwise."""
    to_lonlat = None
    offset = np.zeros(2)
    if georef.referenced:
        from pyproj import Transformer

        to_lonlat = Transformer.from_crs(f"EPSG:{georef.frame.horizontal_epsg}", "EPSG:4326", always_xy=True)
        offset = np.asarray(georef.frame.offset[:2])
    features = []
    for g in gaps:
        rings = []
        for outer, ring in sorted(g["rings"], key=lambda r: not r[0]):  # outer rings first
            xy = ring + offset
            if to_lonlat is not None:
                lon, lat = to_lonlat.transform(xy[:, 0], xy[:, 1])
                xy = np.c_[lon, lat]
            rings.append(np.round(xy, 7 if to_lonlat else 2).tolist())
        centre = np.asarray(g["centroid_local"]) + offset
        props = {k: v for k, v in g.items() if k not in ("rings", "centroid_local")}
        props.update(zone=3, kind="edge of view" if g["touches_view_edge"] else "interior",
                     centroid_map=np.round(centre, 2).tolist(), note="never observed: no valid depth; flagged, not filled")
        # A component can trace several outer rings (touching diagonally); each is its own polygon.
        polygons, current = [], None
        for (outer, _), coords in zip(sorted(g["rings"], key=lambda r: not r[0]), rings):
            if outer:
                current = [coords]
                polygons.append(current)
            elif current is not None:
                current.append(coords)
        geometry = ({"type": "Polygon", "coordinates": polygons[0]} if len(polygons) == 1
                    else {"type": "MultiPolygon", "coordinates": polygons})
        features.append({"type": "Feature", "properties": props, "geometry": geometry})
    doc = {"type": "FeatureCollection", "features": features,
           "properties": {"crs": "EPSG:4326 (lon, lat)" if to_lonlat else "local model frame (not georeferenced)",
                          "map_crs": georef.frame.crs_string if georef.referenced else None,
                          "cell_m": gm.cell, "total_gap_m2": round(sum(g["area_m2"] for g in gaps), 1)}}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")
    return Path(path)


# -- the stage -----------------------------------------------------------------------------------
def run_fusion(track_a: dict[str, Path], georef_path: Path | None, out_dir: Path, cfg: Any, *,
               budget_stage=None) -> dict[str, Any]:
    fcfg = cfg.get_path("fusion")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    downgrades: list[str] = []
    t0 = time.perf_counter()

    georef = Georef.load(georef_path) if georef_path and Path(georef_path).is_file() else None
    if georef is None or not georef.referenced:
        georef = levelled_identity(Path(track_a["sparse"]))
        downgrades.append("no georeferencing: zones in model units (levelled), gaps.geojson in the local frame")
    full_scene: Scene = load_scene(track_a, georef)
    for note in full_scene.notes:
        log_downgrade(log, "zone classification input", full_scene.source, note)
        downgrades.append(note)
    rejected = near_camera(full_scene, float(fcfg.zones.camera_clearance_fraction))
    scene = full_scene.subset(~rejected) if rejected.any() else full_scene
    if rejected.any():
        share = 100.0 * rejected.mean()
        note = (f"{int(rejected.sum()):,} dense points ({share:.1f}%) within {fcfg.zones.camera_clearance_fraction} "
                f"x flight height of a camera rejected: not scene surface (stereo failures or the aircraft)")
        log_event(log, logging.WARNING, note, event="points_rejected", rejected=int(rejected.sum()))
        downgrades.append(note)
    timings["load"] = round(time.perf_counter() - t0, 1)

    t = time.perf_counter()
    zones, voxel_notes = zmod.classify(scene, cfg)
    for note in voxel_notes:
        downgrades.append(f"voxel size coarsened: {note}")
    timings["classify"] = round(time.perf_counter() - t, 1)
    timings["classify_steps"] = zones.timings
    gcfg = fcfg.ground
    t = time.perf_counter()
    measured_map = zmod.ground_map(scene, zones, float(gcfg.cell_m), float(gcfg.max_range_factor))
    measured_stats = measured_map.stats()
    timings["ground"] = round(time.perf_counter() - t, 1)

    # -- Zone 2 fill (§6.3) --
    t = time.perf_counter()
    depth_dir = track_a.get("track_b_depth")
    if depth_dir is None and track_a.get("dense"):
        guess = Path(track_a["dense"]).parent.parent / "track_b_depth"
        depth_dir = guess if guess.is_dir() else None
    source, why_not = mono_source(cfg, depth_dir)
    if source is None:
        fill = anchor.FillResult(reason=why_not)
        log_downgrade(log, "Zone 2 monocular fill", "gaps reported, not filled", why_not)
        downgrades.append(f"Zone 2 fill skipped: {why_not}")
    else:
        fill = anchor.fill(scene, zones, measured_map, cfg, source, max_frames=source.frame_cap,
                           source=source.name, budget_stage=budget_stage)
        downgrades.extend(source.downgrades)
        if fill.reason:
            downgrades.append(f"Zone 2 fill: {fill.reason}")
    timings["fill"] = round(time.perf_counter() - t, 1)
    final_map = zmod.ground_map(scene, zones, float(gcfg.cell_m), float(gcfg.max_range_factor),
                                extra_xy=fill.points[:, :2] if len(fill.points) else None)
    final_stats = final_map.stats()
    gaps = zmod.gap_regions(final_map, float(fcfg.gaps.min_area_m2), int(fcfg.gaps.edge_band_cells))

    # -- outputs --
    artifacts: dict[str, Path] = {}
    zones_path = out_dir / "zones.parquet"
    zones.frame().to_parquet(zones_path, index=False)
    artifacts["zones"] = zones_path
    point_zones = out_dir / "point_zones.npy"
    full_zone = np.zeros(len(full_scene.points), np.uint8)  # 0 = rejected (not scene surface)
    full_zone[~rejected] = zones.point_zone
    np.save(point_zones, full_zone)
    artifacts["point_zones"] = point_zones
    artifacts["fill"] = write_fill_ply(out_dir / "fill.ply", fill)
    artifacts.update(save_zone_map(out_dir, final_map, georef))
    artifacts["gaps"] = write_gaps(out_dir / "gaps.geojson", gaps, georef, final_map)

    mesh_stats = None
    mesh_src = track_a.get("textured") or track_a.get("mesh")
    if mesh_src is not None and Path(mesh_src).is_file():
        t = time.perf_counter()
        try:
            face_zone, mesh_stats = mesh_support(Path(mesh_src), georef, zones, fill.points,
                                                 float(fcfg.mesh.support_voxels) * zones.grid.size)
            path = out_dir / "mesh_face_zones.npy"
            np.save(path, face_zone)
            artifacts["mesh_face_zones"] = path
            mesh_stats["mesh"] = str(mesh_src)
        except Exception as exc:  # noqa: BLE001 - the mesh flags are a layer, not the stage
            reason = f"{type(exc).__name__}: {exc}"
            log_downgrade(log, "mesh support flags", "none", reason)
            downgrades.append(f"mesh support flags skipped: {reason}")
        timings["mesh"] = round(time.perf_counter() - t, 1)

    counts = {z: int((zones.zone == z).sum()) for z in (ZONE1, ZONE2, ZONE3)}
    point_counts = np.bincount(zones.point_zone, minlength=4)
    report = {
        "referenced": scene.metric, "units": "m" if scene.metric else "model units", "source": scene.source,
        "cameras": len(scene.cameras), "points": int(len(full_scene.points)),
        "rejected_near_camera": int(rejected.sum()),
        "rejected_near_camera_pct": round(100.0 * float(rejected.mean()), 2) if len(rejected) else 0.0,
        "voxel": {**zones.grid.to_dict(), "gsd": round(zones.gsd, 4), "voxels": int(len(zones.keys)),
                  "sample_spacing": round(zones.spacing, 4),
                  "angle_source": zones.angle_source},
        "thresholds": {"zone1_min_views": int(fcfg.zones.zone1_min_views),
                       "zone1_min_triangulation_deg": float(fcfg.zones.zone1_min_triangulation_deg),
                       "zone2_min_views": int(fcfg.zones.zone2_min_views)},
        "voxels_by_zone": {f"zone{z}": n for z, n in counts.items()},
        "points_by_zone": {f"zone{z}": int(point_counts[z]) for z in (1, 2, 3)},
        "views_median": float(np.median(zones.views)) if len(zones.views) else None,
        "views_geom_median": float(np.median(zones.views_geom)) if len(zones.views_geom) else None,
        "tri_deg_median": round(float(np.median(zones.tri_deg)), 2) if len(zones.tri_deg) else None,
        "photometric_median": (round(float(np.nanmedian(zones.photo)), 3)
                               if np.isfinite(zones.photo).any() else None),
        "ground_measured": measured_stats, "ground": final_stats,
        "coverage_pct": final_stats["coverage_pct"], "measured_coverage_pct": measured_stats["coverage_pct"],
        "gaps": {"count": len(gaps), "total_m2": round(sum(g["area_m2"] for g in gaps), 1),
                 "largest_m2": gaps[0]["area_m2"] if gaps else 0.0,
                 "interior": sum(not g["touches_view_edge"] for g in gaps),
                 "min_area_m2": float(fcfg.gaps.min_area_m2)},
        "fill": fill.summary(), "fill_frames": fill.frames,
        "mesh": mesh_stats, "timings_s": timings, "downgrades": downgrades,
    }
    timings["total"] = round(time.perf_counter() - t0, 1)
    report_path = out_dir / "fusion_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    artifacts["report"] = report_path
    log_event(log, logging.INFO, "zones classified", coverage_pct=report["coverage_pct"],
              zone1_pct=final_stats["zone1_pct"], gaps=len(gaps), fill_points=int(len(fill.points)),
              seconds=timings["total"])
    metrics = {k: v for k, v in report.items() if k != "fill_frames"}
    return {"artifacts": artifacts, "metrics": metrics}
