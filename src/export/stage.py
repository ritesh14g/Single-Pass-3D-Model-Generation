"""Stage 5b runner: every §8.2 format plus ``metadata.json``.

Each format is written independently: a failing writer (FBX is the fragile one, §8.3) is
recorded with its reason and the rest are still produced. Nothing here changes geometry;
it only moves Track A's outputs into the map frame fitted by the geo stage.
"""

from __future__ import annotations

import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import __version__
from src.core.logging import get_logger, log_downgrade, log_event
from src.export import rasters, writers
from src.geo.georef import Georef

log = get_logger(__name__)
FORMATS = ("obj", "ply", "las", "geotiff", "glb", "fbx")


def _versions() -> dict[str, str]:
    out = {"ps17": __version__, "python": platform.python_version()}
    for module in ("pycolmap", "numpy", "pyproj", "laspy", "rasterio", "trimesh", "torch"):
        try:
            out[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001
            pass
    out["openmvs"] = "2.4.0"
    return out


def run_export(track_a: dict[str, Path], georef_path: Path, out_dir: Path, cfg: Any, *,
               run_info: dict[str, Any] | None = None) -> dict[str, Any]:
    import pycolmap

    ecfg = cfg.get_path("export")
    wanted = [f for f in ecfg.formats if f in FORMATS]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    georef = Georef.load(georef_path)
    frame = georef.frame
    offset = frame.offset if frame else (0.0, 0.0, 0.0)
    crs_string = frame.crs_string if frame else None
    files: list[dict] = []
    failures: dict[str, str] = {}
    timings: dict[str, float] = {}

    def attempt(name: str, fn):
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - one format never takes the others down
            failures[name] = f"{type(exc).__name__}: {exc}"
            log_downgrade(log, f"{name} export", "skipped", failures[name])
            result = None
        timings[name] = round(time.perf_counter() - started, 1)
        return result

    extras = {"crs": crs_string, "offset": list(offset), "vertical_datum": frame.vertical_datum if frame else None,
              "note": "vertices are map coordinates minus offset (east, north, up before glTF's Y-up rotation)"}

    # -- dense cloud: PLY, LAS, rasters ------------------------------------------
    local = rgb = views = confidence = None
    if "dense" in track_a:
        xyz, rgb, views = writers.read_dense(track_a["dense"])
        local = georef.to_local(xyz)
        confidence = writers.confidence_from_views(views, int(ecfg.confidence_full_views))
        map_xyz = local + np.asarray(offset)
        if "ply" in wanted:
            path = attempt("ply", lambda: writers.write_ply(out_dir / "cloud.ply", local, rgb, views, confidence))
            if path:
                files.append(writers.file_entry(path, "ply", points=len(local), frame="local"))
        if "las" in wanted:
            path = attempt("las", lambda: writers.write_las(out_dir / "cloud.las", map_xyz, rgb, views, confidence,
                                                            crs_string, list(ecfg.las.scale), int(ecfg.las.point_format)))
            if path:
                files.append(writers.file_entry(path, "las", points=len(local), frame="map", crs=crs_string))
            if ecfg.tiling.enabled and len(local) > int(ecfg.tiling.max_points_per_tile):
                tiles = attempt("las_tiles", lambda: writers.write_las_tiles(
                    out_dir / "tiles", map_xyz, rgb, views, confidence, crs_string, list(ecfg.las.scale),
                    float(ecfg.tiling.tile_size_m)))
                for tile in tiles or []:
                    files.append(writers.file_entry(tile, "las_tile", frame="map", crs=crs_string))
        if "geotiff" in wanted:
            def geotiffs():
                res_cfg = ecfg.raster.dsm_resolution_m
                res = rasters.auto_resolution(local, views) if str(res_cfg) == "auto" else float(res_cfg)
                grid, dsm, ortho, filled = rasters.rasterize(local, rgb, res, int(ecfg.raster.fill_max_cells),
                                                             float(ecfg.raster.nodata))
                paths = rasters.write_geotiffs(out_dir, grid, dsm, ortho, offset,
                                               frame.horizontal_epsg if frame else None,
                                               frame.vertical_datum if frame else "model units",
                                               float(ecfg.raster.nodata))
                valid = float((dsm != float(ecfg.raster.nodata)).mean())
                return paths, {"resolution_m": res, "width": grid.width, "height": grid.height,
                               "valid_fraction": round(valid, 3), "filled_cells": filled}

            got = attempt("geotiff", geotiffs)
            if got:
                (dsm_path, ortho_path), raster_info = got
                files.append(writers.file_entry(dsm_path, "geotiff_dsm", crs=crs_string, **raster_info))
                files.append(writers.file_entry(ortho_path, "geotiff_ortho", crs=crs_string, **raster_info))

    # -- mesh: OBJ, glTF binary (+ confidence), FBX -------------------------------------
    mesh_src = track_a.get("textured") or track_a.get("mesh")
    obj_path = None
    if mesh_src is None:
        for fmt in ("obj", "glb", "fbx"):
            if fmt in wanted:
                failures[fmt] = "Stage 4 produced no mesh (dense point cloud only)"
    if mesh_src is not None and any(f in wanted for f in ("obj", "glb", "fbx")):
        mesh = attempt("mesh_load", lambda: writers.load_mesh(Path(mesh_src)))
        if mesh is not None:
            mesh.vertices = georef.to_local(np.asarray(mesh.vertices))
            if "obj" in wanted or "fbx" in wanted:
                if track_a.get("textured") is not None and str(mesh_src).lower().endswith(".obj"):
                    obj_path = attempt("obj", lambda: writers.transform_obj(Path(mesh_src), out_dir / "model.obj",
                                                                            georef))
                else:
                    obj_path = attempt("obj", lambda: writers.write_obj(out_dir / "model.obj", mesh))
                if obj_path and "obj" in wanted:
                    files.append(writers.file_entry(obj_path, "obj", faces=len(mesh.faces), frame="local",
                                                    textured=track_a.get("textured") is not None))
            if "glb" in wanted:
                path = attempt("glb", lambda: writers.write_glb(out_dir / "model.glb", mesh, extras))
                if path:
                    files.append(writers.file_entry(path, "glb", faces=len(mesh.faces), frame="local"))
                if local is not None and ecfg.carry_confidence:
                    path = attempt("glb_confidence", lambda: writers.write_confidence_glb(
                        out_dir / "model_confidence.glb", mesh, local, confidence, extras))
                    if path:
                        files.append(writers.file_entry(path, "glb_confidence", frame="local"))
            if "fbx" in wanted:
                blender = writers.find_blender(ecfg.fbx.blender_binary)
                if blender is None:
                    failures["fbx"] = ("no Blender found (export.fbx.blender_binary, tools/blender*/, PATH); "
                                       "FBX needs headless Blender (§8.3)")
                    log_downgrade(log, "fbx export", "skipped", failures["fbx"])
                elif obj_path:
                    path = attempt("fbx", lambda: writers.write_fbx(out_dir / "model.fbx", obj_path, blender))
                    if path:
                        files.append(writers.file_entry(path, "fbx", frame="local", converter=str(blender)))

    # -- coverage and the sidecar -----------------------------------------------------
    coverage = None
    if local is not None and georef.referenced and "sparse" in track_a:
        coverage = attempt("coverage", lambda: rasters.coverage_percent(
            local, georef, pycolmap.Reconstruction(str(track_a["sparse"])), float(ecfg.coverage_cell_m)))
    produced = sorted({f["format"].split("_")[0] for f in files if f["bytes"] > 0} & set(FORMATS))
    bounds = None
    if local is not None:
        lo, hi = local.min(0) + np.asarray(offset), local.max(0) + np.asarray(offset)
        bounds = {"min": lo.round(3).tolist(), "max": hi.round(3).tolist()}
    metadata = {
        "crs": crs_string, "georeferencing": {k: v for k, v in georef.to_dict().items() if k != "per_camera"},
        "coordinate_frames": {"map": crs_string, "local": f"map minus offset {list(offset)}",
                              "offset": list(offset)},
        "scale_residual": {k: georef.stats.get(k) for k in ("rms_all_m", "rms_inliers_m", "horizontal_rms_m",
                                                            "vertical_rms_m", "inliers", "cameras")},
        "coverage": coverage, "bounds_map": bounds,
        "confidence": f"views confirming each point / {int(ecfg.confidence_full_views)}, clipped to 1",
        "files": files, "formats_produced": produced, "formats_failed": failures,
        "processing": run_info or {}, "software": _versions(),
    }
    meta_path = writers.dump_json(out_dir / "metadata.json", metadata)
    log_event(log, logging.INFO, "export finished", produced=produced, failed=sorted(failures))
    metrics = {"formats_wanted": wanted, "formats_produced": produced, "formats_failed": failures,
               "files": files, "coverage": coverage, "timings_s": timings, "crs": crs_string}
    return {"artifacts": {"metadata": meta_path, "folder": out_dir}, "metrics": metrics}
