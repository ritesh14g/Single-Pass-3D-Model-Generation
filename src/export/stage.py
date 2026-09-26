"""Stage 5b runner: every §8.2 format plus ``metadata.json``.

Each format is written independently: a failing writer (FBX is the fragile one, §8.3) is
recorded with its reason and the rest are still produced. Nothing here changes geometry;
it only moves Track A's outputs into the map frame fitted by the geo stage.

When Stage 3 ran, its layers travel with the export: points it rejected as not scene surface
are left out, every point carries ``zone`` (1 well / 2 thinly observed) and ``source`` (0 MVS,
1 anchored monocular fill) in PLY and LAS, ``model_zones.glb`` colours the mesh by zone with
unsupported faces marked inferred, and ``gaps.geojson`` / ``zone_map.tif`` are copied alongside.
The DSM and orthophoto stay measured-only: the monocular fill never enters a raster.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import __version__
from src.core.logging import get_logger, log_downgrade, log_event
from src.export import classify, rasters, writers
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


def _fusion_layers(fusion: dict[str, Path] | None, n_points: int):
    """(keep mask, zone per kept point, fill (xyz, rgb, weight, frames) | None, report) from Stage 3."""
    if not fusion:
        return None, None, None, None
    from src.fusion.stage import read_fill_ply

    keep = zone = fill = None
    if fusion.get("point_zones") and Path(fusion["point_zones"]).is_file():
        point_zone = np.load(fusion["point_zones"])
        if len(point_zone) == n_points:
            keep = point_zone > 0
            zone = point_zone[keep]
    if fusion.get("fill") and Path(fusion["fill"]).is_file():
        fill = read_fill_ply(Path(fusion["fill"]))
        if not len(fill[0]):
            fill = None
    report = (json.loads(Path(fusion["report"]).read_text(encoding="utf-8"))
              if fusion.get("report") and Path(fusion["report"]).is_file() else None)
    return keep, zone, fill, report


def run_export(track_a: dict[str, Path], georef_path: Path, out_dir: Path, cfg: Any, *,
               run_info: dict[str, Any] | None = None, fusion: dict[str, Path] | None = None) -> dict[str, Any]:
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

    mesh_src = track_a.get("textured") or track_a.get("mesh")
    mesh_state: dict[str, Any] = {}

    def mesh_local():
        """The Stage 4 mesh in the local map frame, loaded once; None if absent or rejected."""
        if "mesh" not in mesh_state:
            mesh_state["mesh"] = None
            if mesh_src is not None:
                loaded = attempt("mesh_load", lambda: writers.load_mesh(Path(mesh_src)))
                if loaded is not None:
                    loaded.vertices = georef.to_local(np.asarray(loaded.vertices))
                    broken = writers.mesh_is_broken(np.asarray(loaded.vertices), local,
                                                    float(ecfg.mesh_max_extent_factor))
                    mesh_state["broken"] = broken
                    mesh_state["mesh"] = None if broken else loaded
        return mesh_state["mesh"]

    # -- dense cloud: PLY, LAS, rasters ------------------------------------------
    local = rgb = views = confidence = None
    zones_info: dict[str, Any] | None = None
    if "dense" in track_a:
        xyz, rgb, views = writers.read_dense(track_a["dense"])
        keep, zone, fill, fusion_report = _fusion_layers(fusion, len(xyz))
        if keep is not None:
            xyz, rgb, views = xyz[keep], rgb[keep], views[keep]
        local = georef.to_local(xyz)
        confidence = writers.confidence_from_views(views, int(ecfg.confidence_full_views))
        # Point products (PLY, LAS) = measured points + Stage 3's anchored fill, each flagged.
        cloud_local, cloud_rgb, cloud_views, cloud_conf, extra = local, rgb, views, confidence, None
        if zone is not None:
            extra = {"zone": zone, "source": np.zeros(len(zone), np.uint8)}
            if fill is not None:
                f_xyz, f_rgb, f_weight, f_frames = fill
                cloud_local = np.r_[local, f_xyz]
                cloud_rgb = np.r_[rgb, f_rgb]
                cloud_views = np.r_[views, f_frames].astype(np.int32)
                cloud_conf = np.r_[confidence, f_weight].astype(np.float32)
                extra = {"zone": np.r_[zone, np.full(len(f_xyz), 2, np.uint8)],
                         "source": np.r_[np.zeros(len(zone), np.uint8), np.ones(len(f_xyz), np.uint8)]}
            zones_info = {"rejected_points": int((~keep).sum()), "fill_points": int(len(fill[0])) if fill else 0,
                          "report": fusion_report}
        map_xyz = cloud_local + np.asarray(offset)
        if "ply" in wanted:
            path = attempt("ply", lambda: writers.write_ply(out_dir / "cloud.ply", cloud_local, cloud_rgb, cloud_views,
                                                            cloud_conf, extra=extra))
            if path:
                files.append(writers.file_entry(path, "ply", points=len(cloud_local), frame="local",
                                                layers=sorted(extra or {})))
        if "las" in wanted:
            classes, class_info = None, None
            if ecfg.las.classify.enabled and frame is not None:       # needs metres and a z-up frame
                got = attempt("las_classify", lambda: classify.ground_classes(cloud_local, ecfg.las.classify))
                if got:
                    classes, class_info = got
            path = attempt("las", lambda: writers.write_las(out_dir / "cloud.las", map_xyz, cloud_rgb, cloud_views,
                                                            cloud_conf, crs_string, list(ecfg.las.scale),
                                                            int(ecfg.las.point_format), extra=extra,
                                                            classification=classes))
            if path:
                files.append(writers.file_entry(path, "las", points=len(cloud_local), frame="map", crs=crs_string,
                                                layers=sorted(extra or {}), classification=class_info))
            if ecfg.tiling.enabled and len(cloud_local) > int(ecfg.tiling.max_points_per_tile):
                tiles = attempt("las_tiles", lambda: writers.write_las_tiles(
                    out_dir / "tiles", map_xyz, cloud_rgb, cloud_views, cloud_conf, crs_string, list(ecfg.las.scale),
                    float(ecfg.tiling.tile_size_m), extra=extra, classification=classes))
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
                info = {"resolution_m": res, "width": grid.width, "height": grid.height,
                        "valid_fraction": round(valid, 3), "filled_cells": filled}
                ortho_info = {**info, "source": "dense cloud colours"}
                rendered = _mesh_ortho(grid, dsm) if str(ecfg.raster.ortho_source) != "cloud" else None
                if rendered is not None:
                    ortho_info = rendered
                return paths, info, ortho_info

            def _mesh_ortho(grid, dsm):
                """S5-2: the orthophoto rendered from the textured mesh, at the texture's resolution."""
                mesh = mesh_local() if track_a.get("textured") is not None else None
                texture = rasters.texture_image(mesh) if mesh is not None else None
                if texture is None:
                    return None
                rcfg = ecfg.raster
                vertices, faces, uv = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.visual.uv)
                res = rasters.texel_size(vertices, faces, uv, texture.shape) if str(rcfg.ortho_resolution_m) == "auto" \
                    else float(rcfg.ortho_resolution_m)
                res = max(float(res or grid.res), float(rcfg.ortho_min_resolution_m))
                span_x, span_y = grid.width * grid.res, grid.height * grid.res
                while (span_x / res) * (span_y / res) > float(rcfg.ortho_max_pixels):
                    res *= 1.25
                ogrid = rasters.Grid(grid.x0, grid.y1, res, int(np.ceil(span_x / res)), int(np.ceil(span_y / res)))
                image, hit = rasters.render_ortho(vertices, faces, uv, texture, ogrid,
                                                  max_triangle_px=max(int(float(rcfg.ortho_max_triangle_m2)
                                                                          / res ** 2), 1),
                                                  empty_color=tuple(cfg.get_path("recon.track_a.texture.empty_color")))
                supported = rasters.support_mask(dsm, float(rcfg.nodata), grid, ogrid, int(rcfg.ortho_support_cells))
                image[~supported] = 0
                rasters.write_ortho(out_dir / "orthophoto.tif", ogrid, image, offset,
                                    frame.horizontal_epsg if frame else None,
                                    "orthophoto rendered top-down from the textured mesh; band 4 = alpha")
                return {"resolution_m": round(res, 3), "width": ogrid.width, "height": ogrid.height,
                        "valid_fraction": round(float((image[:, :, 3] > 0).mean()), 3),
                        "source": "textured mesh (rendered)", "mesh_hit_fraction": round(hit, 3)}

            got = attempt("geotiff", geotiffs)
            if got:
                (dsm_path, ortho_path), raster_info, ortho_info = got
                files.append(writers.file_entry(dsm_path, "geotiff_dsm", crs=crs_string, **raster_info))
                files.append(writers.file_entry(ortho_path, "geotiff_ortho", crs=crs_string, **ortho_info))

    # -- mesh: OBJ, glTF binary (+ confidence), FBX -------------------------------------
    obj_path = None
    if mesh_src is None:
        for fmt in ("obj", "glb", "fbx"):
            if fmt in wanted:
                failures[fmt] = "Stage 4 produced no mesh (dense point cloud only)"
    if mesh_src is not None and any(f in wanted for f in ("obj", "glb", "fbx")):
        mesh = mesh_local()
        broken = mesh_state.get("broken")
        if broken:
            # Never hand absurd geometry to the writers: it overflowed float32 in glTF and hung
            # Blender for > 10 min on the box (T-1). The point formats still go out.
            for fmt in ("obj", "glb", "fbx"):
                if fmt in wanted:
                    failures[fmt] = f"Stage 4 mesh rejected: {broken}"
            log_downgrade(log, "mesh export", "skipped (point formats only)", broken)
        if mesh is not None:
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
                face_zones = (fusion or {}).get("mesh_face_zones")
                if face_zones and Path(face_zones).is_file():
                    face_zone = np.load(face_zones)
                    if len(face_zone) == len(mesh.faces):
                        path = attempt("glb_zones", lambda: writers.write_zones_glb(
                            out_dir / "model_zones.glb", mesh, face_zone, extras))
                        if path:
                            files.append(writers.file_entry(path, "glb_zones", frame="local"))
                    else:
                        failures["glb_zones"] = (f"Stage 3 flagged {len(face_zone)} faces, the mesh has "
                                                 f"{len(mesh.faces)}")
            if "fbx" in wanted:
                blender = writers.find_blender(ecfg.fbx.blender_binary)
                if blender is None:
                    failures["fbx"] = ("no Blender found (export.fbx.blender_binary, tools/blender*/, PATH); "
                                       "FBX needs headless Blender (§8.3)")
                    log_downgrade(log, "fbx export", "skipped", failures["fbx"])
                elif obj_path:
                    path = attempt("fbx", lambda: writers.write_fbx(out_dir / "model.fbx", obj_path, blender,
                                                                                 int(ecfg.fbx.timeout_s)))
                    if path:
                        files.append(writers.file_entry(path, "fbx", frame="local", converter=str(blender)))

    # -- Stage 3 gap files ------------------------------------------------------------
    for key, kind in (("gaps", "geojson_gaps"), ("zone_map_tif", "geotiff_zones")):
        src = (fusion or {}).get(key)
        if src and Path(src).is_file():
            dst = out_dir / Path(src).name
            shutil.copyfile(src, dst)
            files.append(writers.file_entry(dst, kind, crs=crs_string if key == "zone_map_tif" else "EPSG:4326"))

    # -- coverage and the sidecar -----------------------------------------------------
    coverage = None
    stage3 = (zones_info or {}).get("report") or {}
    if stage3.get("ground") and georef.referenced:
        # Stage 3 measured it on the same footprint grid, with its rejected points removed: one number.
        g = stage3["ground"]
        coverage = {"coverage_pct": g["coverage_pct"], "visible_m2": g["visible_m2"],
                    "covered_m2": g["zone1_m2"] + g["zone2_m2"], "outside_camera_view_m2": g["outside_view_m2"],
                    "ground_plane_z_m": g["ground_plane_z"], "cell_m": g["cell_m"],
                    "source": "Stage 3 zone map (zone 1 + zone 2 of the ground the cameras saw)"}
    elif local is not None and georef.referenced and "sparse" in track_a:
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
        "coverage": coverage, "bounds_map": bounds, "zones": _zones_block(zones_info),
        "confidence": f"views confirming each point / {int(ecfg.confidence_full_views)}, clipped to 1",
        "files": files, "formats_produced": produced, "formats_failed": failures,
        "processing": run_info or {}, "software": _versions(),
    }
    meta_path = writers.dump_json(out_dir / "metadata.json", metadata)
    log_event(log, logging.INFO, "export finished", produced=produced, failed=sorted(failures))
    metrics = {"formats_wanted": wanted, "formats_produced": produced, "formats_failed": failures,
               "files": files, "coverage": coverage, "timings_s": timings, "crs": crs_string,
               "zones": _zones_block(zones_info)}
    return {"artifacts": {"metadata": meta_path, "folder": out_dir}, "metrics": metrics}


def _zones_block(info: dict[str, Any] | None) -> dict[str, Any] | None:
    """Stage 3's headline numbers for ``metadata.json`` (None when Stage 3 did not run)."""
    if not info:
        return None
    report = info.get("report") or {}
    ground = report.get("ground") or {}
    return {
        "coverage_pct": report.get("coverage_pct"), "measured_coverage_pct": report.get("measured_coverage_pct"),
        "zone1_pct": ground.get("zone1_pct"), "zone2_pct": ground.get("zone2_pct"), "zone3_pct": ground.get("zone3_pct"),
        "gaps": report.get("gaps"), "fill_points": info["fill_points"], "rejected_points": info["rejected_points"],
        "holdout_error_median_m": (report.get("fill") or {}).get("holdout_error_median_m"),
        "mesh": report.get("mesh"),
        "note": "zone 1 = measured, >= zone1_min_views at a sufficient angle; zone 2 = thin or monocular "
                "(anchored, weight < zone 1); zone 3 = never observed, listed in gaps.geojson, never filled",
    }
