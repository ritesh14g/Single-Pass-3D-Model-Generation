"""Track A (§7.1): pycolmap SfM and dense stereo, OpenMVS mesh and texture.

The classical baseline and the submission floor. It must work standalone, so every
step has a fallback that keeps the run going:

  step              GPU first                      fallback (logged downgrade)
  features/matching pycolmap on CUDA               pycolmap on CPU
  mapping           incremental SfM (CPU in COLMAP) -
  dense             COLMAP PatchMatch on CUDA      OpenMVS DensifyPointCloud (CPU), else none
  mesh              OpenMVS ReconstructMesh        pycolmap Poisson + cleaning
  texture           OpenMVS TextureMesh            the untextured mesh

The defaults come from measurements on the institute box (DEVLOG S4-6, S4-8): focal
seeded from telemetry and held fixed, dense at 1280 px with 8 source views and the
geometric pass, OpenMVS Delaunay meshing. Dense stereo is one long GPU call that cannot
be degraded halfway, so its time is projected up front and the resolution stepped down
through ``dense.degrade_sizes`` when the projection overruns the stage budget.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from src.core.device import cpu_thread_budget, resolve_device
from src.core.logging import get_logger, log_downgrade, log_event
from src.recon import alignment, meshing, openmvs

log = get_logger(__name__)


class TrackAError(RuntimeError):
    """Track A produced nothing usable (no SfM model)."""


class _Run:
    """Timings and downgrades for one Track A run, reported alongside the outputs."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.downgrades: list[str] = []

    @contextmanager
    def timed(self, name: str):
        started = time.perf_counter()
        log_event(log, logging.INFO, f"track_a {name} start", step=name)
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - started, 1)
            log_event(log, logging.INFO, f"track_a {name} done", step=name, seconds=self.timings[name])

    def downgrade(self, component: str, fallback: str, reason: str) -> None:
        log_downgrade(log, component, fallback, reason)
        self.downgrades.append(f"{component} -> {fallback}: {reason}")


def run_track_a(
    images_dir: Path,
    out_dir: Path,
    cfg: Any,
    *,
    masks_dir: Path | None = None,
    geo_path: Path | None = None,
    telemetry_path: Path | None = None,
    budget_stage: Any = None,
    depth_predictor: Any = None,
) -> dict[str, Any]:
    """Reconstruct the conditioned frames in ``images_dir`` into ``out_dir``.

    ``depth_predictor`` replaces VGGT in the hybrid (tests); ``None`` loads VGGT.
    Returns ``{"artifacts": {key: path}, "metrics": {...}}`` for the manifest.
    """
    import pycolmap

    tcfg = cfg.get_path("recon.track_a")
    out_dir = Path(out_dir).resolve()
    images_dir = Path(images_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run = _Run()
    threads = cpu_thread_budget()

    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if len(names) < 2:
        raise TrackAError(f"Track A needs at least 2 conditioned frames, found {len(names)} in {images_dir}")

    gpu_visible = resolve_device(str(cfg.get_path("device.prefer", "auto"))) == "cuda"
    use_cuda = gpu_visible and bool(pycolmap.has_cuda)
    if gpu_visible and not pycolmap.has_cuda:
        run.downgrade("COLMAP on CUDA", "COLMAP on CPU",
                      "this pycolmap build has no CUDA (install pycolmap-cuda12)")
    bin_dir = openmvs.find_bin_dir(tcfg.openmvs.bin_dir)
    if bin_dir is None:
        run.downgrade("OpenMVS", "pycolmap-only meshing, no texture",
                      f"OpenMVS tools not found (recon.track_a.openmvs.bin_dir={tcfg.openmvs.bin_dir})")

    hints = alignment.telemetry_hints(telemetry_path)
    gps = alignment.read_geo_enu(geo_path) if geo_path and Path(geo_path).exists() else {}

    rec, sparse_dir, focal_source = _sparse(pycolmap, tcfg, images_dir, masks_dir, out_dir, names,
                                            hints, threads, use_cuda, run)
    reg_names = sorted(im.name for im in rec.images.values())
    by_name = {im.name: im for im in rec.images.values()}
    centres = np.array([by_name[n].projection_center() for n in reg_names])
    sparse_points = np.array([p.xyz for p in rec.points3D.values()])
    metric, transform = alignment.metric_check(reg_names, centres, sparse_points, gps)

    metrics: dict[str, Any] = {
        "frames_in": len(names),
        "registered": rec.num_reg_images(),
        "registered_fraction": round(rec.num_reg_images() / len(names), 3),
        "models": len(list(p for p in sparse_dir.iterdir() if p.is_dir())),
        "sparse_points": rec.num_points3D(),
        "reproj_px": round(float(rec.compute_mean_reprojection_error()), 3),
        "track_length": round(float(rec.compute_mean_track_length()), 2),
        "focal_px": round(float(next(iter(rec.cameras.values())).params[0]), 1),
        "focal_source": focal_source,
        "using_cuda": use_cuda,
        **metric,
    }
    if "expected_agl_m" in hints:
        metrics["expected_agl_m"] = round(hints["expected_agl_m"], 1)
        if "height_above_ground_m" in metrics:
            metrics["height_error_pct"] = round(
                100 * (metrics["height_above_ground_m"] - hints["expected_agl_m"]) / hints["expected_agl_m"], 1)

    artifacts: dict[str, Path] = {"sparse": sparse_dir / str(_best_model_id(sparse_dir, rec))}
    fused, dense_info, mvs_dir = _dense(pycolmap, cfg, tcfg, rec, artifacts["sparse"], images_dir, out_dir,
                                        threads, use_cuda, bin_dir, budget_stage, run,
                                        depth_predictor=depth_predictor)
    metrics["dense"] = dense_info
    if fused is not None:
        artifacts["dense"] = fused
        if (out_dir / "track_b_depth").is_dir():
            artifacts["track_b_depth"] = out_dir / "track_b_depth"
        points = meshing.read_ply_xyz(fused)
        dense_info["points"] = int(len(points))
        area = alignment.footprint_m2(points, transform)
        if area is not None:
            dense_info["footprint_m2"] = round(area)

        mesh, mesh_info = _mesh(pycolmap, tcfg, fused, out_dir, mvs_dir, threads, bin_dir, run)
        metrics["mesh"] = mesh_info
        if mesh is not None:
            artifacts["mesh"] = mesh
            textured = _texture(tcfg, mesh, out_dir, mvs_dir, threads, bin_dir, run)
            metrics["textured"] = textured is not None
            if textured is not None:
                artifacts["textured"] = textured
    metrics["timings_s"] = run.timings
    metrics["downgrades"] = run.downgrades

    report = out_dir / "track_a_report.json"
    report.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    artifacts["report"] = report
    log_event(log, logging.INFO, "track_a finished", registered=metrics["registered"],
              dense_engine=dense_info.get("engine"), mesher=metrics.get("mesh", {}).get("mesher"),
              textured=metrics.get("textured", False))
    return {"artifacts": artifacts, "metrics": metrics}


# -- sparse -----------------------------------------------------------------
def _sparse(pycolmap, tcfg, images_dir, masks_dir, out_dir, names, hints, threads, use_cuda, run):
    sparse_dir = out_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir()
    db = out_dir / "database.db"
    if db.exists():
        db.unlink()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = str(tcfg.camera_model)
    mask_dir = _colmap_masks(images_dir, masks_dir, out_dir, names) if tcfg.use_dynamic_masks else None
    if mask_dir is not None:
        reader.mask_path = str(mask_dir)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = int(tcfg.sift.max_image_size)
    extraction.sift.max_num_features = int(tcfg.sift.max_num_features)
    extraction.num_threads = threads

    def extract(device):
        if db.exists():
            db.unlink()
        pycolmap.extract_features(db, images_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction, device=device)

    _gpu_then_cpu(pycolmap, "sparse_extract", extract, use_cuda, run)

    focal_source = "self_calibrated"
    fix_focal = False
    if tcfg.focal_from_telemetry:
        handle = pycolmap.Database.open(str(db))
        try:
            cam = handle.read_all_cameras()[0]
            focal, focal_source = alignment.focal_from_telemetry(hints, cam.width)
            if focal is not None:
                cam.params = [focal, cam.width / 2, cam.height / 2] + [0.0] * (len(cam.params) - 3)
                cam.has_prior_focal_length = True
                handle.update_camera(cam)
                fix_focal = True
                log_event(log, logging.INFO, f"focal seeded from {focal_source}: {focal:.0f} px, held fixed")
        finally:
            handle.close()
    if tcfg.focal_from_telemetry and not fix_focal:
        run.downgrade("focal length prior", "self-calibrated focal",
                      "telemetry has no field of view or focal length; height/scale may drift on nadir flights")

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = int(tcfg.matching.sequential_overlap)
    pairing.quadratic_overlap = bool(tcfg.matching.quadratic_overlap)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    _gpu_then_cpu(pycolmap, "sparse_match",
                  lambda device: pycolmap.match_sequential(db, matching_options=matching,
                                                           pairing_options=pairing, device=device),
                  use_cuda, run)

    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.num_threads = threads
    mapper.ba_refine_focal_length = not fix_focal
    if float(tcfg.mapper.max_runtime_s) > 0:
        mapper.max_runtime_seconds = float(tcfg.mapper.max_runtime_s)
    with run.timed("sparse_map"):
        recs = pycolmap.incremental_mapping(db, images_dir, sparse_dir, options=mapper)
    if not recs:
        raise TrackAError("SfM produced no model: too few matches between the conditioned frames")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    return rec, sparse_dir, focal_source


def _colmap_masks(images_dir: Path, masks_dir: Path | None, out_dir: Path, names: list[str]) -> Path | None:
    """Stage 2's ``<stem>_exclude.png`` (255 = exclude) in COLMAP's convention:
    ``<image name>.png`` per image, 0 = no features."""
    if masks_dir is None or not Path(masks_dir).is_dir():
        return None
    import cv2

    sources = {n: Path(masks_dir) / f"{Path(n).stem}_exclude.png" for n in names}
    if not any(p.exists() for p in sources.values()):
        return None
    target = out_dir / "masks_colmap"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    shape = None
    for name, src in sources.items():
        if src.exists():
            exclude = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            shape = exclude.shape
            keep = np.where(exclude > 127, 0, 255).astype(np.uint8)
        else:
            if shape is None:
                shape = cv2.imread(str(images_dir / name), cv2.IMREAD_GRAYSCALE).shape
            keep = np.full(shape, 255, dtype=np.uint8)
        cv2.imwrite(str(target / f"{name}.png"), keep)
    return target


def _best_model_id(sparse_dir: Path, rec) -> int:
    import pycolmap

    for d in sorted(p for p in sparse_dir.iterdir() if p.is_dir()):
        if pycolmap.Reconstruction(str(d)).num_reg_images() == rec.num_reg_images():
            return int(d.name)
    return 0


def _gpu_then_cpu(pycolmap, name: str, fn, use_cuda: bool, run: _Run):
    if use_cuda:
        try:
            with run.timed(name):
                return fn(pycolmap.Device.cuda)
        except Exception as exc:  # noqa: BLE001 - any GPU failure downgrades, never crashes
            run.downgrade(f"{name} on CUDA", "CPU", f"{type(exc).__name__}: {exc}")
    with run.timed(name):
        return fn(pycolmap.Device.cpu)


# -- dense ------------------------------------------------------------------
def _dense(pycolmap, cfg, tcfg, rec, sparse_model, images_dir, out_dir, threads, use_cuda, bin_dir,
           budget_stage, run, depth_predictor=None):
    """Returns (fused cloud or None, info, OpenMVS working folder).

    run.mode A: Track A dense. auto / hybrid / B: VGGT depth on these cameras first
    (Track B, ``track_b_vggt``), falling back to Track A dense on any failure.
    """
    dcfg = tcfg.dense
    undist = out_dir / "dense"
    mvs_dir = out_dir / "mvs"
    for stale in (undist, mvs_dir, out_dir / "track_b_depth"):
        if stale.exists():
            shutil.rmtree(stale)
    mode = str(cfg.get_path("run.mode", "auto"))
    engine_gpu = use_cuda
    if mode == "A" and not engine_gpu and bin_dir is None:
        run.downgrade("dense reconstruction", "sparse model only",
                      "no CUDA for COLMAP PatchMatch and no OpenMVS for the CPU path")
        return None, {"engine": None, "mode": mode}, mvs_dir

    dense_model = sparse_model
    frames = rec.num_reg_images()
    every = max(int(dcfg.every), 1)
    if every > 1:
        # The skipped frames must leave the model itself: undistort_images(image_names=)
        # keeps every frame, so PatchMatch's __auto__ source views point at missing files.
        subset = pycolmap.Reconstruction(str(sparse_model))
        keep = set(sorted(im.name for im in subset.images.values())[::every])
        for frame_id in {im.frame_id for im in subset.images.values() if im.name not in keep}:
            subset.deregister_frame(frame_id)
        dense_model = out_dir / "sparse_dense_subset"
        if dense_model.exists():
            shutil.rmtree(dense_model)
        dense_model.mkdir()
        subset.write(str(dense_model))
        frames = subset.num_reg_images()

    # Undistort once at the configured size: the images feed VGGT, PatchMatch (which
    # downsamples further when the budget asks) and the texture.
    undistort = pycolmap.UndistortCameraOptions()
    undistort.max_image_size = int(dcfg.max_image_size)
    with run.timed("undistort"):
        pycolmap.undistort_images(undist, dense_model, images_dir, num_patch_match_src_images=int(dcfg.src_images),
                                  undistort_options=undistort, num_threads=threads)
    fused = undist / "fused.ply"

    if mode != "A":
        from src.recon import track_b_vggt

        try:
            with run.timed("dense_vggt"):
                hybrid = track_b_vggt.depth_cloud(undist, fused, cfg, depth_dir=out_dir / "track_b_depth",
                                                  predictor=depth_predictor)
            return fused, {"mode": mode, "size": int(dcfg.max_image_size), **hybrid}, mvs_dir
        except Exception as exc:  # noqa: BLE001 - Track B never takes the run down (spec §7.4 auto)
            run.downgrade("Track B (VGGT depth)", "Track A dense", f"{type(exc).__name__}: {exc}")
        if not engine_gpu and bin_dir is None:
            run.downgrade("dense reconstruction", "sparse model only",
                          "no CUDA for COLMAP PatchMatch and no OpenMVS for the CPU path")
            return None, {"engine": None, "mode": mode}, mvs_dir

    camera = next(iter(rec.cameras.values()))
    size = _budgeted_size(dcfg, frames, engine_gpu, budget_stage, native_px=max(camera.width, camera.height))
    info: dict[str, Any] = {"mode": mode, "size": size, "src_images": int(dcfg.src_images),
                            "iterations": int(dcfg.iterations), "geom_consistency": bool(dcfg.geom_consistency),
                            "frames": frames}

    if engine_gpu:
        try:
            pm = pycolmap.PatchMatchOptions()
            pm.max_image_size = size
            pm.gpu_index = "0"
            pm.num_iterations = int(dcfg.iterations)
            pm.window_step = int(dcfg.window_step)
            pm.geom_consistency = bool(dcfg.geom_consistency)
            with run.timed("dense_patchmatch"):
                pycolmap.patch_match_stereo(undist, options=pm)
            fusion = pycolmap.StereoFusionOptions()
            fusion.num_threads = threads
            fusion.min_num_pixels = int(dcfg.fusion_min_num_pixels)
            with run.timed("dense_fusion"):
                pycolmap.stereo_fusion(fused, undist, options=fusion, output_type="PLY",
                                       input_type="geometric" if dcfg.geom_consistency else "photometric")
            info["engine"] = "colmap_patchmatch_cuda"
            return fused, info, mvs_dir
        except Exception as exc:  # noqa: BLE001
            if bin_dir is None:
                run.downgrade("dense on CUDA", "sparse model only", f"{type(exc).__name__}: {exc}; no OpenMVS")
                return None, {**info, "engine": None}, mvs_dir
            run.downgrade("dense on CUDA", "OpenMVS DensifyPointCloud on CPU", f"{type(exc).__name__}: {exc}")

    # CPU dense. The images are already undistorted to ``size``, so OpenMVS must not halve them again.
    try:
        with run.timed("openmvs_import"):
            openmvs.run_tool(bin_dir, "InterfaceCOLMAP", mvs_dir, "-i", str(undist), "-o", str(mvs_dir / "scene.mvs"),
                             "--image-folder", str(undist / "images"), threads=threads)
        with run.timed("dense_openmvs"):
            openmvs.run_tool(bin_dir, "DensifyPointCloud", mvs_dir, "-i", str(mvs_dir / "scene.mvs"),
                             "-o", str(mvs_dir / "scene_dense.mvs"), "--resolution-level", "0",
                             "--max-resolution", str(size), threads=threads)
    except openmvs.OpenMvsError as exc:
        run.downgrade("dense on CPU", "sparse model only", str(exc))
        return None, {**info, "engine": None}, mvs_dir
    shutil.copy(mvs_dir / "scene_dense.ply", fused)
    info["engine"] = "openmvs_cpu"
    return fused, info, mvs_dir


def _budgeted_size(dcfg, frames: int, gpu: bool, budget_stage, native_px: int) -> int:
    """Dense resolution that the projected run time fits in the stage budget.

    Cost scales with the pixels actually processed: frames smaller than the target size
    are not upscaled, so they are projected at their own size.
    """
    size = int(dcfg.max_image_size)
    if budget_stage is None or not getattr(budget_stage, "enabled", False):
        return size
    per_frame_1280 = float(dcfg.gpu_s_per_frame_at_1280 if gpu else dcfg.cpu_s_per_frame_at_1280)
    available = budget_stage.remaining_s - float(dcfg.reserve_after_s)

    def projected(px: int) -> float:
        return frames * per_frame_1280 * (min(px, native_px) / 1280.0) ** 2

    for smaller in [int(s) for s in dcfg.degrade_sizes if int(s) < min(size, native_px)]:
        if projected(size) <= available:
            break
        budget_stage.degrade("reduce_resolution", reason="dense stereo projected to overrun",
                             from_px=size, to_px=smaller, projected_s=round(projected(size)),
                             available_s=round(available), frames=frames)
        size = smaller
    return size


# -- mesh and texture -------------------------------------------------------
def _mesh(pycolmap, tcfg, fused, out_dir, mvs_dir, threads, bin_dir, run):
    mcfg = tcfg.mesh
    mesh = out_dir / "mesh.ply"
    scene_dense = mvs_dir / "scene_dense.mvs"
    if bin_dir is not None and str(mcfg.mesher) == "openmvs":
        try:
            if not scene_dense.exists():  # COLMAP's cloud, imported with its per-point visibility
                with run.timed("openmvs_import"):
                    openmvs.run_tool(bin_dir, "InterfaceCOLMAP", mvs_dir, "-i", str(fused.parent), "-p", str(fused),
                                     "-o", str(scene_dense), "--image-folder", str(fused.parent / "images"),
                                     threads=threads)
            with run.timed("mesh_openmvs"):
                openmvs.run_tool(bin_dir, "ReconstructMesh", mvs_dir, "-i", str(scene_dense), "-o", str(mesh),
                                 threads=threads)
            # ReconstructMesh can clean a thin surface down to nothing, exit 0 and write no
            # file (seen on a tiny flat synthetic scene); that is a failure, not a mesh.
            counts = meshing.ply_counts(mesh) if mesh.exists() else {}
            if not counts.get("faces"):
                raise openmvs.OpenMvsError("ReconstructMesh produced an empty mesh")
            return mesh, {"mesher": "openmvs_delaunay", **counts, **_face_ratio(counts)}
        except openmvs.OpenMvsError as exc:
            run.downgrade("OpenMVS ReconstructMesh", "pycolmap Poisson", str(exc))
    raw = out_dir / "mesh_poisson.ply"
    try:
        poisson = pycolmap.PoissonMeshingOptions()
        poisson.depth = int(mcfg.poisson_depth)
        poisson.num_threads = threads
        with run.timed("mesh_poisson"):
            pycolmap.poisson_meshing(fused, raw, options=poisson)
        with run.timed("mesh_clean"):
            stats = meshing.clean_mesh(raw, mesh, float(mcfg.min_component_fraction))
    except Exception as exc:  # noqa: BLE001
        run.downgrade("meshing", "dense point cloud only", f"{type(exc).__name__}: {exc}")
        return None, {"mesher": None}
    return mesh, {"mesher": "colmap_poisson", **stats, **_face_ratio(stats)}


def _face_ratio(counts: dict) -> dict:
    """~2 faces per vertex on a clean surface; ~1 means fragments (S4-6)."""
    if counts.get("vertices"):
        return {"faces_per_vertex": round(counts.get("faces", 0) / counts["vertices"], 2)}
    return {}


def _texture(tcfg, mesh, out_dir, mvs_dir, threads, bin_dir, run):
    if bin_dir is None or not tcfg.texture.enabled:
        return None
    scene = mvs_dir / "scene_dense.mvs"
    try:
        if not scene.exists():
            scene = mvs_dir / "scene.mvs"
            with run.timed("openmvs_import"):
                openmvs.run_tool(bin_dir, "InterfaceCOLMAP", mvs_dir, "-i", str(out_dir / "dense"), "-o", str(scene),
                                 "--image-folder", str(out_dir / "dense" / "images"), threads=threads)
        with run.timed("texture"):
            openmvs.run_tool(bin_dir, "TextureMesh", mvs_dir, "-i", str(scene), "-m", str(mesh),
                             "-o", str(out_dir / "textured.mvs"), "--export-type", str(tcfg.texture.export_type),
                             threads=threads)
    except openmvs.OpenMvsError as exc:
        run.downgrade("TextureMesh", "untextured mesh", str(exc))
        return None
    textured = out_dir / f"textured.{tcfg.texture.export_type}"
    return textured if textured.exists() else None
