"""Stage 4 Track A feasibility probe: pycolmap sparse + dense, OpenMVS texturing.

Runs on the output of ``src.cli run`` (Stages 1-2) and records timings and a metric
check against telemetry, so the laptop CPU run and the GPU box run can be compared
line for line. This is a probe, not the Track A module: it exists to settle the
engine choice and the defaults before ``src/recon/track_a_colmap.py`` is written.

GPU first, CPU fallback: every GPU step is attempted on CUDA when pycolmap was
built with it, and retried on CPU (logged) if it fails. COLMAP's PatchMatch stereo
has no CPU path, so without CUDA dense falls back to OpenMVS DensifyPointCloud.

    python scripts/recon_probe.py --run-dir data/interim/esri --out data/outputs/recon_probe/esri \
        --openmvs-bin tools/openmvs/bin
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.device import cpu_thread_budget, resolve_device  # noqa: E402

TIMINGS: dict[str, float] = {}
NOTES: list[str] = []


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def note(msg: str) -> None:
    NOTES.append(msg)
    log(msg)


class timed:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        log(f"{self.name} ...")

    def __exit__(self, *exc):
        TIMINGS[self.name] = round(time.perf_counter() - self.t0, 1)
        log(f"{self.name} done in {TIMINGS[self.name]} s")


def gpu_then_cpu(name: str, fn, use_cuda: bool):
    """Run ``fn(device)`` on CUDA, and once more on CPU if that raises."""
    if use_cuda:
        try:
            with timed(name):
                return fn(pycolmap.Device.cuda)
        except Exception as exc:  # noqa: BLE001 - any GPU failure downgrades, never crashes
            note(f"DOWNGRADE {name}: CUDA failed ({exc!r}); retrying on CPU")
    with timed(name):
        return fn(pycolmap.Device.cpu)


def read_geo(path: Path) -> dict[str, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    lon = np.array([float(r[1]) for r in rows])
    lat = np.array([float(r[2]) for r in rows])
    alt = np.array([float(r[3]) for r in rows])
    radius = 6378137.0
    east = np.radians(lon - lon[0]) * radius * math.cos(math.radians(lat[0]))
    north = np.radians(lat - lat[0]) * radius
    return {r[0]: np.array([east[i], north[i], alt[i] - alt[0]]) for i, r in enumerate(rows)}


def telemetry_hints(run_dir: Path) -> dict[str, float]:
    """Horizontal FOV and expected height above ground, when the telemetry carries them."""
    hints: dict[str, float] = {}
    path = run_dir / "ingest" / "telemetry.parquet"
    if not path.exists():
        return hints
    tel = pd.read_parquet(path)
    if "hfov_deg" in tel and tel["hfov_deg"].notna().any():
        hints["hfov_deg"] = float(tel["hfov_deg"].median())
    if {"alt_gps", "frame_center_alt"} <= set(tel.columns) and tel["frame_center_alt"].notna().any():
        hints["expected_agl_m"] = float((tel["alt_gps"] - tel["frame_center_alt"]).median())
    return hints


def umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cs, cd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(cd.T @ cs / len(src))
    s_ = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_[2, 2] = -1
    rot = u @ s_ @ vt
    scale = np.trace(np.diag(d) @ s_) / cs.var(0).sum()
    return scale, rot, mu_d - scale * rot @ mu_s


def metric_check(rec: pycolmap.Reconstruction, gps: dict[str, np.ndarray]) -> dict[str, float]:
    ims = [im for im in rec.images.values() if im.name in gps]
    cams = np.array([im.projection_center() for im in ims])
    ref = np.array([gps[im.name] for im in ims])
    scale, rot, trans = umeyama(cams, ref)
    aligned = (scale * (rot @ cams.T)).T + trans
    pts = (scale * (rot @ np.array([p.xyz for p in rec.points3D.values()]).T)).T + trans
    heights = []
    for c in aligned:
        near = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) < 30
        if near.sum() > 20:
            heights.append(c[2] - np.median(pts[near, 2]))
    return {
        "cam_vs_gps_rms_m": round(float(np.sqrt(((aligned - ref) ** 2).sum(1).mean())), 2),
        "height_above_ground_m": round(float(np.median(heights)), 1) if heights else float("nan"),
    }


def run_openmvs(bin_dir: Path, tool: str, work: Path, *args: str, threads: int) -> None:
    exe = bin_dir / (tool + (".exe" if platform.system() == "Windows" else ""))
    # OpenMVS resolves -i/-o against -w, so everything passed here is absolute.
    cmd = [str(exe), "-w", str(work), "--max-threads", str(threads), *args]
    with timed(f"openmvs_{tool}"):
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logs = sorted(work.glob(f"{tool}-*.log"))
        tail = logs[-1].read_text(errors="ignore").splitlines()[-15:] if logs else [result.stderr]
        raise RuntimeError(f"{tool} exited {result.returncode}:\n" + "\n".join(tail))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="output folder of `src.cli run`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--openmvs-bin", type=Path, default=None, help="folder with OpenMVS executables")
    ap.add_argument("--sift-size", type=int, default=1600, help="max image side for SIFT")
    ap.add_argument("--dense-size", type=int, default=1920, help="max image side for dense stereo")
    ap.add_argument("--poisson-depth", type=int, default=11,
                    help="COLMAP's default 13 gave 5.4 M faces on 43 small frames and stalled texturing")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--no-texture", action="store_true")
    args = ap.parse_args()

    run_dir, out = args.run_dir.resolve(), args.out.resolve()
    images = run_dir / "condition" / "images"
    geo_path = run_dir / "condition" / "geo.txt"
    if not images.is_dir():
        log(f"no conditioned images at {images}; run `python -m src.cli run <video> --out {args.run_dir}` first")
        return 2
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    bin_dir = args.openmvs_bin.resolve() if args.openmvs_bin else None

    threads = cpu_thread_budget()
    # CUDA needs both a visible GPU (resolve_device) and a CUDA build of pycolmap
    # (`pip install pycolmap-cuda12`); the plain `pycolmap` wheel is CPU-only.
    gpu_visible = resolve_device(args.device) == "cuda"
    use_cuda = gpu_visible and bool(pycolmap.has_cuda)
    if gpu_visible and not pycolmap.has_cuda:
        note("DOWNGRADE: GPU visible but pycolmap has no CUDA (install pycolmap-cuda12); running on CPU")
    env = {"pycolmap": pycolmap.__version__, "pycolmap_cuda": bool(pycolmap.has_cuda),
           "gpu_visible": gpu_visible, "using_cuda": use_cuda, "cpu_threads": threads, "platform": platform.platform()}
    log(f"environment {env}")
    n_images = len(list(images.glob("*.jpg")))
    hints = telemetry_hints(run_dir)
    log(f"{n_images} conditioned frames; telemetry hints {hints}")

    # ---- sparse -------------------------------------------------------------
    db = out / "database.db"
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = args.sift_size
    extraction.num_threads = threads

    def extract(device):
        if db.exists():
            db.unlink()
        pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction, device=device)
    gpu_then_cpu("sparse_extract", extract, use_cuda)

    # Seed the focal length from telemetry and hold it: on a constant-height nadir
    # flight the images only fix height/focal, so a refined focal drifts (measured on
    # Esri: refined -> 92 m above ground, telemetry-fixed -> 107 m, truth ~111 m).
    fix_focal = "hfov_deg" in hints
    if fix_focal:
        handle = pycolmap.Database.open(str(db))
        cam = handle.read_all_cameras()[0]
        focal = (cam.width / 2) / math.tan(math.radians(hints["hfov_deg"] / 2))
        cam.params = [focal, cam.width / 2, cam.height / 2, 0.0]
        cam.has_prior_focal_length = True
        handle.update_camera(cam)
        handle.close()
        note(f"focal seeded from telemetry HFOV {hints['hfov_deg']:.2f} deg -> {focal:.0f} px, held fixed")
    else:
        note("no telemetry FOV: focal is self-calibrated (expect height/focal drift on nadir flights)")

    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10
    pairing.quadratic_overlap = True
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = threads
    gpu_then_cpu("sparse_match",
                 lambda device: pycolmap.match_sequential(db, matching_options=matching,
                                                          pairing_options=pairing, device=device),
                 use_cuda)

    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.num_threads = threads
    mapper.ba_refine_focal_length = not fix_focal
    sparse_dir = out / "sparse"
    sparse_dir.mkdir()
    with timed("sparse_map"):
        recs = pycolmap.incremental_mapping(db, images, sparse_dir, options=mapper)
    if not recs:
        log("SfM produced no model")
        return 1
    best_id, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    sparse_model = sparse_dir / str(best_id)
    sparse_stats = {
        "models": len(recs), "registered": rec.num_reg_images(), "images": n_images,
        "points": rec.num_points3D(),
        "reproj_px": round(rec.compute_mean_reprojection_error(), 3),
        "track_length": round(rec.compute_mean_track_length(), 2),
        "focal_px": round(float(next(iter(rec.cameras.values())).params[0]), 1),
    }
    if geo_path.exists():
        sparse_stats.update(metric_check(rec, read_geo(geo_path)))
    if "expected_agl_m" in hints:
        sparse_stats["expected_agl_m"] = round(hints["expected_agl_m"], 1)
    log(f"sparse {sparse_stats}")

    # ---- dense --------------------------------------------------------------
    undist = out / "dense"
    with timed("undistort"):
        pycolmap.undistort_images(undist, sparse_model, images,
                                  undistort_options=_undistort_options(args.dense_size))
    fused = undist / "fused.ply"
    dense_engine = None
    if use_cuda:
        try:
            pm = pycolmap.PatchMatchOptions()
            pm.max_image_size = args.dense_size
            pm.gpu_index = "0"
            with timed("dense_patchmatch_gpu"):
                pycolmap.patch_match_stereo(undist, options=pm)
            fusion = pycolmap.StereoFusionOptions()
            fusion.num_threads = threads
            with timed("dense_fusion"):
                pycolmap.stereo_fusion(fused, undist, options=fusion, output_type="PLY")
            dense_engine = "colmap_patchmatch_cuda"
        except Exception as exc:  # noqa: BLE001
            note(f"DOWNGRADE dense: COLMAP PatchMatch on CUDA failed ({exc!r})")
    if dense_engine is None:
        if bin_dir is None:
            note("dense skipped: no CUDA and no --openmvs-bin for the CPU path")
        else:
            note("dense on CPU via OpenMVS DensifyPointCloud")
            mvs = out / "mvs"
            mvs.mkdir()
            run_openmvs(bin_dir, "InterfaceCOLMAP", mvs, "-i", str(undist), "-o", str(mvs / "scene.mvs"),
                        "--image-folder", str(undist / "images"), threads=threads)
            run_openmvs(bin_dir, "DensifyPointCloud", mvs, "-i", str(mvs / "scene.mvs"),
                        "-o", str(mvs / "scene_dense.mvs"), threads=threads)
            shutil.copy(mvs / "scene_dense.ply", fused)
            dense_engine = "openmvs_cpu"

    mesh = out / "mesh.ply"
    if dense_engine:
        with timed("mesh_poisson"):
            poisson = pycolmap.PoissonMeshingOptions()
            poisson.depth = args.poisson_depth
            poisson.num_threads = threads
            pycolmap.poisson_meshing(fused, mesh, options=poisson)

    # ---- texture ------------------------------------------------------------
    textured = None
    if dense_engine and bin_dir is not None and not args.no_texture:
        mvs = out / "mvs"
        mvs.mkdir(exist_ok=True)
        if not (mvs / "scene.mvs").exists():
            run_openmvs(bin_dir, "InterfaceCOLMAP", mvs, "-i", str(undist), "-o", str(mvs / "scene.mvs"),
                        "--image-folder", str(undist / "images"), threads=threads)
        run_openmvs(bin_dir, "TextureMesh", mvs, "-i", str(mvs / "scene.mvs"), "-m", str(mesh),
                    "-o", str(out / "textured.mvs"), "--export-type", "obj", threads=threads)
        textured = out / "textured.obj"

    summary = {
        "environment": env, "telemetry_hints": hints, "sparse": sparse_stats,
        "dense_engine": dense_engine, "timings_s": TIMINGS,
        "total_s": round(sum(TIMINGS.values()), 1), "notes": NOTES,
        "outputs": {k: str(v) for k, v in {"sparse": sparse_model, "dense": fused, "mesh": mesh,
                                           "textured": textured}.items() if v is not None and Path(v).exists()},
    }
    (out / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== PASTE EVERYTHING BELOW BACK TO CLAUDE =====")
    print(json.dumps(summary, indent=2))
    return 0


def _undistort_options(max_size: int):
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = max_size
    return opts


if __name__ == "__main__":
    raise SystemExit(main())
