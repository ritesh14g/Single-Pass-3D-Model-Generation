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
LOG_PATH: Path | None = None


def log(msg: str) -> None:
    line = f"[probe] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


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


def metric_check(rec: pycolmap.Reconstruction, gps: dict[str, np.ndarray]):
    """Fit the model to GPS (similarity) and measure it; returns (stats, transform or None)."""
    ims = [im for im in rec.images.values() if im.name in gps]
    if len(ims) < 3:  # a similarity fit needs three non-collinear positions
        return {"gps_matched_images": len(ims)}, None
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
    }, (scale, rot, trans)


def dense_stats(fused: Path, transform, cell_m: float = 1.0) -> dict[str, float]:
    """Point count and, when the model is GPS-aligned, the ground area the cloud covers.

    Footprint = occupied ``cell_m`` x ``cell_m`` cells in the GPS east/north plane. It is
    the completeness number the dense sweep trades against time.
    """
    from plyfile import PlyData

    vertex = PlyData.read(str(fused))["vertex"]
    xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)
    stats: dict[str, float] = {"points": int(len(xyz))}
    if transform is not None and len(xyz):
        scale, rot, trans = transform
        east_north = ((scale * (rot @ xyz.T)).T + trans)[:, :2]
        cells = np.unique(np.floor(east_north / cell_m).astype(np.int64), axis=0)
        stats["footprint_m2"] = round(float(len(cells)) * cell_m * cell_m, 0)
    return stats


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


def run_sparse(out: Path, images: Path, sparse_dir: Path, hints: dict, fix_focal: bool,
               threads: int, use_cuda: bool, args) -> dict:
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
    sparse_dir.mkdir(exist_ok=True)
    with timed("sparse_map"):
        return pycolmap.incremental_mapping(db, images, sparse_dir, options=mapper)


def ply_counts(path: Path) -> dict[str, int]:
    """Vertex and face counts from a PLY header, without loading the mesh."""
    counts: dict[str, int] = {}
    with path.open("rb") as fh:
        for raw in fh:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element "):
                _, name, n = line.split()
                counts[{"vertex": "vertices", "face": "faces"}.get(name, name)] = int(n)
            if line == "end_header":
                break
    return counts


def clean_mesh(src: Path, dst: Path, min_component_fraction: float) -> dict[str, int]:
    """Drop non-finite vertices, degenerate faces and small fragments before texturing.

    On the box, Poisson on the Esri cloud gave 1.75 M vertices for 1.82 M faces (a clean
    surface has ~2 faces per vertex) and NaN vertex coordinates; TextureMesh segfaulted
    on it and ``trimesh.split`` stalled. Components are found with one sparse
    connected-components pass over the vertex graph, which is linear in the mesh size.
    """
    import trimesh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mesh = trimesh.load(src, process=False)
    stats = {"vertices_in": len(mesh.vertices), "faces_in": len(mesh.faces)}
    finite = np.isfinite(mesh.vertices).all(axis=1)
    stats["nonfinite_vertices"] = int((~finite).sum())
    faces = np.asarray(mesh.faces)
    keep = finite[faces].all(axis=1)
    keep &= (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[keep]

    n = len(mesh.vertices)
    rows = np.concatenate([faces[:, 0], faces[:, 1]])
    cols = np.concatenate([faces[:, 1], faces[:, 2]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    counts = np.bincount(face_label, minlength=n_comp)
    big = counts >= min_component_fraction * counts.max()
    stats.update(components_in=int((counts > 0).sum()), components_kept=int(big.sum()))

    mask = keep.copy()
    mask[keep] = big[face_label]
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    mesh.export(dst)
    stats.update(vertices_out=len(mesh.vertices), faces_out=len(mesh.faces))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="output folder of `src.cli run`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--openmvs-bin", type=Path, default=None, help="folder with OpenMVS executables")
    ap.add_argument("--sift-size", type=int, default=1600, help="max image side for SIFT")
    ap.add_argument("--dense-size", type=int, default=1920, help="max image side for dense stereo")
    # PatchMatch cost ~ pixels x source images x iterations x (2 with geometric consistency).
    # COLMAP's quality defaults took 1133 s for 45 frames on the MIG slice.
    ap.add_argument("--pm-src-images", type=int, default=20, help="source views per reference image")
    ap.add_argument("--pm-iterations", type=int, default=5)
    ap.add_argument("--pm-window-step", type=int, default=1, help="2 samples every other pixel in the window")
    ap.add_argument("--pm-no-geom", action="store_true", help="skip the geometric-consistency pass")
    ap.add_argument("--poisson-depth", type=int, default=11,
                    help="COLMAP's default 13 gave 5.4 M faces on 43 small frames and stalled texturing")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--min-component", type=float, default=0.01,  # Poisson fallback only
                    help="drop mesh fragments smaller than this fraction of the largest piece")
    ap.add_argument("--no-texture", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="keep --out and reuse its sparse model and dense cloud; redo mesh and texture")
    ap.add_argument("--redo-dense", action="store_true",
                    help="with --reuse: keep the sparse model, recompute dense, mesh and texture")
    args = ap.parse_args()

    run_dir, out = args.run_dir.resolve(), args.out.resolve()
    images = run_dir / "condition" / "images"
    geo_path = run_dir / "condition" / "geo.txt"
    if not images.is_dir():
        log(f"no conditioned images at {images}; run `python -m src.cli run <video> --out {args.run_dir}` first")
        return 2
    if out.exists() and not args.reuse:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = out / "probe.log"
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
    sparse_dir = out / "sparse"
    fix_focal = "hfov_deg" in hints
    reused = [d for d in sparse_dir.glob("*") if (d / "images.bin").exists()] if args.reuse else []
    if reused:
        recs = {int(d.name): pycolmap.Reconstruction(str(d)) for d in reused}
        note(f"reused sparse model(s) from {sparse_dir}")
    else:
        recs = run_sparse(out, images, sparse_dir, hints, fix_focal, threads, use_cuda, args)
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
    transform = None
    if geo_path.exists():
        metric, transform = metric_check(rec, read_geo(geo_path))
        sparse_stats.update(metric)
    if "expected_agl_m" in hints:
        sparse_stats["expected_agl_m"] = round(hints["expected_agl_m"], 1)
    log(f"sparse {sparse_stats}")

    # ---- dense --------------------------------------------------------------
    undist = out / "dense"
    fused = undist / "fused.ply"
    dense_engine = None
    dense_params = {"size": args.dense_size, "src_images": args.pm_src_images,
                    "iterations": args.pm_iterations, "window_step": args.pm_window_step,
                    "geom_consistency": not args.pm_no_geom}
    if args.reuse and fused.exists() and not args.redo_dense:
        dense_engine = "reused"
        note(f"reused dense cloud {fused}")
    else:
        for stale in (undist, out / "mvs"):
            if stale.exists():
                shutil.rmtree(stale)
        with timed("undistort"):
            pycolmap.undistort_images(undist, sparse_model, images,
                                      num_patch_match_src_images=args.pm_src_images,
                                      undistort_options=_undistort_options(args.dense_size),
                                      num_threads=threads)
    if dense_engine is None and use_cuda:
        try:
            pm = pycolmap.PatchMatchOptions()
            pm.max_image_size = args.dense_size
            pm.gpu_index = "0"
            pm.num_iterations = args.pm_iterations
            pm.window_step = args.pm_window_step
            pm.geom_consistency = not args.pm_no_geom
            with timed("dense_patchmatch_gpu"):
                pycolmap.patch_match_stereo(undist, options=pm)
            fusion = pycolmap.StereoFusionOptions()
            fusion.num_threads = threads
            with timed("dense_fusion"):
                pycolmap.stereo_fusion(fused, undist, options=fusion, output_type="PLY",
                                       input_type="photometric" if args.pm_no_geom else "geometric")
            dense_engine = "colmap_patchmatch_cuda"
        except Exception as exc:  # noqa: BLE001
            note(f"DOWNGRADE dense: COLMAP PatchMatch on CUDA failed ({exc!r})")
    if dense_engine is None:
        if bin_dir is None:
            note("dense skipped: no CUDA and no --openmvs-bin for the CPU path")
        else:
            note("dense on CPU via OpenMVS DensifyPointCloud")
            mvs = out / "mvs"
            mvs.mkdir(exist_ok=True)
            run_openmvs(bin_dir, "InterfaceCOLMAP", mvs, "-i", str(undist), "-o", str(mvs / "scene.mvs"),
                        "--image-folder", str(undist / "images"), threads=threads)
            run_openmvs(bin_dir, "DensifyPointCloud", mvs, "-i", str(mvs / "scene.mvs"),
                        "-o", str(mvs / "scene_dense.mvs"), threads=threads)
            shutil.copy(mvs / "scene_dense.ply", fused)
            dense_engine = "openmvs_cpu"

    dense = {"params": "reused" if dense_engine == "reused" else dense_params}
    if dense_engine and fused.exists():
        dense.update(dense_stats(fused, transform))
        log(f"dense {dense}")

    # ---- mesh ---------------------------------------------------------------
    # OpenMVS's Delaunay mesher is the default. On the box, pycolmap-cuda12's Poisson
    # turned a clean 447 k-point cloud into 121 k fragments with NaN vertices and
    # TextureMesh segfaulted on it; the same cloud meshed cleanly on the laptop, and
    # OpenMVS meshed it into 659 k faces in 18 s (S4-6). Poisson is the fallback.
    mvs = out / "mvs"
    mesh = out / "mesh.ply"
    scene_dense = mvs / "scene_dense.mvs"
    mesh_stats: dict[str, int] = {}
    mesher = None
    if dense_engine and bin_dir is not None:
        mvs.mkdir(exist_ok=True)
        try:
            if not scene_dense.exists():  # COLMAP's cloud, imported with its per-point visibility
                run_openmvs(bin_dir, "InterfaceCOLMAP", mvs, "-i", str(undist), "-p", str(fused),
                            "-o", str(scene_dense), "--image-folder", str(undist / "images"),
                            threads=threads)
            run_openmvs(bin_dir, "ReconstructMesh", mvs, "-i", str(scene_dense), "-o", str(mesh),
                        threads=threads)
            mesher = "openmvs_delaunay"
            mesh_stats = ply_counts(mesh)
        except Exception as exc:  # noqa: BLE001
            note(f"DOWNGRADE mesh: OpenMVS ReconstructMesh failed ({str(exc).splitlines()[0]}); using Poisson")
    if dense_engine and mesher is None:
        raw = out / "mesh_poisson.ply"
        with timed("mesh_poisson"):
            poisson = pycolmap.PoissonMeshingOptions()
            poisson.depth = args.poisson_depth
            poisson.num_threads = threads
            pycolmap.poisson_meshing(fused, raw, options=poisson)
        with timed("mesh_clean"):
            mesh_stats = clean_mesh(raw, mesh, args.min_component)
        mesher = "colmap_poisson"
    if mesher:
        log(f"mesh ({mesher}) {mesh_stats}")

    # ---- texture ------------------------------------------------------------
    textured = None
    if mesher and bin_dir is not None and not args.no_texture:
        mvs.mkdir(exist_ok=True)
        try:
            scene = scene_dense
            if not scene.exists():
                scene = mvs / "scene.mvs"
                run_openmvs(bin_dir, "InterfaceCOLMAP", mvs, "-i", str(undist), "-o", str(scene),
                            "--image-folder", str(undist / "images"), threads=threads)
            run_openmvs(bin_dir, "TextureMesh", mvs, "-i", str(scene), "-m", str(mesh),
                        "-o", str(out / "textured.mvs"), "--export-type", "obj", threads=threads)
            textured = out / "textured.obj"
        except Exception as exc:  # noqa: BLE001 - the untextured mesh is still a result
            note(f"DOWNGRADE texture: {str(exc).splitlines()[0]}; keeping untextured mesh.ply")

    summary = {
        "environment": env, "telemetry_hints": hints, "sparse": sparse_stats,
        "dense_engine": dense_engine, "dense": dense, "mesher": mesher, "mesh": mesh_stats, "timings_s": TIMINGS,
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
