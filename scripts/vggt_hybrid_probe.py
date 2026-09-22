"""Hybrid Track B probe: VGGT depth on Track A's camera poses (spec §7.4 "Track B depth fills").

The window probe showed VGGT's cameras match COLMAP's to 0.7 m over 8-frame windows but
break beyond ~16 frames (DEVLOG). So VGGT does not carry the flight path here: Track A's
SfM does (~32 s on the box). VGGT supplies depth, in short windows, and each depth map is
scaled into the SfM frame by the Track A sparse points that frame observes -- the same
anchoring §6.3 uses for monocular depth.

It measures, per frame, VGGT depth against COLMAP's geometric depth map (the 19-minute
Track A dense result, kept on the box) and writes the fused hybrid cloud:

    python scripts/vggt_hybrid_probe.py --run-dir data/interim/esri_gpu \\
        --colmap-dense data/outputs/recon_probe/esri_gpu/dense --out data/outputs/vggt_probe/esri_hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.core.device import cpu_thread_budget, resolve_device  # noqa: E402
from src.recon import alignment  # noqa: E402
from src.recon.track_b_vggt import preprocess, preprocess_balanced  # noqa: E402


def log(msg: str) -> None:
    print(f"[hybrid] {msg}", flush=True)


def read_colmap_array(path: Path) -> np.ndarray:
    """COLMAP depth/normal map: ``w&h&c&`` header, then float32 in column-major order."""
    with open(path, "rb") as fh:
        header = b""
        while header.count(b"&") < 3:
            header += fh.read(1)
        width, height, channels = (int(v) for v in header.split(b"&")[:3])
        data = np.fromfile(fh, np.float32)
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()


def windows_for(n: int, size: int) -> list[tuple[int, int, list[int]]]:
    """Half-overlapping windows; each frame is assigned to the window it sits most central in."""
    size = min(size, n)
    starts = list(range(0, max(n - size, 0) + 1, max(size // 2, 1)))
    if starts[-1] != n - size:
        starts.append(n - size)
    owner = {}
    for s in starts:
        centre = s + (size - 1) / 2
        for i in range(s, s + size):
            if i not in owner or abs(i - centre) < abs(i - owner[i][1]):
                owner[i] = (s, centre)
    return [(s, s + size, [i for i in range(s, s + size) if owner[i][0] == s]) for s in starts]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="src.cli run folder (for geo.txt)")
    ap.add_argument("--colmap-dense", type=Path, required=True,
                    help="COLMAP undistorted workspace: images/, sparse/, stereo/depth_maps/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=["vggt", "vggt_omega"], default="vggt",
                    help="vggt_omega: --widths is Omega's image_resolution (512 -> 688x384 on 16:9)")
    ap.add_argument("--weights", default="facebook/VGGT-1B")
    ap.add_argument("--omega-checkpoint", default="vggt_omega_1b_512.pt")
    ap.add_argument("--omega-repo", type=Path, default=ROOT / "tools" / "vggt_omega")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--vggt-repo", type=Path, default=ROOT / "tools" / "vggt")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--widths", default="518", help="VGGT input widths to compare, e.g. 518,700,1036")
    ap.add_argument("--drop-low-conf", type=float, default=0.3, help="drop this fraction of least-confident pixels")
    ap.add_argument("--max-frames", type=int, default=0, help="limit frames (plumbing tests)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    sys.path.insert(0, str(args.vggt_repo.resolve()))
    import cv2
    import pycolmap
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt_probe import load_model, run_chunk

    torch.set_num_threads(cpu_thread_budget())
    device = resolve_device(args.device)
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) \
        if device == "cuda" else torch.float32
    args.out.mkdir(parents=True, exist_ok=True)

    rec = pycolmap.Reconstruction(str(args.colmap_dense / "sparse"))
    images = sorted(rec.images.values(), key=lambda im: im.name)
    if args.max_frames:
        images = images[:args.max_frames]
    names = [im.name for im in images]
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy = (float(v) for v in cam.params[:4])  # PINHOLE after undistortion
    points3d = {pid: np.asarray(p.xyz) for pid, p in rec.points3D.items()}

    geo = args.run_dir / "condition" / "geo.txt"
    gps = alignment.read_geo_enu(geo) if geo.exists() else {}
    centres = np.array([im.projection_center() for im in images])
    metric, transform = alignment.metric_check(names, centres, np.empty((0, 3)), gps)
    to_metres = transform[0] if transform is not None else None
    log(f"{len(names)} frames, camera {cam.width}x{cam.height}, GPS fit {metric}")

    if args.model == "vggt_omega":
        sys.path.insert(0, str(args.omega_repo.resolve()))
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera

        model = VGGTOmega(autocast=device == "cuda")
        if not args.random_weights:
            from huggingface_hub import hf_hub_download

            state = torch.load(hf_hub_download("facebook/VGGT-Omega", args.omega_checkpoint),
                               map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        log(f"VGGT-Omega loaded ({args.omega_checkpoint}{', RANDOM WEIGHTS' if args.random_weights else ''})")
    else:
        model, torch = load_model(args, device)

    def forward(paths, width):
        """(depth [S,H,W], conf [S,H,W], extrinsics [S,3,4], batch, seconds, peak GB) for either model."""
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        if args.model == "vggt_omega":
            batch = preprocess_balanced(paths, width)
            with torch.inference_mode():
                pred = model(batch.to(device))
            ext = encoding_to_camera(pred["pose_enc"], batch.shape[-2:])[0][0]
        else:
            batch = preprocess(paths, width)
            pred, _, _ = run_chunk(model, torch, batch, device, dtype)
            ext = pose_encoding_to_extri_intri(pred["pose_enc"], batch.shape[-2:])[0][0]
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        return (pred["depth"][0, ..., 0].float().cpu().numpy(), pred["depth_conf"][0].float().cpu().numpy(),
                ext.float().cpu().numpy().astype(np.float64), batch, seconds, peak)

    def run_width(width: int) -> dict:
        t_vggt, t_fuse, peak_gb = 0.0, 0.0, 0.0
        per_frame, cloud_pts, cloud_rgb = [], [], []
        for start, stop, owned in windows_for(len(names), args.window):
            depth_all, conf_all, ext, batch, seconds, peak = forward(
                [args.colmap_dense / "images" / n for n in names[start:stop]], width)
            t_vggt += seconds
            peak_gb = max(peak_gb, peak)
            started = time.perf_counter()
            h, w = batch.shape[-2:]
            sx, sy = w / cam.width, h / cam.height

            # Window scale from cameras, rotation-aware (a straight flight line leaves the roll
            # about the path undetermined by centres alone): R from paired orientations, then s.
            rot_c = np.stack([images[i].cam_from_world().rotation.matrix() for i in range(start, stop)])
            rot_v = ext[:, :, :3]
            c_c = centres[start:stop]
            c_v = np.stack([-e[:, :3].T @ e[:, 3] for e in ext])
            u, _, vt = np.linalg.svd(sum(rc.T @ rv for rc, rv in zip(rot_c, rot_v)))
            r_align = u @ vt
            dc, dv = c_c - c_c.mean(0), (r_align @ (c_v - c_v.mean(0)).T).T
            s_cams = float((dc * dv).sum() / max((dv * dv).sum(), 1e-12))

            for i in owned:
                k = i - start
                im = images[i]
                depth = depth_all[k].astype(np.float64)
                conf = conf_all[k]
                keep = conf >= np.quantile(conf, args.drop_low_conf)

                # Per-frame anchoring on the sparse points this frame observes.
                pose = im.cam_from_world()
                r, t = pose.rotation.matrix(), np.asarray(pose.translation)
                uv, z_true = [], []
                for p2d in im.points2D:
                    if p2d.has_point3D() and p2d.point3D_id in points3d:
                        xc = r @ points3d[p2d.point3D_id] + t
                        if xc[2] > 0:
                            uv.append(p2d.xy)
                            z_true.append(xc[2])
                row = {"frame": im.name, "anchors": len(uv), "s_cams": s_cams}
                if to_metres:
                    # Ground spacing of one depth pixel: depth / focal at this resolution.
                    row["gsd_cm"] = round(100 * float(np.median(depth)) * (s_cams or 1) * to_metres / (fx * sx), 1)
                if len(uv) >= 10:
                    uv = np.array(uv)
                    px = np.clip((uv[:, 0] * sx).astype(int), 0, w - 1)
                    py = np.clip((uv[:, 1] * sy).astype(int), 0, h - 1)
                    ratio = np.array(z_true) / np.maximum(depth[py, px], 1e-9)
                    s_frame = float(np.median(ratio))
                    row["s_frame"] = s_frame
                    row["anchor_spread_pct"] = round(100 * float(np.median(np.abs(ratio / s_frame - 1))), 2)
                else:
                    s_frame = s_cams
                    row["s_frame"] = None

                dm_path = args.colmap_dense / "stereo" / "depth_maps" / f"{im.name}.geometric.bin"
                if dm_path.exists():
                    truth = cv2.resize(read_colmap_array(dm_path), (w, h), interpolation=cv2.INTER_NEAREST)
                    valid = (truth > 0) & keep
                    row["colmap_valid_pct"] = round(100 * float((truth > 0).mean()), 1)
                    if valid.sum() > 100:
                        for label, s in (("frame", s_frame), ("cams", s_cams)):
                            rel = np.abs(s * depth[valid] - truth[valid]) / truth[valid]
                            row[f"rel_err_median_pct_{label}"] = round(100 * float(np.median(rel)), 2)
                            row[f"within_5pct_{label}"] = round(100 * float((rel < 0.05).mean()), 1)
                            row[f"within_10pct_{label}"] = round(100 * float((rel < 0.10).mean()), 1)
                            if to_metres:
                                err_m = np.abs(s * depth[valid] - truth[valid]) * to_metres
                                row[f"abs_err_median_m_{label}"] = round(float(np.median(err_m)), 2)

                # Back-project confident pixels with Track A's pose and intrinsics, VGGT's scaled depth.
                ys, xs = np.nonzero(keep[::2, ::2])
                ys, xs = ys * 2, xs * 2
                z = s_frame * depth[ys, xs]
                xc = np.stack([(xs / sx - cx) / fx * z, (ys / sy - cy) / fy * z, z], axis=1)
                cloud_pts.append((r.T @ (xc - t).T).T)
                rgb = (batch[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cloud_rgb.append(rgb[ys, xs])
                per_frame.append(row)
            t_fuse += time.perf_counter() - started
            log(f"window {start}-{stop}: {seconds:.2f} s, s_cams {s_cams:.4f}")

        pts, rgb = np.concatenate(cloud_pts), np.concatenate(cloud_rgb)
        from plyfile import PlyData, PlyElement

        vertex = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        for i, key in enumerate("xyz"):
            vertex[key] = pts[:, i]
        for i, key in enumerate(("red", "green", "blue")):
            vertex[key] = rgb[:, i]
        ply = args.out / f"hybrid_cloud_{width}.ply"
        PlyData([PlyElement.describe(vertex, "vertex")]).write(str(ply))

        def med(key):
            vals = [r[key] for r in per_frame if r.get(key) is not None]
            return round(float(np.median(vals)), 2) if vals else None

        summary = {
            "model": args.model, "width": width, "input_hw": [int(h), int(w)], "frames": len(names), "window": args.window,
            "peak_gpu_gb": round(peak_gb, 2),
            "vggt_seconds": round(t_vggt, 1), "vggt_s_per_frame": round(t_vggt / max(len(names), 1), 3),
            "fuse_seconds": round(t_fuse, 1), "cloud_points": int(len(pts)),
            "footprint_m2": (round(alignment.footprint_m2(pts, transform)) if transform is not None else None),
            "colmap_dense_footprint_m2_for_reference": 60749,
            "median": {k: med(k) for k in (
                "gsd_cm", "anchors", "anchor_spread_pct", "colmap_valid_pct",
                "rel_err_median_pct_frame", "within_5pct_frame", "within_10pct_frame", "abs_err_median_m_frame",
                "rel_err_median_pct_cams", "within_5pct_cams", "within_10pct_cams", "abs_err_median_m_cams")},
            "worst_frame_rel_err_pct": max((r.get("rel_err_median_pct_frame") or 0) for r in per_frame),
            "cloud": str(ply),
        }
        (args.out / f"hybrid_frames_{width}.json").write_text(json.dumps(per_frame, indent=2))
        return summary

    results = {"gps_fit": metric, "device": device, "colmap_dense_footprint_m2_for_reference": 60749, "widths": {}}
    for width in [int(v) for v in args.widths.split(",") if v.strip()]:
        try:
            results["widths"][width] = run_width(width)
            log(f"width {width}: {results['widths'][width]['median']}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results["widths"][width] = {"oom": True}
            log(f"width {width}: out of GPU memory with {args.window}-frame windows")
    (args.out / "hybrid_summary.json").write_text(json.dumps(results, indent=2))
    summary = results
    print("\n===== PASTE EVERYTHING BELOW BACK TO CLAUDE =====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
