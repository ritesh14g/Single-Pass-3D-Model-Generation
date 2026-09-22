"""Track B (§7.2, used as the §7.4 hybrid): VGGT depth on Track A's SfM cameras.

Measured on the box (DEVLOG, Esri): VGGT's cameras agree with COLMAP's to 0.7 m over
8-frame windows but break beyond ~16 frames, so VGGT does not carry the flight path.
Track A's SfM does. VGGT supplies depth in short half-overlapping windows, and each
frame's depth is scaled into the SfM frame by the Track A sparse points it observes
(§6.3-style anchoring). Against COLMAP's 1920 px dense depth that measured 1.04% median
error in 5.9 s instead of 1133 s.

Output is a COLMAP-format ``fused.ply`` plus ``fused.ply.vis``, so OpenMVS meshes and
textures it exactly like a COLMAP dense cloud. A point's visibility is its source frame
plus every neighbouring frame whose own anchored depth agrees there; points no
neighbour confirms are dropped, as COLMAP's fusion drops single-view depths.

``TrackBUnavailable`` means "use Track A dense instead"; the caller logs the downgrade.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.device import resolve_device

ROOT = Path(__file__).resolve().parents[2]
# A predictor takes image paths and returns (depth [S,H,W], confidence [S,H,W], rgb [S,H,W,3] uint8).
Predictor = Callable[[list[Path]], tuple[np.ndarray, np.ndarray, np.ndarray]]
_MODELS: dict[str, Any] = {}


class TrackBUnavailable(RuntimeError):
    """VGGT cannot run or cannot be trusted here; fall back to Track A dense."""


def windows_for(n: int, size: int) -> list[tuple[int, int, list[int]]]:
    """Half-overlapping windows; each frame is owned by the window it sits most central in."""
    size = max(min(size, n), 1)
    starts = list(range(0, max(n - size, 0) + 1, max(size // 2, 1)))
    if starts[-1] != n - size:
        starts.append(n - size)
    owner: dict[int, tuple[int, float]] = {}
    for s in starts:
        centre = s + (size - 1) / 2
        for i in range(s, s + size):
            if i not in owner or abs(i - centre) < abs(i - owner[i][1]):
                owner[i] = (s, centre)
    return [(s, s + size, [i for i in range(s, s + size) if owner[i][0] == s]) for s in starts]


def vggt_predictor(bcfg: Any, device: str) -> Predictor:
    """VGGT-1B from Hugging Face, loaded once per process."""
    repo = Path(str(bcfg.repo_dir))
    repo = repo if repo.is_absolute() else ROOT / repo
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as exc:
        raise TrackBUnavailable(f"VGGT not importable ({exc}); clone it into {repo}") from exc

    key = f"{bcfg.weights}@{device}"
    if key not in _MODELS:
        try:
            _MODELS[key] = VGGT.from_pretrained(str(bcfg.weights)).to(device).eval()
        except Exception as exc:  # noqa: BLE001 - download, auth or CUDA problems all mean "fall back"
            raise TrackBUnavailable(f"VGGT weights {bcfg.weights} failed to load: {type(exc).__name__}: {exc}") from exc
    model = _MODELS[key]
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def predict(paths: list[Path]):
        batch = load_and_preprocess_images([str(p) for p in paths])
        if batch.shape[-2] > batch.shape[-1]:
            # "crop" mode centre-crops portrait frames, which breaks the pixel mapping below.
            raise TrackBUnavailable("portrait frames are not supported by the VGGT crop preprocessing")
        try:
            with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
                pred = model(batch.to(device))
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise TrackBUnavailable(f"VGGT out of GPU memory on a {len(paths)}-frame window") from exc
        depth = pred["depth"][0, ..., 0].float().cpu().numpy()
        conf = pred["depth_conf"][0].float().cpu().numpy()
        rgb = (batch.permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
        return depth, conf, rgb

    return predict


def depth_cloud(undist_dir: Path, fused_path: Path, cfg: Any, *, depth_dir: Path | None = None,
                predictor: Predictor | None = None) -> dict[str, Any]:
    """Fuse anchored VGGT depth over the undistorted COLMAP workspace into ``fused_path``."""
    import pycolmap

    bcfg = cfg.get_path("recon.track_b")
    if predictor is None:
        device = resolve_device(str(cfg.get_path("device.prefer", "auto")))
        if device != "cuda" and bool(bcfg.require_gpu):
            raise TrackBUnavailable("no CUDA device (recon.track_b.require_gpu)")
        predictor = vggt_predictor(bcfg, device)

    rec = pycolmap.Reconstruction(str(Path(undist_dir) / "sparse"))
    # fused.ply.vis indexes images in image-id order (verified on the box's COLMAP output).
    images = [rec.images[i] for i in sorted(rec.images)]
    n = len(images)
    if n < 2:
        raise TrackBUnavailable(f"only {n} registered frame(s)")
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy = (float(v) for v in cam.params[:4])  # PINHOLE after undistortion
    poses = [(im.cam_from_world().rotation.matrix(), np.asarray(im.cam_from_world().translation))
             for im in images]
    points3d = {pid: np.asarray(p.xyz) for pid, p in rec.points3D.items()}
    if depth_dir is not None:
        Path(depth_dir).mkdir(parents=True, exist_ok=True)

    frames: dict[int, dict[str, Any]] = {}
    rejected: dict[str, int] = {}
    spreads, anchor_counts = [], []
    vggt_s = 0.0
    for start, stop, owned in windows_for(n, int(bcfg.window_frames)):
        started = time.perf_counter()
        depth, conf, rgb = predictor([Path(undist_dir) / "images" / images[i].name for i in range(start, stop)])
        vggt_s += time.perf_counter() - started
        h, w = depth.shape[1:]
        sx, sy = w / cam.width, h / cam.height
        for i in owned:
            k = i - start
            rot, trans = poses[i]
            uv, z_true = [], []
            for p2d in images[i].points2D:
                if p2d.has_point3D() and p2d.point3D_id in points3d:
                    z = (rot @ points3d[p2d.point3D_id] + trans)[2]
                    if z > 0:
                        uv.append(p2d.xy)
                        z_true.append(z)
            anchor_counts.append(len(uv))
            if len(uv) < int(bcfg.anchor.min_points):
                rejected["too_few_anchors"] = rejected.get("too_few_anchors", 0) + 1
                continue
            uv = np.asarray(uv)
            px = np.clip((uv[:, 0] * sx).astype(int), 0, w - 1)
            py = np.clip((uv[:, 1] * sy).astype(int), 0, h - 1)
            ratio = np.asarray(z_true) / np.maximum(depth[k][py, px], 1e-9)
            scale = float(np.median(ratio))
            spread = 100 * float(np.median(np.abs(ratio / scale - 1)))
            spreads.append(spread)
            if spread > float(bcfg.anchor.max_spread_pct):
                rejected["anchor_spread"] = rejected.get("anchor_spread", 0) + 1
                continue
            scaled = (scale * depth[k]).astype(np.float32)
            keep = conf[k] >= np.quantile(conf[k], float(bcfg.drop_low_conf))
            frames[i] = {"depth": scaled, "keep": keep, "rgb": rgb[k], "sx": sx, "sy": sy}
            if depth_dir is not None and bool(bcfg.keep_confidence_maps):
                np.savez_compressed(Path(depth_dir) / f"{images[i].name}.npz", depth=scaled.astype(np.float16),
                                    conf=conf[k].astype(np.float16), scale=scale)
    if len(frames) < 2:
        raise TrackBUnavailable(f"only {len(frames)} frame(s) anchored ({rejected})")

    pts, normals, colours, vis, before = _fuse(frames, poses, (fx, fy, cx, cy), bcfg)
    if not len(pts):
        raise TrackBUnavailable("no VGGT depth survived the multi-view consistency check")
    write_colmap_fused(fused_path, pts, normals, colours, vis)
    views = np.fromiter((len(v) for v in vis), dtype=np.int32, count=len(vis))
    return {
        "engine": "vggt_hybrid", "weights": str(bcfg.weights), "window": int(bcfg.window_frames),
        "frames": n, "frames_anchored": len(frames), "frames_rejected": rejected,
        "anchors_median": float(np.median(anchor_counts)) if anchor_counts else 0.0,
        "anchor_spread_median_pct": round(float(np.median(spreads)), 2) if spreads else None,
        "points_before_consistency": before, "views_per_point_median": float(np.median(views)),
        "vggt_seconds": round(vggt_s, 1),
    }


def _fuse(frames, poses, intrinsics, bcfg):
    """Back-project every kept frame; keep points that neighbouring frames confirm."""
    fx, fy, cx, cy = intrinsics
    stride = max(int(bcfg.pixel_stride), 1)
    tol = float(bcfg.consistency.rel_tolerance)
    reach = int(bcfg.consistency.neighbours)
    need = int(bcfg.consistency.min_extra_views)
    order = sorted(frames)
    out_pts, out_nrm, out_rgb, out_vis = [], [], [], []
    before = 0
    for i in order:
        f = frames[i]
        rot, trans = poses[i]
        ys, xs = np.nonzero(f["keep"][::stride, ::stride])
        ys, xs = ys * stride, xs * stride
        z = f["depth"][ys, xs].astype(np.float64)
        cam_pts = np.stack([(xs / f["sx"] - cx) / fx * z, (ys / f["sy"] - cy) / fy * z, z], axis=1)
        world = (rot.T @ (cam_pts - trans).T).T
        before += len(world)
        neighbours = [j for j in order if j != i and abs(j - i) <= reach]
        agree = np.zeros((len(world), len(neighbours)), dtype=bool)
        for c, j in enumerate(neighbours):
            g = frames[j]
            rj, tj = poses[j]
            pj = (rj @ world.T).T + tj
            zj = pj[:, 2]
            u = np.round((fx * pj[:, 0] / np.maximum(zj, 1e-9) + cx) * g["sx"]).astype(int)
            v = np.round((fy * pj[:, 1] / np.maximum(zj, 1e-9) + cy) * g["sy"]).astype(int)
            h, w = g["depth"].shape
            inside = (zj > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            ui, vi = u[inside], v[inside]
            seen = g["depth"][vi, ui]
            ok = g["keep"][vi, ui] & (np.abs(seen - zj[inside]) <= tol * zj[inside])
            agree[np.flatnonzero(inside)[ok], c] = True
        keep = agree.sum(1) >= need
        if not keep.any():
            continue
        centre = -rot.T @ trans
        to_cam = centre - world[keep]
        out_pts.append(world[keep])
        out_nrm.append(to_cam / np.linalg.norm(to_cam, axis=1, keepdims=True))
        out_rgb.append(f["rgb"][ys[keep], xs[keep]])
        ids = np.array([i] + neighbours, dtype=np.uint32)
        mask = np.concatenate([np.ones((keep.sum(), 1), bool), agree[keep]], axis=1)
        out_vis.append((ids, mask))
    if not out_pts:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3), np.uint8), [], before
    vis = [ids[row].tolist() for ids, mask in out_vis for row in mask]
    return np.concatenate(out_pts), np.concatenate(out_nrm), np.concatenate(out_rgb), vis, before


def write_colmap_fused(path: Path, pts: np.ndarray, normals: np.ndarray, rgb: np.ndarray,
                       vis: list[list[int]]) -> None:
    """COLMAP's ``fused.ply`` (x y z nx ny nz red green blue) and ``fused.ply.vis``
    (uint64 count, then per point uint32 n + n uint32 image indices in image-id order)."""
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"),
                                       ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for c, key in enumerate("xyz"):
        vertex[key] = pts[:, c]
    for c, key in enumerate(("nx", "ny", "nz")):
        vertex[key] = normals[:, c]
    for c, key in enumerate(("red", "green", "blue")):
        vertex[key] = rgb[:, c]
    PlyData([PlyElement.describe(vertex, "vertex")], byte_order="<").write(str(path))

    # Offsets in int64: mixing uint32 counts with signed aranges promotes to float64.
    counts = np.fromiter((len(v) for v in vis), dtype=np.int64, count=len(vis))
    total = int(counts.sum())
    flat = np.empty(len(vis) + total, dtype=np.uint32)
    starts = np.concatenate([[0], np.cumsum(counts + 1)[:-1]]).astype(np.int64)
    flat[starts] = counts.astype(np.uint32)
    ids = np.fromiter((i for v in vis for i in v), dtype=np.uint32, count=total)
    rank = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    flat[np.repeat(starts + 1, counts) + rank] = ids
    with open(str(path) + ".vis", "wb") as fh:
        fh.write(np.uint64(len(vis)).tobytes())
        fh.write(flat.tobytes())
