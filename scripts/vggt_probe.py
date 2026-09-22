"""Stage 4 Track B feasibility probe: VGGT on conditioned frames (spec §7.2).

Answers, on the box, before any Track B module is written:
  * does VGGT load (weights, licence) and fit the 20 GB MIG slice, and at what chunk size?
  * how long does one chunk take, and how does that scale with frames?
  * how good is it here: focal vs telemetry, camera centres vs GPS and vs Track A's
    COLMAP cameras, height above ground from its depth vs telemetry?

GPU first, CPU fallback: on CPU (the laptop) it runs a 2-frame smoke test only, which is
what ``--random-weights`` is for: it exercises every code path without the 5 GB download.

    python scripts/vggt_probe.py --run-dir data/interim/esri_gpu --out data/outputs/vggt_probe/esri \\
        --colmap-sparse data/outputs/recon_probe/esri_gpu/sparse/1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.core.device import cpu_thread_budget, resolve_device  # noqa: E402
from recon_probe import read_geo, telemetry_hints, umeyama  # noqa: E402

NOTES: list[str] = []


def log(msg: str) -> None:
    print(f"[vggt] {msg}", flush=True)


def note(msg: str) -> None:
    NOTES.append(msg)
    log(msg)


def licence_info(repo_id: str) -> dict:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id)
        card = getattr(info, "card_data", None) or {}
        found = sorted(m.id for m in api.list_models(author="facebook", search="vggt", limit=50))
        return {"repo": repo_id, "licence": card.get("license") if hasattr(card, "get") else None,
                "gated": getattr(info, "gated", None), "facebook_vggt_models": found}
    except Exception as exc:  # noqa: BLE001 - informational only
        return {"repo": repo_id, "error": f"{type(exc).__name__}: {exc}"}


def load_model(args, device: str):
    import torch
    from vggt.models.vggt import VGGT

    t0 = time.perf_counter()
    if args.random_weights:
        model = VGGT()
        note("RANDOM WEIGHTS: plumbing test only, every quality number below is meaningless")
    else:
        model = VGGT.from_pretrained(args.weights)
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    log(f"model loaded in {time.perf_counter() - t0:.1f} s, {params / 1e9:.2f} B parameters")
    return model, torch


def run_chunk(model, torch, images, device: str, dtype):
    """One forward pass; returns (predictions on CPU, seconds, peak GB) or raises OOM."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
        pred = model(images.to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    keep = ("pose_enc", "depth", "depth_conf", "world_points_conf")
    return {k: pred[k].float().cpu() for k in keep if k in pred}, seconds, peak


def evaluate(pred, names, images_hw, orig_width, hints, gps, colmap):
    """Quality of one chunk against telemetry, GPS and Track A's cameras."""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images_hw)
    ext = extrinsic[0].numpy().astype(np.float64)   # [S, 3, 4] camera-from-world, OpenCV
    intr = intrinsic[0].numpy().astype(np.float64)  # [S, 3, 3] at the preprocessed size
    centres = np.stack([-e[:, :3].T @ e[:, 3] for e in ext])
    out: dict = {"frames": len(names)}

    # "crop" preprocessing scales the width to 518 exactly, so focal scales by width.
    focal_full = float(np.median(intr[:, 0, 0])) * orig_width / images_hw[1]
    out["focal_px_full_res"] = round(focal_full, 1)
    if "hfov_deg" in hints:
        tele = (orig_width / 2) / math.tan(math.radians(hints["hfov_deg"] / 2))
        out["focal_telemetry_px"] = round(tele, 1)
        out["focal_error_pct"] = round(100 * (focal_full - tele) / tele, 1)

    depth = pred["depth"][0, ..., 0].numpy()
    conf = pred["depth_conf"][0].numpy()
    out["depth_conf_median"] = round(float(np.median(conf)), 3)

    matched = [i for i, n in enumerate(names) if n in gps]
    if len(matched) >= 3:
        ref = np.array([gps[names[i]] for i in matched])
        scale, rot, trans = umeyama(centres[matched], ref)
        aligned = (scale * (rot @ centres[matched].T)).T + trans
        out["cam_vs_gps_rms_m"] = round(float(np.sqrt(((aligned - ref) ** 2).sum(1).mean())), 2)
        out["gps_path_m"] = round(float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()), 1)
        h, w = depth.shape[1:]
        centre = depth[:, int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]
        out["height_above_ground_m"] = round(float(np.median(centre)) * scale, 1)
        if "expected_agl_m" in hints:
            out["expected_agl_m"] = round(hints["expected_agl_m"], 1)
            out["height_error_pct"] = round(100 * (out["height_above_ground_m"] - hints["expected_agl_m"])
                                            / hints["expected_agl_m"], 1)

    if colmap is not None:
        try:
            import pycolmap

            rec = pycolmap.Reconstruction(str(colmap))
            by_name = {im.name: np.asarray(im.projection_center()) for im in rec.images.values()}
            common = [i for i, n in enumerate(names) if n in by_name and n in gps]
            if len(common) >= 3:
                col = np.array([by_name[names[i]] for i in common])
                ref = np.array([gps[names[i]] for i in common])
                s, r, t = umeyama(col, ref)
                col_metric = (s * (r @ col.T)).T + t
                s2, r2, t2 = umeyama(centres[common], col_metric)
                vg = (s2 * (r2 @ centres[common].T)).T + t2
                out["cam_vs_colmap_rms_m"] = round(float(np.sqrt(((vg - col_metric) ** 2).sum(1).mean())), 2)
                # Track A's own residual on the same frames: the GPS-noise floor (S4-1).
                out["colmap_vs_gps_rms_m"] = round(float(np.sqrt(((col_metric - ref) ** 2).sum(1).mean())), 2)
                out["colmap_common_frames"] = len(common)
        except Exception as exc:  # noqa: BLE001
            note(f"COLMAP comparison skipped: {type(exc).__name__}: {exc}")
    return out, ext, intr, depth, conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="output folder of `src.cli run`")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--colmap-sparse", type=Path, default=None, help="Track A sparse model to compare with")
    ap.add_argument("--weights", default="facebook/VGGT-1B")
    ap.add_argument("--random-weights", action="store_true", help="skip the download (plumbing test)")
    ap.add_argument("--vggt-repo", type=Path, default=ROOT / "tools" / "vggt")
    ap.add_argument("--chunks", default="8,16,32,all", help="chunk sizes to time; 'all' = every frame")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--cpu-frames", type=int, default=2, help="frames for the CPU smoke test")
    ap.add_argument("--windows", default="8,16,32",
                    help="sliding-window sizes to score along the flight (half-window stride); '' = off")
    args = ap.parse_args()

    sys.path.insert(0, str(args.vggt_repo.resolve()))
    try:
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError as exc:
        if getattr(exc, "name", "") and not str(exc.name).startswith("vggt"):
            log(f"missing dependency {exc.name!r}: pip install torchvision einops safetensors huggingface_hub")
        else:
            log(f"cannot import VGGT ({exc}); git clone https://github.com/facebookresearch/vggt {args.vggt_repo}")
        return 2

    images_dir = args.run_dir / "condition" / "images"
    names = sorted(p.name for p in images_dir.glob("*.jpg"))
    if not names:
        log(f"no conditioned images in {images_dir}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cpu_thread_budget())

    device = resolve_device(args.device)
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        env = {"device": device, "gpu": torch.cuda.get_device_name(0),
               "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
               "dtype": str(dtype).replace("torch.", "")}
    else:
        note("DOWNGRADE: no CUDA; running a CPU smoke test on "
             f"{args.cpu_frames} frames (VGGT-1B is not usable on CPU at scale)")
        env = {"device": device, "dtype": "float32"}
    env["torch"] = torch.__version__
    env["cpu_threads"] = cpu_thread_budget()
    log(f"environment {env}")
    lic = licence_info(args.weights)
    log(f"licence {lic}")

    wanted = [len(names) if c.strip() == "all" else int(c) for c in args.chunks.split(",")]
    wanted = sorted({min(n, len(names)) for n in wanted})
    if device != "cuda":
        wanted = [min(args.cpu_frames, len(names))]
    hints = telemetry_hints(args.run_dir)
    geo_path = args.run_dir / "condition" / "geo.txt"
    gps = read_geo(geo_path) if geo_path.exists() else {}

    t0 = time.perf_counter()
    images = load_and_preprocess_images([str(images_dir / n) for n in names[:max(wanted)]])
    load_s = time.perf_counter() - t0
    from PIL import Image

    orig_width = Image.open(images_dir / names[0]).size[0]
    images_hw = tuple(images.shape[-2:])
    log(f"{len(names)} frames available; preprocessed {images.shape[0]} to {images_hw[1]}x{images_hw[0]} "
        f"in {load_s:.1f} s; chunks {wanted}")

    model, torch = load_model(args, device)
    if device == "cuda":
        run_chunk(model, torch, images[:2], device, dtype)  # warm-up: kernels, allocator

    runs, best = [], None
    for n in wanted:
        try:
            pred, seconds, peak = run_chunk(model, torch, images[:n], device, dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            note(f"chunk {n}: out of GPU memory; larger chunks skipped")
            runs.append({"frames": n, "oom": True})
            break
        row = {"frames": n, "seconds": round(seconds, 2), "s_per_frame": round(seconds / n, 3),
               "peak_gpu_gb": round(peak, 2)}
        runs.append(row)
        log(f"chunk {row}")
        best = (n, pred)

    summary = {"environment": env, "licence": lic, "preprocess_s": round(load_s, 1),
               "input_hw": list(images_hw), "chunks": runs, "notes": NOTES}

    # Sliding windows: does VGGT hold up on short chunks (spec §7.2 chunking) even when a
    # single pass over the whole flight does not?
    sizes = [int(w) for w in args.windows.split(",") if w.strip()]
    windows = {}
    for w in sizes:
        if w > len(images):
            continue
        rows = []
        for start in range(0, len(images) - w + 1, max(w // 2, 1)):
            try:
                pred, _, _ = run_chunk(model, torch, images[start:start + w], device, dtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
            q, *_ = evaluate(pred, names[start:start + w], images_hw, orig_width, hints, gps, args.colmap_sparse)
            rows.append({"start": start, **{k: q.get(k) for k in (
                "gps_path_m", "cam_vs_gps_rms_m", "colmap_vs_gps_rms_m", "cam_vs_colmap_rms_m",
                "height_error_pct", "focal_error_pct")}})
        if rows:
            def med(key):
                vals = [r[key] for r in rows if r.get(key) is not None]
                return round(float(np.median(vals)), 2) if vals else None

            def worst(key):
                vals = [abs(r[key]) for r in rows if r.get(key) is not None]
                return round(float(max(vals)), 2) if vals else None

            windows[w] = {"count": len(rows),
                          "median": {k: med(k) for k in ("gps_path_m", "cam_vs_gps_rms_m", "colmap_vs_gps_rms_m",
                                                           "cam_vs_colmap_rms_m", "height_error_pct",
                                                           "focal_error_pct")},
                          "worst_abs": {k: worst(k) for k in ("cam_vs_colmap_rms_m", "height_error_pct")},
                          "rows": rows}
            log(f"windows of {w}: {windows[w]['median']}")
    summary["windows"] = {w: {k: v for k, v in d.items() if k != "rows"} for w, d in windows.items()}
    (args.out / "vggt_windows.json").write_text(json.dumps(windows, indent=2))
    if best is not None:
        n, pred = best
        quality, ext, intr, depth, conf = evaluate(pred, names[:n], images_hw, orig_width, hints, gps,
                                                   args.colmap_sparse)
        summary["quality_largest_chunk"] = quality
        np.savez_compressed(args.out / "vggt_chunk.npz", names=np.array(names[:n]), extrinsic=ext,
                            intrinsic=intr, depth=depth.astype(np.float16), depth_conf=conf.astype(np.float16))
        log(f"quality {quality}")
    (args.out / "vggt_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== PASTE EVERYTHING BELOW BACK TO CLAUDE =====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
