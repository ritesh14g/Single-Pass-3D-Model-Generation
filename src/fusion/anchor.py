"""Zone 2 handling (spec §6.3): monocular depth, anchored to Zone 1, filling — never correcting.

For each frame chosen to look at thinly observed or unobserved ground:

  1. R = pixels with no Zone 1 surface behind them; B = the Zone 1 pixels within
     ``boundary_band_px`` of R (the boundary band). A second ring beyond B is held out.
  2. Monocular depth for the frame: Track B's saved VGGT maps when the run has them,
     otherwise the Track B predictor run on this one frame.
  3. d_mvs ~ s * d_mono + t over B only, RANSAC (2-point samples), least squares on inliers.
  4. Inlier share below ``min_inlier_ratio`` or residual above ``max_residual_m``: the frame is
     refused and its region stays Zone 3 — a bad anchor is worse than a gap. Track B's saved
     depth is already anchored to Track A, so its fit must also agree with the georeferencing
     scale within ``cache_scale_tolerance``.
  5. The residual at the band is feathered into R over ``blend_band_px`` (C0 at the seam).
  6. R's pixels are back-projected; points whose ground cell is not a gap or thin cell, or whose
     height leaves the measured surface's range, are dropped. So are pixels whose anchored depth
     leaves the band's depth span by more than ``max_extrapolation_rel`` x the band depth: a fit
     is trusted only near the depths it was fitted on.

Fusion is a confidence-weighted voxel average on the Zone 1 voxel grid: a pixel's weight is
``zone2_weight_max`` x its confidence rank x the frame's inlier share, so always below Zone 1's
weight of 1. A voxel needs ``min_weight_to_surface`` in total. **A voxel that holds Zone 1
surface never receives monocular points** (the §6.3 critical rule) — the evaluator re-checks it.

The held-out ring gives an honest Zone 2 accuracy estimate: the anchored depth's error on
Zone 1 pixels the fit never saw, at the distance from the seam the fill extrapolates over.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.core.logging import get_logger
from src.fusion.scene import CameraView, Scene
from src.fusion.zones import ZONE1, ZONE2, ZONE3, GroundMap, ZoneResult

log = get_logger(__name__)
EMPTY = 1e30  # z-buffer values at or above this are "no surface"
# (image path) -> (depth [h,w], confidence [h,w]); depth along the optical axis, any scale.
MonoDepth = Callable[[CameraView], tuple[np.ndarray, np.ndarray]]


class MonoUnavailable(RuntimeError):
    """The depth source itself cannot run (no model, no weights): stop, do not retry per frame."""


@dataclass
class AnchorFit:
    scale: float
    shift: float
    residual_m: float
    inlier_ratio: float
    band_px: int


@dataclass
class FillResult:
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    weight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    frames_per_point: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    frames: list[dict[str, Any]] = field(default_factory=list)
    blocked_by_zone1: int = 0
    below_weight: int = 0
    source: str = "none"
    reason: str | None = None
    seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        used = [f for f in self.frames if f["status"] == "anchored"]
        refused = [f for f in self.frames if f["status"] == "refused"]
        held = [f["holdout_error_m"] for f in used if f.get("holdout_error_m") is not None]
        held_pct = [f["holdout_error_pct"] for f in used if f.get("holdout_error_pct") is not None]
        return {
            "source": self.source, "reason": self.reason, "frames_tried": len(self.frames),
            "frames_anchored": len(used), "frames_refused": len(refused),
            "refusal_reasons": sorted({f["reason"] for f in refused}),
            "fill_points": int(len(self.points)), "blocked_by_zone1": int(self.blocked_by_zone1),
            "below_weight": int(self.below_weight),
            "anchor_residual_median_m": round(float(np.median([f["residual_m"] for f in used])), 3) if used else None,
            "holdout_error_median_m": round(float(np.median(held)), 3) if held else None,
            "holdout_error_median_pct": round(float(np.median(held_pct)), 3) if held_pct else None,
            "weight_max": round(float(self.weight.max()), 3) if len(self.weight) else None,
            "seconds": round(self.seconds, 1),
        }


# -- the fit -----------------------------------------------------------------------------
def fit_scale_shift(mono: np.ndarray, mvs: np.ndarray, *, iterations: int, threshold,
                    rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    """RANSAC over 2-point samples, then least squares on the inliers. Scale must be positive.

    ``threshold`` is a scalar or one value per sample (depth-relative inlier bands)."""
    n = len(mono)
    best = np.zeros(n, bool)
    if n < 2:
        return 1.0, 0.0, best
    for _ in range(iterations):
        i, j = rng.choice(n, 2, replace=False)
        if abs(mono[i] - mono[j]) < 1e-9:
            continue
        s = (mvs[i] - mvs[j]) / (mono[i] - mono[j])
        if s <= 0:
            continue
        t = mvs[i] - s * mono[i]
        inl = np.abs(s * mono + t - mvs) < threshold
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 2:
        return 1.0, 0.0, best
    a = np.c_[mono[best], np.ones(int(best.sum()))]
    (s, t), *_ = np.linalg.lstsq(a, mvs[best], rcond=None)
    if s <= 0:
        return 1.0, 0.0, np.zeros(n, bool)
    return float(s), float(t), np.abs(s * mono + t - mvs) < threshold


def render_depth(cam: CameraView, pts: np.ndarray, shape: tuple[int, int], voxel: float) -> np.ndarray:
    """Nearest depth per pixel of a (h, w) buffer; holes between samples closed by one voxel."""
    import cv2

    h, w = shape
    uv, z = cam.project(pts)
    ok = cam.inside(uv, z)
    sx, sy = w / cam.width, h / cam.height
    col = np.clip((uv[ok, 0] * sx).astype(np.int64), 0, w - 1)
    row = np.clip((uv[ok, 1] * sy).astype(np.int64), 0, h - 1)
    buf = np.full(h * w, np.inf, np.float32)
    np.minimum.at(buf, row * w + col, z[ok].astype(np.float32))
    if ok.any():
        radius = int(np.clip(np.round(np.median(voxel * cam.k[0, 0] * sx / z[ok])), 1, 9))
        buf = cv2.erode(buf.reshape(h, w), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)).ravel()
        buf[buf >= EMPTY] = np.inf  # cv2.erode turns inf into FLT_MAX, which isfinite() accepts
    return buf.reshape(h, w)


def anchor_frame(cam: CameraView, mono: np.ndarray, conf: np.ndarray, zone1_pts: np.ndarray, voxel: float,
                 acfg: Any, rng: np.random.Generator, expected_scale: float | None = None):
    """(filled depth [h,w] with NaN outside R, weight [h,w], AnchorFit | None, report dict).

    ``expected_scale``: the scale the fit should find when the depth is already anchored (Track B's
    saved maps are in Track A's model units, so metric = georef scale x depth); None = any scale."""
    from scipy import ndimage

    h, w = mono.shape
    d1 = render_depth(cam, zone1_pts, (h, w), voxel)
    z1 = np.isfinite(d1)
    valid = np.isfinite(mono) & (mono > 0) & (conf >= np.quantile(conf, float(acfg.drop_low_conf)))
    region = ~z1 & valid
    report: dict[str, Any] = {"frame": cam.name, "region_px": int(region.sum()), "zone1_px": int(z1.sum())}
    if region.sum() < int(acfg.min_region_px):
        return None, None, None, {**report, "status": "skipped", "reason": "no thin or unobserved pixels"}
    dist = ndimage.distance_transform_edt(~region)
    band_px = int(acfg.boundary_band_px)
    band = z1 & valid & (dist <= band_px)
    ring = z1 & valid & (dist > band_px) & (dist <= 2 * band_px)
    if band.sum() < int(acfg.min_band_px):
        return None, None, None, {**report, "status": "refused", "reason": "no Zone 1 boundary to anchor to",
                                  "band_px": int(band.sum())}
    # Monocular depth error grows with range (VGGT: ~0.9% of depth on Esri), so every tolerance is
    # the larger of an absolute floor in metres and a fraction of the Zone 1 depth.
    target = d1[band].astype(np.float64)
    s, t, inl = fit_scale_shift(mono[band].astype(np.float64), target, iterations=int(acfg.ransac_iterations),
                                threshold=np.maximum(float(acfg.ransac_inlier_threshold_m),
                                                     float(acfg.ransac_inlier_threshold_rel) * target), rng=rng)
    ratio = float(inl.mean()) if len(inl) else 0.0
    resid = s * mono[band].astype(np.float64) + t - target
    rms = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.any() else float("inf")
    depth = float(np.median(target))
    limit = max(float(acfg.max_residual_m), float(acfg.max_residual_rel) * depth)
    fit = AnchorFit(s, t, rms, ratio, int(band.sum()))
    report.update(scale=round(s, 5), shift=round(t, 4), residual_m=round(rms, 4), residual_limit_m=round(limit, 3),
                  band_depth_m=round(depth, 2), inlier_ratio=round(ratio, 3), band_px=int(band.sum()))
    if ring.any():
        err = np.abs(s * mono[ring].astype(np.float64) + t - d1[ring])
        report["holdout_error_m"] = round(float(np.median(err)), 4)
        report["holdout_error_pct"] = round(100.0 * float(np.median(err / d1[ring])), 3)
        report["holdout_px"] = int(ring.sum())
    if ratio < float(acfg.min_inlier_ratio):
        return None, None, fit, {**report, "status": "refused", "reason": "inlier share below min_inlier_ratio"}
    if rms > limit:
        return None, None, fit, {**report, "status": "refused", "reason": "anchor residual above the limit"}
    # A narrow band can fit any scale: Esri frame 216 (2,139 band px) found s = 267 on depth whose
    # true scale was 1.0 and extrapolated it over 183k px, a third of the fill (box, 2026-09-23).
    tol = float(acfg.cache_scale_tolerance)
    if expected_scale and expected_scale > 0 and tol > 1 and abs(np.log(s / expected_scale)) > np.log(tol):
        return None, None, fit, {**report, "status": "refused", "expected_scale": round(float(expected_scale), 5),
                                 "reason": "scale disagrees with the georeferencing (depth already anchored)"}

    # C0 at the seam: each R pixel takes the residual of its nearest band inlier, fading to zero
    # over the blend band, so the fill meets Zone 1 instead of stepping off it.
    seam = np.zeros((h, w), bool)
    band_rows, band_cols = np.nonzero(band)
    seam[band_rows[inl], band_cols[inl]] = True
    correction = np.zeros((h, w), np.float32)
    correction[seam] = -(s * mono[seam] + t - d1[seam])
    dseam, (ri, ci) = ndimage.distance_transform_edt(~seam, return_indices=True)
    fade = np.clip(1.0 - dseam / max(float(acfg.blend_band_px), 1.0), 0.0, 1.0)
    filled = np.where(region, s * mono + t + correction[ri, ci] * fade, np.nan).astype(np.float32)
    # Trust the fit only near the depths it was fitted on.
    lo, hi = np.percentile(target[inl], [2, 98])
    reach = float(acfg.max_extrapolation_rel) * depth
    far = region & ((filled < lo - reach) | (filled > hi + reach))
    filled[far] = np.nan
    region = region & ~far
    report["extrapolation_dropped_px"] = int(far.sum())
    # Confidence rank within the frame, so a model's absolute confidence scale does not matter.
    rank = np.zeros((h, w), np.float32)
    if region.any():
        order = np.argsort(np.argsort(conf[region]))
        rank[region] = (order + 1) / len(order)
    weight = np.where(region, float(acfg.zone2_weight_max) * rank * ratio, 0.0).astype(np.float32)
    return filled, weight, fit, {**report, "status": "anchored"}


# -- frame choice ---------------------------------------------------------------------------
def choose_frames(scene: Scene, gm: GroundMap, max_frames: int, min_cells: int) -> list[int]:
    """Greedy set cover of the gap and thin ground cells by camera views."""
    rows, cols = np.nonzero((gm.zone == ZONE3) | (gm.zone == ZONE2))
    if not len(rows) or max_frames <= 0:
        return []
    xy = gm.centres(rows, cols)
    pts = np.c_[xy, np.full(len(xy), gm.ground_z)]
    sees = []
    for cam in scene.cameras:
        uv, z = cam.project(pts)
        sees.append(np.flatnonzero(cam.inside(uv, z)))
    covered = np.zeros(len(pts), bool)
    chosen: list[int] = []
    while len(chosen) < max_frames:
        gains = [0 if i in chosen else int((~covered[s]).sum()) for i, s in enumerate(sees)]
        best = int(np.argmax(gains)) if gains else 0
        if not gains or gains[best] < min_cells:
            break
        chosen.append(best)
        covered[sees[best]] = True
    return chosen


# -- monocular depth sources ---------------------------------------------------------------
def cached_depth(depth_dir: Path | None) -> MonoDepth | None:
    """Track B's anchored VGGT depth (``track_b_depth/<name>.npz``), when the run kept it."""
    if depth_dir is None or not Path(depth_dir).is_dir() or not any(Path(depth_dir).glob("*.npz")):
        return None

    def get(cam: CameraView):
        path = Path(depth_dir) / f"{cam.name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"no saved depth for {cam.name}")
        data = np.load(path)
        return data["depth"].astype(np.float32), data["conf"].astype(np.float32)

    return get


def predictor_depth(cfg: Any, device: str) -> MonoDepth:
    """Track B's VGGT predictor, one frame at a time (monocular). Raises TrackBUnavailable."""
    from src.recon.track_b_vggt import make_predictor

    predict = make_predictor(cfg.get_path("recon.track_b"), device)

    def get(cam: CameraView):
        if cam.image_path is None:
            raise FileNotFoundError(f"no image for {cam.name}")
        depth, conf, _ = predict([cam.image_path])
        return depth[0].astype(np.float32), conf[0].astype(np.float32)

    return get


# -- the whole fill ---------------------------------------------------------------------------
def fill(scene: Scene, zones: ZoneResult, gm: GroundMap, cfg: Any, mono: MonoDepth | None, *,
         max_frames: int, source: str, budget_stage=None) -> FillResult:
    import cv2

    acfg = cfg.get_path("fusion.mono_depth")
    started = time.perf_counter()
    result = FillResult(source=source)
    if mono is None:
        result.reason = "no monocular depth source"
        return result
    frames = choose_frames(scene, gm, max_frames, int(acfg.min_target_cells))
    if not frames:
        result.reason = "no gap or thin ground cell is in view"
        return result
    rng = np.random.default_rng(int(cfg.get_path("run.seed", 0)))
    # Track B's saved maps were anchored in Track A's model units; the scene is metric.
    expected_scale = float(getattr(scene.georef, "scale", 1.0) or 1.0) if source == "track_b_cache" else None
    zone1_pts = scene.points[zones.point_zone == ZONE1]
    z_lo, z_hi = np.percentile(scene.points[:, 2], [1, 99])
    margin = float(acfg.height_margin_m)
    stride = max(int(acfg.pixel_stride), 1)
    pts_all, rgb_all, w_all, frame_all = [], [], [], []
    for n, idx in enumerate(frames):
        cam = scene.cameras[idx]
        try:
            depth, conf = mono(cam)
        except MonoUnavailable as exc:
            result.reason = f"monocular depth unavailable: {str(exc).splitlines()[0][:240]}"
            break
        except Exception as exc:  # noqa: BLE001 - one frame's depth never stops the stage
            result.frames.append({"frame": cam.name, "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        filled, weight, _, report = anchor_frame(cam, depth, conf, zone1_pts, zones.spacing or zones.grid.size,
                                                 acfg, rng, expected_scale=expected_scale)
        result.frames.append(report)
        if filled is not None:
            h, w = filled.shape
            rows, cols = np.mgrid[0:h:stride, 0:w:stride]
            rows, cols = rows.ravel(), cols.ravel()
            keep = np.isfinite(filled[rows, cols])
            rows, cols = rows[keep], cols[keep]
            uv = np.c_[(cols + 0.5) * cam.width / w, (rows + 0.5) * cam.height / h]
            pts = cam.centre + filled[rows, cols, None] * cam.rays(uv)
            grow, gcol = gm.cells(pts[:, :2])
            inside = gm.inside(grow, gcol)
            target = np.zeros(len(pts), bool)
            target[inside] = np.isin(gm.zone[grow[inside], gcol[inside]], (ZONE2, ZONE3))
            target &= (pts[:, 2] >= z_lo - margin) & (pts[:, 2] <= z_hi + margin)
            report["points_in_targets"] = int(target.sum())
            if target.any():
                colour = np.full((int(target.sum()), 3), 150, np.uint8)
                if cam.image_path is not None:
                    img = cv2.imread(str(cam.image_path), cv2.IMREAD_COLOR)
                    if img is not None:
                        px = np.clip((uv[target, 0] * img.shape[1] / cam.width).astype(int), 0, img.shape[1] - 1)
                        py = np.clip((uv[target, 1] * img.shape[0] / cam.height).astype(int), 0, img.shape[0] - 1)
                        colour = img[py, px, ::-1]
                pts_all.append(pts[target])
                rgb_all.append(colour)
                w_all.append(weight[rows[target], cols[target]])
                frame_all.append(np.full(int(target.sum()), n, np.int32))
        # Project the *fill's own* per-frame rate onto the frames left. The stage-level projection
        # (elapsed / progress) counted zone classification as fill time: on Esri it read 49 s at 2/35
        # frames as an 860 s stage and stopped the fill, which then took 2.2 s for those 2 frames.
        left = len(frames) - n - 1
        if budget_stage is not None and getattr(budget_stage, "enabled", True) and left:
            per_frame = (time.perf_counter() - started) / (n + 1)
            if per_frame * left > budget_stage.remaining_s - float(acfg.budget_reserve_s):
                budget_stage.degrade("reduce_frames", reason="time budget", component="Zone 2 monocular fill",
                                     frames_done=n + 1, frames_planned=len(frames),
                                     projected_s=round(per_frame * left, 1),
                                     remaining_s=round(budget_stage.remaining_s, 1))
                result.reason = f"time budget: stopped after {n + 1} of {len(frames)} frames"
                break
    if pts_all:
        _fuse(result, zones, np.concatenate(pts_all), np.concatenate(rgb_all), np.concatenate(w_all),
              np.concatenate(frame_all), float(acfg.min_weight_to_surface), float(acfg.zone2_weight_max))
    result.seconds = time.perf_counter() - started
    return result


def _fuse(result: FillResult, zones: ZoneResult, pts, rgb, weight, frame, min_weight: float, cap: float) -> None:
    keys = zones.grid.keys(pts)
    ok = keys >= 0
    # The critical rule: a voxel holding Zone 1 surface is measured; monocular depth never enters it.
    zone1_keys = zones.keys[zones.zone == ZONE1]
    blocked = np.isin(keys, zone1_keys)
    result.blocked_by_zone1 = int((blocked & ok).sum())
    ok &= ~blocked & (weight > 0)
    keys, pts, rgb, weight, frame = keys[ok], pts[ok], rgb[ok], weight[ok], frame[ok]
    if not len(keys):
        return
    uniq, inv = np.unique(keys, return_inverse=True)
    wsum = np.bincount(inv, weight)
    xyz = np.stack([np.bincount(inv, weight * pts[:, a]) for a in range(3)], 1) / wsum[:, None]
    col = np.stack([np.bincount(inv, weight * rgb[:, a]) for a in range(3)], 1) / wsum[:, None]
    frames = np.bincount(np.unique(inv.astype(np.int64) * 1_000_003 + frame) // 1_000_003, minlength=len(uniq))
    accept = wsum >= min_weight
    result.below_weight = int((~accept).sum())
    result.points = xyz[accept]
    result.rgb = np.clip(col[accept], 0, 255).astype(np.uint8)
    # Stored confidence: accumulated weight, capped strictly below Zone 1's weight of 1.
    result.weight = np.minimum(wsum[accept], cap)
    result.frames_per_point = frames[accept].astype(np.int32)
