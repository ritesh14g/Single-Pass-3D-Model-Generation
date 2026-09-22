"""Image measurements for the input check: quality per frame, motion and rigidity per pair.

Per frame: sharpness (variance of the Laplacian), exposure, texture (feature count), sky.
Per pair of consecutive samples: SIFT matches, the share consistent with one rigid 3-D
scene seen from two positions (fundamental-matrix inliers), and the ground motion as a
homography (translation, rotation, scale). The pair motion is what gets compared with the
GPS; the rigid share is the physical-consistency check — a real, mostly static scene filmed
from a moving camera obeys epipolar geometry; morphing footage does not.
Whole clip: static edges that do not move with the scene (burned-in OSD text, logos,
watermarks) and black letterbox borders.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameStats:
    t: float
    sharpness: float
    brightness: float
    dark_frac: float
    bright_frac: float
    keypoints: int
    sky_frac: float


@dataclass
class PairMotion:
    t0: float
    t1: float
    matches: int
    rigid_inliers: int
    rigid_ratio: float | None
    shift_px: float | None          # image-centre displacement, px at the sampled width
    rotation_deg: float | None
    scale: float | None


def frame_stats(t: float, bgr: np.ndarray, keypoints: int) -> FrameStats:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return FrameStats(t=t, sharpness=sharp, brightness=float(gray.mean()),
                      dark_frac=float((gray < 16).mean()), bright_frac=float((gray > 245).mean()),
                      keypoints=keypoints, sky_frac=sky_fraction(bgr))


def sky_fraction(bgr: np.ndarray) -> float:
    """Share of the frame that is smooth, bright, blue-or-grey and connected to the top edge."""
    small = cv2.resize(bgr, (160, int(160 * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    b, _, r = cv2.split(small.astype(np.int16))
    candidate = ((grad < 12) & (hsv[..., 2] > 120) & (b >= r - 5)).astype(np.uint8)
    n, labels = cv2.connectedComponents(candidate)
    top = set(np.unique(labels[0][candidate[0] > 0]).tolist()) - {0}
    if not top:
        return 0.0
    return float(np.isin(labels, list(top)).mean())


class PairAnalyzer:
    """SIFT on a downscaled copy; one detector per thread (OpenCV releases the GIL)."""

    def __init__(self, max_features: int = 1500, ratio: float = 0.8, rigid_px: float = 1.5, width: int = 640):
        self.max_features, self.width = max_features, width
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio, self.rigid_px = ratio, rigid_px
        import threading

        self._local = threading.local()

    def features(self, bgr: np.ndarray):
        if not hasattr(self._local, "sift"):
            self._local.sift = cv2.SIFT_create(nfeatures=self.max_features)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self.width:
            gray = cv2.resize(gray, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        return self._local.sift.detectAndCompute(gray, None)

    def feature_size(self, bgr: np.ndarray) -> tuple[int, int]:
        h, w = bgr.shape[:2]
        return (self.width, int(round(h * self.width / w))) if w > self.width else (w, h)

    def pair(self, t0: float, f0, t1: float, f1, size: tuple[int, int]) -> PairMotion:
        (k0, d0), (k1, d1) = f0, f1
        empty = PairMotion(t0, t1, 0, 0, None, None, None, None)
        if d0 is None or d1 is None or len(k0) < 8 or len(k1) < 8:
            return empty
        good = [m for m, n in (p for p in self.matcher.knnMatch(d0, d1, k=2) if len(p) == 2)
                if m.distance < self.ratio * n.distance]
        if len(good) < 12:
            return PairMotion(t0, t1, len(good), 0, None, None, None, None)
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        # Two models of "one rigid scene": the fundamental matrix in general, a homography when
        # the ground is flat or the camera barely moves — where the fundamental matrix is
        # degenerate (and its solver can even fail outright). The better fit is the verdict.
        rigid = 0
        try:
            _, fmask = cv2.findFundamentalMat(p0, p1, getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC), self.rigid_px, 0.999)
            rigid = int(fmask.sum()) if fmask is not None else 0
        except cv2.error:
            rigid = 0
        H, hmask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if hmask is not None:
            rigid = max(rigid, int(hmask.sum()))
        shift = rot = scale = None
        if H is not None:
            w, h = size
            c = np.array([w / 2, h / 2, 1.0])
            m = H @ c
            if abs(m[2]) > 1e-9:
                shift = float(np.hypot(m[0] / m[2] - c[0], m[1] / m[2] - c[1]))
            A = H[:2, :2] / (H[2, 2] if abs(H[2, 2]) > 1e-9 else 1.0)
            rot = float(np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1])))
            scale = float(np.sqrt(abs(np.linalg.det(A))))
        return PairMotion(t0, t1, len(good), rigid, rigid / len(good), shift, rot, scale)


def static_overlay(frames: list[np.ndarray], persist: float = 0.9) -> tuple[float, tuple[int, int, int, int] | None]:
    """(share of the frame, bounding box) covered by edges that stay put while the scene moves."""
    if len(frames) < 8:
        return 0.0, None
    edges = []
    for bgr in frames:
        g = cv2.cvtColor(cv2.resize(bgr, (480, int(480 * bgr.shape[0] / bgr.shape[1]))), cv2.COLOR_BGR2GRAY)
        edges.append(cv2.Canny(g, 80, 200) > 0)
    stack = np.stack(edges).astype(np.float32)
    persistent = (stack.mean(0) >= persist).astype(np.uint8)
    # A static scene makes every edge persistent; that is "no motion", not an overlay.
    if stack.mean(0).mean() > 0 and persistent.sum() > 0.5 * (stack.mean(0) > 0.3).sum():
        return 0.0, None
    persistent = cv2.dilate(persistent, np.ones((5, 5), np.uint8))
    frac = float(persistent.mean())
    ys, xs = np.nonzero(persistent)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
    return frac, box


def letterbox(frames: list[np.ndarray], level: int = 10) -> dict[str, float]:
    """Width of black borders on each side, as a share of the frame."""
    if not frames:
        return {}
    mean = np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames], axis=0)
    rows, cols = mean.mean(1), mean.mean(0)

    def run(values: np.ndarray) -> int:
        n = 0
        while n < len(values) // 3 and values[n] < level:
            n += 1
        return n

    h, w = mean.shape
    return {"top": run(rows) / h, "bottom": run(rows[::-1]) / h, "left": run(cols) / w, "right": run(cols[::-1]) / w}
