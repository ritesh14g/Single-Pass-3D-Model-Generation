"""Compression artifact handling (spec §5.3, challenge ii).

Drone video arrives heavily H.264/H.265 compressed. The damage that matters for
photogrammetry is not aesthetic: blocking creates a *regular grid of corners*
that is stationary in image space rather than in the world. Feature detectors
love corners, so they latch onto the grid, and the resulting matches are
consistent between frames while being geometrically meaningless — the worst
possible failure mode, because it looks like good data.

Three defences, in the order the spec prescribes:

  * **Detect** blockiness by comparing gradient energy across block boundaries
    with gradient energy inside blocks. Real scene structure does not care
    where the 8x8 grid falls; compression artifacts sit exactly on it.
  * **Correct** with edge-preserving filtering only. Bilateral/guided filtering
    suppresses the block edge while keeping the real corner. Gaussian blur
    would remove both, which destroys precisely what SfM depends on.
  * **Suppress false features** by rejecting keypoints that land on the block
    grid when the blockiness score is high — a targeted veto rather than a
    blanket one, so a genuine corner that happens to fall near a boundary is
    only lost on frames where the grid is actually a problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.logging import get_logger, log_event

log = get_logger(__name__)


@dataclass
class ArtifactAssessment:
    """Blockiness measurement for one frame."""

    blockiness: float                    # strongest score across tested block sizes
    per_block_size: dict[int, float]
    dominant_block_size: int
    needs_correction: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockiness": round(self.blockiness, 3),
            "per_block_size": {str(k): round(v, 3) for k, v in self.per_block_size.items()},
            "dominant_block_size": self.dominant_block_size,
            "needs_correction": self.needs_correction,
            "threshold": self.threshold,
        }


def blockiness_score(gray: np.ndarray, block_size: int = 8) -> float:
    """Ratio of gradient energy on the block grid to gradient energy off it.

    1.0 means the grid is indistinguishable from the rest of the image (no
    detectable blocking). Values above ~1.3 indicate visible blocking.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    image = gray.astype(np.float32)
    h, w = image.shape[:2]
    if h < 3 * block_size or w < 3 * block_size:
        return 1.0

    # Mean absolute difference across each vertical / horizontal seam.
    column_diff = np.abs(np.diff(image, axis=1)).mean(axis=0)   # length w-1
    row_diff = np.abs(np.diff(image, axis=0)).mean(axis=1)      # length h-1

    def ratio(profile: np.ndarray) -> float:
        # Seam i lies between pixel i and i+1, so a block boundary after every
        # `block_size` pixels sits at indices block_size-1, 2*block_size-1, ...
        positions = np.arange(profile.size)
        on_grid = ((positions + 1) % block_size) == 0
        # Ignore the outermost seams; border effects are not compression.
        interior = np.zeros_like(on_grid)
        interior[block_size:-block_size] = True
        grid_values = profile[on_grid & interior]
        other_values = profile[(~on_grid) & interior]
        if grid_values.size == 0 or other_values.size == 0:
            return 1.0
        off_grid_mean = float(other_values.mean())
        if off_grid_mean <= 1e-6:
            return 1.0
        return float(grid_values.mean() / off_grid_mean)

    return float(np.mean([ratio(column_diff), ratio(row_diff)]))


def assess_artifacts(image: np.ndarray, cfg: Any) -> ArtifactAssessment:
    """Score a frame for blocking at each configured block size."""
    artifact_cfg = cfg.get_path("condition.artifacts")
    threshold = float(artifact_cfg["blockiness_threshold"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    scores = {int(size): blockiness_score(gray, int(size)) for size in artifact_cfg["block_sizes"]}
    dominant = max(scores, key=lambda k: scores[k])
    worst = scores[dominant]
    return ArtifactAssessment(
        blockiness=worst,
        per_block_size=scores,
        dominant_block_size=dominant,
        needs_correction=worst >= threshold,
        threshold=threshold,
    )


def suppress_block_artifacts(image: np.ndarray, cfg: Any, assessment: ArtifactAssessment | None = None) -> np.ndarray:
    """Edge-preserving suppression of block edges.

    Returns the input unchanged when blockiness is below threshold: filtering a
    clean frame costs texture detail for no benefit.
    """
    artifact_cfg = cfg.get_path("condition.artifacts")
    if not bool(artifact_cfg["enabled"]):
        return image
    if assessment is None:
        assessment = assess_artifacts(image, cfg)
    if not assessment.needs_correction:
        return image

    bilateral = artifact_cfg["bilateral"]
    # Bilateral, never Gaussian: the block edge is a low-amplitude step that
    # the range kernel treats as noise, while a real geometric edge exceeds the
    # range sigma and survives.
    return cv2.bilateralFilter(
        image,
        d=int(bilateral["diameter"]),
        sigmaColor=float(bilateral["sigma_color"]),
        sigmaSpace=float(bilateral["sigma_space"]),
    )


def block_grid_mask(shape: tuple[int, int], block_size: int, exclusion_px: int) -> np.ndarray:
    """Boolean mask of pixels within ``exclusion_px`` of the block grid."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for boundary in range(block_size, w, block_size):
        lo, hi = max(boundary - exclusion_px, 0), min(boundary + exclusion_px + 1, w)
        mask[:, lo:hi] = True
    for boundary in range(block_size, h, block_size):
        lo, hi = max(boundary - exclusion_px, 0), min(boundary + exclusion_px + 1, h)
        mask[lo:hi, :] = True
    return mask


def filter_grid_keypoints(
    keypoints: Sequence[Any],
    shape: tuple[int, int],
    assessment: ArtifactAssessment,
    cfg: Any,
) -> tuple[list[Any], int]:
    """Drop keypoints sitting on the compression grid.

    Only applied when blockiness actually exceeds threshold — on a clean frame
    the grid carries no artifacts and vetoing those positions would throw away
    good features for nothing.

    Returns ``(kept_keypoints, dropped_count)``.
    """
    artifact_cfg = cfg.get_path("condition.artifacts")
    if not assessment.needs_correction or not bool(artifact_cfg["enabled"]):
        return list(keypoints), 0

    exclusion = int(artifact_cfg["keypoint_exclusion_px"])
    mask = block_grid_mask(shape, assessment.dominant_block_size, exclusion)
    h, w = shape[:2]
    kept: list[Any] = []
    dropped = 0
    for keypoint in keypoints:
        x, y = _keypoint_xy(keypoint)
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w and mask[yi, xi]:
            dropped += 1
            continue
        kept.append(keypoint)
    if dropped:
        log_event(
            log,
            logging.DEBUG,
            f"dropped {dropped} keypoints on the compression grid",
            blockiness=round(assessment.blockiness, 3),
            block_size=assessment.dominant_block_size,
            kept=len(kept),
        )
    return kept, dropped


def _keypoint_xy(keypoint: Any) -> tuple[float, float]:
    """Accept both ``cv2.KeyPoint`` and plain (x, y) pairs."""
    point = getattr(keypoint, "pt", None)
    if point is not None:
        return float(point[0]), float(point[1])
    return float(keypoint[0]), float(keypoint[1])
