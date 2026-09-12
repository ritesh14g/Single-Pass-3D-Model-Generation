"""Dynamic object handling (spec §5.5, challenge iv).

Cars, people and animals violate the static-scene assumption every
photogrammetric method is built on. Left alone they smear into the mesh as
elongated ghosts and, worse, corrupt the poses that produced them.

The spec asks us to *handle* dynamic objects, not reconstruct them — so the
correct output is a clean static scene with the movers removed, and honest gaps
where they were.

Two detectors, deliberately different in kind:

  * **Semantic masking** (primary) — a lightweight segmentation model over the
    known mover classes. Fast, and catches a parked car that never moves, which
    no geometric test can find.
  * **Geometric consistency** (secondary) — reproject a frame's depth into its
    neighbours and flag pixels whose depth disagrees across views *despite*
    living in a textured region. This catches what the semantic model has no
    class for: debris, floodwater, moving vegetation, the disaster-response
    cases that matter most here.

The texture qualifier in the geometric test is what keeps it from firing on
every untextured roof, where depth disagreement means "MVS had nothing to match
on", not "this surface moved".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.logging import get_logger, log_downgrade, log_event

log = get_logger(__name__)


@dataclass
class DynamicMaskResult:
    """Mask of pixels excluded from feature detection and depth fusion."""

    mask: np.ndarray = field(repr=False)
    fraction: float = 0.0
    classes: dict[str, int] = field(default_factory=dict)
    method: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "masked_fraction": round(self.fraction, 4),
            "classes": self.classes,
            "method": self.method,
        }

    def combined_with(self, other: "DynamicMaskResult") -> "DynamicMaskResult":
        mask = self.mask | other.mask
        classes = dict(self.classes)
        for name, count in other.classes.items():
            classes[name] = classes.get(name, 0) + count
        methods = "+".join(sorted({self.method, other.method} - {"none"})) or "none"
        return DynamicMaskResult(mask=mask, fraction=float(mask.mean()), classes=classes, method=methods)


class DynamicMasker:
    """Semantic mover segmentation with a defined no-model fallback.

    The model is loaded lazily and once. If ultralytics or the weights are not
    available the masker reports ``available=False`` and returns empty masks —
    the pipeline then relies on the geometric test alone, with the downgrade
    recorded loudly rather than silently changing the result.
    """

    def __init__(self, cfg: Any):
        dyn_cfg = cfg.get_path("condition.dynamic")
        self.enabled = bool(dyn_cfg["enabled"])
        self.model_name = str(dyn_cfg["model"])
        self.wanted_classes = {str(c).lower() for c in dyn_cfg["classes"]}
        self.conf_threshold = float(dyn_cfg["conf_threshold"])
        self.dilate_px = int(dyn_cfg["dilate_px"])
        self._model: Any = None
        self._load_attempted = False
        self.available = False
        self.unavailable_reason = "" if self.enabled else "disabled in config"

    def _ensure_model(self) -> bool:
        if self._load_attempted:
            return self.available
        self._load_attempted = True
        if not self.enabled:
            return False
        try:
            from ultralytics import YOLO
        except ImportError:
            self.unavailable_reason = "ultralytics is not installed"
            log_downgrade(log, "semantic dynamic masking", "geometric consistency only",
                          self.unavailable_reason)
            return False
        try:
            self._model = YOLO(self.model_name)
            self.available = True
            log_event(log, logging.INFO, f"loaded dynamic-object model {self.model_name}",
                      classes=sorted(self.wanted_classes))
        except Exception as exc:  # noqa: BLE001 - weights missing or download blocked
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            log_downgrade(log, "semantic dynamic masking", "geometric consistency only",
                          self.unavailable_reason)
        return self.available

    def mask(self, image: np.ndarray) -> DynamicMaskResult:
        """Segment movers in one frame."""
        empty = np.zeros(image.shape[:2], dtype=bool)
        if not self._ensure_model():
            return DynamicMaskResult(mask=empty, method="none")

        try:
            predictions = self._model.predict(
                image, conf=self.conf_threshold, verbose=False, retina_masks=True
            )
        except Exception as exc:  # noqa: BLE001 - inference failure must not kill the run
            log_downgrade(log, "semantic dynamic masking", "geometric consistency only",
                          f"inference failed: {type(exc).__name__}: {exc}")
            self.available = False
            return DynamicMaskResult(mask=empty, method="none")

        mask = empty
        counts: dict[str, int] = {}
        for prediction in predictions:
            names = getattr(prediction, "names", {}) or {}
            masks = getattr(prediction, "masks", None)
            boxes = getattr(prediction, "boxes", None)
            if masks is None or boxes is None:
                continue
            data = masks.data.cpu().numpy() if hasattr(masks.data, "cpu") else np.asarray(masks.data)
            class_ids = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
            for instance, class_id in zip(data, class_ids):
                label = str(names.get(int(class_id), int(class_id))).lower()
                if label not in self.wanted_classes:
                    continue
                resized = instance
                if instance.shape[:2] != image.shape[:2]:
                    resized = cv2.resize(
                        instance.astype(np.float32), (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = mask | (resized > 0.5)
                counts[label] = counts.get(label, 0) + 1

        mask = dilate_mask(mask, self.dilate_px)
        return DynamicMaskResult(mask=mask, fraction=float(mask.mean()), classes=counts, method="semantic")


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow a mask outward.

    Segmentation boundaries sit slightly inside the object, and the pixels just
    outside a moving car carry its motion blur and its shadow. Dilating is
    cheap insurance against both.
    """
    if pixels <= 0 or not mask.any():
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pixels + 1, 2 * pixels + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def texture_variance(image: np.ndarray, window: int = 9) -> np.ndarray:
    """Local intensity variance, used to qualify the geometric test."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    mean = cv2.blur(gray, (window, window))
    mean_square = cv2.blur(gray * gray, (window, window))
    return np.maximum(mean_square - mean * mean, 0.0)


def geometric_dynamic_mask(
    depth: np.ndarray,
    reprojected_depths: Sequence[np.ndarray],
    image: np.ndarray,
    cfg: Any,
) -> DynamicMaskResult:
    """Flag pixels whose depth disagrees across views in textured regions.

    ``reprojected_depths`` holds this frame's depth as predicted by each
    neighbouring view — same shape as ``depth``, NaN where a neighbour does not
    see the pixel. A pixel is dynamic when enough neighbours see it and enough
    of them disagree by more than the relative threshold.

    Disagreement is measured relatively, not in absolute metres: a 20 cm
    mismatch is decisive at 5 m range and meaningless at 200 m.
    """
    geo_cfg = cfg.get_path("condition.dynamic.geometric")
    empty = np.zeros(depth.shape[:2], dtype=bool)
    if not bool(geo_cfg["enabled"]) or not reprojected_depths:
        return DynamicMaskResult(mask=empty, method="none")

    threshold = float(geo_cfg["depth_disagreement"])
    min_views = int(geo_cfg["min_neighbour_views"])
    min_texture = float(geo_cfg["min_texture_variance"])

    stack = np.stack([np.asarray(d, dtype=np.float32) for d in reprojected_depths], axis=0)
    reference = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(stack) & (stack > 0) & np.isfinite(reference)[None, ...] & (reference > 0)[None, ...]

    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.abs(stack - reference[None, ...]) / np.maximum(reference[None, ...], 1e-6)
    disagreeing = valid & (relative > threshold)

    view_count = valid.sum(axis=0)
    disagree_count = disagreeing.sum(axis=0)
    # Require a majority of the views that see the pixel to disagree; a single
    # dissenting neighbour is more likely an occlusion boundary than a mover.
    mask = (view_count >= min_views) & (disagree_count * 2 > view_count)
    mask &= texture_variance(image) >= min_texture

    mask = dilate_mask(mask, int(cfg.get_path("condition.dynamic.dilate_px")))
    fraction = float(mask.mean())
    if fraction > 0:
        log_event(log, logging.DEBUG, "geometric consistency flagged dynamic pixels",
                  fraction=round(fraction, 4), threshold=threshold)
    return DynamicMaskResult(mask=mask, fraction=fraction, method="geometric")


@dataclass
class HoleDecision:
    """What to do with a region a dynamic mask removed."""

    fill_ground_plane: bool
    reason: str
    area_m2: float

    def to_dict(self) -> dict[str, Any]:
        return {"fill_ground_plane": self.fill_ground_plane, "reason": self.reason,
                "area_m2": round(self.area_m2, 2)}


def decide_hole_policy(
    area_m2: float,
    is_ground_like: bool,
    neighbours_confident: bool,
    cfg: Any,
) -> HoleDecision:
    """Masked regions become gaps, not guesses (spec §5.5).

    The single exception is a small patch of road or terrain ringed by
    confident geometry, where interpolating the ground plane is a defensible
    reconstruction of a surface we genuinely observed around all sides.
    Everything else is flagged and travels to the gap report.
    """
    policy = cfg.get_path("condition.dynamic.hole_policy")
    if not bool(policy["fill_ground_plane"]):
        return HoleDecision(False, "ground-plane filling disabled in config", area_m2)
    if not is_ground_like:
        return HoleDecision(False, "region is not road/terrain; flagged rather than guessed", area_m2)
    if not neighbours_confident:
        return HoleDecision(False, "surrounding geometry is not confident enough to interpolate", area_m2)
    max_area = float(policy["max_fill_area_m2"])
    if area_m2 > max_area:
        return HoleDecision(False, f"region area {area_m2:.1f} m2 exceeds the {max_area:.1f} m2 fill limit", area_m2)
    return HoleDecision(True, "small ground region enclosed by confident geometry", area_m2)


def summarize_masks(results: Sequence[DynamicMaskResult]) -> dict[str, Any]:
    """Aggregate dynamic-masking statistics for the QA report."""
    if not results:
        return {"frames": 0}
    fractions = np.array([r.fraction for r in results], dtype=float)
    classes: dict[str, int] = {}
    for result in results:
        for name, count in result.classes.items():
            classes[name] = classes.get(name, 0) + count
    return {
        "frames": len(results),
        "frames_with_movers": int((fractions > 0).sum()),
        "masked_fraction_mean": round(float(fractions.mean()), 4),
        "masked_fraction_max": round(float(fractions.max()), 4),
        "detections_by_class": classes,
        "methods": sorted({r.method for r in results}),
    }
