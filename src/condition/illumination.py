"""Illumination and shadow handling (spec §5.4, challenge iii).

Two different problems live under "variable illumination", and conflating them
is how reconstructions get worse instead of better.

**(a) Exposure drift.** As the drone turns, auto-exposure moves. MVS
photo-consistency assumes a surface looks the same from every view, so an
exposure step between neighbouring frames reads as a depth error. The fix is
radiometric, not geometric: estimate a per-frame gain/bias against the previous
frame, chain those into a global transform per frame, and normalise everything
to one reference. The fit is done on matched luminance *quantiles* rather than
matched pixels, which needs no correspondence and is robust to the fraction of
the scene that actually changed between frames.

**(b) Cast shadows.** Hard shadows are static in the world but change
appearance with view angle, and they hide facade detail. The spec is explicit
that aggressive de-shadowing before reconstruction introduces artifacts, so
this module only *detects* them and hands a mask downstream: MVS down-weights
shadow pixels in photo-consistency scoring, and relighting happens at texturing
time, after geometry is settled.

Shadow detection uses the physics rather than a threshold on brightness alone.
A shadowed surface is lit by sky rather than sun, so it is simultaneously
darker, less saturated, and *bluer* than the same surface in sun — the three
conditions together separate shadow from dark paint, which brightness alone
cannot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.logging import get_logger, log_event

log = get_logger(__name__)

# Quantiles used for the gain/bias fit. The tails are excluded: blown
# highlights and crushed blacks are clipped, so they carry no gain information.
_FIT_QUANTILES = np.linspace(0.05, 0.95, 19)
# Working width for the exposure estimate; this is a global statistic and does
# not need full resolution.
_FIT_WIDTH = 320
# Minimum overlapping pixels for a paired fit to mean anything.
_MIN_PAIRED_PIXELS = 500
# Trimmed least squares: discard the worst-fitting tail and refit, so moving
# objects and parallax cannot tilt the line.
_TRIM_ITERATIONS = 2
_TRIM_PERCENTILE = 90.0


@dataclass
class ExposureTransform:
    """Affine luminance transform taking one frame to the reference frame."""

    gain: float = 1.0
    bias: float = 0.0
    fit_quality: float = 1.0    # correlation of the quantile fit, 0-1
    clamped: bool = False

    def apply(self, image: np.ndarray) -> np.ndarray:
        if abs(self.gain - 1.0) < 1e-3 and abs(self.bias) < 0.5:
            return image
        return cv2.convertScaleAbs(image, alpha=self.gain, beta=self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain": round(self.gain, 4),
            "bias": round(self.bias, 2),
            "fit_quality": round(self.fit_quality, 3),
            "clamped": self.clamped,
        }


@dataclass
class IlluminationResult:
    """Conditioned frame plus the masks and weights it carries downstream."""

    image: np.ndarray = field(repr=False)
    shadow_mask: np.ndarray | None = field(default=None, repr=False)
    shadow_fraction: float = 0.0
    low_light: bool = False
    weight: float = 1.0
    exposure: ExposureTransform | None = None
    mean_luma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "shadow_fraction": round(self.shadow_fraction, 4),
            "low_light": self.low_light,
            "weight": round(self.weight, 3),
            "mean_luma": round(self.mean_luma, 4),
        }
        if self.exposure is not None:
            data["exposure"] = self.exposure.to_dict()
        return data


# --------------------------------------------------------------------------
# (a) Exposure drift
# --------------------------------------------------------------------------
def fit_gain_bias(
    reference: np.ndarray, target: np.ndarray, transform: Sequence[float] | None = None
) -> ExposureTransform:
    """Fit ``reference ~= gain * target + bias`` between two overlapping frames.

    With a ``transform`` — the normalised affine the frame selector already
    measured between these two frames — the fit uses **real pixel
    correspondence**: the reference is warped into the target's frame and the
    two are compared only where they genuinely overlap. That is what spec
    §5.4(a) asks for, and the distinction is not academic. At 75% overlap a
    quarter of each frame is content the other never saw, so comparing whole-
    frame luminance *distributions* charges the exposure estimate for the scene
    changing: panning from a dark field onto a bright roof reads as the camera
    brightening, and the chained gain then drifts across the flight correcting
    an exposure change that never happened.

    Without a transform the function falls back to matching luminance
    quantiles, which needs no registration and is right in spirit, but carries
    exactly the content-change bias described above.

    The paired fit is trimmed rather than plain least squares: moving objects,
    parallax on tall structures, and specular highlights all violate
    brightness constancy locally, and a handful of such pixels would otherwise
    tilt the line.
    """
    if transform is not None:
        fitted = _fit_paired(reference, target, transform)
        if fitted is not None:
            return fitted

    ref_luma = _luma_small(reference)
    tgt_luma = _luma_small(target)
    ref_q = np.quantile(ref_luma, _FIT_QUANTILES)
    tgt_q = np.quantile(tgt_luma, _FIT_QUANTILES)

    spread = float(tgt_q.max() - tgt_q.min())
    if spread < 5.0:
        # A near-flat frame (fog, uniform water) carries no gain information.
        return ExposureTransform(1.0, 0.0, fit_quality=0.0)

    slope, intercept = np.polyfit(tgt_q, ref_q, 1)
    predicted = slope * tgt_q + intercept
    residual = float(np.sqrt(np.mean((ref_q - predicted) ** 2)))
    quality = float(np.clip(1.0 - residual / max(spread, 1e-6), 0.0, 1.0))
    return ExposureTransform(gain=float(slope), bias=float(intercept), fit_quality=quality)


def _fit_paired(
    reference: np.ndarray, target: np.ndarray, transform: Sequence[float]
) -> ExposureTransform | None:
    """Trimmed least-squares gain/bias over the region two frames share."""
    from src.ingest.frame_selector import denormalize_transform

    ref_luma = _luma_small(reference)
    tgt_luma = _luma_small(target)
    if ref_luma.shape != tgt_luma.shape:
        return None

    h, w = tgt_luma.shape[:2]
    matrix = denormalize_transform(transform, w, h)
    warped = cv2.warpAffine(ref_luma, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    coverage = cv2.warpAffine(
        np.ones((h, w), dtype=np.float32), matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    valid = coverage > 0.99
    if int(valid.sum()) < _MIN_PAIRED_PIXELS:
        return None

    x = tgt_luma[valid].astype(np.float64)
    y = warped[valid].astype(np.float64)
    # Clipped pixels carry no radiometric information: a blown highlight stays
    # blown whatever the gain, so including them biases the slope toward 1.
    unclipped = (x > 2) & (x < 253) & (y > 2) & (y < 253)
    if int(unclipped.sum()) < _MIN_PAIRED_PIXELS:
        return None
    x, y = x[unclipped], y[unclipped]

    spread = float(x.max() - x.min())
    if spread < 5.0:
        return ExposureTransform(1.0, 0.0, fit_quality=0.0)

    slope, intercept = np.polyfit(x, y, 1)
    for _ in range(_TRIM_ITERATIONS):
        residual = np.abs(slope * x + intercept - y)
        keep = residual <= max(float(np.percentile(residual, _TRIM_PERCENTILE)), 1.0)
        if int(keep.sum()) < _MIN_PAIRED_PIXELS:
            break
        slope, intercept = np.polyfit(x[keep], y[keep], 1)

    residual = float(np.sqrt(np.mean((slope * x + intercept - y) ** 2)))
    quality = float(np.clip(1.0 - residual / max(spread, 1e-6), 0.0, 1.0))
    return ExposureTransform(gain=float(slope), bias=float(intercept), fit_quality=quality)


class ExposureChain:
    """Global exposure normalisation across the selected frame sequence.

    Pairwise fits are chained so every frame maps to a single reference (the
    first frame by default). Composition matters: if frame *i* maps to *i-1* as
    ``a*x + c``, and *i-1* maps to the reference as ``g*x + b``, then *i* maps
    to the reference as ``(g*a)*x + (g*c + b)``.

    Chaining accumulates drift, so each link is checked: a fit of poor quality
    or an out-of-range gain is replaced by the identity for that step, which
    keeps the chain anchored rather than letting one bad frame skew everything
    after it.
    """

    def __init__(self, cfg: Any):
        chain_cfg = cfg.get_path("condition.illumination.exposure_chain")
        self.enabled = bool(chain_cfg["enabled"])
        self.max_gain = float(chain_cfg["max_gain"])
        self.min_gain = float(chain_cfg["min_gain"])
        self.transforms: dict[int, ExposureTransform] = {}
        self.rejected_links = 0
        self._previous: np.ndarray | None = None
        self._previous_key: int | None = None
        self._cumulative = ExposureTransform(1.0, 0.0)

    def push(
        self, key: int, image: np.ndarray, geometry: Sequence[float] | None = None
    ) -> ExposureTransform:
        """Add the next frame in sequence and return its transform to reference.

        ``geometry`` is the normalised affine from the previous kept frame to
        this one, as measured during frame selection. When present the fit runs
        over the frames' shared region rather than their whole extents.
        """
        if not self.enabled:
            transform = ExposureTransform(1.0, 0.0)
            self.transforms[key] = transform
            return transform

        if self._previous is None:
            self._cumulative = ExposureTransform(1.0, 0.0)
        else:
            step = fit_gain_bias(self._previous, image, geometry)
            if step.fit_quality < 0.5 or not (self.min_gain <= step.gain <= self.max_gain):
                self.rejected_links += 1
                log_event(
                    log,
                    logging.DEBUG,
                    "exposure link rejected; holding the previous transform",
                    frame=key,
                    gain=round(step.gain, 3),
                    fit_quality=round(step.fit_quality, 3),
                )
                step = ExposureTransform(1.0, 0.0, fit_quality=step.fit_quality)
            composed_gain = self._cumulative.gain * step.gain
            composed_bias = self._cumulative.gain * step.bias + self._cumulative.bias
            clamped = False
            if not (self.min_gain <= composed_gain <= self.max_gain):
                composed_gain = float(np.clip(composed_gain, self.min_gain, self.max_gain))
                clamped = True
            self._cumulative = ExposureTransform(
                gain=composed_gain, bias=composed_bias, fit_quality=step.fit_quality, clamped=clamped
            )

        self._previous = image
        self._previous_key = key
        self.transforms[key] = self._cumulative
        return self._cumulative

    def summary(self) -> dict[str, Any]:
        if not self.transforms:
            return {"enabled": self.enabled, "frames": 0}
        gains = np.array([t.gain for t in self.transforms.values()], dtype=float)
        return {
            "enabled": self.enabled,
            "frames": len(self.transforms),
            "gain_min": round(float(gains.min()), 4),
            "gain_max": round(float(gains.max()), 4),
            "gain_span": round(float(gains.max() - gains.min()), 4),
            "rejected_links": self.rejected_links,
            "clamped_frames": sum(1 for t in self.transforms.values() if t.clamped),
        }


def apply_clahe(image: np.ndarray, cfg: Any) -> np.ndarray:
    """CLAHE on the L channel in LAB, so colour fidelity survives for texturing."""
    clahe_cfg = cfg.get_path("condition.illumination.clahe")
    if not bool(clahe_cfg["enabled"]):
        return image
    tile = tuple(int(v) for v in clahe_cfg["tile_grid"])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    equalizer = cv2.createCLAHE(clipLimit=float(clahe_cfg["clip_limit"]), tileGridSize=tile)
    # Only L is touched: running CLAHE per-channel in BGR shifts hue, and the
    # texture stage needs the colours it was given.
    return cv2.cvtColor(cv2.merge([equalizer.apply(lightness), a_channel, b_channel]), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# (b) Cast shadows
# --------------------------------------------------------------------------
def detect_shadows(image: np.ndarray, cfg: Any) -> tuple[np.ndarray, float]:
    """Per-pixel shadow mask from the darker/less-saturated/bluer signature.

    Returns ``(mask, fraction_of_frame)``. The mask is dilated so the shadow
    *edge* — the strongest false feature a shadow produces — is covered too.
    """
    shadow_cfg = cfg.get_path("condition.illumination.shadow")
    if not bool(shadow_cfg["enabled"]):
        empty = np.zeros(image.shape[:2], dtype=bool)
        return empty, 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    blue = image[:, :, 0].astype(np.float32)
    red = image[:, :, 2].astype(np.float32)
    # Shadowed surfaces are lit by the sky, so blue rises relative to red.
    blue_ratio = (blue + 1.0) / (red + 1.0)

    luminance_cut = float(np.percentile(value, float(shadow_cfg["luminance_percentile"])))
    mask = (
        (value <= luminance_cut)
        & (saturation <= float(shadow_cfg["max_saturation"]))
        & (blue_ratio >= float(shadow_cfg["min_blue_ratio"]))
    )

    dilate = int(shadow_cfg["dilate_px"])
    if dilate > 0 and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    return mask, float(mask.mean())


def shadow_weight_map(shadow_mask: np.ndarray, cfg: Any) -> np.ndarray:
    """Per-pixel photo-consistency weights for MVS (1.0 outside shadow)."""
    weight = float(cfg.get_path("condition.illumination.shadow.mvs_weight"))
    weights = np.ones(shadow_mask.shape, dtype=np.float32)
    weights[shadow_mask] = weight
    return weights


# --------------------------------------------------------------------------
# Low light
# --------------------------------------------------------------------------
def condition_low_light(image: np.ndarray, cfg: Any) -> tuple[np.ndarray, bool, float]:
    """Gamma-correct and denoise dim footage *before* feature detection.

    Returns ``(image, is_low_light, mean_luma)``. A low-light frame keeps its
    label all the way to the QA report: spec §5.4 is explicit that low-light
    accuracy will be measurably worse and that the report must show it rather
    than hide it.
    """
    low_cfg = cfg.get_path("condition.illumination.low_light")
    mean_luma = float(_luma_small(image).mean() / 255.0)
    if mean_luma >= float(low_cfg["mean_luma_threshold"]):
        return image, False, mean_luma

    gamma = float(low_cfg["gamma"])
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    brightened = cv2.LUT(image, table)
    # Brightening amplifies sensor noise, which reads as texture to a feature
    # detector; denoise after the gamma lift, not before.
    denoised = cv2.fastNlMeansDenoisingColored(
        brightened, None, h=float(low_cfg["denoise_h"]), hColor=float(low_cfg["denoise_h"]),
        templateWindowSize=7, searchWindowSize=21,
    )
    log_event(
        log, logging.DEBUG, "low-light frame conditioned",
        mean_luma=round(mean_luma, 4), gamma=gamma,
    )
    return denoised, True, mean_luma


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def condition_illumination(
    image: np.ndarray,
    cfg: Any,
    exposure: ExposureTransform | None = None,
    base_weight: float = 1.0,
) -> IlluminationResult:
    """Run the full §5.4 illumination pass over one frame.

    Order matters: exposure normalisation first (it is a global radiometric
    correction), then low-light conditioning (which depends on the normalised
    brightness), then CLAHE for local contrast, and shadow detection last so
    the mask matches the image the reconstruction will actually see.
    """
    illum_cfg = cfg.get_path("condition.illumination")
    if not bool(illum_cfg["enabled"]):
        return IlluminationResult(image=image, weight=base_weight)

    working = exposure.apply(image) if exposure is not None else image
    working, low_light, mean_luma = condition_low_light(working, cfg)
    working = apply_clahe(working, cfg)
    shadow_mask, shadow_fraction = detect_shadows(working, cfg)

    weight = base_weight
    if low_light:
        weight *= float(illum_cfg["low_light"]["fusion_weight"])

    return IlluminationResult(
        image=working,
        shadow_mask=shadow_mask,
        shadow_fraction=shadow_fraction,
        low_light=low_light,
        weight=weight,
        exposure=exposure,
        mean_luma=mean_luma,
    )


def summarize_illumination(results: Sequence[IlluminationResult]) -> dict[str, Any]:
    """Aggregate illumination statistics for the QA report."""
    if not results:
        return {"frames": 0}
    shadow = np.array([r.shadow_fraction for r in results], dtype=float)
    luma = np.array([r.mean_luma for r in results], dtype=float)
    return {
        "frames": len(results),
        "low_light_frames": sum(1 for r in results if r.low_light),
        "shadow_fraction_mean": round(float(shadow.mean()), 4),
        "shadow_fraction_max": round(float(shadow.max()), 4),
        "mean_luma_min": round(float(luma.min()), 4),
        "mean_luma_max": round(float(luma.max()), 4),
    }


def _luma_small(image: np.ndarray) -> np.ndarray:
    """Downscaled grayscale view used for all global brightness statistics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.shape[1] > _FIT_WIDTH:
        scale = _FIT_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, (_FIT_WIDTH, max(int(round(gray.shape[0] * scale)), 1)),
                          interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)
