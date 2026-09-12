"""Motion blur detection and gating (spec §5.2, challenge ii).

Three ideas carry this module.

**The threshold is adaptive, not absolute.** Variance-of-Laplacian is a
scene-dependent quantity — a sharp frame of featureless farmland scores lower
than a blurred frame of a tiled roof. So the whole video is profiled first and
the rejection line drawn from *its own* distribution: a frame is rejected when
it is a robust outlier below that video's typical sharpness, floored by an
absolute minimum so a uniformly-blurred video cannot normalise its own
blurriness into acceptability, and capped so that catastrophic footage loses
most but not all of its coverage.

**Directional blur is worse than uniform softness.** A frame that is soft in
every direction may just be haze or a focus miss and still carries usable
structure. A frame smeared along one axis is drone jerk: its features are
displaced, not merely dulled, and matching them produces *wrong* geometry
rather than weak geometry. The FFT of a motion-blurred image shows energy
collapsing along the blur direction, so the anisotropy of the spectrum's
second moment separates the two cases.

That separation carries more weight than the sharpness score does. Measured on
real footage the two populations overlap in variance-of-Laplacian — a soft but
honest frame can score below a lightly smeared one — while anisotropy splits
them cleanly. So anisotropy is treated as a *detection* of motion smear rather
than a tiebreak, and a frame carrying that signature must be at least as sharp
as a typical frame to survive, not merely clear the outlier line.

**Rejection is safe, up to a point.** Frame selection (§4.3) maintains 70-80%
overlap, so dropping a frame costs coverage only when several consecutive
frames go. Runs of rejections are therefore tracked and reported as coverage
warnings for that flight segment, which propagates to reduced confidence for
the corresponding region rather than to silent gaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from src.core.logging import get_logger, log_event

log = get_logger(__name__)

# Frames are profiled at this width; variance-of-Laplacian is resolution
# dependent, so every comparison must happen at one common scale.
PROFILE_WIDTH = 640

# Annulus of the frequency plane used for the anisotropy measurement, as a
# fraction of the Nyquist radius. Below the inner radius the spectrum is
# dominated by scene layout; above the outer radius, by sensor noise.
_ANISO_INNER = 0.15
_ANISO_OUTER = 0.50


class BlurVerdict(str, Enum):
    CLEAN = "clean"        # use at full weight
    CORRECT = "correct"    # mild blur: sharpen conservatively, down-weight
    REJECT = "reject"      # destroyed: exclude entirely


@dataclass
class BlurAssessment:
    """Per-frame blur verdict and the numbers behind it."""

    score: float                # variance of Laplacian
    anisotropy: float           # 1.0 = isotropic; higher = directional
    verdict: BlurVerdict
    weight: float               # fusion weight in [0, 1]
    directional: bool = False
    reason: str = ""

    @property
    def rejected(self) -> bool:
        return self.verdict is BlurVerdict.REJECT

    def to_dict(self) -> dict[str, Any]:
        return {
            "blur_score": round(self.score, 2),
            "anisotropy": round(self.anisotropy, 3),
            "verdict": self.verdict.value,
            "weight": round(self.weight, 3),
            "directional": self.directional,
            "reason": self.reason,
        }


@dataclass
class BlurProfile:
    """Video-wide blur distribution and the thresholds derived from it."""

    reject_threshold: float
    clean_threshold: float
    directional_reject_threshold: float
    anisotropy_limit: float
    samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    degraded_weight: float = 0.6
    unsharp_amount: float = 0.6
    unsharp_radius: float = 1.5

    @classmethod
    def from_samples(cls, samples: Sequence[float], cfg: Any) -> "BlurProfile":
        """Derive thresholds from the video's own score distribution.

        A frame is rejected when it is a **robust outlier** below the typical
        sharpness of this video, or below the absolute floor — whichever bites
        first. The outlier test uses the median and the median absolute
        deviation rather than the mean and standard deviation, because the
        blurred frames we are trying to find are exactly the samples that would
        drag a mean-based threshold down toward themselves.

        Two guards bound the result. The absolute floor stops a uniformly awful
        video from normalising its own awfulness into acceptability. The
        coverage cap stops a video whose blur is not a tail but a *character*
        from losing most of its frames to the gate.
        """
        blur_cfg = cfg.get_path("condition.blur")
        values = np.asarray([s for s in samples if np.isfinite(s)], dtype=float)
        floor = float(blur_cfg["absolute_floor"])
        sigma_multiple = float(blur_cfg["robust_sigma"])
        max_reject = float(blur_cfg["max_reject_fraction"])
        directional_cfg = blur_cfg["directional"]

        if values.size == 0:
            log_event(log, logging.WARNING, "no blur samples; falling back to the absolute floor")
            return cls(
                reject_threshold=floor,
                clean_threshold=floor * 2.0,
                directional_reject_threshold=floor,
                anisotropy_limit=float(directional_cfg["fft_anisotropy_reject"]),
                samples=values,
                degraded_weight=float(blur_cfg["degraded_weight"]),
                unsharp_amount=float(blur_cfg["unsharp_amount"]),
                unsharp_radius=float(blur_cfg["unsharp_radius"]),
            )

        robust = _robust_low_threshold(values, sigma_multiple)
        reject = max(robust, floor)

        # Coverage cap: if the criterion would reject too much of the video,
        # pull it back to the point where exactly the cap fraction falls below.
        rejected_fraction = float((values < reject).mean())
        capped = False
        if rejected_fraction > max_reject:
            reject = float(np.percentile(values, max_reject * 100.0))
            capped = True

        clean = max(float(np.percentile(values, float(blur_cfg["clean_percentile"]))), reject)

        if directional_cfg["enabled"]:
            # Anisotropy above the limit is a detection, not a hint — a sharp
            # frame has no preferred direction in its spectrum. So a frame
            # carrying the motion-smear signature is held to the sharpness of a
            # typical frame rather than merely to the outlier line. Without
            # this, the two populations can overlap in score (a soft-but-sharp
            # frame scoring below a lightly-smeared one) and the gate lets
            # smeared frames through despite having identified them.
            strict = float(np.percentile(values, float(directional_cfg["strict_percentile"])))
            directional_reject = max(strict, reject)
        else:
            directional_reject = reject

        profile = cls(
            reject_threshold=reject,
            clean_threshold=clean,
            directional_reject_threshold=directional_reject,
            anisotropy_limit=float(directional_cfg["fft_anisotropy_reject"]),
            samples=values,
            degraded_weight=float(blur_cfg["degraded_weight"]),
            unsharp_amount=float(blur_cfg["unsharp_amount"]),
            unsharp_radius=float(blur_cfg["unsharp_radius"]),
        )
        log_event(
            log,
            logging.INFO,
            "blur thresholds derived from the video's own distribution",
            samples=int(values.size),
            reject_threshold=round(profile.reject_threshold, 2),
            directional_reject_threshold=round(profile.directional_reject_threshold, 2),
            clean_threshold=round(profile.clean_threshold, 2),
            median=round(float(np.median(values)), 2),
            robust_threshold=round(robust, 2),
            absolute_floor=floor,
            expected_reject_fraction=round(float((values < profile.reject_threshold).mean()), 4),
            binding_constraint=("coverage_cap" if capped else "floor" if floor >= robust else "outlier_test"),
        )
        return profile

    def to_dict(self) -> dict[str, Any]:
        data = {
            "reject_threshold": round(self.reject_threshold, 2),
            "directional_reject_threshold": round(self.directional_reject_threshold, 2),
            "clean_threshold": round(self.clean_threshold, 2),
            "anisotropy_limit": self.anisotropy_limit,
            "n_samples": int(self.samples.size),
        }
        if self.samples.size:
            data["score_percentiles"] = {
                str(p): round(float(np.percentile(self.samples, p)), 2) for p in (5, 25, 50, 75, 95)
            }
        return data


def _robust_low_threshold(values: np.ndarray, sigma_multiple: float) -> float:
    """Lower outlier bound: ``median - k * sigma``, sigma estimated one-sidedly.

    The spread is measured from the **upper half** of the distribution only.
    This matters more than it looks. The contamination we are hunting — blurred
    frames — sits entirely on the low side, so a two-sided MAD is inflated by
    the very samples it is supposed to help detect. The more blurred frames a
    video contains, the wider the estimated spread becomes, the lower the
    threshold drops, and the more blur the gate lets through: the test gets
    *weaker* exactly as the problem gets worse.

    Taking the deviation spread from the sharp half instead estimates the
    scale of the population we want to keep, uncontaminated by the one we want
    to remove. For a normal distribution the same 1.4826 factor applies, so
    ``sigma_multiple`` still reads as a familiar number of sigmas.

    A degenerate distribution — every frame scoring alike, which happens on
    synthetic or heavily processed footage — has zero spread. Falling back to
    the median there would reject half the video, so the fallback is a fraction
    of the median instead: without spread there is no outlier to find.
    """
    median = float(np.median(values))
    upper = values[values >= median]
    scale = float(np.median(upper - median)) if upper.size else 0.0
    if scale <= 1e-9:
        return median * 0.5
    return median - sigma_multiple * 1.4826 * scale


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def to_profile_gray(image: np.ndarray, width: int = PROFILE_WIDTH) -> np.ndarray:
    """Downscale to the common profiling width and convert to grayscale."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.shape[1] > width:
        scale = width / gray.shape[1]
        gray = cv2.resize(gray, (width, max(int(round(gray.shape[0] * scale)), 1)), interpolation=cv2.INTER_AREA)
    return gray


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Sharpness proxy: the variance of the image's Laplacian response."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def spectral_anisotropy(gray: np.ndarray) -> float:
    """Directionality of the frequency spectrum; 1.0 means isotropic.

    Motion blur is a convolution with a line kernel, whose transform is a sinc
    that zeroes energy *along* the blur direction. The resulting spectrum is
    elongated perpendicular to the motion, so the ratio of the second moment's
    eigenvalues measures how directional the degradation is.
    """
    gray = gray.astype(np.float32)
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 1.0
    # Window first: the FFT of a non-periodic image has a bright cross at the
    # axes from edge discontinuity, which would read as strong anisotropy.
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(gray * window)))

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    v = (yy - cy) / cy
    u = (xx - cx) / cx
    radius = np.sqrt(u**2 + v**2)
    band = (radius >= _ANISO_INNER) & (radius <= _ANISO_OUTER)
    if not band.any():
        return 1.0

    energy = (spectrum[band] ** 2).astype(np.float64)
    total = energy.sum()
    if total <= 0:
        return 1.0
    weights = energy / total
    ub, vb = u[band], v[band]
    cov = np.array(
        [
            [float((weights * ub * ub).sum()), float((weights * ub * vb).sum())],
            [float((weights * ub * vb).sum()), float((weights * vb * vb).sum())],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(cov)
    small, large = float(max(eigenvalues[0], 1e-12)), float(max(eigenvalues[1], 1e-12))
    return float(np.sqrt(large / small))


# --------------------------------------------------------------------------
# Profiling and assessment
# --------------------------------------------------------------------------
def profile_video_blur(
    reader: Any,
    cfg: Any,
    max_samples: int = 300,
    measure_anisotropy: bool = False,
) -> BlurProfile:
    """Sample the video to build its blur distribution before selecting frames.

    Cheap by design: frames are skipped with ``grab()`` and scored at
    :data:`PROFILE_WIDTH`, so a 10-minute 4K clip costs a few seconds rather
    than a full decode.
    """
    frame_count = reader.metadata.frame_count
    step = max(int(frame_count // max_samples), 1) if frame_count else 10
    scores: list[float] = []
    anisotropies: list[float] = []

    for frame in reader.stream(step=step, max_frames=max_samples):
        gray = to_profile_gray(frame.image)
        scores.append(variance_of_laplacian(gray))
        if measure_anisotropy:
            anisotropies.append(spectral_anisotropy(gray))
    reader.close()  # Rewind: selection starts from the beginning again.

    profile = BlurProfile.from_samples(scores, cfg)
    if anisotropies:
        log_event(log, logging.DEBUG, "anisotropy distribution sampled",
                  median=round(float(np.median(anisotropies)), 3))
    return profile


def assess_blur(image: np.ndarray, profile: BlurProfile, cfg: Any) -> BlurAssessment:
    """Classify one frame against the video's blur profile."""
    gray = to_profile_gray(image)
    score = variance_of_laplacian(gray)
    directional_cfg = cfg.get_path("condition.blur.directional")

    anisotropy = 1.0
    directional = False
    threshold = profile.reject_threshold
    if directional_cfg["enabled"]:
        anisotropy = spectral_anisotropy(gray)
        directional = anisotropy >= profile.anisotropy_limit
        if directional:
            threshold = profile.directional_reject_threshold

    if score < threshold:
        reason = (
            f"directional blur (anisotropy {anisotropy:.2f}) with score {score:.1f} < {threshold:.1f}"
            if directional
            else f"score {score:.1f} < {threshold:.1f}"
        )
        return BlurAssessment(score, anisotropy, BlurVerdict.REJECT, 0.0, directional, reason)

    if score < profile.clean_threshold:
        return BlurAssessment(
            score,
            anisotropy,
            BlurVerdict.CORRECT,
            profile.degraded_weight,
            directional,
            f"mild blur: {score:.1f} below clean threshold {profile.clean_threshold:.1f}",
        )

    return BlurAssessment(score, anisotropy, BlurVerdict.CLEAN, 1.0, directional, "")


def sharpen(image: np.ndarray, cfg: Any) -> np.ndarray:
    """Conservative unsharp mask for mildly-blurred frames.

    Deliberately gentle: aggressive sharpening manufactures edges, and a
    manufactured edge is a false feature match waiting to happen. Only frames
    the gate graded CORRECT reach here — a frame whose detail was physically
    destroyed is excluded, never sharpened back into plausibility.
    """
    blur_cfg = cfg.get_path("condition.blur")
    amount = float(blur_cfg["unsharp_amount"])
    radius = float(blur_cfg["unsharp_radius"])
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


# --------------------------------------------------------------------------
# Coverage accounting
# --------------------------------------------------------------------------
@dataclass
class RejectionRun:
    """A consecutive run of rejected frames — a potential coverage hole."""

    start_index: int
    end_index: int
    count: int
    start_time_s: float
    end_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "count": self.count,
            "start_time_s": round(self.start_time_s, 2),
            "end_time_s": round(self.end_time_s, 2),
        }


class RejectionTracker:
    """Tracks consecutive rejections and raises coverage warnings.

    A single dropped frame is free — the next kept frame still overlaps its
    neighbour. A run of them is a stretch of flight path with no usable
    imagery, and the region it covers must be marked reduced-confidence rather
    than quietly reconstructed from whatever else happens to see it.
    """

    def __init__(self, max_consecutive: int):
        self.max_consecutive = int(max_consecutive)
        self.runs: list[RejectionRun] = []
        self._current: list[tuple[int, float]] = []

    def record(self, index: int, timestamp_s: float, rejected: bool) -> None:
        if rejected:
            self._current.append((index, timestamp_s))
            return
        self._close()

    def _close(self) -> None:
        if len(self._current) > self.max_consecutive:
            run = RejectionRun(
                start_index=self._current[0][0],
                end_index=self._current[-1][0],
                count=len(self._current),
                start_time_s=self._current[0][1],
                end_time_s=self._current[-1][1],
            )
            self.runs.append(run)
            log_event(
                log,
                logging.WARNING,
                f"coverage warning: {run.count} consecutive frames rejected for blur",
                event="coverage_warning",
                **run.to_dict(),
            )
        self._current = []

    def finish(self) -> list[RejectionRun]:
        self._close()
        return self.runs


def summarize_assessments(assessments: Iterable[BlurAssessment]) -> dict[str, Any]:
    """Aggregate blur verdicts for the QA report."""
    items = list(assessments)
    if not items:
        return {"frames": 0}
    verdicts = [a.verdict for a in items]
    scores = np.array([a.score for a in items], dtype=float)
    return {
        "frames": len(items),
        "clean": verdicts.count(BlurVerdict.CLEAN),
        "corrected": verdicts.count(BlurVerdict.CORRECT),
        "rejected": verdicts.count(BlurVerdict.REJECT),
        "directional_rejects": sum(1 for a in items if a.directional and a.rejected),
        "score_median": round(float(np.median(scores)), 2),
        "score_min": round(float(scores.min()), 2),
        "score_max": round(float(scores.max()), 2),
    }
