"""Adaptive frame selection (spec §4.3).

Fixed-interval sampling is the wrong tool: a drone that hovers produces a
hundred near-identical frames, and one that races down a corridor produces
consecutive frames with no shared surface at all. Both break photogrammetry —
the first by wasting the time budget, the second by breaking the match graph.

So selection targets *overlap* instead of time. Starting from the last kept
frame, the selector predicts how far ahead the target overlap will be reached,
probes that frame, measures the real overlap by sparse optical flow, and
corrects its stride estimate. Two or three probes per kept frame is typical,
which is what keeps the acceptance test (a 10-minute 4K clip selected in under
60 s) reachable — the alternative, decoding and scoring every frame, is not.

Each candidate then passes the §5.2 blur gate. A rejected candidate is replaced
by the best frame in a short forward window, which also serves the §5.3
preference for I-frames; if the whole window is unusable the run is recorded as
a coverage warning rather than papered over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from src.condition.blur import (
    BlurAssessment,
    BlurProfile,
    BlurVerdict,
    RejectionTracker,
    RollingBaseline,
    assess_blur,
    to_profile_gray,
    summarize_assessments,
)
from src.core.logging import get_logger, log_event
from src.ingest.video_reader import Frame, VideoReader

log = get_logger(__name__)

# Probes spent refining the stride before accepting whatever overlap we have.
MAX_STRIDE_PROBES = 4
# A partial-affine fit needs two correspondences; three keeps it over-determined.
MIN_FIT_POINTS = 3
# Maximum round-trip error, in pixels, for a forward-backward flow check.
FB_ERROR_PX = 1.5
# Reference working width the flow config's pixel values are expressed at.
FLOW_REFERENCE_WIDTH = 640
# Below this warped-area fraction the frames barely intersect, and the
# transform is extrapolation rather than measurement.
MIN_VALID_AREA_FRACTION = 0.05


@dataclass
class SelectedFrame:
    """One kept frame and the evidence for keeping it."""

    index: int
    timestamp_s: float
    overlap_prev: float           # measured overlap with the previously kept frame
    blur_score: float
    blur_verdict: str
    weight: float                 # fusion weight, already reduced for mild blur
    is_keyframe: bool = False
    anisotropy: float = 1.0
    # Normalised affine from the previously kept frame to this one. Downstream
    # stages use it to work over the region the two frames genuinely share.
    transform_prev: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_s": round(self.timestamp_s, 4),
            "overlap_prev": round(self.overlap_prev, 4),
            "blur_score": round(self.blur_score, 2),
            "blur_verdict": self.blur_verdict,
            "weight": round(self.weight, 3),
            "is_keyframe": self.is_keyframe,
            "anisotropy": round(self.anisotropy, 3),
            "transform_prev": list(self.transform_prev) if self.transform_prev else None,
        }


@dataclass
class FrameSelection:
    """The selected frame set plus everything the QA report needs about it."""

    frames: list[SelectedFrame] = field(default_factory=list)
    rejection_runs: list[dict[str, Any]] = field(default_factory=list)
    blur_summary: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    decoded_frames: int = 0
    decode_s: float = 0.0   # time inside VideoCapture read/grab/seek during selection
    # One record per frame the blur gate evaluated (kept or not). Diagnostic
    # only: it is what lets the Stage Lab plot scores against thresholds.
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def evaluations_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.evaluations)
        if not frame.empty:
            frame["selected"] = frame["index"].isin(self.indices)
        return frame

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def indices(self) -> list[int]:
        return [f.index for f in self.frames]

    @property
    def timestamps(self) -> list[float]:
        return [f.timestamp_s for f in self.frames]

    @property
    def overlaps(self) -> np.ndarray:
        """Measured overlaps, excluding the first frame which has no predecessor."""
        return np.array([f.overlap_prev for f in self.frames[1:]], dtype=float)

    def overlap_stats(self) -> dict[str, Any]:
        values = self.overlaps
        if values.size == 0:
            return {"pairs": 0}
        return {
            "pairs": int(values.size),
            "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "p10": round(float(np.percentile(values, 10)), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
            "below_0.5": int((values < 0.5).sum()),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.frames])

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    def summary(self) -> dict[str, Any]:
        return {
            "selected_frames": len(self.frames),
            "decoded_frames": self.decoded_frames,
            "decode_s": round(self.decode_s, 3),
            "overlap": self.overlap_stats(),
            "blur": self.blur_summary,
            "blur_profile": self.profile,
            "coverage_warnings": self.rejection_runs,
            "keyframes_preferred": sum(1 for f in self.frames if f.is_keyframe),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Overlap estimation
# --------------------------------------------------------------------------
@dataclass
class OverlapEstimate:
    """Measured overlap plus the evidence for believing it.

    ``reliable`` is the field callers act on. It is deliberately not a function
    of how many points tracked: a transform confirmed by near-perfect image
    correlation is trustworthy on six correspondences, and a transform agreed
    on by two hundred correspondences that do not actually align the images is
    not trustworthy at all.
    """

    overlap: float
    support: int             # correspondences agreeing with the fitted transform
    correlation: float       # NCC of the frames after alignment, -1 to 1
    reliable: bool
    # Smallest correlation drop when the fitted shift is perturbed along either
    # axis. Near zero means the match is a ridge, not a peak (aperture problem).
    peak_drop: float = 0.0
    # 2x3 affine taking anchor-frame pixels to candidate-frame pixels, stored
    # with translation normalised by frame width/height so it can be re-applied
    # at any resolution. Downstream stages need this to compare the two frames
    # over the region they actually share.
    transform: tuple[float, ...] | None = None

    def __bool__(self) -> bool:
        return self.reliable


def estimate_overlap(
    anchor_gray: np.ndarray,
    candidate_gray: np.ndarray,
    flow_cfg: Any,
) -> OverlapEstimate:
    """Fraction of the anchor frame still visible in the candidate.

    Sparse Lucas-Kanade flow gives point correspondences; a partial-affine fit
    turns those into the transform between the two views, and the overlap is
    the area of the anchor rectangle's image under that transform, intersected
    with the candidate rectangle. Using the transform rather than a raw
    displacement median means rotation and altitude change are accounted for,
    not just pan.

    Correspondences are screened by a **forward-backward** check first. LK
    reports success for a point whose window it merely found *a* minimum in,
    and on repetitive aerial texture that routinely means a point that appears
    not to have moved at all. Those failures are not random noise — they
    cluster at zero displacement, so they survive a median and bias the
    estimate toward "no motion", which would make the selector keep
    near-duplicate frames forever. Tracking back from the candidate to the
    anchor and demanding the round trip return to within a pixel removes them.

    The fitted transform is then verified **photometrically**. Correspondences
    can agree with each other and still be wrong together, and the failure is
    not hypothetical: forward-backward screening removes points that disagree,
    but on content whose coarse pyramid levels are smooth, LK converges to zero
    displacement in *both* directions and the round trip confirms it. Two
    unrelated frames then report near-total overlap, which would make the
    selector step forward forever believing it had not moved. Aligning the
    frames and measuring whether they actually match catches this, because
    agreement between features says nothing about agreement between images.

    When the transform cannot be believed, the result is reported as
    unreliable with zero overlap — the caller's response to low overlap is to
    shorten its stride, which is the safe direction to be wrong in.
    """
    h, w = anchor_gray.shape[:2]
    corners = cv2.goodFeaturesToTrack(
        anchor_gray,
        maxCorners=int(flow_cfg["max_corners"]),
        qualityLevel=float(flow_cfg["quality_level"]),
        minDistance=float(flow_cfg["min_distance"]),
    )
    if corners is None or len(corners) < MIN_FIT_POINTS:
        return _phase_correlation_fallback(anchor_gray, candidate_gray, 0, 0.0, flow_cfg)

    win = int(flow_cfg["win_size"])
    lk_args = {
        "winSize": (win, win),
        "maxLevel": int(flow_cfg["pyramid_levels"]),
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, status, _ = cv2.calcOpticalFlowPyrLK(anchor_gray, candidate_gray, corners, None, **lk_args)
    backward, back_status = None, None
    if forward is not None and status is not None:
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            candidate_gray, anchor_gray, forward, None, **lk_args
        )
    if forward is None or status is None or backward is None or back_status is None:
        return _phase_correlation_fallback(anchor_gray, candidate_gray, 0, 0.0, flow_cfg)

    round_trip = np.linalg.norm(corners.reshape(-1, 2) - backward.reshape(-1, 2), axis=1)
    keep = (
        status.reshape(-1).astype(bool)
        & back_status.reshape(-1).astype(bool)
        & (round_trip <= FB_ERROR_PX)
    )
    source = corners.reshape(-1, 2)[keep]
    target = forward.reshape(-1, 2)[keep]
    if len(source) < MIN_FIT_POINTS:
        return _phase_correlation_fallback(anchor_gray, candidate_gray, int(len(source)), 0.0, flow_cfg)

    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=500
    )
    if matrix is None:
        return _phase_correlation_fallback(anchor_gray, candidate_gray, int(len(source)), 0.0, flow_cfg)

    support = int(inliers.sum()) if inliers is not None else len(source)
    overlap = _overlap_from_transform(matrix, w, h)
    correlation, peak_drop = _verify_alignment(anchor_gray, candidate_gray, matrix, flow_cfg)

    if _verified(correlation, peak_drop, flow_cfg) and support >= MIN_FIT_POINTS:
        return OverlapEstimate(overlap, support, correlation, True, peak_drop,
                               normalize_transform(matrix, w, h))

    return _phase_correlation_fallback(anchor_gray, candidate_gray, support, correlation, flow_cfg)


def _phase_correlation_fallback(
    anchor_gray: np.ndarray,
    candidate_gray: np.ndarray,
    feature_support: int,
    feature_correlation: float,
    flow_cfg: Any,
) -> OverlapEstimate:
    """Estimate overlap from whole-image phase correlation.

    Feature tracking fails on exactly the content aerial video is full of:
    water, tarmac, ploughed fields, anything whose corners are weak or
    repetitive. Phase correlation does not need corners — it finds the
    translation that aligns two images from their Fourier phase, using every
    pixel — so it degrades where feature tracking collapses.

    It recovers translation only, which is fine here: frame-to-frame motion on
    a nadir survey pass is translation-dominated, and this is a fallback for
    when the richer model could not be fitted at all. The result still has to
    pass the same photometric check before it is believed.
    """
    h, w = anchor_gray.shape[:2]
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        anchor_gray.astype(np.float32), candidate_gray.astype(np.float32), window
    )
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    correlation, peak_drop = _verify_alignment(anchor_gray, candidate_gray, matrix, flow_cfg)

    if not _verified(correlation, peak_drop, flow_cfg):
        # Neither model aligns the frames verifiably. Report the better of the
        # two correlations so the log shows how far off we were.
        return OverlapEstimate(0.0, feature_support, max(correlation, feature_correlation), False, peak_drop)

    overlap = _overlap_from_transform(matrix, w, h)
    # Phase correlation has no inlier count; the response peak is its own
    # confidence measure, scaled here to the same "supporting evidence" role.
    return OverlapEstimate(
        overlap, max(feature_support, int(response * 100)), correlation, True, peak_drop,
        normalize_transform(matrix, w, h),
    )


def normalize_transform(matrix: np.ndarray, w: int, h: int) -> tuple[float, ...]:
    """Flatten a 2x3 affine with translation expressed in frame fractions.

    The transform is fitted on downscaled frames but applied later at whatever
    resolution the conditioning stage works at. Both are the same frame scaled
    uniformly, so the linear part is resolution-independent and only the
    translation needs normalising.
    """
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    return (float(a), float(b), float(tx) / w, float(c), float(d), float(ty) / h)


def denormalize_transform(values: Sequence[float], w: int, h: int) -> np.ndarray:
    """Inverse of :func:`normalize_transform` at a target resolution."""
    a, b, tx, c, d, ty = values
    return np.array([[a, b, tx * w], [c, d, ty * h]], dtype=np.float64)


def _overlap_from_transform(matrix: np.ndarray, w: int, h: int) -> float:
    """Fraction of the frame rectangle still inside the frame after transform."""
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    warped = cv2.transform(rect.reshape(1, -1, 2).astype(np.float32), matrix).reshape(-1, 2)
    area, _ = cv2.intersectConvexConvex(warped.astype(np.float32), rect)
    return float(np.clip(float(area) / float(w * h), 0.0, 1.0))


def _verified(correlation: float, peak_drop: float, flow_cfg: Any) -> bool:
    return correlation >= float(flow_cfg["min_alignment_ncc"]) and peak_drop >= float(flow_cfg["min_peak_drop"])


def _verify_alignment(
    anchor_gray: np.ndarray, candidate_gray: np.ndarray, matrix: np.ndarray, flow_cfg: Any
) -> tuple[float, float]:
    """Photometric verification of a fitted transform: ``(correlation, peak_drop)``.

    ``correlation`` is the zero-mean NCC of the frames after alignment; a
    transform that doesn't actually align the images is not believed, however
    many feature correspondences agreed on it.

    ``peak_drop`` guards against the aperture problem. When content is
    invariant along one axis (a road, river or crop row running with the flight
    line) the frames correlate just as well at *any* along-track offset, so the
    fitted shift along that axis is arbitrary and the reported overlap is
    fiction. The fit is perturbed by a few pixels along each axis in both
    directions; a genuine peak drops on both axes, a ridge stays flat on one.
    ``peak_drop`` is the smaller of the two axis drops.

    All five correlations run on frames downscaled to ``verify_width`` and over
    one shared support (the valid region eroded by the probe distance), so they
    are directly comparable and cheap — profiling showed full-resolution
    verification was 43% of selection time.
    """
    h, w = anchor_gray.shape[:2]
    scale = min(1.0, float(flow_cfg["verify_width"]) / w)
    fitted = matrix.astype(np.float64).copy()
    if scale < 1.0:
        size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
        a = cv2.resize(anchor_gray, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(candidate_gray, size, interpolation=cv2.INTER_AREA)
        fitted[:, 2] *= scale  # uniform scaling leaves the linear part unchanged
    else:
        a, b = anchor_gray, candidate_gray
    hs, ws = a.shape[:2]
    probe = max(1, int(round(float(flow_cfg["peak_probe_px"]) * ws / FLOW_REFERENCE_WIDTH)))

    support = cv2.warpAffine(np.full((hs, ws), 255, np.uint8), fitted, (ws, hs),
                             flags=cv2.INTER_NEAREST, borderValue=0)
    support = cv2.erode(support, np.ones((2 * probe + 1, 2 * probe + 1), np.uint8)) > 0
    if support.sum() < MIN_VALID_AREA_FRACTION * ws * hs:
        return 0.0, 0.0

    target = b[support].astype(np.float32)
    target -= target.mean()
    target_norm = float(np.sqrt(np.dot(target, target)))
    if target_norm <= 1e-6:
        return 0.0, 0.0

    def correlate(m: np.ndarray) -> float:
        warped = cv2.warpAffine(a, m, (ws, hs), flags=cv2.INTER_LINEAR, borderValue=0)
        values = warped[support].astype(np.float32)
        values -= values.mean()
        norm = float(np.sqrt(np.dot(values, values)))
        return 0.0 if norm <= 1e-6 else float(np.dot(values, target) / (norm * target_norm))

    correlation = correlate(fitted)
    drops = []
    for axis in (0, 1):
        neighbour = -1.0
        for sign in (1, -1):
            shifted = fitted.copy()
            shifted[axis, 2] += sign * probe
            neighbour = max(neighbour, correlate(shifted))
        drops.append(correlation - neighbour)
    return correlation, float(min(drops))


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def _stride_bounds(sel_cfg: Any, fps: float) -> tuple[int, int]:
    """Resolve the stride guard rails to frame counts for this clip's fps.

    The guard rails exist to stop a bad flow estimate from starving or
    exploding the frame set, but stating them in frames ties them to a frame
    rate. Measured on a 1080p60 forward-oblique clip, the 90-frame cap is 1.5 s
    of a scene whose content barely changes in that time: every kept pair came
    out at 0.997 overlap against a 0.70-0.80 target, and selection spent its
    budget re-verifying near-duplicates. The same cap on 30 fps survey footage
    is 3 s and reasonable. So when a seconds form is configured it wins, and
    the frame count remains the fallback for a clip whose fps is unknown.
    """
    min_frames_cap = int(sel_cfg["min_stride_frames"])
    max_frames_cap = int(sel_cfg["max_stride_frames"])
    min_seconds = sel_cfg.get("min_stride_seconds") if hasattr(sel_cfg, "get") else None
    max_seconds = sel_cfg.get("max_stride_seconds") if hasattr(sel_cfg, "get") else None

    if fps and fps > 0:
        if min_seconds is not None:
            min_frames_cap = int(round(float(min_seconds) * fps))
        if max_seconds is not None:
            max_frames_cap = int(round(float(max_seconds) * fps))

    min_stride = max(1, min_frames_cap)
    max_stride = max(min_stride + 1, max_frames_cap)
    return min_stride, max_stride


def select_frames(
    reader: VideoReader,
    cfg: Any,
    profile: BlurProfile,
    keyframes: set[int] | None = None,
    budget: Any = None,
) -> FrameSelection:
    """Select frames targeting the configured overlap, gated on blur."""
    sel_cfg = cfg.get_path("ingest.frame_selection")
    flow_cfg = sel_cfg["flow"]
    target = float(sel_cfg["target_overlap"])
    tolerance = float(sel_cfg["overlap_tolerance"])
    min_stride, max_stride = _stride_bounds(sel_cfg, reader.metadata.fps)
    min_frames = int(sel_cfg["min_frames"])
    max_frames = int(sel_cfg["max_frames"])
    prefer_keyframes = bool(sel_cfg["prefer_keyframes"]) and bool(keyframes)
    search_radius = int(sel_cfg["keyframe_search_radius"])
    max_consecutive = int(cfg.get_path("condition.blur.max_consecutive_rejects"))
    # Sharpness is scene-dependent, so the gate compares each frame with the
    # frames around it rather than with the whole clip (issue S1-7). None when
    # the rolling baseline is switched off, which restores whole-video thresholds.
    baseline = RollingBaseline.from_config(profile, cfg)

    total_frames = reader.metadata.frame_count
    selection = FrameSelection(profile=profile.to_dict())
    decode_started = reader.decode_s
    tracker = RejectionTracker(max_consecutive)
    # (frame index, timestamp, assessment) for every frame the gate evaluated.
    assessments: list[tuple[int, float, BlurAssessment]] = []
    decoded = 0

    # -- Seed: the first frame that survives the blur gate -------------------
    anchor: Frame | None = None
    anchor_gray: np.ndarray | None = None
    for candidate in reader.stream(step=1, max_frames=search_radius * 4 + 1):
        decoded += 1
        assessment = assess_blur(candidate.image, profile, cfg, baseline)
        assessments.append((candidate.index, candidate.timestamp_s, assessment))
        tracker.record(candidate.index, candidate.timestamp_s, assessment.rejected)
        if not assessment.rejected:
            anchor = candidate
            anchor_gray = to_profile_gray(candidate.image)
            selection.frames.append(
                SelectedFrame(
                    index=candidate.index,
                    timestamp_s=candidate.timestamp_s,
                    overlap_prev=1.0,
                    blur_score=assessment.score,
                    blur_verdict=assessment.verdict.value,
                    weight=assessment.weight,
                    is_keyframe=bool(keyframes and candidate.index in keyframes),
                    anisotropy=assessment.anisotropy,
                )
            )
            break

    if anchor is None or anchor_gray is None:
        selection.notes.append("no frame in the opening window passed the blur gate")
        log_event(log, logging.ERROR, "frame selection found no usable opening frame")
        selection.decoded_frames = decoded
        selection.decode_s = reader.decode_s - decode_started
        selection.blur_summary = summarize_assessments([a for _, _, a in assessments])
        selection.evaluations = _evaluation_records(assessments)
        selection.rejection_runs = [r.to_dict() for r in tracker.finish()]
        return selection

    # -- Main loop: predict stride, probe, correct ---------------------------
    stride = max(min_stride, min(max_stride, int(round(reader.metadata.fps * 0.5)) or min_stride))
    fallback_stride = stride
    # Floor for the next probe. This is what guarantees termination: the anchor
    # advances when a frame is kept, and this cursor advances when one is not,
    # so no iteration can re-probe ground the previous one already covered.
    scan_floor = anchor.index + min_stride

    while True:
        if len(selection.frames) >= max_frames:
            selection.notes.append(f"stopped at the configured maximum of {max_frames} frames")
            break
        if total_frames and max(anchor.index + min_stride, scan_floor) >= total_frames:
            break

        # Budget pressure: widen the stride rather than blow the deadline.
        if budget is not None and len(selection.frames) > 0:
            progress = (anchor.index / total_frames) if total_frames else 0.0
            if budget.should_degrade(progress=progress) and budget.next_action() is not None:
                action = budget.degrade(
                    "reduce_frames",
                    "frame selection projected to overrun its budget",
                    stride_before=stride,
                )
                if action == "reduce_frames":
                    target = max(target - 0.1, 0.4)
                    selection.notes.append(f"overlap target relaxed to {target:.2f} under time pressure")

        probe = _probe_for_overlap(
            reader=reader,
            anchor=anchor,
            anchor_gray=anchor_gray,
            stride=stride,
            target=target,
            tolerance=tolerance,
            min_stride=min_stride,
            max_stride=max_stride,
            total_frames=total_frames,
            flow_cfg=flow_cfg,
            min_index=scan_floor,
            max_stride_growth=float(sel_cfg["max_stride_growth"]),
        )
        decoded += probe.decoded
        probe_index, probe_frame, probe_gray = probe.index, probe.frame, probe.gray
        if probe_frame is None or probe_gray is None:
            break

        if not probe.reliable:
            # Keeping the frame is still right — dropping it would leave a
            # larger hole — but the pair's overlap is unmeasured, so the region
            # it covers must be reported as lower confidence rather than
            # silently counted as a well-connected part of the match graph.
            note = (
                f"overlap between frames {anchor.index} and {probe_index} could not be verified; "
                "this flight segment is reduced-confidence"
            )
            selection.notes.append(note)
            log_event(log, logging.WARNING, note, event="coverage_warning",
                      anchor=anchor.index, probe=probe_index)

        # The stride that worked here is the starting guess for the next step.
        stride = max(min_stride, min(max_stride, probe_index - anchor.index))
        fallback_stride = stride

        (
            chosen, chosen_assessment, chosen_gray, chosen_overlap, decoded_here, chosen_transform
        ) = _choose_frame(
            reader=reader,
            first=probe_frame,
            first_gray=probe_gray,
            first_overlap=probe.overlap,
            first_transform=probe.transform,
            min_overlap=target - tolerance,
            anchor_gray=anchor_gray,
            cfg=cfg,
            profile=profile,
            keyframes=keyframes if prefer_keyframes else None,
            search_radius=search_radius,
            total_frames=total_frames,
            flow_cfg=flow_cfg,
            tracker=tracker,
            assessments=assessments,
            baseline=baseline,
        )
        decoded += decoded_here

        if chosen is None:
            # Nothing usable in the window. Step past it and keep the anchor;
            # the rejection tracker has already logged the coverage warning.
            scan_floor = probe_index + search_radius + 1
            if total_frames and scan_floor >= total_frames:
                break
            stride = min(max_stride, max(min_stride, fallback_stride + search_radius + 1))
            continue

        selection.frames.append(
            SelectedFrame(
                index=chosen.index,
                timestamp_s=chosen.timestamp_s,
                overlap_prev=chosen_overlap,
                blur_score=chosen_assessment.score,
                blur_verdict=chosen_assessment.verdict.value,
                weight=chosen_assessment.weight,
                is_keyframe=bool(keyframes and chosen.index in keyframes),
                anisotropy=chosen_assessment.anisotropy,
                transform_prev=chosen_transform,
            )
        )
        anchor, anchor_gray = chosen, chosen_gray
        scan_floor = anchor.index + min_stride

    selection.decoded_frames = decoded
    selection.decode_s = reader.decode_s - decode_started
    selection.blur_summary = summarize_assessments([a for _, _, a in assessments])
    selection.evaluations = _evaluation_records(assessments)
    selection.rejection_runs = [r.to_dict() for r in tracker.finish()]

    if len(selection.frames) < min_frames:
        note = (
            f"only {len(selection.frames)} frames selected, below the configured minimum of "
            f"{min_frames} — reconstruction will be sparse"
        )
        selection.notes.append(note)
        log_event(log, logging.WARNING, note, event="coverage_warning", selected=len(selection.frames))

    stats = selection.overlap_stats()
    log_event(
        log,
        logging.INFO,
        f"selected {len(selection.frames)} frames from {total_frames or 'unknown'}",
        selected=len(selection.frames),
        decoded=decoded,
        overlap_median=stats.get("median"),
        overlap_p10=stats.get("p10"),
        rejected=selection.blur_summary.get("rejected"),
    )
    return selection


def _probe_for_overlap(
    reader: VideoReader,
    anchor: Frame,
    anchor_gray: np.ndarray,
    stride: int,
    target: float,
    tolerance: float,
    min_stride: int,
    max_stride: int,
    total_frames: int,
    flow_cfg: Any,
    min_index: int = 0,
    max_stride_growth: float = 2.0,
) -> "ProbeResult":
    """Find a frame whose overlap with the anchor is near the target.

    The stride correction assumes overlap loss is roughly proportional to
    distance travelled, which holds over the few-frame scales involved here:
    if we are at overlap ``o`` after ``s`` frames and want ``target``, then
    ``s * (1 - target) / (1 - o)`` is the stride to try next.

    An **unverifiable** probe is never offered as a candidate. If the estimator
    cannot confirm the overlap, the honest reading is "we have gone too far to
    tell", and the response is to halve the stride and look closer in — not to
    keep the frame and record an overlap nobody measured. When every probe
    comes back unverifiable, a final probe at the shortest legal stride is
    taken, since adjacent frames are the case most likely to align.

    ``min_index`` is the floor the caller has already scanned past; probing
    below it would re-examine frames a previous iteration rejected.
    """
    decoded = 0
    best: _Probe | None = None          # reliable candidates only
    last: _Probe | None = None          # last probe of any kind, for the fallback path
    tried: set[int] = set()
    current_stride = stride

    def probe_at(index: int) -> _Probe | None:
        nonlocal decoded
        if total_frames and index >= total_frames:
            index = total_frames - 1
        if index <= anchor.index or index < min_index or index in tried:
            return None
        tried.add(index)
        frame = reader.read_one(index)
        if frame is None:
            return None
        decoded += 1
        gray = to_profile_gray(frame.image)
        estimate = estimate_overlap(anchor_gray, gray, flow_cfg)
        return _Probe(index=index, frame=frame, gray=gray, estimate=estimate)

    for _ in range(MAX_STRIDE_PROBES):
        index = max(anchor.index + max(min_stride, min(max_stride, current_stride)), min_index)
        candidate = probe_at(index)
        if candidate is None:
            break
        last = candidate

        if not candidate.estimate.reliable:
            log_event(
                log, logging.DEBUG, "overlap estimate unverifiable; shortening stride",
                anchor=anchor.index, probe=candidate.index, support=candidate.estimate.support,
                correlation=round(candidate.estimate.correlation, 3),
            )
            halved = max(min_stride, current_stride // 2)
            if halved == current_stride:
                break
            current_stride = halved
            continue

        overlap = candidate.estimate.overlap
        if best is None or abs(overlap - target) < abs(best.estimate.overlap - target):
            best = candidate
        if abs(overlap - target) <= tolerance:
            break

        remaining = max(1.0 - overlap, 1e-3)
        scaled = current_stride * (1.0 - target) / remaining
        # Motion is continuous: cap growth relative to the last accepted stride
        # so one bad estimate cannot fling the search far past the overlap region.
        growth_cap = max(int(stride * max_stride_growth), stride + 1)
        next_stride = int(round(np.clip(scaled, min_stride, min(max_stride, growth_cap))))
        if next_stride == current_stride:
            break
        current_stride = next_stride

    if best is None:
        closest = probe_at(max(anchor.index + min_stride, min_index))
        if closest is not None:
            last = closest
            if closest.estimate.reliable:
                best = closest

    chosen = best or last
    if chosen is None:
        return ProbeResult(index=anchor.index, frame=None, gray=None, overlap=0.0,
                           reliable=False, decoded=decoded)
    return ProbeResult(
        index=chosen.index,
        frame=chosen.frame,
        gray=chosen.gray,
        overlap=chosen.estimate.overlap,
        reliable=chosen.estimate.reliable,
        decoded=decoded,
        transform=chosen.estimate.transform,
    )


@dataclass
class _Probe:
    """One probed frame and its overlap estimate."""

    index: int
    frame: Frame
    gray: np.ndarray
    estimate: OverlapEstimate


@dataclass
class ProbeResult:
    """Outcome of a stride search."""

    index: int
    frame: Frame | None
    gray: np.ndarray | None
    overlap: float
    reliable: bool
    decoded: int
    transform: tuple[float, ...] | None = None


def _choose_frame(
    reader: VideoReader,
    first: Frame,
    first_gray: np.ndarray,
    first_overlap: float,
    first_transform: tuple[float, ...] | None,
    anchor_gray: np.ndarray,
    cfg: Any,
    profile: BlurProfile,
    keyframes: set[int] | None,
    search_radius: int,
    total_frames: int,
    flow_cfg: Any,
    tracker: RejectionTracker,
    assessments: list[tuple[int, float, BlurAssessment]],
    min_overlap: float = 0.0,
    baseline: RollingBaseline | None = None,
) -> tuple[
    Frame | None, BlurAssessment | None, np.ndarray | None, float, int, tuple[float, ...] | None
]:
    """Pick the frame to keep from a short forward window.

    The probe frame is preferred when it is clean. Otherwise the window is
    scanned forward — never backward, since a backward seek on long-GOP video
    costs far more than the few frames it would save — looking for a frame that
    is sharper and, where the container told us frame types, an I-frame.

    **A replacement may not trade away the overlap target.** Each frame forward
    loses overlap with the anchor, so ranking the window by sharpness alone
    systematically picks the last frame in it: measured on a synthetic flight,
    probes that landed at 0.76 overlap were swapped for sharper frames at 0.52,
    dragging the median out of the §4.3 band. When the probe is usable, a
    candidate must have a *verified* overlap of at least ``min_overlap`` to
    replace it, and scanning stops once overlap falls below that line. Only
    when the probe itself is rejected may the window fall back to the
    best-overlapping usable frame below the line.
    """
    decoded = 0
    assessment = assess_blur(first.image, profile, cfg, baseline)
    assessments.append((first.index, first.timestamp_s, assessment))
    tracker.record(first.index, first.timestamp_s, assessment.rejected)

    first_is_keyframe = bool(keyframes and first.index in keyframes)
    if assessment.verdict is BlurVerdict.CLEAN and (first_is_keyframe or keyframes is None):
        return first, assessment, first_gray, first_overlap, decoded, first_transform

    probe_usable = not assessment.rejected
    best: tuple[Frame, BlurAssessment, np.ndarray, float, float, tuple[float, ...] | None] | None = None
    fallback: tuple[Frame, BlurAssessment, np.ndarray, float, float, tuple[float, ...] | None] | None = None
    if probe_usable:
        best = (first, assessment, first_gray, first_overlap,
                _candidate_score(assessment, first_is_keyframe), first_transform)

    for offset in range(1, search_radius + 1):
        index = first.index + offset
        if total_frames and index >= total_frames:
            break
        frame = reader.read_one(index)
        if frame is None:
            break
        decoded += 1
        candidate_assessment = assess_blur(frame.image, profile, cfg, baseline)
        assessments.append((frame.index, frame.timestamp_s, candidate_assessment))
        tracker.record(frame.index, frame.timestamp_s, candidate_assessment.rejected)
        if candidate_assessment.rejected:
            continue
        gray = to_profile_gray(frame.image)
        estimate = estimate_overlap(anchor_gray, gray, flow_cfg)
        # A window candidate sits within a few frames of the probe, so when its
        # own estimate cannot be verified the probe's measured overlap is a
        # better stand-in than an unverified number.
        is_keyframe = bool(keyframes and index in keyframes)
        score = _candidate_score(candidate_assessment, is_keyframe)

        if estimate.reliable and estimate.overlap >= min_overlap:
            if best is None or score > best[4]:
                best = (frame, candidate_assessment, gray, estimate.overlap, score, estimate.transform)
            if candidate_assessment.verdict is BlurVerdict.CLEAN and is_keyframe:
                break  # Cannot do better than a clean I-frame.
            continue

        if probe_usable:
            if estimate.reliable:
                break  # Verified below the band; frames further on only lose more overlap.
            continue   # Unverifiable (e.g. smeared) — never a replacement for a usable probe.

        # Probe rejected: remember the best-overlapping usable frame as a last resort.
        overlap = estimate.overlap if estimate.reliable else first_overlap
        transform = estimate.transform if estimate.reliable else first_transform
        if fallback is None or (estimate.reliable and overlap > fallback[3]):
            fallback = (frame, candidate_assessment, gray, overlap, score, transform)

    chosen = best or fallback
    if chosen is None:
        return None, None, None, 0.0, decoded, None
    return chosen[0], chosen[1], chosen[2], chosen[3], decoded, chosen[5]


def _evaluation_records(assessments: list[tuple[int, float, BlurAssessment]]) -> list[dict[str, Any]]:
    """Flatten gate evaluations, keeping the last verdict if a frame was seen twice."""
    by_index: dict[int, dict[str, Any]] = {}
    for index, timestamp, assessment in assessments:
        by_index[index] = {
            "index": index,
            "timestamp_s": round(timestamp, 4),
            "blur_score": round(assessment.score, 3),
            "anisotropy": round(assessment.anisotropy, 3),
            "verdict": assessment.verdict.value,
            "directional": assessment.directional,
        }
    return [by_index[i] for i in sorted(by_index)]


def _candidate_score(assessment: BlurAssessment, is_keyframe: bool) -> float:
    """Rank window candidates: sharpness first, keyframe status as a tiebreak.

    The keyframe bonus is deliberately small. An I-frame with visibly worse
    sharpness is not worth taking — compression artifacts are correctable
    (§5.3), and blur is not.
    """
    return assessment.score * (1.15 if is_keyframe else 1.0)


def frames_per_chunk(selection: FrameSelection, chunk_frames: int, overlap: int) -> list[list[int]]:
    """Split the selection into overlapping Track B chunks (spec §7.2)."""
    indices = selection.indices
    if chunk_frames <= 0 or len(indices) <= chunk_frames:
        return [indices] if indices else []
    step = max(chunk_frames - overlap, 1)
    chunks: list[list[int]] = []
    for start in range(0, len(indices), step):
        chunk = indices[start : start + chunk_frames]
        if len(chunk) < max(overlap + 1, 2) and chunks:
            chunks[-1].extend(i for i in chunk if i not in chunks[-1])
            break
        chunks.append(chunk)
        if start + chunk_frames >= len(indices):
            break
    return chunks


def load_selection(path: Path | str) -> FrameSelection:
    """Rehydrate a saved selection (resume path)."""
    frame = pd.read_parquet(path)
    selection = FrameSelection()
    for row in frame.to_dict("records"):
        selection.frames.append(
            SelectedFrame(
                index=int(row["index"]),
                timestamp_s=float(row["timestamp_s"]),
                overlap_prev=float(row["overlap_prev"]),
                blur_score=float(row["blur_score"]),
                blur_verdict=str(row["blur_verdict"]),
                weight=float(row["weight"]),
                is_keyframe=bool(row.get("is_keyframe", False)),
                anisotropy=float(row.get("anisotropy", 1.0)),
                transform_prev=(
                    tuple(float(v) for v in row["transform_prev"])
                    if row.get("transform_prev") is not None
                    else None
                ),
            )
        )
    return selection
