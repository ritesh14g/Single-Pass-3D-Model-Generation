"""Cheap, whole-video frame sampling for the input check.

Decoding a 10-minute 4K clip takes minutes; the input check has seconds. So one demux pass
(no decoding) picks keyframe packets about a second apart, and only those are decoded, each
with its own decoder, in parallel across the CPU cores. That is the sampling the
motion-vs-GPS comparison needs. Streams with too few keyframes (intra-refresh encoders) fall
back to seeking to evenly spaced times.

Every decode error is counted with the time it happened, so a truncated or damaged file is
reported before a 15-minute run finds out at minute twelve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Sample:
    t: float
    bgr: np.ndarray          # downscaled colour frame


@dataclass
class SampleSet:
    samples: list[Sample] = field(default_factory=list)
    method: str = "keyframes"
    decode_errors: list[float] = field(default_factory=list)
    last_decoded_s: float = 0.0
    keyframe_interval_s: float | None = None
    seconds: float = 0.0


def _resize(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def sample_frames(path: Path | str, duration_s: float, max_samples: int, width: int,
                  min_gap_s: float, max_keyframe_rate: float = 4.0, workers: int = 3) -> SampleSet:
    """Up to ``max_samples`` frames spread over the whole video (keyframes when possible)."""
    import time

    started = time.monotonic()
    try:
        import av  # noqa: F401
    except ImportError:
        out = _sample_opencv(path, duration_s, max_samples, width)
        out.seconds = time.monotonic() - started
        return out
    # Samples are at least min_gap_s apart and at most max_samples in total, but a short clip
    # is sampled more densely than that: the sync check needs enough points to be meaningful.
    gap = min_gap_s
    if duration_s:
        gap = max(min_gap_s, duration_s / max(max_samples, 1))
        gap = min(gap, max(duration_s / 40.0, 0.2))
    out = _sample_keyframes(path, gap, width, max_keyframe_rate, workers)
    if out is None or len(out.samples) < min(12, max_samples):
        seek = _sample_seek(path, duration_s, max_samples, width)
        if out is not None:
            seek.decode_errors = out.decode_errors + seek.decode_errors
        out = seek
    out.seconds = time.monotonic() - started
    return out


def _sample_keyframes(path: Path | str, gap: float, width: int, max_rate: float, workers: int = 3) -> SampleSet | None:
    """One demux pass picks keyframe packets ``gap`` apart; only those are decoded, in parallel.

    Each chosen keyframe gets a fresh decoder (keyframes decode on their own), so the work
    splits across cores and unneeded keyframes cost only their demux. ``max_rate`` is kept
    for the caller's signature; all-intra streams are handled the same way.
    """
    import av
    from concurrent.futures import ThreadPoolExecutor

    out = SampleSet(method="keyframes")
    try:
        container = av.open(str(path), metadata_errors="ignore")
    except Exception:  # noqa: BLE001
        return None
    with container:
        stream = container.streams.video[0]
        name, extradata = stream.codec_context.name, stream.codec_context.extradata
        cw, ch = stream.codec_context.width, stream.codec_context.height
        key_times: list[float] = []
        chosen: list[tuple[float, Any]] = []
        next_t = -1e9
        for packet in container.demux(stream):
            if packet.pts is None or not packet.is_keyframe or packet.size == 0:
                continue
            t = float(packet.pts * packet.time_base) - float((stream.start_time or 0) * stream.time_base)
            key_times.append(t)
            if t >= next_t:
                chosen.append((t, packet))
                next_t = t + gap
    if len(key_times) > 1:
        out.keyframe_interval_s = float(np.median(np.diff(key_times)))
    ow = min(width, cw or width)
    oh = int(round((ch or 1) * ow / max(cw or 1, 1))) // 2 * 2

    def decode(item: tuple[float, Any]) -> tuple[float, np.ndarray | None]:
        t, packet = item
        try:
            ctx = av.CodecContext.create(name, "r")
            if extradata:
                ctx.extradata = extradata
            frames = list(ctx.decode(packet)) + list(ctx.decode(None))
            if not frames:
                return t, None
            return t, frames[0].reformat(width=ow, height=oh, format="bgr24").to_ndarray()
        except Exception:  # noqa: BLE001 - a corrupt keyframe is counted, not fatal
            return t, None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for t, img in pool.map(decode, chosen):
            if img is None:
                out.decode_errors.append(t)
                continue
            out.samples.append(Sample(t, img))
            out.last_decoded_s = max(out.last_decoded_s, t)
    return out


def _sample_seek(path: Path | str, duration_s: float, n: int, width: int) -> SampleSet:
    import av

    out = SampleSet(method="seek")
    with av.open(str(path), metadata_errors="ignore") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = stream.time_base
        for target in np.linspace(0, max(duration_s - 0.5, 0), n):
            try:
                container.seek(int(target / tb) + (stream.start_time or 0), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.time is not None and frame.time >= target - 1e-3:
                        out.samples.append(Sample(float(frame.time), _resize(frame.to_ndarray(format="bgr24"), width)))
                        out.last_decoded_s = max(out.last_decoded_s, float(frame.time))
                        break
            except Exception:  # noqa: BLE001
                out.decode_errors.append(float(target))
    return out


def _sample_opencv(path: Path | str, duration_s: float, n: int, width: int) -> SampleSet:
    out = SampleSet(method="opencv-seek")
    cap = cv2.VideoCapture(str(path))
    for target in np.linspace(0, max(duration_s - 0.5, 0), n):
        cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
        ok, frame = cap.read()
        if not ok:
            out.decode_errors.append(float(target))
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        out.samples.append(Sample(t, _resize(frame, width)))
        out.last_decoded_s = max(out.last_decoded_s, t)
    cap.release()
    return out
