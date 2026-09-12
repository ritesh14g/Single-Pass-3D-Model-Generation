"""Streaming video decode (spec §4.1).

Three rules from the spec drive this module:

  * accept 1080p and 4K MP4/MOV;
  * use hardware decode (NVDEC) when the build supports it, and downgrade
    loudly rather than failing when it does not;
  * **never decode the whole video into RAM** — a 10-minute 4K clip is ~300 GB
    of raw frames. Everything here streams: score, keep, release.

Two access patterns are supported, because frame selection needs both:

  :meth:`VideoReader.stream` — sequential decode of every frame (or every Nth),
  used for the scoring pass where the cost is dominated by decode anyway.

  :meth:`VideoReader.read_indices` — fetch a specific, sorted set of frames.
  Seeking per frame is slow and unreliable on long-GOP H.264/H.265, so this
  decodes forward and skips by ``grab()`` (which does not decode or convert)
  whenever the gap is small, and only seeks when the gap is large enough to pay
  for itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from src.core.logging import get_logger, log_downgrade, log_event

log = get_logger(__name__)

# Below this many frames, grabbing forward beats seeking: a seek on long-GOP
# video lands on the previous keyframe and re-decodes to the target anyway.
SEEK_THRESHOLD_FRAMES = 60

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".ts"}


@dataclass(frozen=True)
class Frame:
    """One decoded frame with the identity the rest of the pipeline uses."""

    index: int
    timestamp_s: float
    image: np.ndarray  # BGR, uint8

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    fourcc: str
    hardware_decode: bool = False

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 2),
            "fourcc": self.fourcc,
            "hardware_decode": self.hardware_decode,
            "megapixels": round(self.width * self.height / 1e6, 2),
        }


class VideoReader:
    """Streaming reader over a single video file.

    Usable as a context manager; the underlying capture is released on exit.
    """

    def __init__(self, path: Path | str, hardware_decode: bool = True, max_width: int | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        if self.path.suffix.lower() not in SUPPORTED_SUFFIXES:
            log_event(
                log,
                logging.WARNING,
                f"unusual video suffix {self.path.suffix!r}; attempting decode anyway",
                path=str(self.path),
            )
        self.max_width = max_width
        self._cap: cv2.VideoCapture | None = None
        self._position = 0
        self._hardware_requested = hardware_decode
        self._hardware_active = False
        self.metadata = self._probe()

    # -- Lifecycle ----------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        """Open the capture, trying hardware decode first when requested."""
        if self._hardware_requested and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
            params = [
                int(cv2.CAP_PROP_HW_ACCELERATION),
                int(getattr(cv2, "VIDEO_ACCELERATION_ANY", 1)),
            ]
            try:
                cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG, params)
                if cap.isOpened():
                    active = cap.get(cv2.CAP_PROP_HW_ACCELERATION)
                    self._hardware_active = bool(active and active > 0)
                    if not self._hardware_active:
                        log_downgrade(
                            log, "hardware decode", "software decode",
                            "OpenCV build accepted the flag but did not engage an accelerator",
                        )
                    return cap
                cap.release()
            except cv2.error as exc:
                log_downgrade(log, "hardware decode", "software decode", str(exc))
        elif self._hardware_requested:
            log_downgrade(
                log, "hardware decode", "software decode",
                "this OpenCV build has no CAP_PROP_HW_ACCELERATION",
            )

        cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {self.path}")
        return cap

    @property
    def cap(self) -> cv2.VideoCapture:
        if self._cap is None:
            self._cap = self._open()
            self._position = 0
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._position = 0

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- Probing ------------------------------------------------------------
    def _probe(self) -> VideoMetadata:
        cap = self.cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")

        if fps <= 0 or fps > 1000:
            log_event(log, logging.WARNING, "container reports no usable frame rate; assuming 30 fps",
                      path=str(self.path), reported_fps=fps)
            fps = 30.0
        if count <= 0:
            # Some containers lie. Counting by decode is expensive, so trust
            # duration where we can and flag the uncertainty.
            log_event(log, logging.WARNING, "container reports no frame count; will discover it by streaming",
                      path=str(self.path))
            count = 0

        meta = VideoMetadata(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=count,
            duration_s=count / fps if count else 0.0,
            fourcc=fourcc,
            hardware_decode=self._hardware_active,
        )
        log_event(log, logging.INFO, f"opened {self.path.name}", **meta.to_dict())
        return meta

    # -- Frame access -------------------------------------------------------
    def _postprocess(self, image: np.ndarray) -> np.ndarray:
        if self.max_width and image.shape[1] > self.max_width:
            scale = self.max_width / image.shape[1]
            image = cv2.resize(
                image,
                (self.max_width, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return image

    def _timestamp(self, index: int) -> float:
        """Prefer the container's own timestamp; fall back to index/fps."""
        pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms and pos_ms > 0:
            return float(pos_ms) / 1000.0
        return index / self.metadata.fps

    def stream(self, step: int = 1, start: int = 0, max_frames: int | None = None) -> Iterator[Frame]:
        """Yield frames sequentially, decoding only every ``step``-th one.

        Skipped frames are consumed with ``grab()``, which demuxes without
        decoding to RGB — materially cheaper than a full ``read()``.
        """
        if step < 1:
            raise ValueError("step must be >= 1")
        cap = self.cap
        if start > 0:
            self._skip_to(start)

        index = max(start, self._position)
        yielded = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            timestamp = self._timestamp(index)
            self._position = index + 1
            yield Frame(index=index, timestamp_s=timestamp, image=self._postprocess(image))
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
            for _ in range(step - 1):
                if not cap.grab():
                    return
                index += 1
                self._position = index
            index += 1

    def _skip_to(self, target: int) -> None:
        """Advance the capture to ``target`` by grabbing or seeking."""
        cap = self.cap
        gap = target - self._position
        if gap < 0 or gap > SEEK_THRESHOLD_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
            self._position = target
            return
        for _ in range(gap):
            if not cap.grab():
                break
            self._position += 1

    def read_indices(self, indices: Sequence[int]) -> Iterator[Frame]:
        """Yield the requested frames, in ascending index order.

        Indices are sorted and de-duplicated; out-of-range requests are skipped
        with a warning rather than aborting the run.
        """
        wanted = sorted({int(i) for i in indices if i >= 0})
        if not wanted:
            return
        cap = self.cap
        for index in wanted:
            if self.metadata.frame_count and index >= self.metadata.frame_count:
                log_event(log, logging.WARNING, "requested frame past end of video; skipping",
                          index=index, frame_count=self.metadata.frame_count)
                continue
            self._skip_to(index)
            ok, image = cap.read()
            if not ok:
                log_event(log, logging.WARNING, "decode failed; skipping frame", index=index)
                continue
            timestamp = self._timestamp(index)
            self._position = index + 1
            yield Frame(index=index, timestamp_s=timestamp, image=self._postprocess(image))

    def read_one(self, index: int) -> Frame | None:
        for frame in self.read_indices([index]):
            return frame
        return None

    def keyframe_indices(self) -> set[int] | None:
        """Indices of I-frames, or ``None`` when the container can't say.

        Spec §5.3 biases selection toward keyframes because they carry fewer
        inter-frame compression artifacts. OpenCV cannot expose frame types, so
        this needs PyAV; without it the caller simply loses the bias, which is
        an optimisation rather than a requirement.
        """
        try:
            import av
        except ImportError:
            log_downgrade(log, "keyframe detection", "uniform frame treatment",
                          "PyAV is not installed; frame types are unavailable via OpenCV")
            return None
        try:
            keyframes: set[int] = set()
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = "NONKEY"
                time_base = float(stream.time_base) if stream.time_base else 1.0 / self.metadata.fps
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        if frame.pts is None:
                            continue
                        keyframes.add(int(round(frame.pts * time_base * self.metadata.fps)))
            log_event(log, logging.INFO, f"found {len(keyframes)} keyframes", count=len(keyframes))
            return keyframes
        except Exception as exc:  # noqa: BLE001 - any PyAV failure is non-fatal here
            log_downgrade(log, "keyframe detection", "uniform frame treatment", f"{type(exc).__name__}: {exc}")
            return None


def probe_video(path: Path | str, hardware_decode: bool = True) -> VideoMetadata:
    """Open, read metadata, close. Cheap enough to call during validation."""
    with VideoReader(path, hardware_decode=hardware_decode) as reader:
        return reader.metadata
