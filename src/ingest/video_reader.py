"""Streaming video decode (spec §4.1).

Three rules from the spec drive this module:

  * accept 1080p and 4K MP4/MOV;
  * use hardware decode (NVDEC) when the build supports it, and downgrade
    loudly rather than failing when it does not. The order tried is
    PyNvVideoCodec on the GPU (:class:`NvdecCapture`), then OpenCV's own
    hardware flag, then OpenCV software decode. The pip OpenCV wheel never
    engages NVDEC, so on the cloud box the first rung is the one that matters —
    with only 3 CPU cores there, software 4K decode alone would eat the whole
    §9 ingest budget;
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
import time
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
    decoder: str = "opencv"

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
            "decoder": self.decoder,
            "megapixels": round(self.width * self.height / 1e6, 2),
        }


class NvdecFailure(RuntimeError):
    """NVDEC failed after the stream was open; the reader falls back to OpenCV."""


class NvdecCapture:
    """GPU decode through PyNvVideoCodec, shaped like ``cv2.VideoCapture``.

    Only the subset :class:`VideoReader` uses is implemented. Frames decode
    into GPU memory and are converted RGB->BGR there, so the CPU only pays for
    the copy back. ``grab()`` just advances the position: the decoder serves
    random access itself, so skipped frames are never copied to the host.

    Timestamps are index / fps (constant frame rate), which is what the OpenCV
    path falls back to anyway when the container carries none.
    """

    def __init__(self, path: Path, gpu_id: int = 0):
        import PyNvVideoCodec as nvc
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch sees no CUDA device")
        self._torch = torch
        self._decoder = self._make_decoder(nvc, path, gpu_id, scan=False)
        meta = self._decoder.get_stream_metadata()
        if "mpeg4" in str(getattr(meta, "codec_name", "")).lower().replace("-", ""):
            # PyNvVideoCodec warns that MPEG-4 Part 2 headers misreport the
            # frame count, and a wrong count means reading past the real end.
            # Scanning costs a demux pass, so only pay it for this codec.
            self._decoder = self._make_decoder(nvc, path, gpu_id, scan=True)
            meta = self._decoder.get_stream_metadata()
        self.width = int(getattr(meta, "width", 0) or 0)
        self.height = int(getattr(meta, "height", 0) or 0)
        self.fps = float(getattr(meta, "average_fps", 0) or 0)
        self.frame_count = int(len(self._decoder) or getattr(meta, "num_frames", 0) or 0)
        self.codec = str(getattr(meta, "codec_name", "") or "")
        self._position = 0
        # Prove the path end to end before trusting it with the run: an API
        # mismatch or a MIG slice without a decoder engine fails here, at open,
        # where falling back costs nothing.
        first = self._fetch(0)
        if first.ndim != 3 or first.shape[2] != 3:
            raise RuntimeError(f"unexpected NVDEC frame shape {first.shape}")
        self.height, self.width = int(first.shape[0]), int(first.shape[1])

    @staticmethod
    def _make_decoder(nvc, path: Path, gpu_id: int, scan: bool):
        kwargs = dict(gpu_id=gpu_id, use_device_memory=True, output_color_type=nvc.OutputColorType.RGB)
        if scan:
            try:
                return nvc.SimpleDecoder(str(path), need_scanned_stream_metadata=True, **kwargs)
            except TypeError:  # older PyNvVideoCodec without the flag
                pass
        return nvc.SimpleDecoder(str(path), **kwargs)

    def _fetch(self, index: int) -> np.ndarray:
        tensor = self._torch.from_dlpack(self._decoder[index])
        if tensor.ndim == 3 and tensor.shape[0] == 3 and tensor.shape[-1] != 3:
            tensor = tensor.permute(1, 2, 0)  # planar -> interleaved
        return tensor.flip(-1).contiguous().cpu().numpy()

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoCapture's name
        return self._decoder is not None

    def release(self) -> None:
        self._decoder = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._position >= self.frame_count:
            return False, None
        try:
            image = self._fetch(self._position)
        except Exception as exc:  # noqa: BLE001 - the reader decides how to recover
            raise NvdecFailure(f"{type(exc).__name__}: {exc}") from exc
        self._position += 1
        return True, image

    def grab(self) -> bool:
        if self._position >= self.frame_count:
            return False
        self._position += 1
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._position = max(int(value), 0)
            return True
        return False

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FOURCC:
            code = (self.codec.lower() + "    ")[:4]
            return float(sum(ord(c) << (8 * i) for i, c in enumerate(code)))
        if prop == cv2.CAP_PROP_POS_MSEC and self.fps > 0:
            return max(self._position - 1, 0) / self.fps * 1000.0
        return 0.0


def _codec_key(name: str) -> str:
    """Lower-case alphanumerics only: "H.264", "h264" and "cudaVideoCodec_H264" all contain "h264"."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def reader_options(cfg) -> dict:
    """``VideoReader`` keyword arguments from config, shared by every caller.

    ``device.prefer: cpu`` turns NVDEC off as well: a CPU run must not touch the GPU.
    """
    video = cfg.get_path("ingest.video")
    prefer_cpu = str(cfg.get_path("device.prefer", "auto")).lower() == "cpu"
    return {
        "hardware_decode": bool(video["hardware_decode"]),
        "max_width": video.get("max_width"),
        "nvdec": bool(video.get("nvdec", True)) and not prefer_cpu,
        "nvdec_gpu_id": int(video.get("nvdec_gpu_id", 0)),
        "nvdec_codecs": list(video.get("nvdec_codecs") or []) or None,
        "nvdec_allow_ts": bool(video.get("nvdec_allow_ts", False)),
    }


# PyNvVideoCodec missing is a fact about the install, not the video: say it once.
_NVDEC_UNAVAILABLE: str | None = None


class VideoReader:
    """Streaming reader over a single video file.

    Usable as a context manager; the underlying capture is released on exit.
    """

    def __init__(
        self,
        path: Path | str,
        hardware_decode: bool = True,
        max_width: int | None = None,
        nvdec: bool = True,
        nvdec_gpu_id: int = 0,
        nvdec_codecs: Sequence[str] | None = None,
        nvdec_allow_ts: bool = False,
    ):
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
        self._nvdec_requested = hardware_decode and nvdec
        self._nvdec_gpu_id = nvdec_gpu_id
        self._nvdec_codecs = [_codec_key(c) for c in nvdec_codecs] if nvdec_codecs is not None else None
        self._nvdec_allow_ts = nvdec_allow_ts
        self.decoder = "opencv"
        # Cumulative seconds spent inside VideoCapture read/grab/seek. Survives
        # close() so a profiling pass and a selection pass on one reader add up.
        # This is what separates decode cost from analysis cost in the Stage 1
        # speed KPI — on 4K footage they scale very differently.
        self.decode_s = 0.0
        self.metadata = self._probe()

    # -- Lifecycle ----------------------------------------------------------
    def _open(self) -> cv2.VideoCapture | NvdecCapture:
        """Open the capture, trying hardware decode first when requested."""
        if self._nvdec_requested:
            cap = self._open_nvdec()
            if cap is not None:
                return cap
        self.decoder = "opencv"
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
                    if self._hardware_active:
                        self.decoder = "opencv-hw"
                    else:
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

    def _open_nvdec(self) -> NvdecCapture | None:
        global _NVDEC_UNAVAILABLE
        if _NVDEC_UNAVAILABLE is not None:
            return None
        try:
            import PyNvVideoCodec  # noqa: F401
        except ImportError:
            _NVDEC_UNAVAILABLE = "PyNvVideoCodec is not installed"
            log_downgrade(log, "NVDEC decode", "OpenCV decode", _NVDEC_UNAVAILABLE)
            return None
        # PyNvVideoCodec does not raise on inputs it cannot seek: it segfaults, which
        # no fallback can catch. Stage 2's read_indices killed the process on an
        # MPEG-2 transport stream (Esri, S1-15) after Stage 1 streamed it fine. So
        # the inputs NVDEC gets are chosen up front: no TS containers, and only
        # codecs on the allow-list.
        from src.ingest.klv import file_is_ts

        if not self._nvdec_allow_ts and file_is_ts(self.path):
            log_downgrade(log, "NVDEC decode", "OpenCV decode",
                          "MPEG-2 transport stream: PyNvVideoCodec random access segfaults on it "
                          "(ingest.video.nvdec_allow_ts)")
            return None
        try:
            cap = NvdecCapture(self.path, gpu_id=self._nvdec_gpu_id)
        except Exception as exc:  # noqa: BLE001 - any failure here just means "use OpenCV"
            log_downgrade(log, "NVDEC decode", "OpenCV decode", f"{type(exc).__name__}: {exc}")
            return None
        if self._nvdec_codecs is not None:
            codec = _codec_key(getattr(cap, "codec", ""))
            if not any(allowed in codec for allowed in self._nvdec_codecs if allowed):
                cap.release()
                log_downgrade(log, "NVDEC decode", "OpenCV decode",
                              f"codec {codec or 'unknown'!r} is not in ingest.video.nvdec_codecs")
                return None
        self._hardware_active = True
        self.decoder = "nvdec"
        return cap

    def _read(self) -> tuple[bool, np.ndarray | None]:
        """``cap.read()``, recovering from an NVDEC failure mid-stream.

        The frame that failed is re-read through OpenCV at the same index, so
        the caller sees no gap: only a slower run and a logged downgrade.
        """
        try:
            return self._timed(self.cap.read)
        except NvdecFailure as exc:
            position = self._position
            log_downgrade(log, "NVDEC decode", "OpenCV decode",
                          f"failed mid-stream at frame {position}: {exc}")
            self._nvdec_requested = False
            self._hardware_active = False
            self.close()
            # Same grab-or-seek rule as any other jump: a raw seek on long-GOP
            # video can land a frame off.
            self._skip_to(position)
            return self._timed(self.cap.read)

    @property
    def cap(self) -> cv2.VideoCapture | NvdecCapture:
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
            decoder=self.decoder,
        )
        log_event(log, logging.INFO, f"opened {self.path.name}", **meta.to_dict())
        return meta

    # -- Frame access -------------------------------------------------------
    def _timed(self, call, *args):
        started = time.perf_counter()
        try:
            return call(*args)
        finally:
            self.decode_s += time.perf_counter() - started

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
        if start > 0:
            self._skip_to(start)

        index = max(start, self._position)
        yielded = 0
        while True:
            ok, image = self._read()
            if not ok:
                break
            timestamp = self._timestamp(index)
            self._position = index + 1
            yield Frame(index=index, timestamp_s=timestamp, image=self._postprocess(image))
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
            for _ in range(step - 1):
                # Through self.cap, not a local: a mid-stream NVDEC fallback
                # swaps the capture underneath this loop.
                if not self._timed(self.cap.grab):
                    return
                index += 1
                self._position = index + 1  # the grabbed frame is consumed
            index += 1

    def _skip_to(self, target: int) -> None:
        """Advance the capture to ``target`` by grabbing or seeking."""
        cap = self.cap
        gap = target - self._position
        if gap < 0 or gap > SEEK_THRESHOLD_FRAMES:
            self._timed(cap.set, cv2.CAP_PROP_POS_FRAMES, float(target))
            self._position = target
            return
        for _ in range(gap):
            if not self._timed(cap.grab):
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
        for index in wanted:
            if self.metadata.frame_count and index >= self.metadata.frame_count:
                log_event(log, logging.WARNING, "requested frame past end of video; skipping",
                          index=index, frame_count=self.metadata.frame_count)
                continue
            self._skip_to(index)
            ok, image = self._read()
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
