"""What an input file really is, decided by its content, never by its extension.

Extensions lie in this domain: every MISB sample clip is an MPEG-2 transport stream,
named ``.ts``, ``.mp4``, ``.mpeg4`` and ``.H264``; DJI flight logs arrive as
``telemetry.csv``, ``DJI_0001-TxtLogToCsv.csv`` or a binary ``.txt``. So:

  * **Video**: any container and codec FFmpeg can demux and decode is accepted (MP4/MOV,
    MPEG-TS/M2TS/MTS, MKV/WebM, AVI, MXF, FLV, 3GP, ASF/WMV, MPEG-PS, raw H.264/H.265
    elementary streams; H.264, H.265, AV1, VP9, MPEG-2, ProRes, DNxHD, MJPEG ...).
    Support is proven by decoding frames at the start, middle and end, not assumed.
  * **Telemetry**: sniffed from the first bytes — MISB ST 0601 KLV, DJI SRT, CSV/TXT flight
    logs, DJI binary flight records, GPX, KML, GeoJSON/JSON, PX4 ULog, ArduPilot DataFlash
    ``.bin``, MAVLink ``.tlog``; embedded in the video as a KLV data stream, a subtitle
    track (DJI) or a GoPro GPMF stream.

Everything that is not supported gets a plain message saying what the file is, why it
cannot be used and how to convert it (``UnsupportedInput``).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.logging import get_logger

log = get_logger(__name__)

# Suffixes worth scanning next to a video for telemetry. Content decides the kind.
TELEMETRY_SUFFIXES = (".srt", ".csv", ".txt", ".gpx", ".kml", ".json", ".geojson", ".ulg", ".bin",
                      ".tlog", ".klv", ".log", ".dat")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".heic", ".webp", ".bmp", ".raw", ".arw")
CONVERT_HINT = "ffmpeg -i <input> -map 0 -c:v libx264 -crf 16 -c:d copy <output>.mp4"

KLV_KEY = bytes.fromhex("060E2B34020B01010E01030101000000")
_SRT_CUE = re.compile(rb"\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}")


class UnsupportedInput(ValueError):
    """An input the system cannot use, with the reason and the fix in the message."""

    def __init__(self, what: str, reason: str, fix: str = ""):
        self.what, self.reason, self.fix = what, reason, fix
        super().__init__(f"{what}: {reason}" + (f" Fix: {fix}" if fix else ""))


# --------------------------------------------------------------------------
# Video containers
# --------------------------------------------------------------------------
@dataclass
class VideoStreamInfo:
    codec: str
    width: int
    height: int
    fps: float
    fps_guessed: float
    frames: int
    pix_fmt: str | None
    bit_depth: int | None
    profile: str | None
    rotation_deg: float
    field_order: str | None
    color_transfer: str | None
    spherical: bool
    bit_rate: int | None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class ContainerProbe:
    """What FFmpeg (PyAV) says about a file, without decoding it."""

    path: str
    size_bytes: int
    format_name: str | None = None
    duration_s: float | None = None
    bit_rate: int | None = None
    tags: dict[str, str] = field(default_factory=dict)
    video: VideoStreamInfo | None = None
    data_streams: list[dict[str, Any]] = field(default_factory=list)
    subtitle_streams: list[dict[str, Any]] = field(default_factory=list)
    audio_streams: int = 0
    probe_backend: str = "pyav"
    error: str | None = None

    @property
    def embedded_telemetry(self) -> list[str]:
        """Kinds of telemetry carried inside the container."""
        kinds = []
        for s in self.data_streams:
            if s.get("kind"):
                kinds.append(s["kind"])
        if self.subtitle_streams:
            kinds.append("embedded_srt")
        return kinds

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedded_telemetry"] = self.embedded_telemetry
        return d


def _clean_tags(meta: Any) -> dict[str, str]:
    out = {}
    for k, v in dict(meta or {}).items():
        key = "".join(ch for ch in str(k) if ch.isprintable()).strip()
        val = "".join(ch for ch in str(v) if ch.isprintable()).strip()
        if key and key not in out:
            out[key] = val
    return out


def _data_stream_kind(codec: str | None, tag: str | None, handler: str) -> str | None:
    blob = f"{codec or ''} {tag or ''} {handler}".lower()
    if "klv" in blob or "smpte" in blob or tag in ("KLVA",):
        return "klv"
    if "gpmd" in blob or "gopro met" in blob:
        return "gpmf"
    if "djmd" in blob or "dbgi" in blob:
        return "dji_protobuf"
    if "camm" in blob:
        return "camm"
    return None


def probe_container(path: Path | str) -> ContainerProbe:
    """Container, streams and tags via PyAV; OpenCV fallback when PyAV is absent."""
    path = Path(path)
    probe = ContainerProbe(path=str(path), size_bytes=path.stat().st_size if path.exists() else 0)
    try:
        import av
    except ImportError:
        return _probe_opencv(path, probe)
    try:
        container = av.open(str(path), metadata_errors="ignore")
    except Exception as exc:  # noqa: BLE001 - any demuxer failure means "not a container FFmpeg reads"
        probe.error = f"{type(exc).__name__}: {exc}"
        return probe
    try:
        probe.format_name = container.format.name
        probe.duration_s = container.duration / 1e6 if container.duration else None
        probe.bit_rate = int(container.bit_rate) if container.bit_rate else None
        probe.tags = _clean_tags(container.metadata)
        for stream in container.streams:
            handler = str(stream.metadata.get("handler_name", "")) if stream.metadata else ""
            ctx = stream.codec_context
            codec = getattr(ctx, "name", None) if ctx is not None else None
            try:
                tag = getattr(stream, "codec_tag", None)
            except Exception:  # noqa: BLE001 - non-ASCII fourcc on some audio streams
                tag = None
            if stream.type == "video" and probe.video is None:
                probe.video = _video_info(stream, ctx)
            elif stream.type == "audio":
                probe.audio_streams += 1
            elif stream.type == "subtitle":
                probe.subtitle_streams.append({"index": stream.index, "codec": codec, "handler": handler})
            elif stream.type in ("data", "unknown"):
                probe.data_streams.append({"index": stream.index, "codec": codec, "tag": tag, "handler": handler,
                                           "kind": _data_stream_kind(codec, tag, handler)})
        if probe.format_name == "mpegts" and probe.data_streams and not any(s["kind"] for s in probe.data_streams):
            # FFmpeg often leaves MISB KLV PIDs unnamed; the transport stream is sniffed for the key.
            from src.ingest.klv import file_is_ts

            if file_is_ts(path) and _file_contains(path, KLV_KEY, 8 << 20):
                probe.data_streams[0]["kind"] = "klv"
    finally:
        container.close()
    return probe


def _video_info(stream: Any, ctx: Any) -> VideoStreamInfo:
    def rate(value: Any) -> float:
        try:
            return float(value) if value else 0.0
        except (TypeError, ZeroDivisionError):
            return 0.0

    tags = _clean_tags(stream.metadata)
    rotation = 0.0
    for key in ("rotate", "rotation"):
        if key in tags:
            try:
                rotation = float(tags[key])
            except ValueError:
                pass
    try:  # PyAV >= 13 exposes the display matrix through side data
        for sd in getattr(stream, "side_data", {}) or {}:
            if "DISPLAYMATRIX" in str(sd).upper():
                rotation = float(getattr(stream.side_data[sd], "rotation", rotation) or rotation)
    except Exception:  # noqa: BLE001
        pass
    pix_fmt = getattr(ctx, "pix_fmt", None)
    depth = None
    if pix_fmt:
        m = re.search(r"p(\d{2})(le|be)?$", pix_fmt)
        depth = int(m.group(1)) if m else 8
    spherical = any("spherical" in k.lower() or "projection" in k.lower() for k in tags) or \
        "equirectangular" in json.dumps(tags).lower()
    return VideoStreamInfo(
        codec=getattr(ctx, "name", "unknown"), width=int(getattr(ctx, "width", 0) or 0),
        height=int(getattr(ctx, "height", 0) or 0), fps=rate(stream.average_rate),
        fps_guessed=rate(getattr(stream, "guessed_rate", None)), frames=int(stream.frames or 0), pix_fmt=pix_fmt,
        bit_depth=depth, profile=getattr(ctx, "profile", None), rotation_deg=rotation,
        field_order=str(getattr(ctx, "field_order", None) or "") or None,
        color_transfer=str(getattr(ctx, "color_trc", None) or "") or None, spherical=spherical,
        bit_rate=int(ctx.bit_rate) if getattr(ctx, "bit_rate", None) else None, tags=tags)


def _probe_opencv(path: Path, probe: ContainerProbe) -> ContainerProbe:
    import cv2

    probe.probe_backend = "opencv"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        probe.error = "OpenCV could not open the file (install PyAV for a precise diagnosis)"
        return probe
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.video = VideoStreamInfo(
        codec="".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip() or "unknown",
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps, fps_guessed=fps, frames=frames, pix_fmt=None, bit_depth=None, profile=None, rotation_deg=0.0,
        field_order=None, color_transfer=None, spherical=False, bit_rate=None)
    probe.duration_s = frames / fps if fps else None
    cap.release()
    return probe


def _file_contains(path: Path, needle: bytes, limit: int) -> bool:
    with open(path, "rb") as fh:
        return needle in fh.read(limit)


def classify_non_video(path: Path) -> tuple[str, str]:
    """(what it is, fix) for a file FFmpeg could not use as a video."""
    suffix = path.suffix.lower()
    head = path.read_bytes()[:4096] if path.is_file() else b""
    if suffix in IMAGE_SUFFIXES or head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n":
        return ("a still image", "This system reconstructs from a single-pass drone *video*. Supply the flight video; "
                "a photo survey needs a photogrammetry tool instead.")
    if head[:4] in (b"PK\x03\x04", b"Rar!", b"7z\xbc\xaf") or suffix in (".zip", ".rar", ".7z", ".tar", ".gz"):
        return ("an archive", "Extract it and pass the video file inside.")
    if head[:5] == b"%PDF-":
        return ("a PDF document", "Pass the drone video file.")
    kind = sniff_telemetry(path)
    if kind:
        return (f"a telemetry file ({kind})", "Pass the video as the input and this file with --telemetry.")
    if head and all(32 <= b < 127 or b in (9, 10, 13) for b in head[:512]):
        return ("a text file", "Pass the drone video file.")
    return ("not a video FFmpeg can read (unknown or damaged container)",
            f"If it plays elsewhere, remux or convert it: {CONVERT_HINT}")


# --------------------------------------------------------------------------
# Telemetry sniffing and discovery
# --------------------------------------------------------------------------
def sniff_telemetry(path: Path | str) -> str | None:
    """Telemetry kind from the first bytes of a file, or None when it is not telemetry."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(65536)
    except OSError:
        return None
    if not head:
        return None
    if head.startswith(b"ULog\x01\x12\x35"):
        return "ulog"
    if head[:2] == b"\xa3\x95":
        return "dataflash"
    if len(head) > 9 and head[8] in (0xFE, 0xFD) and _looks_like_tlog(head):
        return "tlog"
    if KLV_KEY in head[:4096]:
        return "klv"
    try:
        from src.ingest.dji_flight_record import is_dji_flight_record

        if is_dji_flight_record(path):
            return "dji_flight_record"
    except Exception:  # noqa: BLE001 - a sniff never raises
        pass
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    low = text[:2048].lower()
    if low.startswith(b"<?xml") or low.startswith(b"<"):
        if b"<gpx" in low:
            return "gpx"
        if b"<kml" in low:
            return "kml"
        return None
    if text[:1] in (b"{", b"["):
        return "geojson" if b"featurecollection" in low or b'"feature"' in low else "json"
    if _SRT_CUE.search(head):
        return "srt"
    if _looks_like_delimited(head):
        return "csv"
    return None


def _looks_like_tlog(head: bytes) -> bool:
    """MAVLink telemetry log: 8-byte timestamp then a v1 (0xFE) or v2 (0xFD) frame, repeated."""
    pos, frames = 0, 0
    while pos + 10 < len(head) and frames < 4:
        magic = head[pos + 8]
        if magic == 0xFE:
            length = 6 + head[pos + 9] + 2
        elif magic == 0xFD:
            length = 10 + head[pos + 9] + 2 + (13 if head[pos + 10] & 1 else 0)
        else:
            return False
        pos += 8 + length
        frames += 1
    return frames >= 2


def _looks_like_delimited(head: bytes) -> bool:
    try:
        text = head.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return False
    lines = [ln for ln in text.splitlines()[:20] if ln.strip()]
    if len(lines) < 2:
        return False
    for delim in (",", ";", "\t", "|"):
        counts = [ln.count(delim) for ln in lines[:-1]]
        if counts and min(counts) >= 1 and max(counts) - min(counts) <= 1:
            return True
    return bool(re.match(r"^\s*[-+]?\d+\.\d+\s+[-+]?\d+\.\d+", lines[1]))


# Telemetry kinds the parsers read, and what to tell the user for the ones they do not.
SUPPORTED_TELEMETRY = ("klv", "srt", "embedded_srt", "csv", "dji_flight_record", "gpx", "kml", "geojson", "json",
                       "ulog", "dataflash", "tlog", "gpmf")
UNSUPPORTED_TELEMETRY_HINTS = {
    "dji_protobuf": "Newer DJI drones embed telemetry as protobuf ('djmd'), which is not decoded. Turn on "
                    "'Video Caption' (SRT) in DJI Fly/Pilot, or export the flight log (AirData/Flight Reader CSV).",
    "camm": "Camera Motion Metadata (CAMM, 360 cameras) is not decoded. Export GPS as GPX.",
}


@dataclass
class TelemetryCandidate:
    path: str | None           # None for streams inside the video
    kind: str
    found_by: str              # "explicit" | "same name" | "same folder" | "embedded"
    supported: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_telemetry(video: Path | str, explicit: list[Path | str] | None = None,
                       probe: ContainerProbe | None = None, scan_folder: bool = True) -> list[TelemetryCandidate]:
    """Every telemetry source available for a video, most specific first.

    Order: files the user named, streams embedded in the video, files sharing the video's
    name, then telemetry-looking files in the same folder (DJI logs are often called
    ``telemetry.csv`` or ``<name>-TxtLogToCsv.csv``). A folder file is only taken when the
    folder holds one video, so a log is never attached to the wrong clip.
    """
    video = Path(video)
    found: list[TelemetryCandidate] = []
    seen: set[str] = set()

    def add(path: Path | None, kind: str | None, how: str) -> None:
        if kind is None:
            return
        key = str(path.resolve()) if path else f"embedded:{kind}"
        if key in seen:
            return
        seen.add(key)
        supported = kind in SUPPORTED_TELEMETRY
        found.append(TelemetryCandidate(str(path) if path else None, kind, how, supported,
                                        "" if supported else UNSUPPORTED_TELEMETRY_HINTS.get(kind, "not a supported format")))

    for item in explicit or []:
        p = Path(item)
        kind = sniff_telemetry(p)
        if kind is None:
            found.append(TelemetryCandidate(str(p), "unknown", "explicit", False,
                                            "not a telemetry format this system recognises (KLV, SRT, CSV/TXT, "
                                            "DJI flight record, GPX, KML, GeoJSON/JSON, ULog, DataFlash .bin, .tlog)"))
            seen.add(str(p.resolve()))
        else:
            add(p, kind, "explicit")
    probe = probe or probe_container(video)
    for kind in probe.embedded_telemetry:
        add(None, kind, "embedded")
    folder = video.parent
    if folder.is_dir():
        siblings = sorted(p for p in folder.iterdir() if p.is_file() and p != video)
        for p in siblings:
            if p.stem.lower().startswith(video.stem.lower()) and p.suffix.lower() in TELEMETRY_SUFFIXES:
                add(p, sniff_telemetry(p), "same name")
        if scan_folder:
            videos = [p for p in siblings if p.suffix.lower() in (".mp4", ".mov", ".ts", ".mts", ".m2ts", ".mkv",
                                                                   ".avi", ".mxf", ".h264", ".h265", ".mpeg4")]
            if not videos:
                for p in siblings:
                    if p.suffix.lower() in TELEMETRY_SUFFIXES and p.stat().st_size < (2 << 30):
                        add(p, sniff_telemetry(p), "same folder")
    return found
