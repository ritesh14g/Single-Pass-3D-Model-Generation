"""Synthetic flight fixtures.

Real drone footage is not available at development time and will not be until
the competition dataset lands, so the tests need footage whose ground truth is
known exactly. These helpers build a textured canvas and pan a camera window
across it at a controlled speed, which makes the true frame-to-frame overlap an
arithmetic fact rather than an estimate — exactly what the §4.3 acceptance test
needs to check against.

The same generators back the §8.5 degradation harness: applying a known blur
kernel or a known GPS perturbation to a known-clean input is what turns
"conditioning helps" into a number.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SyntheticFlight:
    """A generated video plus the ground truth used to generate it."""

    video_path: Path
    srt_path: Path | None
    frame_count: int
    fps: float
    width: int
    height: int
    step_px: float           # camera translation per frame, in pixels
    true_overlap: float      # true consecutive-frame overlap fraction
    blurred_indices: list[int]
    lat0: float
    lon0: float


def make_canvas(width: int, height: int, seed: int = 7) -> np.ndarray:
    """A textured 'ground' with structure at several scales.

    The scale mix is the point. Uniform noise is the *pathological* case for
    pyramidal optical flow, not the representative one: averaged down to a
    coarse pyramid level it becomes featureless grey, so the tracker converges
    to zero displacement and confidently reports that nothing moved. Real
    aerial imagery has structure that survives downsampling — field boundaries,
    roads, building footprints — and that is what lets a coarse-to-fine tracker
    find large motions at all.

    So this canvas is built the way a scene is: large regions first, then
    building-sized blocks, then fine texture on top.
    """
    rng = np.random.default_rng(seed)
    canvas = np.full((height, width, 3), 120, dtype=np.uint8)

    # Coarse scale: large regions that survive heavy downsampling.
    for _ in range(max(width * height // 120000, 4)):
        x = int(rng.integers(0, max(width - 200, 1)))
        y = int(rng.integers(0, max(height - 120, 1)))
        w = int(rng.integers(120, max(400, 121)))
        h = int(rng.integers(60, max(200, 61)))
        colour = tuple(int(c) for c in rng.integers(60, 200, size=3))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, -1)

    # Roads: long high-contrast lines, the classic coarse-scale aerial feature.
    for _ in range(max(width // 400, 2)):
        y = int(rng.integers(0, height))
        cv2.line(canvas, (0, y), (width, y), (210, 210, 210), max(height // 60, 3))
    for x in range(0, width, max(width // 12, 120)):
        jitter = int(rng.integers(-20, 21))
        cv2.line(canvas, (x + jitter, 0), (x + jitter, height), (200, 200, 200), 3)

    # Building scale: filled blocks with dark outlines — strong corners.
    for _ in range(max(width * height // 12000, 8)):
        x = int(rng.integers(0, max(width - 90, 1)))
        y = int(rng.integers(0, max(height - 90, 1)))
        w = int(rng.integers(25, 85))
        h = int(rng.integers(25, 85))
        colour = tuple(int(c) for c in rng.integers(40, 230, size=3))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), 2)

    # Fine scale: mild broadband texture, not the dominant signal.
    noise = rng.integers(-25, 26, size=(height, width, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(canvas, (0, 0), 0.8)


def motion_blur(image: np.ndarray, length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    """Convolve with a line kernel — the directional blur of §5.2."""
    length = max(int(length), 3)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return cv2.filter2D(image, -1, kernel)


def make_flight_video(
    path: Path | str,
    frames: int = 60,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    overlap: float = 0.9,
    blur_every: int | None = None,
    blur_length: int = 21,
    seed: int = 7,
) -> SyntheticFlight:
    """Render a constant-velocity pan and write it as an MP4.

    ``overlap`` sets the *true* consecutive-frame overlap; the camera step is
    derived from it, so a test can assert that the selector recovers the
    overlap it was asked to target.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step_px = (1.0 - overlap) * width

    canvas_width = int(width + step_px * frames + 8)
    canvas = make_canvas(canvas_width, height + 8, seed=seed)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a VideoWriter for {path}")

    blurred: list[int] = []
    try:
        for i in range(frames):
            x = int(round(i * step_px))
            frame = canvas[0:height, x : x + width].copy()
            if blur_every and i > 0 and i % blur_every == 0:
                frame = motion_blur(frame, length=blur_length, angle_deg=0.0)
                blurred.append(i)
            writer.write(frame)
    finally:
        writer.release()

    return SyntheticFlight(
        video_path=path,
        srt_path=None,
        frame_count=frames,
        fps=fps,
        width=width,
        height=height,
        step_px=step_px,
        true_overlap=overlap,
        blurred_indices=blurred,
        lat0=12.9716,
        lon0=77.5946,
    )


# --------------------------------------------------------------------------
# Telemetry sidecars — two firmware dialects, deliberately different
# --------------------------------------------------------------------------
def _cue(index: int, start_s: float, end_s: float) -> str:
    def stamp(value: float) -> str:
        hours, rest = divmod(value, 3600)
        minutes, seconds = divmod(rest, 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{millis:03d}"

    return f"{index}\n{stamp(start_s)} --> {stamp(end_s)}"


def write_srt_modern(
    path: Path | str,
    frames: int,
    fps: float = 30.0,
    lat0: float = 12.9716,
    lon0: float = 77.5946,
    speed_mps: float = 5.0,
    rel_alt: float = 60.0,
    abs_alt: float = 920.0,
) -> Path:
    """Mavic-3 era dialect: ``[rel_alt: ... abs_alt: ...]``, gimbal triplet."""
    path = Path(path)
    lines: list[str] = []
    metres_per_degree = 111320.0
    for i in range(frames):
        t = i / fps
        lon = lon0 + (speed_mps * t) / (metres_per_degree * math.cos(math.radians(lat0)))
        lines.append(_cue(i + 1, t, t + 1.0 / fps))
        lines.append(f'<font size="28">SrtCnt : {i + 1}, DiffTime : {int(1000 / fps)}ms')
        lines.append(f"2026-09-12 10:21:{31 + int(t) % 29:02d}.{int((t % 1) * 1000):03d}")
        lines.append(
            "[iso : 100] [shutter : 1/2000.0] [fnum : 2.8] [ev : 0] [ct : 5500] "
            "[color_md : default] [focal_len : 24.00] [dzoom_ratio: 10000, delta:0],"
            f"[latitude: {lat0:.6f}] [longitude: {lon:.6f}] "
            f"[rel_alt: {rel_alt:.3f} abs_alt: {abs_alt:.3f}] "
            "[gb_yaw: 12.3 gb_pitch: -89.9 gb_roll: 0.0] </font>"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_srt_legacy(
    path: Path | str,
    frames: int,
    fps: float = 30.0,
    lat0: float = 12.9716,
    lon0: float = 77.5946,
    speed_mps: float = 5.0,
    altitude: float = 100.5,
) -> Path:
    """Phantom-era dialect: ``FrameCnt``, tenths-of-a-mm focal length, no rel_alt.

    The differences from the modern dialect are the point of this fixture:
    ``altitude`` instead of ``abs_alt``, ``focal_len : 240`` meaning 24 mm, a
    comma-separated timestamp, and no gimbal attitude at all.
    """
    path = Path(path)
    lines: list[str] = []
    metres_per_degree = 111320.0
    for i in range(frames):
        t = i / fps
        lon = lon0 + (speed_mps * t) / (metres_per_degree * math.cos(math.radians(lat0)))
        lines.append(_cue(i + 1, t, t + 1.0 / fps))
        lines.append(f'<font size="36">FrameCnt : {i + 1}, DiffTime : {int(1000 / fps)}ms')
        lines.append(f"2026-09-12 10:21:{31 + int(t) % 29:02d},{int((t % 1) * 1000):03d},000")
        lines.append(
            "[iso : 100] [shutter : 1/240] [fnum : 280] [ev : 0] [ct : 5000] "
            "[color_md : default] [focal_len : 240] "
            f"[latitude : {lat0:.6f}] [longitude : {lon:.6f}] [altitude: {altitude:.6f}] </font>"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_srt_bare(
    path: Path | str,
    frames: int,
    fps: float = 30.0,
    lat0: float = 12.9716,
    lon0: float = 77.5946,
    height: float = 45.0,
) -> Path:
    """OSD-overlay dialect: ``GPS (lon, lat, sats)`` and a bare ``H 45.0m``."""
    path = Path(path)
    lines: list[str] = []
    metres_per_degree = 111320.0
    for i in range(frames):
        t = i / fps
        lon = lon0 + (4.0 * t) / (metres_per_degree * math.cos(math.radians(lat0)))
        lines.append(_cue(i + 1, t, t + 1.0 / fps))
        lines.append(
            f"F/2.8, SS 1000, ISO 100, EV 0, GPS ({lon:.6f}, {lat0:.6f}, 14), "
            f"D 12.3m, H {height:.1f}m, H.S 4.0m/s, V.S 0.0m/s"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_flight_csv(
    path: Path | str,
    frames: int,
    fps: float = 30.0,
    lat0: float = 12.9716,
    lon0: float = 77.5946,
) -> Path:
    """A flight log with headers no parser should hard-code."""
    path = Path(path)
    metres_per_degree = 111320.0
    rows = ["Timestamp,OSD.latitude,OSD.longitude,OSD.altitude [m],OSD.relativeAltitude,OSD.yaw"]
    for i in range(frames):
        t = i / fps
        lon = lon0 + (5.0 * t) / (metres_per_degree * math.cos(math.radians(lat0)))
        rows.append(f"{t:.3f},{lat0:.6f},{lon:.6f},{920.0:.2f},{60.0:.2f},{12.3:.1f}")
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def write_dji_flight_record(
    path: Path | str,
    samples: int = 300,
    rate_hz: float = 10.0,
    recordings: tuple[tuple[float, float], ...] = ((5.0, 15.0),),
    lat0: float = 12.9716,
    lon0: float = 77.5946,
    speed_mps: float = 5.0,
    height_m: float = 60.0,
    gimbal_pitch: float = -90.0,
    version: int = 8,
) -> Path:
    """A binary DJIFlightRecord in the v8 layout: scrambled OSD/gimbal records at
    ``rate_hz`` and a camera record once per second whose flag marks ``recordings``."""
    from src.ingest.dji_flight_record import RECORD_CAMERA, RECORD_GIMBAL, RECORD_OSD, scramble

    path = Path(path)
    metres_per_degree = 111320.0
    body = bytearray()

    def record(record_type: int, plain: bytes, index: int) -> None:
        payload = scramble(plain, record_type, key_byte=(index * 37 + record_type) & 0xFF)
        body.extend(bytes([record_type, len(payload)]) + payload + b"\xff")

    per_second = max(1, int(round(rate_hz)))
    for i in range(samples):
        fly = i / rate_hz
        if i % per_second == 0:
            camera = bytearray(27)
            camera[0] = 0xC0 if any(s <= fly < e for s, e in recordings) else 0x00
            record(RECORD_CAMERA, bytes(camera), i)
        record(RECORD_GIMBAL, struct.pack("<hhh", int(round(gimbal_pitch * 10)), 0, 0) + bytes(6), i)
        osd = bytearray(53)
        lon = lon0 + (speed_mps * fly) / (metres_per_degree * math.cos(math.radians(lat0)))
        struct.pack_into("<dd", osd, 0, math.radians(lon), math.radians(lat0))
        struct.pack_into("<h", osd, 16, int(round(height_m * 10)))
        osd[33], osd[34], osd[36] = 0x80, 5 << 2, 17
        struct.pack_into("<H", osd, 42, int(round(fly * 10)))
        record(RECORD_OSD, bytes(osd), i)

    prefix = bytearray(100)
    struct.pack_into("<QHB", prefix, 0, 100 + len(body), 400, version)
    path.write_bytes(bytes(prefix) + bytes(body) + bytes(400))
    return path


# --------------------------------------------------------------------------
# Ground truth — lets the Stage Lab score a stage against known answers
# --------------------------------------------------------------------------
METRES_PER_DEGREE_LAT = 111320.0
SRT_DIALECTS = ("modern", "legacy", "bare", "csv", "none")


def truth_path_for(video_path: Path | str) -> Path:
    """Ground truth lives next to the video as ``<stem>.truth.json``."""
    video_path = Path(video_path)
    return video_path.with_name(video_path.stem + ".truth.json")


def load_ground_truth(video_path: Path | str) -> dict | None:
    """Return the ground truth for a generated video, or None for real footage."""
    import json

    path = truth_path_for(video_path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def true_lonlat(truth: dict, t: float, lat0: float | None = None) -> tuple[float, float]:
    """Exact lon/lat the telemetry generator wrote for time ``t`` (seconds)."""
    lat0 = truth["lat0"] if lat0 is None else lat0
    lon = truth["lon0"] + (truth["speed_mps"] * t) / (METRES_PER_DEGREE_LAT * math.cos(math.radians(lat0)))
    return lon, lat0


def true_pair_overlap(truth: dict, index_a: int, index_b: int) -> float:
    """True overlap between two frames of a constant-velocity synthetic pan."""
    shift = abs(index_b - index_a) * truth["step_px"]
    return max(0.0, 1.0 - shift / truth["width"])


def generate_synthetic_flight(
    out_dir: Path | str,
    name: str = "synthetic_flight",
    frames: int = 150,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    overlap: float = 0.92,
    blur_every: int | None = 11,
    blur_length: int = 25,
    seed: int = 5,
    telemetry: str = "modern",
    speed_mps: float = 6.0,
) -> tuple[Path, Path | None, Path]:
    """Render a flight, its telemetry sidecar, and a ground-truth JSON.

    Returns ``(video_path, sidecar_path_or_None, truth_path)``. ``telemetry`` is
    one of :data:`SRT_DIALECTS`; "none" exercises the scale-free path.
    """
    import json

    if telemetry not in SRT_DIALECTS:
        raise ValueError(f"telemetry must be one of {SRT_DIALECTS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flight = make_flight_video(
        out_dir / f"{name}.mp4", frames=frames, width=width, height=height, fps=fps,
        overlap=overlap, blur_every=blur_every, blur_length=blur_length, seed=seed,
    )

    sidecar: Path | None = None
    if telemetry == "modern":
        sidecar = write_srt_modern(out_dir / f"{name}.SRT", frames, fps, flight.lat0, flight.lon0, speed_mps)
    elif telemetry == "legacy":
        sidecar = write_srt_legacy(out_dir / f"{name}.SRT", frames, fps, flight.lat0, flight.lon0, speed_mps)
    elif telemetry == "bare":
        # The bare OSD dialect in this module flies at a fixed 4 m/s.
        speed_mps = 4.0
        sidecar = write_srt_bare(out_dir / f"{name}.SRT", frames, fps, flight.lat0, flight.lon0)
    elif telemetry == "csv":
        speed_mps = 5.0
        sidecar = write_flight_csv(out_dir / f"{name}.csv", frames, fps, flight.lat0, flight.lon0)

    truth = {
        "kind": "synthetic_flight",
        "frame_count": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "step_px": flight.step_px,
        "true_overlap_per_frame": overlap,
        "blurred_indices": flight.blurred_indices,
        "lat0": flight.lat0,
        "lon0": flight.lon0,
        "speed_mps": speed_mps if telemetry != "none" else None,
        "telemetry": telemetry,
        "seed": seed,
    }
    truth_file = truth_path_for(flight.video_path)
    truth_file.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return flight.video_path, sidecar, truth_file
