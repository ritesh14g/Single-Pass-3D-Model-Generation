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
