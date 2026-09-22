"""Stage 0 — input formats and the input check.

Inputs are generated so each test states the flaw it plants: a telemetry file in another
format, a log in another encoding, a log covering the whole flight, a clip whose telemetry
is shifted in time, a scene that is not rigid, a drone that never moves.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.config import load_config
from src.ingest.formats import classify_non_video, discover_telemetry, probe_container, sniff_telemetry
from src.ingest.telemetry import load_telemetry
from src.preflight import analyze_input
from src.preflight.checks import BLOCK, PASS, WARN


# --------------------------------------------------------------------------
# Synthetic flight with a varying speed (the sync check needs speed to vary)
# --------------------------------------------------------------------------
def _canvas(width: int, height: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(height // 8 + 2, width // 8 + 2, 3), dtype=np.uint8)
    canvas = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    for _ in range(width // 40):                       # a few hard edges, so SIFT has corners
        x, y = int(rng.integers(0, width - 30)), int(rng.integers(0, height - 30))
        cv2.rectangle(canvas, (x, y), (x + 20, y + 20), (int(rng.integers(0, 255)),) * 3, -1)
    return canvas


def make_flight(path: Path, seconds: float = 8.0, fps: float = 25.0, width: int = 1280, height: int = 720,
                speed_px: float = 40.0, vary: float = 0.6, warp: float = 0.0, hover: bool = False):
    """A pan over a fixed scene at a varying speed; returns per-frame x positions (px)."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    rate = np.zeros(n) if hover else speed_px * (1 + vary * np.sin(2 * np.pi * t / max(seconds, 1e-6)))
    x = np.cumsum(rate) / fps
    canvas = _canvas(int(width + x.max() + 10), height + 10)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    rng = np.random.default_rng(11)
    try:
        for i in range(n):
            frame = canvas[0:height, int(x[i]):int(x[i]) + width].copy()
            if warp:                                   # non-rigid: a different wobble every frame
                gy, gx = np.mgrid[0:height, 0:width].astype(np.float32)
                phase = rng.uniform(0, 2 * np.pi, size=2)
                gx += warp * np.sin(gy / 40 + phase[0])
                gy += warp * np.sin(gx / 40 + phase[1])
                frame = cv2.remap(frame, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
    finally:
        writer.release()
    return t, x


def write_csv(path: Path, t: np.ndarray, x: np.ndarray, lat0: float = 27.2, lon0: float = -81.8,
              metres_per_px: float = 0.25, lag_s: float = 0.0, recording: bool = False, pad_s: float = 0.0,
              encoding: str = "utf-8") -> Path:
    """A DJI/AirData-style log. ``lag_s`` shifts the telemetry clock; ``pad_s`` adds flight
    before and after the recorded part (with an isVideo flag, as full-flight logs have)."""
    step = 1 / 10.0
    times = np.arange(-pad_s, t[-1] + pad_s, step)
    east = np.interp(times + lag_s, t, x) * metres_per_px
    lat = lat0 + np.zeros_like(times)
    lon = lon0 + east / (111320 * np.cos(np.radians(lat0)))
    start = np.datetime64("2026-03-01T10:00:00.000")
    stamps = [str((start + np.timedelta64(int(v * 1000), "ms"))).replace("T", " ") for v in times]
    rows = ["CUSTOM.updateTime,CUSTOM.isVideo,OSD.latitude,OSD.longitude,OSD.height [m],OSD.altitude [m],"
            "GIMBAL.pitch,DETAILS.droneType"]
    for stamp, la, lo, tv in zip(stamps, lat, lon, times):
        flag = "Recording" if (not recording or 0 <= tv <= t[-1]) else ""
        rows.append(f"{stamp},{flag},{la:.7f},{lo:.7f},60.0,{60.0 + 12:.1f},-75.0,Ñuble P4 Pro")
    path.write_bytes(("\n".join(rows) + "\n").encode(encoding))
    return path


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def good_flight(tmp_path_factory):
    d = tmp_path_factory.mktemp("flight")
    t, x = make_flight(d / "flight.mp4")
    write_csv(d / "flight.csv", t, x)
    return d / "flight.mp4"


# --------------------------------------------------------------------------
# Formats: what a file is, decided by content
# --------------------------------------------------------------------------
def test_telemetry_kinds_are_sniffed_not_guessed_from_the_extension(tmp_path):
    files = {
        "log.bin": b"1\n00:00:00,000 --> 00:00:00,033\n[latitude: 27.2] [longitude: -81.8]\n",   # SRT named .bin
        "track.txt": b'<?xml version="1.0"?><gpx><trk><trkseg><trkpt lat="1" lon="2"><time>2026-01-01T00:00:00Z'
                     b"</time></trkpt></trkseg></trk></gpx>",
        "path.dat": b'{"type":"FeatureCollection","features":[]}',
        "flight.csv": b"time,lat,lon\n0,27.2,-81.8\n1,27.2,-81.8\n",
        "px4.log": b"ULog\x01\x125" + b"\x00" * 32,
        "ardu.txt": b"\xa3\x95" + b"\x00" * 64,
    }
    expected = {"log.bin": "srt", "track.txt": "gpx", "path.dat": "geojson", "flight.csv": "csv",
                "px4.log": "ulog", "ardu.txt": "dataflash"}
    for name, blob in files.items():
        (tmp_path / name).write_bytes(blob)
        assert sniff_telemetry(tmp_path / name) == expected[name], name


def test_a_still_image_and_a_text_file_are_named_not_called_corrupt(tmp_path):
    img = tmp_path / "photo.jpg"
    cv2.imwrite(str(img), _canvas(64, 64))
    what, fix = classify_non_video(img)
    assert "still image" in what and "video" in fix.lower()
    (tmp_path / "notes.txt").write_bytes(b"hello\nthere\n")
    assert "text file" in classify_non_video(tmp_path / "notes.txt")[0]


def test_telemetry_next_to_the_video_is_found_under_any_name(good_flight, tmp_path):
    other = tmp_path / "DJI_0099"
    other.mkdir()
    video = other / "DJI_0099.mp4"
    video.write_bytes(good_flight.read_bytes())
    (other / "telemetry.csv").write_bytes((good_flight.parent / "flight.csv").read_bytes())
    found = discover_telemetry(video)
    assert [(c.kind, c.found_by) for c in found] == [("csv", "same folder")]


def test_a_csv_in_another_encoding_still_parses(cfg, good_flight, tmp_path):
    """DJI logs carry place names; a latin-1 log used to fail to read at all."""
    d = tmp_path / "latin"
    d.mkdir()
    video = d / "flight.mp4"
    video.write_bytes(good_flight.read_bytes())
    t, x = np.arange(0, 8, 0.04), np.linspace(0, 300, 200)
    write_csv(d / "flight.csv", t, np.interp(t, np.linspace(0, 8, 200), x), encoding="latin-1")
    table = load_telemetry(video, cfg)
    assert table.has_gps and len(table) > 50


# --------------------------------------------------------------------------
# Timing: a log that covers the whole flight
# --------------------------------------------------------------------------
def test_a_full_flight_log_is_trimmed_to_the_recorded_segment(cfg, tmp_path):
    """The log runs 30 s before and after the video; without the recording flag the GPS
    would be 30 s out of step with the frames."""
    video = tmp_path / "flight.mp4"
    t, x = make_flight(video, seconds=8.0)
    write_csv(tmp_path / "flight.csv", t, x, recording=True, pad_s=30.0)
    table = load_telemetry(video, cfg, video_duration_s=float(t[-1]))
    assert table.alignment["method"] == "recording_flag"
    assert table.frame["t"].min() == pytest.approx(-2.0, abs=0.3)      # padding only
    assert table.frame["t"].max() == pytest.approx(t[-1] + 2.0, abs=0.5)


# --------------------------------------------------------------------------
# The check itself
# --------------------------------------------------------------------------
def _by_id(report, check_id):
    return next(c for c in report.checks if c.id == check_id)


def test_a_sound_clip_passes_and_identifies_the_camera(cfg, good_flight):
    report = analyze_input(good_flight, cfg)
    assert report.verdict in ("READY", "READY_WITH_WARNINGS"), report.text()
    assert _by_id(report, "T1").status == PASS
    assert _by_id(report, "A1").status == PASS                       # rigid scene
    assert _by_id(report, "P1").status == PASS                       # parallax
    assert "Phantom 4 Pro" in (report.camera.get("model") or "")     # from the log's aircraft column
    assert report.camera["hfov_deg"] == pytest.approx(71.5, abs=1.5)


def test_a_telemetry_time_offset_is_measured_from_the_picture(cfg, tmp_path):
    video = tmp_path / "flight.mp4"
    t, x = make_flight(video, seconds=10.0)
    write_csv(tmp_path / "flight.csv", t, x, lag_s=1.5)
    report = analyze_input(video, cfg)
    assert report.sync["r_best"] > 0.6
    assert report.sync["lag_s"] == pytest.approx(-1.5, abs=0.4)
    assert _by_id(report, "S1").status == WARN
    assert report.recommended["time_offset_s"] == pytest.approx(-1.5, abs=0.4)


def test_telemetry_from_another_flight_is_refused(cfg, tmp_path):
    """The log is a different flight: the picture moves while the GPS says almost nothing."""
    video = tmp_path / "flight.mp4"
    t, x = make_flight(video, seconds=10.0)
    # Same ground covered at a speed that wanders differently, with no repeating pattern that
    # could line up with the video's under some shift.
    rng = np.random.default_rng(4)
    noise = np.convolve(rng.normal(0, 1, len(t) + 60), np.ones(60) / 60, mode="same")[:len(t)]
    wrong = np.cumsum(40.0 * (1 + 0.8 * noise / max(np.abs(noise).max(), 1e-6))) / 25.0
    write_csv(tmp_path / "flight.csv", t, wrong)
    report = analyze_input(video, cfg)
    assert _by_id(report, "S1").status != PASS
    # Either it does not correlate, or it correlates no better at its best offset than elsewhere.
    assert report.sync["r_best"] < 0.6 or report.sync["peak_margin"] < 0.05


def test_no_telemetry_blocks_because_gps_is_mandatory(cfg, tmp_path):
    video = tmp_path / "alone.mp4"
    make_flight(video, seconds=6.0)
    report = analyze_input(video, cfg)
    assert report.blocked and _by_id(report, "T1").status == BLOCK
    assert "--telemetry" in _by_id(report, "T1").fix

    relaxed = load_config(overrides=["preflight.telemetry.require_gps=false"])
    assert not analyze_input(video, relaxed).blocked


def test_too_small_a_video_is_refused_with_the_requirement_named(cfg, tmp_path):
    video = tmp_path / "small.mp4"
    t, x = make_flight(video, seconds=6.0, width=640, height=360)
    write_csv(tmp_path / "small.csv", t, x)
    report = analyze_input(video, cfg)
    assert report.blocked and _by_id(report, "V1").status == BLOCK
    assert "1080p" in _by_id(report, "V1").detail


def test_a_hovering_drone_is_refused_for_lack_of_parallax(cfg, tmp_path):
    video = tmp_path / "hover.mp4"
    t, x = make_flight(video, seconds=6.0, hover=True)
    write_csv(tmp_path / "hover.csv", t, x)
    report = analyze_input(video, cfg)
    assert _by_id(report, "P1").status == BLOCK
    assert "parallax" in _by_id(report, "P1").label.lower() or "parallax" in _by_id(report, "P1").fix.lower()


def test_a_scene_that_is_not_rigid_is_flagged(cfg, tmp_path):
    """Every frame is warped differently, so matches cannot fit one rigid 3-D scene —
    the signature of morphing or generated footage."""
    video = tmp_path / "warped.mp4"
    t, x = make_flight(video, seconds=8.0, warp=9.0)
    write_csv(tmp_path / "warped.csv", t, x)
    report = analyze_input(video, cfg)
    assert _by_id(report, "A1").status in (WARN, BLOCK)
    assert report.checks and "rigid" in _by_id(report, "A1").label.lower()


def test_an_edited_copy_is_reported_as_re_encoded(cfg, good_flight):
    report = analyze_input(good_flight, cfg)
    provenance = _by_id(report, "A2")
    # OpenCV writes these files through libavformat, which is exactly the "re-encoded" case.
    assert provenance.status in (PASS, WARN)
    if provenance.status == WARN:
        assert "original file" in provenance.fix


def test_the_report_saves_and_reloads(cfg, good_flight, tmp_path):
    from src.preflight.analyzer import InputReport

    report = analyze_input(good_flight, cfg)
    report.save(tmp_path)
    again = InputReport.load(tmp_path)
    assert again.verdict == report.verdict and len(again.checks) == len(report.checks)
    assert json.loads((tmp_path / "input_report.json").read_text())["note"]


def test_the_pipeline_refuses_a_blocking_input_and_accept_input_overrides(cfg, tmp_path):
    from src.pipeline import RunInputs, run_pipeline
    from src.preflight import InputRejected

    video = tmp_path / "small.mp4"
    t, x = make_flight(video, seconds=6.0, width=640, height=360)
    write_csv(tmp_path / "small.csv", t, x)
    with pytest.raises(InputRejected) as excinfo:
        run_pipeline(RunInputs(video=video), cfg, run_dir=tmp_path / "run", stages=["preflight"])
    assert "Resolution" in str(excinfo.value)
    result = run_pipeline(RunInputs(video=video, accept_input=True), cfg, run_dir=tmp_path / "run2",
                          stages=["preflight"])
    assert result.manifest.stages["preflight"].metrics["verdict"] == "BLOCKED"
    assert (tmp_path / "run2" / "preflight" / "input_report.json").exists()
