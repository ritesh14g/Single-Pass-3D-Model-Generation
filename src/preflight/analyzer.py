"""Stage 0 — the input check (PS §1.3 mandatory inputs; spec §4 ingest; §5.6 GPS).

Before any budget is spent, prove that the input can produce every output — or say exactly
why not and how to fix it. Six groups of checks, each PASS / WARN / BLOCK / INFO:

  Format        container and codec decodable start to end; not 360°, not interlaced.
  Video         the PS's 1080p/4K; frame rate; length; compression; sharpness, exposure,
                texture, sky, burned-in overlays, letterboxing (sampled across the clip).
  Telemetry     GPS present (mandatory), sane, dense and covering the video; altitude and
                its datum; flight metadata (attitude, field of view); placeholder values.
  Sync          the video's own motion against the GPS speed: measures any time offset
                between telemetry and video, and catches a log from the wrong flight.
  Authenticity  physical consistency, not a claim of proof: the scene behaves as one rigid
                3-D scene filmed from a moving camera; the file's own metadata (encoder,
                embedded location and date) agrees with the telemetry; re-encoding noted.
  Feasibility   the drone moved enough for parallax; ground sample distance for the <= 1 m
                target; a known camera for an intrinsics prior; projected processing time.

It samples keyframes only (``sampling.py``), so a 10-minute 4K clip is checked in well under
a minute on the box. What it cannot prove, it says: a clean report is "no problems found in
these checks", never "certified correct".
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_event
from src.preflight.checks import BLOCK, INFO, PASS, WARN, Check, band, soft, verdict

log = get_logger(__name__)

AUTHENTICITY_NOTE = ("These checks test physical consistency (rigid 3-D scene, motion matching GPS, metadata "
                     "agreeing with telemetry). They can expose fabricated or mismatched inputs, but no software "
                     "can certify that footage is genuine.")


class InputRejected(RuntimeError):
    """The input failed a blocking check; ``report`` says which and how to fix it."""

    def __init__(self, report: "InputReport"):
        self.report = report
        blocked = [c for c in report.checks if c.status == BLOCK]
        lines = [f"input rejected by the input check ({len(blocked)} blocking problem(s)):"]
        lines += [f"  - {c.label}: {c.detail}" + (f"\n    fix: {c.fix}" if c.fix else "") for c in blocked]
        lines.append("Re-run with --accept-input to process it anyway.")
        super().__init__("\n".join(lines))


@dataclass
class InputReport:
    video: str
    verdict: str
    checks: list[Check]
    probe: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    sync: dict[str, Any] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    recommended: dict[str, Any] = field(default_factory=dict)
    timing_s: dict[str, float] = field(default_factory=dict)
    series: pd.DataFrame = field(default_factory=pd.DataFrame)
    gps_track: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = AUTHENTICITY_NOTE

    @property
    def blocked(self) -> bool:
        return self.verdict == "BLOCKED"

    def to_dict(self) -> dict[str, Any]:
        return {"video": self.video, "verdict": self.verdict, "checks": [c.to_dict() for c in self.checks],
                "probe": self.probe, "telemetry": self.telemetry, "sync": self.sync, "camera": self.camera,
                "recommended": self.recommended, "timing_s": self.timing_s, "note": self.note}

    def save(self, out_dir: Path | str) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "input_report.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default), encoding="utf-8")
        if not self.series.empty:
            self.series.to_parquet(out_dir / "input_series.parquet", index=False)
        if not self.gps_track.empty:
            self.gps_track.to_parquet(out_dir / "input_gps.parquet", index=False)
        return path

    @classmethod
    def load(cls, out_dir: Path | str) -> "InputReport":
        out_dir = Path(out_dir)
        d = json.loads((out_dir / "input_report.json").read_text(encoding="utf-8"))
        series = pd.read_parquet(out_dir / "input_series.parquet") if (out_dir / "input_series.parquet").exists() \
            else pd.DataFrame()
        gps = pd.read_parquet(out_dir / "input_gps.parquet") if (out_dir / "input_gps.parquet").exists() \
            else pd.DataFrame()
        return cls(video=d["video"], verdict=d["verdict"], checks=[Check(**c) for c in d["checks"]],
                   probe=d.get("probe", {}), telemetry=d.get("telemetry", {}), sync=d.get("sync", {}),
                   camera=d.get("camera", {}), recommended=d.get("recommended", {}), timing_s=d.get("timing_s", {}),
                   series=series, gps_track=gps, note=d.get("note", AUTHENTICITY_NOTE))

    def text(self) -> str:
        """The report as plain text, for the terminal."""
        sym = {PASS: "PASS ", WARN: "WARN ", BLOCK: "BLOCK", INFO: "info "}
        lines = [f"Input check: {Path(self.video).name} -> {self.verdict.replace('_', ' ')}"]
        group = None
        for c in self.checks:
            if c.group != group:
                group = c.group
                lines.append(f"\n  {group}")
            lines.append(f"    [{sym.get(c.status, c.status)}] {c.label}: {c.detail}")
            if c.fix and c.status in (WARN, BLOCK):
                lines.append(f"            fix: {c.fix}")
        lines.append(f"\n  {self.note}")
        return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _cfg(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        value = cfg.get_path(f"preflight.{key}")
    except Exception:  # noqa: BLE001
        return default
    if value is None:
        return default
    return value.to_dict() if hasattr(value, "to_dict") else value


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def analyze_input(video: Path | str, cfg: Any, telemetry_paths: list[Path | str] | None = None,
                  srt: Path | str | None = None, csv: Path | str | None = None) -> InputReport:
    from src.ingest.formats import classify_non_video, discover_telemetry, probe_container

    started = time.monotonic()
    video = Path(video)
    checks: list[Check] = []
    timing: dict[str, float] = {}
    report = InputReport(video=str(video), verdict="BLOCKED", checks=checks)

    if not video.is_file():
        checks.append(Check("F0", "Format", "Input file", BLOCK, None, f"{video} does not exist",
                            "Check the path."))
        report.verdict = verdict(checks)
        return report

    t0 = time.monotonic()
    probe = probe_container(video)
    report.probe = probe.to_dict()
    timing["probe"] = time.monotonic() - t0
    if probe.error or probe.video is None or not probe.video.width:
        what, fix = classify_non_video(video)
        reason = probe.error or "no video stream"
        checks.append(Check("F1", "Format", "Video container", BLOCK, None,
                            f"{video.name} is {what}; FFmpeg: {reason}", fix))
        report.verdict = verdict(checks)
        report.timing_s = {k: round(v, 2) for k, v in timing.items()}
        return report

    v = probe.video
    duration = float(probe.duration_s or (v.frames / v.fps if v.fps else 0) or 0)
    # FFmpeg opens a still image as a one-frame "video"; say what the file is instead of
    # reporting a decode failure further down.
    if (probe.format_name or "").split(",")[0] in ("image2", "jpeg_pipe", "png_pipe", "tiff_pipe", "webp_pipe",
                                                   "bmp_pipe", "gif") or v.frames == 1:
        what, fix = classify_non_video(video)
        checks.append(Check("F1", "Format", "Video container", BLOCK, probe.format_name,
                            f"{video.name} is {what} ({v.width}x{v.height} {v.codec}), not a video", fix))
        report.verdict = verdict(checks)
        report.timing_s = {k: round(val, 2) for k, val in timing.items()}
        return report
    checks += _format_checks(probe, v, video)

    # -- frames ---------------------------------------------------------------
    from src.preflight.sampling import sample_frames

    scfg = _cfg(cfg, "sampling", {})
    workers = int(scfg.get("workers", 3))
    samples = sample_frames(video, duration, int(scfg.get("max_samples", 180)), int(scfg.get("width", 960)),
                            float(scfg.get("min_gap_s", 1.0)), workers=workers)
    timing["decode"] = samples.seconds
    checks.append(_decode_check(samples, duration))
    checks += _video_checks(v, probe, duration, _cfg(cfg, "video", {}))

    t0 = time.monotonic()
    series, vision_checks, pairs_info = _vision(samples, _cfg(cfg, "quality", {}), _cfg(cfg, "authenticity", {}),
                                                 workers, int(scfg.get("feature_width", 640)))
    checks += vision_checks
    timing["vision"] = time.monotonic() - t0

    # -- telemetry --------------------------------------------------------------
    t0 = time.monotonic()
    from src.ingest.telemetry import load_telemetry

    candidates = discover_telemetry(video, [p for p in [srt, csv, *(telemetry_paths or [])] if p], probe=probe,
                                    scan_folder=bool(cfg.get_path("ingest.telemetry.scan_folder", True)))
    telemetry = load_telemetry(video, cfg, srt_path=srt, csv_path=csv, telemetry_paths=telemetry_paths,
                               probe=probe, video_duration_s=duration)
    timing["telemetry"] = time.monotonic() - t0
    report.telemetry = {"summary": telemetry.summary(), "alignment": telemetry.alignment,
                        "candidates": [c.to_dict() for c in candidates]}
    tcfg = _cfg(cfg, "telemetry", {})
    tel_checks, gps = _telemetry_checks(telemetry, candidates, duration, tcfg)
    checks += tel_checks
    if gps is not None:
        report.gps_track = gps

    # -- camera -----------------------------------------------------------------
    report.camera = identify_camera(probe, candidates, telemetry, v.width)

    # -- sync ---------------------------------------------------------------------
    t0 = time.monotonic()
    clock_offset = _clock_offset(probe, telemetry)
    sync, sync_checks, series = _sync(series, telemetry, gps, _cfg(cfg, "sync", {}), duration, clock_offset)
    report.sync = sync
    checks += sync_checks
    timing["sync"] = time.monotonic() - t0
    if sync.get("apply_offset_s") is not None and bool(_cfg(cfg, "apply_time_offset", True)):
        report.recommended["time_offset_s"] = sync["apply_offset_s"]

    # -- authenticity & feasibility ---------------------------------------------------
    checks += _authenticity_checks(probe, telemetry, gps, pairs_info, sync, _cfg(cfg, "authenticity", {}))
    checks += _feasibility_checks(telemetry, gps, report.camera, v, duration, _cfg(cfg, "feasibility", {}), cfg)

    report.series = series
    report.verdict = verdict(checks)
    timing["total"] = time.monotonic() - started
    report.timing_s = {k: round(val, 2) for k, val in timing.items()}
    log_event(log, 20, f"input check: {report.verdict} in {timing['total']:.1f}s", verdict=report.verdict,
              blocks=sum(c.status == BLOCK for c in checks), warns=sum(c.status == WARN for c in checks))
    return report


# --------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------
def _format_checks(probe, v, video: Path) -> list[Check]:
    out = [Check("F1", "Format", "Video container", PASS, probe.format_name,
                 f"{probe.format_name} ({video.suffix or 'no extension'}), {v.codec} {v.width}x{v.height}"
                 + (" — extension does not match the content; read by content" if _ext_mismatch(video, probe) else ""))]
    if v.spherical:
        out.append(Check("F2", "Format", "Projection", BLOCK, "360/spherical",
                         "360° / equirectangular video: the pipeline needs a normal (pinhole) camera view.",
                         "Export a flat, reframed view from the 360 editor at 1080p or more."))
    fo = (v.field_order or "").lower()
    if fo and fo not in ("progressive", "unknown", "none"):
        out.append(Check("F3", "Format", "Scan type", WARN, fo, f"interlaced video ({fo}): combing on moving frames",
                         "Deinterlace: ffmpeg -i in -vf bwdif -c:v libx264 -crf 16 out.mp4"))
    trc = (v.color_transfer or "").lower()
    if trc in ("arib-std-b67", "smpte2084"):
        out.append(Check("F4", "Format", "Colour", WARN, trc, f"HDR video ({trc}); colours are processed as SDR",
                         "Record in SDR, or tone-map to Rec.709 before processing."))
    elif v.bit_depth and v.bit_depth > 8:
        out.append(Check("F4", "Format", "Colour", INFO, f"{v.bit_depth}-bit",
                         f"{v.bit_depth}-bit video (often a flat/log profile): features still match, textures look flat"))
    if v.rotation_deg:
        out.append(Check("F5", "Format", "Rotation", INFO, v.rotation_deg,
                         f"display rotation {v.rotation_deg:.0f}° in the metadata"))
    if v.fps and v.fps_guessed and abs(v.fps - v.fps_guessed) / v.fps > 0.02:
        out.append(Check("F6", "Format", "Frame timing", INFO, "variable",
                         f"variable frame rate (average {v.fps:.2f}, nominal {v.fps_guessed:.2f}); frame times are used, "
                         "not frame counts"))
    return out


def _ext_mismatch(video: Path, probe) -> bool:
    ext = video.suffix.lower()
    fmt = (probe.format_name or "").lower()
    if "mpegts" in fmt:
        return ext not in (".ts", ".mts", ".m2ts", ".trp")
    if "mov" in fmt or "mp4" in fmt:
        return ext not in (".mp4", ".mov", ".m4v", ".3gp", ".lrv", ".insv")
    return False


def _decode_check(samples, duration: float) -> Check:
    reached = samples.last_decoded_s / duration if duration else 0.0
    n = len(samples.samples)
    detail = (f"{n} frames decoded across the clip ({samples.method}"
              + (f", keyframes every {samples.keyframe_interval_s:.2f} s" if samples.keyframe_interval_s else "")
              + f") in {samples.seconds:.1f} s; last at {samples.last_decoded_s:.1f} of {duration:.1f} s")
    if n < 2:
        return Check("F7", "Format", "Decoding", BLOCK, n, "no frames could be decoded: " + detail,
                     "The file is damaged or uses a codec this FFmpeg build lacks. Re-export or convert: "
                     "ffmpeg -i <input> -c:v libx264 -crf 16 <output>.mp4")
    if reached < 0.5:
        return Check("F7", "Format", "Decoding", BLOCK, round(reached, 2), "decodes only part of the clip: " + detail,
                     "The file is truncated or damaged; copy it again from the drone's card.")
    if samples.decode_errors or reached < 0.9:
        errs = ", ".join(f"{t:.1f}s" for t in samples.decode_errors[:5])
        return Check("F7", "Format", "Decoding", WARN, len(samples.decode_errors),
                     detail + (f"; decode errors at {errs}" if errs else "; the end of the clip was not reached"),
                     "Damaged stretches are skipped; re-copy the file from the card if possible.")
    return Check("F7", "Format", "Decoding", PASS, n, detail)


# --------------------------------------------------------------------------
# Video requirements
# --------------------------------------------------------------------------
def _video_checks(v, probe, duration: float, c: dict) -> list[Check]:
    out = []
    short = min(v.width, v.height)
    st = band(short, float(c.get("good_short_side_px", 1080)), float(c.get("min_short_side_px", 720)))
    out.append(Check("V1", "Video", "Resolution", st, f"{v.width}x{v.height}",
                     f"{v.width}x{v.height} (PS: 1080p / 4K)" + ("" if st == PASS else
                     "; below 1080p every metre on the ground gets fewer pixels, so detail and accuracy drop"),
                     "" if st == PASS else "Record at 1080p or 4K."))
    st = band(v.fps, float(c.get("good_fps", 24)), float(c.get("min_fps", 10)))
    out.append(Check("V2", "Video", "Frame rate", st, round(v.fps, 2), f"{v.fps:.2f} fps",
                     "" if st == PASS else "Record at 24 fps or more; low rates leave too little overlap in fast flight."))
    lo, hi = float(c.get("min_duration_s", 5)), float(c.get("max_duration_s", 600))
    if duration < lo:
        st, detail = BLOCK, f"{duration:.1f} s: too short to reconstruct anything (needs >= {lo:.0f} s of flight)"
    elif duration > hi:
        st, detail = WARN, f"{duration / 60:.1f} min: longer than the PS's 10-minute case; the 15-minute budget will degrade quality"
    else:
        st, detail = PASS, f"{duration:.1f} s"
    out.append(Check("V3", "Video", "Duration", st, round(duration, 1), detail,
                     "" if st == PASS else "Trim to the survey part of the flight."))
    rate = probe.bit_rate or v.bit_rate
    if rate and v.width and v.height and v.fps:
        bpp = rate / (v.width * v.height * v.fps)
        st = soft(band(bpp, float(c.get("good_bits_per_pixel", 0.05)), float(c.get("min_bits_per_pixel", 0.02))))
        out.append(Check("V4", "Video", "Compression", st, round(bpp, 3),
                         f"{rate / 1e6:.1f} Mbit/s = {bpp:.3f} bits per pixel per frame"
                         + ("" if st == PASS else "; heavy compression smears fine texture"),
                         "" if st == PASS else "Use the drone's original file, not a re-encoded or streamed copy."))
    return out


def _vision(samples, q: dict, a: dict, workers: int = 3,
            feature_width: int = 640) -> tuple[pd.DataFrame, list[Check], dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    from src.preflight.vision import PairAnalyzer, frame_stats, letterbox, static_overlay

    out: list[Check] = []
    frames = samples.samples
    if len(frames) < 2:
        return pd.DataFrame(), out, {}
    analyzer = PairAnalyzer(width=feature_width)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        feats = list(pool.map(lambda s: analyzer.features(s.bgr), frames))
        stats = list(pool.map(lambda sf: frame_stats(sf[0].t, sf[0].bgr, len(sf[1][0])), zip(frames, feats)))
        size = analyzer.feature_size(frames[0].bgr)
        pairs = list(pool.map(lambda i: analyzer.pair(frames[i].t, feats[i], frames[i + 1].t, feats[i + 1], size),
                              range(len(frames) - 1)))
    h, w = frames[0].bgr.shape[:2]
    series = pd.DataFrame([s.__dict__ for s in stats])
    pm = pd.DataFrame([p.__dict__ for p in pairs])
    if not pm.empty:
        pm["t"] = (pm["t0"] + pm["t1"]) / 2
        pm["dt"] = pm["t1"] - pm["t0"]
        pm["shift_px_s"] = pm["shift_px"] / pm["dt"]
        pm["rot_deg_s"] = pm["rotation_deg"] / pm["dt"]
        series = series.merge(pm[["t0", "matches", "rigid_ratio", "shift_px_s", "rot_deg_s"]], left_on="t",
                              right_on="t0", how="left").drop(columns=["t0"])
    series["width_px"] = size[0]

    blur_floor = float(q.get("blur_sharpness_floor", 25.0))
    blurry = float((series["sharpness"] < blur_floor).mean())
    st = PASS if blurry <= 0.1 else WARN
    out.append(Check("V5", "Video", "Sharpness", st, round(blurry, 2),
                     f"{blurry:.0%} of sampled frames are blurred (Laplacian variance < {blur_floor:g} at {w} px)",
                     "" if st == PASS else "Fly slower or use a faster shutter; blurred frames are dropped by Stage 1."))
    dark = float((series["brightness"] < float(q.get("dark_brightness", 35))).mean())
    over = float((series["bright_frac"] > float(q.get("overexposed_share", 0.25))).mean())
    st = PASS if max(dark, over) <= float(q.get("max_bad_exposure_share", 0.3)) else WARN
    out.append(Check("V6", "Video", "Exposure", st, {"dark": round(dark, 2), "overexposed": round(over, 2)},
                     f"{dark:.0%} of frames too dark, {over:.0%} clipped highlights",
                     "" if st == PASS else "Stage 2 corrects exposure, but night or blown-out footage loses detail."))
    kp = float(series["keypoints"].median())
    st = PASS if kp >= float(q.get("min_keypoints", 150)) else WARN
    out.append(Check("V7", "Video", "Texture", st, int(kp),
                     f"median {kp:.0f} features per frame",
                     "" if st == PASS else "Little texture (water, snow, sky, fog): matching will be sparse there."))
    sky = float(series["sky_frac"].median())
    st = PASS if sky <= float(q.get("max_sky_share", 0.3)) else WARN
    out.append(Check("V8", "Video", "Sky in view", st, round(sky, 2),
                     f"sky fills {sky:.0%} of a typical frame",
                     "" if st == PASS else "Tilt the camera further down (nadir or 45-70° down) for mapping."))
    frac, box = static_overlay([s.bgr for s in frames[:: max(1, len(frames) // 40)]])
    st = WARN if frac >= float(q.get("overlay_min_share", 0.002)) else PASS
    out.append(Check("V9", "Video", "Burned-in overlay", st, round(frac, 4),
                     "no static text or logo detected" if st == PASS else
                     f"static edges (OSD text, logo or watermark) cover {frac:.1%} of the frame, box {box} at 480 px",
                     "" if st == PASS else "Record without the on-screen display; overlays add false features."))
    lb = letterbox([s.bgr for s in frames[:: max(1, len(frames) // 20)]])
    worst = max(lb.values()) if lb else 0.0
    st = WARN if worst >= float(q.get("letterbox_min_share", 0.02)) else PASS
    out.append(Check("V10", "Video", "Black borders", st, {k: round(v, 3) for k, v in lb.items()},
                     "none" if st == PASS else f"black borders up to {worst:.1%} of the frame (letterbox / pillarbox)",
                     "" if st == PASS else "Use the original frame; borders shrink the usable image."))
    rigid = pm["rigid_ratio"].dropna() if not pm.empty else pd.Series(dtype=float)
    info = {"pairs": len(pm), "rigid_median": float(rigid.median()) if len(rigid) else None,
            "rigid_low_share": float((rigid < float(a.get("rigid_min", 0.35))).mean()) if len(rigid) else None,
            "matches_median": float(pm["matches"].median()) if not pm.empty else 0.0}
    return series, out, info


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
def _camera_pitch(frame: pd.DataFrame, gps: pd.DataFrame) -> tuple[float | None, str]:
    """Camera angle below horizontal, from the most direct source the telemetry has.

    ISR (MISB) telemetry carries the *airframe* pitch in the canonical column and the camera
    angle separately, so reading the canonical column alone would call a downward-looking
    sensor "level".
    """
    cols = set(frame.columns)
    if {"slant_range_m", "frame_center_alt", "alt_gps"} <= cols:
        drop = (frame["alt_gps"] - frame["frame_center_alt"]).to_numpy(dtype=float)
        slant = frame["slant_range_m"].to_numpy(dtype=float)
        ok = np.isfinite(drop) & np.isfinite(slant) & (slant > 1)
        if ok.sum() > 3:
            ratio = np.clip(drop[ok] / slant[ok], -1, 1)
            return float(-np.degrees(np.arcsin(np.median(ratio)))), "from sensor altitude and slant range"
    if "sensor_rel_el_deg" in cols and frame["sensor_rel_el_deg"].notna().any():
        platform = float(frame["pitch"].median()) if frame["pitch"].notna().any() else 0.0
        return float(platform + frame["sensor_rel_el_deg"].median()), "sensor elevation plus airframe pitch"
    pitch = gps["pitch"].dropna()
    return (float(pitch.median()), "gimbal pitch") if len(pitch) else (None, "")


def _telemetry_checks(tel, candidates, duration: float, c: dict) -> tuple[list[Check], pd.DataFrame | None]:
    from src.ingest.telemetry import enu_from_geodetic

    out: list[Check] = []
    found = ", ".join(f"{x.kind} ({x.found_by}{': ' + Path(x.path).name if x.path else ''})" for x in candidates) or "none"
    unsupported = [x for x in candidates if not x.supported]
    if tel.is_empty or not tel.has_gps:
        require = bool(c.get("require_gps", True))
        detail = f"no usable GPS. Sources looked at: {found}." + (
            " " + "; ".join(f"{Path(x.path).name if x.path else 'in video'}: {x.note}" for x in unsupported)
            if unsupported else "") + (" " + "; ".join(tel.notes[:3]) if tel.notes else "")
        out.append(Check("T1", "Telemetry", "GPS (mandatory)", BLOCK if require else WARN, False, detail,
                         "Supply the flight's telemetry: the DJI .SRT (turn on 'Video Caption'), the flight log "
                         "(AirData / Flight Reader CSV, DJI .txt), a GPX/KML track, or an autopilot log, with "
                         "--telemetry <file>. Without GPS the model is scale-free and not georeferenced."))
        return out, None
    out.append(Check("T1", "Telemetry", "GPS (mandatory)", PASS, tel.source,
                     f"{len(tel)} fixes from {tel.source}. Sources found: {found}"))
    for x in unsupported:
        out.append(Check("T1b", "Telemetry", "Unused telemetry", INFO, x.kind,
                         f"{Path(x.path).name if x.path else 'stream in the video'}: {x.note}"))

    f = tel.frame.dropna(subset=["lat", "lon"]).sort_values("t")
    t = f["t"].to_numpy(dtype=float)
    lat, lon = f["lat"].to_numpy(dtype=float), f["lon"].to_numpy(dtype=float)
    bad = (np.abs(lat) > 90) | (np.abs(lon) > 180)
    if bad.any():
        out.append(Check("T2", "Telemetry", "Coordinates", BLOCK, int(bad.sum()),
                         f"{bad.sum()} fixes outside valid latitude/longitude ranges",
                         "Check the log's column mapping (lat/lon swapped or in a projected CRS)."))
        return out, None
    alt = f["alt_gps"].to_numpy(dtype=float)
    enu, _ = enu_from_geodetic(lat, lon, np.nan_to_num(alt, nan=0.0))
    gps = pd.DataFrame({"t": t, "lat": lat, "lon": lon, "e": enu[:, 0], "n": enu[:, 1],
                        "alt_gps": alt, "alt_baro": f["alt_baro"].to_numpy(dtype=float),
                        "yaw": f["yaw"].to_numpy(dtype=float), "pitch": f["pitch"].to_numpy(dtype=float)})

    dts = np.diff(t)
    rate = 1.0 / float(np.median(dts[dts > 0])) if (dts > 0).any() else 0.0
    st = band(rate, float(c.get("good_rate_hz", 5)), float(c.get("min_rate_hz", 0.9)))
    whole = float(np.mean(np.isclose(t - np.floor(t), t[0] - np.floor(t[0]), atol=1e-3))) if len(t) > 5 else 0.0
    detail = f"{rate:.1f} fixes/s"
    if rate < 1.5 and whole >= float(c.get("whole_second_share", 0.9)):
        detail += "; timestamps have whole-second resolution (±0.5 s timing on each fix)"
        st = soft(WARN if st == PASS else st)
    out.append(Check("T3", "Telemetry", "Sample rate", soft(st), round(rate, 2), detail,
                     "" if st == PASS else "Per-frame telemetry (SRT, KLV at frame rate) or a 5-10 Hz log is best."))

    lo, hi = max(0.0, t.min()), min(duration, t.max())
    coverage = max(0.0, hi - lo) / duration if duration else 0.0
    st = band(coverage, float(c.get("good_coverage", 0.95)), float(c.get("min_coverage", 0.7)))
    out.append(Check("T4", "Telemetry", "Covers the video", st, round(coverage, 3),
                     f"GPS spans video time {lo:.1f}-{hi:.1f} s of {duration:.1f} s ({coverage:.0%}); "
                     f"alignment: {tel.alignment.get('note', tel.alignment.get('method', 'unknown'))}",
                     "" if st == PASS else "The log does not cover the recording: wrong log, or its clock is off. "
                     "Supply the log of this flight, or set ingest.telemetry.time_offset_s."))

    inside = (t >= -1) & (t <= duration + 1)
    gaps = np.diff(t[inside]) if inside.sum() > 1 else np.array([0.0])
    # Speed over a window, not between neighbouring fixes: a 10 Hz log whose coordinates are
    # rounded shows huge apparent speeds over 0.1 s that are only its own quantisation.
    window = float(c.get("speed_window_s", 1.0))
    ahead = np.searchsorted(t, t + window)
    ok = ahead < len(t)
    speed = np.zeros(int(ok.sum()))
    if ok.any():
        j = ahead[ok]
        dt_w = np.maximum(t[j] - t[ok], 1e-3)
        speed = np.hypot(enu[j, 0] - enu[ok, 0], enu[j, 1] - enu[ok, 1]) / dt_w
    jumps = int((speed > float(c.get("max_speed_mps", 70))).sum())
    st = PASS if gaps.max() <= float(c.get("max_gap_s", 3.0)) and jumps == 0 else WARN
    out.append(Check("T5", "Telemetry", "Gaps and jumps", st, {"max_gap_s": round(float(gaps.max()), 2), "jumps": jumps},
                     f"largest gap {gaps.max():.1f} s; fastest {speed.max() if len(speed) else 0:.0f} m/s over {window:.0f} s "
                     f"windows, {jumps} impossible (> {c.get('max_speed_mps', 70)} m/s)",
                     "" if st == PASS else "Stage 2 rejects outliers and interpolates short gaps; long gaps lose "
                     "georeferencing there."))

    has_abs, has_rel = np.isfinite(alt).any(), np.isfinite(gps["alt_baro"]).any()
    rel_equal = has_abs and has_rel and np.nanmedian(np.abs(alt - gps["alt_baro"].to_numpy())) < 1.0
    if not has_abs and not has_rel:
        st, detail, fix = WARN, "no altitude at all: heights come from the reconstruction only", \
            "Supply a log with altitude (flight metadata is mandatory in the PS)."
    elif rel_equal or (not has_abs and has_rel):
        st, detail, fix = WARN, ("altitude is height above the takeoff point only (no sea-level or ellipsoidal "
                                 "altitude): model heights will be offset by the takeoff elevation"), \
            "Use a log with absolute altitude (DJI SRT abs_alt, KLV, autopilot GPS), or supply GCPs."
    else:
        span = float(np.nanmax(alt) - np.nanmin(alt))
        st, detail, fix = PASS, f"absolute altitude {np.nanmin(alt):.1f}-{np.nanmax(alt):.1f} m (span {span:.1f} m)" + \
            (", plus height above takeoff" if has_rel else ""), ""
    out.append(Check("T6", "Telemetry", "Altitude", st, bool(has_abs and not rel_equal), detail, fix))

    p, source = _camera_pitch(f, gps)
    if p is not None:
        view = "nadir (straight down)" if p <= -70 else "oblique" if p <= -20 else "shallow (horizon likely in view)"
        st = PASS if p <= -20 else WARN
        out.append(Check("T7", "Telemetry", "Camera pitch", st, round(p, 1), f"camera {p:.0f}° from horizontal "
                         f"({source}): {view}",
                         "" if st == PASS else "Point the camera 45-90° down for mapping; a view along the horizon "
                         "gives little parallax on the ground and much sky."))
    else:
        out.append(Check("T7", "Telemetry", "Camera pitch", INFO, None,
                         "no camera attitude in the telemetry; orientation comes from the images alone"))

    # A gimbal held at one pitch is normal; attitude that is exactly zero for the whole flight
    # (a camera "looking at the horizon" while mapping) or a ground height that never changes
    # is how simplified or synthesised metadata looks.
    placeholders = []
    zero = [col for col in ("roll", "pitch") if f[col].notna().sum() > 20 and (f[col].dropna() == 0).all()]
    if len(zero) == 2:
        placeholders.append("camera roll and pitch are exactly 0 for the whole flight")
    for col in f.columns:
        if col.startswith("frame_center_alt") and f[col].notna().sum() > 20 and np.ptp(f[col].dropna()) == 0:
            placeholders.append(f"ground elevation ({col}) constant at {f[col].dropna().iloc[0]:g} m")
    if placeholders:
        out.append(Check("T8", "Telemetry", "Placeholder values", WARN, placeholders,
                         "; ".join(placeholders) + ": usually simplified or synthesised metadata",
                         "Treat these fields as unknown; prefer the drone's original log."))
    rtk = bool(tel.has_rtk)
    out.append(Check("T9", "Telemetry", "RTK / PPK", INFO, rtk,
                     "RTK/PPK corrections present: GPS weight tightened" if rtk else
                     "no RTK/PPK: standard GPS (a few metres absolute)"))
    return out, gps


# --------------------------------------------------------------------------
# Sync: video motion vs GPS speed
# --------------------------------------------------------------------------
def _speed_at(gps: pd.DataFrame, times: np.ndarray, half_window: float = 0.5) -> np.ndarray:
    t = gps["t"].to_numpy()
    e, n = gps["e"].to_numpy(), gps["n"].to_numpy()
    a, b = times - half_window, times + half_window
    de = np.interp(b, t, e, left=np.nan, right=np.nan) - np.interp(a, t, e, left=np.nan, right=np.nan)
    dn = np.interp(b, t, n, left=np.nan, right=np.nan) - np.interp(a, t, n, left=np.nan, right=np.nan)
    return np.hypot(de, dn) / (2 * half_window)


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Correlation of two speed profiles, with extremes clipped.

    One bad frame pair (a cut, a blurred frame, take-off vibration) can be ten times any real
    motion and would otherwise decide the answer on its own; clipping to the 5th-95th
    percentile keeps the shape of the profile without letting an outlier dominate.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return float("nan"), int(ok.sum())
    a, b = _clip(x[ok]), _clip(y[ok])
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(ok.sum())


def _clip(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [5, 95])
    return np.clip(values, lo, hi)


def _clock_offset(probe, tel) -> float | None:
    """Offset between the telemetry clock and the video's own start time, in video seconds.

    An independent second opinion on the measured lag: the camera stamps the file with the
    moment recording began, and the log stamps every row with UTC. Cameras often write local
    time labelled as UTC, so whole time-zone steps are removed first.
    """
    from src.ingest.telemetry_formats import video_start_utc

    if tel.is_empty or "wall_time" not in tel.frame:
        return None
    start = video_start_utc(probe.tags)
    wall = pd.to_datetime(tel.frame["wall_time"], utc=True, errors="coerce")
    if start is None or not wall.notna().any():
        return None
    t = tel.frame["t"].to_numpy(dtype=float)
    ok = wall.notna().to_numpy()
    # Telemetry time (its own clock) at the moment the video says it started.
    at_start = float(np.interp(0.0, t[ok], (wall[ok] - wall[ok].min()).dt.total_seconds().to_numpy()))
    diff = (start - wall[ok].min()).total_seconds() - at_start
    zone = round(diff / 900.0) * 900.0
    residual = diff - zone
    return float(residual) if abs(residual) < 120 else None


def _sync(series: pd.DataFrame, tel, gps: pd.DataFrame | None, c: dict, duration: float,
          clock_offset_s: float | None = None):
    out: list[Check] = []
    info: dict[str, Any] = {"method": "video motion vs GPS ground speed"}
    if gps is None or series.empty or "shift_px_s" not in series:
        out.append(Check("S1", "Sync", "Telemetry-video sync", INFO, None, "not tested (no GPS or no motion samples)"))
        return info, out, series
    tm = series["t"].to_numpy(dtype=float)
    video_speed = series["shift_px_s"].to_numpy(dtype=float).copy()
    # Pairs with few matches give meaningless motion (a cut, a blurred frame, take-off vibration).
    if "matches" in series:
        video_speed[series["matches"].to_numpy(dtype=float) < float(c.get("min_pair_matches", 30))] = np.nan
    series["gps_speed"] = _speed_at(gps, tm)
    # Searching further than half the clip is meaningless: with little overlap left, any two
    # speed profiles can be made to agree somewhere.
    search = min(float(c.get("max_search_s", 30)), max(duration / 2.0, 2.0))
    step = float(c.get("step_s", 0.1))
    lags = np.arange(-search, search + step / 2, step)
    r = np.array([_corr(video_speed, _speed_at(gps, tm + lag))[0] for lag in lags])
    moving = np.nanstd(series["gps_speed"]) / max(np.nanmean(series["gps_speed"]), 1e-6)
    video_cv = np.nanstd(video_speed) / max(np.nanmean(video_speed), 1e-6)
    r0, n0 = _corr(video_speed, series["gps_speed"].to_numpy())
    info.update(r_at_zero=None if math.isnan(r0) else round(r0, 3), points=n0, gps_speed_cv=round(float(moving), 3),
                video_motion_cv=round(float(video_cv), 3))
    # Where GPS says hovering but the picture slides (or the reverse), the log is not this video's.
    gs = series["gps_speed"].to_numpy()
    vs_norm = video_speed / max(np.nanmedian(video_speed[np.isfinite(video_speed)]) if np.isfinite(video_speed).any() else 1, 1e-6)
    still_gps = gs < float(c.get("hover_speed_mps", 0.5))
    moving_img = vs_norm > 0.5
    contradiction = float(np.nanmean(still_gps & moving_img)) if np.isfinite(gs).any() else 0.0
    info["contradiction_share"] = round(contradiction, 3)
    if np.all(np.isnan(r)) or n0 < int(c.get("min_points", 10)):
        out.append(Check("S1", "Sync", "Telemetry-video sync", INFO, None,
                         f"not enough overlapping samples to test ({n0})"))
        return info, out, series
    best = int(np.nanargmax(r))
    lag, rb = float(lags[best]), float(r[best])
    # A real match is a *peak*, not just a high number: an unrelated track can correlate well
    # at some offset by chance, but then it correlates about as well far away from it too.
    far = r[np.abs(lags - lag) > max(3.0, 3 * step)]
    far = far[np.isfinite(far)]
    sharp = rb - float(np.percentile(far, 90)) if far.size else 0.0
    info.update(lag_s=round(lag, 2), r_best=round(rb, 3), peak_margin=round(float(sharp), 3))
    if clock_offset_s is not None:
        info["clock_offset_s"] = round(clock_offset_s, 2)
        info["clock_agrees"] = bool(abs(clock_offset_s - lag) <= float(c.get("clock_agreement_s", 2.0)))
    series["gps_speed_shifted"] = _speed_at(gps, tm + lag)
    min_cv = float(c.get("min_speed_cv", 0.15))
    good, low = float(c.get("good_corr", 0.6)), float(c.get("min_corr", 0.3))
    if moving < min_cv or video_cv < min_cv:
        st, detail = INFO, (f"speed barely varies (GPS CV {moving:.2f}, video CV {video_cv:.2f}), so timing cannot be "
                            f"measured from speed; best match {lag:+.1f} s (r={rb:.2f}) is not reliable")
        fix = ""
    elif rb >= good and sharp >= float(c.get("min_peak_margin", 0.05)):
        ok_lag, max_auto = float(c.get("ok_lag_s", 0.5)), float(c.get("max_auto_lag_s", 5.0))
        if abs(lag) <= ok_lag:
            st, detail, fix = PASS, f"video motion matches GPS speed (r={rb:.2f}) at {lag:+.1f} s", ""
        elif abs(lag) <= max_auto:
            st = WARN
            detail = (f"telemetry is {abs(lag):.1f} s {'behind' if lag > 0 else 'ahead of'} the video "
                      f"(motion vs GPS r={rb:.2f} at {lag:+.1f} s, {r0:.2f} at 0)")
            fix = "Corrected automatically (preflight.apply_time_offset); Stage 4 refines it further."
            info["apply_offset_s"] = round(lag, 2)
        elif info.get("clock_agrees"):
            # Two independent measurements agree: the picture and the two clocks. A large offset
            # is then a fact about the log (a late recording flag, a trimmed clip), not a mismatch.
            st = WARN
            detail = (f"telemetry is {abs(lag):.1f} s {'behind' if lag > 0 else 'ahead of'} the video (r={rb:.2f} at "
                      f"{lag:+.1f} s), which matches the {clock_offset_s:+.1f} s between the video's recording time "
                      "and the log's clock")
            fix = "Corrected automatically; the log's recording marker is late or the clip was trimmed."
            info["apply_offset_s"] = round(lag, 2)
        else:
            st = BLOCK
            detail = (f"telemetry and video are {abs(lag):.1f} s apart (r={rb:.2f} at {lag:+.1f} s): too far to be "
                      "clock jitter — likely the wrong recording segment or a trimmed video"
                      + (f", and the clocks disagree with it ({clock_offset_s:+.1f} s)" if clock_offset_s is not None
                         else ""))
            fix = f"Check the log belongs to this clip; if it does, set ingest.telemetry.time_offset_s={lag:.1f}."
            info["apply_offset_s"] = round(lag, 2)
    elif rb >= good and sharp < float(c.get("min_peak_margin", 0.05)):
        st = WARN
        detail = (f"video motion and GPS speed correlate (r={rb:.2f} at {lag:+.1f} s) but almost as well at other "
                  f"offsets (margin {sharp:.2f}), so the timing cannot be pinned down from motion alone")
        fix = "Stage 4 estimates the offset from the reconstruction as well."
    elif rb < low and contradiction > float(c.get("max_contradiction", 0.2)):
        st = BLOCK
        detail = (f"video motion does not match the GPS at any offset within ±{search:.0f} s (best r={rb:.2f}); the "
                  f"GPS says hovering while the picture moves in {contradiction:.0%} of samples")
        fix = "The telemetry is probably from another flight or another part of this one. Supply the right log."
    elif rb < low:
        st = WARN
        detail = f"weak agreement between video motion and GPS speed (best r={rb:.2f} at {lag:+.1f} s)"
        fix = "Check that the log belongs to this clip."
    else:
        st = WARN if abs(lag) > float(c.get("ok_lag_s", 0.5)) else PASS
        detail = f"moderate agreement (r={rb:.2f}) at {lag:+.1f} s"
        fix = "" if st == PASS else "Stage 4 estimates the offset from the reconstruction as well."
    out.append(Check("S1", "Sync", "Telemetry-video sync", st, {"lag_s": round(lag, 2), "r": round(rb, 3)}, detail, fix))
    return info, out, series


# --------------------------------------------------------------------------
# Authenticity
# --------------------------------------------------------------------------
def _iso6709(value: str) -> tuple[float, float] | None:
    import re

    m = re.match(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)", value or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def _authenticity_checks(probe, tel, gps, pairs_info: dict, sync: dict, a: dict) -> list[Check]:
    from src.ingest.telemetry_formats import video_start_utc

    out: list[Check] = []
    rigid = pairs_info.get("rigid_median")
    if rigid is None:
        out.append(Check("A1", "Authenticity", "Rigid 3-D scene", INFO, None, "too few matched frames to test"))
    else:
        good, low = float(a.get("rigid_good", 0.6)), float(a.get("rigid_min", 0.35))
        st = PASS if rigid >= good else WARN if rigid >= low else BLOCK
        detail = (f"{rigid:.0%} of feature matches between consecutive samples fit one rigid scene seen from two "
                  f"positions (median over {pairs_info['pairs']} pairs; {pairs_info['rigid_low_share']:.0%} of pairs "
                  f"below {low:.0%})")
        fix = "" if st == PASS else ("Much of the image moves inconsistently: water, wind-blown vegetation, crowds - or "
                                     "footage that is not a real camera moving through a real scene (morphing, "
                                     "generated or heavily stabilised/warped video).")
        out.append(Check("A1", "Authenticity", "Rigid 3-D scene", st, round(rigid, 3), detail, fix))

    tags = dict(probe.tags)
    vtags = dict((probe.video.tags if probe.video else {}) or {})
    blob = " ".join(f"{k}={v}" for k, v in {**tags, **vtags}.items())
    maker = next((v for k, v in {**tags, **vtags}.items() if k.lower() in ("make", "com.apple.quicktime.make",
                                                                            "handler_name", "encoder", "model")), "")
    reencoders = [s for s in a.get("reencoders", []) if s.lower() in blob.lower()]
    if reencoders:
        out.append(Check("A2", "Authenticity", "File provenance", WARN, reencoders,
                         f"written by {', '.join(reencoders)} ({blob[:160]}): an edited or re-encoded copy, not the "
                         "camera's original file", "Use the original file from the drone's card: it keeps full quality "
                         "and the camera's own metadata."))
    elif maker:
        out.append(Check("A2", "Authenticity", "File provenance", PASS, maker, f"camera metadata present: {blob[:200]}"))
    else:
        out.append(Check("A2", "Authenticity", "File provenance", INFO, None,
                         "no camera make/model or encoder tags (normal for transport streams; stripped in edited copies)"))

    loc = _iso6709(tags.get("location", "") or tags.get("com.apple.quicktime.location.ISO6709", ""))
    if loc and gps is not None and len(gps):
        d = _haversine(loc[0], loc[1], float(gps["lat"].iloc[0]), float(gps["lon"].iloc[0]))
        dmin = float(np.min(_haversine(loc[0], loc[1], gps["lat"].to_numpy(), gps["lon"].to_numpy())))
        lim = float(a.get("location_match_m", 300))
        st = PASS if dmin <= lim else WARN
        out.append(Check("A3", "Authenticity", "Embedded location vs GPS", st, round(dmin, 1),
                         f"the video's own location tag {loc[0]:.5f}, {loc[1]:.5f} is {dmin:.0f} m from the GPS track "
                         f"({d:.0f} m from its first fix)",
                         "" if st == PASS else "The telemetry may belong to a different flight."))
    start = video_start_utc(tags)
    wall = tel.frame["wall_time"] if (not tel.is_empty and "wall_time" in tel.frame) else None
    if start is not None and wall is not None and pd.to_datetime(wall, utc=True, errors="coerce").notna().any():
        w0 = pd.to_datetime(wall, utc=True, errors="coerce").dropna().min()
        days = abs((start - w0).total_seconds()) / 86400
        st = PASS if days < 1.0 else WARN
        out.append(Check("A4", "Authenticity", "Recording date vs telemetry", st, round(days, 2),
                         f"video created {start:%Y-%m-%d %H:%M} (as tagged), telemetry starts {w0:%Y-%m-%d %H:%M} UTC"
                         + (f"; {tel.alignment.get('note')}" if tel.alignment.get("method") == "wall_clock" else ""),
                         "" if st == PASS else "Video and telemetry are from different days: wrong log?"))
    s = sync or {}
    if s.get("r_best") is not None and s.get("gps_speed_cv", 0) >= 0.15:
        # Weak agreement is not proof of a mismatch: with the camera near the horizon, or on an
        # orbit, image motion is mostly rotation and tracks ground speed only loosely. The Sync
        # check blocks the clear contradictions; this one reports the evidence.
        st = PASS if s["r_best"] >= 0.6 else WARN
        out.append(Check("A5", "Authenticity", "Motion agrees with GPS", st, s["r_best"],
                         f"image motion and GPS speed correlate at r={s['r_best']:.2f} (see Sync)",
                         "" if st == PASS else "Expected with a shallow camera angle or an orbit; otherwise check "
                         "that the telemetry belongs to this video."))
    return out


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(h))


# --------------------------------------------------------------------------
# Camera and feasibility
# --------------------------------------------------------------------------
def identify_camera(probe, candidates, tel, width: int, table: Path | None = None) -> dict[str, Any]:
    """Known camera -> nominal horizontal FOV, from file tags and the flight log's aircraft column."""
    import yaml

    if not tel.is_empty and "hfov_deg" in tel.frame and tel.frame["hfov_deg"].notna().any():
        return {"model": None, "hfov_deg": float(tel.frame["hfov_deg"].median()), "source": "telemetry (KLV)"}
    texts = [f"{k}={v}" for k, v in {**probe.tags, **((probe.video.tags if probe.video else {}) or {})}.items()]
    for c in candidates:
        if c.path and c.kind == "csv":
            try:
                from src.ingest.telemetry import read_text_any

                head = read_text_any(Path(c.path))[:200000].splitlines()
                header = head[0].split(",")
                cols = [i for i, h in enumerate(header) if any(k in h.lower() for k in ("dronetype", "aircraft", "model"))]
                for line in head[1:50]:
                    parts = line.split(",")
                    texts += [parts[i] for i in cols if i < len(parts) and parts[i]]
            except Exception:  # noqa: BLE001
                pass
    table = table or Path(__file__).resolve().parents[2] / "configs" / "cameras.yaml"
    try:
        cameras = yaml.safe_load(table.read_text(encoding="utf-8"))["cameras"]
    except Exception:  # noqa: BLE001
        return {"model": None, "hfov_deg": None, "source": "camera table unavailable"}
    blob = " | ".join(texts).lower()
    best, best_len = None, 0
    for cam in cameras:
        for name in cam["names"]:
            if name.lower() in blob and len(name) > best_len:
                best, best_len = cam, len(name)
    if best is None:
        return {"model": None, "hfov_deg": None, "source": "unknown camera"}
    hfov = 2 * math.degrees(math.atan(17.3 * float(best.get("video_width_fraction", 1.0)) / float(best["focal_35mm"])))
    return {"model": best["model"], "hfov_deg": round(hfov, 2), "focal_35mm": best["focal_35mm"],
            "fisheye": bool(best.get("fisheye", False)), "source": "camera table (nominal)",
            "focal_px": round((width / 2) / math.tan(math.radians(hfov / 2)), 1) if width else None}


def _feasibility_checks(tel, gps, camera: dict, v, duration: float, c: dict, cfg: Any) -> list[Check]:
    out: list[Check] = []
    if gps is not None and len(gps) > 1:
        inside = gps[(gps["t"] >= 0) & (gps["t"] <= duration)]
        track = inside if len(inside) > 1 else gps
        path = float(np.hypot(np.diff(track["e"]), np.diff(track["n"])).sum())
        extent = float(np.hypot(np.ptp(track["e"]), np.ptp(track["n"])))
        agl = None
        if track["alt_baro"].notna().any():
            agl = float(track["alt_baro"].median())
        elif "frame_center_alt" in tel.frame and tel.frame["frame_center_alt"].notna().any():
            agl = float((tel.frame["alt_gps"] - tel.frame["frame_center_alt"]).median())
        need = float(c.get("min_path_m", 20))
        st = PASS if extent >= need else BLOCK
        out.append(Check("P1", "Feasibility", "Camera movement (parallax)", st, round(extent, 1),
                         f"the drone travelled {path:.0f} m (straight-line extent {extent:.0f} m) during the clip"
                         + (f" at ~{agl:.0f} m above takeoff" if agl else ""),
                         "" if st == PASS else "A hovering or rotating-only drone gives no parallax, so no 3-D. "
                         "Fly a pass over the area."))
        hfov = camera.get("hfov_deg")
        if agl and hfov and v.width:
            gsd = 2 * agl * math.tan(math.radians(hfov / 2)) / v.width
            lim = float(c.get("max_gsd_m", 0.3))
            st = PASS if gsd <= lim else WARN
            out.append(Check("P2", "Feasibility", "Ground sample distance", st, round(gsd, 3),
                             f"~{gsd * 100:.1f} cm per pixel at nadir ({agl:.0f} m, {hfov:.0f}° FOV, {v.width} px); "
                             "oblique views are coarser at the far edge",
                             "" if st == PASS else "Fly lower or record at a higher resolution for <= 1 m accuracy."))
    if camera.get("hfov_deg"):
        out.append(Check("P3", "Feasibility", "Camera intrinsics", PASS if "telemetry" in camera["source"] else INFO,
                         camera.get("model") or "from telemetry",
                         f"{camera.get('model') or 'field of view'}: horizontal FOV {camera['hfov_deg']:.1f}° "
                         f"({camera['source']})" + ("; wide-angle lens, strong distortion" if camera.get("fisheye") else "")))
    else:
        out.append(Check("P3", "Feasibility", "Camera intrinsics", WARN, None,
                         "camera not identified and no field of view in the telemetry: focal length will be "
                         "self-calibrated, which lets scale and height drift on straight passes",
                         "Supply the camera model (configs/cameras.yaml) or telemetry with focal length / FOV."))
    budget = float(cfg.get_path("budget.total_s", 900))
    out.append(Check("P4", "Feasibility", "Processing budget", INFO, round(duration, 1),
                     f"{duration / 60:.1f} min of video against a {budget / 60:.0f}-minute budget "
                     f"(PS: 10 min of video in 15 min)"))
    return out
