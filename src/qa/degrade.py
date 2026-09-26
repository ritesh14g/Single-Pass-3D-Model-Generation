"""The synthetic degradation harness (spec §8.5): measured before/after numbers, not claims.

Degradations are injected where frames are decoded (``VideoReader``) and where telemetry is
loaded, driven by ``qa.inject`` (off in every normal run). Injecting at decode instead of
re-encoding a degraded copy of the video means:

  * ingest and conditioning see the *same* degraded frame (deterministic per frame index),
  * the original telemetry (embedded KLV, SRT, flight logs) is untouched, on the same clock,
  * no 4K re-encode per case.

Kinds (§8.5 list):
    motion_blur      directional kernel, length scaled to frame width, direction varies slowly
    compression      JPEG round trip at low quality (a stand-in for H.264 at a low bitrate:
                     both are 8x8-ish block transforms; no ffmpeg binary is assumed)
    low_light        gain + gamma + shot/read noise
    shadow           hard-edged dark polygons drifting slowly across the frame (cloud shadows)
    gps_noise        Gaussian position noise + occasional gross outliers on the parsed telemetry
    dynamic_objects  vehicle-like blocks moving across the frame

The bench (``run_degradation_bench``) reconstructs the clean clip once, then every
kind x severity with conditioning on and off, and measures each against the clean run:
surface deviation (points vs the clean DSM, per zone), camera-centre deviation, frames
registered, coverage, time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.core.logging import get_logger, log_event

log = get_logger(__name__)
KINDS = ("motion_blur", "compression", "low_light", "shadow", "gps_noise", "dynamic_objects")
FRAME_KINDS = tuple(k for k in KINDS if k != "gps_noise")


def _rng(seed: int, index: int, salt: str) -> np.random.Generator:
    digest = hashlib.blake2b(f"{seed}:{index}:{salt}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


# -- frame degradations (image: BGR uint8, index: frame index in the video) --------------------------
def motion_blur(image: np.ndarray, index: int, length_px: float, seed: int) -> np.ndarray:
    length = max(int(round(length_px * image.shape[1] / 1920.0)), 1)
    if length < 2:
        return image
    angle = 25.0 * np.sin(index / 90.0) + float(_rng(seed, index // 30, "blur").uniform(-10, 10))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    return cv2.filter2D(image, -1, kernel / max(kernel.sum(), 1e-6))


def compression(image: np.ndarray, index: int, quality: float, seed: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else image


def low_light(image: np.ndarray, index: int, gain: float, seed: int) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    x = np.power(x, 1.0 + (1.0 - gain)) * gain                   # darker, crushed shadows
    rng = _rng(seed, index, "noise")
    photons = 255.0 * gain * 40.0                                   # fewer photons -> more shot noise
    x = rng.poisson(np.clip(x, 0, 1) * photons) / photons + rng.normal(0, 0.012 / gain, x.shape)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def shadow(image: np.ndarray, index: int, fraction: float, seed: int) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rng = _rng(seed, 0, "shadow")                                   # the same shadows all clip long
    count = 3
    for k in range(count):
        cx0, cy0 = rng.uniform(0, 1, 2)
        vx, vy = rng.uniform(-1, 1, 2) * 0.002
        cx, cy = ((cx0 + vx * index) % 1.2 - 0.1) * w, ((cy0 + vy * index) % 1.2 - 0.1) * h
        radius = np.sqrt(fraction / count / np.pi) * np.hypot(w, h) * 0.75
        angles = np.sort(rng.uniform(0, 2 * np.pi, 7))
        radii = radius * rng.uniform(0.6, 1.3, 7)
        poly = np.c_[cx + radii * np.cos(angles), cy + radii * np.sin(angles)].astype(np.int32)
        cv2.fillPoly(mask, [poly], 1)
    out = image.copy()
    out[mask > 0] = (out[mask > 0].astype(np.float32) * 0.4).astype(np.uint8)
    return out


def dynamic_objects(image: np.ndarray, index: int, count: float, seed: int) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    size = max(int(0.03 * w), 4)
    for k in range(int(count)):
        rng = _rng(seed, k, "vehicle")
        x0, y0 = rng.uniform(0, 1, 2)
        vx, vy = rng.uniform(-1, 1, 2) * 0.004
        x, y = int(((x0 + vx * index) % 1.0) * w), int(((y0 + vy * index) % 1.0) * h)
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(out, (x, y), (x + size, y + size // 2), colour, -1)
        cv2.rectangle(out, (x + size // 5, y + size // 8), (x + 4 * size // 5, y + 3 * size // 8),
                      tuple(255 - c for c in colour), -1)
    return out


_FRAME_FUNCS: dict[str, Callable[[np.ndarray, int, float, int], np.ndarray]] = {
    "motion_blur": motion_blur, "compression": compression, "low_light": low_light,
    "shadow": shadow, "dynamic_objects": dynamic_objects}


def level(cfg: Any, kind: str, severity: str) -> Any:
    return cfg.get_path(f"qa.degrade.levels.{kind}.{severity}")


def frame_degrader(cfg: Any) -> Callable[[np.ndarray, int], np.ndarray] | None:
    """The decode-time hook for ``qa.inject`` (None when nothing is injected)."""
    inject = cfg.get_path("qa.inject", None) or {}
    kind = str(inject.get("kind", "none"))
    if kind not in FRAME_KINDS:
        return None
    value, seed = level(cfg, kind, str(inject.get("severity", "moderate"))), int(inject.get("seed", 0))
    func = _FRAME_FUNCS[kind]
    log_event(log, logging.WARNING, f"QA INJECTION: every decoded frame is degraded ({kind} {value})",
              event="qa_inject", kind=kind, value=value)
    return lambda image, index: func(image, index, float(value), seed)


def inject_gps_noise(frame, cfg: Any):
    """Gaussian noise (sigma m) + gross outliers (rate, outlier m) on lat/lon/alt, seeded."""
    inject = cfg.get_path("qa.inject", None) or {}
    if str(inject.get("kind", "none")) != "gps_noise" or frame.empty:
        return frame, None
    sigma, rate = (float(v) for v in level(cfg, "gps_noise", str(inject.get("severity", "moderate"))))
    outlier_m = float(cfg.get_path("qa.degrade.gps_outlier_m", 30.0))
    rng = np.random.default_rng(int(inject.get("seed", 0)))
    n = len(frame)
    east, north, up = (rng.normal(0, sigma, n) for _ in range(3))
    gross = rng.random(n) < rate
    east[gross] += rng.choice([-1, 1], gross.sum()) * outlier_m
    north[gross] += rng.choice([-1, 1], gross.sum()) * outlier_m
    lat0 = np.deg2rad(float(np.nanmedian(frame["lat"])))
    out = frame.copy()
    out["lat"] = frame["lat"] + np.rad2deg(north / 6378137.0)
    out["lon"] = frame["lon"] + np.rad2deg(east / (6378137.0 * np.cos(lat0)))
    if "alt_gps" in out:
        out["alt_gps"] = frame["alt_gps"] + up
    info = {"sigma_m": sigma, "outlier_rate": rate, "outliers": int(gross.sum()), "rows": n}
    log_event(log, logging.WARNING, "QA INJECTION: GPS noise added to the telemetry", event="qa_inject", **info)
    return out, info


# -- the bench ------------------------------------------------------------------------------------
@dataclass
class Case:
    kind: str
    severity: str
    conditioning: bool

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.severity}_{'cond' if self.conditioning else 'raw'}"


def cases(kinds, severities) -> list[Case]:
    return [Case(k, s, c) for k in kinds for s in severities for c in (True, False)]


def case_overrides(case: Case | None) -> dict[str, Any]:
    """Config overlay for one bench run (None = the clean baseline, conditioning on)."""
    if case is None:
        return {"qa": {"inject": {"kind": "none"}}}
    over: dict[str, Any] = {"qa": {"inject": {"kind": case.kind, "severity": case.severity}}}
    if not case.conditioning:
        over["condition"] = {"enabled": False, "gps": {"enabled": False}}
    return over


def camera_deviation(run_dir: Path, baseline_dir: Path) -> dict[str, Any]:
    """Georeferenced camera centres of a run vs the clean run's, for frames both registered."""
    import pycolmap

    from src.geo.georef import Georef

    def centres(d: Path) -> dict[str, np.ndarray]:
        from src.core.manifest import RunManifest

        m = RunManifest.load(d)
        rec = pycolmap.Reconstruction(str(m.artifact("track_a", "sparse")))
        g = Georef.load(d / "geo" / "georef.json")
        names = [im.name for im in rec.images.values() if im.has_pose]
        pts = np.array([rec.images[i].projection_center() for i in rec.images if rec.images[i].has_pose])
        return dict(zip(names, g.to_local(pts))) if len(pts) else {}

    a, b = centres(run_dir), centres(baseline_dir)
    common = sorted(set(a) & set(b))
    if not common:
        return {"common_frames": 0}
    d = np.array([np.linalg.norm(a[k] - b[k]) for k in common])
    return {"common_frames": len(common), "rms_m": round(float(np.sqrt(np.mean(d ** 2))), 3),
            "median_m": round(float(np.median(d)), 3), "max_m": round(float(d.max()), 3)}


def measure_run(run_dir: Path, baseline_dir: Path | None, cfg: Any) -> dict[str, Any]:
    """The numbers the table needs from one finished bench run."""
    from src.qa.metrics import accuracy_vs_reference

    out: dict[str, Any] = {}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stages = manifest.get("stages", {})
    out["seconds"] = round(sum(float(r.get("duration_s") or 0) for r in stages.values() if r.get("status") == "done"), 1)
    out["failed"] = {k: r.get("error") for k, r in stages.items() if r.get("status") == "failed"}
    ta = (stages.get("track_a") or {}).get("metrics") or {}
    out["registered"] = ta.get("registered")
    out["frames_in"] = ta.get("frames_in")
    meta_path = run_dir / "export" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    g = meta.get("georeferencing") or {}
    out["cam_vs_gps_rms_m"] = g.get("rms_all_m")
    out["coverage_pct"] = (meta.get("coverage") or {}).get("coverage_pct")
    if baseline_dir is not None and meta and (baseline_dir / "export" / "dsm.tif").is_file():
        try:
            acc = accuracy_vs_reference(run_dir / "export", baseline_dir / "export" / "dsm.tif", cfg.get_path("qa.metrics"))
            pts = acc["points_vs_reference_dsm"]
            out["surface_rms_m"] = (pts.get("all_points") or {}).get("rms_m")
            out["surface_nmad_m"] = (pts.get("all_points") or {}).get("nmad_m")
            out["zone1_rms_m"] = (pts.get("zone1_measured") or {}).get("rms_m")
            out["shift_m"] = (acc.get("horizontal_shift") or {}).get("horizontal_m")
        except Exception as exc:  # noqa: BLE001
            out["surface_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["camera_deviation"] = camera_deviation(run_dir, baseline_dir)
        except Exception as exc:  # noqa: BLE001
            out["camera_deviation"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def run_degradation_bench(video: Path, out_dir: Path, cfg: Any, *, kinds=None, severities=None,
                          telemetry: list[Path] | None = None, stages: list[str] | None = None,
                          runner: Callable | None = None) -> dict[str, Any]:
    """Clean baseline + every case; writes degradation.json / .md in ``out_dir``.

    ``runner(inputs, cfg, run_dir, stages)`` defaults to ``src.pipeline.run_pipeline`` (tests
    pass a stand-in). Finished case folders are reused, so an interrupted bench resumes.
    """
    from src.pipeline import RunInputs

    if runner is None:
        from src.pipeline import run_pipeline

        def runner(inputs, run_cfg, run_dir, run_stages):
            return run_pipeline(inputs, run_cfg, run_dir=run_dir, stages=run_stages)

    dcfg = cfg.get_path("qa.degrade")
    kinds = list(kinds or dcfg.types)
    severities = list(severities or dcfg.severities)
    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown degradation kinds {unknown}; known: {list(KINDS)}")
    # No input check: its telemetry offset would be measured on the clean frames for every case.
    stages = list(stages or dcfg.stages)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = RunInputs(video=Path(video), telemetry=list(telemetry or []), accept_input=True)

    def run_one(name: str, case: Case | None) -> dict[str, Any]:
        run_dir = out_dir / name
        run_cfg = cfg.merged(case_overrides(case))
        started = time.perf_counter()
        error = None
        if not (run_dir / "export" / "metadata.json").is_file():
            try:
                runner(inputs, run_cfg, run_dir, stages)
            except Exception as exc:  # noqa: BLE001 - a case that breaks the pipeline is a result
                error = f"{type(exc).__name__}: {exc}"
        row = {"case": name, "kind": case.kind if case else "clean", "severity": case.severity if case else "-",
               "conditioning": case.conditioning if case else True, "wall_s": round(time.perf_counter() - started, 1)}
        if error:
            row["error"] = error
        if (run_dir / "manifest.json").is_file():
            row.update(measure_run(run_dir, None if case is None else out_dir / "clean", run_cfg))
        log_event(log, logging.INFO, f"bench case {name} done", **{k: v for k, v in row.items() if not isinstance(v, dict)})
        return row

    rows = [run_one("clean", None)]
    for case in cases(kinds, severities):
        rows.append(run_one(case.name, case))
        _write(out_dir, rows, video, kinds, severities)
    return _write(out_dir, rows, video, kinds, severities)


def _write(out_dir: Path, rows: list[dict], video: Path, kinds, severities) -> dict[str, Any]:
    result = {"video": str(video), "kinds": list(kinds), "severities": list(severities), "rows": rows,
              "reference": "the clean reconstruction of the same clip (conditioning on)",
              "table_html": table_html(rows), "table_md": table_md(rows)}
    (out_dir / "degradation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out_dir / "degradation.md").write_text(result["table_md"], encoding="utf-8")
    return result


def _pairs(rows: list[dict]):
    """(kind, severity, row with conditioning, row without) in bench order."""
    by = {(r["kind"], r["severity"], r["conditioning"]): r for r in rows}
    seen = []
    for r in rows:
        key = (r["kind"], r["severity"])
        if r["kind"] != "clean" and key not in seen:
            seen.append(key)
            yield key[0], key[1], by.get((*key, True), {}), by.get((*key, False), {})


def _cell(row: dict, key: str, digits: int = 2) -> str:
    if row.get("error"):
        return "failed"
    v = row.get(key)
    return "—" if v is None else f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)


def _cam(row: dict) -> str:
    dev = row.get("camera_deviation") or {}
    return "failed" if row.get("error") else "—" if dev.get("rms_m") is None else f"{dev['rms_m']:.2f}"


COLUMNS = (("Surface RMS vs clean (m)", "surface_rms_m", 2), ("Zone 1 RMS (m)", "zone1_rms_m", 2),
           ("Registered", "registered", 0), ("Coverage %", "coverage_pct", 1))


def table_md(rows: list[dict]) -> str:
    head = "| Degradation | Severity | " + " | ".join(f"{c[0]} with / without" for c in COLUMNS) + \
           " | Cameras vs clean (m) with / without |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 3)]
    for kind, sev, w, wo in _pairs(rows):
        cells = [f"{_cell(w, k, d)} / {_cell(wo, k, d)}" for _, k, d in COLUMNS]
        lines.append(f"| {kind} | {sev} | " + " | ".join(cells) + f" | {_cam(w)} / {_cam(wo)} |")
    clean = next((r for r in rows if r["kind"] == "clean"), {})
    lines.append(f"\nClean baseline: registered {_cell(clean, 'registered', 0)}, coverage "
                 f"{_cell(clean, 'coverage_pct', 1)}%, cameras vs GPS {_cell(clean, 'cam_vs_gps_rms_m')} m, "
                 f"{_cell(clean, 'seconds', 0)} s.")
    return "\n".join(lines) + "\n"


def table_html(rows: list[dict]) -> str:
    import html

    head = "".join(f"<th class='num'>{html.escape(c[0])}<br><span class='muted'>with / without</span></th>" for c in COLUMNS)
    body = []
    for kind, sev, w, wo in _pairs(rows):
        cells = "".join(f"<td class='num'>{_cell(w, k, d)} / {_cell(wo, k, d)}</td>" for _, k, d in COLUMNS)
        body.append(f"<tr><td>{html.escape(kind)}</td><td>{html.escape(sev)}</td>{cells}"
                    f"<td class='num'>{_cam(w)} / {_cam(wo)}</td></tr>")
    return ("<div class='table-wrap'><table><tr><th>Degradation</th><th>Severity</th>" + head +
            "<th class='num'>Cameras vs clean (m)<br><span class='muted'>with / without</span></th></tr>" +
            "".join(body) + "</table></div>")
