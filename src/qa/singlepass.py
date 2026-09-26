"""The single-pass simulation (spec §8.5): how accurate is ONE pass, against a real reference?

    1. Reconstruct a multi-strip survey with every strip -> the pseudo-ground truth.
    2. Find the flight strips in its telemetry (straight runs between turns).
    3. Cut one strip's time window out of the video (+ its telemetry, as a CSV on the clip's
       clock, same altitude datum as the full run) and reconstruct that alone.
    4. Compare the one-strip model with the all-strips DSM: heights per zone, horizontal
       placement, coverage.

Needs footage that actually has several strips over the same ground; ``find_strips`` also
answers whether a clip qualifies (``is_multi_strip``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_event

log = get_logger(__name__)


def _enu(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(np.nanmedian(lat))
    r = 6378137.0
    return np.c_[np.deg2rad(lon - np.nanmedian(lon)) * r * np.cos(lat0), np.deg2rad(lat - np.nanmedian(lat)) * r]


def find_strips(telemetry: pd.DataFrame, scfg: Any) -> dict[str, Any]:
    """Straight runs of the flight: [{t0, t1, heading_deg, length_m}], and whether they form a survey.

    Heading comes from the displacement over ``window_s``; a strip continues while the heading
    stays within ``turn_deg`` of the strip's start and the drone moves faster than ``min_speed_ms``.
    """
    df = telemetry.dropna(subset=["lat", "lon"]).sort_values("t")
    if len(df) < 3:
        return {"strips": [], "is_multi_strip": False, "reason": "fewer than 3 GPS fixes"}
    t = df["t"].to_numpy(float)
    xy = _enu(df["lat"].to_numpy(float), df["lon"].to_numpy(float))
    window = float(scfg.get("window_s", 4.0))
    step = np.searchsorted(t, t + window).clip(max=len(t) - 1)
    d = xy[step] - xy
    dt = np.maximum(t[step] - t, 1e-6)
    speed = np.hypot(d[:, 0], d[:, 1]) / dt
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    turn = float(scfg.get("turn_deg", 30.0))
    moving = speed >= float(scfg.get("min_speed_ms", 1.0))
    strips, start = [], None
    for i in range(len(t)):
        if start is not None:
            diff = abs((heading[i] - heading[start] + 180) % 360 - 180)
            if not moving[i] or diff > turn:
                strips.append((start, i - 1))
                start = None
        if start is None and moving[i]:
            start = i
    if start is not None:
        strips.append((start, len(t) - 1))
    min_len = float(scfg.get("min_length_m", 100.0))
    out = []
    for a, b in strips:
        length = float(np.hypot(*(xy[b] - xy[a])))
        if length >= min_len and b > a:
            # Direction fitted to every fix of the strip (its ends lean into the turns).
            pts = xy[a:b + 1]
            centre = pts.mean(0)
            direction = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
            direction = direction if direction @ (xy[b] - xy[a]) >= 0 else -direction
            h = np.degrees(np.arctan2(direction[0], direction[1])) % 360
            out.append({"t0": round(float(t[a]), 2), "t1": round(float(t[b]), 2), "heading_deg": round(float(h), 1),
                        "length_m": round(length, 1), "_line": (centre, direction)})
    # A survey: two strips side by side (parallel or antiparallel, laterally apart) over the same ground.
    pairs = []
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            dh = abs((out[i]["heading_deg"] - out[j]["heading_deg"] + 180) % 360 - 180)
            if min(dh, 180 - dh) > turn:
                continue
            centre, direction = out[i]["_line"]
            lateral = abs(np.cross(direction, out[j]["_line"][0] - centre))
            if float(scfg.get("min_spacing_m", 20.0)) <= lateral <= float(scfg.get("max_spacing_m", 400.0)):
                pairs.append((i, j, round(float(lateral), 1)))
    for s in out:
        s.pop("_line")
    return {"strips": out, "is_multi_strip": bool(pairs), "side_by_side": pairs,
            "reason": "" if pairs else "no two strips run side by side over the same ground"}


def make_strip_input(video: Path, telemetry: pd.DataFrame, t0: float, t1: float, out_dir: Path) -> tuple[Path, Path]:
    """(clip, telemetry CSV) for [t0, t1]: the clip re-encoded from the original frames, the CSV
    on the clip's clock with a first row interpolated exactly at the cut (the CSV reader counts
    time from its first row)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip = out_dir / "strip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    first, last = int(round(t0 * fps)), int(round(t1 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    written = 0
    for _ in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    if written < 2:
        raise RuntimeError(f"could not read frames {first}-{last} of {video}")

    df = telemetry.dropna(subset=["lat", "lon"]).sort_values("t")
    cols = [c for c in ("lat", "lon", "alt_gps", "yaw", "pitch", "roll", "hfov_deg") if c in df and df[c].notna().any()]
    start = {c: float(np.interp(t0, df["t"], df[c].interpolate().bfill().ffill())) for c in cols}
    inside = df[(df["t"] > t0) & (df["t"] <= t0 + written / fps)]
    rows = pd.concat([pd.DataFrame([{"t": t0, **start}]), inside[["t", *cols]]], ignore_index=True)
    rows["time_s"] = rows.pop("t") - t0
    names = {"alt_gps": "altitude", "hfov_deg": "hfov_deg"}
    csv = out_dir / "strip_telemetry.csv"
    rows.rename(columns=names)[["time_s", *[names.get(c, c) for c in cols]]].to_csv(csv, index=False)
    return clip, csv


def run_single_pass(video: Path, out_dir: Path, cfg: Any, *, telemetry: list[Path] | None = None,
                    full_run: Path | None = None, strip: int | None = None,
                    runner: Callable | None = None) -> dict[str, Any]:
    """All strips vs one strip; writes single_pass.json in ``out_dir``."""
    from src.pipeline import RunInputs
    from src.qa.metrics import accuracy_vs_reference

    if runner is None:
        from src.pipeline import run_pipeline

        def runner(inputs, run_cfg, run_dir, run_stages):
            return run_pipeline(inputs, run_cfg, run_dir=run_dir, stages=run_stages)

    scfg = cfg.get_path("qa.single_pass")
    stages = list(scfg.stages)
    out_dir = Path(out_dir)
    full = Path(full_run) if full_run else out_dir / "all_strips"
    if not (full / "export" / "metadata.json").is_file():
        runner(RunInputs(video=Path(video), telemetry=list(telemetry or []), accept_input=True), cfg, full,
               ["preflight", *stages])
    tel = pd.read_parquet(full / "ingest" / "telemetry.parquet")
    found = find_strips(tel, scfg)
    result: dict[str, Any] = {"video": str(video), "all_strips_run": str(full), **found}
    if not found["strips"]:
        result["error"] = "no flight strip found"
        return _write(out_dir, result)
    if not found["is_multi_strip"]:
        result["warning"] = ("not a multi-strip survey: the 'all strips' model is not a better reference "
                             "than one strip; the comparison below only checks repeatability")
    index = int(strip) if strip is not None else int(np.argmax([s["length_m"] for s in found["strips"]]))
    chosen = found["strips"][index]
    result["strip"] = {"index": index, **chosen}
    one = out_dir / f"strip_{index}"
    if not (one / "export" / "metadata.json").is_file():
        clip, csv = make_strip_input(Path(video), tel, chosen["t0"], chosen["t1"], one / "input")
        meta = json.loads((full / "export" / "metadata.json").read_text(encoding="utf-8"))
        datum = (((meta.get("georeferencing") or {}).get("frame")) or {}).get("gps_altitude_datum")
        # The strip's CSV must be read in the same altitude datum as the full run's telemetry.
        strip_cfg = cfg.merged({"geo": {"vertical": {"gps_altitude_datum": datum}}}) if datum else cfg
        runner(RunInputs(video=clip, telemetry=[csv], accept_input=True), strip_cfg, one, stages)
    acc = accuracy_vs_reference(one / "export", full / "export" / "dsm.tif", cfg.get_path("qa.metrics"))
    m_one = json.loads((one / "export" / "metadata.json").read_text(encoding="utf-8"))
    m_all = json.loads((full / "export" / "metadata.json").read_text(encoding="utf-8"))
    result.update(accuracy=acc, coverage_one=(m_one.get("coverage") or {}).get("coverage_pct"),
                  coverage_all=(m_all.get("coverage") or {}).get("coverage_pct"),
                  zones_one=m_one.get("zones"), zones_all=m_all.get("zones"))
    log_event(log, logging.INFO, "single-pass simulation done", strip=index,
              zone1_rms=((acc["points_vs_reference_dsm"].get("zone1_measured") or {}).get("rms_m")))
    return _write(out_dir, result)


def _write(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["summary_html"] = summary_html(result)
    (out_dir / "single_pass.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def summary_html(r: dict[str, Any]) -> str:
    import html

    if r.get("error"):
        return f"<p>{html.escape(r['error'])}</p>"
    acc = r.get("accuracy") or {}
    pts = acc.get("points_vs_reference_dsm") or {}
    s = r.get("strip") or {}
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td class='num'>{(pts.get(k) or {}).get('n', 0):,}</td>"
        f"<td class='num'>{(pts.get(k) or {}).get('rms_m', '—')}</td><td class='num'>{(pts.get(k) or {}).get('nmad_m', '—')}</td></tr>"
        for k, label in (("zone1_measured", "Zone 1"), ("zone2_measured", "Zone 2 (measured)"),
                         ("zone2_fill", "Zone 2 (anchored fill)"), ("all_points", "All")))
    warn = f"<p class='warn'>{html.escape(r['warning'])}</p>" if r.get("warning") else ""
    shift = (acc.get("horizontal_shift") or {}).get("horizontal_m")
    return (f"{warn}<p>Strip {s.get('index')} ({s.get('length_m')} m, heading {s.get('heading_deg')}°, "
            f"{s.get('t0')}–{s.get('t1')} s) of {len(r.get('strips') or [])} found, reconstructed alone and "
            f"compared with the all-strips model: horizontal placement {shift} m; coverage "
            f"{r.get('coverage_one')}% vs {r.get('coverage_all')}%.</p><div class='table-wrap'><table><tr><th>Points</th>"
            f"<th class='num'>n</th><th class='num'>RMS (m)</th><th class='num'>NMAD (m)</th></tr>{rows}</table></div>")
