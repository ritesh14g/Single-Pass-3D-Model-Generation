"""Telemetry readers beyond KLV / SRT / CSV / DJI flight records (spec §4.2, "use when present").

Each reader returns the canonical ``TelemetryTable`` (``t, lat, lon, alt_gps, alt_baro, roll,
pitch, yaw, focal_mm``) with ``wall_time`` when the format carries a clock, and records how
its clock relates to the video in ``table.alignment``:

  * GPX / KML / GeoJSON / JSON — track logs with UTC times;
  * PX4 ULog (``.ulg``), ArduPilot DataFlash (``.bin``), MAVLink telemetry logs (``.tlog``) —
    autopilot logs, UTC from the GPS; these carry *vehicle* attitude, so only the heading is
    kept (the camera gimbal is not in them);
  * a DJI subtitle track inside the MP4/MOV — the SRT text, on the video's own clock;
  * GoPro GPMF (``gpmd`` stream) — GPS5 / GPS9 samples, on the video's own clock.

``align_to_video`` then ties a log with a UTC clock to the video's creation time, removing a
time-zone offset (cameras often write local time labelled as UTC).
"""

from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_downgrade
from src.ingest.telemetry import (TelemetryTable, parse_dji_srt_text, read_text_any, resolve_csv_columns,
                                  table_from_frame)

log = get_logger(__name__)


def _from_rows(rows: list[dict[str, Any]], source: str, method: str, note: str) -> TelemetryTable:
    rows = [r for r in rows if r.get("lat") is not None and r.get("lon") is not None
            and np.isfinite(r["lat"]) and np.isfinite(r["lon"]) and not (r["lat"] == 0 and r["lon"] == 0)]
    if not rows:
        return TelemetryTable.empty(f"{source}: no usable GPS rows")
    if "wall_time" in rows[0] and rows[0]["wall_time"] is not None:
        t0 = min(r["wall_time"] for r in rows if r.get("wall_time") is not None)
        for r in rows:
            if r.get("wall_time") is not None:
                r["t"] = (r["wall_time"] - t0).total_seconds()
    table = TelemetryTable.from_records(rows, source=source)
    table.alignment = {"method": method, "note": note}
    return table


def _utc(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# --------------------------------------------------------------------------
# GPX / KML
# --------------------------------------------------------------------------
def parse_gpx(path: Path | str) -> TelemetryTable:
    path = Path(path)
    try:
        root = ET.fromstring(read_text_any(path).encode("utf-8"))
    except ET.ParseError as exc:
        return TelemetryTable.empty(f"{path.name}: not valid GPX XML ({exc})")
    rows = []
    for el in root.iter():
        if _local(el.tag) not in ("trkpt", "rtept"):
            continue
        row: dict[str, Any] = {"lat": float(el.get("lat")), "lon": float(el.get("lon"))}
        for child in el.iter():
            name, text = _local(child.tag).lower(), (child.text or "").strip()
            if name == "ele" and text:
                row["alt_gps"] = float(text)
            elif name == "time" and text:
                row["wall_time"] = _utc(text)
            elif name in ("course", "heading", "bearing") and text:
                row["yaw"] = float(text)
        rows.append(row)
    if rows and not any(r.get("wall_time") is not None for r in rows):
        return TelemetryTable.empty(f"{path.name}: GPX track has no <time> stamps, so it cannot be synced to video")
    return _from_rows(rows, f"gpx:{path.name}", "wall_clock", "GPX UTC times")


def parse_kml(path: Path | str) -> TelemetryTable:
    path = Path(path)
    try:
        root = ET.fromstring(read_text_any(path).encode("utf-8"))
    except ET.ParseError as exc:
        return TelemetryTable.empty(f"{path.name}: not valid KML XML ({exc})")
    rows: list[dict[str, Any]] = []
    for track in (el for el in root.iter() if _local(el.tag) == "Track"):          # gx:Track
        whens = [_utc(c.text) for c in track if _local(c.tag) == "when"]
        coords = [(c.text or "").split() for c in track if _local(c.tag) == "coord"]
        for when, coord in zip(whens, coords):
            if len(coord) >= 2:
                rows.append({"lon": float(coord[0]), "lat": float(coord[1]),
                             "alt_gps": float(coord[2]) if len(coord) > 2 else None, "wall_time": when})
    if not rows:                                                                   # timestamped Placemarks
        for pm in (el for el in root.iter() if _local(el.tag) == "Placemark"):
            when = next((_utc(e.text) for e in pm.iter() if _local(e.tag) == "when"), None)
            coord = next(((e.text or "").strip().split(",") for e in pm.iter() if _local(e.tag) == "coordinates"), None)
            if when is not None and coord and len(coord) >= 2 and " " not in coord[1].strip():
                rows.append({"lon": float(coord[0]), "lat": float(coord[1]),
                             "alt_gps": float(coord[2]) if len(coord) > 2 else None, "wall_time": when})
    if not rows:
        return TelemetryTable.empty(f"{path.name}: KML has no timestamped track (gx:Track or timed Placemarks); "
                                    "a plain LineString has no times and cannot be synced to video")
    return _from_rows(rows, f"kml:{path.name}", "wall_clock", "KML UTC times")


# --------------------------------------------------------------------------
# JSON / GeoJSON
# --------------------------------------------------------------------------
def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = obj
    return out


def _records_in(obj: Any) -> list[dict[str, Any]] | None:
    """The first list of objects in a JSON document (``[...]`` or ``{"data": [...]}`` etc.)."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = _records_in(value)
            if found:
                return found
    return None


def parse_json(path: Path | str, column_map: Mapping[str, Sequence[str]],
               video_duration_s: float | None = None, recording: Mapping[str, Any] | None = None) -> TelemetryTable:
    path = Path(path)
    try:
        doc = json.loads(read_text_any(path))
    except json.JSONDecodeError as exc:
        return TelemetryTable.empty(f"{path.name}: not valid JSON ({exc})")
    if isinstance(doc, dict) and doc.get("type") in ("FeatureCollection", "Feature"):
        return _parse_geojson(path, doc)
    records = _records_in(doc)
    if not records:
        return TelemetryTable.empty(f"{path.name}: JSON holds no list of telemetry records")
    frame = pd.DataFrame([_flatten(r) for r in records])
    resolved = resolve_csv_columns(frame.columns, column_map)
    table = table_from_frame(frame, resolved, f"json:{path.name}", path, video_duration_s, recording)
    if not table.is_empty:
        table.notes.append(f"columns resolved: {resolved}")
    return table


def _parse_geojson(path: Path, doc: dict) -> TelemetryTable:
    features = doc.get("features", [doc])
    rows: list[dict[str, Any]] = []
    for feat in features:
        geom, props = feat.get("geometry") or {}, feat.get("properties") or {}
        coords = geom.get("coordinates")
        time_key = next((k for k in props if k.lower() in ("time", "timestamp", "datetime", "when")), None)
        if geom.get("type") == "Point" and coords and time_key:
            rows.append({"lon": float(coords[0]), "lat": float(coords[1]),
                         "alt_gps": float(coords[2]) if len(coords) > 2 else None, "wall_time": _utc(props[time_key])})
        elif geom.get("type") == "LineString" and coords:
            times = props.get("coordTimes") or props.get("times") or []
            for c, when in zip(coords, times):
                rows.append({"lon": float(c[0]), "lat": float(c[1]),
                             "alt_gps": float(c[2]) if len(c) > 2 else None, "wall_time": _utc(when)})
    if not rows:
        return TelemetryTable.empty(f"{path.name}: GeoJSON has no timestamped points (Point features with a time "
                                    "property, or a LineString with coordTimes)")
    return _from_rows(rows, f"geojson:{path.name}", "wall_clock", "GeoJSON UTC times")


# --------------------------------------------------------------------------
# Autopilot logs: PX4 ULog, ArduPilot DataFlash, MAVLink tlog
# --------------------------------------------------------------------------
def parse_ulog(path: Path | str) -> TelemetryTable:
    path = Path(path)
    try:
        from pyulog import ULog
    except ImportError:
        return TelemetryTable.empty(f"{path.name}: PX4 ULog needs the 'pyulog' package (pip install pyulog)")
    try:
        ulog = ULog(str(path), ["vehicle_gps_position", "sensor_gps", "vehicle_global_position", "vehicle_attitude"])
    except Exception as exc:  # noqa: BLE001
        return TelemetryTable.empty(f"{path.name}: could not read ULog ({type(exc).__name__}: {exc})")
    topics = {d.name: d.data for d in ulog.data_list}
    gps = topics.get("sensor_gps") or topics.get("vehicle_gps_position")
    if gps is None:
        return TelemetryTable.empty(f"{path.name}: ULog has no GPS topic (sensor_gps / vehicle_gps_position)")
    ts = gps["timestamp"].astype(float) / 1e6
    lat = gps["latitude_deg"] if "latitude_deg" in gps else gps["lat"] / 1e7
    lon = gps["longitude_deg"] if "longitude_deg" in gps else gps["lon"] / 1e7
    alt = gps["altitude_msl_m"] if "altitude_msl_m" in gps else gps["alt"] / 1e3
    utc = gps.get("time_utc_usec")
    frame = pd.DataFrame({"t": ts - ts[0], "lat": lat, "lon": lon, "alt_gps": alt})
    if utc is not None and (utc > 0).any():
        boot_to_utc = np.median((utc[utc > 0] / 1e6) - ts[utc > 0])
        frame["wall_time"] = pd.to_datetime(ts + boot_to_utc, unit="s", utc=True)
    att = topics.get("vehicle_attitude")
    if att is not None and "q[0]" in att:
        q0, q1, q2, q3 = (att[f"q[{i}]"] for i in range(4))
        yaw = np.degrees(np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2)))
        frame["yaw"] = np.interp(ts, att["timestamp"] / 1e6, np.unwrap(np.radians(yaw)) * 180 / np.pi) % 360
    table = TelemetryTable.from_records(frame.to_dict("records"), source=f"ulog:{path.name}")
    table.alignment = {"method": "wall_clock" if "wall_time" in frame else "log_start", "note": "PX4 GPS UTC"}
    table.notes.append("autopilot log: vehicle heading kept; camera gimbal attitude is not in the log")
    return table


GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_LEAP_S = 18


def parse_mavlink(path: Path | str) -> TelemetryTable:
    """ArduPilot DataFlash ``.bin`` or a MAVLink ``.tlog`` (ground-station telemetry log)."""
    path = Path(path)
    try:
        from pymavlink import mavutil
    except ImportError:
        return TelemetryTable.empty(f"{path.name}: ArduPilot/MAVLink logs need 'pymavlink' (pip install pymavlink)")
    try:
        conn = mavutil.mavlink_connection(str(path), dialect="ardupilotmega")
    except Exception as exc:  # noqa: BLE001
        return TelemetryTable.empty(f"{path.name}: could not open ({type(exc).__name__}: {exc})")
    rows: list[dict[str, Any]] = []
    headings: list[tuple[float, float]] = []
    while True:
        try:
            msg = conn.recv_match(type=["GPS", "ATT", "GLOBAL_POSITION_INT", "ATTITUDE"], blocking=False)
        except Exception:  # noqa: BLE001 - a corrupt tail ends the log, it does not fail it
            break
        if msg is None:
            break
        kind = msg.get_type()
        if kind == "GPS" and getattr(msg, "Status", 3) >= 3:                     # DataFlash, 3-D fix
            wall = GPS_EPOCH + timedelta(weeks=int(msg.GWk), milliseconds=int(msg.GMS)) - timedelta(seconds=GPS_LEAP_S)
            rows.append({"boot_s": msg.TimeUS / 1e6, "lat": msg.Lat, "lon": msg.Lng, "alt_gps": msg.Alt,
                         "wall_time": pd.Timestamp(wall), "sats": msg.NSats})
        elif kind == "GLOBAL_POSITION_INT":                                       # tlog
            rows.append({"boot_s": msg.time_boot_ms / 1e3, "lat": msg.lat / 1e7, "lon": msg.lon / 1e7,
                         "alt_gps": msg.alt / 1e3, "alt_baro": msg.relative_alt / 1e3,
                         "wall_time": pd.Timestamp(msg._timestamp, unit="s", tz="UTC"),
                         "yaw": msg.hdg / 100.0 if msg.hdg != 65535 else None})
        elif kind == "ATT":
            headings.append((msg.TimeUS / 1e6, float(msg.Yaw)))
    if not rows:
        return TelemetryTable.empty(f"{path.name}: no GPS fixes (GPS with a 3-D fix / GLOBAL_POSITION_INT)")
    if headings and not any(r.get("yaw") is not None for r in rows):
        ht = np.array(headings)
        yaw = np.interp([r["boot_s"] for r in rows], ht[:, 0], np.unwrap(np.radians(ht[:, 1])) * 180 / np.pi) % 360
        for r, y in zip(rows, yaw):
            r["yaw"] = float(y)
    for r in rows:
        r.pop("boot_s", None)
    table = _from_rows(rows, f"{'dataflash' if path.suffix.lower() == '.bin' else 'tlog'}:{path.name}",
                       "wall_clock", "autopilot GPS UTC")
    table.notes.append("autopilot log: vehicle heading kept; camera gimbal attitude is not in the log")
    return table


# --------------------------------------------------------------------------
# Streams inside the video: DJI subtitle track, GoPro GPMF
# --------------------------------------------------------------------------
def _fmt_cue(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def extract_subtitle_srt(video: Path | str) -> str | None:
    """The first subtitle track of a video, rebuilt as SRT text (DJI embeds its SRT this way)."""
    import av

    with av.open(str(video), metadata_errors="ignore") as container:
        streams = [s for s in container.streams if s.type == "subtitle"]
        if not streams:
            return None
        stream, cues = streams[0], []
        for packet in container.demux(stream):
            if packet.pts is None or packet.size == 0:
                continue
            data = bytes(packet)
            if stream.codec_context.name == "mov_text" and len(data) >= 2:
                n = int.from_bytes(data[:2], "big")
                data = data[2:2 + n]
            text = data.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            start = float(packet.pts * packet.time_base)
            end = start + float((packet.duration or 0) * packet.time_base)
            cues.append(f"{len(cues) + 1}\n{_fmt_cue(start)} --> {_fmt_cue(max(end, start + 0.001))}\n{text}\n")
    return "\n".join(cues) if cues else None


def parse_embedded_srt(video: Path | str) -> TelemetryTable:
    video = Path(video)
    try:
        text = extract_subtitle_srt(video)
    except Exception as exc:  # noqa: BLE001
        return TelemetryTable.empty(f"{video.name}: could not read the subtitle track ({type(exc).__name__}: {exc})")
    if not text:
        return TelemetryTable.empty(f"{video.name}: subtitle track is empty")
    table = parse_dji_srt_text(text, video.name, source_prefix="embedded_srt")
    if table.is_empty:
        table.notes.append("the video's subtitle track carries no GPS (not a DJI telemetry caption)")
    return table


_GPMF_TYPES = {"b": ("b", 1), "B": ("B", 1), "s": ("h", 2), "S": ("H", 2), "l": ("i", 4), "L": ("I", 4),
               "f": ("f", 4), "d": ("d", 8), "j": ("q", 8), "J": ("Q", 8)}


def _gpmf_items(buf: bytes):
    """Yield (key, type, struct_size, repeat, payload) for one GPMF level."""
    pos = 0
    while pos + 8 <= len(buf):
        key = buf[pos:pos + 4].decode("latin-1")
        typ = chr(buf[pos + 4])
        size, repeat = buf[pos + 5], struct.unpack(">H", buf[pos + 6:pos + 8])[0]
        length = size * repeat
        payload = buf[pos + 8:pos + 8 + length]
        yield key, typ, size, repeat, payload
        pos += 8 + ((length + 3) & ~3)


def _gpmf_values(typ: str, size: int, repeat: int, payload: bytes) -> np.ndarray | None:
    if typ not in _GPMF_TYPES:
        return None
    fmt, width = _GPMF_TYPES[typ]
    per = size // width
    vals = struct.unpack(f">{per * repeat}{fmt}", payload[:per * repeat * width])
    return np.array(vals, dtype=float).reshape(repeat, per)


def parse_gpmf_payload(payload: bytes) -> list[dict[str, Any]]:
    """GPS samples in one GPMF packet (GPS5: lat, lon, alt, 2D, 3D speed; GPS9 adds date/time)."""
    samples: list[dict[str, Any]] = []
    for key, typ, size, repeat, body in _gpmf_items(payload):
        if key != "DEVC" or typ != "\x00":
            continue
        for skey, styp, _, _, sbody in _gpmf_items(body):
            if skey != "STRM" or styp != "\x00":
                continue
            scal, data, name, utc, fix = None, None, None, None, 3
            for k, t, sz, rp, b in _gpmf_items(sbody):
                if k == "SCAL":
                    scal = _gpmf_values(t, sz, rp, b)
                elif k in ("GPS5", "GPS9"):
                    data, name = _gpmf_values(t, sz, rp, b), k
                elif k == "GPSU" and t == "U":
                    utc = b[:16].decode("ascii", errors="ignore")
                elif k == "GPSF":
                    fix = int(_gpmf_values(t, sz, rp, b).ravel()[0])
            if data is None or fix < 2:
                continue
            scale = scal.ravel() if scal is not None else np.ones(data.shape[1])
            if scale.size == 1:
                scale = np.repeat(scale, data.shape[1])
            vals = data / scale[: data.shape[1]]
            for row in vals:
                sample = {"lat": row[0], "lon": row[1], "alt_gps": row[2]}
                if name == "GPS9" and len(row) >= 7:
                    sample["wall_time"] = (pd.Timestamp("2000-01-01", tz="UTC") + pd.Timedelta(days=row[5])
                                           + pd.Timedelta(seconds=row[6]))
                samples.append(sample)
            if utc and samples and "wall_time" not in samples[0]:
                try:
                    stamp = pd.Timestamp(datetime.strptime(utc, "%y%m%d%H%M%S.%f"), tz="UTC")
                    samples[-len(vals)]["gpsu"] = stamp
                except ValueError:
                    pass
    return samples


def parse_gpmf(video: Path | str) -> TelemetryTable:
    """GoPro GPS from the ``gpmd`` data stream: packet times spread over each packet's samples."""
    import av

    video = Path(video)
    rows: list[dict[str, Any]] = []
    try:
        with av.open(str(video), metadata_errors="ignore") as container:
            def is_gpmf(s: Any) -> bool:
                try:
                    tag = str(getattr(s, "codec_tag", "") or "")
                except Exception:  # noqa: BLE001
                    tag = ""
                return tag == "gpmd" or "gopro met" in str(s.metadata.get("handler_name", "")).lower()

            streams = [s for s in container.streams if s.type in ("data", "unknown") and is_gpmf(s)]
            if not streams:
                return TelemetryTable.empty(f"{video.name}: no GoPro GPMF stream")
            for packet in container.demux(streams[0]):
                if packet.pts is None or packet.size == 0:
                    continue
                start = float(packet.pts * packet.time_base)
                dur = float((packet.duration or 0) * packet.time_base) or 1.0
                samples = parse_gpmf_payload(bytes(packet))
                for i, smp in enumerate(samples):
                    smp["t"] = start + dur * i / max(len(samples), 1)
                    smp.pop("gpsu", None)
                    rows.append(smp)
    except Exception as exc:  # noqa: BLE001
        return TelemetryTable.empty(f"{video.name}: could not read GPMF ({type(exc).__name__}: {exc})")
    if not rows:
        return TelemetryTable.empty(f"{video.name}: GPMF stream has no GPS lock (GPSF < 2)")
    for r in rows:
        r.pop("wall_time", None)          # packet times are already on the video clock
    table = TelemetryTable.from_records(rows, source=f"gpmf:{video.name}")
    table.alignment = {"method": "video_clock", "note": "GPMF packets are timed on the video timeline"}
    return table


# --------------------------------------------------------------------------
# Clock alignment
# --------------------------------------------------------------------------
def video_start_utc(tags: Mapping[str, str]) -> pd.Timestamp | None:
    for key in ("creation_time", "com.apple.quicktime.creationdate", "date"):
        if tags.get(key):
            ts = pd.to_datetime(tags[key], utc=True, errors="coerce")
            if not pd.isna(ts) and ts.year > 2000:
                return ts
    return None


def align_to_video(table: TelemetryTable, video_start: pd.Timestamp | None, video_duration_s: float | None,
                   offset_s: float | None, tz_step_s: float, tolerance_s: float, pad_s: float = 2.0) -> TelemetryTable:
    """Re-time a table so t = 0 is the first video frame; record how in ``table.alignment``."""
    if table.is_empty:
        return table
    method = (table.alignment or {}).get("method", "log_start")
    frame = table.frame
    if offset_s is not None:
        frame["t"] = frame["t"] - float(offset_s)
        table.alignment = {"method": "offset", "offset_s": float(offset_s),
                           "note": f"video t=0 at telemetry t={offset_s:+.2f} s (ingest.telemetry.time_offset_s)"}
    elif method in ("video_clock", "recording_flag"):
        return table
    elif "wall_time" in frame and frame["wall_time"].notna().any() and video_start is not None:
        wall = pd.to_datetime(frame["wall_time"], utc=True, errors="coerce")
        diff = (video_start - wall.min()).total_seconds()
        zone = round(diff / tz_step_s) * tz_step_s if tz_step_s else 0.0
        residual = diff - zone
        if abs(residual) <= tolerance_s and (video_duration_s is None or residual < (wall.max() - wall.min()).total_seconds()):
            frame["t"] = (wall - wall.min()).dt.total_seconds() - residual
            hours = zone / 3600.0
            table.alignment = {"method": "wall_clock", "offset_s": round(residual, 3), "timezone_offset_h": hours,
                               "note": (f"log UTC clock vs video creation time: video starts {residual:+.1f} s into the "
                                        f"log" + (f" after removing a {hours:+.2f} h clock/time-zone offset" if zone else ""))}
        else:
            table.alignment = {"method": "log_start", "clock_mismatch_s": round(diff, 1),
                               "note": (f"log clock and video creation time disagree by {diff / 3600:.2f} h and are not a "
                                        "time-zone step apart; log assumed to start with the video")}
    if video_duration_s:
        keep = (frame["t"] >= -pad_s) & (frame["t"] <= video_duration_s + pad_s)
        if keep.sum() >= 2:
            frame = frame[keep].reset_index(drop=True)
    table.frame = frame
    return table


def parse_by_kind(kind: str, path: Path | None, video: Path, column_map: Mapping[str, Sequence[str]],
                  video_duration_s: float | None, recording: Mapping[str, Any] | None,
                  flight_record: Mapping[str, Any] | None, headerless_order: Sequence[str]) -> TelemetryTable:
    """Dispatch a sniffed telemetry kind to its reader."""
    from src.ingest.telemetry import parse_dji_srt, parse_flight_csv

    try:
        if kind == "srt":
            return parse_dji_srt(path)
        if kind == "embedded_srt":
            return parse_embedded_srt(video)
        if kind == "gpmf":
            return parse_gpmf(video)
        if kind == "klv":
            from src.ingest.klv import parse_klv

            table = parse_klv(path or video)
            table.alignment = {"method": "video_clock", "note": "KLV multiplexed with the video"}
            return table
        if kind == "csv":
            return parse_flight_csv(path, column_map, headerless_order=headerless_order,
                                    video_duration_s=video_duration_s, recording=recording)
        if kind == "dji_flight_record":
            from src.ingest.dji_flight_record import parse_dji_flight_record

            table = parse_dji_flight_record(path, video_duration_s=video_duration_s, **dict(flight_record or {}))
            table.alignment = {"method": "recording_flag", "note": "DJI flight record recording segment"}
            return table
        if kind == "gpx":
            return parse_gpx(path)
        if kind == "kml":
            return parse_kml(path)
        if kind in ("json", "geojson"):
            return parse_json(path, column_map, video_duration_s, recording)
        if kind == "ulog":
            return parse_ulog(path)
        if kind in ("dataflash", "tlog"):
            return parse_mavlink(path)
    except Exception as exc:  # noqa: BLE001 - one bad file is a downgrade, never a crash
        name = path.name if path else video.name
        log_downgrade(log, f"telemetry {name}", "next source", f"{type(exc).__name__}: {exc}")
        return TelemetryTable.empty(f"{name}: {kind} reader failed ({type(exc).__name__}: {exc})")
    return TelemetryTable.empty(f"no reader for telemetry kind {kind!r}")
