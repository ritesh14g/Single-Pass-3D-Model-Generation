"""DJI binary flight records (``DJIFlightRecord_*.txt``) — spec §4.2 tier 2.

DJI GO 4 / DJI Pilot save flight logs with a ``.txt`` extension, but the file is
binary: a prefix, a stream of ``[type][length][payload][0xFF]`` records, then a
details block and an embedded JPEG thumbnail. From format v7 every payload is
XOR-scrambled with a key derived from its first byte and the record type; from
v13 it is also AES-encrypted with keys only DJI's servers issue, so v13+ is
reported as unreadable rather than guessed at.

The scrambling scheme and field layouts follow the MIT-licensed
``dji-log-parser`` (Rust) and its ``pydjirecord`` port. pydjirecord 1.3.0 is not
used directly: it always reads a 16-bit record length, while v8 logs use 8 bits,
so it decodes zero records from them. The length width is probed here instead.

The log's clock is flight time at ~10 Hz, not video time, and one flight can
contain several recordings. Camera records carry the recording flag, so the
recording segments are recovered and telemetry is re-based onto the one that
matches the video (``ingest.telemetry.flight_record`` in the config).
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_downgrade, log_event
from src.ingest.telemetry import TelemetryTable

log = get_logger(__name__)

PREFIX_SIZE = 100
OLD_PREFIX_SIZE = 12
V12_RECORDS_START = PREFIX_SIZE + 436
FIRST_SCRAMBLED_VERSION = 7
FIRST_ENCRYPTED_VERSION = 13
MAX_PLAUSIBLE_VERSION = 40
RECORD_END = 0xFF

RECORD_OSD = 1
RECORD_GIMBAL = 3
RECORD_CAMERA = 25
_WANTED_RECORDS = {RECORD_OSD, RECORD_GIMBAL, RECORD_CAMERA}

# OSD payload: lon f64 @0, lat f64 @8 (radians), height i16 @16 (dm above takeoff),
# GPS-valid bit 0x80 @33, GPS level bits 0x3C @34, satellites @36, fly time u16 @42 (ds).
_OSD_MIN_LENGTH = 44
_GIMBAL_MIN_LENGTH = 6
_SNIFF_BYTES = 4096
_FRAMING_PROBE_RECORDS = 4

_XOR_MAGIC = 0x123456789ABCDEF0
_MASK64 = (1 << 64) - 1
_CRC64_POLY = 0x95AC9329AC4BC9B5

SAMPLE_COLUMNS = [
    "fly_time", "lat", "lon", "height", "gimbal_pitch", "gimbal_roll", "gimbal_yaw",
    "recording", "gps_valid", "gps_level", "satellites",
]


# --------------------------------------------------------------------------
# Descrambling
# --------------------------------------------------------------------------
def _build_crc64_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC64_POLY if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC64_TABLE = _build_crc64_table()


def _crc64(seed: int, data: bytes) -> int:
    crc = seed & _MASK64
    for byte in data:
        crc = _CRC64_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc & _MASK64


@lru_cache(maxsize=None)
def xor_key(key_byte: int, record_type: int) -> bytes:
    """The 8-byte XOR key for a v7+ record, derived from its first payload byte."""
    seed = (key_byte + record_type) & 0xFF
    key_input = ((_XOR_MAGIC * key_byte) & _MASK64).to_bytes(8, "little")
    return _crc64(seed, key_input).to_bytes(8, "little")


def _xor(data: bytes, key: bytes) -> bytes:
    if not data:
        return b""
    tiled = np.resize(np.frombuffer(key, dtype=np.uint8), len(data))
    return (np.frombuffer(data, dtype=np.uint8) ^ tiled).tobytes()


def descramble(payload: bytes, record_type: int) -> bytes:
    """Undo v7+ scrambling. The first payload byte is the key seed, not data."""
    if not payload:
        return payload
    return _xor(payload[1:], xor_key(payload[0], record_type))


def scramble(plain: bytes, record_type: int, key_byte: int) -> bytes:
    """Inverse of :func:`descramble`, for writing test logs."""
    return bytes([key_byte]) + _xor(plain, xor_key(key_byte, record_type))


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FlightRecordHeader:
    version: int
    records_start: int
    records_end: int

    @property
    def scrambled(self) -> bool:
        return self.version >= FIRST_SCRAMBLED_VERSION

    @property
    def encrypted(self) -> bool:
        return self.version >= FIRST_ENCRYPTED_VERSION


def read_header(head: bytes, file_size: int) -> FlightRecordHeader | None:
    """Parse the prefix; ``None`` when these bytes cannot start a DJI flight record."""
    if file_size < PREFIX_SIZE or len(head) < PREFIX_SIZE:
        return None
    detail_offset = struct.unpack_from("<Q", head, 0)[0]
    version = head[10]
    if not 1 <= version <= MAX_PLAUSIBLE_VERSION or detail_offset > file_size:
        return None
    # The 100-byte prefix ends in a zero-filled reserved block; text never does.
    if version >= 6 and head[20:PREFIX_SIZE].count(0) < 60:
        return None

    if version < 6:
        start, end = OLD_PREFIX_SIZE, detail_offset
    elif version < 12:
        start, end = PREFIX_SIZE, detail_offset
    elif version == 12:
        start, end = V12_RECORDS_START, file_size
    else:
        start, end = detail_offset, file_size
    if start >= end:
        return None
    return FlightRecordHeader(version=version, records_start=start, records_end=end)


def _length_width(data: bytes, header: FlightRecordHeader) -> int | None:
    """Bytes in the record length field: 1 in the v8 logs measured, 2 in newer ones."""
    for width in (1, 2):
        pos = header.records_start
        for _ in range(_FRAMING_PROBE_RECORDS):
            body = pos + 1 + width
            stop = body + int.from_bytes(data[pos + 1:body], "little")
            if body > len(data) or stop >= len(data) or data[stop] != RECORD_END:
                break
            pos = stop + 1
        else:
            return width
    return None


def is_dji_flight_record(path: Path | str) -> bool:
    """Sniff whether a sidecar is a binary DJI flight record rather than a text log."""
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(_SNIFF_BYTES)
    except OSError:
        return False
    header = read_header(head, size)
    if header is None:
        return False
    if header.encrypted:
        return True
    return header.records_start < len(head) and _length_width(head, header) is not None


@dataclass
class FlightRecordLog:
    """Decoded OSD samples, each carrying the latest gimbal and camera state."""

    version: int
    samples: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def segments(self) -> list[tuple[float, float]]:
        """``(start, end)`` flight time of each continuous video recording."""
        if self.samples.empty:
            return []
        recording = self.samples["recording"].to_numpy(dtype=bool)
        fly = self.samples["fly_time"].to_numpy(dtype=float)
        edges = np.flatnonzero(np.diff(recording.astype(np.int8))) + 1
        bounds = np.r_[0, edges, len(recording)]
        return [(float(fly[s]), float(fly[e - 1])) for s, e in zip(bounds[:-1], bounds[1:]) if recording[s]]


def read_flight_record(path: Path | str) -> FlightRecordLog:
    path = Path(path)
    data = path.read_bytes()
    header = read_header(data, len(data))
    if header is None:
        raise ValueError(f"{path.name} is not a DJI flight record")
    empty = pd.DataFrame(columns=SAMPLE_COLUMNS)
    if header.encrypted:
        return FlightRecordLog(header.version, empty, [
            f"{path.name} is DJI log format v{header.version}, which is AES-encrypted with keys only DJI's "
            "servers issue; export it to CSV with a DJI log viewer (e.g. Airdata) and upload the CSV instead"
        ])
    width = _length_width(data, header)
    if width is None:
        return FlightRecordLog(header.version, empty, [
            f"{path.name}: record framing not recognised for DJI log format v{header.version}"
        ])

    rows: list[tuple[Any, ...]] = []
    gimbal = (math.nan, math.nan, math.nan)
    recording = False
    pos = header.records_start
    while pos + 1 + width <= header.records_end:
        record_type = data[pos]
        body = pos + 1 + width
        stop = body + int.from_bytes(data[pos + 1:body], "little")
        if stop >= len(data) or data[stop] != RECORD_END:
            break  # end of the record stream (v8 logs run straight into the thumbnail)
        if record_type in _WANTED_RECORDS:
            payload = data[body:stop]
            if header.scrambled:
                payload = descramble(payload, record_type)
            if record_type == RECORD_OSD and len(payload) >= _OSD_MIN_LENGTH:
                lon, lat = struct.unpack_from("<dd", payload, 0)
                rows.append((
                    struct.unpack_from("<H", payload, 42)[0] / 10.0,
                    math.degrees(lat),
                    math.degrees(lon),
                    struct.unpack_from("<h", payload, 16)[0] / 10.0,
                    *gimbal,
                    recording,
                    bool(payload[33] & 0x80),
                    (payload[34] & 0x3C) >> 2,
                    payload[36],
                ))
            elif record_type == RECORD_GIMBAL and len(payload) >= _GIMBAL_MIN_LENGTH:
                gimbal = tuple(v / 10.0 for v in struct.unpack_from("<hhh", payload, 0))
            elif record_type == RECORD_CAMERA and payload:
                recording = bool(payload[0] & 0xC0)
        pos = stop + 1

    samples = pd.DataFrame(rows, columns=SAMPLE_COLUMNS)
    log_event(log, logging.INFO, f"decoded {len(samples)} OSD samples from {path.name}",
              source="dji_log", version=header.version, stopped_at=pos, records_end=header.records_end)
    return FlightRecordLog(header.version, samples)


# --------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------
def parse_dji_flight_record(
    path: Path | str,
    *,
    video_duration_s: float | None,
    segment: int | str,
    offset_s: float | None,
    duration_tolerance_s: float,
    segment_padding_s: float,
) -> TelemetryTable:
    """Decode a flight record into canonical telemetry timed from the video's start."""
    path = Path(path)
    try:
        flight = read_flight_record(path)
    except (OSError, ValueError) as exc:
        log_downgrade(log, f"DJI flight record {path.name}", "next telemetry source", str(exc))
        return TelemetryTable.empty(f"could not read {path.name}: {exc}")
    if flight.samples.empty:
        reason = flight.notes[0] if flight.notes else f"{path.name} held no OSD samples"
        log_downgrade(log, f"DJI flight record {path.name}", "next telemetry source", reason)
        return TelemetryTable.empty(reason)

    start, end, notes = _video_window(flight, video_duration_s, segment, offset_s, duration_tolerance_s)
    fly = flight.samples["fly_time"].to_numpy(dtype=float)
    window = flight.samples[(fly >= start - segment_padding_s) & (fly <= end + segment_padding_s)]
    gps = window["gps_valid"].to_numpy(dtype=bool)
    frame = pd.DataFrame({
        "t": window["fly_time"].to_numpy(dtype=float) - start,
        "lat": np.where(gps, window["lat"].to_numpy(dtype=float), np.nan),
        "lon": np.where(gps, window["lon"].to_numpy(dtype=float), np.nan),
        "alt_baro": window["height"].to_numpy(dtype=float),
        "roll": window["gimbal_roll"].to_numpy(dtype=float),
        "pitch": window["gimbal_pitch"].to_numpy(dtype=float),
        "yaw": window["gimbal_yaw"].to_numpy(dtype=float),
    })

    table = TelemetryTable.from_records(frame.to_dict("records"), source=f"dji_log:{path.name}")
    table.notes.extend(notes)
    table.notes.append(
        f"DJI log format v{flight.version}: GPS sampled at ~10 Hz and interpolated onto video frames; "
        "alt_baro is height above takeoff, absolute altitude is not decoded"
    )
    log_event(log, logging.INFO, f"parsed {len(table)} telemetry rows from {path.name}",
              source="dji_log", rows=len(table), start_fly_time_s=start, has_gps=table.has_gps)
    return table


def _segment_index(segment: int | str) -> int | None:
    if isinstance(segment, bool):
        return None
    if isinstance(segment, int):
        return segment
    if isinstance(segment, str) and segment.strip().isdigit():
        return int(segment)
    return None


def _video_window(
    flight: FlightRecordLog,
    video_duration_s: float | None,
    segment: int | str,
    offset_s: float | None,
    tolerance_s: float,
) -> tuple[float, float, list[str]]:
    """Flight-time span the video covers, plus notes explaining how it was chosen."""
    fly = flight.samples["fly_time"]
    flight_start, flight_end = float(fly.iloc[0]), float(fly.iloc[-1])
    if offset_s is not None:
        start = float(offset_s)
        end = start + video_duration_s if video_duration_s else flight_end
        return start, end, [f"video start fixed at flight time {start:.1f} s by ingest.telemetry.flight_record.offset_s"]

    segments = flight.segments
    if not segments:
        note = ("log records no video recording; telemetry is timed from the start of the log — set "
                "ingest.telemetry.flight_record.offset_s if the video started later")
        log_event(log, logging.WARNING, note)
        return flight_start, flight_end, [note]

    listing = ", ".join(f"#{i} {s:.1f}-{e:.1f} s ({e - s:.1f} s)" for i, (s, e) in enumerate(segments))
    prefix = f"recordings in log: {listing}; "
    index = _segment_index(segment)
    if index is not None:
        if 0 <= index < len(segments):
            start, end = segments[index]
            return start, end, [prefix + f"using #{index} as configured"]
        prefix += f"configured recording #{index} does not exist; "

    durations = np.array([e - s for s, e in segments])
    if video_duration_s:
        best = int(np.argmin(np.abs(durations - video_duration_s)))
        gap = float(abs(durations[best] - video_duration_s))
        if gap <= tolerance_s:
            start, end = segments[best]
            # Recording on/off is logged once per second, so each edge is only known to ~1 s.
            return start, end, [prefix + f"video ({video_duration_s:.1f} s) matched #{best}; start known to about ±1 s"]
        mismatch = f"no recording matches the video duration ({video_duration_s:.1f} s, nearest is {gap:.1f} s off)"
    else:
        mismatch = "video duration unknown"

    longest = int(np.argmax(durations))
    start, end = segments[longest]
    warning = (prefix + f"{mismatch}, so using the longest (#{longest}). GPS may be misaligned with the video "
               "— set ingest.telemetry.flight_record.segment or offset_s")
    log_event(log, logging.WARNING, warning)
    return start, end, [warning]
