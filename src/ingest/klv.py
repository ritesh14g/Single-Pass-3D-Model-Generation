"""MISB ST 0601 KLV telemetry embedded in the video stream — spec §4.2.

Military-intelligence and disaster-response imagery is usually STANAG 4609:
an MPEG-2 transport stream carrying, alongside the video, a data PID whose
payload is KLV (Key-Length-Value) metadata in the MISB ST 0601 "UAS Datalink
Local Set". Unlike a DJI ``.SRT`` sidecar, this telemetry travels *inside* the
file, multiplexed against the video's own clock, so it cannot be lost, mismatched
or drift out of sync. That removes the whole class of alignment problem the
``flight_record`` settings exist to work around.

Three containers are accepted:

  * **MPEG-2 TS** (``.ts``/``.m2ts``/``.mpg``) — depacketized per PID here; no
    PAT/PMT walk is needed because the 16-byte universal key identifies the
    local sets unambiguously once the 4-byte TS headers are stripped.
  * **A raw KLV stream** (``.klv``) — what ``ffmpeg -map 0:d -c copy`` writes.
  * **MP4/MOV**, when its KLV samples happen to be contiguous in ``mdat``.
    Off by default (``scan_mp4``): a DJI MP4 never carries KLV, and scanning a
    4K file to prove that costs a full read on every run.

Neither ``ffmpeg`` nor ``klvdata`` is required. The parser is self-contained,
like ``dji_flight_record``, because the pipeline must not gain a binary
dependency to read a format the problem statement puts on the critical path.

Field scaling follows MISB ST 0601: each value is a fixed-point integer mapped
onto a documented range, and the all-ones negative (e.g. ``0x8000`` for an
int16) is the standard "out of range" marker, decoded to NaN rather than to a
plausible-looking angle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_event
from src.ingest.telemetry import TelemetryTable

log = get_logger(__name__)

# SMPTE 336M universal label for the ST 0601 UAS Datalink Local Set.
UAS_LS_KEY = bytes.fromhex("060E2B34020B01010E01030101000000")

TS_PACKET_SIZE = 188
TS_SYNC_BYTE = 0x47
# A TS PID is 13 bits; 0x1FFF is the null-packet PID and carries no payload.
TS_NULL_PID = 0x1FFF

CHECKSUM_TAG = 1
TIMESTAMP_TAG = 2

# Containers whose KLV is reachable by depacketizing or scanning directly.
TS_SUFFIXES = (".ts", ".m2ts", ".mts", ".mpg", ".mpeg")
RAW_SUFFIXES = (".klv", ".bin")
MP4_SUFFIXES = (".mp4", ".mov", ".m4v")


# --------------------------------------------------------------------------
# ST 0601 value decoding
# --------------------------------------------------------------------------
def _uint(data: bytes) -> int:
    return int.from_bytes(data, "big", signed=False)


def _sint(data: bytes) -> int:
    return int.from_bytes(data, "big", signed=True)


def _unsigned_range(data: bytes, low: float, high: float) -> float:
    """Map an unsigned fixed-point integer onto ``[low, high]``."""
    span = (1 << (8 * len(data))) - 1
    if span == 0:
        return float("nan")
    return low + _uint(data) * (high - low) / span


def _signed_range(data: bytes, limit: float) -> float:
    """Map a signed fixed-point integer onto ``[-limit, +limit]``.

    ST 0601 reserves the most negative value (``0x8000``, ``0x80000000``, …) to
    mean "value out of range"; it decodes to NaN so a missing reading never
    masquerades as a real one.
    """
    bits = 8 * len(data)
    raw = _sint(data)
    if raw == -(1 << (bits - 1)):
        return float("nan")
    return raw * limit / ((1 << (bits - 1)) - 1)


# tag -> (field name, decoder). Only the tags that carry geometry the pipeline
# can use are decoded; the rest are counted and skipped, so an unknown tag is
# never mistaken for a parse failure.
TAG_DECODERS: dict[int, tuple[str, Any]] = {
    2:  ("unix_us",           _uint),
    5:  ("yaw",               lambda d: _unsigned_range(d, 0.0, 360.0)),
    6:  ("pitch",             lambda d: _signed_range(d, 20.0)),
    7:  ("roll",              lambda d: _signed_range(d, 50.0)),
    13: ("lat",               lambda d: _signed_range(d, 90.0)),
    14: ("lon",               lambda d: _signed_range(d, 180.0)),
    15: ("alt_gps",           lambda d: _unsigned_range(d, -900.0, 19000.0)),
    16: ("hfov_deg",          lambda d: _unsigned_range(d, 0.0, 180.0)),
    17: ("vfov_deg",          lambda d: _unsigned_range(d, 0.0, 180.0)),
    18: ("sensor_rel_az_deg", lambda d: _unsigned_range(d, 0.0, 360.0)),
    19: ("sensor_rel_el_deg", lambda d: _signed_range(d, 180.0)),
    20: ("sensor_rel_roll_deg", lambda d: _unsigned_range(d, 0.0, 360.0)),
    21: ("slant_range_m",     lambda d: _unsigned_range(d, 0.0, 5_000_000.0)),
    23: ("frame_center_lat",  lambda d: _signed_range(d, 90.0)),
    24: ("frame_center_lon",  lambda d: _signed_range(d, 180.0)),
    25: ("frame_center_alt",  lambda d: _unsigned_range(d, -900.0, 19000.0)),
    65: ("st0601_version",    _uint),
}

# Columns carried beyond the canonical telemetry schema. Frame centre and slant
# range are georeferencing priors (§8); the sensor-relative angles are the
# gimbal pointing Stage 4 needs to separate camera attitude from airframe
# attitude.
EXTRA_COLUMNS = (
    "frame_center_lat", "frame_center_lon", "frame_center_alt",
    "slant_range_m", "sensor_rel_az_deg", "sensor_rel_el_deg",
    "sensor_rel_roll_deg", "hfov_deg", "vfov_deg",
)


# --------------------------------------------------------------------------
# BER decoding
# --------------------------------------------------------------------------
def _read_ber_length(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a BER length. Returns ``(length, bytes_consumed)``."""
    if offset >= len(data):
        raise ValueError("truncated BER length")
    first = data[offset]
    if first < 0x80:
        return first, 1
    count = first & 0x7F
    if count == 0 or offset + 1 + count > len(data):
        # Indefinite form is not permitted in a local set.
        raise ValueError("unsupported or truncated BER long-form length")
    return _uint(data[offset + 1: offset + 1 + count]), 1 + count


def _read_ber_oid(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a BER-OID tag (7 bits per byte, MSB = continuation)."""
    value = 0
    consumed = 0
    while offset + consumed < len(data):
        byte = data[offset + consumed]
        value = (value << 7) | (byte & 0x7F)
        consumed += 1
        if not byte & 0x80:
            return value, consumed
    raise ValueError("truncated BER-OID tag")


def _checksum(local_set: bytes) -> int:
    """ST 0601 checksum (``bcc_16``) over the packet up to the checksum value.

    Not a plain byte sum: the standard's reference implementation adds each byte
    into alternating halves of a 16-bit accumulator — even indices into the high
    byte, odd into the low one::

        for i in range(len): bcc += buff[i] << (8 * ((i + 1) % 2))

    A plain sum matches zero real packets across all eight MISB sample clips, so
    getting this wrong rejects every stream when ``require_checksum`` is on.
    """
    bcc = 0
    for i, byte in enumerate(local_set):
        bcc += byte << (8 * ((i + 1) % 2))
    return bcc & 0xFFFF


# --------------------------------------------------------------------------
# Container handling
# --------------------------------------------------------------------------
def _looks_like_ts(data: bytes) -> bool:
    """Two sync bytes one packet apart is the standard TS probe."""
    return (
        len(data) >= TS_PACKET_SIZE * 2
        and data[0] == TS_SYNC_BYTE
        and data[TS_PACKET_SIZE] == TS_SYNC_BYTE
    )


def depacketize_ts(data: bytes) -> dict[int, bytes]:
    """Strip TS framing and return the payload bytes of each PID.

    PES headers are left in place: the universal key search below skips over
    them, and the local set's own BER length says where each packet ends, so
    parsing PES framing would add failure modes without adding information.
    """
    streams: dict[int, list[bytes]] = {}
    for start in range(0, len(data) - TS_PACKET_SIZE + 1, TS_PACKET_SIZE):
        packet = data[start:start + TS_PACKET_SIZE]
        if packet[0] != TS_SYNC_BYTE:
            continue                                    # lost sync; skip
        pid = ((packet[1] & 0x1F) << 8) | packet[2]
        if pid == TS_NULL_PID:
            continue
        adaptation = (packet[3] >> 4) & 0x3
        if adaptation in (0, 2):
            continue                                    # no payload
        offset = 4
        if adaptation == 3:
            offset += 1 + packet[4]                     # skip adaptation field
        if offset >= TS_PACKET_SIZE:
            continue
        streams.setdefault(pid, []).append(packet[offset:])
    return {pid: b"".join(chunks) for pid, chunks in streams.items()}


def iter_local_sets(data: bytes, max_packets: int) -> Iterator[tuple[bytes, bytes]]:
    """Yield ``(header_through_length, value_bytes)`` for each ST 0601 set."""
    offset = 0
    found = 0
    while found < max_packets:
        index = data.find(UAS_LS_KEY, offset)
        if index < 0:
            return
        value_start = index + len(UAS_LS_KEY)
        try:
            length, consumed = _read_ber_length(data, value_start)
        except ValueError:
            offset = index + 1
            continue
        payload_start = value_start + consumed
        payload_end = payload_start + length
        if length <= 0 or payload_end > len(data):
            offset = index + 1
            continue
        yield data[index:payload_start], data[payload_start:payload_end]
        found += 1
        offset = payload_end


# --------------------------------------------------------------------------
# Packet decoding
# --------------------------------------------------------------------------
@dataclass
class KlvStats:
    """What the decode saw, for the QA report and the downgrade log."""

    packets: int = 0
    checksum_ok: int = 0
    checksum_bad: int = 0
    unknown_tags: set[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unknown_tags is None:
            self.unknown_tags = set()


def decode_local_set(header: bytes, payload: bytes, stats: KlvStats) -> dict[str, Any]:
    """Decode one ST 0601 local set into canonical field names."""
    row: dict[str, Any] = {}
    offset = 0
    while offset < len(payload):
        try:
            tag, tag_len = _read_ber_oid(payload, offset)
            length, len_len = _read_ber_length(payload, offset + tag_len)
        except ValueError:
            break
        start = offset + tag_len + len_len
        end = start + length
        if end > len(payload):
            break
        value = payload[start:end]
        if tag == CHECKSUM_TAG and length == 2:
            # The checksum covers the key, the length and every byte before the
            # checksum's own value.
            expected = _checksum(header + payload[:start])
            if expected == _uint(value):
                stats.checksum_ok += 1
            else:
                stats.checksum_bad += 1
        elif tag in TAG_DECODERS:
            name, decoder = TAG_DECODERS[tag]
            if length:
                try:
                    row[name] = decoder(value)
                except Exception:                        # a malformed field is not fatal
                    stats.unknown_tags.add(tag)
        elif length:
            stats.unknown_tags.add(tag)
        offset = end
    return row


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def file_is_ts(path: Path, probe_bytes: int = TS_PACKET_SIZE + 1) -> bool:
    """Sniff for MPEG-2 TS framing: sync bytes exactly one packet apart.

    Extensions lie in this domain. Every one of the eight MISB sample clips is
    an MPEG-2 TS, but they are named ``.ts``, ``.mp4``, ``.mpeg4`` and ``.H264``.
    Gating on the suffix therefore skips real ISR footage, so the container is
    identified by reading 189 bytes instead.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(probe_bytes)
    except OSError:
        return False
    return (len(head) > TS_PACKET_SIZE
            and head[0] == TS_SYNC_BYTE
            and head[TS_PACKET_SIZE] == TS_SYNC_BYTE)


def find_klv_source(video_path: Path, sidecar_suffixes: Sequence[str]) -> Path | None:
    """The file to read KLV from: the video itself, or a sidecar beside it."""
    video_path = Path(video_path)
    if video_path.suffix.lower() in TS_SUFFIXES + RAW_SUFFIXES:
        return video_path
    if file_is_ts(video_path):
        return video_path
    for suffix in sidecar_suffixes:
        candidate = video_path.with_suffix(suffix if suffix.startswith(".") else f".{suffix}")
        if candidate.is_file() and candidate != video_path:
            return candidate
    return None


def parse_klv(
    path: Path | str,
    max_packets: int = 200_000,
    require_checksum: bool = False,
) -> TelemetryTable:
    """Parse MISB ST 0601 telemetry from a TS, raw KLV stream or MP4."""
    path = Path(path)
    data = path.read_bytes()

    if _looks_like_ts(data):
        streams = depacketize_ts(data)
        candidates = [(pid, payload) for pid, payload in streams.items()
                      if UAS_LS_KEY in payload]
        if not candidates:
            return TelemetryTable.empty(f"{path.name}: no ST 0601 KLV stream in any TS PID")
        pid, blob = max(candidates, key=lambda item: item[1].count(UAS_LS_KEY))
        log_event(log, logging.INFO, f"KLV data PID 0x{pid:04X} in {path.name}",
                  source="klv", pid=pid, container="mpegts")
    else:
        blob = data
        if UAS_LS_KEY not in blob:
            return TelemetryTable.empty(f"{path.name}: no ST 0601 universal key found")

    stats = KlvStats()
    rows: list[dict[str, Any]] = []
    for header, payload in iter_local_sets(blob, max_packets):
        stats.packets += 1
        row = decode_local_set(header, payload, stats)
        if row:
            rows.append(row)

    if require_checksum and stats.checksum_bad:
        rows = []                                        # refuse a stream we cannot trust

    rows = [r for r in rows if "unix_us" in r]
    if not rows:
        reason = (f"{path.name}: {stats.packets} KLV packets but none carried a "
                  "precision timestamp")
        if require_checksum and stats.checksum_bad:
            reason = f"{path.name}: {stats.checksum_bad} packets failed the ST 0601 checksum"
        return TelemetryTable.empty(reason)

    # The local sets are timestamped in absolute UTC microseconds. The canonical
    # schema wants seconds from the start of the recording, which is what the
    # first packet marks — the KLV is multiplexed against the video's own clock,
    # so no offset search is needed.
    origin = min(int(r["unix_us"]) for r in rows)
    records = []
    for row in rows:
        record = {k: v for k, v in row.items() if k not in ("unix_us", "st0601_version")}
        record["t"] = (int(row["unix_us"]) - origin) / 1e6
        record["wall_time"] = pd.to_datetime(int(row["unix_us"]), unit="us", utc=True)
        records.append(record)

    table = TelemetryTable.from_records(records, source=f"klv:{path.name}")
    versions = {int(r["st0601_version"]) for r in rows if "st0601_version" in r}
    if versions:
        table.notes.append(f"MISB ST 0601 version {'/'.join(str(v) for v in sorted(versions))}")
    if stats.checksum_bad:
        table.notes.append(
            f"{stats.checksum_bad} of {stats.packets} KLV packets failed the ST 0601 checksum")
    if stats.unknown_tags:
        table.notes.append(
            f"ignored {len(stats.unknown_tags)} unmapped ST 0601 tags: "
            + ", ".join(str(t) for t in sorted(stats.unknown_tags)[:12]))
    log_event(
        log, logging.INFO, f"parsed {len(table)} telemetry rows from {path.name}",
        source="klv", rows=len(table), packets=stats.packets,
        checksum_ok=stats.checksum_ok, checksum_bad=stats.checksum_bad,
        has_gps=table.has_gps, has_attitude=table.has_attitude,
    )
    return table


def load_klv_for_video(video_path: Path | str, settings: dict[str, Any]) -> TelemetryTable | None:
    """Resolve and parse the KLV source for a video, honouring the config.

    Returns ``None`` when there is nothing to read, which is the ordinary case
    for DJI footage and must not be logged as a failure.
    """
    video_path = Path(video_path)
    if not bool(settings.get("enabled", True)):
        return None

    suffix = video_path.suffix.lower()
    sidecars = [str(s) for s in settings.get("sidecar_suffixes", [".klv", ".ts"])]
    source = find_klv_source(video_path, sidecars)

    if source is None:
        # Only a genuine MP4 reaches here — a TS misnamed .mp4 was already
        # identified by its framing. Scanning a real MP4 costs a full file read
        # to prove a DJI clip has no KLV, so it stays opt-in.
        if suffix in MP4_SUFFIXES and bool(settings.get("scan_mp4", False)):
            source = video_path
        else:
            return None
    elif source == video_path and suffix not in TS_SUFFIXES + RAW_SUFFIXES:
        log_event(log, logging.INFO,
                  f"{video_path.name} is an MPEG-2 transport stream despite its {suffix} "
                  "extension; reading KLV from it",
                  source="klv", container="mpegts", suffix=suffix)

    return parse_klv(
        source,
        max_packets=int(settings.get("max_packets", 200_000)),
        require_checksum=bool(settings.get("require_checksum", False)),
    )
