"""Telemetry parsing (spec §4.2).

Priority order, each level degrading into the next without crashing:

  1. DJI ``.SRT`` sidecar — per-frame GPS, altitude, gimbal attitude, focal length.
  2. Separate flight-log CSV/TXT — delimiter and column auto-detection, with a
     positional fallback for headerless numeric files (bare "lat, lon" dumps).
  3. EXIF GPS on extracted frames.
  4. Nothing at all — the pipeline still runs and produces a **scale-free**
     model, and says so loudly in the QA report. This path must not crash.

The SRT format varies materially by firmware, so parsing is deliberately
token-based rather than template-based: strip markup, harvest every
``key : value`` pair and every known bare pattern from the block, then map
whatever was found onto canonical column names. A firmware that spells relative
altitude ``rel_alt`` and one that writes ``H 50.0m`` both land in ``alt_baro``,
and a firmware that reports a field nobody has seen before is ignored rather
than fatal.

Output schema (one row per video timestamp), written to ``telemetry.parquet``:

    t, lat, lon, alt_gps, alt_baro, roll, pitch, yaw, focal_mm, valid_flags
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_downgrade, log_event

log = get_logger(__name__)

SCHEMA_COLUMNS = ["t", "lat", "lon", "alt_gps", "alt_baro", "roll", "pitch", "yaw", "focal_mm", "valid_flags"]

# Every telemetry source ``load_telemetry`` knows how to read, in the §4.2
# priority order. The Stage Lab offers these, so a source added here appears
# in the UI without the two lists drifting apart.
TELEMETRY_SOURCES = ("klv", "srt", "csv", "exif")


class TelemetryFlags(IntFlag):
    """Per-row validity bitmask carried through to the QA report."""

    NONE = 0
    HAS_GPS = 1 << 0          # lat/lon present
    HAS_ALT_GPS = 1 << 1      # absolute (ellipsoidal) altitude present
    HAS_BARO = 1 << 2         # barometric / relative altitude present
    HAS_ATTITUDE = 1 << 3     # at least one of roll/pitch/yaw present
    HAS_FOCAL = 1 << 4        # focal length present -> intrinsics prior
    HAS_RTK = 1 << 5          # RTK/PPK fix -> tighten GPS weight (§5.6)
    GPS_OUTLIER = 1 << 6      # rejected by the §5.6 motion-envelope filter
    INTERPOLATED = 1 << 7     # value filled between measured rows
    SMOOTHED = 1 << 8         # value replaced by the Kalman estimate


# --------------------------------------------------------------------------
# Canonical field aliases. Extend this table when a new firmware shows up;
# never branch on firmware version in the parsing code.
# --------------------------------------------------------------------------
SRT_ALIASES: dict[str, str] = {
    "latitude": "lat",
    "lat": "lat",
    "gps_lat": "lat",
    "osd_lat": "lat",
    "longitude": "lon",
    "long": "lon",
    "lon": "lon",
    "gps_long": "lon",
    "gps_lon": "lon",
    "osd_lon": "lon",
    "altitude": "alt_gps",
    "abs_alt": "alt_gps",
    "absolute_altitude": "alt_gps",
    "gps_alt": "alt_gps",
    "osd_alt": "alt_gps",
    "rel_alt": "alt_baro",
    "relative_altitude": "alt_baro",
    "height": "alt_baro",
    "osd_relalt": "alt_baro",
    "baro_alt": "alt_baro",
    "gb_roll": "roll",
    "gimbal_roll": "roll",
    "roll": "roll",
    "gb_pitch": "pitch",
    "gimbal_pitch": "pitch",
    "pitch": "pitch",
    "gb_yaw": "yaw",
    "gimbal_yaw": "yaw",
    "yaw": "yaw",
    "heading": "yaw",
    "focal_len": "focal_mm",
    "focallength": "focal_mm",
    "focal_length": "focal_mm",
    "satellites": "sats",
    "sat": "sats",
    "gps_sats": "sats",
}

# Keys whose mere presence indicates an RTK/PPK-corrected fix.
RTK_MARKERS = {"rtk", "rtk_flag", "rtkflag", "ppk", "rtk_status"}

_TAG_RE = re.compile(r"<[^>]+>")
_CUE_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)
# Key must start with a letter, so the wall-clock line "10:21:31.041" cannot be
# mistaken for a key/value pair.
_KV_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*([^\s,\]\[]+)")
# "GPS (lon, lat, sats)" — note DJI's longitude-first ordering in this variant.
_GPS_TUPLE_RE = re.compile(r"GPS\s*[\(\[]\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)")
# Bare "H 50.0m" (height above takeoff) / "D 12.3m" (distance from home).
_HEIGHT_RE = re.compile(r"(?<![A-Za-z])H\s+([-+]?\d+\.?\d*)\s*m", re.IGNORECASE)

# DJI firmwares below a certain vintage report focal length and aperture in
# tenths (240 -> 24.0 mm). No drone camera has an 80 mm+ native focal length,
# so a value above this threshold is a units artefact, not a long lens.
FOCAL_TENTHS_THRESHOLD = 80.0


@dataclass
class TelemetryTable:
    """Parsed telemetry plus the provenance the QA report needs."""

    frame: pd.DataFrame
    source: str = "none"
    notes: list[str] = field(default_factory=list)
    wall_time: pd.Series | None = None

    # -- Construction -------------------------------------------------------
    @classmethod
    def empty(cls, reason: str) -> "TelemetryTable":
        """The no-telemetry path: a valid, empty table that never crashes."""
        frame = pd.DataFrame({c: pd.Series(dtype="float64") for c in SCHEMA_COLUMNS})
        frame["valid_flags"] = frame["valid_flags"].astype("int64")
        return cls(frame=frame, source="none", notes=[reason])

    @classmethod
    def from_records(cls, records: Sequence[Mapping[str, Any]], source: str) -> "TelemetryTable":
        if not records:
            return cls.empty(f"{source} produced no rows")
        frame = pd.DataFrame(list(records))
        for column in SCHEMA_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        frame = frame.sort_values("t").reset_index(drop=True)
        frame["valid_flags"] = _compute_flags(frame)
        wall = frame["wall_time"] if "wall_time" in frame else None
        ordered = SCHEMA_COLUMNS + [c for c in frame.columns if c not in SCHEMA_COLUMNS]
        return cls(frame=frame[ordered], source=source, wall_time=wall)

    # -- Properties ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame)

    @property
    def is_empty(self) -> bool:
        return len(self.frame) == 0

    @property
    def has_gps(self) -> bool:
        return not self.is_empty and bool(self.frame["lat"].notna().any() and self.frame["lon"].notna().any())

    @property
    def has_baro(self) -> bool:
        return not self.is_empty and bool(self.frame["alt_baro"].notna().any())

    @property
    def has_attitude(self) -> bool:
        return not self.is_empty and bool(
            self.frame[["roll", "pitch", "yaw"]].notna().any().any()
        )

    @property
    def has_focal(self) -> bool:
        return not self.is_empty and bool(self.frame["focal_mm"].notna().any())

    @property
    def has_rtk(self) -> bool:
        if self.is_empty:
            return False
        return bool((self.frame["valid_flags"].to_numpy() & int(TelemetryFlags.HAS_RTK)).any())

    @property
    def scale_free(self) -> bool:
        """True when nothing here can give the reconstruction a metric scale."""
        return not self.has_gps

    @property
    def duration_s(self) -> float:
        if self.is_empty:
            return 0.0
        return float(self.frame["t"].iloc[-1] - self.frame["t"].iloc[0])

    # -- Queries ------------------------------------------------------------
    def at_times(self, times: Iterable[float]) -> pd.DataFrame:
        """Interpolate telemetry onto arbitrary video timestamps.

        Linear on position and altitude; angles are interpolated on the unit
        circle so a heading crossing 360 degrees does not swing the long way
        round. Rows produced here are flagged INTERPOLATED.
        """
        times = np.asarray(list(times), dtype=float)
        if self.is_empty:
            out = pd.DataFrame({c: np.full(times.shape, np.nan) for c in SCHEMA_COLUMNS})
            out["t"] = times
            out["valid_flags"] = int(TelemetryFlags.NONE)
            return out

        source_t = self.frame["t"].to_numpy(dtype=float)
        out: dict[str, np.ndarray] = {"t": times}
        for column in ("lat", "lon", "alt_gps", "alt_baro", "focal_mm"):
            out[column] = _interp_masked(times, source_t, self.frame[column].to_numpy(dtype=float))
        for column in ("roll", "pitch", "yaw"):
            out[column] = _interp_angles(times, source_t, self.frame[column].to_numpy(dtype=float))

        frame = pd.DataFrame(out)
        flags = _compute_flags(frame)
        # Mark anything that did not land exactly on a measured row.
        measured = np.isin(times, source_t)
        flags = np.where(measured, flags, flags | int(TelemetryFlags.INTERPOLATED))
        if self.has_rtk:
            flags = flags | int(TelemetryFlags.HAS_RTK)
        frame["valid_flags"] = flags.astype("int64")
        return frame[SCHEMA_COLUMNS]

    def summary(self) -> dict[str, Any]:
        """Everything the QA report says about the telemetry we were given."""
        info: dict[str, Any] = {
            "source": self.source,
            "rows": len(self.frame),
            "duration_s": round(self.duration_s, 2),
            "has_gps": self.has_gps,
            "has_baro": self.has_baro,
            "has_attitude": self.has_attitude,
            "has_focal": self.has_focal,
            "has_rtk": self.has_rtk,
            "scale_free": self.scale_free,
            "notes": list(self.notes),
        }
        if self.has_gps:
            info["bounds"] = {
                "lat_min": float(self.frame["lat"].min()),
                "lat_max": float(self.frame["lat"].max()),
                "lon_min": float(self.frame["lon"].min()),
                "lon_max": float(self.frame["lon"].max()),
            }
            if self.frame["alt_gps"].notna().any():
                info["alt_gps_range"] = [
                    float(self.frame["alt_gps"].min()),
                    float(self.frame["alt_gps"].max()),
                ]
        return info

    # -- Persistence --------------------------------------------------------
    def to_parquet(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.frame.copy()
        frame.attrs["source"] = self.source
        frame.attrs["notes"] = self.notes
        frame.to_parquet(path, index=False)
        return path

    @classmethod
    def from_parquet(cls, path: Path | str) -> "TelemetryTable":
        frame = pd.read_parquet(path)
        return cls(
            frame=frame,
            source=str(frame.attrs.get("source", "parquet")),
            notes=list(frame.attrs.get("notes", [])),
        )

    def merged_with(self, other: "TelemetryTable") -> "TelemetryTable":
        """Fill this table's missing columns from ``other``.

        Used when, say, an SRT carries GPS but a separate CSV carries the
        barometric altitude. The primary source always wins where both have a
        value; the secondary only fills gaps.
        """
        if other.is_empty:
            return self
        if self.is_empty:
            return other
        filled = other.at_times(self.frame["t"].to_numpy(dtype=float))
        frame = self.frame.copy()
        added: list[str] = []
        for column in ("lat", "lon", "alt_gps", "alt_baro", "roll", "pitch", "yaw", "focal_mm"):
            missing = frame[column].isna()
            if missing.any() and filled[column].notna().any():
                frame.loc[missing, column] = filled.loc[missing, column]
                added.append(column)
        frame["valid_flags"] = _compute_flags(frame)
        notes = list(self.notes)
        if added:
            notes.append(f"filled {', '.join(sorted(set(added)))} from {other.source}")
        return TelemetryTable(frame=frame, source=f"{self.source}+{other.source}", notes=notes)


# --------------------------------------------------------------------------
# DJI SRT
# --------------------------------------------------------------------------
def parse_dji_srt(path: Path | str) -> TelemetryTable:
    """Parse a DJI SRT sidecar into canonical telemetry rows.

    Tolerant by construction: unknown keys are ignored, missing keys are NaN,
    and a block that yields nothing usable is skipped rather than fatal.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    unknown_keys: set[str] = set()
    blocks = _split_srt_blocks(text)

    for block in blocks:
        cue = _CUE_RE.search(block)
        if cue is None:
            continue
        start_s = _cue_seconds(cue, offset=0)
        payload = _TAG_RE.sub(" ", block[cue.end():])
        row: dict[str, Any] = {"t": start_s}

        # Bracketed / colon-separated key-value pairs, any firmware spelling.
        for key, raw in _KV_RE.findall(payload):
            canonical = SRT_ALIASES.get(key.lower())
            if canonical is None:
                if key.lower() in RTK_MARKERS:
                    row["rtk"] = True
                else:
                    unknown_keys.add(key.lower())
                continue
            value = _to_float(raw)
            if value is not None:
                row[canonical] = value

        # "GPS (lon, lat, sats)" variant.
        gps = _GPS_TUPLE_RE.search(payload)
        if gps:
            row.setdefault("lon", float(gps.group(1)))
            row.setdefault("lat", float(gps.group(2)))
            row.setdefault("sats", float(gps.group(3)))

        # Bare "H 50.0m" height-above-takeoff variant.
        if "alt_baro" not in row:
            height = _HEIGHT_RE.search(payload)
            if height:
                row["alt_baro"] = float(height.group(1))

        wall = _parse_wall_time(payload)
        if wall is not None:
            row["wall_time"] = wall

        if len(row) > 1:
            records.append(row)

    if not records:
        return TelemetryTable.empty(f"{path.name} parsed to zero usable rows")

    table = TelemetryTable.from_records(records, source=f"srt:{path.name}")
    table = _normalize_units(table)
    if unknown_keys:
        note = f"ignored unrecognised SRT keys: {', '.join(sorted(unknown_keys))}"
        table.notes.append(note)
        log_event(log, logging.DEBUG, note, path=str(path))
    log_event(
        log,
        logging.INFO,
        f"parsed {len(table)} telemetry rows from {path.name}",
        source="srt",
        rows=len(table),
        has_gps=table.has_gps,
        has_baro=table.has_baro,
        has_attitude=table.has_attitude,
    )
    return table


def _split_srt_blocks(text: str) -> list[str]:
    """Split on blank lines, tolerating CRLF and a missing trailing newline."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]
    if len(blocks) <= 1 and _CUE_RE.search(normalized):
        # Some firmwares emit no blank lines between cues; split before each
        # subtitle index that precedes a cue line instead.
        blocks = [b.strip() for b in re.split(r"\n(?=\d+\s*\n\d{1,2}:\d{2}:\d{2})", normalized) if b.strip()]
    return blocks


def _cue_seconds(match: re.Match[str], offset: int = 0) -> float:
    hours, minutes, seconds, millis = (match.group(1 + offset + i) for i in range(4))
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000.0


def _parse_wall_time(payload: str) -> pd.Timestamp | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:[.,](\d+))?", payload)
    if not match:
        return None
    fractional = (match.group(3) or "0")[:6].ljust(6, "0")
    try:
        return pd.Timestamp(f"{match.group(1)} {match.group(2)}.{fractional}")
    except ValueError:
        return None


def _normalize_units(table: TelemetryTable) -> TelemetryTable:
    """Undo DJI's tenths-of-a-unit reporting for focal length."""
    frame = table.frame
    if "focal_mm" in frame and frame["focal_mm"].notna().any():
        tenths = frame["focal_mm"] > FOCAL_TENTHS_THRESHOLD
        if tenths.any():
            frame.loc[tenths, "focal_mm"] = frame.loc[tenths, "focal_mm"] / 10.0
            table.notes.append("focal length reported in tenths of a mm; divided by 10")
    if "rtk" in frame:
        rtk_rows = frame["rtk"].fillna(False).astype(bool)
        if rtk_rows.any():
            frame.loc[rtk_rows, "valid_flags"] = (
                frame.loc[rtk_rows, "valid_flags"] | int(TelemetryFlags.HAS_RTK)
            )
            table.notes.append("RTK/PPK fixes detected; GPS weight will be tightened")
    return table


# --------------------------------------------------------------------------
# Flight-log CSV / TXT
# --------------------------------------------------------------------------
_CANONICAL_LOG_FIELDS = ("t", "lat", "lon", "alt_gps", "alt_baro", "roll", "pitch", "yaw", "focal_mm")


def parse_flight_csv(
    path: Path | str,
    column_map: Mapping[str, Sequence[str]],
    headerless_order: Sequence[str] = ("lat", "lon", "alt_gps"),
) -> TelemetryTable:
    """Parse a flight-log CSV/TXT, auto-detecting delimiter and columns.

    Covers the three shapes a flight-log sidecar shows up in: a comma header
    row (the common ground-station export), the same with a different
    delimiter (tab/space/semicolon, as hand-exported ``.txt`` GPS dumps often
    use), and a bare numeric file with no header at all — whose columns are
    then assigned positionally from ``headerless_order`` rather than guessed.
    """
    path = Path(path)
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except Exception as exc:  # noqa: BLE001 - a malformed log is a downgrade, not a crash
        log_downgrade(log, f"flight log {path.name}", "next telemetry source", f"{type(exc).__name__}: {exc}")
        return TelemetryTable.empty(f"could not read {path.name}: {exc}")

    headerless = _looks_headerless(frame.columns)
    if headerless:
        try:
            frame = pd.read_csv(path, sep=None, engine="python", header=None)
        except Exception as exc:  # noqa: BLE001
            log_downgrade(log, f"flight log {path.name}", "next telemetry source", f"{type(exc).__name__}: {exc}")
            return TelemetryTable.empty(f"could not read {path.name}: {exc}")
        frame.columns = [
            headerless_order[i] if i < len(headerless_order) else f"col{i}" for i in range(frame.shape[1])
        ]
        resolved = {c: c for c in frame.columns if c in _CANONICAL_LOG_FIELDS}
    else:
        resolved = resolve_csv_columns(frame.columns, column_map)

    if "lat" not in resolved or "lon" not in resolved:
        note = f"{path.name} has no recognisable latitude/longitude columns (saw {list(frame.columns)[:12]})"
        log_downgrade(log, f"flight log {path.name}", "next telemetry source", note)
        return TelemetryTable.empty(note)

    out = pd.DataFrame()
    out["t"] = _csv_time_column(frame, resolved.get("t"), path)
    for canonical in ("lat", "lon", "alt_gps", "alt_baro", "roll", "pitch", "yaw", "focal_mm"):
        if canonical in resolved:
            out[canonical] = pd.to_numeric(frame[resolved[canonical]], errors="coerce")

    records = out.to_dict("records")
    table = TelemetryTable.from_records(records, source=f"csv:{path.name}")
    if headerless:
        table.notes.append(f"no header row detected; columns assigned positionally as {list(resolved)}")
    else:
        table.notes.append(f"columns resolved: {resolved}")
    log_event(log, logging.INFO, f"parsed {len(table)} telemetry rows from {path.name}",
              source="csv", rows=len(table), resolved=resolved, headerless=headerless)
    return table


def _looks_headerless(columns: Iterable[Any]) -> bool:
    """True when the "header" row is actually numeric data, not field names.

    A bare GPS dump (``12.9716,77.5946`` per line, no column names) parses its
    first row as a header of two floats; treat a majority-numeric header as a
    sign there is no header at all, so it can be re-read positionally.
    """
    values = list(columns)
    if not values:
        return False
    numeric = sum(1 for v in values if _to_float(str(v)) is not None)
    return numeric / len(values) >= 0.6


def _csv_time_column(frame: pd.DataFrame, column: str | None, path: Path) -> pd.Series:
    """Return elapsed seconds from whatever the log calls time."""
    if column is None:
        rate_note = f"{path.name} has no time column; assuming rows are evenly spaced at 1 Hz"
        log_event(log, logging.WARNING, rate_note, path=str(path))
        return pd.Series(np.arange(len(frame), dtype=float))

    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().mean() > 0.9:
        values = numeric.astype(float)
        # Heuristic: a "time" column whose span exceeds a day of seconds is
        # almost certainly milliseconds since boot or an epoch stamp.
        span = float(values.max() - values.min())
        if span > 86400:
            values = values / 1000.0
        return values - values.min()

    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.notna().mean() > 0.5:
        return (parsed - parsed.min()).dt.total_seconds().astype(float)

    log_event(log, logging.WARNING, f"could not interpret time column {column!r}; using row index",
              path=str(path))
    return pd.Series(np.arange(len(frame), dtype=float))


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Substring matching below this length is too loose to be safe — "alt" would
# claim any header containing those three letters in sequence.
_MIN_SUBSTRING_SPELLING = 4


def resolve_csv_columns(
    columns: Iterable[str], column_map: Mapping[str, Sequence[str]]
) -> dict[str, str]:
    """Map canonical field names onto a flight log's actual headers.

    Flight logs name the same quantity a dozen ways — ``OSD.latitude``,
    ``GPS(lat)``, ``latitude_deg`` — so matching happens in two passes over
    punctuation-stripped, lowercased headers:

      1. **Exact match** against a configured spelling.
      2. **Substring match** for anything still unresolved, taking the
         *shortest* matching header, since the shortest one contains the least
         extra qualification and is therefore the most likely to be the plain
         field rather than a derived variant.

    Headers are claimed as they are matched, so ``OSD.altitude [m]`` cannot be
    assigned to both ``alt_gps`` and ``alt_baro``; the canonical fields are
    resolved in the order the config declares them, which makes the outcome
    deterministic and reviewable. The resolution is logged, because a silently
    mis-mapped altitude column is a metric-accuracy bug that would otherwise
    surface only as a strange reconstruction.
    """
    lookup: dict[str, str] = {}
    for column in columns:
        lookup.setdefault(_normalize_header(column), str(column))

    resolved: dict[str, str] = {}
    claimed: set[str] = set()

    for canonical, spellings in column_map.items():
        for spelling in spellings:
            key = _normalize_header(spelling)
            if key in lookup and lookup[key] not in claimed:
                resolved[canonical] = lookup[key]
                claimed.add(lookup[key])
                break

    for canonical, spellings in column_map.items():
        if canonical in resolved:
            continue
        candidates: list[tuple[int, str]] = []
        for spelling in spellings:
            key = _normalize_header(spelling)
            if len(key) < _MIN_SUBSTRING_SPELLING:
                continue
            for normalized, original in lookup.items():
                if original in claimed:
                    continue
                if key in normalized:
                    candidates.append((len(normalized), original))
        if candidates:
            chosen = min(candidates)[1]
            resolved[canonical] = chosen
            claimed.add(chosen)

    return resolved


# --------------------------------------------------------------------------
# EXIF on extracted frames
# --------------------------------------------------------------------------
def parse_exif(paths: Sequence[Path | str], timestamps: Sequence[float] | None = None) -> TelemetryTable:
    """Read GPS EXIF from extracted frames — the last structured fallback."""
    try:
        from PIL import ExifTags, Image
    except ImportError:
        log_downgrade(log, "EXIF telemetry", "no telemetry", "Pillow is not installed")
        return TelemetryTable.empty("Pillow unavailable; cannot read EXIF")

    gps_tag = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), None)
    records: list[dict[str, Any]] = []
    for i, image_path in enumerate(paths):
        try:
            with Image.open(image_path) as image:
                exif = image.getexif()
                gps = exif.get_ifd(gps_tag) if gps_tag else None
        except Exception:  # noqa: BLE001 - a frame without EXIF is normal
            continue
        if not gps:
            continue
        lat = _exif_coordinate(gps.get(2), gps.get(1))
        lon = _exif_coordinate(gps.get(4), gps.get(3))
        if lat is None or lon is None:
            continue
        row: dict[str, Any] = {
            "t": float(timestamps[i]) if timestamps is not None and i < len(timestamps) else float(i),
            "lat": lat,
            "lon": lon,
        }
        altitude = gps.get(6)
        if altitude is not None:
            value = float(altitude)
            row["alt_gps"] = -value if gps.get(5) in (1, b"\x01") else value
        records.append(row)

    if not records:
        return TelemetryTable.empty("no GPS EXIF found on extracted frames")
    table = TelemetryTable.from_records(records, source="exif")
    log_event(log, logging.INFO, f"parsed {len(table)} telemetry rows from EXIF", source="exif", rows=len(table))
    return table


def _exif_coordinate(dms: Any, ref: Any) -> float | None:
    """Convert EXIF degrees/minutes/seconds plus a hemisphere ref to decimal."""
    if not dms:
        return None
    try:
        degrees, minutes, seconds = (float(v) for v in dms)
    except (TypeError, ValueError):
        return None
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode("ascii", "ignore")
    if str(ref).upper() in ("S", "W"):
        value = -value
    return value


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def find_sidecar(video_path: Path, suffixes: Sequence[str]) -> Path | None:
    """Find a sidecar next to the video, case-insensitively.

    DJI writes ``DJI_0001.SRT`` next to ``DJI_0001.MP4``; some ground stations
    write ``DJI_0001.mp4.srt``. Both are found here.
    """
    for suffix in suffixes:
        for candidate in (
            video_path.with_suffix(suffix),
            video_path.with_suffix(suffix.upper()),
            video_path.parent / (video_path.name + suffix),
        ):
            if candidate.is_file():
                return candidate
    # Fall back to a case-insensitive directory scan.
    wanted = {s.lower().lstrip(".") for s in suffixes}
    stem = video_path.stem.lower()
    for candidate in video_path.parent.iterdir():
        if candidate.is_file() and candidate.suffix.lower().lstrip(".") in wanted:
            if candidate.stem.lower() in (stem, video_path.name.lower()):
                return candidate
    return None


def load_telemetry(
    video_path: Path | str,
    cfg: Any,
    srt_path: Path | str | None = None,
    csv_path: Path | str | None = None,
    frame_paths: Sequence[Path] | None = None,
    frame_timestamps: Sequence[float] | None = None,
    video_duration_s: float | None = None,
) -> TelemetryTable:
    """Load telemetry by the §4.2 priority order, filling gaps from lower tiers.

    The first source that yields GPS becomes primary; later sources only fill
    columns the primary is missing. When every source comes up empty the result
    is an empty table with ``scale_free`` set — a supported operating mode, not
    an error.
    """
    video_path = Path(video_path)
    sources = list(cfg.get_path("ingest.telemetry.sources", ["klv", "srt", "csv", "exif"]))
    column_map = cfg.get_path("ingest.telemetry.csv_column_map", {})
    if hasattr(column_map, "to_dict"):
        column_map = column_map.to_dict()
    headerless_order = list(cfg.get_path("ingest.telemetry.headerless_column_order", ["lat", "lon", "alt_gps"]))

    primary: TelemetryTable | None = None
    for source in sources:
        table: TelemetryTable | None = None
        if source == "srt":
            path = Path(srt_path) if srt_path else find_sidecar(video_path, [".srt"])
            if path is None:
                log_event(log, logging.INFO, "no SRT sidecar found", video=video_path.name)
            else:
                table = parse_dji_srt(path)
        elif source == "csv":
            path = Path(csv_path) if csv_path else find_sidecar(video_path, [".csv", ".txt"])
            if path is None:
                log_event(log, logging.INFO, "no flight-log CSV/TXT found", video=video_path.name)
            else:
                # Imported here: dji_flight_record builds on TelemetryTable from this module.
                from src.ingest.dji_flight_record import is_dji_flight_record, parse_dji_flight_record

                if is_dji_flight_record(path):
                    settings = cfg.get_path("ingest.telemetry.flight_record")
                    settings = settings.to_dict() if hasattr(settings, "to_dict") else dict(settings)
                    table = parse_dji_flight_record(path, video_duration_s=video_duration_s, **settings)
                else:
                    table = parse_flight_csv(path, column_map, headerless_order=headerless_order)
        elif source == "klv":
            # Imported here: klv builds on TelemetryTable from this module.
            from src.ingest.klv import load_klv_for_video

            settings = cfg.get_path("ingest.telemetry.klv", {})
            settings = settings.to_dict() if hasattr(settings, "to_dict") else dict(settings)
            table = load_klv_for_video(video_path, settings)
            if table is None:
                log_event(log, logging.INFO, "no KLV/STANAG 4609 metadata for this video",
                          video=video_path.name)
        elif source == "exif":
            if frame_paths:
                table = parse_exif(frame_paths, frame_timestamps)
        else:
            log_event(log, logging.WARNING, f"unknown telemetry source {source!r} in config; ignoring")

        if table is None or table.is_empty:
            continue
        if primary is None:
            primary = table
        else:
            primary = primary.merged_with(table)
        if primary.has_gps and primary.has_baro and primary.has_attitude:
            break  # Nothing left for a lower-priority source to add.

    if primary is None or primary.is_empty:
        log_event(
            log,
            logging.WARNING,
            "NO TELEMETRY FOUND — reconstruction will be scale-free and not georeferenced",
            event="downgrade",
            component="telemetry",
            fallback="scale-free reconstruction",
            reason="no KLV, SRT, CSV or EXIF source yielded usable rows",
        )
        return TelemetryTable.empty("no telemetry source available; model is scale-free")

    if not primary.has_gps:
        log_downgrade(log, "GPS", "scale-free reconstruction", "telemetry had no usable lat/lon")
    if not primary.has_baro:
        log_downgrade(log, "barometric altitude", "GPS altitude only",
                      "no barometric/relative altitude column in telemetry")
    if not primary.has_focal:
        log_downgrade(log, "focal length prior", "self-calibrated intrinsics",
                      "telemetry carries no focal length")
    return primary


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _compute_flags(frame: pd.DataFrame) -> np.ndarray:
    """Derive the per-row validity bitmask from which columns are populated."""
    n = len(frame)
    flags = np.zeros(n, dtype="int64")
    have = lambda c: frame[c].notna().to_numpy() if c in frame else np.zeros(n, dtype=bool)  # noqa: E731
    flags |= (have("lat") & have("lon")) * int(TelemetryFlags.HAS_GPS)
    flags |= have("alt_gps") * int(TelemetryFlags.HAS_ALT_GPS)
    flags |= have("alt_baro") * int(TelemetryFlags.HAS_BARO)
    flags |= (have("roll") | have("pitch") | have("yaw")) * int(TelemetryFlags.HAS_ATTITUDE)
    flags |= have("focal_mm") * int(TelemetryFlags.HAS_FOCAL)
    if "valid_flags" in frame:
        existing = pd.to_numeric(frame["valid_flags"], errors="coerce").fillna(0).to_numpy(dtype="int64")
        # Preserve sticky bits set elsewhere (RTK, outlier, smoothed).
        sticky = int(TelemetryFlags.HAS_RTK | TelemetryFlags.GPS_OUTLIER | TelemetryFlags.SMOOTHED)
        flags |= existing & sticky
    return flags


def _interp_masked(targets: np.ndarray, source_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation that ignores NaNs and does not extrapolate."""
    mask = np.isfinite(values)
    if mask.sum() < 1:
        return np.full(targets.shape, np.nan)
    if mask.sum() == 1:
        out = np.full(targets.shape, np.nan)
        out[:] = values[mask][0]
        return out
    out = np.interp(targets, source_t[mask], values[mask], left=np.nan, right=np.nan)
    # np.interp clamps rather than extrapolating; restore NaN outside the range
    # so callers can tell "no measurement here" from "measured at the edge".
    out[targets < source_t[mask][0]] = np.nan
    out[targets > source_t[mask][-1]] = np.nan
    return out


def _interp_angles(targets: np.ndarray, source_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Interpolate degrees on the unit circle, so 359 -> 1 goes the short way."""
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return _interp_masked(targets, source_t, values)
    radians = np.deg2rad(values[mask])
    sin = np.interp(targets, source_t[mask], np.sin(radians), left=np.nan, right=np.nan)
    cos = np.interp(targets, source_t[mask], np.cos(radians), left=np.nan, right=np.nan)
    out = np.rad2deg(np.arctan2(sin, cos))
    out[targets < source_t[mask][0]] = np.nan
    out[targets > source_t[mask][-1]] = np.nan
    # Keep the output in the same convention as the input (-180..180 vs 0..360).
    if np.nanmin(values) >= 0.0 and np.nanmax(values) > 180.0:
        out = np.mod(out, 360.0)
    return out


def enu_from_geodetic(
    lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, origin: tuple[float, float, float] | None = None
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Convert geodetic coordinates to a local ENU frame in metres.

    Used for velocity checks in §5.6 and for the similarity fit in §8.1. The
    origin defaults to the first finite fix, which keeps coordinates small and
    numerically well behaved.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    alt = np.asarray(alt, dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    if origin is None:
        if not finite.any():
            raise ValueError("no finite GPS fixes to establish an ENU origin")
        first = int(np.argmax(finite))
        origin = (float(lat[first]), float(lon[first]), float(alt[first]) if np.isfinite(alt[first]) else 0.0)

    lat0, lon0, alt0 = origin
    # WGS-84 radii of curvature at the origin latitude.
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)
    sin_lat0 = math.sin(math.radians(lat0))
    meridional = a * (1 - e2) / (1 - e2 * sin_lat0**2) ** 1.5
    transverse = a / math.sqrt(1 - e2 * sin_lat0**2)

    east = np.radians(lon - lon0) * transverse * math.cos(math.radians(lat0))
    north = np.radians(lat - lat0) * meridional
    up = np.where(np.isfinite(alt), alt - alt0, np.nan)
    return np.column_stack([east, north, up]), origin
