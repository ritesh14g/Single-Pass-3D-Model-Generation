"""GPS and sensor-noise conditioning (spec §5.6, challenge v).

Consumer GPS on a moving UAV produces three distinct problems, and each gets
its own treatment here rather than one catch-all smoother:

  * **Gross outliers** — a fix that implies the drone teleported. Caught by a
    motion-envelope test against the platform's physical limits, then by a
    median filter over a sliding window, because a median is the only cheap
    estimator that a single wild value cannot move.
  * **Zero-mean noise** — caught by a constant-velocity Kalman filter. A drone
    flying a survey line is very close to constant velocity, so the motion
    model does real work here rather than merely smoothing.
  * **Vertical bias** — GPS altitude error runs 2-3x the horizontal. Where a
    barometer is present it has excellent *relative* precision and a drifting
    absolute reference, which is the exact complement of GPS. A complementary
    filter takes the high-frequency detail from the barometer and the
    low-frequency absolute reference from GPS.

RTK/PPK, when detected, is not merely "better GPS": it changes the weight the
whole pipeline should place on position priors, and it is the path to the
sub-metre accuracy target. Detection is automatic and the weight boost is
applied without asking (§5.6).

This module lives under ``condition/`` rather than ``geo/`` because it is
Stage 2 work in the spec — it conditions a noisy input signal, exactly like the
blur and artifact modules do for imagery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger, log_downgrade, log_event
from src.ingest.telemetry import TelemetryFlags, TelemetryTable, enu_from_geodetic

log = get_logger(__name__)


@dataclass
class GpsFilterReport:
    """What the filter did, for the QA report and the honest-limitations list."""

    input_fixes: int = 0
    envelope_outliers: int = 0
    median_outliers: int = 0
    smoothed_fixes: int = 0
    rtk_detected: bool = False
    gps_weight: float = 1.0
    altitude_source: str = "none"
    baro_gps_offset_m: float | None = None
    max_speed_observed: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_fixes": self.input_fixes,
            "envelope_outliers": self.envelope_outliers,
            "median_outliers": self.median_outliers,
            "smoothed_fixes": self.smoothed_fixes,
            "outlier_fraction": round(
                (self.envelope_outliers + self.median_outliers) / max(self.input_fixes, 1), 4
            ),
            "rtk_detected": self.rtk_detected,
            "gps_weight": self.gps_weight,
            "altitude_source": self.altitude_source,
            "baro_gps_offset_m": (
                round(self.baro_gps_offset_m, 3) if self.baro_gps_offset_m is not None else None
            ),
            "max_speed_observed_mps": (
                round(self.max_speed_observed, 2) if self.max_speed_observed is not None else None
            ),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Outlier rejection
# --------------------------------------------------------------------------
def envelope_outliers(
    enu: np.ndarray,
    times: np.ndarray,
    max_speed: float,
    max_accel: float,
    min_baseline_s: float = 0.5,
    position_quantum_m: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Flag fixes implying impossible motion for the platform.

    A fix is rejected when reaching it from the previous *accepted* fix would
    demand more speed or acceleration than the airframe has. Comparing against
    the last accepted fix rather than the immediate predecessor matters: a
    single wild fix would otherwise reject its innocent successor too, since
    the return trip looks equally impossible.

    **The comparison needs a time baseline.** Telemetry positions are
    quantized — six decimal places of latitude is about 0.11 m, and per-frame
    SRT timestamps are quantized to the millisecond. Differentiating quantized
    positions over a 1/30 s interval amplifies that quantum enormously:
    acceleration goes as ``dx / dt^2``, so 0.11 m over 33 ms reads as roughly
    100 m/s^2 — far outside any airframe's envelope, on perfectly good data.
    A filter that differentiates adjacent high-rate samples therefore rejects
    almost everything, and the cleaner the GPS the more confidently it does so.

    So velocity and acceleration are evaluated over at least ``min_baseline_s``
    of flight, and the acceleration limit additionally carries the quantization
    noise floor implied by that baseline. Samples between baselines are not
    skipped — they are simply not used as differentiation endpoints.
    """
    n = len(times)
    flags = np.zeros(n, dtype=bool)
    if n < 2:
        return flags, 0.0

    finite = np.isfinite(enu).all(axis=1)
    flags[~finite] = True

    last_good: int | None = None
    last_velocity: np.ndarray | None = None
    last_velocity_at: int | None = None
    max_observed = 0.0

    for i in range(n):
        if not finite[i]:
            continue
        if last_good is None:
            last_good = i
            continue

        dt = float(times[i] - times[last_good])
        if dt <= 0:
            flags[i] = True
            continue

        delta = enu[i] - enu[last_good]
        # A gross outlier is obvious at any baseline, so the speed test still
        # runs on short intervals — but with the quantization floor added, so
        # ordinary jitter cannot trip it.
        speed = float(np.linalg.norm(delta)) / dt
        speed_allowance = max_speed + position_quantum_m / max(dt, 1e-6)
        max_observed = max(max_observed, speed)
        if speed > speed_allowance:
            flags[i] = True
            continue

        if dt < min_baseline_s:
            # Accepted, but too close in time to differentiate against: leave
            # `last_good` where it is so the next sample gets a longer baseline.
            continue

        velocity = delta / dt
        if last_velocity is not None and last_velocity_at is not None:
            accel_dt = max(float(times[i] - times[last_velocity_at]), min_baseline_s)
            accel = float(np.linalg.norm(velocity - last_velocity)) / accel_dt
            accel_allowance = max_accel + 2.0 * position_quantum_m / (accel_dt * dt)
            if accel > accel_allowance:
                flags[i] = True
                continue

        last_velocity = velocity
        last_velocity_at = i
        last_good = i

    return flags, max_observed


def median_filter_outliers(
    enu: np.ndarray, window: int, tolerance_m: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window median, plus the residuals against it.

    Returns ``(median_track, outlier_flags)``. The median track is the
    robust reference the Kalman filter starts from; anything further than
    ``tolerance_m`` from it is treated as an outlier measurement.
    """
    n = len(enu)
    if n == 0:
        return enu.copy(), np.zeros(0, dtype=bool)
    window = max(int(window) | 1, 3)  # odd window
    half = window // 2
    smoothed = np.full_like(enu, np.nan)
    for i in range(n):
        lo, hi = max(i - half, 0), min(i + half + 1, n)
        block = enu[lo:hi]
        for axis in range(block.shape[1]):
            values = block[:, axis]
            finite = values[np.isfinite(values)]
            # A window with no finite sample on this axis has no median; leave
            # it NaN rather than letting nanmedian warn about an all-NaN slice.
            if finite.size:
                smoothed[i, axis] = np.median(finite)
    residual = np.linalg.norm(enu - smoothed, axis=1)
    return smoothed, np.isfinite(residual) & (residual > tolerance_m)


# --------------------------------------------------------------------------
# Constant-velocity Kalman filter
# --------------------------------------------------------------------------
def _initial_velocity(enu: np.ndarray, times: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Seed velocity from the first two accepted fixes.

    Starting a constant-velocity filter at zero velocity makes it lag for as
    long as the process noise takes to accelerate the estimate, which on a
    drone flying a straight survey line is most of the flight. A one-line
    finite difference removes that entirely.
    """
    indices = np.flatnonzero(valid & np.isfinite(enu).all(axis=1))
    if len(indices) < 2:
        return np.zeros(3)
    i, j = int(indices[0]), int(indices[1])
    dt = float(times[j] - times[i])
    if dt <= 0:
        return np.zeros(3)
    return (enu[j] - enu[i]) / dt


def kalman_smooth(
    enu: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    process_noise: float,
    measurement_noise_h: float,
    measurement_noise_v: float,
) -> np.ndarray:
    """Constant-velocity Kalman filter with an RTS backward smoothing pass.

    State is ``[e, n, u, ve, vn, vu]``. Rejected or missing fixes skip the
    update step, so the filter coasts on its motion model — exactly the
    behaviour wanted across a short GPS dropout.

    The backward pass matters. A forward-only filter is a *causal* estimator:
    at every sample it knows only the past, so it necessarily lags a manoeuvre
    and, at the start of a track, lags the initial velocity too. This pipeline
    is offline — the whole flight is on disk before filtering begins — so
    withholding the future from the estimate buys nothing and costs accuracy.
    The Rauch-Tung-Striebel smoother runs the filter forward, then propagates
    information backward, giving every sample an estimate conditioned on the
    entire track.

    Horizontal and vertical measurement noise stay separate because GPS
    vertical error runs 2-3x the horizontal; one shared value would let
    altitude noise drag the horizontal solution.
    """
    n = len(times)
    if n == 0:
        return enu.copy()

    measurement_matrix = np.zeros((3, 6))
    measurement_matrix[:, :3] = np.eye(3)
    measurement_cov = np.diag([measurement_noise_h**2, measurement_noise_h**2, measurement_noise_v**2])

    first = int(np.argmax(valid)) if valid.any() else 0
    state = np.zeros(6, dtype=float)
    state[:3] = enu[first] if np.isfinite(enu[first]).all() else 0.0
    state[3:] = _initial_velocity(enu, times, valid)
    covariance = np.diag([100.0, 100.0, 100.0, 25.0, 25.0, 25.0])

    predicted_states = np.zeros((n, 6))
    predicted_covs = np.zeros((n, 6, 6))
    updated_states = np.zeros((n, 6))
    updated_covs = np.zeros((n, 6, 6))
    transitions = np.zeros((n, 6, 6))

    previous_t = float(times[0])
    for i in range(n):
        dt = max(float(times[i]) - previous_t, 0.0)
        previous_t = float(times[i])

        transition = np.eye(6)
        transition[0, 3] = transition[1, 4] = transition[2, 5] = dt
        # Continuous white-noise acceleration model, discretised.
        q = process_noise**2
        position_q = q * dt**4 / 4.0
        cross_q = q * dt**3 / 2.0
        velocity_q = q * dt**2
        process_cov = np.zeros((6, 6))
        for axis in range(3):
            process_cov[axis, axis] = position_q
            process_cov[axis, axis + 3] = cross_q
            process_cov[axis + 3, axis] = cross_q
            process_cov[axis + 3, axis + 3] = velocity_q

        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_cov
        transitions[i] = transition
        predicted_states[i] = state
        predicted_covs[i] = covariance

        if valid[i] and np.isfinite(enu[i]).all():
            innovation = enu[i] - measurement_matrix @ state
            innovation_cov = measurement_matrix @ covariance @ measurement_matrix.T + measurement_cov
            gain = covariance @ measurement_matrix.T @ np.linalg.pinv(innovation_cov)
            state = state + gain @ innovation
            covariance = (np.eye(6) - gain @ measurement_matrix) @ covariance

        updated_states[i] = state
        updated_covs[i] = covariance

    # Backward RTS pass.
    smoothed_states = updated_states.copy()
    smoothed_covs = updated_covs.copy()
    for i in range(n - 2, -1, -1):
        gain = updated_covs[i] @ transitions[i + 1].T @ np.linalg.pinv(predicted_covs[i + 1])
        smoothed_states[i] = updated_states[i] + gain @ (smoothed_states[i + 1] - predicted_states[i + 1])
        smoothed_covs[i] = updated_covs[i] + gain @ (smoothed_covs[i + 1] - predicted_covs[i + 1]) @ gain.T

    return smoothed_states[:, :3]


# --------------------------------------------------------------------------
# Altitude fusion
# --------------------------------------------------------------------------
def fuse_altitude(
    alt_gps: np.ndarray, alt_baro: np.ndarray, alpha: float
) -> tuple[np.ndarray, str, float | None]:
    """Complementary filter over GPS and barometric altitude.

    The barometer gives excellent relative precision with a drifting absolute
    reference; GPS gives a noisy but unbiased absolute reference. So the
    barometer supplies the *shape* of the altitude profile and GPS supplies the
    *datum*: the constant offset between them is estimated robustly (median),
    and the recursive filter leans hard on the barometric increments.

    Returns ``(altitude, source_description, estimated_offset)``.
    """
    have_gps = np.isfinite(alt_gps)
    have_baro = np.isfinite(alt_baro)

    if not have_baro.any():
        return alt_gps.copy(), "gps", None
    if not have_gps.any():
        # Baro alone is relative to takeoff — usable for shape, not for datum.
        return alt_baro.copy(), "baro_relative_only", None

    both = have_gps & have_baro
    if both.sum() < 3:
        return alt_gps.copy(), "gps", None

    offset = float(np.median(alt_gps[both] - alt_baro[both]))
    baro_absolute = alt_baro + offset

    fused = np.full_like(alt_gps, np.nan, dtype=float)
    estimate: float | None = None
    for i in range(len(fused)):
        if estimate is None:
            estimate = float(baro_absolute[i]) if have_baro[i] else (
                float(alt_gps[i]) if have_gps[i] else None
            )
            if estimate is None:
                continue
            fused[i] = estimate
            continue
        if have_baro[i] and have_baro[i - 1]:
            estimate = estimate + float(alt_baro[i] - alt_baro[i - 1])
        if have_gps[i]:
            estimate = alpha * estimate + (1.0 - alpha) * float(alt_gps[i])
        fused[i] = estimate

    return fused, "baro+gps_complementary", offset


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def filter_telemetry(table: TelemetryTable, cfg: Any) -> tuple[TelemetryTable, GpsFilterReport]:
    """Run the full §5.6 GPS conditioning pass.

    Returns a new table with filtered positions and updated ``valid_flags``,
    plus a report. An empty or GPS-free table passes through untouched — the
    scale-free path is a supported mode, not an error.
    """
    report = GpsFilterReport()
    gps_cfg = cfg.get_path("condition.gps")

    if table.is_empty or not table.has_gps:
        report.notes.append("no GPS to filter; reconstruction will be scale-free")
        log_downgrade(log, "GPS filtering", "scale-free reconstruction", "telemetry has no GPS fixes")
        return table, report
    if not bool(gps_cfg["enabled"]):
        report.notes.append("GPS filtering disabled in config")
        return table, report

    frame = table.frame.copy()
    times = frame["t"].to_numpy(dtype=float)
    report.input_fixes = int(np.isfinite(frame["lat"].to_numpy(dtype=float)).sum())

    altitude_for_enu = frame["alt_gps"].to_numpy(dtype=float)
    if not np.isfinite(altitude_for_enu).any():
        altitude_for_enu = np.zeros_like(times)
    enu, origin = enu_from_geodetic(
        frame["lat"].to_numpy(dtype=float),
        frame["lon"].to_numpy(dtype=float),
        altitude_for_enu,
    )

    # 1. Motion-envelope rejection.
    envelope_flags, max_speed = envelope_outliers(
        enu,
        times,
        float(gps_cfg["max_speed_mps"]),
        float(gps_cfg["max_accel_mps2"]),
        min_baseline_s=float(gps_cfg["envelope_min_baseline_s"]),
        position_quantum_m=float(gps_cfg["position_quantum_m"]),
    )
    report.envelope_outliers = int(envelope_flags.sum())
    report.max_speed_observed = max_speed

    # 2. Median filter over the survivors.
    working = enu.copy()
    working[envelope_flags] = np.nan
    _, median_flags = median_filter_outliers(working, int(gps_cfg["median_window"]))
    report.median_outliers = int(median_flags.sum())

    rejected = envelope_flags | median_flags
    valid = ~rejected & np.isfinite(enu).all(axis=1)
    if not valid.any():
        report.notes.append("every GPS fix was rejected as an outlier; keeping the raw track")
        log_downgrade(log, "GPS outlier filtering", "raw GPS track",
                      "all fixes failed the motion envelope — check the platform limits in config")
        valid = np.isfinite(enu).all(axis=1)
        rejected = ~valid

    # 3. Kalman smoothing.
    kalman_cfg = gps_cfg["kalman"]
    smoothed = kalman_smooth(
        enu,
        times,
        valid,
        float(kalman_cfg["process_noise"]),
        float(kalman_cfg["measurement_noise_horizontal"]),
        float(kalman_cfg["measurement_noise_vertical"]),
    )
    report.smoothed_fixes = int(np.isfinite(smoothed).all(axis=1).sum())

    # 4. Back to geodetic.
    lat, lon = _enu_to_geodetic(smoothed, origin)
    frame["lat"] = lat
    frame["lon"] = lon

    # 5. Altitude fusion.
    altitude_cfg = gps_cfg["altitude"]
    if bool(altitude_cfg["prefer_baro"]):
        fused, source, offset = fuse_altitude(
            frame["alt_gps"].to_numpy(dtype=float),
            frame["alt_baro"].to_numpy(dtype=float),
            float(altitude_cfg["complementary_alpha"]),
        )
        frame["alt_gps"] = fused
        report.altitude_source = source
        report.baro_gps_offset_m = offset
        if source == "gps":
            log_downgrade(log, "barometric altitude fusion", "GPS altitude",
                          "no usable barometric altitude in telemetry")
        elif source == "baro_relative_only":
            report.notes.append(
                "barometric altitude is relative to takeoff and no GPS altitude was available; "
                "vertical datum is unknown"
            )
    else:
        report.altitude_source = "gps"

    # 6. RTK detection and weighting.
    report.rtk_detected = table.has_rtk
    report.gps_weight = float(gps_cfg["rtk"]["weight_boost"]) if report.rtk_detected else 1.0
    if report.rtk_detected:
        log_event(log, logging.INFO,
                  "RTK/PPK fixes detected; GPS priors tightened for sub-metre georeferencing",
                  gps_weight=report.gps_weight)

    # 7. Flags.
    flags = frame["valid_flags"].to_numpy(dtype="int64")
    flags = np.where(rejected, flags | int(TelemetryFlags.GPS_OUTLIER), flags)
    flags = flags | int(TelemetryFlags.SMOOTHED)
    frame["valid_flags"] = flags

    outlier_fraction = (report.envelope_outliers + report.median_outliers) / max(report.input_fixes, 1)
    if outlier_fraction > 0.2:
        note = (
            f"{outlier_fraction:.0%} of GPS fixes were rejected as outliers — absolute "
            "georeferencing accuracy will be degraded"
        )
        report.notes.append(note)
        log_event(log, logging.WARNING, note, event="coverage_warning",
                  outlier_fraction=round(outlier_fraction, 3))

    log_event(
        log,
        logging.INFO,
        "GPS conditioning complete",
        **{k: v for k, v in report.to_dict().items() if k != "notes"},
    )
    filtered = TelemetryTable(
        frame=frame,
        source=f"{table.source}+filtered",
        notes=list(table.notes) + report.notes,
    )
    return filtered, report


def huber_weights(residuals: np.ndarray, delta: float) -> np.ndarray:
    """Huber IRLS weights for GPS position residuals.

    Used wherever GPS enters an optimisation (§5.6, §7.3, §8.1). Residuals
    within ``delta`` keep full weight; beyond it weight falls as ``delta/|r|``,
    so a handful of bad fixes cannot drag the whole block while still
    contributing their direction.
    """
    magnitude = np.abs(np.asarray(residuals, dtype=float))
    return np.where(magnitude <= delta, 1.0, delta / np.maximum(magnitude, 1e-9))


def _enu_to_geodetic(enu: np.ndarray, origin: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`~src.ingest.telemetry.enu_from_geodetic`."""
    import math

    lat0, lon0, _ = origin
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2 - f)
    sin_lat0 = math.sin(math.radians(lat0))
    meridional = a * (1 - e2) / (1 - e2 * sin_lat0**2) ** 1.5
    transverse = a / math.sqrt(1 - e2 * sin_lat0**2)

    lat = lat0 + np.degrees(enu[:, 1] / meridional)
    lon = lon0 + np.degrees(enu[:, 0] / (transverse * math.cos(math.radians(lat0))))
    return lat, lon


def track_length_m(table: TelemetryTable) -> float:
    """Total flown distance, used for sanity checks and the QA report."""
    if table.is_empty or not table.has_gps:
        return 0.0
    frame = table.frame
    altitude = frame["alt_gps"].to_numpy(dtype=float)
    if not np.isfinite(altitude).any():
        altitude = np.zeros(len(frame))
    enu, _ = enu_from_geodetic(
        frame["lat"].to_numpy(dtype=float), frame["lon"].to_numpy(dtype=float), altitude
    )
    finite = np.isfinite(enu).all(axis=1)
    track = enu[finite]
    if len(track) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())


def interpolate_to_frames(
    table: TelemetryTable, timestamps: list[float]
) -> pd.DataFrame:
    """Telemetry resampled onto the selected frames' timestamps."""
    return table.at_times(timestamps)
