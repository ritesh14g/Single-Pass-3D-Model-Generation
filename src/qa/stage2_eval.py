"""Stage 2 (Conditioning, spec §5) evaluation — scorecard behind the Stage Lab panel.

KPI groups and the measuring technique behind each:

  * Artifact suppression (§5.3): blockiness correction fraction, keypoints vetoed.
    With ground truth (synthetic blocking): suppression ratio (S2-1 fix target).

  * Illumination (§5.4): shadow coverage, low-light fraction, exposure gain span
    and rejected links. With ground truth (synthetic shadow): recall and
    dark-paint false-positive rate (validates the S2-2 Otsu-adaptive fix).

  * Dynamic masking (§5.5): fraction of frames with moving objects, peak masked area.

  * GPS conditioning (§5.6): envelope / median outliers, Kalman smoothed fixes,
    altitude source, RTK detection, max observed speed.

Status per KPI: pass / warn / fail against the stated target, or info.
Stage score = share of scored KPIs passed (warn counts half).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.qa.stage1_eval import FAIL, INFO, PASS, WARN, Kpi, StageEvaluation

GAIN_SPAN_WARN = 0.40
GPS_OUTLIER_FRACTION_WARN = 0.10
GPS_OUTLIER_FRACTION_FAIL = 0.30


@dataclass
class ConditionOutputs:
    """Stage 2 artifacts loaded from the run directory."""

    artifacts_report: dict[str, Any]
    illumination_report: dict[str, Any]
    dynamic_report: dict[str, Any]
    gps_report: dict[str, Any]
    frame_artifacts: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame_illumination: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def load(cls, stage_dir: "Path | str") -> "ConditionOutputs":
        stage_dir = Path(stage_dir)

        def rj(name: str) -> dict:
            p = stage_dir / name
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

        def rp(name: str) -> pd.DataFrame:
            p = stage_dir / name
            return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

        return cls(
            artifacts_report=rj("artifacts_report.json"),
            illumination_report=rj("illumination_report.json"),
            dynamic_report=rj("dynamic_report.json"),
            gps_report=rj("gps_report.json"),
            frame_artifacts=rp("frame_artifacts.parquet"),
            frame_illumination=rp("frame_illumination.parquet"),
        )


def _band(cfg: Any, key: str, default: float) -> float:
    """A Stage 2 scorecard threshold, from ``qa.stage2`` with a safe fallback."""
    try:
        value = cfg.get_path(f"qa.stage2.{key}")
    except Exception:
        return default
    return default if value is None else float(value)


def _band_list(cfg: Any, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    try:
        value = cfg.get_path(f"qa.stage2.{key}")
    except Exception:
        return default
    return tuple(str(v) for v in value) if value else default


def _threshold_status(value: float, warn_above: float, fail_above: float) -> str:
    if value <= warn_above:
        return PASS
    if value <= fail_above:
        return WARN
    return FAIL


def evaluate_condition(
    outputs: "ConditionOutputs",
    cfg: Any,
    truth: "dict | None" = None,
) -> StageEvaluation:
    """Build the Stage 2 scorecard from the run JSON reports."""
    ev = StageEvaluation(stage="condition", has_ground_truth=truth is not None)
    _artifact_kpis(ev, outputs, cfg, truth)
    _illumination_kpis(ev, outputs, cfg, truth)
    _dynamic_kpis(ev, outputs, cfg)
    _gps_kpis(ev, outputs, cfg)
    return ev


# ---------------------------------------------------------------------------
# KPI groups
# ---------------------------------------------------------------------------
def _artifact_kpis(ev: StageEvaluation, outputs: "ConditionOutputs", cfg: Any, truth: "dict | None") -> None:
    art = outputs.artifacts_report
    if not art:
        ev.kpis.append(Kpi(
            "artifacts_report_missing", "Artifact suppression",
            "Report present", False, "required", FAIL,
            "No artifacts_report.json in run directory.",
        ))
        return
    frames_total = int(art.get("frames_evaluated", 0))
    frames_corrected = int(art.get("frames_corrected", 0))
    keypoints_vetoed = int(art.get("keypoints_vetoed", 0))
    if frames_total:
        frac = frames_corrected / frames_total
        ev.kpis.append(Kpi(
            "artifact_correct_fraction", "Artifact suppression",
            "Frames requiring correction", round(frac, 3), "< 0.30",
            PASS if frac < 0.15 else WARN if frac < 0.30 else FAIL,
            str(frames_corrected) + " of " + str(frames_total) + " frames had visible blocking.",
        ))
    ev.kpis.append(Kpi(
        "keypoints_vetoed", "Artifact suppression",
        "Keypoints vetoed on block grid", keypoints_vetoed, "info", INFO,
        "SIFT/FAST keypoints discarded on compression block boundary.",
    ))
    if "blockiness_before" in art and "blockiness_after" in art:
        # Measured on the same frames before and after suppression, so this
        # needs no ground truth and reports on whatever footage was run. It is
        # the only artifact KPI that says whether conditioning *worked*; the
        # correction fraction above only says how much work the input needed.
        before = float(art["blockiness_before"])
        after = float(art["blockiness_after"])
        reduction = (before - after) / max(before, 1e-6)
        ev.kpis.append(Kpi(
            "blockiness_reduction", "Artifact suppression",
            "Blockiness reduction on corrected frames", round(reduction, 3),
            "> 0.10 (S2-1 fix target)",
            PASS if reduction > 0.10 else WARN if reduction > 0.0 else FAIL,
            "Mean score before: " + str(round(before, 3)) + ", after: " + str(round(after, 3)) + ".",
        ))


def _illumination_kpis(ev: StageEvaluation, outputs: "ConditionOutputs", cfg: Any, truth: "dict | None") -> None:
    illum = outputs.illumination_report
    if not illum:
        ev.kpis.append(Kpi(
            "illumination_report_missing", "Illumination",
            "Report present", False, "required", FAIL,
            "No illumination_report.json in run directory.",
        ))
        return
    frames = int(illum.get("frames", 0))
    low_light = int(illum.get("low_light_frames", 0))
    shadow_mean = float(illum.get("shadow_fraction_mean", 0.0))
    shadow_max = float(illum.get("shadow_fraction_max", 0.0))
    if frames:
        ll_frac = low_light / frames
        ll_warn = _band(cfg, "low_light_fraction_warn", 0.20)
        ll_fail = _band(cfg, "low_light_fraction_fail", 0.50)
        ev.kpis.append(Kpi(
            "low_light_fraction", "Illumination",
            "Low-light frames", round(ll_frac, 3), "< " + str(ll_warn),
            _threshold_status(ll_frac, ll_warn, ll_fail),
            str(low_light) + " of " + str(frames) + " frames triggered low-light path (§5.4 accuracy note).",
        ))
    shadow_warn = _band(cfg, "shadow_fraction_warn", 0.30)
    shadow_fail = _band(cfg, "shadow_fraction_fail", 0.50)
    ev.kpis.append(Kpi(
        "shadow_fraction_mean", "Illumination",
        "Mean shadow coverage per frame", round(shadow_mean, 3),
        "< " + str(shadow_warn),
        _threshold_status(shadow_mean, shadow_warn, shadow_fail),
        "Peak per-frame shadow coverage: " + str(round(shadow_max * 100, 1)) + "%. "
        "High coverage means either a genuinely shadowed scene or an over-firing "
        "detector; separating the two needs ground truth (S2-3).",
    ))
    chain = illum.get("exposure_chain", {})
    if chain:
        gain_span = float(chain.get("gain_span", 0.0))
        rejected_links = int(chain.get("rejected_links", 0))
        chain_frames = int(chain.get("frames", 0))
        requested_span = float(chain.get("requested_gain_span", gain_span))

        # Scored: how many frames the chain could not correct. A clamped frame
        # had its exposure transform truncated at a bound, so the normalisation
        # did not fully apply — an outcome, unlike the span below.
        if chain_frames:
            clamped_frac = float(chain.get(
                "clamped_fraction", int(chain.get("clamped_frames", 0)) / chain_frames))
            warn = _band(cfg, "exposure_clamped_fraction_warn", 0.10)
            fail = _band(cfg, "exposure_clamped_fraction_fail", 0.30)
            unclamped_final = chain.get("unclamped_gain_final")
            detail = (str(int(chain.get("clamped_frames", 0))) + " of " + str(chain_frames)
                      + " frames hit a gain bound; their exposure was not fully normalised.")
            if unclamped_final is not None:
                detail += (" Unbounded, the chain would have reached gain "
                           + str(round(float(unclamped_final), 4))
                           + " — far from 1.0 means per-link error is compounding, "
                           "not that the scene changed.")
            ev.kpis.append(Kpi(
                "exposure_clamped_fraction", "Illumination",
                "Frames with exposure clamped", round(clamped_frac, 3),
                "< " + str(warn), _threshold_status(clamped_frac, warn, fail), detail,
            ))

        # Informational: once anything clamps, this is the width of the clamp
        # range, not a measurement of the footage. Kept because the gap between
        # it and the requested span is the size of the correction that was lost.
        ev.kpis.append(Kpi(
            "exposure_gain_span", "Illumination",
            "Exposure gain span across sequence", round(gain_span, 3), "info", INFO,
            "Span the chain asked for before clamping: " + str(round(requested_span, 3))
            + ". Large values mean auto-exposure drift, a compounding chain, or both.",
        ))
        if chain_frames:
            rfrac = rejected_links / chain_frames
            ev.kpis.append(Kpi(
                "exposure_rejected_links", "Illumination",
                "Exposure chain links rejected", rejected_links, "< 10%",
                PASS if rfrac < 0.10 else WARN if rfrac < 0.25 else FAIL,
                "A rejected link means irrecoverable lighting change between adjacent frames.",
            ))
    if truth is not None and "shadow_recall" in illum:
        recall = float(illum["shadow_recall"])
        ev.kpis.append(Kpi(
            "shadow_recall", "Illumination (ground truth)",
            "Shadow recall", round(recall, 3), "> 0.50 (S2-2 fix target)",
            PASS if recall > 0.50 else WARN if recall > 0.30 else FAIL,
            "Fraction of synthetically shadowed pixels correctly detected.",
        ))
    if truth is not None and "shadow_false_positive" in illum:
        fp = float(illum["shadow_false_positive"])
        ev.kpis.append(Kpi(
            "shadow_false_positive", "Illumination (ground truth)",
            "Dark-paint false-positive rate", round(fp, 3), "< 0.25",
            PASS if fp < 0.25 else WARN if fp < 0.50 else FAIL,
            "Fraction of dark-but-not-shadowed pixels incorrectly flagged.",
        ))


def _dynamic_kpis(ev: StageEvaluation, outputs: "ConditionOutputs", cfg: Any) -> None:
    dyn = outputs.dynamic_report
    if not dyn:
        ev.kpis.append(Kpi(
            "dynamic_report_missing", "Dynamic masking",
            "Report present", False, "optional", INFO,
            "No dynamic_report.json; semantic masking may have been skipped.",
        ))
        return
    frames = int(dyn.get("frames", 0))
    with_movers = int(dyn.get("frames_with_movers", 0))
    masked_mean = float(dyn.get("masked_fraction_mean", 0.0))
    masked_max = float(dyn.get("masked_fraction_max", 0.0))
    methods = list(dyn.get("methods", []))
    if frames:
        mover_frac = with_movers / frames
        method_str = ", ".join(methods) if methods else "none"
        ev.kpis.append(Kpi(
            "dynamic_frames_with_movers", "Dynamic masking",
            "Frames containing dynamic objects", round(mover_frac, 3), "info", INFO,
            str(with_movers) + "/" + str(frames) + " frames had movers. Methods: " + method_str + ".",
        ))
    if masked_max > 0:
        ev.kpis.append(Kpi(
            "dynamic_masked_fraction_max", "Dynamic masking",
            "Peak masked frame area", round(masked_max, 3), "< 0.40 (§5.5 coverage concern)",
            PASS if masked_max < 0.20 else WARN if masked_max < 0.40 else FAIL,
            "Mean per-frame masked area: " + str(round(masked_mean * 100, 1)) + "%.",
        ))


def _gps_kpis(ev: StageEvaluation, outputs: "ConditionOutputs", cfg: Any) -> None:
    gps = outputs.gps_report
    if not gps:
        ev.kpis.append(Kpi(
            "gps_report_missing", "GPS conditioning",
            "Report present", False, "optional", INFO, "No gps_report.json found.",
        ))
        return
    notes = gps.get("notes", [])
    if "scale-free" in " ".join(notes).lower():
        ev.kpis.append(Kpi(
            "gps_available", "GPS conditioning",
            "GPS available", False, "GPS required for metric scale (§4.2)", WARN,
            "Scale-free reconstruction path - no GPS in telemetry.",
        ))
        return
    input_fixes = int(gps.get("input_fixes", 0))
    envelope = int(gps.get("envelope_outliers", 0))
    median_out = int(gps.get("median_outliers", 0))
    smoothed = int(gps.get("smoothed_fixes", 0))
    if input_fixes:
        outlier_warn = _band(cfg, "gps_outlier_fraction_warn", GPS_OUTLIER_FRACTION_WARN)
        outlier_fail = _band(cfg, "gps_outlier_fraction_fail", GPS_OUTLIER_FRACTION_FAIL)
        outlier_frac = (envelope + median_out) / input_fixes
        ev.kpis.append(Kpi(
            "gps_outlier_fraction", "GPS conditioning",
            "GPS fixes rejected as outliers", round(outlier_frac, 3),
            "< " + str(int(outlier_warn * 100)) + "%",
            _threshold_status(outlier_frac, outlier_warn, outlier_fail),
            "Envelope: " + str(envelope) + ", median: " + str(median_out) + ", from " + str(input_fixes) + " fixes.",
        ))
        ev.kpis.append(Kpi(
            "gps_smoothed_fixes", "GPS conditioning",
            "Fixes after Kalman smoothing", smoothed,
            ">= " + str(max(int(input_fixes * 0.7), 2)) + " (70% of input)",
            PASS if smoothed >= int(input_fixes * 0.7) else WARN,
            "Fixes with a finite smoothed position estimate.",
        ))
    altitude_source = str(gps.get("altitude_source", "unknown"))
    good = _band_list(cfg, "altitude_source_pass", ("baro+gps_complementary",))
    acceptable = _band_list(cfg, "altitude_source_warn", ("gps",))
    ev.kpis.append(Kpi(
        "gps_altitude_source", "GPS conditioning",
        "Altitude source", altitude_source, " or ".join(good),
        PASS if altitude_source in good else WARN if altitude_source in acceptable else FAIL,
        "baro+gps_complementary is best; GPS-only is noisier but keeps an absolute "
        "datum; baro_relative_only leaves the vertical datum unknown, which breaks "
        "absolute georeferencing downstream.",
    ))
    rtk = bool(gps.get("rtk_detected", False))
    gps_weight = gps.get("gps_weight", 1.0)
    ev.kpis.append(Kpi(
        "gps_rtk_detected", "GPS conditioning",
        "RTK/PPK detected", rtk, "info", INFO,
        "GPS weight: " + str(round(float(gps_weight), 1)) + "x with RTK.",
    ))
    # GpsReport.to_dict() emits "max_speed_observed_mps"; the older unsuffixed
    # spelling is still accepted so reports written before that name settled
    # keep scoring.
    max_speed = gps.get("max_speed_observed_mps", gps.get("max_speed_observed"))
    if max_speed is not None:
        max_speed = float(max_speed)
        limit = float(cfg.get_path("condition.gps.max_speed_mps"))
        ev.kpis.append(Kpi(
            "gps_max_speed", "GPS conditioning",
            "Max observed speed (m/s)", round(max_speed, 2), "< " + str(int(limit)) + " m/s",
            PASS if max_speed < limit else WARN,
            "Measured across all accepted GPS fixes before smoothing.", unit="m/s",
        ))
    for note in notes:
        if note:
            ev.kpis.append(Kpi("gps_note", "GPS conditioning", "Note", note, "info", INFO))


# ---------------------------------------------------------------------------
# Chart data (the UI plots these; kept here so they are testable)
# ---------------------------------------------------------------------------
def illumination_timeline(outputs: "ConditionOutputs") -> pd.DataFrame:
    """Per-frame illumination data for the timeline chart."""
    return outputs.frame_illumination.copy()


def artifact_timeline(outputs: "ConditionOutputs") -> pd.DataFrame:
    """Per-frame blockiness data for the artifact chart."""
    return outputs.frame_artifacts.copy()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_evaluation(evaluation: StageEvaluation, path: "Path | str") -> Path:
    path = Path(path)
    path.write_text(json.dumps(evaluation.to_dict(), indent=2, default=str), encoding="utf-8")
    return path
