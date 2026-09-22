"""Stage 2 end-to-end: run the pipeline on a synthetic flight and score it.

The Stage 2 scorecard is a standalone reader — it takes a run directory and
nothing else, so the Stage Lab can score a run it did not launch. That makes the
contract between what conditioning *writes* and what the evaluator *reads* easy
to break silently: it was broken from the day the evaluator was written
(DEVLOG S2-6) and nothing caught it, because the UI smoke test only checked that
the panel loaded. These tests hold that contract.
"""

from __future__ import annotations

import pytest

from src.core.config import Config, load_config
from src.core.manifest import RunManifest
from src.pipeline import RunInputs, run_pipeline
from src.qa.stage2_eval import ConditionOutputs, evaluate_condition
from src.qa.synthetic import generate_synthetic_flight

REPORTS = ("artifacts_report", "illumination_report", "dynamic_report", "gps_report")
FRAME_TABLES = ("frame_artifacts", "frame_illumination")


@pytest.fixture(scope="module")
def conditioned_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("stage2")
    video, _, _ = generate_synthetic_flight(tmp_path / "in", frames=60)
    cfg = load_config(overrides=["run.resume=false", "logging.level=ERROR"])
    run_dir = tmp_path / "run"
    run_pipeline(RunInputs(video=video), cfg, run_dir=run_dir, stages=["ingest", "condition"])
    manifest = RunManifest.load(run_dir)
    outputs = ConditionOutputs.load(run_dir / "condition")
    evaluation = evaluate_condition(outputs, Config(manifest.config))
    return run_dir, manifest, outputs, evaluation


class TestReportsAreWritten:
    @pytest.mark.parametrize("name", REPORTS)
    def test_report_json_exists(self, conditioned_run, name):
        run_dir, *_ = conditioned_run
        assert (run_dir / "condition" / f"{name}.json").is_file()

    @pytest.mark.parametrize("name", FRAME_TABLES)
    def test_frame_table_exists(self, conditioned_run, name):
        run_dir, *_ = conditioned_run
        assert (run_dir / "condition" / f"{name}.parquet").is_file()

    @pytest.mark.parametrize("name", REPORTS + FRAME_TABLES)
    def test_report_is_recorded_in_the_manifest(self, conditioned_run, name):
        _, manifest, *_ = conditioned_run
        assert name in manifest.stages["condition"].artifacts


class TestOutputsLoad:
    def test_every_report_loads_non_empty(self, conditioned_run):
        _, _, outputs, _ = conditioned_run
        assert outputs.artifacts_report and outputs.illumination_report
        assert outputs.dynamic_report and outputs.gps_report

    def test_frame_tables_carry_the_columns_the_charts_plot(self, conditioned_run):
        _, _, outputs, _ = conditioned_run
        assert list(outputs.frame_artifacts.columns) == ["index", "blockiness"]
        assert {"index", "shadow_fraction"} <= set(outputs.frame_illumination.columns)
        assert not outputs.frame_artifacts.empty


class TestScorecard:
    def test_the_stage_scores_rather_than_reporting_missing_reports(self, conditioned_run):
        """The S2-6 regression: every KPI was "report missing" and the score 0."""
        *_, evaluation = conditioned_run
        missing = [k for k in evaluation.kpis if k.key.endswith("_report_missing")]
        assert not missing, f"reports not found by the evaluator: {[k.key for k in missing]}"
        assert evaluation.score is not None and evaluation.score > 0

    def test_all_four_kpi_groups_are_represented(self, conditioned_run):
        *_, evaluation = conditioned_run
        groups = {k.group.split(" (")[0] for k in evaluation.kpis}
        assert {"Artifact suppression", "Illumination", "GPS conditioning"} <= groups

    def test_gps_max_speed_kpi_fires(self, conditioned_run):
        """Guards the key mismatch: the report says max_speed_observed_mps."""
        *_, evaluation = conditioned_run
        assert any(k.key == "gps_max_speed" for k in evaluation.kpis)

    def test_shadow_truth_kpis_stay_dormant_without_per_pixel_truth(self, conditioned_run):
        # Honesty guard: shadow recall and the dark-paint false-positive rate
        # need per-pixel truth the synthetic generator does not produce, so they
        # must stay absent rather than be filled with numbers nothing measured.
        # Blockiness reduction is no longer in this group — it is measured
        # directly, before and after, on whatever frames were corrected.
        *_, evaluation = conditioned_run
        keys = {k.key for k in evaluation.kpis}
        assert "shadow_recall" not in keys
        assert "shadow_false_positive" not in keys


# ---------------------------------------------------------------------------
# KPI threshold logic
#
# The tests above run the real pipeline, which only ever exercises the happy
# path: on a clean synthetic flight every scored KPI passes. These drive
# evaluate_condition with synthesised reports instead, so the warn and fail
# branches - the ones that matter when a run goes wrong - are covered too.
# Each case sits on a threshold boundary, because that is where an off-by-one
# comparison hides.
# ---------------------------------------------------------------------------
from src.qa.stage2_eval import (  # noqa: E402
    FAIL,
    INFO,
    PASS,
    WARN,
    GAIN_SPAN_WARN,
    _threshold_status,
)

TRUTH = {"synthetic": True}     # any non-None value unlocks the ground-truth KPIs


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _outputs(artifacts=None, illumination=None, dynamic=None, gps=None) -> ConditionOutputs:
    return ConditionOutputs(
        artifacts_report=artifacts or {},
        illumination_report=illumination or {},
        dynamic_report=dynamic or {},
        gps_report=gps or {},
    )


def _status(evaluation, key: str):
    kpi = next((k for k in evaluation.kpis if k.key == key), None)
    return kpi.status if kpi else None


class TestThresholdHelper:
    @pytest.mark.parametrize("value,expected", [
        (0.00, PASS), (0.10, PASS),     # warn_above is inclusive of pass
        (0.11, WARN), (0.30, WARN),     # fail_above is inclusive of warn
        (0.31, FAIL), (1.00, FAIL),
    ])
    def test_boundaries(self, value, expected):
        assert _threshold_status(value, 0.10, 0.30) == expected


class TestMissingReports:
    def test_required_reports_fail_and_optional_ones_inform(self, cfg):
        ev = evaluate_condition(_outputs(), cfg)
        assert _status(ev, "artifacts_report_missing") == FAIL
        assert _status(ev, "illumination_report_missing") == FAIL
        assert _status(ev, "dynamic_report_missing") == INFO
        assert _status(ev, "gps_report_missing") == INFO
        assert ev.score == 0.0


class TestArtifactKpis:
    @pytest.mark.parametrize("corrected,expected", [
        (14, PASS), (15, WARN), (29, WARN), (30, FAIL),
    ])
    def test_correction_fraction_bands(self, cfg, corrected, expected):
        ev = evaluate_condition(
            _outputs(artifacts={"frames_evaluated": 100, "frames_corrected": corrected}), cfg)
        assert _status(ev, "artifact_correct_fraction") == expected

    @pytest.mark.parametrize("after,expected", [
        (0.85, PASS),   # 15% reduction, clears the S2-1 target
        (0.95, WARN),   # 5%, some improvement but under target
        (1.00, FAIL),   # no change
        (1.10, FAIL),   # suppression made blocking worse
    ])
    def test_blockiness_reduction_bands(self, cfg, after, expected):
        report = {"frames_evaluated": 10, "frames_corrected": 1,
                  "blockiness_before": 1.0, "blockiness_after": after}
        ev = evaluate_condition(_outputs(artifacts=report), cfg, TRUTH)
        assert _status(ev, "blockiness_reduction") == expected

    def test_blockiness_reduction_needs_no_ground_truth(self, cfg):
        """Before and after are measured on the same real frames.

        This KPI used to be gated on ``truth`` alongside the shadow ones, which
        left it dormant on every real run. Nothing about a before/after pair
        requires synthetic truth, and it is the only artifact KPI that reports
        whether suppression actually worked.
        """
        report = {"frames_evaluated": 10, "frames_corrected": 1,
                  "blockiness_before": 1.0, "blockiness_after": 0.85}
        ev = evaluate_condition(_outputs(artifacts=report), cfg)
        assert _status(ev, "blockiness_reduction") == PASS

    def test_reduction_is_absent_when_the_stage_measured_nothing(self, cfg):
        """No frame needed correction, so there is no before/after pair."""
        ev = evaluate_condition(
            _outputs(artifacts={"frames_evaluated": 10, "frames_corrected": 0}), cfg)
        assert _status(ev, "blockiness_reduction") is None


class TestIlluminationKpis:
    def test_gain_span_is_informational_because_it_saturates(self, cfg):
        """Once a frame clamps, the span is the clamp width, not a measurement."""
        ev = evaluate_condition(_outputs(illumination={
            "frames": 10, "exposure_chain": {"frames": 10, "gain_span": 0.6,
                                             "requested_gain_span": 0.72,
                                             "clamped_frames": 9, "rejected_links": 0},
        }), cfg)
        assert _status(ev, "exposure_gain_span") == INFO
        # The number that was truncated is still reported, in the detail.
        span_kpi = next(k for k in ev.kpis if k.key == "exposure_gain_span")
        assert "0.72" in span_kpi.detail

    @pytest.mark.parametrize("clamped,expected", [
        (0, PASS), (1, PASS), (2, WARN), (3, WARN), (4, FAIL), (9, FAIL),
    ])
    def test_exposure_clamped_fraction_bands(self, cfg, clamped, expected):
        """The outcome measure: frames the chain could not fully correct."""
        ev = evaluate_condition(_outputs(illumination={
            "frames": 10, "exposure_chain": {"frames": 10, "gain_span": 0.6,
                                             "clamped_frames": clamped, "rejected_links": 0},
        }), cfg)
        assert _status(ev, "exposure_clamped_fraction") == expected

    def test_a_runaway_chain_is_named_in_the_detail(self, cfg):
        """Drift and a genuinely changing scene look identical without this."""
        ev = evaluate_condition(_outputs(illumination={
            "frames": 10, "exposure_chain": {"frames": 10, "gain_span": 0.6,
                                             "clamped_frames": 9, "rejected_links": 0,
                                             "unclamped_gain_final": 0.0004},
        }), cfg)
        detail = next(k for k in ev.kpis if k.key == "exposure_clamped_fraction").detail
        assert "0.0004" in detail and "compounding" in detail

    @pytest.mark.parametrize("rejected,expected", [
        (9, PASS), (10, WARN), (24, WARN), (25, FAIL),
    ])
    def test_rejected_link_bands(self, cfg, rejected, expected):
        ev = evaluate_condition(_outputs(illumination={
            "frames": 100,
            "exposure_chain": {"frames": 100, "gain_span": 0.0, "rejected_links": rejected},
        }), cfg)
        assert _status(ev, "exposure_rejected_links") == expected

    @pytest.mark.parametrize("recall,expected", [
        (0.51, PASS), (0.50, WARN), (0.31, WARN), (0.30, FAIL),
    ])
    def test_shadow_recall_bands(self, cfg, recall, expected):
        ev = evaluate_condition(
            _outputs(illumination={"frames": 10, "shadow_recall": recall}), cfg, TRUTH)
        assert _status(ev, "shadow_recall") == expected

    @pytest.mark.parametrize("fp,expected", [
        (0.24, PASS), (0.25, WARN), (0.49, WARN), (0.50, FAIL),
    ])
    def test_dark_paint_false_positive_bands(self, cfg, fp, expected):
        ev = evaluate_condition(
            _outputs(illumination={"frames": 10, "shadow_false_positive": fp}), cfg, TRUTH)
        assert _status(ev, "shadow_false_positive") == expected


class TestDynamicKpis:
    @pytest.mark.parametrize("masked_max,expected", [
        (0.19, PASS), (0.20, WARN), (0.39, WARN), (0.40, FAIL),
    ])
    def test_peak_masked_area_bands(self, cfg, masked_max, expected):
        ev = evaluate_condition(_outputs(dynamic={
            "frames": 10, "frames_with_movers": 1,
            "masked_fraction_mean": 0.05, "masked_fraction_max": masked_max,
        }), cfg)
        assert _status(ev, "dynamic_masked_fraction_max") == expected

    def test_no_movers_means_no_masked_area_kpi(self, cfg):
        ev = evaluate_condition(_outputs(dynamic={
            "frames": 10, "frames_with_movers": 0,
            "masked_fraction_mean": 0.0, "masked_fraction_max": 0.0,
        }), cfg)
        assert _status(ev, "dynamic_masked_fraction_max") is None


class TestGpsKpis:
    def test_scale_free_run_warns_and_skips_the_rest(self, cfg):
        ev = evaluate_condition(
            _outputs(gps={"notes": ["no GPS in telemetry; scale-free reconstruction"]}), cfg)
        assert _status(ev, "gps_available") == WARN
        # The remaining GPS KPIs are meaningless without fixes and must not appear.
        assert _status(ev, "gps_outlier_fraction") is None

    @pytest.mark.parametrize("envelope,expected", [
        (10, PASS), (11, WARN), (30, WARN), (31, FAIL),
    ])
    def test_outlier_fraction_bands(self, cfg, envelope, expected):
        ev = evaluate_condition(_outputs(gps={
            "input_fixes": 100, "envelope_outliers": envelope,
            "median_outliers": 0, "smoothed_fixes": 100,
        }), cfg)
        assert _status(ev, "gps_outlier_fraction") == expected

    @pytest.mark.parametrize("smoothed,expected", [(70, PASS), (69, WARN)])
    def test_smoothed_fix_survival(self, cfg, smoothed, expected):
        ev = evaluate_condition(_outputs(gps={
            "input_fixes": 100, "envelope_outliers": 0,
            "median_outliers": 0, "smoothed_fixes": smoothed,
        }), cfg)
        assert _status(ev, "gps_smoothed_fixes") == expected

    @pytest.mark.parametrize("key", ["max_speed_observed_mps", "max_speed_observed"])
    def test_max_speed_reads_either_spelling(self, cfg, key):
        """The canonical name and the older one the evaluator used to expect."""
        limit = float(cfg.get_path("condition.gps.max_speed_mps"))
        fast = evaluate_condition(_outputs(gps={"notes": [], key: limit + 1}), cfg)
        slow = evaluate_condition(_outputs(gps={"notes": [], key: limit - 1}), cfg)
        assert _status(fast, "gps_max_speed") == WARN
        assert _status(slow, "gps_max_speed") == PASS


class TestScoreArithmetic:
    def test_warn_counts_half_and_info_does_not_count(self, cfg):
        # Five KPIs score here: artifact fraction, low-light (0.1, under the
        # 0.2 band), shadow coverage (0.0) and rejected links all pass, gain
        # span warns. 4.5 / 5 = 90.0. The INFO KPIs must not dilute it.
        ev = evaluate_condition(_outputs(
            artifacts={"frames_evaluated": 100, "frames_corrected": 0},      # pass
            illumination={"frames": 10, "low_light_frames": 1,               # pass
                          "exposure_chain": {"frames": 10, "gain_span": GAIN_SPAN_WARN,
                                             "clamped_frames": 3,            # warn
                                             "rejected_links": 0}},          # pass
        ), cfg)
        assert _status(ev, "artifact_correct_fraction") == PASS
        assert _status(ev, "exposure_clamped_fraction") == WARN
        assert _status(ev, "exposure_gain_span") == INFO
        assert _status(ev, "exposure_rejected_links") == PASS
        assert _status(ev, "low_light_fraction") == PASS
        assert ev.score == 90.0


# ---------------------------------------------------------------------------
# S2-8: the scorecard must be able to fail
#
# Stage 2 scored 100/100 with five of twelve KPIs reporting "info" — shadow
# coverage, low-light, altitude provenance among them — so a degraded run could
# not move the number. A scorecard that cannot go red cannot localise a fault,
# which is the whole reason the Stage Lab exists. These pin the promoted bands
# and, most importantly, prove the card can reach a failing score.
# ---------------------------------------------------------------------------
class TestShadowCoverageIsScored:
    @pytest.mark.parametrize("coverage,expected", [
        (0.29, PASS), (0.30, PASS),     # warn_above is inclusive of pass
        (0.31, WARN), (0.50, WARN),
        (0.51, FAIL), (0.95, FAIL),
    ])
    def test_bands(self, cfg, coverage, expected):
        ev = evaluate_condition(_outputs(illumination={
            "frames": 10, "shadow_fraction_mean": coverage, "shadow_fraction_max": coverage,
        }), cfg)
        assert _status(ev, "shadow_fraction_mean") == expected

    def test_the_band_comes_from_config_not_the_source(self, cfg):
        """Rule 6: thresholds are tunable, so a preset can move this band."""
        strict = load_config(overrides=["qa.stage2.shadow_fraction_warn=0.10",
                                        "qa.stage2.shadow_fraction_fail=0.20"])
        report = {"frames": 10, "shadow_fraction_mean": 0.25, "shadow_fraction_max": 0.25}
        assert _status(evaluate_condition(_outputs(illumination=report), cfg),
                       "shadow_fraction_mean") == PASS
        assert _status(evaluate_condition(_outputs(illumination=report), strict),
                       "shadow_fraction_mean") == FAIL


class TestLowLightIsScored:
    @pytest.mark.parametrize("low_light,expected", [
        (0, PASS), (20, PASS), (21, WARN), (50, WARN), (51, FAIL),
    ])
    def test_bands(self, cfg, low_light, expected):
        ev = evaluate_condition(
            _outputs(illumination={"frames": 100, "low_light_frames": low_light}), cfg)
        assert _status(ev, "low_light_fraction") == expected


class TestAltitudeSourceIsScored:
    @pytest.mark.parametrize("source,expected", [
        ("baro+gps_complementary", PASS),   # absolute datum, baro-smoothed
        ("gps", WARN),                      # noisier, but still absolute
        ("baro_relative_only", FAIL),       # datum unknown: georeferencing is a guess
        ("none", FAIL),
        ("unknown", FAIL),
    ])
    def test_bands(self, cfg, source, expected):
        ev = evaluate_condition(_outputs(gps={
            "notes": [], "input_fixes": 10, "envelope_outliers": 0,
            "median_outliers": 0, "smoothed_fixes": 10, "altitude_source": source,
        }), cfg)
        assert _status(ev, "gps_altitude_source") == expected


class TestTheCardCanFail:
    """The S2-8 regression: a bad run must produce a bad number."""

    def _degraded(self):
        return _outputs(
            artifacts={"frames_evaluated": 100, "frames_corrected": 90},   # fail
            illumination={
                "frames": 100, "low_light_frames": 80,                     # fail
                "shadow_fraction_mean": 0.85, "shadow_fraction_max": 0.95, # fail
                "exposure_chain": {"frames": 100, "gain_span": 1.2,
                                   "clamped_frames": 90,                   # fail
                                   "rejected_links": 40},                  # fail
            },
            dynamic={"frames": 100, "frames_with_movers": 70,
                     "masked_fraction_mean": 0.60,
                     "masked_fraction_max": 0.85},                          # fail
            gps={"notes": [], "input_fixes": 100, "envelope_outliers": 60,  # fail
                 "median_outliers": 0, "smoothed_fixes": 10,                # warn
                 "altitude_source": "baro_relative_only"},                  # fail
        )

    def test_a_thoroughly_degraded_run_scores_near_zero(self, cfg):
        ev = evaluate_condition(self._degraded(), cfg)
        counts = ev.counts()
        assert counts["fail"] >= 6, ev.table()
        assert ev.score is not None and ev.score < 15.0

    def test_the_illumination_group_alone_can_fail(self, cfg):
        """S2-3 is an illumination defect; that group must be able to go red."""
        ev = evaluate_condition(self._degraded(), cfg)
        # Only the scored ones: exposure_gain_span is deliberately INFO because
        # it saturates at the clamp width once anything clamps.
        scored = [k for k in ev.kpis
                  if k.group.startswith("Illumination") and k.status != INFO]
        assert scored, "the illumination group scored nothing"
        assert all(k.status == FAIL for k in scored), [(k.key, k.status) for k in scored]

    def test_only_genuinely_contextual_kpis_stay_unscored(self, cfg):
        """Every remaining INFO KPI is context, not a quality signal.

        keypoints_vetoed: grid vetoing happens in Stage 4, so it is always 0 here.
        gps_rtk_detected: most platforms have no RTK; its absence is not a defect.
        dynamic_frames_with_movers: how *much* of a frame is masked is the
        reconstruction-relevant quantity and is scored separately; a survey over
        a highway can have movers in every frame with negligible masked area.
        """
        ev = evaluate_condition(self._degraded(), cfg)
        info = {k.key for k in ev.kpis if k.status == INFO}
        assert info <= {"keypoints_vetoed", "gps_rtk_detected",
                        "dynamic_frames_with_movers", "exposure_gain_span", "gps_note"}


class TestSaturationIsScored:
    """S2-10: the only illumination KPI measured on the output pixels.

    Every other one describes the input or the transform. A runaway bias blew
    20 of 53 conditioned frames to solid white and no KPI noticed, because none
    of them asked whether the image was still there.
    """

    @pytest.mark.parametrize("mean,expected", [
        (0.001, PASS), (0.02, PASS), (0.03, WARN), (0.10, WARN), (0.11, FAIL), (0.80, FAIL),
    ])
    def test_bands(self, cfg, mean, expected):
        ev = evaluate_condition(_outputs(illumination={
            "frames": 53, "saturated_fraction_mean": mean, "saturated_fraction_max": mean,
        }), cfg)
        assert _status(ev, "saturated_fraction") == expected

    def test_blown_frames_are_named_in_the_detail(self, cfg):
        ev = evaluate_condition(_outputs(illumination={
            "frames": 53, "saturated_fraction_mean": 0.38,
            "saturated_fraction_max": 1.0, "blown_frames": 20,
        }), cfg)
        kpi = next(k for k in ev.kpis if k.key == "saturated_fraction")
        assert kpi.status == FAIL
        assert "20 frame(s)" in kpi.detail and "no recoverable detail" in kpi.detail

    def test_absent_on_a_run_that_never_measured_it(self, cfg):
        ev = evaluate_condition(_outputs(illumination={"frames": 10}), cfg)
        assert _status(ev, "saturated_fraction") is None

    def test_quantile_fallbacks_are_surfaced_on_the_clamp_kpi(self, cfg):
        """Points at fit_gain_bias's documented content-change bias."""
        ev = evaluate_condition(_outputs(illumination={
            "frames": 10, "exposure_chain": {"frames": 10, "gain_span": 0.6,
                                             "clamped_frames": 9, "rejected_links": 0,
                                             "quantile_fallbacks": 7},
        }), cfg)
        detail = next(k for k in ev.kpis if k.key == "exposure_clamped_fraction").detail
        assert "7 link(s) fell back to quantile matching" in detail
