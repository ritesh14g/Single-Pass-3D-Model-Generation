"""Stage 1 end-to-end: run the pipeline on synthetic flights and score it.

These double as regression tests for the Stage Lab scorecard — if a change
drops a Stage 1 KPI from pass to fail on known-good input, this fails first.
"""

from __future__ import annotations

import pytest

from src.core.config import Config, load_config
from src.core.manifest import RunManifest, StageStatus
from src.pipeline import RunInputs, run_pipeline
from src.qa.stage1_eval import (
    FAIL,
    PASS,
    IngestOutputs,
    blur_timeline,
    evaluate_ingest,
    overlap_series,
    track_metres,
)
from src.qa.synthetic import generate_synthetic_flight, load_ground_truth, true_pair_overlap
from src.stages import STAGES, StageStatus as BuildStatus, built_manifest_stages, get_stage


def _run(tmp_path, telemetry="modern", **kwargs):
    video, sidecar, _ = generate_synthetic_flight(tmp_path / "in", telemetry=telemetry, **kwargs)
    cfg = load_config(overrides=["run.resume=false", "logging.level=ERROR"])
    run_dir = tmp_path / "run"
    # Stages 1-2 only: Stage 4 is built too, and a CPU reconstruction of a 150-frame
    # flight would add minutes to a test that is about ingest.
    run_pipeline(RunInputs(video=video), cfg, run_dir=run_dir, stages=["ingest", "condition"])
    manifest = RunManifest.load(run_dir)
    outputs = IngestOutputs.load(run_dir / "ingest", manifest.stages["ingest"].metrics)
    truth = load_ground_truth(video)
    return manifest, outputs, evaluate_ingest(outputs, Config(manifest.config), truth), truth


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("stage1"), frames=150, blur_every=11)


class TestRegistry:
    def test_built_stages(self):
        assert built_manifest_stages() == ["ingest", "condition", "track_b", "refine_ba", "track_a"]
        assert get_stage("ingest").status is BuildStatus.BUILT
        assert get_stage("condition").status is BuildStatus.BUILT
        assert get_stage("recon").status is BuildStatus.BUILT

    def test_every_manifest_stage_belongs_to_exactly_one_spec_stage(self):
        from src.core.manifest import STAGE_ORDER

        owned = [name for stage in STAGES for name in stage.manifest_stages]
        assert sorted(owned) == sorted(STAGE_ORDER)


class TestStageOneRun:
    def test_only_requested_stages_run(self, good_run):
        manifest, *_ = good_run
        assert manifest.stages["ingest"].status is StageStatus.DONE
        assert manifest.stages["condition"].status is StageStatus.DONE
        assert manifest.stages["fusion"].status is StageStatus.SKIPPED
        assert "planned" in manifest.stages["fusion"].skip_reason
        assert manifest.stages["track_a"].status is StageStatus.SKIPPED

    def test_telemetry_is_written_unfiltered(self, good_run):
        # GPS filtering is §5.6 (Stage 2); Stage 1 must record the input as parsed.
        manifest, outputs, *_ = good_run
        assert "gps_filter" not in manifest.stages["ingest"].metrics
        assert "filtered" not in manifest.stages["ingest"].metrics["telemetry"]["source"]

    def test_blur_evaluations_are_recorded(self, good_run):
        _, outputs, *_ = good_run
        assert len(outputs.evaluations) >= len(outputs.selection)
        assert set(outputs.selection["index"]) <= set(outputs.evaluations["index"])


class TestScorecard:
    def test_known_good_input_passes_the_acceptance_kpis(self, good_run):
        _, _, evaluation, _ = good_run
        kpis = {k.key: k for k in evaluation.kpis}
        for key in ("overlap_median", "overlap_low_pairs", "blurred_frames_kept",
                    "gps_frame_coverage", "gps_parse_error_m", "overlap_mae_vs_truth"):
            assert kpis[key].status == PASS, f"{key}: {kpis[key].value} ({kpis[key].target})"
        assert evaluation.score is not None and evaluation.score >= 70

    def test_ground_truth_kpis_only_appear_with_truth(self, good_run):
        manifest, outputs, _, _ = good_run
        without = evaluate_ingest(outputs, Config(manifest.config), truth=None)
        assert not any(k.group == "Ground truth" for k in without.kpis)
        assert not without.has_ground_truth

    def test_scale_free_input_fails_gps_coverage(self, tmp_path):
        _, _, evaluation, _ = _run(tmp_path, telemetry="none", frames=90, blur_every=None)
        kpis = {k.key: k for k in evaluation.kpis}
        assert kpis["gps_frame_coverage"].status == FAIL

    def test_score_is_share_of_passed_kpis(self, good_run):
        _, _, evaluation, _ = good_run
        counts = evaluation.counts()
        scored = counts["pass"] + counts["warn"] + counts["fail"]
        assert evaluation.score == pytest.approx(100 * (counts["pass"] + 0.5 * counts["warn"]) / scored, abs=0.1)


class TestChartData:
    def test_overlap_series_matches_truth_helper(self, good_run):
        _, outputs, _, truth = good_run
        series = overlap_series(outputs, truth)
        assert len(series) == len(outputs.selection) - 1
        first = outputs.selection["index"].iloc[:2].tolist()
        assert series["true"].iloc[0] == pytest.approx(true_pair_overlap(truth, *first))

    def test_blur_timeline_marks_selected_frames(self, good_run):
        _, outputs, *_ = good_run
        timeline = blur_timeline(outputs)
        assert timeline["selected"].sum() == len(outputs.selection)

    def test_track_is_metric(self, good_run):
        _, outputs, _, truth = good_run
        track = track_metres(outputs.telemetry)
        span = track["east_m"].max() - track["east_m"].min()
        duration = outputs.telemetry["t"].max() - outputs.telemetry["t"].min()
        assert span == pytest.approx(truth["speed_mps"] * duration, rel=0.02)
