"""Tests for config, budget and manifest — the infrastructure every stage leans on."""

from __future__ import annotations

import json
import time

import pytest

from src.core.budget import Budget, BudgetExceeded
from src.core.config import ConfigError, deep_merge, load_config, parse_override
from src.core.device import (
    cgroup_cpu_quota,
    chunk_frames_for_memory,
    configure_runtime,
    cpu_thread_budget,
    peak_memory_gb,
)
from src.core.manifest import RunManifest, StageStatus


class TestConfig:
    def test_loads_defaults(self, cfg):
        assert cfg.budget.total_s == 900
        assert cfg.fusion.zones.zone1_min_views == 4

    def test_missing_key_raises_rather_than_returning_none(self, cfg):
        # A typo in a tuning parameter must fail loudly, not silently disable a feature.
        with pytest.raises(ConfigError):
            _ = cfg.condition.blur.absolut_floor

    def test_presets_overlay_defaults(self):
        fast = load_config("fast")
        default = load_config()
        assert fast.preset == "fast"
        # A preset changes what it names and inherits everything else.
        assert fast.recon.track_a.pc_quality != default.recon.track_a.pc_quality
        assert fast.fusion.zones.zone1_min_triangulation_deg == default.fusion.zones.zone1_min_triangulation_deg

    def test_cli_overrides_win(self):
        cfg = load_config(overrides=["budget.total_s=600", "condition.enabled=false"])
        assert cfg.budget.total_s == 600
        assert cfg.condition.enabled is False

    def test_override_parses_yaml_types(self):
        assert parse_override("a.b=3") == ("a.b", 3)
        assert parse_override("a.b=false") == ("a.b", False)
        assert parse_override("a.b=[x, y]") == ("a.b", ["x", "y"])
        with pytest.raises(ValueError):
            parse_override("no-equals-sign")

    def test_deep_merge_replaces_lists_wholesale(self):
        merged = deep_merge({"a": {"b": [1, 2, 3], "c": 1}}, {"a": {"b": [9]}})
        assert merged == {"a": {"b": [9], "c": 1}}

    def test_get_path_with_default(self, cfg):
        assert cfg.get_path("does.not.exist", "fallback") == "fallback"
        assert cfg.get_path("budget.total_s") == 900

    def test_provenance_is_recorded(self):
        cfg = load_config("fast", overrides=["budget.total_s=42"])
        provenance = cfg.get_path("_provenance")
        assert "budget.total_s=42" in provenance["overrides"]
        assert any("fast" in s for s in provenance["sources"])


class TestBudget:
    def test_stage_timing_is_recorded(self):
        budget = Budget(total_s=10, stages={"ingest": 5})
        with budget.stage("ingest"):
            # Comfortably above Windows' ~15.6 ms timer granularity: a 10 ms
            # sleep here measured as 0.0 under full-suite load and flaked.
            time.sleep(0.05)
        assert budget.stage_times["ingest"] > 0
        assert budget.summary()["within_budget"] is True

    def test_projection_predicts_overrun_before_the_deadline(self):
        budget = Budget(total_s=100, stages={"condition": 1.0}, ladder=["reduce_resolution"])
        with budget.stage("condition") as stage:
            time.sleep(0.3)
            # 30% of the allowance spent on 10% of the work projects to a 3x
            # overrun, and must be caught while most of the budget is unspent.
            assert stage.should_degrade(progress=0.10) is True
            assert stage.should_degrade(progress=0.99) is False

    def test_no_spurious_degradation_at_the_start_of_a_stage(self):
        # Dividing elapsed time by a progress of nearly zero predicts an
        # infinite runtime, which would degrade every stage on its first
        # iteration and never recover.
        budget = Budget(total_s=100, stages={"condition": 10.0}, ladder=["reduce_resolution"])
        with budget.stage("condition") as stage:
            time.sleep(0.05)
            assert stage.should_degrade(progress=0.0) is False
            assert stage.should_degrade(progress=1 / 500) is False
            assert budget.degradations == []

    def test_ladder_is_consumed_in_order_then_exhausts(self):
        budget = Budget(total_s=100, stages={"fusion": 1.0}, ladder=["a", "b"])
        with budget.stage("fusion") as stage:
            assert stage.degrade() == "a"
            assert stage.degrade() == "b"
            assert stage.degrade() is None
        assert [d.action for d in budget.degradations] == ["a", "b"]

    def test_exhausted_ladder_past_deadline_raises(self):
        budget = Budget(total_s=100, stages={"export": 0.01}, ladder=[])
        with pytest.raises(BudgetExceeded):
            with budget.stage("export") as stage:
                time.sleep(0.05)
                stage.check()

    def test_slack_is_not_clawed_back(self):
        # An early stage finishing under budget leaves later stages untouched:
        # they keep their full nominal allowance and simply have more slack.
        budget = Budget(total_s=1.0, stages={"a": 0.5, "b": 0.5})
        with budget.stage("a"):
            time.sleep(0.05)
        assert budget.allotment_for("b") == 0.5

    def test_later_stages_inherit_the_squeeze(self):
        # An overrunning early stage must shrink what later stages are promised,
        # rather than letting the total quietly exceed the 15-minute ceiling.
        budget = Budget(total_s=1.0, stages={"a": 0.5, "b": 0.5})
        with budget.stage("a"):
            time.sleep(0.7)
        # Read the clock *before* the allotment: remaining_s only shrinks, so
        # this bound holds exactly. Reading it after raced by ~7 µs on the
        # 3-core cloud box.
        remaining_before = budget.remaining_s
        squeezed = budget.allotment_for("b")
        assert squeezed < 0.5
        assert squeezed <= remaining_before

    def test_disabled_budget_never_degrades(self):
        budget = Budget(total_s=0.001, stages={"x": 0.001}, enabled=False)
        with budget.stage("x") as stage:
            time.sleep(0.01)
            assert stage.should_degrade(progress=0.01) is False
            assert stage.exceeded() is False


class TestManifest:
    def test_round_trips(self, tmp_path):
        manifest = RunManifest(tmp_path / "run1", config={"ingest": {"a": 1}})
        with manifest.stage("ingest") as stage:
            stage.add_artifact("frames", tmp_path / "run1" / "frames.parquet")
            stage.add_metric("selected", 42)
        (tmp_path / "run1" / "frames.parquet").write_text("x")

        reloaded = RunManifest.load(tmp_path / "run1")
        assert reloaded.stages["ingest"].status is StageStatus.DONE
        assert reloaded.stages["ingest"].metrics["selected"] == 42
        assert reloaded.artifact("ingest", "frames").name == "frames.parquet"

    def test_completed_stage_is_reused(self, tmp_path):
        manifest = RunManifest(tmp_path / "run", config={"ingest": {"a": 1}})
        artifact = tmp_path / "run" / "out.txt"
        with manifest.stage("ingest") as stage:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("done")
            stage.add_artifact("out", artifact)
        assert manifest.should_run("ingest") is False

    def test_config_change_invalidates_the_stage(self, tmp_path):
        manifest = RunManifest(tmp_path / "run", config={"ingest": {"a": 1}})
        artifact = tmp_path / "run" / "out.txt"
        with manifest.stage("ingest") as stage:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("done")
            stage.add_artifact("out", artifact)

        manifest.config = {"ingest": {"a": 2}}
        assert manifest.should_run("ingest") is True

    def test_missing_artifact_invalidates_the_stage(self, tmp_path):
        manifest = RunManifest(tmp_path / "run", config={"ingest": {}})
        artifact = tmp_path / "run" / "gone.txt"
        with manifest.stage("ingest") as stage:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("x")
            stage.add_artifact("out", artifact)
        artifact.unlink()
        assert manifest.should_run("ingest") is True

    def test_invalidation_cascades_downstream(self, tmp_path):
        manifest = RunManifest(tmp_path / "run", config={})
        for name in ("ingest", "condition", "export"):
            with manifest.stage(name):
                pass
        manifest.invalidate_from("condition")
        assert manifest.stages["ingest"].status is StageStatus.DONE
        assert manifest.stages["condition"].status is StageStatus.PENDING
        assert manifest.stages["export"].status is StageStatus.PENDING

    def test_failure_is_persisted_and_reraised(self, tmp_path):
        manifest = RunManifest(tmp_path / "run", config={})
        with pytest.raises(ValueError):
            with manifest.stage("ingest"):
                raise ValueError("boom")
        reloaded = RunManifest.load(tmp_path / "run")
        assert reloaded.stages["ingest"].status is StageStatus.FAILED
        assert "boom" in reloaded.stages["ingest"].error

    def test_save_is_atomic_and_leaves_no_temp_files(self, tmp_path):
        manifest = RunManifest(tmp_path / "run", config={})
        manifest.save()
        manifest.save()
        leftovers = [p for p in (tmp_path / "run").iterdir() if p.name.startswith(".manifest-")]
        assert leftovers == []
        assert json.loads(manifest.path.read_text())["run_id"] == "run"


class TestDeviceSizing:
    def test_peak_memory_matches_the_published_table(self):
        # Anchor points from the official VGGT-Omega benchmark in spec §7.2.
        assert peak_memory_gb(1) == pytest.approx(6.0)
        assert peak_memory_gb(100) == pytest.approx(13.4)
        assert peak_memory_gb(500) == pytest.approx(43.2)
        assert 9.7 < peak_memory_gb(75) < 13.4

    def test_chunk_sizing_respects_headroom_on_a_24gb_card(self):
        # Spec §7.2: a 24 GB card should land near 180-200 frames, and must not
        # run right up to the 20.8 GB line at 200 frames.
        chunk = chunk_frames_for_memory(total_gb=24.0, headroom_gb=4.0, requested=None)
        assert 150 <= chunk <= 210
        assert peak_memory_gb(chunk) <= 20.0

    def test_requested_chunk_is_a_ceiling_not_a_floor(self):
        assert chunk_frames_for_memory(24.0, 4.0, requested=128) == 128
        # A request bigger than the card can take is cut down to what fits.
        assert chunk_frames_for_memory(12.0, 4.0, requested=400) < 400

    def test_tiny_card_still_returns_a_usable_chunk(self):
        assert chunk_frames_for_memory(total_gb=6.0, headroom_gb=4.0) >= 8

    def test_institute_mig_slice_fits_the_default_chunk(self, cfg):
        # The team's box is a 20 GB H100 MIG slice (CLOUD_GPU_GUIDE.md §2).
        # With the configured headroom the spec's default 128-frame chunk must
        # still fit, or Track B would start life already cut down.
        chunk = chunk_frames_for_memory(
            total_gb=cfg.device.gpu_memory_gb, headroom_gb=cfg.device.gpu_headroom_gb, requested=128
        )
        assert chunk == 128
        assert peak_memory_gb(chunk) <= cfg.device.gpu_memory_gb - cfg.device.gpu_headroom_gb


class TestCpuThreads:
    def test_cgroup_v2_quota_is_read_in_cores(self, tmp_path):
        (tmp_path / "cpu.max").write_text("300000 100000")
        assert cgroup_cpu_quota(tmp_path) == pytest.approx(3.0)
        assert cpu_thread_budget(tmp_path) <= 3

    def test_cgroup_v2_unlimited_means_no_quota(self, tmp_path):
        (tmp_path / "cpu.max").write_text("max 100000")
        assert cgroup_cpu_quota(tmp_path) is None

    def test_cgroup_v1_quota_is_read_in_cores(self, tmp_path):
        (tmp_path / "cpu").mkdir()
        (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("250000")
        (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
        # Fractional quotas round down, but never below one thread.
        assert cgroup_cpu_quota(tmp_path) == pytest.approx(2.5)
        assert cpu_thread_budget(tmp_path) <= 2

    def test_no_cgroup_files_means_no_quota(self, tmp_path):
        assert cgroup_cpu_quota(tmp_path) is None
        assert cpu_thread_budget(tmp_path) >= 1

    def test_configured_thread_count_is_applied_to_opencv(self, cfg):
        import cv2

        before = cv2.getNumThreads()
        try:
            applied = configure_runtime(cfg.merged({"device": {"cpu_threads": 2}}))
            assert applied == {"cpu_threads": 2, "cpu_threads_source": "config"}
            assert cv2.getNumThreads() == 2
        finally:
            cv2.setNumThreads(before)
