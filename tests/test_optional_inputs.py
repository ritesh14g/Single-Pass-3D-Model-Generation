"""The optional-input ledger — what a run had, and what ran without it.

The pipeline is built to degrade rather than fail: no barometer still
reconstructs, no GPS reconstructs scale-free, no ultralytics masks movers
geometrically. Each fallback was only ever a warning in run.jsonl. These tests
pin the ledger that states them, and in particular that "the stage never ran"
stays distinct from "the input was absent".
"""

from __future__ import annotations

import json

import pytest

from src.core.inputs import (
    ABSENT,
    OPTIONAL_INPUTS,
    PRESENT,
    UNKNOWN,
    describe_optional_inputs,
    missing_inputs,
    summarize,
)
from src.core.manifest import RunManifest, StageStatus


def _manifest(tmp_path, ingest=None, condition=None):
    manifest = RunManifest(tmp_path / "run")
    if ingest is not None:
        manifest.stages["ingest"].status = StageStatus.DONE
        manifest.stages["ingest"].metrics = ingest
    if condition is not None:
        manifest.stages["condition"].status = StageStatus.DONE
        manifest.stages["condition"].metrics = condition
    return manifest


FULL_TELEMETRY = {
    "telemetry": {"source": "klv:mission.ts", "has_gps": True, "has_baro": True,
                  "has_attitude": True, "has_focal": True, "has_rtk": True},
    "video": {"hardware_decode": True, "fourcc": "avc1"},
    "selection": {"keyframes_preferred": True},
}
FULL_CONDITION = {"dynamic_model_available": True,
                  "gps_filter": {"altitude_source": "baro+gps_complementary"}}


class TestRegistry:
    def test_every_entry_declares_a_fallback_and_a_cost(self):
        for spec in OPTIONAL_INPUTS:
            assert spec.fallback and spec.impact, spec.key
            assert spec.stage in ("ingest", "condition")

    def test_keys_are_unique(self):
        keys = [s.key for s in OPTIONAL_INPUTS]
        assert len(keys) == len(set(keys))


class TestEverythingAvailable:
    def test_a_fully_equipped_run_reports_no_absences(self, tmp_path):
        ledger = describe_optional_inputs(_manifest(tmp_path, FULL_TELEMETRY, FULL_CONDITION))
        assert missing_inputs(ledger) == []
        assert summarize(ledger)[PRESENT] == len(OPTIONAL_INPUTS)


class TestAbsences:
    def test_missing_barometer_is_reported_with_its_fallback(self, tmp_path):
        ingest = json.loads(json.dumps(FULL_TELEMETRY))
        ingest["telemetry"]["has_baro"] = False
        ledger = describe_optional_inputs(_manifest(tmp_path, ingest, FULL_CONDITION))
        row = next(r for r in ledger if r["key"] == "alt_baro")
        assert row["status"] == ABSENT
        assert "GPS altitude only" in row["fallback"]
        assert row["impact"]

    def test_altitude_fusion_follows_the_source_the_stage_chose(self, tmp_path):
        condition = {"dynamic_model_available": True,
                     "gps_filter": {"altitude_source": "baro_relative_only"}}
        ledger = describe_optional_inputs(_manifest(tmp_path, FULL_TELEMETRY, condition))
        assert next(r for r in ledger if r["key"] == "altitude_fusion")["status"] == ABSENT

    def test_semantic_masking_carries_the_real_reason(self, tmp_path):
        condition = {"dynamic_model_available": False,
                     "dynamic_unavailable_reason": "ultralytics is not installed",
                     "gps_filter": {"altitude_source": "baro+gps_complementary"}}
        ledger = describe_optional_inputs(_manifest(tmp_path, FULL_TELEMETRY, condition))
        row = next(r for r in ledger if r["key"] == "semantic_masking")
        assert row["status"] == ABSENT
        assert "ultralytics" in row["detail"]

    def test_pose_prior_is_absent_for_non_klv_telemetry(self, tmp_path):
        ingest = json.loads(json.dumps(FULL_TELEMETRY))
        ingest["telemetry"]["source"] = "srt:dji.srt"
        ledger = describe_optional_inputs(_manifest(tmp_path, ingest, FULL_CONDITION))
        assert next(r for r in ledger if r["key"] == "pose_prior")["status"] == ABSENT

    @pytest.mark.parametrize("key,patch", [
        ("gps", ("telemetry", "has_gps")),
        ("attitude", ("telemetry", "has_attitude")),
        ("focal_mm", ("telemetry", "has_focal")),
        ("rtk", ("telemetry", "has_rtk")),
        ("hardware_decode", ("video", "hardware_decode")),
        ("keyframe_index", ("selection", "keyframes_preferred")),
    ])
    def test_each_flag_drives_its_own_row(self, tmp_path, key, patch):
        ingest = json.loads(json.dumps(FULL_TELEMETRY))
        section, flag = patch
        ingest[section][flag] = False
        ledger = describe_optional_inputs(_manifest(tmp_path, ingest, FULL_CONDITION))
        assert next(r for r in ledger if r["key"] == key)["status"] == ABSENT
        # Nothing else changed, so nothing else may claim to be missing.
        assert [r["key"] for r in missing_inputs(ledger)] == [key]


class TestUnknownIsNotAbsent:
    def test_a_stage_that_never_ran_yields_unknown(self, tmp_path):
        """The honest distinction: we did not look, so we do not know."""
        ledger = describe_optional_inputs(_manifest(tmp_path))
        assert summarize(ledger)[UNKNOWN] == len(OPTIONAL_INPUTS)
        assert missing_inputs(ledger) == []

    def test_condition_unknown_while_ingest_is_known(self, tmp_path):
        ledger = describe_optional_inputs(_manifest(tmp_path, FULL_TELEMETRY))
        by_key = {r["key"]: r for r in ledger}
        assert by_key["gps"]["status"] == PRESENT
        assert by_key["semantic_masking"]["status"] == UNKNOWN
        assert by_key["altitude_fusion"]["status"] == UNKNOWN

    def test_a_metric_the_run_never_recorded_is_unknown(self, tmp_path):
        """An older run has no such key; that is not evidence of absence."""
        ledger = describe_optional_inputs(_manifest(tmp_path, {}, {}))
        assert all(r["status"] == UNKNOWN for r in ledger)
