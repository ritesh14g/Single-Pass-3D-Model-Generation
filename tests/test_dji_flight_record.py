"""Binary DJI flight record (DJIFlightRecord_*.txt) decoding and video alignment."""

from __future__ import annotations

import struct

import pytest

from src.ingest.dji_flight_record import (
    descramble,
    is_dji_flight_record,
    parse_dji_flight_record,
    read_flight_record,
    scramble,
)
from src.ingest.telemetry import load_telemetry
from tests import fixtures


def _settings(cfg, **overrides):
    settings = cfg.get_path("ingest.telemetry.flight_record").to_dict()
    settings.update(overrides)
    return settings


class TestContainer:
    def test_scramble_round_trips(self):
        plain = bytes(range(60))
        assert descramble(scramble(plain, record_type=1, key_byte=0xA7), record_type=1) == plain

    def test_sniffs_binary_log_but_not_text(self, tmp_path):
        log = fixtures.write_dji_flight_record(tmp_path / "DJIFlightRecord_a.txt")
        text = tmp_path / "gps.txt"
        text.write_text("12.9716,77.5946\n" * 200, encoding="utf-8")
        csv = fixtures.write_flight_csv(tmp_path / "log.csv", frames=50)

        assert is_dji_flight_record(log)
        assert not is_dji_flight_record(text)
        assert not is_dji_flight_record(csv)

    def test_decodes_samples_and_recording_segments(self, tmp_path):
        path = fixtures.write_dji_flight_record(tmp_path / "f.txt", samples=300,
                                                recordings=((5.0, 15.0), (20.0, 28.0)))
        flight = read_flight_record(path)

        assert len(flight.samples) == 300
        assert flight.samples["lat"].iloc[0] == pytest.approx(12.9716, abs=1e-6)
        assert flight.samples["height"].iloc[0] == pytest.approx(60.0)
        assert flight.samples["gimbal_pitch"].iloc[10] == pytest.approx(-90.0)
        assert flight.samples["gps_valid"].all()
        starts = [s for s, _ in flight.segments]
        assert starts == pytest.approx([5.0, 20.0])

    def test_encrypted_versions_degrade_with_a_reason(self, tmp_path, cfg):
        path = tmp_path / "v13.txt"
        prefix = bytearray(100)
        struct.pack_into("<QHB", prefix, 0, 100, 400, 13)
        path.write_bytes(bytes(prefix) + bytes(400))

        assert is_dji_flight_record(path)
        table = parse_dji_flight_record(path, video_duration_s=10.0, **_settings(cfg))
        assert table.is_empty
        assert any("encrypted" in note for note in table.notes)


class TestVideoAlignment:
    @pytest.fixture
    def two_recordings(self, tmp_path):
        return fixtures.write_dji_flight_record(tmp_path / "f.txt", samples=300,
                                                recordings=((5.0, 15.0), (20.0, 28.0)))

    def test_auto_picks_the_recording_matching_the_video(self, two_recordings, cfg):
        settings = _settings(cfg)
        table = parse_dji_flight_record(two_recordings, video_duration_s=8.0, **settings)

        assert table.has_gps and table.has_baro and table.has_attitude
        # Recording #1 starts at flight time 20 s, so video t=0 is flight time 20 s.
        assert table.frame["t"].min() == pytest.approx(-settings["segment_padding_s"], abs=0.11)
        assert table.frame["t"].max() == pytest.approx(7.9 + settings["segment_padding_s"], abs=0.11)

    def test_no_matching_duration_uses_longest_and_warns(self, two_recordings, cfg):
        table = parse_dji_flight_record(two_recordings, video_duration_s=60.0, **_settings(cfg))
        assert any("misaligned" in note for note in table.notes)
        assert any("#0" in note and "longest" in note for note in table.notes)

    def test_configured_segment_index_wins(self, two_recordings, cfg):
        table = parse_dji_flight_record(two_recordings, video_duration_s=8.0, **_settings(cfg, segment=0))
        assert any("using #0 as configured" in note for note in table.notes)

    def test_offset_overrides_segments(self, two_recordings, cfg):
        settings = _settings(cfg, offset_s=12.0)
        table = parse_dji_flight_record(two_recordings, video_duration_s=5.0, **settings)
        assert table.frame["t"].min() == pytest.approx(-settings["segment_padding_s"], abs=0.11)
        assert table.frame["t"].max() == pytest.approx(5.0 + settings["segment_padding_s"], abs=0.11)

    def test_load_telemetry_routes_binary_txt_to_the_decoder(self, two_recordings, tmp_path, cfg):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        table = load_telemetry(video, cfg, csv_path=two_recordings, video_duration_s=10.0)
        assert table.source.startswith("dji_log:")
        assert table.has_gps
