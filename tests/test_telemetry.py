"""Telemetry parsing tests.

Spec §4.2 asks explicitly for tolerant parsing unit-tested against at least two
firmware variants, because the DJI SRT format is not stable across firmwares
and the competition dataset's dialect is unknown until the day. Three dialects
are covered here, plus the no-telemetry path, which must produce a usable
scale-free result rather than an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.condition.gps_filter import filter_telemetry, track_length_m
from src.ingest.telemetry import (
    TelemetryFlags,
    TelemetryTable,
    enu_from_geodetic,
    find_sidecar,
    load_telemetry,
    parse_dji_srt,
    parse_flight_csv,
)
from tests import fixtures


class TestSrtDialects:
    def test_modern_mavic_dialect(self, tmp_path):
        path = fixtures.write_srt_modern(tmp_path / "DJI_0001.SRT", frames=30)
        table = parse_dji_srt(path)

        assert len(table) == 30
        assert table.has_gps and table.has_baro and table.has_attitude
        assert table.frame["lat"].iloc[0] == pytest.approx(12.9716, abs=1e-6)
        # rel_alt -> alt_baro, abs_alt -> alt_gps: the two must not be swapped.
        assert table.frame["alt_baro"].iloc[0] == pytest.approx(60.0)
        assert table.frame["alt_gps"].iloc[0] == pytest.approx(920.0)
        assert table.frame["pitch"].iloc[0] == pytest.approx(-89.9)
        assert table.frame["focal_mm"].iloc[0] == pytest.approx(24.0)

    def test_legacy_phantom_dialect(self, tmp_path):
        path = fixtures.write_srt_legacy(tmp_path / "DJI_0002.SRT", frames=30)
        table = parse_dji_srt(path)

        assert len(table) == 30
        assert table.has_gps
        # 'altitude' in this dialect is absolute, and there is no relative altitude.
        assert table.frame["alt_gps"].iloc[0] == pytest.approx(100.5)
        assert not table.has_baro
        # focal_len : 240 means 24 mm, not a 240 mm lens.
        assert table.frame["focal_mm"].iloc[0] == pytest.approx(24.0)
        # This dialect carries no gimbal attitude at all, and that is not an error.
        assert not table.has_attitude

    def test_bare_osd_dialect(self, tmp_path):
        path = fixtures.write_srt_bare(tmp_path / "DJI_0003.SRT", frames=20)
        table = parse_dji_srt(path)

        assert len(table) == 20
        assert table.has_gps
        # "GPS (lon, lat, sats)" is longitude-first; getting this backwards puts
        # the flight in the wrong hemisphere.
        assert table.frame["lat"].iloc[0] == pytest.approx(12.9716, abs=1e-4)
        assert table.frame["lon"].iloc[0] == pytest.approx(77.5946, abs=1e-3)
        assert table.frame["alt_baro"].iloc[0] == pytest.approx(45.0)

    def test_timestamps_track_the_cue_times(self, tmp_path):
        path = fixtures.write_srt_modern(tmp_path / "a.srt", frames=10, fps=30.0)
        table = parse_dji_srt(path)
        assert table.frame["t"].iloc[0] == pytest.approx(0.0)
        assert table.frame["t"].iloc[9] == pytest.approx(9 / 30.0, abs=0.002)

    def test_garbage_input_degrades_instead_of_raising(self, tmp_path):
        path = tmp_path / "junk.srt"
        path.write_text("this is not a subtitle file at all\n\nneither is this\n", encoding="utf-8")
        table = parse_dji_srt(path)
        assert table.is_empty
        assert table.scale_free

    def test_unknown_keys_are_ignored_not_fatal(self, tmp_path):
        path = tmp_path / "future.srt"
        path.write_text(
            "1\n00:00:00,000 --> 00:00:00,033\n"
            "[latitude: 1.5] [longitude: 2.5] [warp_core_temp: 9000] [flux: 3]\n",
            encoding="utf-8",
        )
        table = parse_dji_srt(path)
        assert len(table) == 1
        assert table.frame["lat"].iloc[0] == pytest.approx(1.5)
        assert any("warp_core_temp" in note for note in table.notes)


class TestCsv:
    def test_column_auto_detection(self, tmp_path, cfg):
        path = fixtures.write_flight_csv(tmp_path / "log.csv", frames=25)
        column_map = cfg.get_path("ingest.telemetry.csv_column_map").to_dict()
        table = parse_flight_csv(path, column_map)

        assert len(table) == 25
        assert table.has_gps
        # "OSD.altitude [m]" and "OSD.relativeAltitude" must land in different columns.
        assert table.frame["alt_gps"].iloc[0] == pytest.approx(920.0)
        assert table.frame["alt_baro"].iloc[0] == pytest.approx(60.0)

    def test_unrecognisable_csv_degrades(self, tmp_path, cfg):
        path = tmp_path / "wrong.csv"
        path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        table = parse_flight_csv(path, cfg.get_path("ingest.telemetry.csv_column_map").to_dict())
        assert table.is_empty


class TestLoadTelemetry:
    def test_finds_the_sidecar_next_to_the_video(self, tmp_path, cfg):
        video = tmp_path / "DJI_0010.MP4"
        video.write_bytes(b"not really a video")
        fixtures.write_srt_modern(tmp_path / "DJI_0010.SRT", frames=10)

        assert find_sidecar(video, [".srt"]) is not None
        table = load_telemetry(video, cfg)
        assert table.has_gps

    def test_no_telemetry_yields_a_usable_scale_free_table(self, tmp_path, cfg):
        video = tmp_path / "lonely.mp4"
        video.write_bytes(b"not really a video")

        table = load_telemetry(video, cfg)
        assert table.is_empty
        assert table.scale_free
        # The scale-free path must survive every downstream query it will face.
        assert table.summary()["scale_free"] is True
        interpolated = table.at_times([0.0, 1.0, 2.0])
        assert len(interpolated) == 3
        assert interpolated["lat"].isna().all()

    def test_secondary_source_fills_gaps_only(self, tmp_path, cfg):
        video = tmp_path / "DJI_0020.MP4"
        video.write_bytes(b"x")
        # The legacy SRT has no barometric altitude; the CSV does.
        fixtures.write_srt_legacy(tmp_path / "DJI_0020.SRT", frames=20)
        fixtures.write_flight_csv(tmp_path / "DJI_0020.csv", frames=20)

        table = load_telemetry(video, cfg)
        assert table.has_baro, "CSV should have filled the altitude the SRT lacked"
        # The SRT stays primary where both have a value.
        assert table.frame["alt_gps"].iloc[0] == pytest.approx(100.5)


class TestInterpolation:
    def test_angles_take_the_short_way_round(self, tmp_path):
        table = TelemetryTable.from_records(
            [{"t": 0.0, "lat": 1.0, "lon": 2.0, "yaw": 359.0},
             {"t": 1.0, "lat": 1.0, "lon": 2.0, "yaw": 1.0}],
            source="test",
        )
        midpoint = float(table.at_times([0.5])["yaw"].iloc[0])
        # Naive linear interpolation would say 180 degrees — the wrong way round.
        assert min(abs(midpoint - 0.0), abs(midpoint - 360.0)) < 1e-6

    def test_no_extrapolation_past_the_measured_range(self, tmp_path):
        table = TelemetryTable.from_records(
            [{"t": 1.0, "lat": 1.0, "lon": 2.0}, {"t": 2.0, "lat": 1.1, "lon": 2.1}], source="test"
        )
        out = table.at_times([0.0, 1.5, 3.0])
        assert np.isnan(out["lat"].iloc[0])
        assert out["lat"].iloc[1] == pytest.approx(1.05)
        assert np.isnan(out["lat"].iloc[2])

    def test_interpolated_rows_are_flagged(self):
        table = TelemetryTable.from_records(
            [{"t": 0.0, "lat": 1.0, "lon": 2.0}, {"t": 1.0, "lat": 1.1, "lon": 2.1}], source="test"
        )
        flags = table.at_times([0.0, 0.5])["valid_flags"].to_numpy()
        assert not flags[0] & int(TelemetryFlags.INTERPOLATED)
        assert flags[1] & int(TelemetryFlags.INTERPOLATED)


class TestEnu:
    def test_round_trip_distance_is_metric(self):
        # One degree of latitude is ~111 km; this is the sanity check that the
        # whole metric-accuracy claim rests on.
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 0.0])
        alt = np.array([0.0, 0.0])
        enu, _ = enu_from_geodetic(lat, lon, alt)
        assert enu[1, 1] == pytest.approx(110574, rel=0.01)

    def test_track_length(self, tmp_path):
        path = fixtures.write_srt_modern(tmp_path / "t.srt", frames=61, fps=30.0, speed_mps=5.0)
        table = parse_dji_srt(path)
        # 60 frames at 30 fps is 2 s; at 5 m/s that is 10 m of flight.
        assert track_length_m(table) == pytest.approx(10.0, rel=0.05)


class TestGpsFiltering:
    def _clean_table(self, n: int = 60) -> TelemetryTable:
        records = []
        for i in range(n):
            records.append(
                {
                    "t": i * 0.1,
                    "lat": 12.9716 + i * 1e-5,
                    "lon": 77.5946,
                    "alt_gps": 920.0,
                    "alt_baro": 60.0,
                }
            )
        return TelemetryTable.from_records(records, source="test")

    def test_gross_outlier_is_rejected(self, cfg):
        table = self._clean_table()
        table.frame.loc[30, "lat"] = 13.5    # ~58 km jump in 0.1 s
        filtered, report = filter_telemetry(table, cfg)

        assert report.envelope_outliers + report.median_outliers >= 1
        flags = filtered.frame["valid_flags"].to_numpy()
        assert flags[30] & int(TelemetryFlags.GPS_OUTLIER)
        # The smoothed track must not follow the spike.
        assert abs(filtered.frame["lat"].iloc[30] - 12.9719) < 0.01

    def test_an_outlier_does_not_condemn_its_neighbour(self, cfg):
        table = self._clean_table()
        table.frame.loc[20, "lat"] = 13.5
        filtered, _ = filter_telemetry(table, cfg)
        flags = filtered.frame["valid_flags"].to_numpy()
        assert flags[20] & int(TelemetryFlags.GPS_OUTLIER)
        assert not flags[21] & int(TelemetryFlags.GPS_OUTLIER)

    def test_noise_is_smoothed_toward_the_truth(self, cfg):
        rng = np.random.default_rng(3)
        table = self._clean_table()
        truth = table.frame["lat"].to_numpy(dtype=float).copy()
        table.frame["lat"] = truth + rng.normal(0, 2e-5, size=len(truth))

        filtered, _ = filter_telemetry(table, cfg)
        noisy_error = np.abs(table.frame["lat"].to_numpy() - truth).mean()
        filtered_error = np.abs(filtered.frame["lat"].to_numpy() - truth).mean()
        assert filtered_error < noisy_error

    def test_baro_supplies_shape_and_gps_supplies_datum(self, cfg):
        table = self._clean_table()
        n = len(table.frame)
        # Barometer: precise relative profile. GPS: same profile, noisy, offset.
        profile = np.linspace(60.0, 80.0, n)
        rng = np.random.default_rng(5)
        table.frame["alt_baro"] = profile
        table.frame["alt_gps"] = profile + 860.0 + rng.normal(0, 4.0, size=n)

        filtered, report = filter_telemetry(table, cfg)
        assert report.altitude_source == "baro+gps_complementary"
        assert report.baro_gps_offset_m == pytest.approx(860.0, abs=3.0)
        fused = filtered.frame["alt_gps"].to_numpy()
        raw_error = np.abs(table.frame["alt_gps"].to_numpy() - (profile + 860.0)).mean()
        fused_error = np.abs(fused - (profile + 860.0)).mean()
        assert fused_error < raw_error

    def test_missing_baro_degrades_to_gps(self, cfg):
        table = self._clean_table()
        table.frame["alt_baro"] = np.nan
        _, report = filter_telemetry(table, cfg)
        assert report.altitude_source == "gps"

    def test_scale_free_input_passes_through(self, cfg):
        table = TelemetryTable.empty("none")
        filtered, report = filter_telemetry(table, cfg)
        assert filtered.is_empty
        assert report.input_fixes == 0
        assert any("scale-free" in note for note in report.notes)

    def test_rtk_boosts_the_gps_weight(self, cfg):
        table = self._clean_table()
        table.frame["valid_flags"] = table.frame["valid_flags"] | int(TelemetryFlags.HAS_RTK)
        _, report = filter_telemetry(table, cfg)
        assert report.rtk_detected is True
        assert report.gps_weight == cfg.condition.gps.rtk.weight_boost
