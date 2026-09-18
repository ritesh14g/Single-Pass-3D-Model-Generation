"""MISB ST 0601 KLV telemetry — spec §4.2, the STANAG 4609 ingest path.

There is no sample ISR clip in the repo, so these tests *encode* ST 0601 local
sets to the standard's own fixed-point rules, wrap them in MPEG-2 TS packets,
and assert the parser recovers the values that went in. An encoder written
against the spec is a real test of a decoder written against the spec: the two
halves are independent, so a scaling error in either one shows up as a mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import load_config
from src.ingest.klv import (
    UAS_LS_KEY,
    depacketize_ts,
    find_klv_source,
    load_klv_for_video,
    parse_klv,
)
from src.ingest.telemetry import load_telemetry

# A short nadir survey leg, in the units a ground station would report.
FLIGHT = [
    # (unix_us,          lat,        lon,       alt,   heading, pitch, roll)
    (1_600_000_000_000_000, 12.971600, 77.594600, 450.0, 90.0,  -5.0,  2.0),
    (1_600_000_000_100_000, 12.971650, 77.594700, 450.5, 90.5,  -5.2,  1.8),
    (1_600_000_000_200_000, 12.971700, 77.594800, 451.0, 91.0,  -5.4,  1.6),
]


# --------------------------------------------------------------------------
# An ST 0601 encoder, written from the standard's scaling rules
# --------------------------------------------------------------------------
def _ber_length(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def _bcc_16(buf: bytes) -> int:
    """ST 0601's checksum, written from the standard's reference loop."""
    bcc = 0
    for i, byte in enumerate(buf):
        bcc += byte << (8 * ((i + 1) % 2))
    return bcc & 0xFFFF


def _item(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _ber_length(len(value)) + value


def _u(value: float, low: float, high: float, size: int) -> bytes:
    span = (1 << (8 * size)) - 1
    raw = int(round((value - low) * span / (high - low)))
    return max(0, min(span, raw)).to_bytes(size, "big")


def _s(value: float, limit: float, size: int) -> bytes:
    peak = (1 << (8 * size - 1)) - 1
    return int(round(value * peak / limit)).to_bytes(size, "big", signed=True)


def build_packet(unix_us, lat, lon, alt, heading, pitch, roll, extras=b"") -> bytes:
    """One ST 0601 local set, checksum included."""
    body = b"".join([
        _item(2, unix_us.to_bytes(8, "big")),
        _item(5, _u(heading, 0.0, 360.0, 2)),
        _item(6, _s(pitch, 20.0, 2)),
        _item(7, _s(roll, 50.0, 2)),
        _item(13, _s(lat, 90.0, 4)),
        _item(14, _s(lon, 180.0, 4)),
        _item(15, _u(alt, -900.0, 19000.0, 2)),
        _item(65, bytes([13])),
    ]) + extras
    checksum_prefix = bytes([1]) + _ber_length(2)
    header = UAS_LS_KEY + _ber_length(len(body) + len(checksum_prefix) + 2)
    checksum = _bcc_16(header + body + checksum_prefix)
    return header + body + checksum_prefix + checksum.to_bytes(2, "big")


def build_stream(flight=FLIGHT, extras=b"") -> bytes:
    return b"".join(build_packet(*row, extras=extras) for row in flight)


def wrap_in_ts(payload: bytes, pid: int = 0x0201) -> bytes:
    """Packetize into 188-byte TS packets on one PID, plus a decoy video PID."""
    packets = []
    counter = 0
    for start in range(0, len(payload), 184):
        chunk = payload[start:start + 184]
        pusi = 0x40 if start == 0 else 0x00
        header = bytes([
            0x47,
            pusi | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            0x10 | (counter & 0x0F),
        ])
        packets.append(header + chunk + b"\xFF" * (184 - len(chunk)))
        counter += 1
    # A second PID carrying non-KLV bytes, so PID selection is actually tested.
    decoy = bytes([0x47, 0x01, 0x00, 0x10]) + b"\x00" * 184
    return decoy + b"".join(packets) + decoy


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def klv_file(tmp_path):
    path = tmp_path / "mission.klv"
    path.write_bytes(build_stream())
    return path


@pytest.fixture
def ts_file(tmp_path):
    path = tmp_path / "mission.ts"
    path.write_bytes(wrap_in_ts(build_stream()))
    return path


# --------------------------------------------------------------------------
class TestRawStream:
    def test_every_packet_becomes_a_row(self, klv_file):
        assert len(parse_klv(klv_file)) == len(FLIGHT)

    def test_position_round_trips(self, klv_file):
        frame = parse_klv(klv_file).frame
        # int32 over +/-90 deg is ~4e-8 deg; well inside a millimetre on the ground.
        assert frame["lat"].to_numpy() == pytest.approx([r[1] for r in FLIGHT], abs=1e-6)
        assert frame["lon"].to_numpy() == pytest.approx([r[2] for r in FLIGHT], abs=1e-6)
        # Altitude is uint16 over a 19900 m range: one count is ~0.3 m.
        assert frame["alt_gps"].to_numpy() == pytest.approx([r[3] for r in FLIGHT], abs=0.5)

    def test_attitude_round_trips(self, klv_file):
        frame = parse_klv(klv_file).frame
        assert frame["yaw"].to_numpy() == pytest.approx([r[4] for r in FLIGHT], abs=0.01)
        assert frame["pitch"].to_numpy() == pytest.approx([r[5] for r in FLIGHT], abs=0.01)
        assert frame["roll"].to_numpy() == pytest.approx([r[6] for r in FLIGHT], abs=0.01)

    def test_time_is_seconds_from_the_first_packet(self, klv_file):
        table = parse_klv(klv_file)
        assert table.frame["t"].to_numpy() == pytest.approx([0.0, 0.1, 0.2], abs=1e-6)

    def test_absolute_utc_is_preserved_as_wall_time(self, klv_file):
        table = parse_klv(klv_file)
        assert table.wall_time is not None
        assert str(table.wall_time.iloc[0]).startswith("2020-09-13")

    def test_source_and_version_are_recorded(self, klv_file):
        table = parse_klv(klv_file)
        assert table.source == "klv:mission.klv"
        assert any("ST 0601 version 13" in note for note in table.notes)

    def test_reports_gps_and_attitude(self, klv_file):
        table = parse_klv(klv_file)
        assert table.has_gps and table.has_attitude
        # KLV carries no barometric altitude; the pipeline must see that honestly.
        assert not table.has_baro


class TestTransportStream:
    def test_ts_and_raw_stream_agree(self, ts_file, klv_file):
        from_ts = parse_klv(ts_file).frame
        from_raw = parse_klv(klv_file).frame
        assert from_ts["lat"].to_numpy() == pytest.approx(from_raw["lat"].to_numpy())
        assert len(from_ts) == len(FLIGHT)

    def test_depacketizer_separates_pids(self, ts_file):
        streams = depacketize_ts(ts_file.read_bytes())
        carrying = [pid for pid, blob in streams.items() if UAS_LS_KEY in blob]
        assert carrying == [0x0201]

    def test_packets_split_across_ts_boundaries_survive(self, tmp_path):
        """A local set is ~60 bytes, so 10 of them straddle many 184-byte payloads."""
        flight = [(1_600_000_000_000_000 + i * 100_000, 12.97 + i * 1e-4, 77.59,
                   450.0, 90.0, -5.0, 2.0) for i in range(10)]
        path = tmp_path / "long.ts"
        path.write_bytes(wrap_in_ts(build_stream(flight)))
        assert len(parse_klv(path)) == 10


class TestGeoreferencingExtras:
    def _extras(self):
        return b"".join([
            _item(23, _s(12.9720, 90.0, 4)),        # frame centre latitude
            _item(24, _s(77.5950, 180.0, 4)),       # frame centre longitude
            _item(25, _u(280.0, -900.0, 19000.0, 2)),
            _item(21, _u(1500.0, 0.0, 5_000_000.0, 4)),   # slant range
            _item(16, _u(35.0, 0.0, 180.0, 2)),     # horizontal FOV
            _item(18, _u(270.0, 0.0, 360.0, 4)),    # sensor relative azimuth
        ])

    def test_frame_centre_and_pointing_are_carried(self, tmp_path):
        path = tmp_path / "isr.klv"
        path.write_bytes(build_stream(extras=self._extras()))
        frame = parse_klv(path).frame
        assert frame["frame_center_lat"].iloc[0] == pytest.approx(12.9720, abs=1e-6)
        assert frame["frame_center_lon"].iloc[0] == pytest.approx(77.5950, abs=1e-6)
        assert frame["slant_range_m"].iloc[0] == pytest.approx(1500.0, abs=1.0)
        assert frame["hfov_deg"].iloc[0] == pytest.approx(35.0, abs=0.01)
        assert frame["sensor_rel_az_deg"].iloc[0] == pytest.approx(270.0, abs=0.01)


class TestMalformedInput:
    def test_out_of_range_marker_decodes_to_nan_not_an_angle(self, tmp_path):
        """ST 0601 reserves 0x8000 for "no reading"; -20 deg would be a lie."""
        body = build_packet(*FLIGHT[0])
        # Replace the pitch value (tag 6, 2 bytes) with the error marker.
        good = _item(6, _s(FLIGHT[0][5], 20.0, 2))
        assert good in body
        path = tmp_path / "bad_pitch.klv"
        path.write_bytes(body.replace(good, _item(6, b"\x80\x00")))
        assert np.isnan(parse_klv(path).frame["pitch"].iloc[0])

    def test_a_file_with_no_klv_is_empty_not_an_error(self, tmp_path):
        path = tmp_path / "nothing.klv"
        path.write_bytes(b"\x00" * 4096)
        table = parse_klv(path)
        assert table.is_empty
        assert "no ST 0601 universal key" in table.notes[0]

    def test_a_ts_without_a_klv_pid_is_empty_not_an_error(self, tmp_path):
        path = tmp_path / "video_only.ts"
        path.write_bytes(bytes([0x47, 0x01, 0x00, 0x10]) + b"\x00" * 184 +
                         bytes([0x47, 0x01, 0x00, 0x11]) + b"\x00" * 184)
        assert parse_klv(path).is_empty

    def test_corrupt_checksum_is_counted(self, tmp_path):
        stream = bytearray(build_stream())
        stream[-1] ^= 0xFF                              # break the last checksum
        path = tmp_path / "corrupt.klv"
        path.write_bytes(bytes(stream))
        table = parse_klv(path)
        assert len(table) == len(FLIGHT)                # lenient by default
        assert any("failed the ST 0601 checksum" in n for n in table.notes)

    def test_require_checksum_rejects_the_stream(self, tmp_path):
        stream = bytearray(build_stream())
        stream[-1] ^= 0xFF
        path = tmp_path / "corrupt.klv"
        path.write_bytes(bytes(stream))
        assert parse_klv(path, require_checksum=True).is_empty

    def test_truncated_final_packet_does_not_lose_the_earlier_ones(self, tmp_path):
        stream = build_stream()
        path = tmp_path / "cut.klv"
        path.write_bytes(stream[:-8])
        assert len(parse_klv(path)) == len(FLIGHT) - 1


class TestSourceResolution:
    def test_a_ts_video_is_its_own_klv_source(self, ts_file):
        assert find_klv_source(ts_file, [".klv"]) == ts_file

    def test_a_sidecar_is_found_next_to_an_mp4(self, tmp_path, klv_file):
        video = tmp_path / "mission.mp4"
        video.write_bytes(b"\x00" * 64)
        assert find_klv_source(video, [".klv", ".ts"]) == klv_file

    def test_a_bare_mp4_has_no_klv_source(self, tmp_path):
        video = tmp_path / "dji.mp4"
        video.write_bytes(b"\x00" * 64)
        assert find_klv_source(video, [".klv", ".ts"]) is None

    def test_mp4_is_not_scanned_unless_asked(self, tmp_path):
        video = tmp_path / "isr.mp4"
        video.write_bytes(build_stream())            # KLV really is in there
        assert load_klv_for_video(video, {"enabled": True, "scan_mp4": False}) is None
        table = load_klv_for_video(video, {"enabled": True, "scan_mp4": True})
        assert table is not None and len(table) == len(FLIGHT)

    def test_disabled_in_config_means_no_read(self, ts_file):
        assert load_klv_for_video(ts_file, {"enabled": False}) is None


class TestPipelineIntegration:
    def test_klv_is_the_first_source_tried(self, cfg):
        assert list(cfg.get_path("ingest.telemetry.sources"))[0] == "klv"

    def test_load_telemetry_picks_up_klv_without_a_sidecar(self, ts_file, cfg):
        table = load_telemetry(ts_file, cfg)
        assert table.source.startswith("klv:")
        assert len(table) == len(FLIGHT)
        assert table.has_gps

    def test_dji_style_mp4_still_falls_through_to_srt(self, tmp_path, cfg):
        """The regression that matters: adding KLV must not break DJI ingest."""
        video = tmp_path / "dji.mp4"
        video.write_bytes(b"\x00" * 64)
        srt = tmp_path / "dji.srt"
        srt.write_text(
            "1\n00:00:00,000 --> 00:00:00,033\n"
            "[latitude: 12.9716] [longitude: 77.5946] [rel_alt: 45.0]\n\n"
            "2\n00:00:00,033 --> 00:00:00,066\n"
            "[latitude: 12.9717] [longitude: 77.5947] [rel_alt: 45.2]\n\n",
            encoding="utf-8",
        )
        table = load_telemetry(video, cfg)
        assert table.source.startswith("srt:")
        assert table.has_gps


# ---------------------------------------------------------------------------
# A real packet, lifted from a real STANAG 4609 clip
#
# Everything above encodes its own test data, which validates the fixed-point
# scaling (encoder and decoder are independent implementations of the standard)
# but cannot validate the checksum: the first version of these tests used a
# plain byte sum on BOTH sides, so the round-trip agreed with itself while
# rejecting every real packet in all eight MISB sample clips. ST 0601 actually
# specifies bcc_16, which adds bytes into alternating halves of the accumulator.
#
# This is the packet that caught it: local set 0 of klv_metadata_test_sync.ts
# (QGISFMV sample media). A vector from outside this repo is the only thing that
# can catch a mistake shared by the encoder and the decoder.
# ---------------------------------------------------------------------------
REAL_PACKET = bytes.fromhex(
    "060e2b34020b01010e0103010100000081dc02080005602866187a93480800055df6b0d09f"
    "d60a085072656461746f720c0e47656f64657469632057475338340502a2cf060200000702"
    "00460d0431d18bf60e04c7b07cec0f023aa110020acf1102081b120490c87b051304e2127e"
    "951404000000000b07454f204e6f73651602123b15040046461e170431d46e921804c7b830"
    "1119020c71030a313233343536372d38391a0207011b0205e11c02fbd31d02091f1e02f9a7"
    "1f02fa33200203a32102f760410106302001010102010203052f2f555341050006035553"
    "410c01020d0355534116020d090102ff59"
)


class TestRealPacket:
    @pytest.fixture
    def real_file(self, tmp_path):
        path = tmp_path / "real.klv"
        path.write_bytes(REAL_PACKET)
        return path

    def test_it_parses(self, real_file):
        assert len(parse_klv(real_file)) == 1

    def test_its_checksum_validates(self, real_file):
        """The regression: bcc_16, not a plain byte sum."""
        table = parse_klv(real_file, require_checksum=True)
        assert len(table) == 1
        assert not any("checksum" in note for note in table.notes)

    def test_plain_byte_sum_would_reject_it(self):
        """Pins why: the two algorithms disagree on this real packet."""
        from src.ingest.klv import _checksum, _read_ber_length
        length, consumed = _read_ber_length(REAL_PACKET, len(UAS_LS_KEY))
        upto = len(REAL_PACKET) - 2            # everything before the checksum value
        stated = int.from_bytes(REAL_PACKET[-2:], "big")
        assert _checksum(REAL_PACKET[:upto]) == stated
        assert (sum(REAL_PACKET[:upto]) & 0xFFFF) != stated

    def test_decoded_values_are_a_plausible_flight(self, real_file):
        frame = parse_klv(real_file).frame
        assert -90.0 <= frame["lat"].iloc[0] <= 90.0
        assert -180.0 <= frame["lon"].iloc[0] <= 180.0
        assert -900.0 <= frame["alt_gps"].iloc[0] <= 19000.0
        assert 0.0 <= frame["yaw"].iloc[0] <= 360.0
        # This clip is the MISB sync test: it carries frame centre and slant range.
        assert "frame_center_lat" in frame.columns
        assert frame["slant_range_m"].iloc[0] > 0


class TestMisnamedContainers:
    """Extensions lie: all eight MISB sample clips are TS, named .ts/.mp4/.mpeg4/.H264."""

    @pytest.mark.parametrize("name", ["clip.H264", "clip.mpeg4", "clip.mp4", "clip.bin"])
    def test_a_ts_is_found_whatever_it_is_called(self, tmp_path, name):
        path = tmp_path / name
        path.write_bytes(wrap_in_ts(build_stream()))
        table = load_klv_for_video(path, {"enabled": True, "scan_mp4": False})
        assert table is not None and len(table) == len(FLIGHT)

    def test_a_real_mp4_is_still_not_scanned_by_default(self, tmp_path):
        """The sniff must not turn the opt-in MP4 scan back on."""
        path = tmp_path / "dji.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + build_stream())
        assert load_klv_for_video(path, {"enabled": True, "scan_mp4": False}) is None
        assert load_klv_for_video(path, {"enabled": True, "scan_mp4": True}) is not None

    def test_sniffing_reads_only_the_header(self, tmp_path):
        from src.ingest.klv import file_is_ts
        path = tmp_path / "big.mp4"
        path.write_bytes(b"\x00" * 10_000_000)
        assert file_is_ts(path) is False
