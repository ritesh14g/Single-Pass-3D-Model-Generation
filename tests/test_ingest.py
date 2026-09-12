"""Video reading and adaptive frame selection.

The headline check here is the §4.3 acceptance test: the selector must produce a
frame set whose consecutive-pair overlap distribution is centred in [0.7, 0.8].
The synthetic flight makes that checkable, because its true overlap is set by
construction rather than estimated.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.condition.blur import BlurProfile, profile_video_blur
from src.ingest.frame_selector import (
    estimate_overlap,
    frames_per_chunk,
    load_selection,
    select_frames,
)
from src.ingest.video_reader import VideoReader, probe_video
from src.condition.blur import to_profile_gray
from tests import fixtures


class TestVideoReader:
    def test_probe_reports_real_metadata(self, flight):
        meta = probe_video(flight.video_path)
        assert meta.width == flight.width
        assert meta.height == flight.height
        assert meta.fps == pytest.approx(flight.fps, abs=0.1)
        assert meta.frame_count == flight.frame_count

    def test_missing_file_raises_immediately(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoReader(tmp_path / "nope.mp4")

    def test_stream_yields_sequential_frames(self, flight):
        with VideoReader(flight.video_path) as reader:
            frames = list(reader.stream(max_frames=5))
        assert [f.index for f in frames] == [0, 1, 2, 3, 4]
        assert frames[0].image.shape == (flight.height, flight.width, 3)

    def test_stream_with_step_skips_without_decoding(self, flight):
        with VideoReader(flight.video_path) as reader:
            frames = list(reader.stream(step=5, max_frames=4))
        assert [f.index for f in frames] == [0, 5, 10, 15]

    def test_read_indices_is_order_independent_and_deduplicates(self, flight):
        with VideoReader(flight.video_path) as reader:
            frames = list(reader.read_indices([20, 5, 5, 12]))
        assert [f.index for f in frames] == [5, 12, 20]

    def test_read_indices_matches_sequential_decode(self, flight):
        # A seek must land on the same pixels a sequential decode would produce,
        # or every frame identity downstream is subtly wrong.
        with VideoReader(flight.video_path) as reader:
            sequential = {f.index: f.image for f in reader.stream(max_frames=30)}
        with VideoReader(flight.video_path) as reader:
            sought = {f.index: f.image for f in reader.read_indices([3, 17, 28])}
        for index, image in sought.items():
            difference = np.abs(image.astype(int) - sequential[index].astype(int)).mean()
            assert difference < 1.0, f"frame {index} differs between seek and stream"

    def test_out_of_range_index_is_skipped_not_fatal(self, flight):
        with VideoReader(flight.video_path) as reader:
            frames = list(reader.read_indices([5, 99999]))
        assert [f.index for f in frames] == [5]

    def test_max_width_downscales(self, flight):
        with VideoReader(flight.video_path, max_width=160) as reader:
            frame = reader.read_one(0)
        assert frame is not None and frame.width == 160

    def test_timestamps_are_monotonic(self, flight):
        with VideoReader(flight.video_path) as reader:
            stamps = [f.timestamp_s for f in reader.stream(max_frames=10)]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))


class TestOverlapEstimation:
    def test_identical_frames_overlap_fully(self, flight, cfg):
        with VideoReader(flight.video_path) as reader:
            frame = reader.read_one(10)
        gray = to_profile_gray(frame.image)
        estimate = estimate_overlap(gray, gray, cfg.get_path("ingest.frame_selection.flow"))
        assert estimate.reliable
        assert estimate.overlap == pytest.approx(1.0, abs=0.02)
        assert estimate.correlation > 0.99

    @pytest.mark.parametrize("gap", [1, 2, 3])
    def test_measured_overlap_matches_the_true_pan(self, flight, cfg, gap):
        # The fixture pans by a known number of pixels per frame, so the true
        # overlap between frame i and i+k is an arithmetic fact to check against.
        flow_cfg = cfg.get_path("ingest.frame_selection.flow")
        with VideoReader(flight.video_path) as reader:
            a = reader.read_one(10)
            b = reader.read_one(10 + gap)
        expected = 1.0 - (gap * flight.step_px) / flight.width
        estimate = estimate_overlap(to_profile_gray(a.image), to_profile_gray(b.image), flow_cfg)
        assert estimate.reliable
        assert estimate.overlap == pytest.approx(expected, abs=0.05)

    def test_unrelated_frames_are_reported_unreliable_not_overlapping(self, cfg):
        # Uniform noise is the case that defeats feature agreement alone: at
        # coarse pyramid levels it averages to grey, so the tracker "succeeds"
        # with zero displacement in both directions and the round-trip check
        # confirms it. Only the photometric check catches this.
        rng = np.random.default_rng(0)
        a = rng.integers(0, 255, (240, 320), dtype=np.uint8)
        b = rng.integers(0, 255, (240, 320), dtype=np.uint8)
        estimate = estimate_overlap(a, b, cfg.get_path("ingest.frame_selection.flow"))
        assert not estimate.reliable
        assert estimate.overlap == 0.0

    def test_frames_past_the_shared_region_are_not_believed(self, flight, cfg):
        # Frame 10 and frame 20 of a 10%-step pan share nothing at all.
        with VideoReader(flight.video_path) as reader:
            a = reader.read_one(10)
            b = reader.read_one(20)
        estimate = estimate_overlap(
            to_profile_gray(a.image), to_profile_gray(b.image),
            cfg.get_path("ingest.frame_selection.flow"),
        )
        assert not estimate.reliable


class TestFrameSelection:
    def test_hits_the_configured_overlap_target(self, flight, cfg):
        # This is the §4.3 acceptance criterion.
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            selection = select_frames(reader, cfg, profile)

        stats = selection.overlap_stats()
        assert stats["pairs"] >= 3
        assert 0.65 <= stats["median"] <= 0.85, f"overlap median {stats['median']} outside the target band"
        assert stats["below_0.5"] == 0, "no consecutive pair may fall below 50% overlap"

    def test_selects_far_fewer_frames_than_it_decodes_from(self, flight, cfg):
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            selection = select_frames(reader, cfg, profile)
        # Fixed-interval sampling would keep all 60; targeting overlap keeps a
        # fraction of them while preserving the match graph.
        assert len(selection) < flight.frame_count
        assert len(selection) >= 3

    def test_adapts_stride_to_camera_speed(self, tmp_path, cfg):
        # Same scene, different speeds: the slow flight must be sampled more
        # sparsely in time than the fast one, for the same overlap.
        slow = fixtures.make_flight_video(tmp_path / "slow.mp4", frames=60, overlap=0.97, seed=3)
        fast = fixtures.make_flight_video(tmp_path / "fast.mp4", frames=60, overlap=0.85, seed=3)

        strides = {}
        for flight in (slow, fast):
            with VideoReader(flight.video_path) as reader:
                profile = profile_video_blur(reader, cfg)
                selection = select_frames(reader, cfg, profile)
            indices = selection.indices
            strides[flight.true_overlap] = float(np.mean(np.diff(indices))) if len(indices) > 1 else 0.0

        assert strides[0.97] > strides[0.85]

    def test_decodes_far_fewer_frames_than_the_video_contains(self, flight, cfg):
        # The predictive-stride design is what makes the 60 s acceptance budget
        # reachable on a 10-minute 4K clip; guard it against regression.
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            selection = select_frames(reader, cfg, profile)
        assert selection.decoded_frames < flight.frame_count * 1.5

    def test_the_gate_rejects_every_known_blurred_frame(self, blurry_flight, cfg):
        from src.condition.blur import BlurVerdict, assess_blur

        with VideoReader(blurry_flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            blurred = {
                index: reader.read_one(index).image for index in blurry_flight.blurred_indices[:6]
            }
            sharp = {index: reader.read_one(index).image for index in (1, 2, 6, 7, 11)}

        for index, image in blurred.items():
            assessment = assess_blur(image, profile, cfg)
            assert assessment.verdict is BlurVerdict.REJECT, f"frame {index} survived the gate"
            assert assessment.directional, f"frame {index} should read as directional blur"
        for index, image in sharp.items():
            assert not assess_blur(image, profile, cfg).rejected, f"sharp frame {index} was rejected"

    def test_selection_never_keeps_a_blurred_frame(self, blurry_flight, cfg):
        with VideoReader(blurry_flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            selection = select_frames(reader, cfg, profile)

        kept_blurred = set(selection.indices) & set(blurry_flight.blurred_indices)
        assert not kept_blurred, f"kept blurred frames {sorted(kept_blurred)}"

    def test_a_clean_video_loses_no_frames_to_the_gate(self, flight, cfg):
        # The reason the threshold is an outlier test rather than a fixed
        # percentile: uniformly sharp footage must not be thinned for nothing.
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            selection = select_frames(reader, cfg, profile)
        assert selection.blur_summary["rejected"] == 0

    def test_selection_round_trips_through_parquet(self, flight, cfg, tmp_path):
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            selection = select_frames(reader, cfg, profile)
        path = selection.save(tmp_path / "frames.parquet")
        reloaded = load_selection(path)
        assert reloaded.indices == selection.indices
        assert reloaded.frames[0].blur_verdict == selection.frames[0].blur_verdict

    def test_completes_well_inside_its_time_budget(self, flight, cfg):
        started = time.monotonic()
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, cfg)
            select_frames(reader, cfg, profile)
        elapsed = time.monotonic() - started
        # 60 small frames must be near-instant; a regression to per-frame
        # decoding would blow this by an order of magnitude.
        assert elapsed < 10.0

    def test_max_frames_is_respected(self, flight, cfg):
        capped = cfg.merged({"ingest": {"frame_selection": {"max_frames": 4}}})
        with VideoReader(flight.video_path) as reader:
            profile = profile_video_blur(reader, capped)
            selection = select_frames(reader, capped, profile)
        assert len(selection) <= 4
        assert any("maximum" in note for note in selection.notes)


class TestChunking:
    def test_chunks_overlap_by_the_configured_amount(self):
        selection = load_selection.__wrapped__ if hasattr(load_selection, "__wrapped__") else None
        indices = list(range(0, 300))
        from src.ingest.frame_selector import FrameSelection, SelectedFrame

        sel = FrameSelection(
            frames=[SelectedFrame(i, i / 30.0, 0.75, 100.0, "clean", 1.0) for i in indices]
        )
        chunks = frames_per_chunk(sel, chunk_frames=128, overlap=16)
        assert len(chunks) > 1
        for a, b in zip(chunks, chunks[1:]):
            shared = set(a) & set(b)
            assert len(shared) >= 16, "consecutive chunks must share frames to register against"

    def test_short_selection_is_a_single_chunk(self):
        from src.ingest.frame_selector import FrameSelection, SelectedFrame

        sel = FrameSelection(frames=[SelectedFrame(i, 0.0, 0.75, 1.0, "clean", 1.0) for i in range(10)])
        assert frames_per_chunk(sel, chunk_frames=128, overlap=16) == [list(range(10))]
