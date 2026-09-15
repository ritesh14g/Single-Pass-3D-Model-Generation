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


def test_content_invariant_along_the_flight_line_is_not_believed(cfg):
    # The aperture problem: a road or river running with the flight line looks
    # the same at any along-track offset, so along-track overlap cannot be
    # measured from it. Found by the Stage Lab ground-truth scorecard, where it
    # let the selector skip 92 frames on a "0.87 overlap" between disjoint views.
    rng = np.random.default_rng(4)
    rows = rng.integers(40, 220, size=(240, 1), dtype=np.uint8)
    stripes = np.repeat(rows, 320, axis=1)          # constant along x
    shifted = np.roll(stripes, 60, axis=1)          # identical after any x-shift
    estimate = estimate_overlap(stripes, shifted, cfg.get_path("ingest.frame_selection.flow"))
    assert not estimate.reliable


def test_selection_tracks_ground_truth_on_road_aligned_synthetic_flight(tmp_path, cfg):
    from src.qa.synthetic import generate_synthetic_flight, load_ground_truth, true_pair_overlap

    video, _, _ = generate_synthetic_flight(tmp_path, frames=150, blur_every=11)
    truth = load_ground_truth(video)
    with VideoReader(video) as reader:
        profile = profile_video_blur(reader, cfg)
        selection = select_frames(reader, cfg, profile)
    idx = selection.indices
    errors = [abs(f.overlap_prev - true_pair_overlap(truth, a, b))
              for f, a, b in zip(selection.frames[1:], idx[:-1], idx[1:])]
    assert max(errors) < 0.1, f"worst overlap error {max(errors):.3f}"
    assert max(np.diff(idx)) <= 30, "selector skipped far past the overlap region"


class TestStrideBoundsFollowFrameRate:
    """The guard rails are a duration, not a frame count (see `_stride_bounds`).

    A cap stated in frames means 1.5 s at 60 fps and 3 s at 30 fps. On a real
    1080p60 clip that turned the cap into the binding constraint on overlap:
    every kept pair came out at 0.997 against a 0.70-0.80 target because 90
    frames of that scene is not a meaningful step.
    """

    def test_the_cap_converts_to_the_same_duration_at_any_frame_rate(self, cfg):
        from src.ingest.frame_selector import _stride_bounds

        sel = cfg.get_path("ingest.frame_selection")
        seconds = float(sel["max_stride_seconds"])
        _, cap_30 = _stride_bounds(sel, fps=30.0)
        _, cap_60 = _stride_bounds(sel, fps=59.94)
        assert cap_30 == pytest.approx(seconds * 30.0, abs=1)
        assert cap_60 == pytest.approx(seconds * 59.94, abs=1)
        assert cap_60 > cap_30

    def test_frame_counts_are_the_fallback_when_seconds_are_unset(self, cfg):
        from src.ingest.frame_selector import _stride_bounds

        plain = cfg.merged({"ingest": {"frame_selection": {
            "min_stride_seconds": None, "max_stride_seconds": None}}})
        sel = plain.get_path("ingest.frame_selection")
        assert _stride_bounds(sel, fps=59.94) == (int(sel["min_stride_frames"]),
                                                  int(sel["max_stride_frames"]))

    def test_unknown_frame_rate_falls_back_to_frame_counts(self, cfg):
        from src.ingest.frame_selector import _stride_bounds

        sel = cfg.get_path("ingest.frame_selection")
        assert _stride_bounds(sel, fps=0.0) == (int(sel["min_stride_frames"]),
                                                int(sel["max_stride_frames"]))


class TestRollingBlurBaseline:
    """Issue S1-7: a change of ground cover must not read as blur.

    Sharpness is scene-dependent, so a whole-video threshold rejected 450
    consecutive in-focus frames where a real clip flew over pasture. The
    baseline follows the scene; what must survive that is the ability to still
    reject frames that are soft *for where they are*.
    """

    def _baseline(self, cfg, video_scores):
        from src.condition.blur import BlurProfile, RollingBaseline

        profile = BlurProfile.from_samples(video_scores, cfg)
        baseline = RollingBaseline.from_config(profile, cfg)
        assert baseline is not None, "rolling baseline is enabled in the default config"
        return profile, baseline

    def _run(self, profile, baseline, scores):
        rejected = []
        for i, score in enumerate(scores):
            reject, _, _ = baseline.thresholds(profile)
            baseline.observe(score)
            if score < reject:
                rejected.append(i)
        return rejected

    def _flight(self):
        # A clip that flies from textured ground (~2800) onto plain pasture
        # (~1400). Proportions matter: the real clip was ~2/3 textured, which
        # puts the whole-video outlier line above the pasture population.
        rng = np.random.default_rng(7)
        return rng, list(rng.normal(2800, 60, 200)), list(rng.normal(1400, 40, 100))

    def test_the_whole_video_threshold_reproduces_the_bug(self, cfg):
        _, textured, pasture = self._flight()
        profile, _ = self._baseline(cfg, textured + pasture)
        assert profile.reject_threshold > max(pasture)

    def test_a_gradual_change_of_ground_cover_is_not_rejected(self, cfg):
        # What a drone actually sees: content changes over a second or two.
        rng, textured, pasture = self._flight()
        ramp = list(np.linspace(2800, 1400, 60) + rng.normal(0, 50, 60))
        profile, baseline = self._baseline(cfg, textured + pasture)
        rejected = self._run(profile, baseline, textured + ramp + pasture)
        assert rejected == [], f"{len(rejected)} in-focus frames rejected across the transition"

    def test_a_hard_cut_costs_at_most_half_a_window(self, cfg):
        # The worst case for any trailing baseline: an instant 50% step. It
        # rejects until the window's median crosses over, and no longer.
        _, textured, pasture = self._flight()
        profile, baseline = self._baseline(cfg, textured + pasture)
        rejected = self._run(profile, baseline, textured + pasture)
        assert len(rejected) <= baseline.window // 2 + 1, f"{len(rejected)} frames lost to one scene cut"
        assert all(i >= len(textured) for i in rejected), "textured frames rejected before the cut"

    def test_genuine_dips_inside_a_low_texture_stretch_are_rejected(self, cfg):
        _, textured, pasture = self._flight()
        profile, baseline = self._baseline(cfg, textured + pasture)
        dips = [0.4 * 1400.0] * 5
        scores = pasture[:50] + dips + pasture[50:]
        rejected = self._run(profile, baseline, scores)
        assert set(range(50, 55)) <= set(rejected), "blurred frames survived the local test"
        warmup = [i for i in rejected if i < 50]
        assert len(warmup) <= baseline.min_samples, f"{len(warmup)} warm-up frames rejected"

    def test_the_hard_floor_survives_a_long_blurred_run(self, cfg):
        # The failure mode of any adaptive threshold: feed it nothing but blur
        # and it normalises the blur. The floor is a fraction of the whole-video
        # median, so the window cannot argue its way below it.
        rng = np.random.default_rng(7)
        profile, baseline = self._baseline(cfg, list(rng.normal(2000, 80, 300)))
        for _ in range(baseline.window * 2):
            baseline.observe(30.0)
        reject, _, _ = baseline.thresholds(profile)
        assert reject == pytest.approx(baseline.hard_floor)
        assert 30.0 < baseline.hard_floor, "a wholly blurred window must still reject"

    def test_detected_motion_smear_is_held_to_the_whole_video_line(self, cfg):
        # Anisotropy is a detection. A window diluted by a nearby soft stretch
        # must not lower the bar for a frame already identified as smeared —
        # found on the blurry synthetic flight, where frame 45 (anisotropy 9.07,
        # score 147) passed a local line of ~150 against a whole-video 188.7.
        rng = np.random.default_rng(7)
        profile, baseline = self._baseline(cfg, list(rng.normal(250, 20, 200)))
        for score in rng.normal(110, 5, baseline.window):
            baseline.observe(score)
        _, directional, _ = baseline.thresholds(profile)
        assert directional >= profile.directional_reject_threshold
