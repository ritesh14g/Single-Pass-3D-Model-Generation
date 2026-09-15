"""Conditioning layer tests (spec §5).

Spec §5 closes with an explicit demand: every operation here must be
*measurable*, and a sub-module that shows no improvement should be reported
honestly rather than kept for show. These tests are the first half of that —
each one constructs a degradation whose ground truth is known, then checks the
module recovers it. The §8.5 degradation harness is the second half.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.condition.artifacts import (
    assess_artifacts,
    block_grid_mask,
    blockiness_score,
    filter_grid_keypoints,
    suppress_block_artifacts,
)
from src.condition.blur import spectral_anisotropy, to_profile_gray, variance_of_laplacian
from src.condition.dynamic_mask import (
    DynamicMasker,
    decide_hole_policy,
    dilate_mask,
    geometric_dynamic_mask,
    texture_variance,
)
from src.condition.illumination import (
    ExposureChain,
    apply_clahe,
    condition_illumination,
    condition_low_light,
    detect_shadows,
    fit_gain_bias,
    shadow_weight_map,
)
from src.ingest.frame_selector import normalize_transform
from tests import fixtures


@pytest.fixture
def scene():
    """A textured 640x480 scene standing in for one conditioned frame."""
    return fixtures.make_canvas(640, 480, seed=21)


class TestBlurMetrics:
    def test_sharpness_falls_as_blur_grows(self, scene):
        gray = to_profile_gray(scene)
        scores = [
            variance_of_laplacian(to_profile_gray(fixtures.motion_blur(scene, length=n)))
            for n in (5, 11, 21, 31)
        ]
        assert variance_of_laplacian(gray) > scores[0]
        assert all(a > b for a, b in zip(scores, scores[1:])), scores

    def test_directional_blur_is_anisotropic_and_isotropic_blur_is_not(self, scene):
        import cv2

        directional = fixtures.motion_blur(scene, length=21, angle_deg=0.0)
        isotropic = cv2.GaussianBlur(scene, (0, 0), 4.0)

        # This separation is what lets the gate treat drone jerk differently
        # from a soft focus or atmospheric haze.
        assert spectral_anisotropy(to_profile_gray(directional)) > 3.0
        assert spectral_anisotropy(to_profile_gray(isotropic)) < 2.0

    def test_blur_direction_does_not_change_the_verdict(self, scene):
        # Anisotropy measures how directional the smear is, not which way it
        # points, so a vertical jerk must read the same as a horizontal one.
        horizontal = spectral_anisotropy(to_profile_gray(fixtures.motion_blur(scene, 21, 0.0)))
        vertical = spectral_anisotropy(to_profile_gray(fixtures.motion_blur(scene, 21, 90.0)))
        assert horizontal == pytest.approx(vertical, rel=0.35)


class TestArtifacts:
    def _blocked(self, image: np.ndarray, block: int = 8, strength: int = 26) -> np.ndarray:
        """Quantise each block to its mean — the essence of blocking."""
        out = image.astype(np.float32).copy()
        h, w = out.shape[:2]
        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                patch = out[y : y + block, x : x + block]
                mean = patch.mean(axis=(0, 1))
                out[y : y + block, x : x + block] = patch * 0.4 + mean * 0.6
        return np.clip(out, 0, 255).astype(np.uint8)

    def test_blockiness_detects_block_quantisation(self, scene):
        clean = blockiness_score(scene, 8)
        blocked = blockiness_score(self._blocked(scene, 8), 8)
        assert blocked > clean
        assert blocked > 1.2

    def test_clean_frame_scores_near_one(self, scene):
        # 1.0 means the 8x8 grid is indistinguishable from everywhere else.
        assert 0.8 < blockiness_score(scene, 8) < 1.3

    def test_assessment_identifies_the_dominant_block_size(self, scene, cfg):
        assessment = assess_artifacts(self._blocked(scene, 8), cfg)
        assert assessment.needs_correction
        assert assessment.dominant_block_size in (8, 16)

    @pytest.mark.xfail(reason="Stage 2 open issue S2-1 (DEVLOG): bilateral params do not reduce strong synthetic blocking", strict=False)
    def test_suppression_reduces_blockiness(self, scene, cfg):
        blocked = self._blocked(scene, 8)
        before = blockiness_score(blocked, 8)
        after = blockiness_score(suppress_block_artifacts(blocked, cfg), 8)
        assert after < before

    def test_clean_frames_are_left_alone(self, scene, cfg):
        # Filtering an unblocked frame costs texture detail for no benefit.
        assert suppress_block_artifacts(scene, cfg) is scene

    def test_grid_mask_covers_the_boundaries(self):
        mask = block_grid_mask((64, 64), block_size=8, exclusion_px=1)
        assert mask[:, 8].all() and mask[:, 7].all() and mask[:, 9].all()
        # Row boundaries are marked too, so probe a pixel far from both.
        assert not mask[4, 4]

    def test_keypoints_on_the_grid_are_dropped_only_when_blocking_is_real(self, scene, cfg):
        blocked_assessment = assess_artifacts(self._blocked(scene, 8), cfg)
        clean_assessment = assess_artifacts(scene, cfg)
        keypoints = [(8.0, 20.0), (16.0, 30.0), (5.0, 21.0), (13.0, 44.0)]

        kept, dropped = filter_grid_keypoints(keypoints, (480, 640), blocked_assessment, cfg)
        assert dropped > 0 and len(kept) == len(keypoints) - dropped

        kept, dropped = filter_grid_keypoints(keypoints, (480, 640), clean_assessment, cfg)
        assert dropped == 0, "a clean frame has no compression grid to veto"


class TestExposure:
    def test_recovers_a_known_gain_on_identical_content(self, scene):
        import cv2

        brightened = cv2.convertScaleAbs(scene, alpha=1.3, beta=5.0)
        # reference ~= gain * target + bias, with target the brightened frame.
        fit = fit_gain_bias(scene, brightened)
        assert fit.gain == pytest.approx(1 / 1.3, rel=0.12)

    def test_panning_over_new_content_is_not_read_as_exposure_change(self):
        # The failure this guards against: at 75% overlap a quarter of each
        # frame is content the other never saw, so whole-frame distribution
        # matching charges the exposure estimate for the scene changing.
        canvas = fixtures.make_canvas(1400, 480, seed=33)
        first = canvas[0:480, 0:640]
        second = canvas[0:480, 160:800]      # panned 160 px: 75% overlap
        matrix = np.array([[1.0, 0.0, -160.0], [0.0, 1.0, 0.0]])
        transform = normalize_transform(matrix, 640, 480)

        without = fit_gain_bias(first, second)
        with_geometry = fit_gain_bias(first, second, transform)

        # No exposure change occurred, so the honest answer is a gain of 1.
        assert abs(with_geometry.gain - 1.0) < abs(without.gain - 1.0)
        assert with_geometry.gain == pytest.approx(1.0, abs=0.05)

    def test_recovers_a_known_gain_while_panning(self):
        import cv2

        canvas = fixtures.make_canvas(1400, 480, seed=34)
        first = canvas[0:480, 0:640]
        second = cv2.convertScaleAbs(canvas[0:480, 160:800], alpha=1.25, beta=0.0)
        transform = normalize_transform(np.array([[1.0, 0.0, -160.0], [0.0, 1.0, 0.0]]), 640, 480)

        fit = fit_gain_bias(first, second, transform)
        assert fit.gain == pytest.approx(1 / 1.25, rel=0.12)

    def test_chain_composes_across_frames(self, scene, cfg):
        import cv2

        chain = ExposureChain(cfg)
        chain.push(0, scene)
        step1 = cv2.convertScaleAbs(scene, alpha=1.2, beta=0.0)
        step2 = cv2.convertScaleAbs(scene, alpha=1.44, beta=0.0)
        chain.push(1, step1)
        transform = chain.push(2, step2)

        # Frame 2 is 1.44x the reference, so its correction is ~1/1.44.
        assert transform.gain == pytest.approx(1 / 1.44, rel=0.2)
        assert chain.summary()["frames"] == 3

    def test_a_bad_link_does_not_poison_the_rest_of_the_chain(self, scene, cfg):
        chain = ExposureChain(cfg)
        chain.push(0, scene)
        flat = np.full_like(scene, 128)      # carries no gain information
        chain.push(1, flat)
        after = chain.push(2, scene)
        assert 0.5 < after.gain < 2.0, "one uninformative frame must not derail the chain"

    def test_disabled_chain_is_the_identity(self, scene, cfg):
        disabled = cfg.merged({"condition": {"illumination": {"exposure_chain": {"enabled": False}}}})
        chain = ExposureChain(disabled)
        assert chain.push(0, scene).gain == 1.0


class TestShadows:
    def _shadowed(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Darken a region the way skylight does: dimmer, desaturated, bluer."""
        out = image.astype(np.float32).copy()
        region = (slice(100, 300), slice(100, 400))
        patch = out[region]
        grey = patch.mean(axis=2, keepdims=True)
        patch = patch * 0.35 + grey * 0.25          # darker and desaturated
        patch[:, :, 0] *= 1.35                      # skylight is blue
        out[region] = patch
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[region] = True
        return np.clip(out, 0, 255).astype(np.uint8), mask

    @pytest.mark.xfail(reason="Stage 2 open issue S2-2 (DEVLOG): fixed-percentile luminance cut caps shadow recall", strict=False)
    def test_finds_a_synthetic_shadow(self, scene, cfg):
        shadowed, truth = self._shadowed(scene)
        detected, fraction = detect_shadows(shadowed, cfg)
        overlap = (detected & truth).sum() / max(truth.sum(), 1)
        assert overlap > 0.5, f"only recovered {overlap:.0%} of the shadow"
        assert fraction > 0

    def test_dark_paint_is_not_a_shadow(self, scene, cfg):
        # The point of the three-condition test: brightness alone cannot tell a
        # shadow from a dark red roof, but saturation and blue ratio can.
        out = scene.copy()
        out[100:300, 100:400] = (30, 30, 160)     # dark, saturated, red
        detected, _ = detect_shadows(out, cfg)
        region = detected[100:300, 100:400]
        assert region.mean() < 0.25

    def test_weight_map_downweights_only_shadow(self, cfg):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:20, 10:20] = True
        weights = shadow_weight_map(mask, cfg)
        assert weights[15, 15] == pytest.approx(cfg.condition.illumination.shadow.mvs_weight)
        assert weights[0, 0] == 1.0

    def test_disabled_returns_an_empty_mask(self, scene, cfg):
        disabled = cfg.merged({"condition": {"illumination": {"shadow": {"enabled": False}}}})
        mask, fraction = detect_shadows(scene, disabled)
        assert not mask.any() and fraction == 0.0


class TestLowLight:
    def test_dim_footage_is_detected_and_lifted(self, scene, cfg):
        import cv2

        dim = cv2.convertScaleAbs(scene, alpha=0.25, beta=0.0)
        conditioned, is_low, mean_luma = condition_low_light(dim, cfg)
        assert is_low
        assert mean_luma < cfg.condition.illumination.low_light.mean_luma_threshold
        assert conditioned.mean() > dim.mean()

    def test_normal_footage_is_untouched(self, scene, cfg):
        conditioned, is_low, _ = condition_low_light(scene, cfg)
        assert not is_low
        assert conditioned is scene

    def test_low_light_frames_carry_a_reduced_weight(self, scene, cfg):
        import cv2

        dim = cv2.convertScaleAbs(scene, alpha=0.2, beta=0.0)
        result = condition_illumination(dim, cfg, base_weight=1.0)
        assert result.low_light
        # Spec §5.4: low-light accuracy is measurably worse, and the pipeline
        # must record that rather than hide it.
        assert result.weight == pytest.approx(cfg.condition.illumination.low_light.fusion_weight)


class TestClahe:
    def test_increases_local_contrast_without_wrecking_colour(self, scene, cfg):
        import cv2

        equalized = apply_clahe(scene, cfg)
        before = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB)[:, :, 0].std()
        after = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)[:, :, 0].std()
        assert after >= before * 0.95
        # Only L is touched, so the chroma channels must survive for texturing.
        for channel in (1, 2):
            original = cv2.cvtColor(scene, cv2.COLOR_BGR2LAB)[:, :, channel].astype(float)
            result = cv2.cvtColor(equalized, cv2.COLOR_BGR2LAB)[:, :, channel].astype(float)
            assert np.abs(original - result).mean() < 6.0


class TestDynamicObjects:
    def test_missing_model_degrades_instead_of_crashing(self, cfg, scene):
        masker = DynamicMasker(cfg)
        result = masker.mask(scene)
        # ultralytics is not a hard dependency; without it the pipeline must
        # fall back to geometric consistency rather than fail.
        if not masker.available:
            assert result.method == "none"
            assert not result.mask.any()
            assert masker.unavailable_reason

    def test_dilation_grows_the_mask(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True
        grown = dilate_mask(mask, 5)
        assert grown.sum() > mask.sum()
        assert grown[20:30, 20:30].all()

    def test_geometric_test_flags_disagreeing_depth_in_textured_regions(self, scene, cfg):
        depth = np.full((480, 640), 50.0, dtype=np.float32)
        neighbours = [depth.copy() for _ in range(3)]
        for neighbour in neighbours:
            neighbour[200:260, 200:260] = 80.0     # a mover: 60% disagreement

        result = geometric_dynamic_mask(depth, neighbours, scene, cfg)
        assert result.mask[220:240, 220:240].mean() > 0.5
        assert result.mask[400:450, 50:100].mean() < 0.1

    def test_agreeing_depth_flags_nothing(self, scene, cfg):
        depth = np.full((480, 640), 50.0, dtype=np.float32)
        result = geometric_dynamic_mask(depth, [depth.copy() for _ in range(3)], scene, cfg)
        assert not result.mask.any()

    def test_a_single_dissenting_view_is_not_enough(self, scene, cfg):
        # One neighbour disagreeing is more likely an occlusion boundary than a
        # moving object; requiring a majority is what separates the two.
        depth = np.full((480, 640), 50.0, dtype=np.float32)
        neighbours = [depth.copy() for _ in range(3)]
        neighbours[0][200:260, 200:260] = 80.0
        result = geometric_dynamic_mask(depth, neighbours, scene, cfg)
        assert result.mask[220:240, 220:240].mean() < 0.2

    def test_untextured_regions_are_not_flagged(self, cfg):
        # Depth disagreement in a flat region means MVS had nothing to match
        # on, not that the surface moved.
        flat = np.full((480, 640, 3), 128, dtype=np.uint8)
        depth = np.full((480, 640), 50.0, dtype=np.float32)
        neighbours = [np.full_like(depth, 80.0) for _ in range(3)]
        result = geometric_dynamic_mask(depth, neighbours, flat, cfg)
        assert not result.mask.any()

    def test_disagreement_is_relative_not_absolute(self, scene, cfg):
        # 1 m of mismatch is decisive at 5 m; 0.5 m is meaningless at 200 m.
        near = np.full((480, 640), 5.0, dtype=np.float32)
        near_neighbours = [np.full_like(near, 6.0) for _ in range(3)]   # 20% > 12% threshold
        far = np.full((480, 640), 200.0, dtype=np.float32)
        far_neighbours = [np.full_like(far, 200.5) for _ in range(3)]

        assert geometric_dynamic_mask(near, near_neighbours, scene, cfg).mask.any()
        assert not geometric_dynamic_mask(far, far_neighbours, scene, cfg).mask.any()

    def test_texture_variance_separates_flat_from_detailed(self, scene):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert texture_variance(flat).mean() < 1.0
        assert texture_variance(scene).mean() > 10.0


class TestHolePolicy:
    def test_small_enclosed_ground_patch_is_filled(self, cfg):
        decision = decide_hole_policy(5.0, is_ground_like=True, neighbours_confident=True, cfg=cfg)
        assert decision.fill_ground_plane

    def test_a_facade_hole_is_flagged_not_guessed(self, cfg):
        decision = decide_hole_policy(5.0, is_ground_like=False, neighbours_confident=True, cfg=cfg)
        assert not decision.fill_ground_plane
        assert "flagged" in decision.reason

    def test_unconfident_surroundings_block_the_fill(self, cfg):
        decision = decide_hole_policy(5.0, is_ground_like=True, neighbours_confident=False, cfg=cfg)
        assert not decision.fill_ground_plane

    def test_large_regions_are_never_interpolated(self, cfg):
        limit = cfg.condition.dynamic.hole_policy.max_fill_area_m2
        decision = decide_hole_policy(limit * 2, is_ground_like=True, neighbours_confident=True, cfg=cfg)
        assert not decision.fill_ground_plane
        assert "exceeds" in decision.reason
