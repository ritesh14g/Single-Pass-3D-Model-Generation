"""Stage 3 (§6): zone classification, gap reporting, anchored monocular fill and its rules.

Synthetic scene: nadir cameras 50 m over a gently undulating ground (GSD 0.1 m), whose
dense cloud has a 10 x 10 m hole and a strip that only two frames confirmed. Everything
the stage decides has a known answer here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.core.config import load_config
from src.fusion import anchor, zones as zm
from src.fusion.scene import CameraView, Scene, near_camera
from src.fusion.zones import ZONE1, ZONE2
from src.geo.crs import MapFrame
from src.geo.georef import Georef

HEIGHT, FOCAL, W, H = 50.0, 500.0, 640, 480
HOLE = (25.0, 35.0, -10.0, 0.0)          # x0, x1, y0, y1 of the unobserved square
THIN = (-20.0, -10.0)                    # x band only two frames confirmed


def ground_z(x, y):
    return 0.04 * x + 1.5 * np.sin(y / 6.0)


def nadir(name: str, x: float, y: float) -> CameraView:
    # Camera x = east, y = south, z = down (looking straight at the ground).
    k = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
    return CameraView(name, W, H, k, np.diag([1.0, -1.0, -1.0]), np.array([x, y, HEIGHT]))


def georef() -> Georef:
    frame = MapFrame(32643, None, "WGS84 ellipsoidal", False, "ellipsoidal", True, (781000.0, 1435000.0, 0.0))
    return Georef(frame, 1.0, np.eye(3), np.zeros(3), {"referenced": True})


def make_scene(xs=(-30, -20, -10, 0, 10, 20, 30), spacing=0.5, hole=True) -> Scene:
    cams = [nadir(f"f{i:02d}.jpg", float(x), 0.0) for i, x in enumerate(xs)]
    g = np.arange(-80, 80, spacing)
    x, y = [a.ravel() for a in np.meshgrid(g, g)]
    keep = ~((x >= HOLE[0]) & (x < HOLE[1]) & (y >= HOLE[2]) & (y < HOLE[3])) if hole else np.ones(len(x), bool)
    pts = np.c_[x, y, ground_z(x, y)][keep]
    sees = np.stack([cam.inside(*cam.project(pts)) for cam in cams], 1)        # [points, cameras]
    thin = (pts[:, 0] >= THIN[0]) & (pts[:, 0] < THIN[1])
    sees[thin] &= np.cumsum(sees[thin], axis=1) <= 2                          # only the first two views confirm
    counts = sees.sum(1)
    ptr = np.r_[0, np.cumsum(counts)]
    idx = np.nonzero(sees)[1].astype(np.int64)
    rgb = np.full((len(pts), 3), 120, np.uint8)
    return Scene(pts, rgb, counts.astype(np.int32), ptr, idx, cams, georef(), "dense")


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def classified(scene, cfg):
    zones, _ = zm.classify(scene, cfg)
    return zones


# -- §6.2 classification ---------------------------------------------------------------------
def test_voxel_follows_the_ground_sample_distance(scene, cfg, classified):
    assert zm.ground_sample_distance(scene) == pytest.approx(HEIGHT / FOCAL, rel=0.05)
    assert classified.grid.size == pytest.approx(2.0 * HEIGHT / FOCAL, rel=0.05)


def test_many_wide_views_are_zone1_and_two_views_are_zone2(scene, classified):
    x = classified.centroid[:, 0]
    y = classified.centroid[:, 1]
    centre = (np.abs(x) < 5) & (np.abs(y) < 5)
    thin = (x > THIN[0] + 1) & (x < THIN[1] - 1) & (np.abs(y) < 5)
    assert (classified.zone[centre] == ZONE1).mean() > 0.95
    assert (classified.zone[thin] == ZONE2).all()
    assert classified.angle_source == "confirming views"


def test_many_views_at_a_poor_angle_are_not_zone1(cfg):
    # Seven frames within 1.2 m of each other: plenty of views, ~1.4° of triangulation angle.
    tight = make_scene(xs=(-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6), spacing=1.0, hole=False)
    zones, _ = zm.classify(tight, cfg)
    near = np.linalg.norm(zones.centroid[:, :2], axis=1) < 10
    assert np.median(zones.views[near]) >= 4
    assert np.median(zones.tri_deg[near]) < 5.0
    assert (zones.zone[near] == ZONE2).all()


def test_the_hole_is_a_zone3_gap_with_its_area(scene, classified):
    gm = zm.ground_map(scene, classified, 1.0, 3.0)
    gaps = zm.gap_regions(gm, 4.0)
    interior = [g for g in gaps if not g["touches_view_edge"]]
    assert len(interior) == 1
    assert interior[0]["area_m2"] == pytest.approx(100.0, rel=0.15)
    cx, cy = interior[0]["centroid_local"]
    assert cx == pytest.approx(30.0, abs=1.0) and cy == pytest.approx(-5.0, abs=1.0)
    stats = gm.stats()
    edge = sum(g["area_m2"] for g in gaps if g["touches_view_edge"])
    assert stats["zone3_m2"] - edge == pytest.approx(100.0, rel=0.15)
    assert edge < 0.02 * stats["visible_m2"]           # flat-plane footprint vs +-2.4 m of relief
    assert stats["coverage_pct"] > 95.0


def test_occluded_voxels_are_not_counted_as_seen(cfg):
    # A 10 m roof over part of the ground hides the ground from the cameras above it.
    base = make_scene(xs=(-10, 0, 10), spacing=1.0, hole=False)
    g = np.arange(-5, 5, 0.25)
    rx, ry = [a.ravel() for a in np.meshgrid(g, g)]
    roof = np.c_[rx, ry, np.full(len(rx), 10.0)]
    pts = np.r_[base.points, roof]
    scene = Scene(pts, np.full((len(pts), 3), 100, np.uint8), np.r_[base.views, np.full(len(roof), 3)].astype(np.int32),
                  None, None, base.cameras, base.georef, "dense")
    zones, _ = zm.classify(scene, cfg)
    under = (np.abs(zones.centroid[:, 0]) < 3) & (np.abs(zones.centroid[:, 1]) < 3) & (zones.centroid[:, 2] < 5)
    open_ground = (np.abs(zones.centroid[:, 0] - 20) < 3) & (np.abs(zones.centroid[:, 1]) < 3)
    assert zones.views_geom[under].max() == 0
    assert zones.views_geom[open_ground].min() >= 2


def test_points_next_to_the_flight_path_are_rejected(scene):
    floaters = np.array([[0.0, 0.5, HEIGHT - 1.0], [10.0, -0.3, HEIGHT - 2.0]])
    with_floaters = Scene(np.r_[scene.points, floaters], np.r_[scene.rgb, np.zeros((2, 3), np.uint8)],
                          np.r_[scene.views, [3, 3]].astype(np.int32), None, None, scene.cameras, scene.georef, "dense")
    rejected = near_camera(with_floaters, 0.1)
    assert rejected[-2:].all() and not rejected[:-2].any()


# -- §6.3 anchoring ---------------------------------------------------------------------------
def test_scale_shift_fit_survives_outliers():
    rng = np.random.default_rng(0)
    mono = rng.uniform(1, 3, 2000)
    mvs = 20.0 * mono + 7.0 + rng.normal(0, 0.05, 2000)
    bad = rng.random(2000) < 0.3
    mvs[bad] += rng.uniform(-20, 20, bad.sum())
    s, t, inl = anchor.fit_scale_shift(mono, mvs, iterations=300, threshold=0.25, rng=rng)
    assert s == pytest.approx(20.0, rel=0.01) and t == pytest.approx(7.0, abs=0.1)
    assert inl.mean() == pytest.approx(0.7, abs=0.05)


def _true_depth(cam: CameraView, shape):
    """Depth of the full (hole-free) surface for every pixel of a (h, w) buffer."""
    full = make_scene(xs=(0,), spacing=0.25, hole=False)
    return anchor.render_depth(cam, full.points, shape, 0.25)


def test_anchored_depth_matches_the_hidden_surface(scene, classified, cfg):
    cam = scene.cameras[6]                        # x = 30: the hole is in view
    truth = _true_depth(cam, (240, 320))
    mono = ((truth - 12.0) / 25.0).astype(np.float32)   # relative depth: unknown scale and shift
    conf = np.ones_like(mono)
    zone1 = scene.points[classified.point_zone == ZONE1]
    filled, weight, fit, report = anchor.anchor_frame(cam, mono, conf, zone1, classified.grid.size,
                                                      cfg.get_path("fusion.mono_depth"), np.random.default_rng(0))
    assert report["status"] == "anchored", report
    assert fit.scale == pytest.approx(25.0, rel=0.01) and fit.shift == pytest.approx(12.0, abs=0.3)
    region = np.isfinite(filled)
    assert region.sum() > 500
    err = np.abs(filled[region] - truth[region])
    assert np.median(err) < 0.1                       # metres, at 50 m range
    assert weight[region].max() <= cfg.get_path("fusion.mono_depth.zone2_weight_max") + 1e-6
    assert report["holdout_error_pct"] < 0.5


def test_a_bad_anchor_is_refused_not_fused(scene, classified):
    # This ground spans only +-1.5 m at 50 m, inside the default 2% band, so even a flat fit
    # would pass it; the gate itself is what is tested, at tolerances the relief exceeds.
    tight = load_config(overrides=["fusion.mono_depth.ransac_inlier_threshold_rel=0.002",
                                   "fusion.mono_depth.ransac_inlier_threshold_m=0.1",
                                   "fusion.mono_depth.max_residual_rel=0.002", "fusion.mono_depth.max_residual_m=0.1"])
    cam = scene.cameras[6]
    truth = _true_depth(cam, (240, 320))
    rng = np.random.default_rng(3)
    mono = rng.uniform(1.0, 2.0, truth.shape).astype(np.float32)   # unrelated to the surface
    zone1 = scene.points[classified.point_zone == ZONE1]
    filled, _, _, report = anchor.anchor_frame(cam, mono, np.ones_like(mono), zone1, classified.grid.size,
                                               tight.get_path("fusion.mono_depth"), rng)
    assert filled is None and report["status"] == "refused"


def test_already_anchored_depth_must_fit_the_georef_scale(scene, classified, cfg):
    # Esri box frame 216: Track B's saved (anchored) depth fitted s = 267 against an expected 1.0.
    cam = scene.cameras[6]
    truth = _true_depth(cam, (240, 320))
    mono = ((truth - 12.0) / 25.0).astype(np.float32)
    zone1 = scene.points[classified.point_zone == ZONE1]
    acfg = cfg.get_path("fusion.mono_depth")
    args = (cam, mono, np.ones_like(mono), zone1, classified.grid.size, acfg)
    filled, _, _, report = anchor.anchor_frame(*args, np.random.default_rng(0), expected_scale=1.0)
    assert filled is None and report["status"] == "refused" and "georeferencing" in report["reason"]
    filled, _, _, report = anchor.anchor_frame(*args, np.random.default_rng(0), expected_scale=24.0)
    assert report["status"] == "anchored" and np.isfinite(filled).any()


def test_fill_is_not_extrapolated_far_from_the_band_depths(scene, classified, cfg):
    cam = scene.cameras[6]
    truth = _true_depth(cam, (240, 320))
    zone1 = scene.points[classified.point_zone == ZONE1]
    region = ~np.isfinite(anchor.render_depth(cam, zone1, truth.shape, classified.grid.size)) & np.isfinite(truth)
    # Right on the band, 20 m too deep inside the gap: 50 m band depth x 0.15 allows 7.5 m.
    mono = ((np.where(region, truth + 20.0, truth) - 12.0) / 25.0).astype(np.float32)
    filled, _, _, report = anchor.anchor_frame(cam, mono, np.ones_like(mono), zone1, classified.grid.size,
                                               cfg.get_path("fusion.mono_depth"), np.random.default_rng(0))
    assert report["status"] == "anchored"
    assert report["extrapolation_dropped_px"] > 0.9 * report["region_px"]
    assert np.isfinite(filled).sum() < 0.1 * report["region_px"]


class _TruthMono:
    """A 'monocular' source that is the true surface under an unknown affine map."""

    def __call__(self, cam):
        truth = _true_depth(cam, (240, 320))
        return ((truth - 3.0) / 7.0).astype(np.float32), np.ones(truth.shape, np.float32)


def test_fill_closes_the_hole_without_touching_zone1(scene, classified, cfg):
    gm = zm.ground_map(scene, classified, 1.0, 3.0)
    result = anchor.fill(scene, classified, gm, cfg, _TruthMono(), max_frames=4, source="truth")
    assert result.summary()["frames_anchored"] >= 1
    assert len(result.points)
    x, y = result.points[:, 0], result.points[:, 1]
    in_hole = (x >= HOLE[0]) & (x < HOLE[1]) & (y >= HOLE[2]) & (y < HOLE[3])
    row, col = gm.cells(result.points[:, :2])
    assert np.isin(gm.zone[row, col], (ZONE2, 3)).all()   # only gap and thin ground is filled
    assert in_hole.sum() > 0.5 * 100.0 / classified.grid.size ** 2 / 4
    assert np.median(np.abs(result.points[:, 2] - ground_z(x, y))) < 0.15
    zone1_keys = classified.keys[classified.zone == ZONE1]
    assert not np.isin(classified.grid.keys(result.points), zone1_keys).any()
    assert result.weight.max() < 1.0
    after = zm.ground_map(scene, classified, 1.0, 3.0, extra_xy=result.points[:, :2])
    cells = after.centres(*np.nonzero(after.zone == 3))
    hole_left = ((cells[:, 0] > HOLE[0]) & (cells[:, 0] < HOLE[1]) & (cells[:, 1] > HOLE[2]) & (cells[:, 1] < HOLE[3]))
    assert hole_left.sum() <= 5                          # the hole is closed (was 90 cells)
    assert after.stats()["zone3_m2"] < gm.stats()["zone3_m2"] - 80


def test_monocular_points_never_enter_a_zone1_voxel(classified):
    zone1 = classified.centroid[classified.zone == ZONE1][:50]
    result = anchor.FillResult()
    anchor._fuse(result, classified, zone1, np.zeros((50, 3), np.uint8), np.full(50, 0.4),
                 np.zeros(50, np.int32), 0.1, 0.45)
    assert result.blocked_by_zone1 == 50 and len(result.points) == 0


def test_gpu_failure_falls_back_to_cpu(monkeypatch, cfg):
    from src.fusion import stage

    calls = []

    def fake_predictor(config, device):
        calls.append(device)
        if device == "cuda":
            def broken(cam):
                raise RuntimeError("CUDA out of memory")
            return broken
        return lambda cam: (np.ones((4, 4), np.float32), np.ones((4, 4), np.float32))

    monkeypatch.setattr(anchor, "predictor_depth", fake_predictor)
    source = stage.MonoSource(cfg, "cuda", None, cfg.get_path("fusion.mono_depth"))
    depth, _ = source(nadir("a.jpg", 0, 0))
    assert calls == ["cuda", "cpu"] and depth.shape == (4, 4)
    assert source.device == "cpu" and "GPU -> CPU" in source.downgrades[0]


def test_a_missing_model_stops_the_fill_once(monkeypatch, scene, classified, cfg):
    from src.fusion import stage

    def unavailable(config, device):
        raise RuntimeError("gated repo")

    monkeypatch.setattr(anchor, "predictor_depth", unavailable)
    source = stage.MonoSource(cfg, "cpu", None, cfg.get_path("fusion.mono_depth"))
    gm = zm.ground_map(scene, classified, 1.0, 3.0)
    result = anchor.fill(scene, classified, gm, cfg, source, max_frames=4, source="x")
    assert result.frames == [] and "unavailable" in result.reason


# -- §6.4 outputs -------------------------------------------------------------------------------
def test_gaps_geojson_is_wgs84_with_areas(tmp_path, scene, classified):
    from src.fusion.stage import write_gaps

    gm = zm.ground_map(scene, classified, 1.0, 3.0)
    gaps = zm.gap_regions(gm, 4.0)
    doc = json.loads(write_gaps(tmp_path / "gaps.geojson", gaps, scene.georef, gm).read_text())
    assert doc["type"] == "FeatureCollection" and len(doc["features"]) == len(gaps)
    ring = np.asarray(doc["features"][0]["geometry"]["coordinates"][0])
    assert np.all((ring[:, 0] > 76) & (ring[:, 0] < 79) & (ring[:, 1] > 12) & (ring[:, 1] < 14))  # Bengaluru
    assert doc["properties"]["total_gap_m2"] == pytest.approx(sum(g["area_m2"] for g in gaps))


def test_evaluator_catches_monocular_points_inside_zone1(tmp_path, scene, classified):
    """Mutation check for the critical-rule KPI: plant a fill point in a Zone 1 voxel."""
    from src.fusion.stage import write_fill_ply
    from src.qa.stage3_eval import FusionOutputs, evaluate_fusion

    folder = tmp_path / "fusion"
    folder.mkdir()
    classified.frame().to_parquet(folder / "zones.parquet", index=False)
    planted = classified.centroid[classified.zone == ZONE1][:3]
    write_fill_ply(folder / "fill.ply", anchor.FillResult(points=planted, rgb=np.zeros((3, 3), np.uint8),
                                                          weight=np.full(3, 0.3), frames_per_point=np.ones(3, np.int32)))
    report = {"voxel": classified.grid.to_dict() | {"voxels": len(classified.keys)}, "ground": {"zone1_pct": 80.0},
              "coverage_pct": 95.0, "gaps": {"count": 0, "total_m2": 0.0}, "fill": {"weight_max": 0.3},
              "timings_s": {"total": 1.0}}
    (folder / "fusion_report.json").write_text(json.dumps(report))
    kpis = {k.key: k for k in evaluate_fusion(FusionOutputs.load(tmp_path), load_config()).kpis}
    assert kpis["zone1_untouched"].status == "fail" and kpis["zone1_untouched"].value == 3


class _Budget:
    """Just enough of StageBudget for the fill: time left and the degradation record."""

    def __init__(self, remaining):
        self.remaining_s, self.enabled, self.degradations = remaining, True, []

    def degrade(self, action, **details):
        self.degradations.append((action, details))


def test_fill_budget_uses_the_fills_own_rate(scene, classified, cfg):
    """Esri, box: the stage-level projection counted 43 s of classification as fill time and stopped
    the fill after 2 of 35 frames, which had taken 2.2 s. Plenty of time left must mean every frame."""
    gm = zm.ground_map(scene, classified, 1.0, 3.0)
    roomy = _Budget(remaining=500.0)
    full = anchor.fill(scene, classified, gm, cfg, _TruthMono(), max_frames=4, source="truth", budget_stage=roomy)
    assert not roomy.degradations and full.reason is None and len(full.frames) >= 2
    tight = _Budget(remaining=0.0)   # nothing left beyond the reserve: stop after the first frame
    cut = anchor.fill(scene, classified, gm, cfg, _TruthMono(), max_frames=4, source="truth", budget_stage=tight)
    assert tight.degradations and len(cut.frames) == 1 and "time budget" in cut.reason
