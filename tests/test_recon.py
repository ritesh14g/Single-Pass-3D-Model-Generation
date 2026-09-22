"""Stage 4 Track A: alignment maths, budget-driven dense resolution, masks, meshing,
the scorecard, and one end-to-end run on a tiny synthetic flight (CPU path).

The synthetic flight is a flat canvas panned in image space, so it cannot judge 3D
quality (focal and curvature are unobservable, DEVLOG §4). The end-to-end test checks
plumbing only: every step runs, every artifact lands, every fallback is reported.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.config import load_config
from src.qa.stage1_eval import FAIL, INFO, PASS, WARN
from src.qa.stage4_eval import TrackAOutputs, evaluate_track_a
from src.recon import alignment, meshing, openmvs
from src.recon.track_a_colmap import _budgeted_size, _colmap_masks
from tests import fixtures


# -- alignment -----------------------------------------------------------------
class TestAlignment:
    def test_umeyama_recovers_a_known_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(20, 3))
        angle = 0.7
        rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        dst = 3.5 * (rot @ src.T).T + np.array([10.0, -4.0, 2.0])
        scale, r, t = alignment.umeyama(src, dst)
        assert scale == pytest.approx(3.5)
        assert np.allclose(r, rot)
        assert np.allclose(alignment.apply((scale, r, t), src), dst)

    def test_metric_check_needs_three_gps_frames(self):
        names = ["a.jpg", "b.jpg"]
        stats, transform = alignment.metric_check(names, np.zeros((2, 3)), np.zeros((0, 3)),
                                                  {"a.jpg": np.zeros(3), "b.jpg": np.ones(3)})
        assert transform is None and stats == {"gps_matched_frames": 2}

    def test_metric_check_measures_height_in_metres(self):
        # Cameras 100 m above a flat ground, in a model 10x smaller and shifted.
        names = [f"{i}.jpg" for i in range(6)]
        gps = {n: np.array([20.0 * i, 3.0 * (i % 2), 0.0]) for i, n in enumerate(names)}
        centres = np.array([gps[n] for n in names]) / 10 + 5
        grid = np.array([[x + 0.5, y + 0.5, -100.0] for x in range(-10, 110, 2) for y in range(-20, 25, 2)])
        points = grid / 10 + 5
        stats, transform = alignment.metric_check(names, centres, points, gps)
        assert transform is not None
        assert stats["cam_vs_gps_rms_m"] == pytest.approx(0.0, abs=1e-6)
        assert stats["height_above_ground_m"] == pytest.approx(100.0, abs=0.1)
        # One point per 1 m cell (points sit at cell centres, away from the edges).
        assert alignment.footprint_m2(points, transform, cell_m=1.0) == pytest.approx(len(grid))

    def test_focal_from_hfov_and_from_35mm_equivalent(self):
        focal, source = alignment.focal_from_telemetry({"hfov_deg": 81.0}, 3840)
        assert source == "telemetry_hfov" and focal == pytest.approx(2248, abs=1)
        focal, source = alignment.focal_from_telemetry({"focal_mm_35": 24.0}, 3840)
        assert source == "telemetry_focal_35mm" and focal == pytest.approx(2560.0)
        assert alignment.focal_from_telemetry({}, 3840) == (None, "self_calibrated")

    def test_telemetry_hints_read_the_parquet(self, tmp_path):
        path = tmp_path / "telemetry.parquet"
        pd.DataFrame({"hfov_deg": [81.0, 81.0], "focal_mm": [np.nan, np.nan],
                      "alt_gps": [117.0, 117.0], "frame_center_alt": [6.0, 6.0]}).to_parquet(path)
        hints = alignment.telemetry_hints(path)
        assert hints == {"hfov_deg": 81.0, "expected_agl_m": 111.0}
        assert alignment.telemetry_hints(tmp_path / "missing.parquet") == {}

    def test_read_geo_enu(self, tmp_path):
        geo = tmp_path / "geo.txt"
        geo.write_text("EPSG:4326\na.jpg 77.0 12.0 100\nb.jpg 77.001 12.0 110\n")
        enu = alignment.read_geo_enu(geo)
        assert np.allclose(enu["a.jpg"], 0)
        assert enu["b.jpg"][0] == pytest.approx(108.9, abs=0.5) and enu["b.jpg"][2] == 10


# -- dense budget ---------------------------------------------------------------
class _FakeStage:
    enabled = True

    def __init__(self, remaining_s):
        self.remaining_s = remaining_s
        self.actions = []

    def degrade(self, action, reason="", **details):
        self.actions.append((action, details["from_px"], details["to_px"]))
        return action


class TestDenseBudget:
    @pytest.fixture
    def dense(self):
        return load_config().recon.track_a.dense

    def test_no_budget_keeps_the_configured_size(self, dense):
        assert _budgeted_size(dense, 45, True, None, native_px=3840) == 1280

    def test_projection_steps_down_until_it_fits(self, dense):
        # 45 frames x 13.6 s = 612 s at 1280; 344 s at 960; 153 s at 640.
        stage = _FakeStage(remaining_s=400 + dense.reserve_after_s)
        assert _budgeted_size(dense, 45, True, stage, native_px=3840) == 960
        assert stage.actions == [("reduce_resolution", 1280, 960)]

    def test_small_frames_are_projected_at_their_own_size(self, dense):
        # 640 px frames are never upscaled, so the 1280 px projection must not trigger a degrade.
        stage = _FakeStage(remaining_s=200 + dense.reserve_after_s)
        assert _budgeted_size(dense, 45, True, stage, native_px=640) == 1280
        assert stage.actions == []

    def test_ladder_exhausted_keeps_the_smallest_size(self, dense):
        stage = _FakeStage(remaining_s=1.0)
        assert _budgeted_size(dense, 45, True, stage, native_px=3840) == 640
        assert [a[2] for a in stage.actions] == [960, 640]


# -- masks, meshing, OpenMVS lookup --------------------------------------------
def test_stage2_masks_become_colmap_masks(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(), masks.mkdir()
    for name in ("f0.jpg", "f1.jpg"):
        cv2.imwrite(str(images / name), np.zeros((8, 10, 3), np.uint8))
    exclude = np.zeros((8, 10), np.uint8)
    exclude[:, :4] = 255
    cv2.imwrite(str(masks / "f0_exclude.png"), exclude)
    out = _colmap_masks(images, masks, tmp_path, ["f0.jpg", "f1.jpg"])
    m0 = cv2.imread(str(out / "f0.jpg.png"), cv2.IMREAD_GRAYSCALE)
    m1 = cv2.imread(str(out / "f1.jpg.png"), cv2.IMREAD_GRAYSCALE)
    assert (m0[:, :4] == 0).all() and (m0[:, 4:] == 255).all()   # excluded -> no features
    assert (m1 == 255).all()                                     # no mask -> features everywhere
    assert _colmap_masks(images, tmp_path / "none", tmp_path, ["f0.jpg"]) is None


def test_clean_mesh_drops_nan_vertices_and_fragments(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    big = trimesh.creation.icosphere(subdivisions=3)
    small = trimesh.creation.icosphere(subdivisions=0)
    small.apply_translation([5, 0, 0])
    joined = trimesh.util.concatenate([big, small])
    vertices = joined.vertices.copy()
    vertices[0] = np.nan
    trimesh.Trimesh(vertices, joined.faces, process=False).export(tmp_path / "in.ply")
    stats = meshing.clean_mesh(tmp_path / "in.ply", tmp_path / "out.ply", 0.1)
    out = trimesh.load(tmp_path / "out.ply", process=False)
    assert stats["nonfinite_vertices"] == 1 and stats["components_kept"] == 1
    assert np.isfinite(out.vertices).all()
    assert len(out.faces) == len(big.faces) - 5  # the 5 faces around the NaN vertex are gone
    assert meshing.ply_counts(tmp_path / "out.ply") == {"vertices": len(out.vertices), "faces": len(out.faces)}


def test_missing_openmvs_is_reported_not_raised(tmp_path):
    assert openmvs.find_bin_dir(str(tmp_path / "nowhere")) is None


# -- scorecard ------------------------------------------------------------------
GOOD = {
    "frames_in": 50, "registered": 49, "registered_fraction": 0.98, "models": 1, "reproj_px": 1.2,
    "track_length": 3.4, "focal_px": 2248.0, "focal_source": "telemetry_hfov", "using_cuda": True,
    "gps_matched_frames": 49, "cam_vs_gps_rms_m": 0.8, "height_above_ground_m": 108.0,
    "expected_agl_m": 110.8, "height_error_pct": -2.5,
    "dense": {"mode": "auto", "engine": "vggt_hybrid", "window": 8, "frames": 49, "frames_anchored": 49,
              "frames_rejected": {}, "anchors_median": 980.0, "anchor_spread_median_pct": 0.85,
              "points_before_consistency": 1500000, "views_per_point_median": 3.0, "vggt_seconds": 6.4,
              "points": 1200000, "footprint_m2": 110000},
    "mesh": {"mesher": "openmvs_delaunay", "vertices": 300000, "faces": 600000, "faces_per_vertex": 2.0},
    "textured": True, "timings_s": {"sparse_map": 14.0, "dense_patchmatch": 600.0}, "downgrades": [],
}


def _status(report, key, cfg):
    kpis = {k.key: k for k in evaluate_track_a(TrackAOutputs(report=report), cfg).kpis}
    return kpis[key].status


class TestScorecard:
    def test_a_good_run_passes_every_scored_kpi(self, cfg):
        ev = evaluate_track_a(TrackAOutputs(report=GOOD), cfg)
        assert ev.counts()["fail"] == 0 and ev.counts()["warn"] == 0
        assert ev.score == 100.0

    def test_mode_a_is_not_penalised_for_skipping_track_b(self, cfg):
        report = {**GOOD, "dense": {"mode": "A", "engine": "colmap_patchmatch_cuda", "size": 1920,
                                    "src_images": 12, "frames": 49, "points": 450000}}
        ev = evaluate_track_a(TrackAOutputs(report=report), cfg)
        assert _status(report, "track_b_used", cfg) == INFO and ev.score == 100.0

    def test_track_b_fallback_warns_and_says_why(self, cfg):
        report = {**GOOD, "dense": {"mode": "auto", "engine": "colmap_patchmatch_cuda", "frames": 49},
                  "downgrades": ["Track B (VGGT depth) -> Track A dense: TrackBUnavailable: no CUDA device"]}
        kpi = {k.key: k for k in evaluate_track_a(TrackAOutputs(report=report), cfg).kpis}["track_b_used"]
        assert kpi.status == WARN and "no CUDA device" in kpi.detail

    @pytest.mark.parametrize("spread,status", [(0.85, PASS), (3.0, WARN), (7.0, FAIL)])
    def test_anchor_spread_band(self, cfg, spread, status):
        report = {**GOOD, "dense": {**GOOD["dense"], "anchor_spread_median_pct": spread}}
        assert _status(report, "anchor_spread_pct", cfg) == status

    def test_missing_report_fails(self, cfg):
        ev = evaluate_track_a(TrackAOutputs(report={}), cfg)
        assert ev.counts()["fail"] == 1 and ev.score == 0.0

    @pytest.mark.parametrize("value,status", [(0.9, PASS), (1.0, PASS), (4.9, WARN), (7.11, FAIL)])
    def test_gps_rms_uses_the_ps_one_metre_target(self, cfg, value, status):
        assert _status({**GOOD, "cam_vs_gps_rms_m": value}, "cam_vs_gps_rms_m", cfg) == status

    @pytest.mark.parametrize("value,status", [(-4.2, PASS), (6.0, WARN), (-16.0, FAIL)])
    def test_height_error_band_is_symmetric(self, cfg, value, status):
        assert _status({**GOOD, "height_error_pct": value}, "height_error_pct", cfg) == status

    @pytest.mark.parametrize("value,status", [(2.0, PASS), (1.5, WARN), (1.04, FAIL)])
    def test_fragmented_mesh_fails(self, cfg, value, status):
        report = {**GOOD, "mesh": {**GOOD["mesh"], "faces_per_vertex": value}}
        assert _status(report, "faces_per_vertex", cfg) == status

    @pytest.mark.parametrize("models,status", [(1, PASS), (2, WARN), (3, FAIL)])
    def test_split_flight_is_flagged(self, cfg, models, status):
        assert _status({**GOOD, "models": models}, "models", cfg) == status

    def test_self_calibrated_focal_and_cpu_only_warn(self, cfg):
        report = {**GOOD, "focal_source": "self_calibrated", "using_cuda": False}
        assert _status(report, "focal_source", cfg) == WARN
        assert _status(report, "using_cuda", cfg) == WARN

    def test_no_dense_fails_and_scale_free_is_info(self, cfg):
        report = {k: v for k, v in GOOD.items() if k not in ("cam_vs_gps_rms_m", "height_error_pct")}
        report["dense"] = {"engine": None}
        assert _status(report, "dense_engine", cfg) == FAIL
        assert _status(report, "cam_vs_gps_rms_m", cfg) == INFO
        assert _status(report, "height_error_pct", cfg) == INFO


def test_presets_pick_the_measured_dense_sizes():
    sizes = {p: load_config(p).recon.track_a.dense.max_image_size for p in (None, "fast", "accurate")}
    assert sizes == {None: 1280, "fast": 960, "accurate": 1920}


# -- end to end -------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_frames(tmp_path_factory):
    root = tmp_path_factory.mktemp("track_a")
    flight = fixtures.make_flight_video(root / "flight.mp4", frames=64, width=320, height=240, overlap=0.93)
    images = root / "images"
    images.mkdir()
    cap = cv2.VideoCapture(str(flight.video_path))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % 4 == 0:
            cv2.imwrite(str(images / f"frame_{index:06d}.jpg"), frame)
        index += 1
    cap.release()
    return root, images


def test_track_a_runs_end_to_end_on_cpu(tiny_frames):
    pytest.importorskip("pycolmap")
    from src.recon.track_a_colmap import run_track_a

    root, images = tiny_frames
    cfg = load_config(overrides=["device.prefer=cpu", "recon.track_a.dense.max_image_size=320"])
    outcome = run_track_a(images, root / "track_a", cfg)
    metrics, artifacts = outcome["metrics"], outcome["artifacts"]

    assert metrics["registered"] >= 0.8 * metrics["frames_in"]
    assert metrics["focal_source"] == "self_calibrated"          # no telemetry given
    assert any("focal length prior" in d for d in metrics["downgrades"])
    assert artifacts["report"].exists() and artifacts["sparse"].is_dir()
    if openmvs.find_bin_dir("auto") is None:
        assert any("OpenMVS" in d for d in metrics["downgrades"])
        pytest.skip("OpenMVS not installed: dense needs it on CPU; sparse path verified")
    assert metrics["dense"]["engine"] == "openmvs_cpu"
    assert metrics["dense"]["points"] > 1000
    # This flat, tiny scene is degenerate for meshing: from run to run OpenMVS meshes it,
    # or cleans the surface away (exit 0, no file) and Poisson takes over, or both come
    # back empty. Each outcome must be handled and reported, never raised: the first two
    # were crashes before the fallback chain was fixed.
    mesher = metrics["mesh"]["mesher"]
    if mesher == "openmvs_delaunay":
        assert metrics["mesh"]["faces"] > 0 and artifacts["mesh"].exists()
    elif mesher == "colmap_poisson":
        assert any("ReconstructMesh" in d for d in metrics["downgrades"])
        assert metrics["mesh"]["faces"] > 0 and artifacts["mesh"].exists()
    else:
        assert mesher is None and "mesh" not in artifacts
        assert any(d.startswith("meshing -> dense point cloud only") and "empty mesh" in d
                   for d in metrics["downgrades"])
        assert artifacts["dense"].exists()
    ev = evaluate_track_a(TrackAOutputs.load(root / "track_a"), cfg)
    assert ev.score is not None


# -- Track B hybrid (VGGT depth on Track A cameras) --------------------------------
def test_windows_cover_every_frame_once():
    from src.recon.track_b_vggt import windows_for

    for n, size in [(45, 8), (10, 8), (8, 8), (3, 8), (100, 6)]:
        windows = windows_for(n, size)
        owned = sorted(i for _, _, own in windows for i in own)
        assert owned == list(range(n))
        assert all(stop - start == min(size, n) for start, stop, _ in windows)


def test_fused_vis_round_trips(tmp_path):
    import struct

    from plyfile import PlyData

    from src.recon.track_b_vggt import write_colmap_fused

    pts = np.arange(12, dtype=float).reshape(4, 3)
    vis = [[0], [1, 2], [3, 0, 5], [7]]
    write_colmap_fused(tmp_path / "fused.ply", pts, np.ones((4, 3)), np.zeros((4, 3), np.uint8), vis)
    raw = (tmp_path / "fused.ply.vis").read_bytes()
    n, back, off = struct.unpack_from("<Q", raw)[0], [], 8
    for _ in range(n):
        k = struct.unpack_from("<I", raw, off)[0]
        back.append(list(struct.unpack_from(f"<{k}I", raw, off + 4)))
        off += 4 + 4 * k
    assert back == vis and off == len(raw)
    v = PlyData.read(str(tmp_path / "fused.ply"))["vertex"]
    assert [p.name for p in v.properties] == ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"]
    assert np.allclose(np.c_[v["x"], v["y"], v["z"]], pts)


def _plane_predictor(undist, true_scale=3.7):
    """Stands in for VGGT on the synthetic (planar) flight: the true depth of the SfM ground
    plane, divided by an unknown scale the anchoring has to recover."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(undist / "sparse"))
    by_name = {im.name: im for im in rec.images.values()}
    cam = next(iter(rec.cameras.values()))
    fx, fy, cx, cy = (float(v) for v in cam.params[:4])
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    centre = np.median(xyz, axis=0)
    normal = np.linalg.svd(xyz - centre)[2][2]
    h, w = 90, 120

    def predict(paths):
        depths, rgbs = [], []
        for path in paths:
            pose = by_name[path.name].cam_from_world()
            rot, trans = pose.rotation.matrix(), np.asarray(pose.translation)
            n_cam = rot @ normal
            d_cam = float(n_cam @ (rot @ centre + trans))
            ys, xs = np.mgrid[0:h, 0:w]
            rays = np.stack([(xs * cam.width / w - cx) / fx, (ys * cam.height / h - cy) / fy, np.ones((h, w))], -1)
            depths.append(d_cam / (rays @ n_cam) / true_scale)
            rgbs.append(cv2.resize(cv2.imread(str(path)), (w, h))[..., ::-1])
        return np.stack(depths), np.ones((len(paths), h, w)), np.stack(rgbs).astype(np.uint8)

    return predict


def test_hybrid_depth_is_anchored_fused_and_meshed(tiny_frames):
    pytest.importorskip("pycolmap")
    from src.recon import track_a_colmap
    from src.recon.track_a_colmap import run_track_a

    root, images = tiny_frames
    cfg = load_config(overrides=["device.prefer=cpu", "recon.track_a.dense.max_image_size=320"])
    out = root / "hybrid"
    real_dense = track_a_colmap._dense

    def dense_with_plane(*args, **kwargs):
        undist = out / "dense"
        kwargs["depth_predictor"] = lambda paths: _plane_predictor(undist)(paths)
        return real_dense(*args, **kwargs)

    track_a_colmap._dense = dense_with_plane
    try:
        outcome = run_track_a(images, out, cfg)
    finally:
        track_a_colmap._dense = real_dense
    dense = outcome["metrics"]["dense"]
    assert dense["engine"] == "vggt_hybrid", outcome["metrics"]["downgrades"]
    assert dense["frames_anchored"] >= 0.8 * dense["frames"]
    assert dense["anchor_spread_median_pct"] < 2.0          # the unknown 3.7x scale was recovered
    assert dense["points"] > 1000 and dense["views_per_point_median"] >= 2
    assert (out / "dense" / "fused.ply.vis").exists()
    assert any((out / "track_b_depth").glob("*.npz"))       # confidence maps kept for Stage 3
    if openmvs.find_bin_dir("auto") is not None:
        mesh = outcome["metrics"]["mesh"]
        assert mesh["mesher"] in ("openmvs_delaunay", "colmap_poisson", None)
        if mesh["mesher"] == "openmvs_delaunay":
            # one vertex per unique surface sample: points / views per point, x2 faces
            expected = int(2.0 * dense["points"] / dense["views_per_point_median"])
            assert mesh["target_faces"] == expected


def test_hybrid_falls_back_to_track_a_dense(tiny_frames):
    pytest.importorskip("pycolmap")
    from src.recon.track_a_colmap import run_track_a

    root, images = tiny_frames
    cfg = load_config(overrides=["device.prefer=cpu", "recon.track_a.dense.max_image_size=320"])

    def broken(paths):
        raise RuntimeError("simulated VGGT crash")

    outcome = run_track_a(images, root / "fallback", cfg, depth_predictor=broken)
    assert any(d.startswith("Track B (VGGT depth) -> Track A dense") for d in outcome["metrics"]["downgrades"])
    if openmvs.find_bin_dir("auto") is not None:
        assert outcome["metrics"]["dense"]["engine"] == "openmvs_cpu"


def test_mode_a_never_calls_vggt(tiny_frames):
    pytest.importorskip("pycolmap")
    from src.recon.track_a_colmap import run_track_a

    root, images = tiny_frames
    cfg = load_config(overrides=["device.prefer=cpu", "run.mode=A", "recon.track_a.dense.max_image_size=320"])

    def must_not_run(paths):
        raise AssertionError("VGGT called in mode A")

    outcome = run_track_a(images, root / "mode_a", cfg, depth_predictor=must_not_run)
    assert not any("Track B" in d for d in outcome["metrics"]["downgrades"])
    assert outcome["metrics"]["dense"].get("mode") == "A"


# -- merging SfM pieces through GPS ----------------------------------------------
@pytest.fixture(scope="module")
def tiny_model(tiny_frames):
    pytest.importorskip("pycolmap")
    from src.recon.track_a_colmap import run_track_a

    root, images = tiny_frames
    cfg = load_config(overrides=["device.prefer=cpu", "run.mode=A", "recon.track_a.texture.enabled=false",
                                 "recon.track_a.dense.max_image_size=320"])
    outcome = run_track_a(images, root / "for_merge", cfg)
    import pycolmap

    return pycolmap.Reconstruction(str(outcome["artifacts"]["sparse"]))


def _split(model, gps_scale=7.3):
    import pycolmap

    from src.recon.merge import posed

    names = sorted(im.name for im in posed(model))
    gps = {im.name: np.asarray(im.projection_center()) * gps_scale + np.array([100.0, -40.0, 3.0])
           for im in posed(model)}

    def piece(keep):
        r = pycolmap.Reconstruction(model)
        for im in list(r.images.values()):
            if im.name not in keep:
                r.deregister_frame(im.frame_id)
        return r

    half = set(names[: len(names) // 2])
    a, b = piece(half), piece(set(names) - half)
    b.transform(pycolmap.Sim3d(0.37, pycolmap.Rotation3d(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])),
                               np.array([5.0, 9.0, -2.0])))
    return names, gps, a, b


def test_merge_places_a_moved_piece_back_exactly(tiny_model):
    from src.recon.merge import merge_by_gps, posed

    names, gps, a, b = _split(tiny_model)
    merged, info = merge_by_gps([a, b], gps)
    got = {im.name: np.asarray(im.projection_center()) for im in posed(merged)}
    assert info["merged"] == 1 and len(got) == len(names)
    for name, centre in got.items():
        assert np.allclose(centre, tiny_model.find_image_with_name(name).projection_center(), atol=1e-6)


def test_merge_refuses_a_piece_gps_cannot_place(tiny_model):
    from src.recon.merge import merge_by_gps, posed

    names, gps, a, b = _split(tiny_model)
    rng = np.random.default_rng(1)
    for im in posed(b):  # wreck the GPS of the second piece only
        gps[im.name] = gps[im.name] + rng.normal(0, 200.0, 3)
    merged, info = merge_by_gps([a, b], gps, max_rms_m=25.0)
    assert info["merged"] == 0 and info["skipped"]
    assert len(posed(merged)) == len(posed(a))
