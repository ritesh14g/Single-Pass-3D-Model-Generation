"""Stage 5: CRS and datum choice, the RANSAC similarity (incl. straight flight lines), rasters,
every writer read back, and an end-to-end georeference + export on a tiny synthetic flight."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.config import load_config
from src.export import rasters, writers
from src.geo import crs as geocrs
from src.geo.georef import Georef, fit_similarity
from src.qa.stage1_eval import FAIL, PASS
from src.recon.alignment import apply
from tests import fixtures


# -- CRS and datum --------------------------------------------------------------
def test_utm_zone_selection():
    assert geocrs.utm_epsg(77.59, 12.97) == 32643     # Bengaluru, zone 43N
    assert geocrs.utm_epsg(-81.86, 27.27) == 32617    # Esri clip, Florida, 17N
    assert geocrs.utm_epsg(151.2, -33.9) == 32756     # Sydney, 56S


def test_altitude_datum_follows_the_telemetry_source():
    assert geocrs.gps_altitude_datum("klv:clip.ts", "auto") == ("orthometric", False)
    assert geocrs.gps_altitude_datum("srt:dji.srt", "auto") == ("ellipsoidal", True)
    assert geocrs.gps_altitude_datum("srt:dji.srt", "orthometric") == ("orthometric", False)


def test_missing_geoid_falls_back_and_says_so(monkeypatch):
    import pyproj

    real = pyproj.Transformer.from_crs

    def no_grid(src, dst, **kwargs):
        if "+" in str(dst):
            raise pyproj.exceptions.ProjError("grid not found")
        return real(src, dst, **kwargs)

    monkeypatch.setattr(pyproj.Transformer, "from_crs", staticmethod(no_grid))
    enh, vertical, applied, note = geocrs.gps_to_map(np.array([77.59]), np.array([12.97]), np.array([900.0]),
                                                     horizontal_epsg=32643, geoid_model="EGM96",
                                                     want_orthometric=True, gps_datum="ellipsoidal", network=False)
    assert vertical is None and not applied and "geoid unavailable" in note
    assert enh[0, 2] == 900.0  # heights left as given, not silently relabelled


# -- similarity fit ----------------------------------------------------------------
def _rotation(a, b):
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return rz @ rx


def test_ransac_flags_gross_gps_outliers():
    rng = np.random.default_rng(0)
    rot, s, t = _rotation(0.6, 0.2), 4.0, np.array([10.0, -5.0, 110.0])
    world = np.c_[np.linspace(0, 300, 30), 40 * np.sin(np.linspace(0, 3, 30)), np.full(30, 111.0)]
    model = ((world - t) @ rot) / s
    noisy = world + rng.normal(0, 0.5, world.shape)
    noisy[[4, 17]] += [60.0, -40.0, 0.0]
    transform, inliers, info, _ = fit_similarity(model, noisy, iterations=2000, threshold_m=10, min_inliers=8)
    assert sorted(np.flatnonzero(~inliers)) == [4, 17]
    assert transform[0] == pytest.approx(s, rel=0.01)
    assert info["rms_inliers_m"] < 1.5 < info["rms_all_m"]  # the headline keeps the outliers in


def test_straight_flight_line_needs_the_ground_constraint():
    rng = np.random.default_rng(1)
    rot, s, t = _rotation(0.6, 0.2), 4.0, np.array([10.0, -5.0, 110.0])
    world = np.c_[np.linspace(0, 300, 30), np.zeros(30), np.full(30, 111.0)]
    ground_w = np.c_[rng.uniform(-20, 320, 3000), rng.uniform(-80, 80, 3000), np.zeros(3000)]
    model, ground_m = ((world - t) @ rot) / s, ((ground_w - t) @ rot) / s
    noisy = world + rng.normal(0, 0.3, world.shape)  # real GPS noise is what lets the roll wander
    free, _, info_free, _ = fit_similarity(model, noisy, iterations=300, threshold_m=10, min_inliers=8)
    fixed, _, info, _ = fit_similarity(model, noisy, iterations=300, threshold_m=10, min_inliers=8,
                                       ground_points=ground_m)
    assert not info_free["ground_constraint"] and info["ground_constraint"]
    assert np.ptp(apply(fixed, ground_m)[:, 2]) < 0.5            # ground comes out level
    assert np.ptp(apply(free, ground_m)[:, 2]) > 10              # without it, it can roll


# -- rasters and writers ------------------------------------------------------------
def test_dsm_rasterises_heights_and_fills_small_gaps_only():
    xs, ys = np.meshgrid(np.arange(0, 20, 0.25), np.arange(0, 10, 0.25))
    pts = np.c_[xs.ravel(), ys.ravel(), 100 + 0.1 * xs.ravel()]
    keep = ~((pts[:, 0] > 5) & (pts[:, 0] < 5.6))                 # a narrow strip of missing data
    keep &= ~((pts[:, 0] > 12) & (pts[:, 0] < 17))                # and a wide one
    grid, dsm, ortho, filled = rasters.rasterize(pts[keep], np.full((keep.sum(), 3), 120, np.uint8), 0.5, 2, -9999.0)
    row, col = grid.cells(np.array([[10.2, 5.2], [5.3, 5.2], [14.5, 5.2]]))
    assert dsm[row[0], col[0]] == pytest.approx(101.0, abs=0.06)
    assert dsm[row[1], col[1]] != -9999.0 and filled > 0          # narrow gap filled
    assert dsm[row[2], col[2]] == -9999.0                         # wide gap stays nodata
    assert ortho[row[0], col[0], 3] == 255


def test_auto_resolution_counts_unique_samples():
    rng = np.random.default_rng(2)
    pts = np.c_[rng.uniform(0, 100, 40000), rng.uniform(0, 100, 40000), np.zeros(40000)]
    # 40k points over 10 000 m^2 seen 4 times each -> 10k unique samples -> ~1 m spacing
    assert rasters.auto_resolution(pts, np.full(40000, 4)) == pytest.approx(1.0, abs=0.1)


def test_las_and_geotiff_carry_the_compound_crs(tmp_path):
    import laspy
    import rasterio

    xyz = np.c_[415000 + np.arange(10.0), 3016700 + np.arange(10.0), 140 + np.zeros(10)]
    writers.write_las(tmp_path / "c.las", xyz, np.zeros((10, 3), np.uint8), np.arange(10),
                      np.linspace(0, 1, 10).astype(np.float32), "EPSG:32617+5773", [0.001] * 3)
    las = laspy.read(tmp_path / "c.las")
    crs = las.header.parse_crs()
    assert crs.is_compound and [c.to_epsg() for c in crs.sub_crs_list] == [32617, 5773]
    assert np.allclose(las.x, xyz[:, 0]) and las.confidence[-1] == pytest.approx(1.0)
    grid = rasters.Grid(0.0, 10.0, 1.0, 10, 10)
    dsm = np.full((10, 10), 5.0, np.float32)
    ortho = np.zeros((10, 10, 4), np.uint8)
    dsm_path, _ = rasters.write_geotiffs(tmp_path, grid, dsm, ortho, (415000.0, 3016700.0, 0.0), 32617,
                                         "EGM96 orthometric", -9999.0)
    with rasterio.open(dsm_path) as src:
        assert src.crs.to_epsg() == 32617 and src.transform.c == 415000.0 and src.tags()["VERTICAL_DATUM"]


def test_textured_obj_is_transformed_losslessly(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "tex.jpg").write_bytes(b"\xff\xd8 not really a jpeg, but it must be copied byte for byte")
    (src / "m.mtl").write_text("newmtl a\nmap_Kd tex.jpg\n")
    (src / "m.obj").write_text("mtllib m.mtl\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvt 0 0\nvn 0 0 1\nusemtl a\nf 1/1/1 2/1/1 3/1/1\n")
    g = Georef(None, 2.0, _rotation(0.3, 0.0), np.array([5.0, 6.0, 7.0]), {})
    out = tmp_path / "out"
    out.mkdir()
    writers.transform_obj(src / "m.obj", out / "m.obj", g)
    verts = [list(map(float, ln.split()[1:])) for ln in (out / "m.obj").read_text().splitlines() if ln.startswith("v ")]
    assert np.allclose(verts, g.to_local(np.eye(3)), atol=1e-4)
    assert (out / "tex.jpg").read_bytes() == (src / "tex.jpg").read_bytes()
    assert "f 1/1/1 2/1/1 3/1/1" in (out / "m.obj").read_text()


def test_glb_is_y_up_and_keeps_the_offset(tmp_path):
    import pygltflib
    import trimesh

    writers.write_glb(tmp_path / "b.glb", trimesh.creation.box(extents=[4, 2, 1]), {"offset": [1, 2, 0]})
    assert pygltflib.GLTF2().load(str(tmp_path / "b.glb")).extras["offset"] == [1, 2, 0]
    assert np.allclose(trimesh.load(tmp_path / "b.glb", force="mesh").extents, [4, 1, 2])


def test_vis_counts_are_read_back(tmp_path):
    from src.recon.track_b_vggt import write_colmap_fused

    vis = [[0], [1, 2], [3, 0, 5]]
    write_colmap_fused(tmp_path / "f.ply", np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3), np.uint8), vis)
    xyz, rgb, views = writers.read_dense(tmp_path / "f.ply")
    assert views.tolist() == [1, 2, 3]


# -- end to end ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def georeferenced_run(tmp_path_factory):
    pytest.importorskip("pycolmap")
    import pycolmap

    from src.export.stage import run_export
    from src.geo.stage import run_geo
    from src.recon.merge import posed
    from src.recon.track_a_colmap import run_track_a

    root = tmp_path_factory.mktemp("stage5")
    # T-1: the same pan as fixtures.make_flight_video (64 frames, 320x240, 93% overlap, every 4th
    # frame kept), cropped straight from the canvas. Through an mp4v encode/decode the pixels
    # differed between the Windows and Linux OpenCV builds, and on this flat scene with a
    # self-calibrated focal SfM sometimes degenerated (the box: float32 overflow in glTF, a
    # Blender hang). A fixed field of view, as real telemetry gives, makes the focal known.
    width, height, frames, overlap = 320, 240, 64, 0.93
    step = (1.0 - overlap) * width
    canvas = fixtures.make_canvas(int(width + step * frames + 8), height + 8, seed=7)
    images = root / "images"
    images.mkdir()
    for index in range(0, frames, 4):
        x = int(round(index * step))
        cv2.imwrite(str(images / f"frame_{index:06d}.jpg"), canvas[0:height, x:x + width],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    telemetry = root / "telemetry.parquet"
    pd.DataFrame({"t": [0.0], "hfov_deg": [60.0]}).to_parquet(telemetry)
    cfg = load_config(overrides=["device.prefer=cpu", "run.mode=A", "recon.track_a.dense.max_image_size=320"])
    track_a = run_track_a(images, root / "track_a", cfg, telemetry_path=telemetry)["artifacts"]

    # GPS = the SfM cameras, levelled (ground plane horizontal, as in the real world), scaled 12x
    # and moved to Bengaluru: the fit must recover exactly that.
    rec = pycolmap.Reconstruction(str(track_a["sparse"]))
    from src.geo.georef import _ground_normal

    pts = np.array([p.xyz for p in rec.points3D.values()])
    centres = np.array([im.projection_center() for im in posed(rec)])
    up = _ground_normal(pts, centres)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    angle = math.acos(np.clip(up @ [0.0, 0.0, 1.0], -1, 1))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / max(np.linalg.norm(axis), 1e-12)
    level = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    lines = ["EPSG:4326"]
    for im in posed(rec):
        e, n, h = level @ np.asarray(im.projection_center()) * 12.0 + np.array([781000.0, 1435000.0, 900.0])
        lon, lat = to_ll.transform(e, n)
        lines.append(f"{im.name} {lon:.9f} {lat:.9f} {h:.3f}")
    geo_txt = root / "geo.txt"
    geo_txt.write_text("\n".join(lines) + "\n")
    geo = run_geo(track_a["sparse"], geo_txt, root / "run" / "geo", cfg, telemetry_source="srt:x")
    export = run_export(track_a, geo["artifacts"]["georef"], root / "run" / "export", cfg)
    return root / "run", geo, export, cfg, track_a


def test_end_to_end_georeference_recovers_the_gps(georeferenced_run):
    _, geo, _, _, _ = georeferenced_run
    m = geo["metrics"]
    assert m["referenced"] and m["crs"].startswith("EPSG:32643")
    assert m["rms_all_m"] < 0.05 and m["scale_model_to_m"] == pytest.approx(12.0, rel=1e-3)


def test_end_to_end_formats_are_written_and_verified(georeferenced_run):
    from src.qa.stage5_eval import ExportOutputs, evaluate_export

    run_dir, _, export, cfg, _ = georeferenced_run
    produced = set(export["metrics"]["formats_produced"])
    assert {"ply", "las", "geotiff"} <= produced
    has_mesh = "mesh" in export["metrics"].get("track_a_keys", []) or "obj" in produced
    if has_mesh:  # this flat 16-frame scene sometimes meshes to nothing (Stage 4 reports it)
        assert {"obj", "glb"} <= produced
        if writers.find_blender(None) is not None:
            assert "fbx" in produced
    else:
        assert "no mesh" in export["metrics"]["formats_failed"]["obj"]
    ev = evaluate_export(ExportOutputs.load(run_dir), cfg)
    kpis = {k.key: k for k in ev.kpis}
    for key in ("format_ply", "format_las", "format_geotiff_dsm", "format_geotiff_ortho") + (
            ("format_obj", "format_glb") if has_mesh else ()):
        assert kpis[key].status == PASS, kpis[key].detail
    assert kpis["cam_vs_gps_rms_m"].status == PASS
    assert kpis["format_fbx"].status in (PASS, FAIL)  # FAIL only where no Blender is installed


def test_end_to_end_stage3_zones_travel_with_the_export(georeferenced_run, tmp_path):
    """Stage 3 on real Track A output, then export with its layers: zones in PLY/LAS, gaps copied."""
    import json

    import laspy

    from src.export.stage import run_export
    from src.fusion.stage import run_fusion
    from src.qa.stage3_eval import FusionOutputs, evaluate_fusion

    run_dir, _, _, cfg, track_a = georeferenced_run
    georef = run_dir / "geo" / "georef.json"
    fusion = run_fusion(track_a, georef, tmp_path / "fusion", cfg)
    m = fusion["metrics"]
    assert m["voxel"]["voxels"] > 0 and m["referenced"] and m["units"] == "m"
    assert 0.0 <= m["coverage_pct"] <= 100.0
    assert json.loads((tmp_path / "fusion" / "gaps.geojson").read_text())["type"] == "FeatureCollection"
    export = run_export(track_a, georef, tmp_path / "export", cfg, fusion=fusion["artifacts"])
    zones = export["metrics"]["zones"]
    assert zones is not None and zones["coverage_pct"] == m["coverage_pct"]
    assert export["metrics"]["coverage"]["coverage_pct"] == m["coverage_pct"]   # one coverage number
    las = laspy.read(tmp_path / "export" / "cloud.las")
    assert set(np.unique(np.asarray(las.zone))) <= {1, 2} and len(las.x) == m["points"] - m["rejected_near_camera"]
    assert "zone" in (tmp_path / "export" / "cloud.ply").read_bytes()[:2048].decode("ascii", "ignore")
    assert (tmp_path / "export" / "gaps.geojson").is_file()
    (tmp_path / "manifest.json").write_text(json.dumps({"config": cfg.to_dict()}))
    kpis = {k.key: k for k in evaluate_fusion(FusionOutputs.load(tmp_path), cfg).kpis}
    assert kpis["gaps_geojson"].status == PASS and kpis["zone1_untouched"].status == PASS


def test_broken_mesh_is_named_not_exported():
    cloud = np.random.default_rng(1).uniform(0, 100, (500, 3))
    good = cloud[:50] + 1.0
    assert writers.mesh_is_broken(good, cloud, 10.0) is None
    assert "not finite" in writers.mesh_is_broken(np.r_[good, [[np.inf, 0, 0]]], cloud, 10.0)
    assert "outside it" in writers.mesh_is_broken(np.r_[good, [[1e30, 0, 0]]], cloud, 10.0)
    assert writers.mesh_is_broken(np.zeros((0, 3)), cloud, 10.0) == "the mesh has no vertices"


def test_end_to_end_absurd_mesh_skips_mesh_formats_keeps_points(georeferenced_run, tmp_path):
    """T-1: a degenerate mesh overflowed glTF's float32 and hung Blender on the box."""
    from src.export.stage import run_export

    run_dir, _, _, cfg, track_a = georeferenced_run
    bad = tmp_path / "bad.obj"
    bad.write_text("\n".join(["v 0 0 0", "v 1 0 0", "v 0 1 0", "v 1e30 1e30 0", "f 1 2 3", "f 2 3 4"]) + "\n")
    broken = {k: v for k, v in track_a.items() if k != "textured"}
    broken["mesh"] = bad
    export = run_export(broken, run_dir / "geo" / "georef.json", tmp_path / "export", cfg)
    failed, produced = export["metrics"]["formats_failed"], set(export["metrics"]["formats_produced"])
    assert all("mesh rejected" in failed[f] for f in ("obj", "glb", "fbx"))
    assert {"ply", "las", "geotiff"} <= produced and not {"obj", "glb", "fbx"} & produced


def test_unreferenced_run_is_reported_not_crashed(tmp_path):
    pytest.importorskip("pycolmap")
    from src.geo.stage import run_geo

    # A GPS file with no matching frames: the result must be an explicit, unreferenced georef.
    geo_txt = tmp_path / "geo.txt"
    geo_txt.write_text("EPSG:4326\nnot_a_frame.jpg 77.0 12.0 900\n")
    sparse = next(Path(__file__).resolve().parents[1].glob("data/interim/*/track_a/sparse/*/images.bin"), None)
    if sparse is None:
        pytest.skip("no local sparse model to reuse")
    out = run_geo(sparse.parent, geo_txt, tmp_path / "geo", load_config())
    assert out["metrics"]["referenced"] is False and out["metrics"]["downgrades"]
