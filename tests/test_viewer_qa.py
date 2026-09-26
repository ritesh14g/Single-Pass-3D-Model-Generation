"""Stage 6: the web viewer package, accuracy against a reference surface, the QA report."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src.core.config import load_config
from src.export import writers
from src.qa.stage1_eval import FAIL, INFO, PASS, WARN

OFFSET = [781000.0, 1435000.0, 0.0]
EPSG = 32643


def _terrain(n=21, size=100.0):
    """A textured height field: vertices (local, z-up), faces, per-vertex UV."""
    xs = np.linspace(0.0, size, n)
    gx, gy = np.meshgrid(xs, xs)
    gz = 900.0 + 3.0 * np.sin(gx / 15.0) * np.cos(gy / 20.0)
    vertices = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]
    uv = np.c_[gx.ravel() / size, gy.ravel() / size]
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a = r * n + c
            faces += [[a, a + 1, a + n + 1], [a, a + n + 1, a + n]]
    return vertices, np.array(faces), uv


def _textured_mesh(vertices, faces, uv):
    import trimesh
    from PIL import Image

    img = np.zeros((64, 64, 3), np.uint8)
    img[..., 0] = np.linspace(0, 255, 64)[None, :]
    img[..., 1] = np.linspace(0, 255, 64)[:, None]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(img))
    return mesh


def _gaps_geojson(path: Path, ring_local: np.ndarray):
    from pyproj import Transformer

    to_ll = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(ring_local[:, 0] + OFFSET[0], ring_local[:, 1] + OFFSET[1])
    doc = {"type": "FeatureCollection", "properties": {"crs": "EPSG:4326 (lon, lat)"},
           "features": [{"type": "Feature", "properties": {"area_m2": 100.0, "kind": "interior", "zone": 3},
                         "geometry": {"type": "Polygon", "coordinates": [np.c_[lon, lat].tolist()]}}]}
    path.write_text(json.dumps(doc))


def _write_dsm(path: Path, z: np.ndarray, x0: float, y1: float, res: float, crs: str):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                       crs=crs, transform=from_origin(x0, y1, res, res), nodata=-9999.0) as dst:
        dst.write(np.where(np.isfinite(z), z, -9999.0).astype(np.float32), 1)


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    """A georeferenced export folder the way Stage 5 writes one, with Stage 3's layers."""
    root = tmp_path_factory.mktemp("s6")
    run = root / "run"
    export = run / "export"
    export.mkdir(parents=True)
    vertices, faces, uv = _terrain()
    mesh = _textured_mesh(vertices, faces, uv)
    mesh.export(str(export / "model.obj"))
    # cloud: every vertex is a point; confidence = x / 100; zone 2 for x > 60
    conf = (vertices[:, 0] / 100.0).astype(np.float32)
    zone = np.where(vertices[:, 0] > 60, 2, 1).astype(np.uint8)
    writers.write_ply(export / "cloud.ply", vertices, np.full((len(vertices), 3), 128, np.uint8),
                      np.full(len(vertices), 3), conf, extra={"zone": zone, "source": np.zeros(len(vertices), np.uint8)})
    face_zone = np.where(vertices[faces].mean(1)[:, 1] > 80, 3, np.where(vertices[faces].mean(1)[:, 0] > 60, 2, 1))
    writers.write_zones_glb(export / "model_zones.glb", mesh, face_zone.astype(np.uint8), {})
    _gaps_geojson(export / "gaps.geojson", np.array([[20, 20], [40, 20], [40, 40], [20, 40], [20, 20]], float))
    meta = {"crs": f"EPSG:{EPSG}+5773",
            "georeferencing": {"frame": {"horizontal_epsg": EPSG, "vertical_datum": "EGM96 orthometric"},
                               "rms_all_m": 0.42, "rms_inliers_m": 0.4, "horizontal_rms_m": 0.3, "vertical_rms_m": 0.2,
                               "inliers": 10, "cameras": 10},
            "coordinate_frames": {"offset": OFFSET},
            "coverage": {"coverage_pct": 81.5, "ground_plane_z_m": 900.0},
            "zones": {"zone1_pct": 60.0, "zone2_pct": 21.5, "zone3_pct": 18.5, "gaps": {"count": 1, "total_m2": 400.0}},
            "confidence": "views confirming each point / 5, clipped to 1",
            "formats_produced": ["fbx", "geotiff", "glb", "las", "obj", "ply"], "formats_failed": {},
            "processing": {"preset": "default", "stage_seconds": {"ingest": 10.0, "track_a": 50.0}, "video": "x/clip.mp4"}}
    (export / "metadata.json").write_text(json.dumps(meta))
    return export


# -- viewer package ------------------------------------------------------------------------------
def test_scene_glb_is_y_up_with_custom_layers(tmp_path):
    import pygltflib

    from src.viewer.package import write_scene_glb

    v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    uv = np.array([[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]])
    path = write_scene_glb(tmp_path / "s.glb", v, np.array([[0, 1, 2]]), uv, b"\xff\xd8fakejpeg",
                           np.array([0.5, -1.0, 1.0]), np.array([1, 2, 3]), {"offset": OFFSET})
    g = pygltflib.GLTF2().load(str(path))
    blob = g.binary_blob()
    attrs = g.meshes[0].primitives[0].attributes

    def read(index, dtype, width):
        acc = g.accessors[index]
        view = g.bufferViews[acc.bufferView]
        return np.frombuffer(blob[view.byteOffset:view.byteOffset + view.byteLength], dtype).reshape(-1, width)

    assert np.allclose(read(attrs.POSITION, np.float32, 3)[0], [1.0, 3.0, -2.0])        # (x, z, -y)
    assert np.allclose(read(attrs.TEXCOORD_0, np.float32, 2)[1], [1.0, 0.75])          # v flipped
    assert np.allclose(read(attrs._CONFIDENCE, np.float32, 1).ravel(), [0.5, -1.0, 1.0])
    assert np.allclose(read(attrs._ZONE, np.float32, 1).ravel(), [1, 2, 3])
    assert g.images[0].mimeType == "image/jpeg" and g.extras["offset"] == OFFSET


def test_viewer_package_carries_texture_confidence_zones_and_gaps(export_dir, tmp_path):
    from src.viewer.package import SCENE_MARKER, build_viewer

    out = build_viewer(export_dir, tmp_path / "viewer", load_config().get_path("qa.viewer"))
    m = out["metrics"]
    assert m["layers"] == {"texture": True, "confidence": True, "zones": True}
    assert m["confidence_known_pct"] == 100.0 and m["gaps_drawn"] == 1
    assert m["zone_pct_vertices"]["3"] > 0 and m["zone_pct_vertices"]["1"] > 0
    page = (tmp_path / "viewer" / "index.html").read_text(encoding="utf-8")
    assert SCENE_MARKER not in page and '"clip.mp4"' in page
    assert (tmp_path / "viewer" / "vendor" / "three.module.min.js").is_file()
    scene = json.loads((tmp_path / "viewer" / "scene.json").read_text())
    ring = np.array(scene["gaps"][0]["rings"][0])
    # gaps.geojson (lon/lat) back in the local frame, draped on the terrain
    assert np.allclose(ring[:4, :2], [[20, 20], [40, 20], [40, 40], [20, 40]], atol=0.05)
    assert np.all(np.abs(ring[:, 2] - 900.0) <= 3.01)
    assert scene["stats"]["cam_vs_gps_rms_m"] == 0.42 and scene["stats"]["total_seconds"] == 60.0


def test_viewer_zones_come_from_model_zones_glb(export_dir, tmp_path):
    import pygltflib

    from src.viewer.package import build_viewer

    build_viewer(export_dir, tmp_path / "v", load_config().get_path("qa.viewer"))
    g = pygltflib.GLTF2().load(str(tmp_path / "v" / "scene.glb"))
    blob, attrs = g.binary_blob(), g.meshes[0].primitives[0].attributes

    def read(index, width):
        acc = g.accessors[index]
        view = g.bufferViews[acc.bufferView]
        return np.frombuffer(blob[view.byteOffset:view.byteOffset + view.byteLength], np.float32).reshape(-1, width)

    pos, zone = read(attrs.POSITION, 3), read(attrs._ZONE, 1).ravel()
    north = -pos[:, 2]                        # three.js -z = north
    assert np.all(zone[north > 95] == 3)      # every vertex of the northern faces is inferred
    assert np.all(zone[(north < 75) & (pos[:, 0] < 55)] == 1)


def test_viewer_without_stage3_says_so(export_dir, tmp_path):
    import shutil

    from src.viewer.package import build_viewer

    bare = tmp_path / "export"
    shutil.copytree(export_dir, bare)
    (bare / "model_zones.glb").unlink()
    (bare / "gaps.geojson").unlink()
    ply = bare / "cloud.ply"
    v, _, _ = _terrain()
    writers.write_ply(ply, v, np.full((len(v), 3), 128, np.uint8), np.full(len(v), 3),
                      np.full(len(v), 0.8, np.float32))
    m = build_viewer(bare, tmp_path / "v", load_config().get_path("qa.viewer"))["metrics"]
    assert m["layers"]["zones"] is False and m["gaps_drawn"] == 0
    assert any("Stage 3 did not run" in n for n in m["notes"])


def test_viewer_server_sends_module_scripts_as_javascript(export_dir, tmp_path):
    from src.viewer.package import build_viewer
    from src.viewer.serve import make_server, url_of

    build_viewer(export_dir, tmp_path / "v", load_config().get_path("qa.viewer"))
    server = make_server(tmp_path / "v", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = url_of(server)
        for path, kind in (("", "text/html"), ("vendor/three.module.min.js", "text/javascript"),
                           ("scene.glb", "model/gltf-binary")):
            with urllib.request.urlopen(url + path) as resp:
                assert resp.headers["Content-Type"].startswith(kind), path
    finally:
        server.shutdown()
        server.server_close()


# -- accuracy against a reference -------------------------------------------------------------------
def _surface(n=200, seed=3):
    """Terrain-like heights (1 m cells) with structure at several scales, so correlation locks on."""
    import cv2

    rng = np.random.default_rng(seed)
    z = sum(cv2.GaussianBlur(rng.normal(0, a, (n, n)), (0, 0), s) * s for a, s in ((1.0, 2), (1.0, 5), (1.0, 12)))
    return 50.0 + z


def test_error_stats_are_robust_and_signed():
    from src.qa.metrics import error_stats

    d = np.r_[np.full(99, 0.5), 50.0, np.nan]
    s = error_stats(d)
    assert s["n"] == 100 and s["median_m"] == 0.5 and s["nmad_m"] == 0.0 and s["mean_m"] > 0.5
    assert error_stats(np.array([np.nan]))["n"] == 0


def test_surface_shift_recovers_a_known_displacement():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    from src.qa.metrics import Surface, surface_shift

    truth = _surface()
    ours = np.roll(np.roll(truth, 3, axis=1), 2, axis=0)      # 3 m east, 2 m south
    t = from_origin(0, 200, 1.0, 1.0)
    shift = surface_shift(Surface(ours, t, CRS.from_epsg(EPSG)), Surface(truth, t, CRS.from_epsg(EPSG)))
    assert shift["estimated"]
    assert shift["east_m"] == pytest.approx(3.0, abs=0.3) and shift["north_m"] == pytest.approx(-2.0, abs=0.3)


def test_cloud_to_cloud_between_reconstructions():
    from src.qa.metrics import cloud_to_cloud

    a = np.random.default_rng(0).uniform(0, 10, (500, 3))
    s = cloud_to_cloud(a + [0.0, 0.0, 0.1], a, 1.0)
    assert s["median_m"] == pytest.approx(0.1, abs=1e-6) and s["matched_pct"] == 100.0


@pytest.fixture()
def accuracy_export(tmp_path):
    """Our dsm.tif + cloud.las over a known surface: zone 1 +0.2 m, zone 2 noisy, fill -1 m."""
    truth = _surface(120)
    x0, y1 = OFFSET[0], OFFSET[1] + 120
    export = tmp_path / "export"
    export.mkdir()
    _write_dsm(export / "dsm.tif", truth, x0, y1, 1.0, f"EPSG:{EPSG}")
    rng = np.random.default_rng(1)
    n = 6000
    xy = rng.uniform(5, 115, (n, 2))
    col, row = xy[:, 0] - 0.5, (120 - xy[:, 1]) - 0.5
    from scipy.ndimage import map_coordinates

    z_true = map_coordinates(truth, [row, col], order=1)
    zone = np.where(np.arange(n) < 3000, 1, 2).astype(np.uint8)
    source = np.where(np.arange(n) >= 5000, 1, 0).astype(np.uint8)
    z = z_true + np.where(zone == 1, 0.2, rng.normal(0, 0.5, n))
    z[source == 1] = z_true[source == 1] - 1.0
    classes = np.where(np.arange(n) % 2 == 0, 2, 1).astype(np.uint8)
    map_xyz = np.c_[xy[:, 0] + x0, xy[:, 1] + OFFSET[1], z]
    writers.write_las(export / "cloud.las", map_xyz, np.full((n, 3), 100, np.uint8), np.full(n, 3),
                      np.ones(n, np.float32), f"EPSG:{EPSG}", [0.001] * 3, extra={"zone": zone, "source": source},
                      classification=classes)
    return export, truth, (x0, y1)


def test_accuracy_per_zone_against_a_reference_dsm(accuracy_export, tmp_path):
    from src.qa.metrics import accuracy_vs_reference

    export, truth, (x0, y1) = accuracy_export
    ref = tmp_path / "ref.tif"
    _write_dsm(ref, truth, x0, y1, 1.0, f"EPSG:{EPSG}")
    acc = accuracy_vs_reference(export, ref, load_config().get_path("qa.metrics"))
    pts = acc["points_vs_reference_dsm"]
    assert pts["zone1_measured"]["mean_m"] == pytest.approx(0.2, abs=0.02)
    assert pts["zone1_measured"]["nmad_m"] < 0.05
    assert pts["zone2_measured"]["nmad_m"] == pytest.approx(0.5, abs=0.1)
    assert pts["zone2_fill"]["median_m"] == pytest.approx(-1.0, abs=0.02)
    assert acc["horizontal_shift"]["horizontal_m"] < 0.3
    assert "taken as-is" in acc["reference"]["vertical"]


def test_accuracy_against_a_lidar_las_in_another_crs(accuracy_export, tmp_path):
    """Reference points in WGS84 lon/lat: reprojected; its class-2 ground gives the ground figure."""
    from pyproj import Transformer

    from src.qa.metrics import accuracy_vs_reference

    export, truth, (x0, y1) = accuracy_export
    gy, gx = np.mgrid[0:120, 0:120]
    e, n = x0 + gx.ravel() + 0.5, y1 - gy.ravel() - 0.5
    lon, lat = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True).transform(e, n)
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales, header.offsets = [1e-7, 1e-7, 0.001], [np.floor(lon.min()), np.floor(lat.min()), 0.0]
    header.add_crs(__import__("pyproj").CRS.from_epsg(4326))
    las = laspy.LasData(header)
    las.x, las.y, las.z = lon, lat, truth.ravel()
    las.classification = np.full(len(lon), 2, np.uint8)
    las.write(str(tmp_path / "ref.las"))
    acc = accuracy_vs_reference(export, tmp_path / "ref.las", load_config().get_path("qa.metrics"))
    assert acc["reference"]["kind"] == "point cloud" and acc["reference"]["ground_points"] > 10000
    # max-per-cell of a 1 m lidar grid vs bilinear heights: a few cm of discretisation
    assert acc["points_vs_reference_dsm"]["zone1_measured"]["median_m"] == pytest.approx(0.2, abs=0.25)
    assert acc["ground_points"]["against"].startswith("reference ground")


# -- stage runner, report, scorecard ---------------------------------------------------------------
def test_qa_stage_writes_viewer_and_report_and_scores_them(export_dir):
    from src.qa.stage import run_qa
    from src.qa.stage6_eval import QaOutputs, evaluate_qa

    run = export_dir.parent
    (run / "manifest.json").write_text(json.dumps({"config": {}, "stages": {
        "ingest": {"status": "done", "duration_s": 10.0, "metrics": {"video": {"duration_s": 60.0}}},
        "export": {"status": "done", "duration_s": 5.0}}}))
    cfg = load_config()
    out = run_qa(run, run / "qa", cfg)
    assert not out["metrics"]["failures"], out["metrics"]["failures"]
    assert {"viewer", "report_json", "report_html"} <= set(out["artifacts"])
    html = (run / "qa" / "report.html").read_text(encoding="utf-8")
    assert "Limitations" in html
    report = json.loads((run / "qa" / "report.json").read_text())
    assert report["scorecards"]["ingest"]["skipped"] == "did not run in this run"
    assert any("reference surface" in x for x in report["limitations"])
    kpis = {k.key: k for k in evaluate_qa(QaOutputs.load(run), cfg).kpis}
    for key in ("viewer_built", "scene_glb", "layer_confidence", "layer_zones", "layer_texture", "report"):
        assert kpis[key].status == PASS, (key, kpis[key].detail)
    assert kpis["reference"].status == INFO and kpis["zone_accuracy"].status == WARN
    assert kpis["projected_10min_s"].value == pytest.approx(15.0 * 600 / 60 / 60, abs=0.1)


def test_qa_scorecard_fails_without_a_viewer(tmp_path):
    from src.qa.stage6_eval import QaOutputs, evaluate_qa

    kpis = {k.key: k for k in evaluate_qa(QaOutputs.load(tmp_path), load_config()).kpis}
    assert kpis["viewer_built"].status == FAIL and kpis["report"].status == FAIL


def test_cli_qa_on_a_copied_run(export_dir, tmp_path):
    import shutil

    from click.testing import CliRunner

    from src.cli import cli

    run = tmp_path / "copied"
    shutil.copytree(export_dir.parent, run, ignore=shutil.ignore_patterns("qa"))
    result = CliRunner().invoke(cli, ["qa", str(run)])
    assert result.exit_code == 0, result.output
    assert (run / "qa" / "viewer" / "scene.glb").is_file() and (run / "qa" / "report.html").is_file()


def test_local_accuracy_removes_each_tiles_offset(tmp_path):
    """Our model 4 m east and 1 m high of the truth, shape exact: absolute error large, local ~0."""
    from scipy.ndimage import map_coordinates

    from src.qa.metrics import accuracy_vs_reference

    size = 260
    truth = _surface(size, seed=5)
    ours = np.roll(truth, 4, axis=1) + 1.0
    x0, y1 = OFFSET[0], OFFSET[1] + size
    export = tmp_path / "export"
    export.mkdir()
    _write_dsm(export / "dsm.tif", ours, x0, y1, 1.0, f"EPSG:{EPSG}")
    _write_dsm(tmp_path / "ref.tif", truth, x0, y1, 1.0, f"EPSG:{EPSG}")
    rng = np.random.default_rng(2)
    xy = rng.uniform(25, size - 25, (20000, 2))
    z = map_coordinates(ours, [(size - xy[:, 1]) - 0.5, xy[:, 0] - 0.5], order=1)
    n = len(xy)
    zone = np.where(np.arange(n) % 4 == 0, 2, 1).astype(np.uint8)
    writers.write_las(export / "cloud.las", np.c_[xy[:, 0] + x0, xy[:, 1] + OFFSET[1], z], np.full((n, 3), 90, np.uint8),
                      np.full(n, 3), np.ones(n, np.float32), f"EPSG:{EPSG}", [0.001] * 3,
                      extra={"zone": zone, "source": np.zeros(n, np.uint8)})
    acc = accuracy_vs_reference(export, tmp_path / "ref.tif", load_config().get_path("qa.metrics"))
    assert acc["points_vs_reference_dsm"]["zone1_measured"]["rms_m"] > 1.0
    local = acc["local"]
    assert local["tiles"] >= 2
    assert local["placement"]["horizontal_median_m"] == pytest.approx(4.0, abs=0.3)
    assert local["placement"]["vertical_median_m"] == pytest.approx(1.0, abs=0.1)
    assert local["points_after_tile_offset"]["zone1_measured"]["nmad_m"] < 0.1
