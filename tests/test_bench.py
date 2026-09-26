"""§8.5 benchmarks: the degradation harness and the single-pass simulation (orchestration and the
measurable pieces; the full reconstructions run on the GPU box)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.core.config import load_config
from src.qa import degrade
from src.qa.synthetic import make_canvas, make_flight_video


@pytest.fixture(scope="module")
def image():
    return make_canvas(640, 360, seed=3)


def _sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# -- frame degradations ------------------------------------------------------------------------------
def test_every_frame_degradation_is_deterministic_and_does_what_it_says(image):
    cfg = load_config()
    for kind in degrade.FRAME_KINDS:
        f = degrade._FRAME_FUNCS[kind]
        value = float(degrade.level(cfg, kind, "moderate"))
        a, b = f(image, 42, value, 0), f(image, 42, value, 0)
        assert np.array_equal(a, b), kind
        assert a.shape == image.shape and a.dtype == np.uint8 and not np.array_equal(a, image), kind
    blur = degrade.motion_blur(image, 0, 15, 0)
    assert _sharpness(blur) < 0.5 * _sharpness(image)
    assert degrade.low_light(image, 0, 0.35, 0).mean() < 0.5 * image.mean()
    dark = degrade.shadow(image, 0, 0.3, 0).astype(int).sum(-1) < 0.6 * image.astype(int).sum(-1)
    assert 0.1 < dark.mean() < 0.5
    moved = degrade.dynamic_objects(image, 0, 8, 0) != degrade.dynamic_objects(image, 30, 8, 0)
    assert moved.any()                                   # vehicles move between frames
    from src.condition.artifacts import blockiness_score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    jpeg = cv2.cvtColor(degrade.compression(image, 0, 8, 0), cv2.COLOR_BGR2GRAY)
    assert blockiness_score(jpeg) > blockiness_score(gray)


def test_severities_are_ordered(image):
    cfg = load_config()
    blur = [_sharpness(degrade.motion_blur(image, 0, degrade.level(cfg, "motion_blur", s), 0))
            for s in ("light", "moderate", "severe")]
    dark = [degrade.low_light(image, 0, degrade.level(cfg, "low_light", s), 0).mean()
            for s in ("light", "moderate", "severe")]
    assert blur[0] > blur[1] > blur[2] and dark[0] > dark[1] > dark[2]


def test_injection_is_off_by_default_and_reaches_the_decoded_frames(tmp_path):
    from src.ingest.video_reader import VideoReader, reader_options

    cfg = load_config(overrides=["device.prefer=cpu"])
    assert degrade.frame_degrader(cfg) is None and reader_options(cfg)["degrade"] is None
    video = make_flight_video(tmp_path / "v.mp4", frames=12).video_path
    blurred = cfg.merged({"qa": {"inject": {"kind": "motion_blur", "severity": "severe"}}})
    with VideoReader(video, **reader_options(cfg)) as r:
        clean = [f.image for f in r.read_indices([3, 7])]
    with VideoReader(video, **reader_options(blurred)) as r:
        dirty = [f.image for f in r.read_indices([3, 7])]
    for c, d in zip(clean, dirty):
        assert _sharpness(d) < 0.5 * _sharpness(c)


def test_injection_invalidates_cached_ingest_and_conditioning():
    from src.core.manifest import STAGE_CONFIG_DEPS

    assert "qa.inject" in STAGE_CONFIG_DEPS["ingest"] and "qa.inject" in STAGE_CONFIG_DEPS["condition"]


def test_gps_noise_injection():
    cfg = load_config()
    n = 4000
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "lat": np.full(n, 12.97), "lon": np.full(n, 77.59),
                          "alt_gps": np.full(n, 900.0)})
    same, info = degrade.inject_gps_noise(frame, cfg)
    assert info is None and same is frame
    noisy, info = degrade.inject_gps_noise(frame, cfg.merged({"qa": {"inject": {"kind": "gps_noise",
                                                                              "severity": "moderate"}}}))
    north = np.deg2rad(noisy["lat"] - 12.97) * 6378137.0
    assert info["sigma_m"] == 3.0 and 40 < info["outliers"] < 120            # 2% of 4000
    inliers = np.abs(north) < 15
    assert np.std(north[inliers]) == pytest.approx(3.0, rel=0.1)
    assert (np.abs(north) > 20).sum() >= info["outliers"] // 3


# -- the bench orchestration -------------------------------------------------------------------------
def _fake_export(run_dir: Path, z_offset: float = 0.0):
    """A finished run folder: manifest, export metadata, dsm.tif and cloud.las."""
    from tests.test_viewer_qa import _surface, _write_dsm

    from src.export import writers

    export = run_dir / "export"
    export.mkdir(parents=True, exist_ok=True)
    truth = _surface(80)
    _write_dsm(export / "dsm.tif", truth + z_offset, 500000.0, 1000080.0, 1.0, "EPSG:32643")
    n = 400
    xy = np.random.default_rng(0).uniform(5, 75, (n, 2))
    z = truth[(80 - xy[:, 1]).astype(int), xy[:, 0].astype(int)] + z_offset
    writers.write_las(export / "cloud.las", np.c_[xy + [500000.0, 1000000.0], z], np.full((n, 3), 90, np.uint8),
                      np.full(n, 3), np.ones(n, np.float32), "EPSG:32643", [0.001] * 3,
                      extra={"zone": np.ones(n, np.uint8), "source": np.zeros(n, np.uint8)})
    (export / "metadata.json").write_text(json.dumps({
        "georeferencing": {"rms_all_m": 1.0, "frame": {"gps_altitude_datum": "orthometric"}},
        "coverage": {"coverage_pct": 70.0}}))
    (run_dir / "manifest.json").write_text(json.dumps({"stages": {
        "track_a": {"status": "done", "duration_s": 10.0, "metrics": {"registered": 40, "frames_in": 42}}}}))


def test_degradation_bench_runs_every_case_both_ways_and_resumes(tmp_path):
    cfg = load_config()
    calls = []

    def runner(inputs, run_cfg, run_dir, stages):
        inject = run_cfg.get_path("qa.inject")
        calls.append((Path(run_dir).name, inject["kind"], run_cfg.get_path("condition.enabled"),
                      run_cfg.get_path("condition.gps.enabled"), stages))
        _fake_export(Path(run_dir), 0.0 if inject["kind"] == "none" else (0.5 if run_cfg.get_path("condition.enabled") else 2.0))

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    result = degrade.run_degradation_bench(video, tmp_path / "bench", cfg, kinds=["motion_blur", "gps_noise"],
                                           severities=["moderate"], runner=runner)
    assert [c[0] for c in calls] == ["clean", "motion_blur_moderate_cond", "motion_blur_moderate_raw",
                                     "gps_noise_moderate_cond", "gps_noise_moderate_raw"]
    assert calls[0][1] == "none" and calls[2][2] is False and calls[2][3] is False and calls[1][2] is True
    assert "preflight" not in calls[0][4]
    rows = {r["case"]: r for r in result["rows"]}
    assert rows["motion_blur_moderate_cond"]["surface_rms_m"] == pytest.approx(0.5, abs=0.02)
    assert rows["motion_blur_moderate_raw"]["surface_rms_m"] == pytest.approx(2.0, abs=0.02)
    assert "| motion_blur | moderate | 0.50 / 2.00 |" in result["table_md"]
    assert (tmp_path / "bench" / "degradation.json").is_file()
    calls.clear()
    degrade.run_degradation_bench(video, tmp_path / "bench", cfg, kinds=["motion_blur", "gps_noise"],
                                  severities=["moderate"], runner=runner)
    assert calls == []                                    # finished cases are reused

    from src.qa.stage import load_benchmarks

    assert "degradation" in load_benchmarks([tmp_path / "bench"])


def test_bench_rejects_unknown_kinds(tmp_path):
    with pytest.raises(ValueError, match="unknown degradation"):
        degrade.run_degradation_bench(tmp_path / "v.mp4", tmp_path / "b", load_config(), kinds=["fog"],
                                      runner=lambda *a: None)


def test_a_case_that_breaks_the_pipeline_is_a_result(tmp_path):
    def runner(inputs, run_cfg, run_dir, stages):
        if run_cfg.get_path("qa.inject.kind") != "none":
            raise RuntimeError("SfM registered 0 frames")
        _fake_export(Path(run_dir))

    result = degrade.run_degradation_bench(tmp_path / "v.mp4", tmp_path / "b", load_config(), kinds=["low_light"],
                                           severities=["severe"], runner=runner)
    assert "SfM registered 0 frames" in result["rows"][1]["error"]
    assert "failed / failed" in result["table_md"]


# -- single pass ----------------------------------------------------------------------------------------
def _lawnmower(strips=3, length=400.0, spacing=60.0, speed=10.0, hz=2.0):
    """A survey flown as parallel strips joined by short turns; lat/lon around Bengaluru."""
    pts, t = [], 0.0
    for k in range(strips):
        xs = np.arange(0, length, speed / hz)
        xs = xs if k % 2 == 0 else xs[::-1]
        for x in xs:
            pts.append((t, x, k * spacing))
            t += 1 / hz
        for y in np.linspace(k * spacing, (k + 1) * spacing, 8)[1:-1]:
            pts.append((t, xs[-1], y))
            t += 1 / hz
    t, x, y = np.array(pts).T
    lat0, lon0 = 12.97, 77.59
    return pd.DataFrame({"t": t, "lat": lat0 + np.rad2deg(y / 6378137.0),
                         "lon": lon0 + np.rad2deg(x / (6378137.0 * np.cos(np.deg2rad(lat0)))),
                         "alt_gps": 900.0, "hfov_deg": 81.0})


def test_find_strips_on_a_survey_and_on_a_single_line():
    from src.qa.singlepass import find_strips

    scfg = load_config().get_path("qa.single_pass")
    found = find_strips(_lawnmower(), scfg)
    assert len(found["strips"]) == 3 and found["is_multi_strip"]
    assert all(s["length_m"] > 350 for s in found["strips"])
    assert all(abs(s["heading_deg"] % 180 - 90) < 1 for s in found["strips"])
    assert found["side_by_side"][0][2] == pytest.approx(60.0, abs=2)
    line = find_strips(_lawnmower(strips=1), scfg)
    assert len(line["strips"]) == 1 and not line["is_multi_strip"]


def test_strip_input_is_on_the_clip_clock_and_reads_back(tmp_path):
    from src.ingest.telemetry import parse_flight_csv
    from src.qa.singlepass import make_strip_input

    flight = make_flight_video(tmp_path / "v.mp4", frames=90, fps=30.0)
    tel = _lawnmower(strips=1, hz=2.0)
    clip, csv = make_strip_input(flight.video_path, tel, 0.7, 2.2, tmp_path / "strip")
    cap = cv2.VideoCapture(str(clip))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == pytest.approx(46, abs=1)
    cap.release()
    table = parse_flight_csv(csv, load_config().get_path("ingest.telemetry.csv_column_map"))
    f = table.frame
    assert f["t"].iloc[0] == 0.0 and f["t"].iloc[1] == pytest.approx(0.3)     # first KLV row after the cut: 1.0 s
    assert f["lon"].iloc[0] == pytest.approx(np.interp(0.7, tel["t"], tel["lon"]))
    assert f["hfov_deg"].iloc[0] == 81.0 and f["alt_gps"].iloc[0] == 900.0


def test_single_pass_compares_one_strip_with_all_strips(tmp_path):
    from src.qa.singlepass import run_single_pass

    cfg = load_config()
    full = tmp_path / "all"
    _fake_export(full)
    (full / "ingest").mkdir()
    _lawnmower().to_parquet(full / "ingest" / "telemetry.parquet")
    seen = {}

    def runner(inputs, run_cfg, run_dir, stages):
        seen.update(video=inputs.video, telemetry=inputs.telemetry, stages=stages,
                    datum=run_cfg.get_path("geo.vertical.gps_altitude_datum"))
        _fake_export(Path(run_dir), 0.3)

    flight = make_flight_video(tmp_path / "v.mp4", frames=30)
    import src.qa.singlepass as sp

    real = sp.make_strip_input
    sp.make_strip_input = lambda video, tel, t0, t1, out: real(video, tel, 0.0, 0.5, out)
    try:
        result = run_single_pass(flight.video_path, tmp_path / "sp", cfg, full_run=full, runner=runner)
    finally:
        sp.make_strip_input = real
    assert seen["datum"] == "orthometric" and seen["video"].name == "strip.mp4" and "preflight" not in seen["stages"]
    assert result["is_multi_strip"] and result["strip"]["index"] in (0, 1, 2)
    z1 = result["accuracy"]["points_vs_reference_dsm"]["zone1_measured"]
    assert z1["mean_m"] == pytest.approx(0.3, abs=0.02)
    assert (tmp_path / "sp" / "single_pass.json").is_file() and "Strip" in result["summary_html"]
