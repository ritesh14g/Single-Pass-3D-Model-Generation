"""S5-1: the telemetry altitude's datum is measured against terrain, and above-takeoff
"altitude" is made absolute (src/geo/vertical.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.core.config import load_config
from src.geo import vertical
from src.geo.crs import gps_altitude_datum
from src.ingest.telemetry import TelemetryTable, resolve_csv_columns

# DJI_0047 (AirData CSV): home -19.91 m, terrain 7 m, EGM96 N -26.3 m at the home point.
HOME = {"lat": 27.223305, "lon": -81.882038, "alt": -19.91}
N = -26.32


def _dem(tmp_path, height=7.0):
    path = tmp_path / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(-81.9, 27.25, 0.005, 0.005)) as ds:
        ds.write(np.full((1, 20, 20), height, np.float32))
    return path


def _cfg(tmp_path, **extra):
    over = [f"geo.vertical.datum_check.dem_file={_dem(tmp_path)}", "geo.vertical.datum_check.online=false"]
    return load_config(overrides=over + [f"{k}={v}" for k, v in extra.items()])


def _table(alt, rel, home=None):
    n = len(rel)
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "lat": np.full(n, 27.2218), "lon": np.full(n, -81.8769),
                          "alt_gps": alt, "alt_baro": rel})
    return TelemetryTable.from_records(frame.to_dict("records"), source="csv:test") if home is None else \
        _with_home(TelemetryTable.from_records(frame.to_dict("records"), source="csv:test"), home)


def _with_home(table, home):
    table.home = dict(home)
    return table


@pytest.mark.parametrize("ground, terrain, n, expected", [
    (-19.91, 7.0, N, "ellipsoidal"),        # DJI_0047
    (7.5, 7.0, N, "orthometric"),
    (920.0, 918.0, -86.4, "orthometric"),   # Bengaluru, MSL log
    (832.0, 918.0, -86.4, "ellipsoidal"),
    (3.0, 7.0, -4.0, None),                 # |N| too small to tell apart
    (60.0, 7.0, N, None),                   # neither fits
])
def test_decide_datum(ground, terrain, n, expected):
    datum, detail = vertical.decide_datum(ground, terrain, n, 8.0)
    assert datum == expected, detail


def test_above_takeoff_altitude_gets_the_home_height_and_the_datum_is_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(vertical, "geoid_undulation", lambda *a, **k: N)
    rel = np.full(10, 57.0)
    table = _table(rel.copy(), rel, HOME)          # "altitude" == height above takeoff
    record = vertical.resolve_altitude(table, _cfg(tmp_path))
    assert "home altitude" in record["relative_fixed"]
    assert np.allclose(table.frame["alt_gps"], -19.91 + 57.0)
    assert record["datum"] == "ellipsoidal" and record["method"] == "terrain check"


def test_above_takeoff_altitude_without_home_uses_the_terrain(tmp_path):
    rel = np.full(10, 57.0)
    table = _table(np.full(10, np.nan), rel)
    record = vertical.resolve_altitude(table, _cfg(tmp_path))
    assert record["datum"] == "orthometric" and record["method"] == "terrain at takeoff"
    assert np.allclose(table.frame["alt_gps"], 7.0 + 57.0)


def test_absolute_minus_relative_gives_the_takeoff_ground(tmp_path, monkeypatch):
    monkeypatch.setattr(vertical, "geoid_undulation", lambda *a, **k: N)
    rel = np.linspace(50, 60, 10)
    table = _table(7.0 + rel, rel)                 # an MSL log (SRT abs_alt + rel_alt)
    record = vertical.resolve_altitude(table, _cfg(tmp_path))
    assert record["datum"] == "orthometric" and record["ground_alt_m"] == pytest.approx(7.0)
    assert np.allclose(table.frame["alt_gps"], 7.0 + rel)         # untouched


def test_offline_keeps_the_assumption_and_says_why(tmp_path):
    rel = np.full(10, 57.0)
    table = _table(rel.copy() + 30, rel)
    cfg = load_config(overrides=["geo.vertical.datum_check.online=false"])
    record = vertical.resolve_altitude(table, cfg)
    assert record["datum"] is None and record["method"] == "assumed" and "no terrain model" in record["detail"]


def test_measured_datum_wins_over_the_assumption_but_not_over_config():
    assert gps_altitude_datum("csv:x", "auto", "orthometric") == ("orthometric", False)
    assert gps_altitude_datum("csv:x", "auto", None) == ("ellipsoidal", True)
    assert gps_altitude_datum("klv", "auto", None) == ("orthometric", False)
    assert gps_altitude_datum("csv:x", "ellipsoidal", "orthometric") == ("ellipsoidal", False)


def test_home_columns_match_exactly_never_the_go_home_height():
    cfg = load_config()
    cols = ["OSD.latitude", "OSD.longitude", "OSD.height [m]", "OSD.altitude [m]", "HOME.goHomeHeight [m]",
            "HOME.latitude", "HOME.longitude"]
    resolved = resolve_csv_columns(cols, cfg.get_path("ingest.telemetry.csv_column_map"))
    assert "home_alt" not in resolved and resolved["home_lat"] == "HOME.latitude"
    resolved = resolve_csv_columns(cols + ["HOME.height [m]"], cfg.get_path("ingest.telemetry.csv_column_map"))
    assert resolved["home_alt"] == "HOME.height [m]"
