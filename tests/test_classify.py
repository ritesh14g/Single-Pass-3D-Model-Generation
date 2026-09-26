"""S5-3: ground / above-ground / low-noise classes for LAS (src/export/classify.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import load_config
from src.export.classify import GROUND, LOW_NOISE, UNCLASSIFIED, ground_classes


def _scene(terrain_slope=0.05, water_noise=False, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 200, (2, 300000))
    ground = terrain_slope * x + 0.5 * np.sin(y / 15)
    z = ground + rng.normal(0, 0.05, len(x))
    building = (x > 80) & (x < 100) & (y > 80) & (y < 110)
    z[building] = ground[building] + 8.0
    tree = (x - 150) ** 2 + (y - 50) ** 2 < 64
    z[tree] = ground[tree] + rng.uniform(2, 10, tree.sum())
    noise = rng.random(len(x)) < 0.002
    z[noise] = ground[noise] - 5
    if water_noise:                       # a patch of reflections tens of metres under a river
        river = (y > 150) & (y < 170) & (rng.random(len(x)) < 0.05)
        z[river] = ground[river] - 30
        noise |= river
    return np.c_[x, y, z], ~building & ~tree & ~noise, building, tree, noise


@pytest.fixture(scope="module")
def ccfg():
    return load_config().get_path("export.las.classify")


def test_terrain_is_ground_objects_are_not_and_noise_is_found(ccfg):
    xyz, is_ground, building, tree, noise = _scene()
    classes, info = ground_classes(xyz, ccfg)
    assert (classes[is_ground] == GROUND).mean() > 0.99
    assert (classes[building] == GROUND).mean() == 0.0
    assert (classes[tree] == GROUND).mean() < 0.03                   # measured 1.3%: canopy-edge cells
    assert (classes[noise] == LOW_NOISE).mean() > 0.95
    assert info["counts"]["ground"] + info["counts"]["above_ground"] + info["counts"]["low_noise"] == len(xyz)


def test_clustered_water_noise_does_not_sink_the_terrain(ccfg):
    # Esri: 1% of the cloud lay > 36 m under ~10 m terrain; with the minimum per cell the whole
    # model classed as above ground (16.6% ground, river included).
    xyz, is_ground, _, _, noise = _scene(water_noise=True)
    classes, _ = ground_classes(xyz, ccfg)
    assert (classes[is_ground] == GROUND).mean() > 0.98
    assert (classes[noise] == LOW_NOISE).mean() > 0.9


def test_steeper_terrain_stays_ground(ccfg):
    xyz, is_ground, building, _, _ = _scene(terrain_slope=0.15)
    classes, _ = ground_classes(xyz, ccfg)
    assert (classes[is_ground] == GROUND).mean() > 0.99 and (classes[building] == GROUND).mean() == 0.0


def test_tiny_cloud_is_left_unclassified(ccfg):
    classes, info = ground_classes(np.zeros((5, 3)), ccfg)
    assert (classes == UNCLASSIFIED).all() and info["method"] == "too few points"
