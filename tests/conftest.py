"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.core.logging import setup_logging  # noqa: E402
from tests import fixtures  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging():
    """Keep test output readable; the JSONL sink is exercised in its own test."""
    setup_logging(run_dir=None, level="ERROR", jsonl=False, console_color=False)


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def flight(tmp_path_factory):
    """A clean synthetic flight with 90% true overlap, shared across tests."""
    directory = tmp_path_factory.mktemp("flight")
    return fixtures.make_flight_video(directory / "flight.mp4", frames=60, overlap=0.9)


@pytest.fixture(scope="session")
def blurry_flight(tmp_path_factory):
    """A flight with a directional blur burst every fifth frame."""
    directory = tmp_path_factory.mktemp("blurry")
    return fixtures.make_flight_video(
        directory / "blurry.mp4", frames=60, overlap=0.9, blur_every=5, blur_length=25, seed=11
    )
