"""Stage Lab smoke tests — keep every panel loadable as the pipeline grows.

The UI is preserved across stages by policy; these tests are what enforce it.
They drive the real Streamlit script headlessly and fail on any exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.stages import STAGES  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120)


def test_pipeline_page_loads():
    at = _app().run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("stage", STAGES, ids=lambda s: s.key)
def test_every_stage_page_loads(stage):
    at = _app().run()
    label = next(o for o in at.sidebar.radio(key="page").options if f"Stage {stage.number}:" in o)
    at.sidebar.radio(key="page").set_value(label).run()
    assert not at.exception, at.exception


def test_stage2_runs_prerequisite_stages_from_the_ui():
    """Stage 2 owns only ``condition``, which cannot run without ingest output.

    The Lab must run the prerequisite chain (``manifest_stages_through``), so
    clicking Run on the Stage 2 page produces a scorecard instead of raising
    "conditioning needs a completed ingest stage".
    """
    at = _app().run()
    label = next(o for o in at.sidebar.radio(key="page").options if "Stage 2:" in o)
    at.sidebar.radio(key="page").set_value(label).run()
    at.number_input(key="condition_syn_frames").set_value(40).run()
    at.button(key="condition_run").click().run()
    assert not at.exception, at.exception
    assert any("Stage score" in m.label for m in at.metric)


def test_stage1_runs_end_to_end_from_the_ui():
    at = _app().run()
    label = next(o for o in at.sidebar.radio(key="page").options if "Stage 1:" in o)
    at.sidebar.radio(key="page").set_value(label).run()
    at.number_input(key="ingest_syn_frames").set_value(60).run()
    at.button(key="ingest_run").click().run()
    assert not at.exception, at.exception
    assert any("Stage score" in m.label for m in at.metric)
