"""One pipeline process per run folder (src/core/runlock.py)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from src.core.runlock import LOCK_NAME, RunLocked, holder, run_lock


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_lock_is_taken_and_released(tmp_path):
    with run_lock(tmp_path) as path:
        assert path.exists() and holder(tmp_path)["pid"] > 0
    assert not (tmp_path / LOCK_NAME).exists() and holder(tmp_path) is None


def test_second_process_in_the_same_folder_is_refused(tmp_path):
    # The box, 2026-09-23: a second DJI_0047 run deleted database.db under the first one.
    with run_lock(tmp_path):
        with pytest.raises(RunLocked, match="in use by pid"):
            with run_lock(tmp_path):
                pass
        assert (tmp_path / LOCK_NAME).exists()      # the refused one leaves the holder's lock alone


def test_lock_of_a_finished_process_is_taken_over(tmp_path):
    (tmp_path / LOCK_NAME).write_text(json.dumps({"pid": _dead_pid(), "started": 0}))
    assert holder(tmp_path) is None
    with run_lock(tmp_path):
        pass


def test_shell_guard_exit_code(tmp_path):
    guard = [sys.executable, "-m", "src.core.runlock", str(tmp_path)]
    assert subprocess.run(guard).returncode == 0
    with run_lock(tmp_path):
        assert subprocess.run(guard, capture_output=True).returncode == 1
