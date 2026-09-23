"""One pipeline process per run folder.

Two processes in one run folder destroy each other's work: on the box (2026-09-23) a second
DJI_0047 run started Track A in the same folder, deleted ``database.db`` under the first one's
mapper, and the first run's GPS-prior pass then found 0 images. ``run_lock`` refuses the second
process with the holder's pid instead. A lock left by a process that is gone is taken over.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_NAME = ".run.lock"


class RunLocked(RuntimeError):
    """Another live process is running in this run folder."""


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if sys.platform == "win32":
        # os.kill(pid, 0) would *terminate* the process on Windows.
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(run_dir: Path) -> dict | None:
    try:
        return json.loads((Path(run_dir) / LOCK_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextmanager
def run_lock(run_dir: Path) -> Iterator[Path]:
    path = Path(run_dir) / LOCK_NAME
    me = {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time(), "argv": sys.argv[:6]}
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = read_lock(run_dir) or {}
            same_host = held.get("host") in (None, me["host"])
            if held and (not same_host or _alive(int(held.get("pid", -1)))):
                since = time.strftime("%H:%M:%S", time.localtime(float(held.get("started", 0))))
                raise RunLocked(
                    f"{run_dir} is in use by pid {held.get('pid')} on {held.get('host')} (since {since}): "
                    f"wait for it to finish (bash scripts/box_status.sh), or if that process is gone, "
                    f"delete {path}") from None
            try:                                   # stale: its process is gone
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(me, fh)
        break
    else:
        raise RunLocked(f"could not take {path}")
    try:
        yield path
    finally:
        if (read_lock(run_dir) or {}).get("pid") == me["pid"]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def holder(run_dir: Path) -> dict | None:
    """The live process holding ``run_dir``, or None when the folder is free."""
    held = read_lock(run_dir)
    if not held:
        return None
    if held.get("host") not in (None, socket.gethostname()) or _alive(int(held.get("pid", -1))):
        return held
    return None


if __name__ == "__main__":
    # Shell guard before deleting a run folder: python -m src.core.runlock <run dir> || exit 1
    busy = [(d, h) for d in sys.argv[1:] if (h := holder(Path(d)))]
    for d, h in busy:
        print(f"!! {d} is in use by pid {h.get('pid')} ({' '.join(h.get('argv') or [])}): not touching it",
              file=sys.stderr)
    sys.exit(1 if busy else 0)
