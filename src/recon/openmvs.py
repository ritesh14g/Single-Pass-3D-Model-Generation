"""OpenMVS command-line tools (InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, TextureMesh).

The prebuilt 2.4.0 binaries live in ``tools/openmvs`` (git-ignored, fetched per machine):
the Linux build is static and CPU-only, the Windows build CPU-only too (DEVLOG Stage 4).
OpenMVS writes its log to a file in the working folder rather than the console, and
resolves relative ``-i``/``-o`` against ``-w``, so every path passed here is absolute.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN_DIRS = (ROOT / "tools" / "openmvs" / "bin", ROOT / "tools" / "openmvs" / "vc17" / "x64" / "Release")


class OpenMvsError(RuntimeError):
    """An OpenMVS tool exited non-zero; the message carries the tail of its log."""


def _exe(bin_dir: Path, tool: str) -> Path:
    return bin_dir / (tool + (".exe" if platform.system() == "Windows" else ""))


def find_bin_dir(configured: str | None) -> Path | None:
    """The folder holding the OpenMVS tools, or ``None`` when they are not installed."""
    candidates = DEFAULT_BIN_DIRS if configured in (None, "", "auto") else (Path(configured),)
    for candidate in candidates:
        if _exe(candidate, "TextureMesh").exists():
            return candidate.resolve()
    return None


def run_tool(bin_dir: Path, tool: str, work: Path, *args: str, threads: int) -> float:
    """Run one tool; returns its wall time, raises :class:`OpenMvsError` on failure."""
    work.mkdir(parents=True, exist_ok=True)
    cmd = [str(_exe(bin_dir, tool)), "-w", str(work.resolve()), "--max-threads", str(threads), *args]
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        logs = sorted(work.glob(f"{tool}-*.log"))
        tail = logs[-1].read_text(errors="ignore").splitlines()[-12:] if logs else [result.stderr]
        raise OpenMvsError(f"{tool} exited {result.returncode}: " + " | ".join(line.strip() for line in tail))
    return elapsed
