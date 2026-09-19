"""Device selection and GPU memory sizing.

Track B chunk size is not guessed. Spec §7.2 carries the official
VGGT-Omega-1B-512 peak-memory measurements (single GPU, 624x416 input), and
:func:`chunk_frames_for_memory` interpolates that curve to find the largest
chunk that fits the available card with headroom left for the stages running
alongside it. Profiling on the real instance should replace the table, not the
method — so the table lives in one place, annotated.

The policy everywhere is **GPU first, CPU fallback, never a crash**: every
consumer asks :func:`resolve_device` and gets ``"cuda"`` or ``"cpu"``, with a
logged downgrade when it asked for the GPU and didn't get it. The team's
target box is a 20 GB MIG slice of an H100 with only 3 CPU cores
(``CLOUD_GPU_GUIDE.md`` §2), so :func:`configure_runtime` also caps CPU threads
to the container's real quota — a notebook container often *sees* every core
on the host and oversubscribes the few it is allowed to use.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

from src.core.logging import get_logger, log_downgrade, log_event

log = get_logger(__name__)

# Frames -> peak GPU memory in GB, from the official VGGT-Omega-1B-512 benchmark
# reproduced in build spec §7.2.
VGGT_OMEGA_PEAK_MEMORY_GB: list[tuple[int, float]] = [
    (1, 6.0),
    (10, 6.7),
    (25, 7.8),
    (50, 9.7),
    (100, 13.4),
    (200, 20.8),
    (300, 28.3),
    (400, 35.7),
    (500, 43.2),
]


# The answer cannot change within a process, and every stage asks; log the
# downgrade once rather than on every call.
_RESOLVED: dict[str, str] = {}


def resolve_device(prefer: str = "auto") -> str:
    """Return the torch device string, logging a downgrade if CUDA is absent."""
    prefer = (prefer or "auto").lower()
    if prefer not in _RESOLVED:
        _RESOLVED[prefer] = _resolve_device(prefer)
    return _RESOLVED[prefer]


def _resolve_device(prefer: str) -> str:
    try:
        import torch
    except ImportError:
        if prefer == "cuda":
            log_downgrade(log, "cuda", "cpu", "torch is not installed")
        return "cpu"

    cuda_available = torch.cuda.is_available()
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda" and not cuda_available:
        log_downgrade(log, "cuda", "cpu", "torch.cuda.is_available() is False")
        return "cpu"
    if not cuda_available:
        log_downgrade(log, "cuda", "cpu", "no CUDA device visible; Track B will be slow")
        return "cpu"
    return "cuda"


def device_info(prefer: str = "auto") -> dict[str, Any]:
    """Describe the compute device for the manifest and the QA report."""
    device = resolve_device(prefer)
    info: dict[str, Any] = {"device": device}
    try:
        import torch

        info["torch"] = torch.__version__
        if device == "cuda":
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
            info["cuda"] = torch.version.cuda
            # A MIG slice reports its own memory, but shares the card's
            # decoders and PCIe link with other tenants — worth recording.
            info["mig"] = "MIG" in str(props.name)
    except ImportError:
        info["torch"] = None
    return info


def use_half_precision(cfg: Any) -> bool:
    """FP16 inference only makes sense on the GPU."""
    return resolve_device(cfg.get_path("device.prefer", "auto")) == "cuda" and bool(
        cfg.get_path("device.half_precision", True)
    )


# -- CPU threads ------------------------------------------------------------
def cgroup_cpu_quota(root: Path | str = "/sys/fs/cgroup") -> float | None:
    """CPU limit imposed by the container, in cores, or ``None`` if unlimited.

    Reads cgroup v2 ``cpu.max`` first, then v1 ``cpu.cfs_quota_us``. Neither
    exists on Windows or macOS, which is the correct "no limit" answer there.
    """
    root = Path(root)
    try:
        v2 = root / "cpu.max"
        if v2.is_file():
            quota, period = v2.read_text().split()[:2]
            if quota != "max" and int(period) > 0:
                return int(quota) / int(period)
            return None
        quota_file, period_file = root / "cpu" / "cpu.cfs_quota_us", root / "cpu" / "cpu.cfs_period_us"
        if quota_file.is_file() and period_file.is_file():
            quota, period = int(quota_file.read_text()), int(period_file.read_text())
            if quota > 0 and period > 0:
                return quota / period
    except (OSError, ValueError):
        pass
    return None


def cpu_thread_budget(root: Path | str = "/sys/fs/cgroup") -> int:
    """Cores this process may actually use: affinity mask, capped by the cgroup quota."""
    try:
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # Windows / macOS
        cores = os.cpu_count() or 1
    quota = cgroup_cpu_quota(root)
    if quota:
        cores = min(cores, max(int(math.floor(quota)), 1))
    return max(cores, 1)


def configure_runtime(cfg: Any) -> dict[str, Any]:
    """Apply process-wide compute settings once, at the start of a run.

    Caps OpenCV's and torch's thread pools at ``device.cpu_threads`` (``auto``
    = the container's real quota). Returns what was applied, for the manifest.
    """
    import cv2

    configured = cfg.get_path("device.cpu_threads", "auto")
    if configured in (None, "auto"):
        threads, source = cpu_thread_budget(), "auto"
    else:
        threads, source = max(int(configured), 1), "config"
    cv2.setNumThreads(threads)
    # Only touch torch if something already imported it or it is cheap to get;
    # torch.set_num_threads is harmless on a CPU-only build.
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(threads)
    else:
        try:
            import torch

            torch.set_num_threads(threads)
        except ImportError:
            pass
    log_event(log, logging.INFO, f"compute threads capped at {threads}", source=source,
              quota_cores=cgroup_cpu_quota())
    return {"cpu_threads": threads, "cpu_threads_source": source}


def available_gpu_memory_gb(configured: float, prefer: str = "auto") -> float:
    """Memory budget to size chunks against.

    Trust the real card when one is visible; fall back to the configured value
    (the rented instance's spec) when planning on a machine without a GPU.
    """
    info = device_info(prefer)
    detected = info.get("gpu_memory_gb")
    if detected:
        if abs(detected - configured) > 2.0:
            log_event(
                log,
                logging.INFO,
                "GPU memory differs from config; using the detected value",
                configured_gb=configured,
                detected_gb=detected,
            )
        return float(detected)
    return float(configured)


def peak_memory_gb(frames: int) -> float:
    """Interpolate the VGGT-Omega peak-memory curve for a frame count."""
    if frames <= VGGT_OMEGA_PEAK_MEMORY_GB[0][0]:
        return VGGT_OMEGA_PEAK_MEMORY_GB[0][1]
    for (f0, m0), (f1, m1) in zip(VGGT_OMEGA_PEAK_MEMORY_GB, VGGT_OMEGA_PEAK_MEMORY_GB[1:]):
        if frames <= f1:
            t = (frames - f0) / (f1 - f0)
            return m0 + t * (m1 - m0)
    # Beyond the measured range the curve is close to linear in frames; keep
    # extrapolating on the last measured slope rather than refusing to answer.
    (f0, m0), (f1, m1) = VGGT_OMEGA_PEAK_MEMORY_GB[-2], VGGT_OMEGA_PEAK_MEMORY_GB[-1]
    slope = (m1 - m0) / (f1 - f0)
    return m1 + (frames - f1) * slope


def chunk_frames_for_memory(
    total_gb: float,
    headroom_gb: float = 4.0,
    requested: int | None = None,
    minimum: int = 8,
) -> int:
    """Largest safe chunk size for the card, capped by ``requested``.

    ``headroom_gb`` is reserved for everything else on the card — conditioning
    batches, a concurrent Track A MVS pass, allocator fragmentation. Spec §7.2
    is explicit that running right up to the measured line is a mistake.
    """
    budget = max(total_gb - headroom_gb, 0.0)
    if budget <= VGGT_OMEGA_PEAK_MEMORY_GB[0][1]:
        # Not even a single frame fits the measured curve; caller will have to
        # fall back to CPU or a smaller model, but never to a chunk of zero.
        safe = minimum
    else:
        safe = minimum
        for frames, _ in VGGT_OMEGA_PEAK_MEMORY_GB:
            if peak_memory_gb(frames) <= budget:
                safe = max(safe, frames)
        # Refine between the last fitting table row and the next one.
        upper = safe * 2
        while peak_memory_gb(upper) <= budget:
            upper *= 2
        lo, hi = safe, upper
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if peak_memory_gb(mid) <= budget:
                lo = mid
            else:
                hi = mid
        safe = lo

    chosen = min(requested, safe) if requested else safe
    chosen = max(chosen, minimum)
    log_event(
        log,
        logging.INFO,
        f"Track B chunk size {chosen} frames",
        gpu_gb=total_gb,
        headroom_gb=headroom_gb,
        safe_max_frames=safe,
        requested=requested,
        projected_peak_gb=round(peak_memory_gb(chosen), 1),
    )
    return chosen
