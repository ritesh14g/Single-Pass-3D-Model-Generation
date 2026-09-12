"""Device selection and GPU memory sizing.

Track B chunk size is not guessed. Spec §7.2 carries the official
VGGT-Omega-1B-512 peak-memory measurements (single GPU, 624x416 input), and
:func:`chunk_frames_for_memory` interpolates that curve to find the largest
chunk that fits the available card with headroom left for the stages running
alongside it. Profiling on the real instance should replace the table, not the
method — so the table lives in one place, annotated.
"""

from __future__ import annotations

import logging
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


def resolve_device(prefer: str = "auto") -> str:
    """Return the torch device string, logging a downgrade if CUDA is absent."""
    prefer = (prefer or "auto").lower()
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
    except ImportError:
        info["torch"] = None
    return info


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
