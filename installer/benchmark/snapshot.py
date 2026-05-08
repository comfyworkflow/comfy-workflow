"""Hardware metrics sampler for ComfyUI workflow execution.

Single responsibility: sample GPU (NVML) and RAM (psutil) at a fixed
interval in a background thread, then aggregate into peak / average
statistics. This module does NOT orchestrate benchmarks (runner.py),
talk to the ComfyUI server (interface.py), or download models
(installer.py).

Architecture note: this module imports ``pynvml``, which requires NVIDIA
drivers and a local GPU. It must run on the machine whose GPU is being
sampled. The benchmark coordinator (Itapoá) has no NVML visibility into
remote GPUs; ``runner.py`` (future block) dispatches snapshot.py via SSH
so it executes inside each executor (cg-3060 / cg-4090 / cg-5090). The
Bloco 12 self-test runs locally on Itapoá's RTX 3060.

Metrics covered (per DA-008)::

    peak_vram_mb               NVML
    peak_ram_gb                psutil
    gpu_avg_utilization_pct    NVML
    gpu_avg_temp_c             NVML
    gpu_avg_power_w            NVML

Metrics NOT covered (handled elsewhere)::

    wallclock_seconds          measured by runner.py around
                               interface.queue_prompt + poll_history
    sampler_steps_per_second   derived from ComfyUI progress events
                               (future block in interface.py / runner.py)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SnapshotError(Exception):
    """Base class for all snapshot module errors."""


class NVMLInitError(SnapshotError):
    """Raised when NVML initialization or device handle acquisition fails."""


class SnapshotStateError(SnapshotError):
    """Raised on illegal lifecycle transitions.

    Examples: ``start()`` called twice, ``stop()`` called without a prior
    ``start()``, ``aggregate()`` called before ``stop()``.
    """


def _safe_nvml[T](
    fn: Callable[[], T],
    errors: list[str],
    err_msg: str,
) -> T | None:
    """Call ``fn`` and append a contextualized error to ``errors`` on failure.

    Parallel to :func:`installer.benchmark.interface._safe_request`. Wrap
    NVML calls so the failure of an individual reading does not poison the
    entire sampling loop. Returns the value of ``fn`` on success, ``None``
    if any exception was raised.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for fault tolerance
        errors.append(f"{err_msg}: {exc}")
        logger.debug("Suppressed exception in _safe_nvml (%s): %r", err_msg, exc)
        return None


@dataclass(frozen=True, slots=True)
class Sample:
    """Single point-in-time hardware reading.

    Attributes:
        timestamp_monotonic: ``time.monotonic()`` value when the sample was
            captured. Suitable for measuring intervals; not wall-clock time.
        vram_used_mb: VRAM bytes in use, converted to MiB
            (``bytes // 1024**2``). Field name uses MB by convention but the
            unit is binary MiB to align with NVML output.
        ram_used_gb: System RAM bytes in use, converted to GiB
            (``bytes / 1024**3``). Same MB/MiB caveat.
        gpu_utilization_pct: GPU SM utilization, integer percent (0-100).
        gpu_temp_c: GPU temperature in degrees Celsius (integer).
        gpu_power_w: GPU instantaneous power draw in Watts. NVML reports
            milliwatts; this field is already converted.
    """

    timestamp_monotonic: float
    vram_used_mb: int
    ram_used_gb: float
    gpu_utilization_pct: int
    gpu_temp_c: int
    gpu_power_w: float


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Aggregated hardware metrics over a collection window.

    Returned by :meth:`SnapshotCollector.aggregate`. The 5 metric fields
    correspond to the 5 metrics covered by snapshot.py per DA-008
    (peak VRAM, peak RAM, average GPU utilization / temperature / power).
    ``errors_during_collection`` aggregates per-sample NVML failures
    captured by :func:`_safe_nvml`; an empty list indicates a clean run.
    """

    peak_vram_mb: int
    peak_ram_gb: float
    gpu_avg_utilization_pct: float
    gpu_avg_temp_c: float
    gpu_avg_power_w: float
    samples_collected: int
    duration_seconds: float
    errors_during_collection: list[str]
