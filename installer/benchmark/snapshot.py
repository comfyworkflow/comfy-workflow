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

import argparse
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import psutil
import pynvml

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


class SnapshotCollector:
    """Thread-based hardware sampler.

    Runs a polling thread that captures NVML + psutil samples at a fixed
    interval. Designed to run in parallel with a ComfyUI workflow execution.
    Single-instance, single-target — do NOT share an instance across threads.

    Lifecycle: ``__init__`` → :meth:`start` → ``...workload runs...`` →
    :meth:`stop` → :meth:`aggregate`. Calling :meth:`aggregate` before
    :meth:`stop`, or :meth:`start` twice, raises :class:`SnapshotStateError`.

    The polling thread is a daemon thread, so an unhandled exit of the main
    thread is not blocked by an in-flight collector.

    Internal note on locking: ``list.append`` on ``_samples`` is atomic in
    CPython, and :meth:`aggregate` is only invoked after the polling thread
    has been joined by :meth:`stop`, so no explicit lock is required.
    """

    def __init__(self, device_index: int = 0, poll_interval_ms: int = 100) -> None:
        """Initialize the collector.

        Args:
            device_index: NVML device index to sample. Default 0 (first GPU).
            poll_interval_ms: Sampling period in milliseconds. Must be > 0.
                Default 100 ms (10 samples/sec).

        Raises:
            ValueError: ``poll_interval_ms`` is not positive.
        """
        if poll_interval_ms <= 0:
            raise ValueError(
                f"poll_interval_ms must be positive, got {poll_interval_ms}"
            )
        self._device_index: int = device_index
        self._poll_interval_s: float = poll_interval_ms / 1000.0
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._samples: list[Sample] = []
        self._errors: list[str] = []
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._nvml_handle: object | None = None

    def _init_nvml(self) -> object:
        """Initialize NVML and return the device handle.

        Raises:
            NVMLInitError: NVML init failed, or the device handle could not
                be acquired for ``self._device_index``.
        """
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # noqa: BLE001 — wrap arbitrary NVML failures
            raise NVMLInitError(f"NVML init failed: {exc}") from exc
        try:
            return pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception as exc:  # noqa: BLE001 — wrap arbitrary NVML failures
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
            raise NVMLInitError(
                f"NVML device handle acquisition failed for "
                f"index {self._device_index}: {exc}"
            ) from exc

    def _shutdown_nvml(self) -> None:
        """Shutdown NVML, suppressing any errors (best-effort cleanup)."""
        try:
            pynvml.nvmlShutdown()
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            logger.debug("Suppressed NVML shutdown error: %r", exc)

    def _collect_one(self) -> Sample:
        """Capture a single sample of GPU + RAM metrics.

        Each NVML / psutil call is wrapped in :func:`_safe_nvml` so that
        an individual read failure does not abort the polling loop. Failed
        readings default to zero and an entry is appended to ``self._errors``.

        Unit conversions:
            - VRAM bytes → MiB (``// 1024**2``); field name uses ``_mb`` by
              convention but the unit is binary MiB to align with NVML output.
            - RAM bytes → GiB (``/ 1024**3``); same MB/GB-vs-MiB/GiB caveat.
            - Power milliwatts → Watts (``/ 1000``).
        """
        handle = self._nvml_handle
        timestamp = time.monotonic()

        vram_used_bytes = _safe_nvml(
            lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used,
            self._errors,
            "nvmlDeviceGetMemoryInfo failed",
        )
        util_pct = _safe_nvml(
            lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            self._errors,
            "nvmlDeviceGetUtilizationRates failed",
        )
        temp_c = _safe_nvml(
            lambda: pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ),
            self._errors,
            "nvmlDeviceGetTemperature failed",
        )
        power_mw = _safe_nvml(
            lambda: pynvml.nvmlDeviceGetPowerUsage(handle),
            self._errors,
            "nvmlDeviceGetPowerUsage failed",
        )
        ram_used_bytes = _safe_nvml(
            lambda: psutil.virtual_memory().used,
            self._errors,
            "psutil.virtual_memory failed",
        )

        return Sample(
            timestamp_monotonic=timestamp,
            vram_used_mb=(
                int(vram_used_bytes // (1024**2)) if vram_used_bytes is not None else 0
            ),
            ram_used_gb=(
                ram_used_bytes / (1024**3) if ram_used_bytes is not None else 0.0
            ),
            gpu_utilization_pct=int(util_pct) if util_pct is not None else 0,
            gpu_temp_c=int(temp_c) if temp_c is not None else 0,
            gpu_power_w=(power_mw / 1000.0) if power_mw is not None else 0.0,
        )

    def start(self) -> None:
        """Start the polling thread.

        Initializes NVML, records the start timestamp, and spawns a daemon
        polling thread that calls :meth:`_collect_one` once per
        ``poll_interval``.

        Raises:
            SnapshotStateError: ``start`` was already called on this
                instance.
            NVMLInitError: NVML init failed; propagated from
                :meth:`_init_nvml`.
        """
        if self._thread is not None:
            raise SnapshotStateError("start() already called on this collector")
        self._nvml_handle = self._init_nvml()
        self._started_at = time.monotonic()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="SnapshotPoller",
        )
        self._thread.start()

    def _poll_loop(self) -> None:
        """Polling thread body. Runs until ``self._stop_event`` is set.

        Uses ``Event.wait(timeout)`` instead of ``time.sleep`` so the loop
        responds to a stop signal immediately, without waiting for the
        remainder of the current poll interval.
        """
        assert self._stop_event is not None  # invariant: set by start()
        while not self._stop_event.is_set():
            sample = self._collect_one()
            self._samples.append(sample)
            self._stop_event.wait(self._poll_interval_s)

    def stop(self) -> None:
        """Signal the polling thread to stop, join it, and shut down NVML.

        After ``stop`` returns, :meth:`aggregate` may be called to retrieve
        the :class:`SnapshotResult`.

        Raises:
            SnapshotStateError: ``start`` was not called, or ``stop`` was
                already called on this collector.
        """
        if self._thread is None:
            raise SnapshotStateError("stop() called before start()")
        if self._stopped_at is not None:
            raise SnapshotStateError("stop() already called on this collector")
        assert self._stop_event is not None  # invariant: set by start()
        self._stop_event.set()
        self._thread.join(timeout=2 * self._poll_interval_s + 1.0)
        self._stopped_at = time.monotonic()
        self._shutdown_nvml()

    def aggregate(self) -> SnapshotResult:
        """Compute peak / average statistics over the collected samples.

        Must be called after :meth:`stop`. Calling earlier raises
        :class:`SnapshotStateError`. If no samples were collected (e.g.
        ``stop`` was called before the first interval elapsed), the result
        contains zeros and an entry "no samples collected" is added to
        ``errors_during_collection``.

        Returns:
            A frozen :class:`SnapshotResult` summarizing the collection
            window. ``errors_during_collection`` is a defensive copy of
            the collector's internal list.

        Raises:
            SnapshotStateError: :meth:`stop` has not yet been called.
        """
        if self._stopped_at is None:
            raise SnapshotStateError("aggregate() called before stop()")

        started_at = self._started_at if self._started_at is not None else self._stopped_at
        duration_seconds = self._stopped_at - started_at
        errors = list(self._errors)

        if not self._samples:
            errors.append("no samples collected")
            return SnapshotResult(
                peak_vram_mb=0,
                peak_ram_gb=0.0,
                gpu_avg_utilization_pct=0.0,
                gpu_avg_temp_c=0.0,
                gpu_avg_power_w=0.0,
                samples_collected=0,
                duration_seconds=duration_seconds,
                errors_during_collection=errors,
            )

        n = len(self._samples)
        return SnapshotResult(
            peak_vram_mb=max(s.vram_used_mb for s in self._samples),
            peak_ram_gb=max(s.ram_used_gb for s in self._samples),
            gpu_avg_utilization_pct=sum(s.gpu_utilization_pct for s in self._samples) / n,
            gpu_avg_temp_c=sum(s.gpu_temp_c for s in self._samples) / n,
            gpu_avg_power_w=sum(s.gpu_power_w for s in self._samples) / n,
            samples_collected=n,
            duration_seconds=duration_seconds,
            errors_during_collection=errors,
        )


def main() -> None:
    """Self-test CLI for :class:`SnapshotCollector`.

    Spawns a collector, samples for ``--duration`` seconds, then prints the
    aggregated :class:`SnapshotResult` fields. Runs locally on the machine
    where snapshot.py is invoked — for a CG executor, this would be via SSH
    dispatch by ``runner.py`` (future block); for the Bloco 12 self-test,
    on the Itapoá's local RTX 3060.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Self-test for SnapshotCollector: samples GPU + RAM for the "
            "given duration and prints aggregated metrics."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Collection duration in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="NVML device index (default: 0).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Poll interval in milliseconds (default: 100).",
    )
    args = parser.parse_args()
    duration = cast(float, args.duration)
    device = cast(int, args.device)
    interval = cast(int, args.interval)

    print(f"device index: {device}")
    print(f"poll interval: {interval} ms")

    collector = SnapshotCollector(device_index=device, poll_interval_ms=interval)
    collector.start()
    print(f"collecting for {duration:.1f}s...")
    time.sleep(duration)
    collector.stop()

    result = collector.aggregate()
    print(f"samples_collected: {result.samples_collected}")
    print(f"duration_seconds: {result.duration_seconds:.2f}")
    print(f"peak_vram_mb: {result.peak_vram_mb}")
    print(f"peak_ram_gb: {result.peak_ram_gb:.2f}")
    print(f"gpu_avg_utilization_pct: {result.gpu_avg_utilization_pct:.1f}")
    print(f"gpu_avg_temp_c: {result.gpu_avg_temp_c:.1f}")
    print(f"gpu_avg_power_w: {result.gpu_avg_power_w:.1f}")
    if result.errors_during_collection:
        print(f"errors_during_collection: {len(result.errors_during_collection)}")
        for err in result.errors_during_collection:
            print(f"  - {err}")
    else:
        print("errors_during_collection: 0")


if __name__ == "__main__":
    main()
