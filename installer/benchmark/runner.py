"""Benchmark orchestrator with lifecycle-aligned snapshot dispatch.

Single responsibility: run a workflow on a remote ComfyUI executor while
``snapshot.py`` samples hardware metrics on the executor's GPU. Unlike
``dry_run.py``, the snapshot lifecycle is *aligned* with the workflow:
snapshot starts immediately before ``queue_prompt`` and stops immediately
after ``poll_history`` returns, eliminating the idle window that biases
average metrics. See V2 débito #7 in ``internal_docs/notas_de_execucao.md``.

Bloco 15 ships a minimal V1 with one run per invocation. Bloco 16+ will
extend with the DA-008 mechanic (5 runs cold + 4 warm, discard min/max,
report mean/stddev/p50). The dataclass shapes here already accommodate
that growth: :class:`RunnerSummary.runs` is a ``list`` (length 1 in V1,
length N in V2+) and :attr:`RunnerSummary.aggregated` is reserved for
future statistics (``None`` in V1).

This module does NOT touch NVML directly (snapshot.py does, on the
executor) and does not orchestrate ComfyUI lifecycle (preserved by humans
via RDP per DA-013). It dispatches snapshot.py via SSH and signals stop
via a file flag on the executor (default
``%USERPROFILE%\\runner_stop.flag``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class RunnerError(Exception):
    """Base class for all runner module errors."""


class RunnerSSHError(RunnerError):
    """Raised when an SSH command fails (non-zero exit, timeout, etc.)."""


class RunnerWorkflowError(RunnerError):
    """Raised when a workflow execution fails.

    Examples: queue rejection by ComfyUI, missing outputs in the history
    entry, malformed workflow JSON.
    """


@dataclass(frozen=True, slots=True)
class RunResult:
    """Output of a single benchmark run.

    Attributes:
        prompt_id: UUID returned by ComfyUI's ``/prompt`` endpoint.
        wallclock_seconds: End-to-end wall time from ``queue_prompt`` to
            ``poll_history`` return, measured with :func:`time.monotonic`.
        seed: KSampler seed injected for this run.
        history_entry: Raw ``/history/<prompt_id>`` entry returned by the
            server, kept for forensic inspection.
        outputs: List of enriched output dicts with ``filename``,
            ``subfolder``, ``type``, ``local_path``, ``size_bytes``, and
            ``is_valid_png``.
        snapshot: Parsed snapshot output (8 fields produced by
            ``snapshot.SnapshotResult``).
        errors_during_run: Aggregated string errors captured during the
            run (e.g. SSH or snapshot parsing warnings that were
            recoverable). Empty list = clean run.
    """

    prompt_id: str
    wallclock_seconds: float
    seed: int
    history_entry: dict[str, Any]
    outputs: list[dict[str, Any]]
    snapshot: dict[str, Any]
    errors_during_run: list[str]


@dataclass(frozen=True, slots=True)
class RunnerSummary:
    """Schema-versioned runner output (``schema_version=1``).

    Attributes:
        schema_version: Output schema version. Currently ``1``.
        machine_id: Identifier of the executor machine (e.g. ``"cg_3060"``).
        workflow: Path or name of the workflow JSON used.
        timestamp_utc: ISO-8601 UTC timestamp of the runner invocation.
        runs: List of :class:`RunResult`. V1 minimal contains exactly one
            element; V2+ DA-008 mechanic will return five.
        aggregated: Aggregated multi-run statistics (mean, stddev, p50)
            or ``None`` in V1 minimal. V2+ will populate.
    """

    schema_version: int
    machine_id: str
    workflow: str
    timestamp_utc: str
    runs: list[RunResult]
    aggregated: dict[str, Any] | None
