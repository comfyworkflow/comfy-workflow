"""End-to-end dry run orchestrator for the ComfyUI benchmark pipeline.

First integration test of ``interface.py`` and ``snapshot.py`` running
together against a real workflow. Validates POST endpoints
(``queue_prompt``, polling, ``get_image``), SSH dispatch of
``snapshot.py`` to an executor, and a structured ``schema_version=1``
JSON output suitable for future aggregation.

Architecture:
    - This script runs on the coordinator (Itapoá) and dispatches
      ``snapshot.py`` via SSH to one executor (default ``cg-3060``).
      NVML is local to the executor; this script never touches NVML
      directly and never imports ``snapshot``.
    - ``interface.ComfyUIClient`` runs locally and reaches the executor's
      ComfyUI server via Tailscale.
    - The two run concurrently: ``snapshot.py`` polls hardware in a
      subprocess while a workflow is queued and polled to completion.

Scope:
    - Bloco 13: validates against cg-3060 only (princípio do elo fraco).
    - Bloco 14+: cross-CG validation on cg-4090 and cg-5090.
    - ``runner.py`` (later block): replaces dry_run with full DA-008
      mechanics (5 runs, cold/warm split, min/max discard, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class DryRunError(Exception):
    """Base class for all dry_run module errors."""


class DryRunSSHError(DryRunError):
    """Raised when an SSH command fails (non-zero exit, timeout, etc.)."""


class DryRunWorkflowError(DryRunError):
    """Raised when a workflow execution fails.

    Examples: queue rejection by ComfyUI, schema validation, missing
    outputs in the history entry.
    """


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Result of a single workflow execution.

    Attributes:
        prompt_id: UUID assigned by ComfyUI's ``/prompt`` endpoint.
        wallclock_seconds: End-to-end wall time from queue submission to
            completion (``poll_history`` return).
        history_entry: Raw ``/history/<prompt_id>`` entry returned by the
            server.
        outputs: List of ``{filename, subfolder, type}`` dicts extracted
            from the ``SaveImage`` node's history entry.
    """

    prompt_id: str
    wallclock_seconds: float
    history_entry: dict[str, Any]
    outputs: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class DryRunSummary:
    """Schema-versioned dry-run output (``schema_version=1``).

    The shape of this dataclass is the canonical V1 dry-run JSON schema.
    Future ``runner.py`` output will extend it (cold/warm split, multi-run
    statistics) but maintain backwards compatibility on these fields.

    Attributes:
        schema_version: Output schema version. Currently ``1``.
        machine_id: Identifier of the executor machine (e.g. ``"cg_3060"``).
        workflow: Path or name of the workflow JSON used.
        timestamp_utc: ISO-8601 UTC timestamp of the run.
        wallclock_seconds: End-to-end workflow execution time.
        prompt_id: UUID returned by ComfyUI's ``/prompt`` endpoint.
        snapshot: Parsed snapshot output, dict with the 8 fields produced
            by ``snapshot.SnapshotResult``.
        outputs: List of ``{filename, subfolder, type, local_path,
            size_bytes, is_valid_png}`` dicts.
    """

    schema_version: int
    machine_id: str
    workflow: str
    timestamp_utc: str
    wallclock_seconds: float
    prompt_id: str
    snapshot: dict[str, Any]
    outputs: list[dict[str, Any]]
