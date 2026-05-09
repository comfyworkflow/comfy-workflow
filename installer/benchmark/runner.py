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
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Canonical path to the public repo on each executor (cg-3060/cg-4090/cg-5090).
# Set up by the bootstrap audit (Phase 0). Forward slashes are accepted by both
# cmd and PowerShell on Windows.
REMOTE_REPO_PATH = "C:/ComfyWorkflowVS/comfy-workflow"

# Default stop-flag location on the remote. Uses cmd-style env var; the
# remote shell (cmd by default on Windows OpenSSH) expands ``%USERPROFILE%``
# before passing the argument to Python or to ``type nul``.
REMOTE_STOP_FLAG_DEFAULT = "%USERPROFILE%\\runner_stop.flag"


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


def _ssh_run(host: str, command: str, timeout: int = 30) -> str:
    """Run a command on a remote host via SSH and return its stdout.

    Mirrors :func:`installer.benchmark.dry_run._ssh_run` (copied, not
    imported, per Bloco 15 plan to avoid coupling). When dry_run.py is
    deprecated (Bloco 16+), the helper consolidates here.

    Args:
        host: SSH host alias (e.g. ``"cg-3060"``).
        command: Single command line passed verbatim to ``ssh``. Quoting
            must be valid for the remote shell (cmd by default on Windows
            OpenSSH).
        timeout: Seconds to wait before raising :class:`RunnerSSHError`.

    Returns:
        Captured stdout, decoded as UTF-8.

    Raises:
        RunnerSSHError: ``ssh`` exited non-zero or the call timed out.
    """
    args = ["ssh", host, command]
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RunnerSSHError(
            f"ssh {host}: command timed out after {timeout}s: {command[:100]!r}"
        ) from exc
    if result.returncode != 0:
        raise RunnerSSHError(
            f"ssh {host}: exit {result.returncode}: "
            f"stderr={result.stderr.strip()!r}"
        )
    return result.stdout


def _ssh_pull(host: str, repo_path: str = REMOTE_REPO_PATH) -> str:
    """Run ``git pull`` on the remote host's clone. Returns stdout for logging.

    Args:
        host: SSH host alias.
        repo_path: Absolute path to the repo on the remote (forward
            slashes). Defaults to :data:`REMOTE_REPO_PATH`.

    Raises:
        RunnerSSHError: ``git pull`` failed or timed out.
    """
    return _ssh_run(host, f'git -C "{repo_path}" pull', timeout=60)


def _signal_snapshot_stop(
    host: str, flag_path: str = REMOTE_STOP_FLAG_DEFAULT
) -> None:
    """Create the snapshot stop flag file on the remote host (idempotent).

    Uses cmd's redirection (``type nul > <path>``) to create or overwrite
    an empty file. Idempotent: re-running just overwrites with a fresh
    empty file. The remote shell expands cmd-style env vars (e.g.
    ``%USERPROFILE%``) before executing.

    Args:
        host: SSH host alias.
        flag_path: Path on the remote. Default
            :data:`REMOTE_STOP_FLAG_DEFAULT`.

    Raises:
        RunnerSSHError: ``ssh`` exited non-zero or timed out.
    """
    _ssh_run(host, f"type nul > {flag_path}", timeout=15)
