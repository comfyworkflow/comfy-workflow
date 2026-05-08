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
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Canonical path to the public repo on each executor (cg-3060/cg-4090/cg-5090).
# Set up by the bootstrap audit (Phase 0). Forward slashes are accepted by both
# cmd and PowerShell on Windows.
REMOTE_REPO_PATH = "C:/ComfyWorkflowVS/comfy-workflow"


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


def _ssh_run(host: str, command: str, timeout: int = 30) -> str:
    """Run a command on a remote host via SSH and return its stdout.

    Args:
        host: SSH host alias (e.g. ``"cg-3060"``, configured in
            ``~/.ssh/config``).
        command: Single command line to execute. Passed as a single
            argument to ``ssh``; quoting must be valid for the remote
            shell (cmd or PowerShell on the Windows executors).
        timeout: Seconds to wait before raising :class:`DryRunSSHError`.

    Returns:
        Captured stdout, decoded as UTF-8 text.

    Raises:
        DryRunSSHError: ``ssh`` exited non-zero or the call timed out.
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
        raise DryRunSSHError(
            f"ssh {host}: command timed out after {timeout}s: {command[:100]!r}"
        ) from exc
    if result.returncode != 0:
        raise DryRunSSHError(
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

    Returns:
        Captured stdout (e.g. ``"Already up to date."`` or fast-forward
        summary), useful for logging.

    Raises:
        DryRunSSHError: ``git pull`` failed or timed out.
    """
    return _ssh_run(host, f'git -C "{repo_path}" pull', timeout=60)


def _spawn_snapshot(
    host: str,
    duration_seconds: float,
    device_index: int = 0,
    poll_interval_ms: int = 100,
) -> subprocess.Popen[str]:
    """Spawn ``snapshot.py`` on the remote host via SSH (asynchronous).

    The snapshot process polls hardware on the executor's GPU for the
    given duration and prints its summary to stdout. The returned
    :class:`subprocess.Popen` is alive; use ``.wait(timeout)`` to block
    on completion, then read ``.stdout``.

    Args:
        host: SSH host alias.
        duration_seconds: Forwarded to snapshot.py's ``--duration`` flag.
            Coerced to int via ``round``.
        device_index: Forwarded to snapshot.py's ``--device`` flag.
        poll_interval_ms: Forwarded to snapshot.py's ``--interval`` flag.

    Returns:
        A live :class:`subprocess.Popen` with stdout and stderr captured
        as text. Caller is responsible for ``.wait`` / ``.communicate``.
    """
    duration_int = int(round(duration_seconds))
    remote_command = (
        f'cd "{REMOTE_REPO_PATH}" && '
        f"python -m installer.benchmark.snapshot "
        f"--duration {duration_int} "
        f"--device {device_index} "
        f"--interval {poll_interval_ms}"
    )
    args = ["ssh", host, remote_command]
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# Field name → parser. Mirrors the 8 fields printed by snapshot.main() in its
# self-test entrypoint. Adding fields here when snapshot.py grows is the only
# coupling point between the two modules.
_SNAPSHOT_FIELD_PARSERS: dict[str, Callable[[str], Any]] = {
    "samples_collected": int,
    "duration_seconds": float,
    "peak_vram_mb": int,
    "peak_ram_gb": float,
    "gpu_avg_utilization_pct": float,
    "gpu_avg_temp_c": float,
    "gpu_avg_power_w": float,
    "errors_during_collection": int,
}


def _parse_snapshot_stdout(stdout: str) -> dict[str, Any]:
    """Parse the stdout of ``snapshot.main()`` into a structured dict.

    Expects the format produced by ``snapshot.py``'s ``__main__`` self-test
    entrypoint::

        samples_collected: <int>
        duration_seconds: <float>
        peak_vram_mb: <int>
        peak_ram_gb: <float>
        gpu_avg_utilization_pct: <float>
        gpu_avg_temp_c: <float>
        gpu_avg_power_w: <float>
        errors_during_collection: <int>

    Lines outside this set (header lines like ``device index: 0``,
    ``poll interval: 100 ms``, ``collecting for 3.0s...``) are ignored.

    Args:
        stdout: Captured stdout from a snapshot.py run.

    Returns:
        Dict with the 8 fields above, parsed to their proper numeric types.

    Raises:
        DryRunWorkflowError: One or more required fields are missing or
            have an unparseable value.
    """
    parsed: dict[str, Any] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        value = raw_value.strip()
        parser = _SNAPSHOT_FIELD_PARSERS.get(key)
        if parser is None:
            continue
        try:
            parsed[key] = parser(value)
        except (ValueError, TypeError) as exc:
            raise DryRunWorkflowError(
                f"failed to parse snapshot field {key!r}={value!r}: {exc}"
            ) from exc

    missing = set(_SNAPSHOT_FIELD_PARSERS) - parsed.keys()
    if missing:
        raise DryRunWorkflowError(
            f"snapshot stdout missing required fields: {sorted(missing)}"
        )
    return parsed
