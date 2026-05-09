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

import argparse
import json
import logging
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from installer.benchmark import interface

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


def _extract_outputs(
    history_entry: dict[str, Any], prompt_id: str
) -> list[dict[str, str]]:
    """Extract image outputs from a ComfyUI ``/history/<prompt_id>`` entry.

    Iterates over ``history_entry["outputs"]`` (a dict keyed by node ID),
    aggregating any ``images`` arrays. Each output is normalized to
    ``{filename, subfolder, type}`` with string values. Non-image outputs
    and malformed entries are silently skipped.

    Raises:
        DryRunWorkflowError: ``outputs`` key is missing or no images were
            produced.
    """
    raw_outputs = history_entry.get("outputs")
    if not isinstance(raw_outputs, dict):
        raise DryRunWorkflowError(
            f"history entry for prompt_id={prompt_id} missing 'outputs' dict"
        )

    images: list[dict[str, str]] = []
    for node_outputs in raw_outputs.values():
        if not isinstance(node_outputs, dict):
            continue
        node_images = node_outputs.get("images")
        if not isinstance(node_images, list):
            continue
        for img in node_images:
            if not isinstance(img, dict):
                continue
            filename = img.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            images.append({
                "filename": filename,
                "subfolder": str(img.get("subfolder", "")),
                "type": str(img.get("type", "output")),
            })

    if not images:
        raise DryRunWorkflowError(
            f"history entry for prompt_id={prompt_id} has no image outputs"
        )
    return images


def _run_workflow(
    client: interface.ComfyUIClient,
    workflow_path: Path,
    ckpt_filename: str,
    workflow_timeout: int = 60,
) -> WorkflowResult:
    """Load a workflow JSON, queue it on ComfyUI, poll until completion.

    The full lifecycle: read and validate the workflow JSON, sanity-check
    that the expected checkpoint is registered on the server, generate a
    fresh ``client_id``, time the queue → completion roundtrip with
    :func:`time.monotonic`, then extract image outputs from the history
    entry.

    Args:
        client: A live :class:`interface.ComfyUIClient` pointing at the
            executor's ComfyUI server.
        workflow_path: Path to a workflow JSON file in ComfyUI API format
            (i.e. ``{node_id: {class_type, inputs}}``).
        ckpt_filename: Expected checkpoint filename (e.g.
            ``"sd_xl_base_1.0.safetensors"``). Sanity-checked against the
            server's :meth:`interface.ComfyUIClient.list_checkpoints`
            before queueing.
        workflow_timeout: Seconds to wait for the workflow to finish.

    Returns:
        :class:`WorkflowResult` with ``prompt_id``, ``wallclock_seconds``,
        the raw ``history_entry``, and parsed image outputs.

    Raises:
        DryRunWorkflowError: Workflow file missing or malformed; checkpoint
            not registered; outputs missing from the history entry.
        interface.ComfyUIError: Any failure from the underlying HTTP calls
            (network, queue rejection, timeout, etc.).
    """
    if not workflow_path.is_file():
        raise DryRunWorkflowError(f"workflow file not found: {workflow_path}")

    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DryRunWorkflowError(
            f"workflow file {workflow_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(workflow, dict):
        raise DryRunWorkflowError(
            f"workflow file {workflow_path} is not a JSON object"
        )

    registered = client.list_checkpoints()
    if ckpt_filename not in registered:
        raise DryRunWorkflowError(
            f"checkpoint {ckpt_filename!r} not registered on the server "
            f"(found: {registered})"
        )

    client_id = uuid.uuid4().hex

    t0 = time.monotonic()
    prompt_id = client.queue_prompt(workflow, client_id=client_id)
    logger.info("queued workflow prompt_id=%s client_id=%s", prompt_id, client_id)

    history_entry = client.poll_history(
        prompt_id,
        timeout=workflow_timeout,
        poll_interval=1.0,
        progress_callback=lambda elapsed: logger.debug(
            "polling /history/%s elapsed=%.1fs", prompt_id, elapsed
        ),
    )
    wallclock = time.monotonic() - t0

    outputs = _extract_outputs(history_entry, prompt_id)
    return WorkflowResult(
        prompt_id=prompt_id,
        wallclock_seconds=wallclock,
        history_entry=history_entry,
        outputs=outputs,
    )


# PNG signature: first 8 bytes of any valid PNG file.
# https://www.w3.org/TR/PNG/#5PNG-file-signature
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _download_outputs(
    client: interface.ComfyUIClient,
    outputs: list[dict[str, str]],
    dest_dir: Path,
) -> list[dict[str, Any]]:
    """Download each workflow output via ``/view`` and validate as PNG.

    For each entry in ``outputs`` (a list of ``{filename, subfolder, type}``),
    fetches the image bytes via :meth:`interface.ComfyUIClient.get_image`,
    writes them to ``dest_dir/filename``, and inspects the first 8 bytes
    against the PNG signature. The returned dicts are enriched with
    ``local_path``, ``size_bytes``, and ``is_valid_png``.

    Args:
        client: A live :class:`interface.ComfyUIClient`.
        outputs: List of output descriptors from :class:`WorkflowResult`.
        dest_dir: Destination directory; created if it does not exist.

    Returns:
        New list of dicts (one per input) with the enriched fields.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict[str, Any]] = []
    for out in outputs:
        filename = out["filename"]
        subfolder = out["subfolder"]
        image_type = out["type"]
        data = client.get_image(filename, subfolder, image_type)
        local_path = dest_dir / filename
        local_path.write_bytes(data)
        is_valid_png = len(data) >= 8 and data[:8] == _PNG_MAGIC
        enriched.append({
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type,
            "local_path": str(local_path),
            "size_bytes": len(data),
            "is_valid_png": is_valid_png,
        })
    return enriched


def _save_summary(summary: DryRunSummary, output_path: Path) -> None:
    """Write a :class:`DryRunSummary` as pretty JSON (indent=2, UTF-8).

    Creates ``output_path.parent`` if it does not exist. Trailing newline
    appended for POSIX-friendliness.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``main()``. Defaults target cg-3060."""
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end dry run of the ComfyUI benchmark pipeline: spawns "
            "snapshot.py via SSH on an executor, queues a workflow on the "
            "executor's ComfyUI server, downloads outputs, and writes a "
            "schema_version=1 JSON summary."
        ),
    )
    parser.add_argument(
        "--ssh-host",
        default="cg-3060",
        help="SSH host alias for the executor (default: cg-3060).",
    )
    parser.add_argument(
        "--target",
        default="http://100.72.255.77:8188",
        help="Base URL of the executor's ComfyUI server "
        "(default: cg-3060 Tailscale).",
    )
    parser.add_argument(
        "--workflow",
        default="installer/benchmark/workflows/sdxl_base_dry_run.json",
        help="Path to workflow JSON (default: SDXL base dry-run).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./dry_run_outputs/<UTC-timestamp>).",
    )
    parser.add_argument(
        "--snapshot-duration",
        type=int,
        default=30,
        help="snapshot.py --duration arg, seconds (default: 30).",
    )
    parser.add_argument(
        "--snapshot-device",
        type=int,
        default=0,
        help="snapshot.py --device arg, NVML index (default: 0).",
    )
    parser.add_argument(
        "--snapshot-interval-ms",
        type=int,
        default=100,
        help="snapshot.py --interval arg, milliseconds (default: 100).",
    )
    parser.add_argument(
        "--workflow-timeout",
        type=int,
        default=60,
        help="poll_history timeout, seconds (default: 60).",
    )
    parser.add_argument(
        "--checkpoint",
        default="sd_xl_base_1.0.safetensors",
        help="Expected checkpoint filename for sanity check.",
    )
    return parser


def main() -> None:
    """End-to-end dry run orchestrator (Bloco 13 entrypoint).

    Sequence:
        1. ``git pull`` on the executor (pre-step).
        2. Sanity-check ComfyUI server (``is_alive``, ``list_checkpoints``).
        3. Spawn ``snapshot.py`` via SSH (asynchronous Popen).
        4. Brief sleep so snapshot's NVML init completes.
        5. Run workflow (queue + poll), measure wallclock.
        6. Wait for snapshot to finish, parse its stdout.
        7. Download workflow outputs, validate PNG headers.
        8. Build :class:`DryRunSummary`, save as pretty JSON.
        9. Print summary to stdout.

    Raises:
        DryRunError, DryRunSSHError, DryRunWorkflowError, interface.ComfyUIError:
            On any step failure; the script exits non-zero via the
            uncaught exception.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _build_argparser().parse_args()
    ssh_host = cast(str, args.ssh_host)
    target = cast(str, args.target)
    workflow_path = Path(cast(str, args.workflow))
    output_dir_arg = cast("str | None", args.output_dir)
    snapshot_duration = cast(int, args.snapshot_duration)
    snapshot_device = cast(int, args.snapshot_device)
    snapshot_interval_ms = cast(int, args.snapshot_interval_ms)
    workflow_timeout = cast(int, args.workflow_timeout)
    checkpoint = cast(str, args.checkpoint)

    if output_dir_arg is None:
        ts_dir = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("dry_run_outputs") / ts_dir
    else:
        output_dir = Path(output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pre-step git pull
    pull_out = _ssh_pull(ssh_host)
    pull_summary = pull_out.strip().splitlines()[0] if pull_out.strip() else "(empty)"
    logger.info("git pull on %s: %s", ssh_host, pull_summary)

    # 2) Sanity checks
    client = interface.ComfyUIClient(target, timeout=workflow_timeout)
    if not client.is_alive():
        raise DryRunError(f"ComfyUI server at {target} did not respond")
    logger.info("is_alive: True")
    registered = client.list_checkpoints()
    if checkpoint not in registered:
        raise DryRunError(
            f"checkpoint {checkpoint!r} not registered on the server "
            f"(found: {registered})"
        )
    logger.info("checkpoint registered: True (%s)", checkpoint)

    # 3) Spawn snapshot
    logger.info(
        "spawning snapshot.py on %s for %ds (device=%d, interval=%dms)",
        ssh_host, snapshot_duration, snapshot_device, snapshot_interval_ms,
    )
    snapshot_proc = _spawn_snapshot(
        ssh_host, snapshot_duration, snapshot_device, snapshot_interval_ms,
    )

    # 4) Brief sleep to let snapshot init NVML before workflow starts
    time.sleep(2.0)
    if snapshot_proc.poll() is not None:
        stdout, stderr = snapshot_proc.communicate()
        raise DryRunSSHError(
            f"snapshot exited prematurely (returncode={snapshot_proc.returncode}): "
            f"stderr={stderr.strip()[:500]!r}"
        )

    # 5) Run workflow
    workflow_result = _run_workflow(
        client, workflow_path, checkpoint, workflow_timeout,
    )
    logger.info(
        "workflow completed in %.2fs (prompt_id=%s)",
        workflow_result.wallclock_seconds, workflow_result.prompt_id,
    )

    # 6) Wait snapshot finish
    logger.info("waiting for snapshot.py to finish...")
    try:
        snap_stdout, snap_stderr = snapshot_proc.communicate(
            timeout=snapshot_duration + 30,
        )
    except subprocess.TimeoutExpired as exc:
        snapshot_proc.kill()
        raise DryRunSSHError(
            f"snapshot did not finish within {snapshot_duration + 30}s"
        ) from exc
    if snapshot_proc.returncode != 0:
        raise DryRunSSHError(
            f"snapshot exited non-zero ({snapshot_proc.returncode}): "
            f"stderr={snap_stderr.strip()[:500]!r}"
        )
    snapshot_dict = _parse_snapshot_stdout(snap_stdout)
    logger.info(
        "snapshot finished: %d samples in %.2fs",
        snapshot_dict["samples_collected"], snapshot_dict["duration_seconds"],
    )

    # 7) Download outputs
    outputs_with_local = _download_outputs(
        client, workflow_result.outputs, output_dir,
    )
    valid_count = sum(1 for o in outputs_with_local if o["is_valid_png"])
    logger.info(
        "downloaded %d image(s), %d valid PNG",
        len(outputs_with_local), valid_count,
    )

    # 8) Build & save summary
    machine_id = ssh_host.replace("-", "_")
    timestamp_utc = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    summary = DryRunSummary(
        schema_version=1,
        machine_id=machine_id,
        workflow=str(workflow_path),
        timestamp_utc=timestamp_utc,
        wallclock_seconds=workflow_result.wallclock_seconds,
        prompt_id=workflow_result.prompt_id,
        snapshot=snapshot_dict,
        outputs=outputs_with_local,
    )
    summary_path = output_dir / "summary.json"
    _save_summary(summary, summary_path)
    logger.info("saved summary to %s", summary_path)

    # 9) Print pretty JSON to stdout
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
