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

import argparse
import copy
import json
import logging
import random
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from installer.benchmark import interface

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


def _spawn_snapshot_until_signal(
    host: str,
    stop_flag_path: str = REMOTE_STOP_FLAG_DEFAULT,
    device_index: int = 0,
    poll_interval_ms: int = 100,
) -> subprocess.Popen[str]:
    """Spawn ``snapshot.py`` on the remote in ``--until-signal`` mode (asynchronous).

    The remote process polls hardware until ``stop_flag_path`` appears
    (created by :func:`_signal_snapshot_stop` from the coordinator).

    Args:
        host: SSH host alias.
        stop_flag_path: Remote path of the stop flag. cmd-style env vars
            are expanded by the remote shell.
        device_index: Forwarded to snapshot.py's ``--device``.
        poll_interval_ms: Forwarded to snapshot.py's ``--interval``.

    Returns:
        Live :class:`subprocess.Popen` with stdout / stderr captured as
        text. Caller is responsible for ``communicate`` / ``wait``.
    """
    remote_command = (
        f'cd "{REMOTE_REPO_PATH}" && '
        f"python -m installer.benchmark.snapshot --until-signal "
        f"--stop-flag {stop_flag_path} "
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


def _find_ksampler_node_id(workflow: dict[str, Any]) -> str:
    """Return the single KSampler node ID, or raise.

    Mirrors :func:`installer.benchmark.dry_run._find_ksampler_node_id`
    (copy, not import). Bloco 16+ may consolidate when dry_run.py is
    deprecated.
    """
    ksampler_ids = [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == "KSampler"
    ]
    if len(ksampler_ids) != 1:
        raise RunnerWorkflowError(
            f"workflow must contain exactly one KSampler node, "
            f"found {len(ksampler_ids)}: {ksampler_ids}"
        )
    return ksampler_ids[0]


def _inject_seed(workflow: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a deep-copy of ``workflow`` with KSampler ``inputs.seed = seed``.

    Mirrors :func:`installer.benchmark.dry_run._inject_seed`.
    """
    result = copy.deepcopy(workflow)
    ksampler_id = _find_ksampler_node_id(result)
    result[ksampler_id]["inputs"]["seed"] = seed
    return result


# Field name → parser. Mirrors snapshot.main() print format.
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
    """Parse stdout of ``snapshot.main()`` into a structured dict.

    Mirrors :func:`installer.benchmark.dry_run._parse_snapshot_stdout`.
    Header lines (``device index:``, ``poll interval:``, ``watching stop
    flag:``, etc.) are silently skipped.
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
            raise RunnerWorkflowError(
                f"failed to parse snapshot field {key!r}={value!r}: {exc}"
            ) from exc

    missing = set(_SNAPSHOT_FIELD_PARSERS) - parsed.keys()
    if missing:
        raise RunnerWorkflowError(
            f"snapshot stdout missing required fields: {sorted(missing)}"
        )
    return parsed


def _extract_outputs(
    history_entry: dict[str, Any], prompt_id: str
) -> list[dict[str, Any]]:
    """Extract image outputs from a ComfyUI ``/history/<prompt_id>`` entry.

    Mirrors :func:`installer.benchmark.dry_run._extract_outputs`. Returns
    pre-download dicts; enrichment with ``local_path``/``size_bytes``/
    ``is_valid_png`` is the responsibility of :func:`_download_outputs`
    (Sub-tarefa 5).
    """
    raw_outputs = history_entry.get("outputs")
    if not isinstance(raw_outputs, dict):
        raise RunnerWorkflowError(
            f"history entry for prompt_id={prompt_id} missing 'outputs' dict"
        )

    images: list[dict[str, Any]] = []
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
        raise RunnerWorkflowError(
            f"history entry for prompt_id={prompt_id} has no image outputs"
        )
    return images


def _run_single(
    client: interface.ComfyUIClient,
    workflow_path: Path,
    ckpt_filename: str,
    seed: int,
    snapshot_proc: subprocess.Popen[str],
    stop_flag_remote_path: str,
    host: str,
    workflow_timeout: int = 60,
) -> RunResult:
    """Execute one workflow run with lifecycle-aligned snapshot collection.

    Assumes ``snapshot_proc`` is already running on the remote (spawned
    pre-call by the caller via :func:`_spawn_snapshot_until_signal`).
    Queues the workflow on ``client``, blocks until completion, then
    signals the snapshot to stop and collects its stdout. Snapshot
    duration ≈ workflow wallclock, eliminating the idle window of
    dry_run.py (V2 débito #7).

    Args:
        client: Live ComfyUIClient.
        workflow_path: Workflow JSON path.
        ckpt_filename: Expected checkpoint, sanity-checked.
        seed: KSampler seed (positive int32).
        snapshot_proc: Already-running remote snapshot process.
        stop_flag_remote_path: Remote flag path used by ``snapshot.py``;
            the caller must supply the same value passed to
            :func:`_spawn_snapshot_until_signal`.
        host: SSH host alias used to signal stop.
        workflow_timeout: ``poll_history`` timeout in seconds.

    Returns:
        :class:`RunResult` with ``prompt_id``, ``wallclock_seconds``,
        ``seed``, ``history_entry``, raw ``outputs``, parsed
        ``snapshot``, and an ``errors_during_run`` list (currently
        always empty in V1; reserved for V2+ accumulation).

    Raises:
        RunnerWorkflowError: workflow file or response invalid.
        RunnerSSHError: snapshot signaling or stdout collection failed.
        interface.ComfyUIError: HTTP-level failure.
    """
    if not workflow_path.is_file():
        raise RunnerWorkflowError(f"workflow file not found: {workflow_path}")

    try:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerWorkflowError(
            f"workflow file {workflow_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(workflow, dict):
        raise RunnerWorkflowError(
            f"workflow file {workflow_path} is not a JSON object"
        )

    registered = client.list_checkpoints()
    if ckpt_filename not in registered:
        raise RunnerWorkflowError(
            f"checkpoint {ckpt_filename!r} not registered on the server "
            f"(found: {registered})"
        )

    workflow = _inject_seed(workflow, seed)
    logger.info("queueing workflow with seed=%d", seed)

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
    logger.info("workflow completed in %.2fs", wallclock)

    # Lifecycle alignment: signal snapshot to stop now that workflow is done.
    errors_during_run: list[str] = []
    _signal_snapshot_stop(host, stop_flag_remote_path)
    logger.info("snapshot stop signaled; collecting stdout...")
    try:
        snap_stdout, snap_stderr = snapshot_proc.communicate(timeout=10)
    except subprocess.TimeoutExpired as exc:
        snapshot_proc.kill()
        raise RunnerSSHError(
            "snapshot did not finish within 10s after stop signal"
        ) from exc
    if snapshot_proc.returncode != 0:
        raise RunnerSSHError(
            f"snapshot exited non-zero ({snapshot_proc.returncode}): "
            f"stderr={snap_stderr.strip()[:500]!r}"
        )

    snapshot_dict = _parse_snapshot_stdout(snap_stdout)
    outputs = _extract_outputs(history_entry, prompt_id)

    return RunResult(
        prompt_id=prompt_id,
        wallclock_seconds=wallclock,
        seed=seed,
        history_entry=history_entry,
        outputs=outputs,
        snapshot=snapshot_dict,
        errors_during_run=errors_during_run,
    )


# PNG signature: first 8 bytes of any valid PNG file.
# https://www.w3.org/TR/PNG/#5PNG-file-signature
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _download_outputs(
    client: interface.ComfyUIClient,
    outputs: list[dict[str, Any]],
    dest_dir: Path,
) -> list[dict[str, Any]]:
    """Download each workflow output via ``/view`` and validate as PNG.

    Mirrors :func:`installer.benchmark.dry_run._download_outputs`. For
    each entry, fetches the image bytes via
    :meth:`interface.ComfyUIClient.get_image`, writes them to
    ``dest_dir/filename``, and inspects the first 8 bytes against the
    PNG signature. Returned dicts include ``local_path``, ``size_bytes``,
    and ``is_valid_png`` in addition to the input fields.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    enriched: list[dict[str, Any]] = []
    for out in outputs:
        filename = str(out["filename"])
        subfolder = str(out["subfolder"])
        image_type = str(out["type"])
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


def _save_summary(summary: RunnerSummary, output_path: Path) -> None:
    """Write a :class:`RunnerSummary` as pretty JSON (indent=2, UTF-8).

    Creates ``output_path.parent`` if it does not exist. Trailing
    newline appended for POSIX-friendliness. Mirrors
    :func:`installer.benchmark.dry_run._save_summary`.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``main()``. Defaults target cg-3060."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark runner with lifecycle-aligned snapshot dispatch. "
            "Spawns snapshot.py via SSH (--until-signal mode), queues a "
            "workflow on the executor's ComfyUI server, signals snapshot "
            "stop on completion, downloads outputs, and writes a "
            "schema_version=1 RunnerSummary JSON. V1 minimal: one run."
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
        help="Output directory (default: ./runner_outputs/<UTC-timestamp>).",
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
        "--seed",
        type=int,
        default=None,
        help="KSampler seed. Default: random per run (logged at INFO).",
    )
    parser.add_argument(
        "--checkpoint",
        default="sd_xl_base_1.0.safetensors",
        help="Expected checkpoint filename for sanity check.",
    )
    parser.add_argument(
        "--stop-flag-remote",
        default=REMOTE_STOP_FLAG_DEFAULT,
        help=(
            "Remote path of the snapshot stop flag (default: "
            f"{REMOTE_STOP_FLAG_DEFAULT}). cmd-style env vars expanded "
            "by remote shell."
        ),
    )
    return parser


def main() -> None:
    """End-to-end runner orchestrator (Bloco 15+, V1 minimal one-run).

    Sequence:
        1. ``git pull`` on the executor (pre-step).
        2. Sanity-check ComfyUI server (``is_alive``, ``list_checkpoints``).
        3. Spawn ``snapshot.py`` in ``--until-signal`` mode (asynchronous,
           BEFORE ``queue_prompt`` for lifecycle alignment).
        4. Brief sleep so snapshot's NVML init completes.
        5. :func:`_run_single`: queue + poll + signal stop + collect.
        6. Download workflow outputs, validate PNG headers.
        7. Build :class:`RunnerSummary` (one run, ``aggregated=None``),
           save as pretty JSON.
        8. Print summary to stdout.

    Raises:
        RunnerError, RunnerSSHError, RunnerWorkflowError,
        interface.ComfyUIError: On any step failure; the script exits
        non-zero via the uncaught exception.
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
    snapshot_device = cast(int, args.snapshot_device)
    snapshot_interval_ms = cast(int, args.snapshot_interval_ms)
    workflow_timeout = cast(int, args.workflow_timeout)
    seed_arg = cast("int | None", args.seed)
    checkpoint = cast(str, args.checkpoint)
    stop_flag_remote = cast(str, args.stop_flag_remote)

    if output_dir_arg is None:
        ts_dir = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("runner_outputs") / ts_dir
    else:
        output_dir = Path(output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve effective seed (random per-run if not provided) and validate.
    effective_seed = (
        seed_arg if seed_arg is not None else random.randint(0, 2**31 - 1)
    )
    if effective_seed < 0 or effective_seed >= 2**31:
        raise RunnerError(
            f"seed out of range: {effective_seed} "
            f"(must be in [0, 2**31)); pass --seed N with N in that range."
        )
    logger.info(
        "effective seed: %d (provided=%s)",
        effective_seed, seed_arg is not None,
    )

    # 1) Pre-step git pull
    pull_out = _ssh_pull(ssh_host)
    pull_summary = pull_out.strip().splitlines()[0] if pull_out.strip() else "(empty)"
    logger.info("git pull on %s: %s", ssh_host, pull_summary)

    # 2) Sanity checks
    client = interface.ComfyUIClient(target, timeout=workflow_timeout)
    if not client.is_alive():
        raise RunnerError(f"ComfyUI server at {target} did not respond")
    logger.info("is_alive: True")
    registered = client.list_checkpoints()
    if checkpoint not in registered:
        raise RunnerError(
            f"checkpoint {checkpoint!r} not registered on the server "
            f"(found: {registered})"
        )
    logger.info("checkpoint registered: True (%s)", checkpoint)

    # 3) Spawn snapshot in --until-signal mode (BEFORE queue_prompt for
    #    lifecycle alignment).
    logger.info(
        "spawning snapshot.py on %s in --until-signal mode "
        "(device=%d, interval=%dms, flag=%s)",
        ssh_host, snapshot_device, snapshot_interval_ms, stop_flag_remote,
    )
    snapshot_proc = _spawn_snapshot_until_signal(
        ssh_host, stop_flag_remote, snapshot_device, snapshot_interval_ms,
    )

    # 4) Brief sleep so snapshot inits NVML before workflow starts.
    time.sleep(2.0)
    if snapshot_proc.poll() is not None:
        stdout, stderr = snapshot_proc.communicate()
        raise RunnerSSHError(
            f"snapshot exited prematurely (returncode={snapshot_proc.returncode}): "
            f"stderr={stderr.strip()[:500]!r}"
        )

    # 5) Run workflow and collect snapshot via lifecycle-aligned dispatch.
    run_result = _run_single(
        client=client,
        workflow_path=workflow_path,
        ckpt_filename=checkpoint,
        seed=effective_seed,
        snapshot_proc=snapshot_proc,
        stop_flag_remote_path=stop_flag_remote,
        host=ssh_host,
        workflow_timeout=workflow_timeout,
    )

    # 6) Download outputs and enrich.
    enriched_outputs = _download_outputs(client, run_result.outputs, output_dir)
    valid_count = sum(1 for o in enriched_outputs if o["is_valid_png"])
    logger.info(
        "downloaded %d image(s), %d valid PNG",
        len(enriched_outputs), valid_count,
    )

    # 7) Build & save summary. RunResult is frozen, so use dataclasses.replace
    #    to swap in the enriched outputs.
    enriched_run = replace(run_result, outputs=enriched_outputs)
    machine_id = ssh_host.replace("-", "_")
    timestamp_utc = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    summary = RunnerSummary(
        schema_version=1,
        machine_id=machine_id,
        workflow=str(workflow_path),
        timestamp_utc=timestamp_utc,
        runs=[enriched_run],
        aggregated=None,
    )
    summary_path = output_dir / "summary.json"
    _save_summary(summary, summary_path)
    logger.info("saved summary to %s", summary_path)

    # 8) Print pretty JSON to stdout.
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
