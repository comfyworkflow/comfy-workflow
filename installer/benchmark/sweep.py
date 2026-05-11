"""Benchmark sweep orchestrator — host-major iteration over V1 workflows.

Single responsibility: orchestrate the V1 benchmark matrix
(M workflows × N hosts × K runs) into a single consolidated JSON. The
inner-loop execution (snapshot dispatch, workflow queue, history poll,
output download) is delegated to :mod:`installer.benchmark.runner`;
this module composes those primitives across the matrix.

Sweep order is **host-major**: host A runs every workflow × K runs,
then host B, then host C. This minimizes cold/warm context switches
within a single host (model weights stay in VRAM cache across runs of
the same workflow) and isolates each host's wall-clock budget.

Bloco 20 Sub-tarefa 1 ships the dataclass/CLI scaffold only — the
helper function bodies raise :class:`NotImplementedError`. Sub-tarefa 2
will implement orchestration using
:func:`installer.benchmark.runner._run_single`,
:func:`installer.benchmark.runner._spawn_snapshot_until_signal`,
:func:`installer.benchmark.runner._signal_snapshot_stop`, and
:func:`installer.benchmark.interface.ComfyUIClient.upload_image` for
WAN's :class:`LoadImage` dependency.

DA-007 dispatch SSH single-coordinator. DA-008 5-runs mechanic
(1 cold + 4 warm, discard min/max, mean/stddev/p50 of middle 3).
DA-011 OOM matrix hardcoded per Bloco 19a finding: cg-3060 8 GB
cannot fit FLUX fp16 or Qwen-Image fp8 even with offload.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from installer.benchmark import interface, runner

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Combinations known to OOM on the target GPU. Bloco 19a coverage matrix:
# 13/15 datapoints viable. cg-3060 (RTX 3060 8 GB) cannot hold FLUX fp16
# (~23.8 GiB UNet) nor Qwen-Image fp8 (~19 GiB UNet + ~9.4 GiB encoder)
# even with --offload — the deficit is too large.
OOM_SKIP_MATRIX: frozenset[tuple[str, str]] = frozenset({
    ("cg-3060", "flux_dev_fp16"),
    ("cg-3060", "qwen_image_fp8"),
})

# Canonical input image for WAN 2.2 i2v workflow. Sub-tarefa 2 uploads
# this once per host via :func:`interface.ComfyUIClient.upload_image`
# before running the WAN workflow there.
WAN_INPUT_PNG_LOCAL: Path = Path(
    "installer/benchmark/workflows/inputs/wan22_input.png"
)

# Workflow basename whose :class:`LoadImage` references the WAN input.
# Used by Sub-tarefa 2 to gate the upload to the right workflow.
WAN_WORKFLOW_BASENAME: str = "wan22_i2v_fp8"


# ============================================================================
# Exceptions
# ============================================================================

class SweepError(Exception):
    """Base class for all sweep module errors."""


class SweepConfigError(SweepError):
    """Raised when sweep CLI/configuration is invalid (e.g. workflows_dir empty)."""


class SweepHostError(SweepError):
    """Raised when a target host is unreachable or misconfigured at pre-flight."""


# ============================================================================
# Dataclasses (DA-008 + Bloco 20 spec)
# ============================================================================

@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Immutable configuration captured for the sweep summary header."""

    workflows_dir: Path
    hosts: tuple[str, ...]
    num_runs: int
    num_cold: int
    seed_base: int
    output_dir: Path
    skip_oom_combinations: bool


@dataclass(frozen=True, slots=True)
class SweepRunResult:
    """Outcome of one (host, workflow, run_idx) execution.

    Telemetry fields are populated when ``status == "success"``. On
    error or skip they are ``None`` and ``error_message`` carries the
    diagnostic. ``gpu_util_p50`` is the p50 of intra-run snapshot
    samples (not a cross-run aggregate); ``_aggregate_runs`` later
    aggregates these per-run p50s across the 5-run series.

    Bloco 22 additive: ``ram_peak_mib`` (host RAM peak, MiB) defaulted
    to ``None`` for backward compatibility with summaries written
    before the snapshot RAM-MiB telemetry expansion.
    """

    host: str
    workflow_name: str
    run_idx: int
    is_cold: bool
    wall_clock_s: float | None
    vram_peak_mib: float | None
    gpu_util_p50: float | None
    gpu_util_peak: float | None
    power_draw_avg_w: float | None
    status: str  # "success" | "error" | "skipped"
    error_message: str | None
    ram_peak_mib: float | None = None


@dataclass(frozen=True, slots=True)
class SweepAggregatedStats:
    """Aggregate of a 5-run series after DA-008 discard min+max of middle 3.

    All fields are ``None`` when aggregation is not possible (fewer
    than 5 successful runs for the (host, workflow) pair).
    """

    mean: float | None
    stddev: float | None
    p50: float | None
    included_indices: tuple[int, int, int] | None
    excluded_min_idx: int | None
    excluded_max_idx: int | None


@dataclass(frozen=True, slots=True)
class SweepWorkflowResult:
    """All 5 runs + aggregates for one (host, workflow) pair.

    ``power_draw_stats`` aggregates per-run :attr:`SweepRunResult.power_draw_avg_w`
    across the 5-run series; joules-per-image (a V1 efficiency metric)
    is derived downstream via ``power_draw_mean × wall_clock_mean``.
    """

    host: str
    workflow_name: str
    runs: tuple[SweepRunResult, ...]
    wall_clock_stats: SweepAggregatedStats
    vram_peak_stats: SweepAggregatedStats
    gpu_util_stats: SweepAggregatedStats
    power_draw_stats: SweepAggregatedStats


@dataclass(frozen=True, slots=True)
class SweepSkipEntry:
    """Record of a (host, workflow) pair skipped without execution."""

    host: str
    workflow: str
    reason: str


@dataclass(frozen=True, slots=True)
class SweepSummary:
    """Consolidated sweep output (single JSON file at sweep end)."""

    schema_version: int
    sweep_id: str
    config: dict[str, Any]
    results: tuple[SweepWorkflowResult, ...]
    skipped: tuple[SweepSkipEntry, ...]


# ============================================================================
# Helpers (stubs — Sub-tarefa 2 implements)
# ============================================================================

_DRY_RUN_BASENAME_RE = re.compile(r".*_dry_run\.json$")


def _discover_workflows(workflows_dir: Path) -> list[Path]:
    """Discover workflow JSONs at ``workflows_dir``, excluding ``*_dry_run.json``.

    Returns a deterministic order (sorted by basename) so the sweep
    matrix is reproducible across invocations.

    Raises:
        SweepConfigError: ``workflows_dir`` does not exist, is not a
            directory, or no production workflow JSONs were found
            after the dry-run filter.
    """
    if not workflows_dir.is_dir():
        raise SweepConfigError(
            f"workflows_dir does not exist or is not a directory: {workflows_dir}"
        )
    candidates = sorted(workflows_dir.glob("*.json"))
    workflows = [p for p in candidates if not _DRY_RUN_BASENAME_RE.match(p.name)]
    if not workflows:
        raise SweepConfigError(
            f"no production workflow JSONs found in {workflows_dir} "
            f"(found {len(candidates)} candidates, all filtered as *_dry_run.json)"
        )
    return workflows


def _should_skip(host: str, workflow_name: str) -> bool:
    """Return ``True`` if ``(host, workflow_name)`` is in :data:`OOM_SKIP_MATRIX`.

    ``workflow_name`` is the basename without the ``.json`` extension
    (e.g. ``"flux_dev_fp16"``).
    """
    return (host, workflow_name) in OOM_SKIP_MATRIX


# Default workflow timeout per run (seconds). Sized for WAN dual-expert
# 14B i2v cold (~12min observed in Bloco 18 Sub-tarefa 3); generous margin
# accommodates contention/cold-load variance on cg-3060.
WORKFLOW_TIMEOUT_S: int = 1800


def _terminate_snapshot_silently(
    host: str,
    snapshot_proc: Any,
) -> None:
    """Best-effort cleanup of a snapshot subprocess after a failed run.

    Called from error branches: signal stop, wait briefly, hard-kill on
    timeout. Errors are logged at debug level and never re-raised so the
    error branch can still return the original failure as a
    :class:`SweepRunResult`.
    """
    if snapshot_proc is None or snapshot_proc.poll() is not None:
        return
    try:
        runner._signal_snapshot_stop(host)
        snapshot_proc.communicate(timeout=10)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.debug("snapshot cleanup on %s failed: %r — killing", host, exc)
        with contextlib.suppress(Exception):
            snapshot_proc.kill()


def _execute_single_run(
    host: str,
    client: interface.ComfyUIClient,
    workflow_path: Path,
    run_idx: int,
    is_cold: bool,
    seed_base: int,
    output_dir: Path,
) -> SweepRunResult:
    """Execute one (host, workflow, run_idx) by orchestrating the runner.

    Inner sequence:
        1. Spawn :func:`runner._spawn_snapshot_until_signal` (snapshot.py
           --until-signal mode on the executor).
        2. :func:`runner._run_single` with ``ckpt_filename=None`` (skips
           the SDXL-only sanity check so UNETLoader workflows pass) and
           ``seed = seed_base + run_idx``.
        3. ``runner._run_single`` itself calls
           :func:`runner._signal_snapshot_stop` and parses snapshot
           stdout into ``RunResult.snapshot``.
        4. :func:`runner._download_outputs` saves binaries into the
           run directory.
        5. Extract metrics into :class:`SweepRunResult`.

    Failure handling (spec #5 — skip-and-continue per run):

    - :class:`runner.RunnerError` (including ``RunnerSSHError``,
      ``RunnerWorkflowError``) and :class:`interface.ComfyUIError` are
      captured and converted to ``status="error"``.
    - Any other exception is logged at WARN and converted to
      ``status="error"`` with ``error_message`` prefixed
      ``"unexpected: ..."``.

    Snapshot mapping (V1):

    - ``wall_clock_s`` ← :attr:`runner.RunResult.wallclock_seconds`
    - ``vram_peak_mib`` ← ``snapshot["peak_vram_mb"]``
    - ``gpu_util_p50`` ← ``snapshot["gpu_avg_utilization_pct"]`` (V1
      proxy; snapshot.py exposes only avg, not p50 — débito V2)
    - ``gpu_util_peak`` ← ``None`` (V1 snapshot does not track peak
      utilization — débito V2)
    - ``power_draw_avg_w`` ← ``snapshot["gpu_avg_power_w"]``
    """
    workflow_name = workflow_path.stem
    seed = seed_base + run_idx
    run_dir = output_dir / host / workflow_name / f"run_{run_idx}"

    def _error_result(msg: str) -> SweepRunResult:
        return SweepRunResult(
            host=host,
            workflow_name=workflow_name,
            run_idx=run_idx,
            is_cold=is_cold,
            wall_clock_s=None,
            vram_peak_mib=None,
            gpu_util_p50=None,
            gpu_util_peak=None,
            power_draw_avg_w=None,
            status="error",
            error_message=msg,
        )

    snapshot_proc: Any = None
    try:
        snapshot_proc = runner._spawn_snapshot_until_signal(host)
        run_result = runner._run_single(
            client=client,
            workflow_path=workflow_path,
            ckpt_filename=None,
            seed=seed,
            snapshot_proc=snapshot_proc,
            stop_flag_remote_path=runner.REMOTE_STOP_FLAG_DEFAULT,
            host=host,
            workflow_timeout=WORKFLOW_TIMEOUT_S,
        )
    except (runner.RunnerError, interface.ComfyUIError) as exc:
        _terminate_snapshot_silently(host, snapshot_proc)
        return _error_result(f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 — last-resort skip-and-continue
        _terminate_snapshot_silently(host, snapshot_proc)
        logger.warning(
            "unexpected exception in %s/%s run %d: %r",
            host, workflow_name, run_idx, exc,
        )
        return _error_result(f"unexpected: {type(exc).__name__}: {exc}")

    # Success path — download binaries (best-effort, telemetry preserved
    # if download fails) then extract metrics.
    try:
        runner._download_outputs(client, run_result.outputs, run_dir)
    except Exception as exc:  # noqa: BLE001 — telemetry is the primary signal
        logger.warning(
            "output download failed for %s/%s run %d: %r — telemetry preserved",
            host, workflow_name, run_idx, exc,
        )

    snap = run_result.snapshot
    peak_vram = snap.get("peak_vram_mb")
    avg_util = snap.get("gpu_avg_utilization_pct")
    avg_power = snap.get("gpu_avg_power_w")
    ram_peak = snap.get("ram_peak_mib")

    return SweepRunResult(
        host=host,
        workflow_name=workflow_name,
        run_idx=run_idx,
        is_cold=is_cold,
        wall_clock_s=run_result.wallclock_seconds,
        vram_peak_mib=float(peak_vram) if peak_vram is not None else None,
        gpu_util_p50=float(avg_util) if avg_util is not None else None,
        gpu_util_peak=None,
        power_draw_avg_w=float(avg_power) if avg_power is not None else None,
        status="success",
        error_message=None,
        ram_peak_mib=float(ram_peak) if ram_peak is not None else None,
    )


_EMPTY_STATS = SweepAggregatedStats(
    mean=None,
    stddev=None,
    p50=None,
    included_indices=None,
    excluded_min_idx=None,
    excluded_max_idx=None,
)


def _aggregate_runs(
    runs: tuple[SweepRunResult, ...],
    metric_extractor: Callable[[SweepRunResult], float | None],
) -> SweepAggregatedStats:
    """Apply DA-008: discard min+max, ``mean``/``stddev``/``p50`` of middle 3.

    Mirrors :func:`runner._aggregate_stats` semantics but operates on
    :class:`SweepRunResult` via a callback, enabling aggregation of
    wall-clock, VRAM peak, GPU utilization, etc. with a single function.

    Only runs with ``status == "success"`` and a non-``None`` value
    from ``metric_extractor`` are considered. If exactly 5 successful
    samples are found, discards the min and max (by value) and
    reports ``mean``/``stddev``/``p50`` of the remaining 3. Any other
    count returns an all-``None`` :class:`SweepAggregatedStats`.

    ``included_indices`` are the original run indices (0-based,
    sorted ascending) of the 3 retained runs; ``excluded_{min,max}_idx``
    are the original indices of the discarded runs.
    """
    samples: list[tuple[int, float]] = []
    for run in runs:
        if run.status != "success":
            continue
        value = metric_extractor(run)
        if value is None:
            continue
        samples.append((run.run_idx, value))

    if len(samples) != 5:
        return _EMPTY_STATS

    sorted_by_value = sorted(samples, key=lambda pair: pair[1])
    excluded_min_idx = sorted_by_value[0][0]
    excluded_max_idx = sorted_by_value[-1][0]
    middle = sorted_by_value[1:-1]
    middle_values = [v for _, v in middle]
    middle_indices = sorted(idx for idx, _ in middle)

    return SweepAggregatedStats(
        mean=statistics.mean(middle_values),
        stddev=statistics.stdev(middle_values),
        p50=statistics.median(middle_values),
        included_indices=(middle_indices[0], middle_indices[1], middle_indices[2]),
        excluded_min_idx=excluded_min_idx,
        excluded_max_idx=excluded_max_idx,
    )


def _save_summary(summary: SweepSummary, output_path: Path) -> None:
    """Serialize :class:`SweepSummary` as pretty JSON (indent=2, UTF-8).

    Creates parent directories as needed. Mirrors
    :func:`runner._save_summary` / :func:`installer._save_summary` for
    consistency across the benchmark suite.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        # default=str coerces Path (in SweepConfig) and any non-JSON-native
        # primitive into its repr — robust against schema additions.
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    logger.info("saved sweep summary to %s", output_path)


# ============================================================================
# CLI
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for :func:`main` (ships full args in Sub-tarefa 1)."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sweep orchestrator. Iterates the V1 matrix "
            "(workflows × hosts × runs) in host-major order (host A all "
            "workflows × K runs, then host B, then C). Skips known-OOM "
            "combos per the hardcoded matrix. Delegates inner-loop "
            "execution (snapshot + workflow + history poll) to runner.py."
        ),
    )
    parser.add_argument(
        "--workflows-dir",
        default="installer/benchmark/workflows",
        help=(
            "Directory containing workflow JSONs "
            "(default: installer/benchmark/workflows)."
        ),
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        required=True,
        help="One or more SSH host aliases (e.g. cg-3060 cg-4090 cg-5090).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Total runs per (host, workflow). Default 5 (DA-008).",
    )
    parser.add_argument(
        "--num-cold",
        type=int,
        default=1,
        help=(
            "Number of leading runs treated as cold (info-only — the first "
            "run is always semantically cold). Default 1."
        ),
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help=(
            "Base seed; effective seed per run is seed_base + run_idx "
            "(débito #5; default 42)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./sweep_outputs/<UTC-timestamp>).",
    )
    parser.add_argument(
        "--skip-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip combinations in the hardcoded OOM matrix (default: True).",
    )
    parser.add_argument(
        "--workflow-filter",
        default=None,
        help="Optional regex; only workflows whose basename matches are included.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything.",
    )
    return parser


def _print_dry_run_plan(
    sweep_id: str,
    hosts: tuple[str, ...],
    workflows: list[Path],
    unhealthy_hosts: set[str],
    skip_oom: bool,
) -> None:
    """Print the execution plan as a tabular preview (PARTE 7 spec)."""
    print(f"\n=== SWEEP PLAN (--dry-run, sweep_id={sweep_id}) ===")
    print(f"{'host':<10} {'workflow':<22} action")
    print(f"{'-' * 10} {'-' * 22} {'-' * 16}")
    for host in hosts:
        for wf in workflows:
            if host in unhealthy_hosts:
                action = "skip_unhealthy"
            elif skip_oom and _should_skip(host, wf.stem):
                action = "skip_oom"
            else:
                action = "execute"
            print(f"{host:<10} {wf.stem:<22} {action}")


def main() -> None:
    """End-to-end sweep orchestrator (Bloco 20 Sub-tarefa 2b).

    Sequence:
        1. Parse args; resolve output_dir / sweep_id.
        2. Discover + optionally filter workflows.
        3. Pre-flight health-check each host (instantiates a
           :class:`interface.ComfyUIClient` per host, ``is_alive``).
        4. If ``--dry-run``: print plan, exit.
        5. Host-major sweep loop. Per host: best-effort git pull, then
           iterate workflows. For WAN workflows, upload the canonical
           input PNG once per host (cached in ``wan_uploaded`` set).
           Run each (host, workflow) 5 times via
           :func:`_execute_single_run`. Aggregate per metric. Prune
           PNG/WEBP binaries from runs 1..N-1 (keep run_0 sample).
        6. Build :class:`SweepSummary`; save consolidated JSON.
        7. Print tabular summary to stderr.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    parser = _build_argparser()
    args = parser.parse_args()

    sweep_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workflows_dir = Path(args.workflows_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(f"sweep_outputs/{sweep_id}")
    )

    config = SweepConfig(
        workflows_dir=workflows_dir,
        hosts=tuple(args.hosts),
        num_runs=args.num_runs,
        num_cold=args.num_cold,
        seed_base=args.seed_base,
        output_dir=output_dir,
        skip_oom_combinations=args.skip_oom,
    )

    workflows = _discover_workflows(workflows_dir)
    if args.workflow_filter:
        regex = re.compile(args.workflow_filter)
        workflows = [w for w in workflows if regex.search(w.stem)]
        if not workflows:
            raise SweepConfigError(
                f"--workflow-filter {args.workflow_filter!r} matched 0 workflows"
            )

    logger.info(
        "sweep_id=%s workflows=%d hosts=%s num_runs=%d output_dir=%s",
        sweep_id, len(workflows), config.hosts, config.num_runs, output_dir,
    )

    # Dry-run gate: print plan and exit before any remote I/O (spec #7).
    if args.dry_run:
        _print_dry_run_plan(
            sweep_id=sweep_id,
            hosts=config.hosts,
            workflows=workflows,
            unhealthy_hosts=set(),
            skip_oom=config.skip_oom_combinations,
        )
        return

    # Pre-flight health checks. Short timeout so DOWN hosts fail fast.
    healthy_clients: dict[str, interface.ComfyUIClient] = {}
    unhealthy_hosts: set[str] = set()
    for host in config.hosts:
        url = f"http://{host}:8188"
        probe = interface.ComfyUIClient(url, timeout=5)
        alive = False
        try:
            alive = probe.is_alive()
        except Exception as exc:  # noqa: BLE001
            logger.warning("host %s pre-flight raised: %r", host, exc)
        if alive:
            healthy_clients[host] = interface.ComfyUIClient(url, timeout=60)
            logger.info("  [PRE-FLIGHT] %s OK", host)
        else:
            unhealthy_hosts.add(host)
            logger.warning("  [PRE-FLIGHT] %s UNHEALTHY", host)

    results: list[SweepWorkflowResult] = []
    skipped: list[SweepSkipEntry] = []
    wan_uploaded: set[str] = set()

    for host in config.hosts:
        if host in unhealthy_hosts:
            for wf in workflows:
                skipped.append(SweepSkipEntry(
                    host=host,
                    workflow=wf.stem,
                    reason="host_unhealthy_pre_flight",
                ))
            continue

        client = healthy_clients[host]
        logger.info("=== host %s ===", host)

        # Best-effort pull. Keep going on failure — the executor may
        # already be up-to-date from a prior manual sync.
        try:
            pull_out = runner._ssh_pull(host)
            first_line = pull_out.strip().splitlines()[0] if pull_out.strip() else "<empty>"
            logger.info("[%s] git pull: %s", host, first_line)
        except runner.RunnerSSHError as exc:
            logger.warning("[%s] git pull failed (continuing): %r", host, exc)

        for wf_path in workflows:
            wf_name = wf_path.stem
            if config.skip_oom_combinations and _should_skip(host, wf_name):
                skipped.append(SweepSkipEntry(
                    host=host, workflow=wf_name, reason="oom_matrix",
                ))
                logger.info("[%s] %s: skipped (OOM matrix)", host, wf_name)
                continue

            # WAN canonical input upload (once per host).
            if wf_name == WAN_WORKFLOW_BASENAME and host not in wan_uploaded:
                try:
                    canonical = client.upload_image(WAN_INPUT_PNG_LOCAL)
                    wan_uploaded.add(host)
                    logger.info("[%s] uploaded WAN input as %r", host, canonical)
                except interface.ComfyUIError as exc:
                    logger.error(
                        "[%s] WAN input upload failed: %r — skipping WAN",
                        host, exc,
                    )
                    skipped.append(SweepSkipEntry(
                        host=host, workflow=wf_name,
                        reason=f"wan_upload_failed: {exc}",
                    ))
                    continue

            logger.info(
                "[%s] %s: running %d runs (cold=run_0, warm=run_1..N-1)",
                host, wf_name, config.num_runs,
            )
            runs: list[SweepRunResult] = []
            for run_idx in range(config.num_runs):
                result = _execute_single_run(
                    host=host,
                    client=client,
                    workflow_path=wf_path,
                    run_idx=run_idx,
                    is_cold=(run_idx == 0),
                    seed_base=config.seed_base,
                    output_dir=output_dir,
                )
                runs.append(result)
                if result.status == "success":
                    logger.info(
                        "  run %d: OK wallclock=%.2fs vram_peak=%.0fMiB",
                        run_idx,
                        result.wall_clock_s or 0.0,
                        result.vram_peak_mib or 0.0,
                    )
                else:
                    logger.warning(
                        "  run %d: ERROR %s", run_idx, result.error_message,
                    )

            runs_t = tuple(runs)
            ws = _aggregate_runs(runs_t, lambda r: r.wall_clock_s)
            vs = _aggregate_runs(runs_t, lambda r: r.vram_peak_mib)
            gs = _aggregate_runs(runs_t, lambda r: r.gpu_util_p50)
            ps = _aggregate_runs(runs_t, lambda r: r.power_draw_avg_w)

            # Output retention: keep run_0 (cold) sample, prune binaries
            # in run_1..N-1 (telemetry JSON, if any, is preserved).
            for run_idx in range(1, config.num_runs):
                run_dir = output_dir / host / wf_name / f"run_{run_idx}"
                if not run_dir.is_dir():
                    continue
                for binary in list(run_dir.glob("*.png")):
                    binary.unlink()
                for binary in list(run_dir.glob("*.webp")):
                    binary.unlink()

            results.append(SweepWorkflowResult(
                host=host,
                workflow_name=wf_name,
                runs=runs_t,
                wall_clock_stats=ws,
                vram_peak_stats=vs,
                gpu_util_stats=gs,
                power_draw_stats=ps,
            ))

    summary = SweepSummary(
        schema_version=1,
        sweep_id=sweep_id,
        config=asdict(config),
        results=tuple(results),
        skipped=tuple(skipped),
    )
    _save_summary(summary, output_dir / "summary.json")

    total_runs = sum(len(r.runs) for r in results)
    success_runs = sum(
        1 for r in results for run in r.runs if run.status == "success"
    )
    error_runs = sum(
        1 for r in results for run in r.runs if run.status == "error"
    )
    print(f"\n=== SWEEP DONE (sweep_id={sweep_id}) ===", file=sys.stderr)
    print(
        f"hosts: {len(config.hosts)} ({len(unhealthy_hosts)} unhealthy)",
        file=sys.stderr,
    )
    print(f"workflows discovered: {len(workflows)}", file=sys.stderr)
    print(
        f"results: {len(results)} (host, workflow) pairs", file=sys.stderr,
    )
    print(
        f"runs: {success_runs} success / {error_runs} error / {total_runs} total",
        file=sys.stderr,
    )
    print(f"skipped: {len(skipped)}", file=sys.stderr)
    print(f"output: {output_dir / 'summary.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
