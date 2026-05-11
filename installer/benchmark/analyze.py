"""Benchmark analysis toolchain — ingest sweep summaries, render reports.

Single responsibility: read one or more
``sweep_outputs/<sweep_id>/summary.json`` files produced by
:mod:`installer.benchmark.sweep`, consolidate them into a single
matrix, and emit a self-contained report under ``reports/<timestamp>/``
with:

- ``summary.md`` — markdown narrative (tables, speedup commentary)
- ``data.json`` — :class:`ConsolidatedMatrix` serialized
- ``charts/`` — 6 canonical PNG charts (matplotlib static)

The dataclasses :class:`SweepSummary`, :class:`SweepWorkflowResult`,
:class:`SweepRunResult`, and :class:`SweepAggregatedStats` are
imported directly from :mod:`installer.benchmark.sweep` — single
source of truth for schema. This module never mutates the source
summary files; it derives :class:`ConsolidatedMatrix` and writes
only to the output directory.

Bloco 21 Sub-tarefa 1 ships the dataclass/CLI scaffold only — helper
function bodies raise :class:`NotImplementedError`. Sub-tarefa 2a
implements ingestion + matrix building + speedup computation;
Sub-tarefa 2b implements the 6 matplotlib charts; Sub-tarefa 3
implements markdown rendering + data JSON serialization + ``main``.

DA-008 mechanic (mean/stddev/p50 of middle 3 from sweep aggregation)
is preserved at the matrix layer — ``wall_clock_mean`` etc come
directly from :attr:`SweepWorkflowResult.wall_clock_stats.mean`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from installer.benchmark.sweep import (
    SweepAggregatedStats,
    SweepRunResult,
    SweepSkipEntry,
    SweepSummary,
    SweepWorkflowResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

CHART_NAMES: frozenset[str] = frozenset({
    "speedup_per_workflow_bars",
    "speedup_matrix_heatmap",
    "fp16_vs_fp8_cg5090_highlight",
    "power_scaling_per_workflow",
    "vram_peak_per_workflow",
    "cold_warm_separation",
})

DEFAULT_REFERENCE_HOST: str = "cg-3060"

# Canonical V1 ordering for plot axes. Hosts left-to-right reflect
# generation (3060→4090→5090); workflows left-to-right reflect rough
# size/complexity ascending. Module-level so rendering is consistent
# across charts.
HOST_ORDER: tuple[str, ...] = ("cg-3060", "cg-4090", "cg-5090")
HOST_COLORS: dict[str, str] = {
    "cg-3060": "#1f77b4",  # matplotlib blue
    "cg-4090": "#ff7f0e",  # matplotlib orange
    "cg-5090": "#2ca02c",  # matplotlib green
}
# Physical VRAM ceilings (GiB) per host — drawn as horizontal dashed
# reference lines on the VRAM chart. cg-3060 = RTX 3060 12 GB variant.
HOST_VRAM_CEILING_GIB: dict[str, float] = {
    "cg-3060": 12.0,
    "cg-4090": 24.0,
    "cg-5090": 32.0,
}

WORKFLOW_ORDER: tuple[str, ...] = (
    "sdxl_base",
    "flux_dev_fp8",
    "flux_dev_fp16",
    "qwen_image_fp8",
    "wan22_i2v_fp8",
)

DEFAULT_DPI: int = 150

# Status literal mirrors the four observed outcomes in V1 sweeps:
# - "success": all 5 runs OK; aggregated stats populated.
# - "skip_oom": preempted by OOM_SKIP_MATRIX before any run.
# - "error": at least one run raised RunnerError / ComfyUIError before
#   completion; aggregated stats may be None.
# - "timeout": run exceeded workflow_timeout (special-cased here from
#   "error" when the error_message contains "ComfyUITimeoutError" so
#   reports can distinguish throughput-bound from architectural failures).
CellStatus = Literal["success", "skip_oom", "error", "timeout"]

# Chart palette for non-success cells. Used by heatmap to overlay
# status string when ratio is unavailable.
STATUS_COLORS: dict[CellStatus, str] = {
    "success": "#2ca02c",  # green
    "skip_oom": "#7f7f7f",  # gray
    "error": "#d62728",    # red
    "timeout": "#ff7f0e",  # orange
}


# ============================================================================
# Exceptions
# ============================================================================

class AnalysisError(Exception):
    """Base class for all analyze module errors."""


class AnalysisConfigError(AnalysisError):
    """Raised when the analyze CLI / configuration is invalid."""


class AnalysisRenderError(AnalysisError):
    """Raised when a chart or markdown render step fails."""


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Immutable configuration captured at CLI parse time."""

    summary_glob: str
    output_dir: Path
    workflow_filter: str | None
    host_filter: str | None
    reference_host: str
    include_charts: bool


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One (host, workflow) cell in the consolidated matrix.

    When ``status == "success"`` the four ``*_mean`` fields carry the
    DA-008 aggregated mean (middle 3 of the 5-run series); otherwise
    they are ``None`` and ``error_message`` may carry diagnostic text.
    """

    host: str
    workflow: str
    status: CellStatus
    wall_clock_mean: float | None
    vram_peak_mean: float | None
    gpu_util_mean: float | None
    power_draw_mean: float | None
    error_count: int
    error_message: str | None


@dataclass(frozen=True, slots=True)
class SpeedupCell:
    """Cross-host speedup datapoint for a single workflow.

    ``speedup_ratio = reference_wall_clock_s / target_wall_clock_s``
    (i.e. >1.0 means target is faster than reference). Returns
    ``None`` when either side is missing or non-positive.
    """

    workflow: str
    reference_host: str
    target_host: str
    speedup_ratio: float | None
    reference_wall_clock_s: float | None
    target_wall_clock_s: float | None


@dataclass(frozen=True, slots=True)
class ConsolidatedMatrix:
    """Consolidated view across one or more sweep summaries."""

    schema_version: int
    generated_at: str
    source_sweep_ids: tuple[str, ...]
    cells: tuple[MatrixCell, ...]
    speedups: tuple[SpeedupCell, ...]


# ============================================================================
# Ingestion + matrix building (Sub-tarefa 2a)
# ============================================================================

_TIMEOUT_PATTERN = re.compile(r"ComfyUITimeoutError")


def _dict_to_aggregated_stats(d: dict[str, Any]) -> SweepAggregatedStats:
    """Reconstruct :class:`SweepAggregatedStats` from a JSON-decoded dict.

    JSON arrays become Python lists; the schema declares
    ``included_indices`` as a tuple, so we coerce here. Missing keys
    default to ``None`` (additive-evolution safe).
    """
    included = d.get("included_indices")
    if included is not None:
        included_tuple = (int(included[0]), int(included[1]), int(included[2]))
    else:
        included_tuple = None
    return SweepAggregatedStats(
        mean=d.get("mean"),
        stddev=d.get("stddev"),
        p50=d.get("p50"),
        included_indices=included_tuple,
        excluded_min_idx=d.get("excluded_min_idx"),
        excluded_max_idx=d.get("excluded_max_idx"),
    )


_EMPTY_AGGREGATED_STATS = SweepAggregatedStats(
    mean=None,
    stddev=None,
    p50=None,
    included_indices=None,
    excluded_min_idx=None,
    excluded_max_idx=None,
)


def _dict_to_run_result(d: dict[str, Any]) -> SweepRunResult:
    """Reconstruct :class:`SweepRunResult` from a JSON-decoded dict."""
    return SweepRunResult(
        host=d["host"],
        workflow_name=d["workflow_name"],
        run_idx=d["run_idx"],
        is_cold=d["is_cold"],
        wall_clock_s=d.get("wall_clock_s"),
        vram_peak_mib=d.get("vram_peak_mib"),
        gpu_util_p50=d.get("gpu_util_p50"),
        gpu_util_peak=d.get("gpu_util_peak"),
        power_draw_avg_w=d.get("power_draw_avg_w"),
        status=d["status"],
        error_message=d.get("error_message"),
    )


def _dict_to_workflow_result(d: dict[str, Any]) -> SweepWorkflowResult:
    """Reconstruct :class:`SweepWorkflowResult` from a JSON-decoded dict.

    Missing aggregated-stats blocks (e.g. ``power_draw_stats`` in
    pre-Bloco 20 summaries) default to ``_EMPTY_AGGREGATED_STATS``
    rather than raising — additive-evolution safe.
    """
    runs = tuple(_dict_to_run_result(r) for r in d.get("runs", []))

    def _stats(key: str) -> SweepAggregatedStats:
        block = d.get(key)
        if not isinstance(block, dict):
            return _EMPTY_AGGREGATED_STATS
        return _dict_to_aggregated_stats(block)

    return SweepWorkflowResult(
        host=d["host"],
        workflow_name=d["workflow_name"],
        runs=runs,
        wall_clock_stats=_stats("wall_clock_stats"),
        vram_peak_stats=_stats("vram_peak_stats"),
        gpu_util_stats=_stats("gpu_util_stats"),
        power_draw_stats=_stats("power_draw_stats"),
    )


def _dict_to_skip_entry(d: dict[str, Any]) -> SweepSkipEntry:
    """Reconstruct :class:`SweepSkipEntry` from a JSON-decoded dict."""
    return SweepSkipEntry(
        host=d["host"],
        workflow=d["workflow"],
        reason=d["reason"],
    )


def _discover_summaries(glob_pattern: str) -> list[Path]:
    """Discover ``summary.json`` files matching ``glob_pattern``.

    Pattern is resolved relative to :func:`Path.cwd`. Returns a
    deterministic order (sorted by path) so report generation is
    reproducible across invocations. Empty result raises
    :class:`AnalysisConfigError` — at least one summary must exist
    to produce a report.
    """
    base = Path.cwd()
    candidates = sorted(base.glob(glob_pattern))
    summaries = [p for p in candidates if p.is_file()]
    if not summaries:
        raise AnalysisConfigError(
            f"no summary files found matching {glob_pattern!r} (cwd={base})"
        )
    return summaries


def _load_summary(path: Path) -> SweepSummary:
    """Load and parse one ``summary.json`` into a :class:`SweepSummary`.

    Field-by-field reconstruction via ``_dict_to_*`` helpers. Validates
    that ``schema_version == 1`` and that the top-level is a JSON
    object; otherwise raises :class:`AnalysisConfigError` with the
    path + detected version for diagnostic clarity.
    """
    try:
        raw_obj: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisConfigError(
            f"summary {path}: invalid JSON: {exc}"
        ) from exc
    if not isinstance(raw_obj, dict):
        raise AnalysisConfigError(
            f"summary {path}: top-level is not a JSON object "
            f"(got {type(raw_obj).__name__})"
        )
    raw: dict[str, Any] = raw_obj
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise AnalysisConfigError(
            f"summary {path}: schema_version={schema_version!r}, expected 1"
        )
    return SweepSummary(
        schema_version=1,
        sweep_id=raw["sweep_id"],
        config=raw.get("config", {}),
        results=tuple(
            _dict_to_workflow_result(r) for r in raw.get("results", [])
        ),
        skipped=tuple(
            _dict_to_skip_entry(e) for e in raw.get("skipped", [])
        ),
    )


_OUTCOME_PRECEDENCE: tuple[CellStatus, ...] = (
    "success",
    "timeout",
    "error",
    "skip_oom",
)


def _classify_workflow_result(r: SweepWorkflowResult) -> CellStatus:
    """Per-summary classification of one :class:`SweepWorkflowResult`.

    Returns the outcome for the (host, workflow) pair as observed
    *within a single summary* — global best-outcome dedup happens
    later in :func:`_build_matrix`. ``timeout`` is returned only
    when every error run carries a ``ComfyUITimeoutError`` message
    (matching :data:`_TIMEOUT_PATTERN`) — surfaces throughput-bound
    failures distinctly from architectural ones (e.g. the
    pre-Bloco 20-Sub-tarefa-5 KSampler bug).
    """
    if r.wall_clock_stats.mean is not None:
        return "success"
    error_runs = [run for run in r.runs if run.status == "error"]
    if error_runs and all(
        run.error_message is not None
        and bool(_TIMEOUT_PATTERN.search(run.error_message))
        for run in error_runs
    ):
        return "timeout"
    return "error"


def _build_matrix(summaries: Iterable[SweepSummary]) -> ConsolidatedMatrix:
    """Consolidate one or more summaries into a :class:`ConsolidatedMatrix`.

    Deduplication strategy is **best-outcome-wins** per
    (host, workflow), with precedence:
    ``success`` > ``timeout`` > ``error`` > ``skip_oom``.

    Each summary independently classifies its (host, workflow)
    via :func:`_classify_workflow_result`; the consolidated cell
    picks the highest-precedence outcome across summaries. When
    multiple ``success`` results exist for the same pair, the
    lexicographically greater ``sweep_id`` wins (sweep_ids are UTC
    timestamps in ``YYYYMMDDTHHMMSSZ`` form, so this corresponds
    to the most recent run). ``error_count`` aggregates all
    ``status="error"`` runs across all summaries for the cell.
    """
    materialized = list(summaries)

    success_by_key: dict[tuple[str, str], tuple[str, SweepWorkflowResult]] = {}
    error_runs_by_key: dict[tuple[str, str], list[SweepRunResult]] = {}
    outcomes_by_key: dict[tuple[str, str], set[CellStatus]] = {}
    # PARTE 0 alignment fix: track the first non-empty error message
    # per (key, classified-outcome). When building the cell, pick the
    # message that matches the winning status — surfacing
    # "ComfyUITimeoutError" for the timeout cell even when an earlier
    # summary (under a different classification) contributed
    # different error text first.
    msg_by_key_outcome: dict[tuple[str, str], dict[CellStatus, str]] = {}

    for summary in materialized:
        for r in summary.results:
            key = (r.host, r.workflow_name)
            outcome = _classify_workflow_result(r)
            outcomes_by_key.setdefault(key, set()).add(outcome)
            if outcome == "success":
                prior = success_by_key.get(key)
                if prior is None or summary.sweep_id > prior[0]:
                    success_by_key[key] = (summary.sweep_id, r)
            else:
                error_runs_iter = [run for run in r.runs if run.status == "error"]
                first_msg = next(
                    (run.error_message for run in error_runs_iter
                     if run.error_message),
                    None,
                )
                if first_msg is not None:
                    per_outcome = msg_by_key_outcome.setdefault(key, {})
                    per_outcome.setdefault(outcome, first_msg)
                for run in error_runs_iter:
                    error_runs_by_key.setdefault(key, []).append(run)
        for e in summary.skipped:
            skip_key = (e.host, e.workflow)
            outcomes_by_key.setdefault(skip_key, set()).add("skip_oom")

    cells: list[MatrixCell] = []
    for key in sorted(outcomes_by_key):
        host, workflow = key
        outcomes = outcomes_by_key[key]
        best: CellStatus = next(o for o in _OUTCOME_PRECEDENCE if o in outcomes)

        error_runs = error_runs_by_key.get(key, [])
        error_count = len(error_runs)

        if best == "success":
            _, r = success_by_key[key]
            cells.append(MatrixCell(
                host=host,
                workflow=workflow,
                status="success",
                wall_clock_mean=r.wall_clock_stats.mean,
                vram_peak_mean=r.vram_peak_stats.mean,
                gpu_util_mean=r.gpu_util_stats.mean,
                power_draw_mean=r.power_draw_stats.mean,
                error_count=error_count,
                error_message=None,
            ))
        elif best in ("timeout", "error"):
            aligned_msg = msg_by_key_outcome.get(key, {}).get(best)
            cells.append(MatrixCell(
                host=host,
                workflow=workflow,
                status=best,
                wall_clock_mean=None,
                vram_peak_mean=None,
                gpu_util_mean=None,
                power_draw_mean=None,
                error_count=error_count,
                error_message=aligned_msg,
            ))
        else:  # skip_oom
            cells.append(MatrixCell(
                host=host,
                workflow=workflow,
                status="skip_oom",
                wall_clock_mean=None,
                vram_peak_mean=None,
                gpu_util_mean=None,
                power_draw_mean=None,
                error_count=0,
                error_message=None,
            ))

    source_sweep_ids = tuple(sorted({s.sweep_id for s in materialized}))
    return ConsolidatedMatrix(
        schema_version=1,
        generated_at=datetime.now(UTC).isoformat(),
        source_sweep_ids=source_sweep_ids,
        cells=tuple(cells),
        speedups=(),
    )


def _compute_speedups(
    matrix: ConsolidatedMatrix,
    reference_host: str,
) -> tuple[SpeedupCell, ...]:
    """Compute ``reference/target`` wall-clock ratios per workflow.

    For each workflow in :attr:`matrix.cells`, looks up the
    ``reference_host`` success cell and emits one
    :class:`SpeedupCell` per non-reference host. ``speedup_ratio``
    is ``None`` whenever either side is missing, non-success, or
    has a non-positive wall_clock value.

    Output is sorted by (workflow, target_host) for deterministic
    downstream rendering.
    """
    cell_lookup: dict[tuple[str, str], MatrixCell] = {
        (c.host, c.workflow): c for c in matrix.cells
    }
    workflows = sorted({c.workflow for c in matrix.cells})
    hosts = sorted({c.host for c in matrix.cells})
    target_hosts = [h for h in hosts if h != reference_host]

    speedups: list[SpeedupCell] = []
    for workflow in workflows:
        ref = cell_lookup.get((reference_host, workflow))
        ref_wall: float | None = (
            ref.wall_clock_mean
            if ref is not None and ref.status == "success"
            else None
        )
        for target_host in target_hosts:
            target = cell_lookup.get((target_host, workflow))
            target_wall: float | None = (
                target.wall_clock_mean
                if target is not None and target.status == "success"
                else None
            )
            if (
                ref_wall is not None
                and target_wall is not None
                and target_wall > 0
            ):
                ratio: float | None = ref_wall / target_wall
            else:
                ratio = None
            speedups.append(SpeedupCell(
                workflow=workflow,
                reference_host=reference_host,
                target_host=target_host,
                speedup_ratio=ratio,
                reference_wall_clock_s=ref_wall,
                target_wall_clock_s=target_wall,
            ))
    return tuple(speedups)


# ============================================================================
# Chart rendering (Sub-tarefa 2b — matplotlib static)
# ============================================================================

def _cell_lookup(matrix: ConsolidatedMatrix) -> dict[tuple[str, str], MatrixCell]:
    """Internal helper: index cells by ``(host, workflow)``."""
    return {(c.host, c.workflow): c for c in matrix.cells}


def _speedup_lookup(
    matrix: ConsolidatedMatrix,
) -> dict[tuple[str, str], SpeedupCell]:
    """Internal helper: index speedups by ``(workflow, target_host)``."""
    return {(s.workflow, s.target_host): s for s in matrix.speedups}


def _filter_workflows_present(matrix: ConsolidatedMatrix) -> list[str]:
    """Return the WORKFLOW_ORDER subset that appears in ``matrix.cells``."""
    seen = {c.workflow for c in matrix.cells}
    return [w for w in WORKFLOW_ORDER if w in seen]


def _filter_hosts_present(matrix: ConsolidatedMatrix) -> list[str]:
    """Return the HOST_ORDER subset that appears in ``matrix.cells``."""
    seen = {c.host for c in matrix.cells}
    return [h for h in HOST_ORDER if h in seen]


def _render_chart_speedup_bars(
    matrix: ConsolidatedMatrix,
    summaries: list[SweepSummary],
    output_path: Path,
) -> None:
    """Grouped bars: one cluster per workflow, 3 bars (one per host).

    Y-axis log-scaled (speedups span 1.2x–7.8x; log clarifies
    the smaller cg-5090/cg-4090 deltas). Cells whose ``reference``
    is non-success render all 3 bars as 0-height with "N/A" overlay.
    Bars labeled with their ratio in ``"X.XXx"`` form.
    """
    del summaries  # not used; signature uniform for batch render

    workflows = _filter_workflows_present(matrix)
    hosts = _filter_hosts_present(matrix)
    if not workflows or not hosts:
        raise AnalysisRenderError(
            "speedup_bars: no workflows/hosts available to render"
        )

    cells_idx = _cell_lookup(matrix)
    speedups_idx = _speedup_lookup(matrix)
    reference_host = matrix.speedups[0].reference_host if matrix.speedups else DEFAULT_REFERENCE_HOST

    n_groups = len(workflows)
    n_bars = len(hosts)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, host in enumerate(hosts):
        ratios: list[float] = []
        labels: list[str] = []
        for wf in workflows:
            if host == reference_host:
                ref_cell = cells_idx.get((reference_host, wf))
                if ref_cell is not None and ref_cell.status == "success":
                    ratios.append(1.0)
                    labels.append("1.00x")
                else:
                    ratios.append(0.0)
                    labels.append("N/A")
            else:
                sp = speedups_idx.get((wf, host))
                if sp is not None and sp.speedup_ratio is not None:
                    ratios.append(sp.speedup_ratio)
                    labels.append(f"{sp.speedup_ratio:.2f}x")
                else:
                    ratios.append(0.0)
                    labels.append("N/A")

        offsets = x + (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets, ratios, bar_width,
            label=host, color=HOST_COLORS.get(host, "#888888"),
            edgecolor="black", linewidth=0.5,
        )
        for bar, label in zip(bars, labels, strict=True):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(height, 1.0) * 1.05,
                label,
                ha="center", va="bottom", fontsize=8, rotation=0,
            )

    ax.set_yscale("log")
    ax.set_ylim(0.8, 12.0)
    ax.set_xticks(x)
    ax.set_xticklabels(workflows, rotation=15, ha="right")
    ax.set_ylabel(f"Speedup vs {reference_host} (log scale)")
    ax.set_title(
        f"V1 Speedup vs {reference_host} baseline "
        "(mean of DA-008 middle 3)"
    )
    ax.axhline(1.0, color="#666666", linestyle=":", linewidth=1, alpha=0.7)
    ax.grid(True, axis="y", color="#cccccc", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", title="GPU")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _render_chart_speedup_heatmap(
    matrix: ConsolidatedMatrix,
    summaries: list[SweepSummary],
    output_path: Path,
) -> None:
    """Heatmap: rows=workflows, cols=hosts, cell=speedup vs reference.

    LogNorm color scale clipped to [1.0, 8.0] keeps the 7.80×
    flux_dev_fp8 outlier visible without saturating smaller deltas.
    Non-success cells render as gray with the status overlaid
    (``skip_oom`` / ``error`` / ``timeout``).
    """
    del summaries

    workflows = _filter_workflows_present(matrix)
    hosts = _filter_hosts_present(matrix)
    if not workflows or not hosts:
        raise AnalysisRenderError(
            "speedup_heatmap: no workflows/hosts available to render"
        )

    cells_idx = _cell_lookup(matrix)
    speedups_idx = _speedup_lookup(matrix)
    reference_host = matrix.speedups[0].reference_host if matrix.speedups else DEFAULT_REFERENCE_HOST

    n_rows = len(workflows)
    n_cols = len(hosts)
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    overlay: list[list[str]] = [["" for _ in hosts] for _ in workflows]

    for i, wf in enumerate(workflows):
        for j, host in enumerate(hosts):
            cell = cells_idx.get((host, wf))
            if cell is None:
                overlay[i][j] = "—"
                continue
            if cell.status != "success":
                overlay[i][j] = cell.status
                continue
            if host == reference_host:
                grid[i, j] = 1.0
                overlay[i][j] = "1.00x"
            else:
                sp = speedups_idx.get((wf, host))
                if sp is not None and sp.speedup_ratio is not None:
                    grid[i, j] = sp.speedup_ratio
                    overlay[i][j] = f"{sp.speedup_ratio:.2f}x"
                else:
                    overlay[i][j] = "N/A"

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#bbbbbb")
    norm = LogNorm(vmin=1.0, vmax=8.0)
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(hosts)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(workflows)
    ax.set_xlabel("GPU host")
    ax.set_ylabel("Workflow")
    ax.set_title(
        f"Speedup Matrix (rows=workflows, cols=GPUs, baseline={reference_host})"
    )

    for i in range(n_rows):
        for j in range(n_cols):
            value = grid[i, j]
            if np.isnan(value):
                ax.text(
                    j, i, overlay[i][j],
                    ha="center", va="center",
                    color="black", fontsize=9,
                )
            else:
                # White text on green dark cells, black on light
                text_color = "white" if value > 4.0 else "black"
                ax.text(
                    j, i, overlay[i][j],
                    ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold",
                )

    cbar = fig.colorbar(im, ax=ax, label=f"Speedup vs {reference_host} (log)")
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _render_chart_fp16_vs_fp8(
    matrix: ConsolidatedMatrix,
    summaries: list[SweepSummary],
    output_path: Path,
) -> None:
    """Counterintuitive insight: fp16 native vs fp8 dequant on cg-5090.

    On RTX 5090, fp16 native (~8.64s) edges out fp8 dequant (~8.74s)
    by ~1.2% — small but reproducible across Bloco 18 + Bloco 20.
    Chart shows the side-by-side wall_clock_mean for both
    cg-4090 (sanity reference) and cg-5090 (highlight host).
    """
    del summaries

    cells_idx = _cell_lookup(matrix)
    hosts_compare = ("cg-4090", "cg-5090")

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(hosts_compare))
    width = 0.35

    fp8_values: list[float] = []
    fp16_values: list[float] = []
    for host in hosts_compare:
        fp8_cell = cells_idx.get((host, "flux_dev_fp8"))
        fp16_cell = cells_idx.get((host, "flux_dev_fp16"))
        fp8_values.append(
            fp8_cell.wall_clock_mean
            if fp8_cell is not None and fp8_cell.wall_clock_mean is not None
            else 0.0
        )
        fp16_values.append(
            fp16_cell.wall_clock_mean
            if fp16_cell is not None and fp16_cell.wall_clock_mean is not None
            else 0.0
        )

    bars_fp8 = ax.bar(x - width / 2, fp8_values, width, label="fp8 (dequant)",
                      color="#1f77b4", edgecolor="black")
    bars_fp16 = ax.bar(x + width / 2, fp16_values, width, label="fp16 (native)",
                       color="#ff7f0e", edgecolor="black")

    for bar in (*bars_fp8, *bars_fp16):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.15,
                f"{h:.2f}s", ha="center", va="bottom", fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(hosts_compare)
    ax.set_ylabel("Wall-clock (s, mean of DA-008 middle 3)")
    ax.set_title("FLUX dev fp16 native vs fp8 dequant — RTX 5090 counterintuitive insight")
    ax.legend()
    ax.grid(True, axis="y", color="#cccccc", linestyle="--", alpha=0.5)

    if len(fp8_values) == 2 and fp8_values[1] > 0 and fp16_values[1] > 0:
        delta_ms = (fp8_values[1] - fp16_values[1]) * 1000.0
        rel_pct = ((fp8_values[1] - fp16_values[1]) / fp8_values[1]) * 100.0
        ax.text(
            0.02, 0.98,
            (
                f"cg-5090: fp16 wins by {delta_ms:.0f} ms "
                f"({rel_pct:.2f}%, reproducible cross-sweep)"
            ),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox={"facecolor": "#fffacd", "edgecolor": "#888", "boxstyle": "round"},
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _render_chart_power(
    matrix: ConsolidatedMatrix,
    summaries: list[SweepSummary],
    output_path: Path,
) -> None:
    """Grouped bars: power_draw_mean per (workflow, host).

    Non-success cells render as 0-height with "N/A" overlay.
    Linear y-axis (power scales linearly across hosts; log unnecessary).
    """
    del summaries

    workflows = _filter_workflows_present(matrix)
    hosts = _filter_hosts_present(matrix)
    cells_idx = _cell_lookup(matrix)

    n_groups = len(workflows)
    n_bars = len(hosts)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, host in enumerate(hosts):
        values: list[float] = []
        labels: list[str] = []
        for wf in workflows:
            cell = cells_idx.get((host, wf))
            if cell is not None and cell.power_draw_mean is not None:
                values.append(cell.power_draw_mean)
                labels.append(f"{cell.power_draw_mean:.0f}W")
            else:
                values.append(0.0)
                labels.append("N/A")
        offsets = x + (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets, values, bar_width,
            label=host, color=HOST_COLORS.get(host, "#888888"),
            edgecolor="black", linewidth=0.5,
        )
        for bar, label in zip(bars, labels, strict=True):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, max(h, 10) + 10,
                label, ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(workflows, rotation=15, ha="right")
    ax.set_ylabel("Power draw (W, mean of DA-008 middle 3)")
    ax.set_title("Power Draw per Workflow")
    ax.legend(loc="upper left", title="GPU")
    ax.grid(True, axis="y", color="#cccccc", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _render_chart_vram(
    matrix: ConsolidatedMatrix,
    summaries: list[SweepSummary],
    output_path: Path,
) -> None:
    """Grouped bars: vram_peak_mean per (workflow, host) in GiB.

    Horizontal dashed reference lines mark each host's physical VRAM
    capacity (cg-3060=12GiB, cg-4090=24GiB, cg-5090=32GiB). Bars
    exceeding 90% of the ceiling get a star annotation (headroom
    risk visual).
    """
    del summaries

    workflows = _filter_workflows_present(matrix)
    hosts = _filter_hosts_present(matrix)
    cells_idx = _cell_lookup(matrix)

    n_groups = len(workflows)
    n_bars = len(hosts)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, host in enumerate(hosts):
        ceiling = HOST_VRAM_CEILING_GIB.get(host)
        values: list[float] = []
        labels: list[str] = []
        is_high: list[bool] = []
        for wf in workflows:
            cell = cells_idx.get((host, wf))
            if cell is not None and cell.vram_peak_mean is not None:
                gib = cell.vram_peak_mean / 1024.0
                values.append(gib)
                labels.append(f"{gib:.1f}")
                is_high.append(ceiling is not None and gib >= 0.9 * ceiling)
            else:
                values.append(0.0)
                labels.append("N/A")
                is_high.append(False)
        offsets = x + (i - n_bars / 2 + 0.5) * bar_width
        bars = ax.bar(
            offsets, values, bar_width,
            label=host, color=HOST_COLORS.get(host, "#888888"),
            edgecolor="black", linewidth=0.5,
        )
        for bar, label, high in zip(bars, labels, is_high, strict=True):
            h = bar.get_height()
            text = f"{label}*" if high else label
            ax.text(
                bar.get_x() + bar.get_width() / 2, max(h, 1) + 0.5,
                text, ha="center", va="bottom", fontsize=8,
            )

    # Horizontal ceilings
    for host, ceiling in HOST_VRAM_CEILING_GIB.items():
        if host in hosts:
            color = HOST_COLORS.get(host, "#888888")
            ax.axhline(
                ceiling, color=color, linestyle="--", linewidth=1.2,
                alpha=0.6,
                label=f"{host} ceiling ({ceiling:.0f} GiB)",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(workflows, rotation=15, ha="right")
    ax.set_ylabel("VRAM peak (GiB, mean of DA-008 middle 3)")
    ax.set_title(
        "VRAM Peak per Workflow (★ marks bars ≥ 90% of host ceiling)"
    )
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, axis="y", color="#cccccc", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 36)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def _select_cleanest_5run_series(
    summaries: list[SweepSummary],
    explicit_key: tuple[str, str] | None = None,
) -> tuple[tuple[SweepRunResult, ...], str, str, str] | None:
    """Pick the cleanest 5-run series across summaries for cold/warm chart.

    "Cleanest" = smallest ``stddev / mean`` ratio of ``wall_clock_stats``
    (warmest set has the lowest relative dispersion — best demonstration
    of DA-008's discard-min/max mechanic). When ``explicit_key`` is
    given, restricts candidates to that ``(host, workflow)``.

    Returns ``(runs, host, workflow, sweep_id)`` or ``None`` when no
    candidate satisfies the 5/5-success requirement.
    """
    candidates: list[
        tuple[float, tuple[SweepRunResult, ...], str, str, str]
    ] = []
    for s in summaries:
        for r in s.results:
            if explicit_key is not None and (r.host, r.workflow_name) != explicit_key:
                continue
            stats = r.wall_clock_stats
            if (
                stats.mean is None
                or stats.stddev is None
                or stats.mean <= 0
                or len(r.runs) < 5
            ):
                continue
            if not all(run.status == "success" for run in r.runs[:5]):
                continue
            rel = stats.stddev / stats.mean
            candidates.append((rel, r.runs, r.host, r.workflow_name, s.sweep_id))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, runs, host, workflow, sweep_id = candidates[0]
    return runs, host, workflow, sweep_id


def _render_chart_cold_warm(
    matrix: ConsolidatedMatrix,
    summaries: list[SweepSummary],
    output_path: Path,
    *,
    explicit_key: tuple[str, str] | None = None,
) -> None:
    """Cold vs warm wall-clock split for one (host, workflow) pair.

    Auto-discovers the cleanest 5-run series via
    :func:`_select_cleanest_5run_series` (smallest ``stddev/mean``
    ratio) unless ``explicit_key`` overrides the (host, workflow).
    Plots the 5 individual run wall_clock values, highlighting the
    cold outlier and the DA-008 discarded extremes.

    When no candidate is found (e.g. all summaries have <5 success
    runs), raises :class:`AnalysisRenderError`; the caller (main)
    is expected to log a warning and skip this chart gracefully.
    """
    del matrix

    selection = _select_cleanest_5run_series(summaries, explicit_key)
    if selection is None:
        raise AnalysisRenderError(
            "cold_warm: no 5/5-success series available in any summary "
            f"(explicit_key={explicit_key})"
        )
    target_runs, target_host, target_workflow, target_sweep_id = selection

    runs_sorted = sorted(target_runs, key=lambda r: r.run_idx)[:5]
    values = [r.wall_clock_s or 0.0 for r in runs_sorted]
    labels = [f"run_{r.run_idx}\n{'cold' if r.is_cold else 'warm'}" for r in runs_sorted]

    # Identify excluded min/max (by value) — mirrors DA-008
    min_idx = min(range(len(values)), key=lambda i: values[i])
    max_idx = max(range(len(values)), key=lambda i: values[i])

    colors: list[str] = []
    for i, run in enumerate(runs_sorted):
        if i in (max_idx, min_idx):
            colors.append("#999999")  # discarded
        elif run.is_cold:
            colors.append("#d62728")  # cold (red) — typically max
        else:
            colors.append("#1f77b4")  # warm included

    cold_value = next((r.wall_clock_s for r in runs_sorted if r.is_cold), None) or 0.0
    warm_values = [r.wall_clock_s or 0.0 for r in runs_sorted if not r.is_cold]
    warm_mean = sum(warm_values) / len(warm_values) if warm_values else 0.0
    ratio = (cold_value / warm_mean) if warm_mean else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        range(len(values)), values, color=colors, edgecolor="black", linewidth=0.7,
    )
    for i, bar in enumerate(bars):
        h = bar.get_height()
        tag = ""
        if i == max_idx:
            tag = "\nexcluded\n(max)"
        elif i == min_idx:
            tag = "\nexcluded\n(min)"
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.2,
            f"{h:.2f}s{tag}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock (s)")
    ax.set_title(
        f"Cold vs Warm Separation — {target_workflow} on {target_host} "
        f"(sweep {target_sweep_id})"
    )
    ax.grid(True, axis="y", color="#cccccc", linestyle="--", alpha=0.5)

    annotation = (
        f"Cold/warm ratio: {ratio:.2f}x\n"
        f"DA-008 discards min+max; remaining 3 averaged.\n"
        f"Warm mean (kept runs): {warm_mean:.2f}s"
    )
    ax.text(
        0.98, 0.98, annotation,
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"facecolor": "#fffacd", "edgecolor": "#888", "boxstyle": "round"},
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Markdown + data serialization (Sub-tarefa 3)
# ============================================================================

def _derive_insights(
    matrix: ConsolidatedMatrix,
    reference_host: str,
) -> list[str]:
    """Auto-derive bullet insights from the consolidated matrix.

    Templates filled programmatically — no hardcoded values. Returns
    a list of markdown bullet lines (without leading ``- ``).
    """
    insights: list[str] = []
    cell_lookup = {(c.host, c.workflow): c for c in matrix.cells}

    valid_by_target: dict[str, list[tuple[str, float]]] = {}
    for sp in matrix.speedups:
        if sp.speedup_ratio is not None:
            valid_by_target.setdefault(sp.target_host, []).append(
                (sp.workflow, sp.speedup_ratio)
            )
    for target_host in sorted(valid_by_target):
        items = sorted(valid_by_target[target_host], key=lambda x: x[1], reverse=True)
        if not items:
            continue
        max_wf, max_ratio = items[0]
        min_wf, min_ratio = items[-1]
        insights.append(
            f"**{reference_host} → {target_host}** speedup range: max "
            f"`{max_wf}` {max_ratio:.2f}×, min `{min_wf}` {min_ratio:.2f}×"
        )

    # Cross-newer-gen (cg-4090 → cg-5090 implicit ratio via reference)
    fp16_4090 = cell_lookup.get(("cg-4090", "flux_dev_fp16"))
    fp16_5090 = cell_lookup.get(("cg-5090", "flux_dev_fp16"))
    if (
        fp16_4090 is not None and fp16_5090 is not None
        and fp16_4090.wall_clock_mean is not None
        and fp16_5090.wall_clock_mean is not None
        and fp16_5090.wall_clock_mean > 0
    ):
        ratio = fp16_4090.wall_clock_mean / fp16_5090.wall_clock_mean
        insights.append(
            f"**cg-4090 → cg-5090 on `flux_dev_fp16`:** "
            f"{fp16_4090.wall_clock_mean:.2f}s → "
            f"{fp16_5090.wall_clock_mean:.2f}s ({ratio:.2f}× — fp16 native "
            "gives the largest cross-newer-gen lift)"
        )

    # fp16 vs fp8 counterintuitive insight on cg-5090
    fp8_cg5090 = cell_lookup.get(("cg-5090", "flux_dev_fp8"))
    fp16_cg5090 = cell_lookup.get(("cg-5090", "flux_dev_fp16"))
    if (
        fp8_cg5090 is not None and fp16_cg5090 is not None
        and fp8_cg5090.wall_clock_mean is not None
        and fp16_cg5090.wall_clock_mean is not None
    ):
        delta_ms = (fp8_cg5090.wall_clock_mean - fp16_cg5090.wall_clock_mean) * 1000.0
        winner = "fp16" if fp16_cg5090.wall_clock_mean < fp8_cg5090.wall_clock_mean else "fp8"
        op = "<" if winner == "fp16" else ">"
        insights.append(
            f"**FLUX cg-5090 counterintuitive:** `flux_dev_fp16` "
            f"({fp16_cg5090.wall_clock_mean:.2f}s) {op} `flux_dev_fp8` "
            f"({fp8_cg5090.wall_clock_mean:.2f}s) — **{winner}** wins by "
            f"{abs(delta_ms):.0f} ms (reproducible cross-sweep)"
        )

    skip_cells = sorted(
        (c for c in matrix.cells if c.status == "skip_oom"),
        key=lambda c: (c.host, c.workflow),
    )
    if skip_cells:
        skip_list = ", ".join(f"`{c.host}/{c.workflow}`" for c in skip_cells)
        insights.append(f"**Skipped (OOM a priori):** {skip_list}")

    timeout_cells = sorted(
        (c for c in matrix.cells if c.status == "timeout"),
        key=lambda c: (c.host, c.workflow),
    )
    if timeout_cells:
        descs = [
            f"`{c.host}/{c.workflow}` (err={c.error_count})"
            for c in timeout_cells
        ]
        insights.append(f"**Timed out (workflow_timeout exceeded):** {', '.join(descs)}")

    sdxl_ref = cell_lookup.get((reference_host, "sdxl_base"))
    sdxl_cg5090 = cell_lookup.get(("cg-5090", "sdxl_base"))
    if (
        sdxl_ref is not None and sdxl_cg5090 is not None
        and sdxl_ref.power_draw_mean is not None
        and sdxl_cg5090.power_draw_mean is not None
        and sdxl_ref.power_draw_mean > 0
    ):
        ratio = sdxl_cg5090.power_draw_mean / sdxl_ref.power_draw_mean
        insights.append(
            f"**Power scaling (`sdxl_base`):** {reference_host} "
            f"{sdxl_ref.power_draw_mean:.1f} W → cg-5090 "
            f"{sdxl_cg5090.power_draw_mean:.1f} W ({ratio:.2f}×)"
        )

    return insights


_CHART_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Speedup Bars", "speedup_per_workflow_bars"),
    ("Speedup Heatmap", "speedup_matrix_heatmap"),
    ("FP16 vs FP8 cg-5090", "fp16_vs_fp8_cg5090_highlight"),
    ("Power Scaling", "power_scaling_per_workflow"),
    ("VRAM Peak", "vram_peak_per_workflow"),
    ("Cold/Warm Separation", "cold_warm_separation"),
)


def _render_markdown(
    matrix: ConsolidatedMatrix,
    output_path: Path,
) -> None:
    """Render ``report.md`` with consolidated tables + auto-derived insights.

    Sections: header, methodology, status summary, consolidated matrix,
    speedup matrix, key insights (programmatic), chart inline refs,
    error details (when any cell has error_count > 0).
    """
    reference_host = (
        matrix.speedups[0].reference_host
        if matrix.speedups
        else DEFAULT_REFERENCE_HOST
    )

    lines: list[str] = []
    lines.append("# V1 Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {matrix.generated_at}")
    lines.append(
        "**Source sweeps:** " + ", ".join(f"`{sid}`" for sid in matrix.source_sweep_ids)
    )
    lines.append(f"**Reference host:** `{reference_host}`")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "DA-008 mechanic: 5 runs per (host, workflow); discard min+max; "
        "mean of middle 3."
    )
    lines.append(
        "Status taxonomy: `success` / `skip_oom` / `error` / `timeout`."
    )
    lines.append("")

    status_counts: dict[CellStatus, int] = {}
    for c in matrix.cells:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1
    lines.append("## Status Summary")
    lines.append("")
    for status in ("success", "skip_oom", "timeout", "error"):
        # mypy: iterating literal strings; cast each to CellStatus for lookup
        key = cast_status_literal(status)
        lines.append(f"- **{status}:** {status_counts.get(key, 0)}")
    total = sum(status_counts.values())
    lines.append(f"- _total cells:_ {total}")
    lines.append("")

    lines.append("## Consolidated Matrix")
    lines.append("")
    lines.append(
        "| host | workflow | status | wall_mean (s) | vram_peak (MiB) | "
        "gpu_util (%) | power (W) | err |"
    )
    lines.append(
        "|------|----------|--------|---------------|------------------|"
        "---------------|-----------|-----|"
    )
    for c in matrix.cells:
        wc = f"{c.wall_clock_mean:.2f}" if c.wall_clock_mean is not None else "n/a"
        vr = f"{c.vram_peak_mean:.0f}" if c.vram_peak_mean is not None else "n/a"
        gu = f"{c.gpu_util_mean:.1f}" if c.gpu_util_mean is not None else "n/a"
        pw = f"{c.power_draw_mean:.1f}" if c.power_draw_mean is not None else "n/a"
        lines.append(
            f"| {c.host} | `{c.workflow}` | `{c.status}` | {wc} | {vr} | "
            f"{gu} | {pw} | {c.error_count} |"
        )
    lines.append("")

    hosts_present = sorted({c.host for c in matrix.cells})
    workflows_present = sorted({c.workflow for c in matrix.cells})
    hosts_ordered = [reference_host] + [
        h for h in hosts_present if h != reference_host
    ]
    sp_lookup = {(s.workflow, s.target_host): s for s in matrix.speedups}
    cell_lookup = {(c.host, c.workflow): c for c in matrix.cells}

    lines.append(f"## Speedup Matrix (baseline: `{reference_host}`)")
    lines.append("")
    header = "| workflow | " + " | ".join(hosts_ordered) + " |"
    sep = "|" + "----------|" * (len(hosts_ordered) + 1)
    lines.append(header)
    lines.append(sep)
    for wf in workflows_present:
        row_parts: list[str] = [f"`{wf}`"]
        for host in hosts_ordered:
            if host == reference_host:
                ref_cell = cell_lookup.get((reference_host, wf))
                if ref_cell is not None and ref_cell.status == "success":
                    row_parts.append("1.00×")
                elif ref_cell is not None:
                    row_parts.append(f"_({ref_cell.status})_")
                else:
                    row_parts.append("n/a")
            else:
                sp = sp_lookup.get((wf, host))
                target_cell = cell_lookup.get((host, wf))
                if sp is not None and sp.speedup_ratio is not None:
                    row_parts.append(f"{sp.speedup_ratio:.2f}×")
                elif target_cell is not None and target_cell.status != "success":
                    row_parts.append(f"_({target_cell.status})_")
                else:
                    row_parts.append("n/a")
        lines.append("| " + " | ".join(row_parts) + " |")
    lines.append("")

    lines.append("## Key Insights")
    lines.append("")
    insights = _derive_insights(matrix, reference_host)
    for ins in insights:
        lines.append(f"- {ins}")
    lines.append("")

    lines.append("## Charts")
    lines.append("")
    for title, fname in _CHART_SECTIONS:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}](charts/{fname}.png)")
        lines.append("")

    error_cells = [c for c in matrix.cells if c.error_count > 0]
    if error_cells:
        lines.append("## Error Details")
        lines.append("")
        for c in error_cells:
            msg = c.error_message or "(no message)"
            lines.append(
                f"- **`{c.host}/{c.workflow}`** "
                f"(`{c.status}`, err_count={c.error_count}): `{msg}`"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("saved markdown report to %s", output_path)


def cast_status_literal(value: str) -> CellStatus:
    """Narrow ``str`` to :data:`CellStatus` for dict lookups in templates."""
    if value not in ("success", "skip_oom", "error", "timeout"):
        raise ValueError(f"not a valid CellStatus: {value!r}")
    # mypy: explicit cast — `value` is one of the literals at runtime
    return value  # type: ignore[return-value]


def _save_data_json(
    matrix: ConsolidatedMatrix,
    output_path: Path,
) -> None:
    """Serialize :class:`ConsolidatedMatrix` as pretty JSON.

    Mirrors :func:`installer.benchmark.sweep._save_summary` —
    ``indent=2``, ``ensure_ascii=False``, ``default=str`` for Path
    coercion (none in V1 but additive-safe), parent dir created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(matrix), f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    logger.info("saved analyze data to %s", output_path)


# ============================================================================
# CLI
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for :func:`main` (Sub-tarefa 1 ships full args)."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark analysis toolchain. Ingests one or more "
            "sweep_outputs/<sweep_id>/summary.json files and emits a "
            "self-contained report (markdown + data JSON + 6 PNG charts) "
            "under reports/<UTC-timestamp>/."
        ),
    )
    parser.add_argument(
        "--summary-glob",
        default="sweep_outputs/*/summary.json",
        help=(
            "Glob pattern matching sweep summary files "
            "(default: sweep_outputs/*/summary.json)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./reports/<UTC-timestamp>).",
    )
    parser.add_argument(
        "--workflow-filter",
        default=None,
        help="Optional regex; only workflows whose basename matches are included.",
    )
    parser.add_argument(
        "--host-filter",
        default=None,
        help="Optional regex; only hosts whose alias matches are included.",
    )
    parser.add_argument(
        "--reference-host",
        default=DEFAULT_REFERENCE_HOST,
        help=(
            f"Host used as the speedup denominator "
            f"(default: {DEFAULT_REFERENCE_HOST})."
        ),
    )
    parser.add_argument(
        "--include-charts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render the 6 PNG charts (default: True; --no-include-charts skips).",
    )
    parser.add_argument(
        "--cold-warm-key",
        default=None,
        help=(
            "Optional 'host:workflow' override for the cold/warm chart's "
            "5-run series (default: auto-pick the cleanest stddev/mean)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the discovered summaries and execution plan without "
            "writing any output files."
        ),
    )
    return parser


def main() -> None:
    """End-to-end analyze orchestrator (Bloco 21 Sub-tarefa 3)."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    parser = _build_argparser()
    args = parser.parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(f"reports/{timestamp}")
    )

    summary_paths = _discover_summaries(args.summary_glob)
    logger.info(
        "discovered %d summary files; output_dir=%s",
        len(summary_paths), output_dir,
    )

    if args.dry_run:
        print("\n=== ANALYZE PLAN (--dry-run) ===")
        print(f"output_dir (would create): {output_dir}")
        print(f"include_charts: {args.include_charts}")
        print(f"reference_host: {args.reference_host}")
        if args.workflow_filter:
            print(f"workflow_filter: {args.workflow_filter!r}")
        if args.host_filter:
            print(f"host_filter: {args.host_filter!r}")
        if args.cold_warm_key:
            print(f"cold_warm_key override: {args.cold_warm_key!r}")
        print(f"summaries ({len(summary_paths)}):")
        for p in summary_paths:
            print(f"  - {p}")
        return

    summaries = [_load_summary(p) for p in summary_paths]

    if args.workflow_filter:
        wf_re = re.compile(args.workflow_filter)
        filtered: list[SweepSummary] = []
        for s in summaries:
            filtered_results = tuple(
                r for r in s.results if wf_re.search(r.workflow_name)
            )
            filtered_skipped = tuple(
                e for e in s.skipped if wf_re.search(e.workflow)
            )
            filtered.append(
                replace(s, results=filtered_results, skipped=filtered_skipped)
            )
        summaries = filtered
    if args.host_filter:
        host_re = re.compile(args.host_filter)
        filtered = []
        for s in summaries:
            filtered_results = tuple(
                r for r in s.results if host_re.search(r.host)
            )
            filtered_skipped = tuple(
                e for e in s.skipped if host_re.search(e.host)
            )
            filtered.append(
                replace(s, results=filtered_results, skipped=filtered_skipped)
            )
        summaries = filtered

    matrix0 = _build_matrix(summaries)
    speedups = _compute_speedups(matrix0, args.reference_host)
    matrix = replace(matrix0, speedups=speedups)

    output_dir.mkdir(parents=True, exist_ok=True)
    _render_markdown(matrix, output_dir / "report.md")
    _save_data_json(matrix, output_dir / "data.json")

    chart_count = 0
    if args.include_charts:
        charts_dir = output_dir / "charts"
        for name, fn in (
            ("speedup_per_workflow_bars", _render_chart_speedup_bars),
            ("speedup_matrix_heatmap", _render_chart_speedup_heatmap),
            ("fp16_vs_fp8_cg5090_highlight", _render_chart_fp16_vs_fp8),
            ("power_scaling_per_workflow", _render_chart_power),
            ("vram_peak_per_workflow", _render_chart_vram),
        ):
            try:
                fn(matrix, summaries, charts_dir / f"{name}.png")
                chart_count += 1
            except AnalysisRenderError as exc:
                logger.warning("chart %s skipped: %s", name, exc)

        explicit_key: tuple[str, str] | None = None
        if args.cold_warm_key:
            parts = args.cold_warm_key.split(":", 1)
            if len(parts) == 2:
                explicit_key = (parts[0], parts[1])
            else:
                logger.warning(
                    "ignoring malformed --cold-warm-key %r (expected 'host:workflow')",
                    args.cold_warm_key,
                )
        try:
            _render_chart_cold_warm(
                matrix, summaries,
                charts_dir / "cold_warm_separation.png",
                explicit_key=explicit_key,
            )
            chart_count += 1
        except AnalysisRenderError as exc:
            logger.warning("chart cold_warm_separation skipped: %s", exc)

    status_counts: dict[str, int] = {}
    for c in matrix.cells:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1

    print("\n=== ANALYZE DONE ===")
    print(f"report:  {output_dir / 'report.md'}")
    print(f"data:    {output_dir / 'data.json'}")
    if args.include_charts:
        print(f"charts:  {output_dir / 'charts'}/ ({chart_count} PNGs)")
    print(f"cells:   {sum(status_counts.values())} total")
    for status in ("success", "timeout", "error", "skip_oom"):
        if status in status_counts:
            print(f"  - {status}: {status_counts[status]}")


if __name__ == "__main__":
    main()
