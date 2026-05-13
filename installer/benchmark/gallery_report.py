"""Gallery analysis + visual report (V2 gallery deliverable).

Single responsibility: ingest one or more
:class:`installer.benchmark.gallery.GallerySummary` JSON files + the
per-cell output artifacts (t2i.png / i2v.webp) and render:

1. ``gallery_report.md`` — visual markdown with comparison grids
   (host × variant × aspect) embedding the rendered PNGs/WEBPs.
2. 4 cross-cutting charts (matplotlib):
   - cross-CG t2i wall-clock matrix (host × variant×aspect heatmap)
   - VRAM vs RAM offload scatter (regime classification)
   - GPU power efficiency (joules-per-image, grouped bars)
   - i2v coverage heatmap (host × variant, status-colored)
3. ``gallery_data.json`` — flattened indexable cell data for
   downstream tooling (web viewer, comparison filter UI).

V2 gallery semantics differ from V1 sweep — gallery cells are 1-run
(no DA-008 5-run aggregates) and carry an aspect_ratio dimension +
t2i/i2v dual-phase telemetry. Hence this module is a SIBLING of
:mod:`installer.benchmark.analyze` (V1 sweep analysis) rather than
an extension. Helper duplication is small (~50 LOC: atomic JSON
save, summary discovery glob, markdown header conventions) — clean
separation > DRY for these.

Bloco 22e Sub-tarefa 1 ships dataclasses + CLI + stubs only — the
chart renderers, markdown builder, and comparison-grid builder raise
:class:`NotImplementedError`. Sub-tarefa 2 implements the body.

Reads inputs that conform to the gallery schema (``schema_version=1``);
on schema drift the loader raises :class:`GalleryReportConfigError`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D

from installer.benchmark.gallery import (
    DEFAULT_ASPECT_RATIOS,
    VARIANT_CATALOG,
    AspectRatio,
    GalleryCell,
    GalleryCellStatus,
    GallerySummary,
    ModelVariant,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Styling constants (mirror installer.benchmark.analyze conventions)
# ============================================================================

HOST_ORDER: tuple[str, ...] = ("cg-3060", "cg-4090", "cg-5090")
HOST_COLORS: dict[str, str] = {
    "cg-3060": "#1f77b4",
    "cg-4090": "#ff7f0e",
    "cg-5090": "#2ca02c",
}
HOST_MARKERS: dict[str, str] = {
    "cg-3060": "o",
    "cg-4090": "s",
    "cg-5090": "^",
}

# Canonical ordering of t2i variant labels — built from VARIANT_CATALOG
# at import time to keep gallery + report consistent.
T2I_VARIANT_LABELS: tuple[str, ...] = tuple(
    f"{v.family}:{v.precision}"
    for v in VARIANT_CATALOG
    if not v.is_video
)

# Variant family colors (6 unique families across the t2i catalog).
FAMILY_ORDER: tuple[str, ...] = (
    "sdxl", "flux1", "flux2", "qwen-image", "qwen-2512", "hunyuan-2.1",
)
FAMILY_COLORS: dict[str, tuple[float, float, float, float]] = dict(
    zip(
        FAMILY_ORDER,
        plt.get_cmap("tab10").colors[: len(FAMILY_ORDER)],  # type: ignore[attr-defined]
        strict=True,
    )
)

DEFAULT_DPI: int = 150


# ============================================================================
# Exceptions
# ============================================================================

class GalleryReportError(Exception):
    """Base class for all gallery_report errors."""


class GalleryReportConfigError(GalleryReportError):
    """Raised when CLI/configuration or input summary is invalid.

    Examples: ``--summary-glob`` matches zero files; loaded
    summary.json missing required fields; schema_version drift;
    output directory cannot be created.
    """


class GalleryReportRenderError(GalleryReportError):
    """Raised when chart/markdown rendering fails.

    Per-chart failures are caught and converted to warnings (skip-
    and-continue) — this exception is reserved for failures that
    prevent producing any meaningful output.
    """


# ============================================================================
# Type aliases
# ============================================================================

ArtifactStatus = Literal["both", "t2i_only", "missing"]
"""Per-cell artifact presence flag.

- ``both``: t2i.png AND i2v.webp present on disk
- ``t2i_only``: t2i.png present; no i2v (either skipped or failed)
- ``missing``: t2i.png absent — cell can't render in visual grid
  (status != "success" or download/atomicity failure)
"""


# ============================================================================
# Dataclasses (Bloco 22e spec)
# ============================================================================

@dataclass(frozen=True, slots=True)
class GalleryReportConfig:
    """Immutable configuration captured for the gallery report header.

    ``summary_glob`` typically points at one ``summary.json`` (V2 gallery
    final-state file) but the helper supports multiple summaries for
    cross-dispatch consolidation (V3 use case).
    """

    summary_glob: str
    output_dir: Path
    variant_filter: str | None = None
    host_filter: str | None = None
    include_charts: bool = True


@dataclass(frozen=True, slots=True)
class GalleryGridCell:
    """Per-cell intermediate row used to build the comparison grid.

    Carries identity, status + telemetry snapshot, and artifact
    paths RELATIVE to the source summary's directory (Bloco 22e.2a
    default — :func:`_render_markdown_gallery` in Sub-tarefa 2b
    re-rebases to its output_dir when embedding). ``summary_source_dir``
    is kept on the cell so 2b can reconstruct absolute paths without
    re-loading the parent summary list.

    Constructed by :func:`_build_comparison_grid` from a
    :class:`GalleryCell` + the summary file's parent directory.
    """

    host: str
    variant_family: str
    variant_precision: str
    aspect_ratio: AspectRatio
    status: GalleryCellStatus
    t2i_wall_clock_s: float | None
    t2i_vram_peak_mib: float | None
    t2i_ram_peak_mib: float | None
    t2i_gpu_util_avg_pct: float | None
    t2i_gpu_power_avg_w: float | None
    i2v_wall_clock_s: float | None
    i2v_vram_peak_mib: float | None
    i2v_ram_peak_mib: float | None
    error_message: str | None
    artifact_status: ArtifactStatus
    summary_source_dir: Path
    t2i_relative_path: str | None
    i2v_relative_path: str | None


@dataclass(frozen=True, slots=True)
class GalleryReportSummary:
    """Top-level consolidated output of the gallery analysis pipeline.

    ``cells`` is the flat list keyed by (host, family, precision,
    aspect_name); downstream tooling can build any indexed view it
    wants. ``rendered_charts`` records which charts succeeded — a
    chart that failed mid-render emits a warning and is omitted here.
    """

    schema_version: int
    report_id: str
    config: dict[str, Any]
    source_summaries: tuple[str, ...]
    cells: tuple[GalleryGridCell, ...]
    rendered_charts: tuple[str, ...]


# ============================================================================
# Helpers (stubs — Sub-tarefa 2 implements)
# ============================================================================

def _discover_gallery_summaries(glob_pattern: str) -> list[Path]:
    """Discover gallery ``summary.json`` files matching ``glob_pattern``.

    Returns sorted absolute paths. Sort is lexicographic on the full
    path string — since gallery output dirs follow the convention
    ``gallery_<UTC-timestamp>/`` where the timestamp is ISO-8601
    ``YYYYMMDDTHHMMSSZ``, lexicographic order = chronological order.

    Raises:
        GalleryReportConfigError: zero matches (the user almost
            certainly intends a real ingest; silent empty result
            masks misconfigured ``--summary-glob``).
    """
    matches = sorted(Path.cwd().glob(glob_pattern))
    if not matches:
        raise GalleryReportConfigError(
            f"--summary-glob {glob_pattern!r} matched 0 files "
            f"(cwd={Path.cwd()})"
        )
    return [p.resolve() for p in matches]


def _hydrate_aspect_ratio(d: dict[str, Any]) -> AspectRatio:
    """Reconstruct an :class:`AspectRatio` from a JSON dict."""
    return AspectRatio(
        name=str(d["name"]),
        width=int(d["width"]),
        height=int(d["height"]),
    )


def _hydrate_model_variant(d: dict[str, Any]) -> ModelVariant:
    """Reconstruct a :class:`ModelVariant` from a JSON dict."""
    low = d.get("unet_filename_low")
    return ModelVariant(
        family=str(d["family"]),
        precision=str(d["precision"]),
        workflow_name=str(d["workflow_name"]),
        unet_filename=str(d["unet_filename"]),
        is_video=bool(d.get("is_video", False)),
        unet_filename_low=str(low) if low is not None else None,
    )


def _hydrate_gallery_cell(d: dict[str, Any]) -> GalleryCell:
    """Reconstruct a :class:`GalleryCell` from a JSON dict.

    Cell fields with ``None`` semantics (telemetry, paths,
    error_message) round-trip directly; status is validated
    against the Literal alias by ``GalleryCell.__init__``.
    """
    return GalleryCell(
        host=str(d["host"]),
        variant_family=str(d["variant_family"]),
        variant_precision=str(d["variant_precision"]),
        aspect_ratio_name=str(d["aspect_ratio_name"]),
        status=d["status"],
        t2i_wall_clock_s=d.get("t2i_wall_clock_s"),
        t2i_vram_peak_mib=d.get("t2i_vram_peak_mib"),
        t2i_ram_peak_mib=d.get("t2i_ram_peak_mib"),
        t2i_gpu_util_avg_pct=d.get("t2i_gpu_util_avg_pct"),
        t2i_gpu_power_avg_w=d.get("t2i_gpu_power_avg_w"),
        t2i_output_path=d.get("t2i_output_path"),
        i2v_wall_clock_s=d.get("i2v_wall_clock_s"),
        i2v_vram_peak_mib=d.get("i2v_vram_peak_mib"),
        i2v_ram_peak_mib=d.get("i2v_ram_peak_mib"),
        i2v_output_path=d.get("i2v_output_path"),
        error_message=d.get("error_message"),
    )


def _load_gallery_summary(path: Path) -> GallerySummary:
    """Parse a gallery ``summary.json`` into a :class:`GallerySummary`.

    Re-hydrates the nested dataclasses (``aspect_ratios``,
    ``variants``, ``cells``) so the rest of the pipeline operates on
    typed objects, not raw dicts. Missing optional cell fields are
    coerced to ``None`` (legacy summaries pre-Bloco-22d may lack
    some i2v fields).

    Raises:
        GalleryReportConfigError: file missing, non-JSON, non-object,
            wrong ``schema_version``, missing required top-level
            fields, or cell hydration shape mismatch.
    """
    if not path.is_file():
        raise GalleryReportConfigError(f"summary not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GalleryReportConfigError(
            f"summary at {path} is corrupted (JSON parse): {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise GalleryReportConfigError(
            f"summary at {path} is not a JSON object"
        )

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise GalleryReportConfigError(
            f"summary at {path} schema_version={schema_version!r} "
            "(expected 1)"
        )

    gallery_id = data.get("gallery_id")
    if not isinstance(gallery_id, str) or not gallery_id:
        raise GalleryReportConfigError(
            f"summary at {path} missing valid 'gallery_id'"
        )

    config = data.get("config")
    if not isinstance(config, dict):
        raise GalleryReportConfigError(
            f"summary at {path} 'config' must be a dict"
        )

    raw_aspects = data.get("aspect_ratios", [])
    if not isinstance(raw_aspects, list):
        raise GalleryReportConfigError(
            f"summary at {path} 'aspect_ratios' must be a list"
        )
    aspect_ratios = tuple(
        _hydrate_aspect_ratio(a) for a in raw_aspects
        if isinstance(a, dict)
    )

    raw_variants = data.get("variants", [])
    if not isinstance(raw_variants, list):
        raise GalleryReportConfigError(
            f"summary at {path} 'variants' must be a list"
        )
    variants = tuple(
        _hydrate_model_variant(v) for v in raw_variants
        if isinstance(v, dict)
    )

    raw_cells = data.get("cells", [])
    if not isinstance(raw_cells, list):
        raise GalleryReportConfigError(
            f"summary at {path} 'cells' must be a list"
        )
    cells: list[GalleryCell] = []
    for i, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            raise GalleryReportConfigError(
                f"summary at {path} cell index {i} is not an object"
            )
        try:
            cells.append(_hydrate_gallery_cell(raw))
        except (KeyError, TypeError, ValueError) as exc:
            raise GalleryReportConfigError(
                f"summary at {path} cell index {i} hydration failed: {exc}"
            ) from exc

    return GallerySummary(
        schema_version=schema_version,
        gallery_id=gallery_id,
        config=config,
        aspect_ratios=aspect_ratios,
        variants=variants,
        cells=tuple(cells),
    )


def _build_comparison_grid(
    summary_path_pairs: Iterable[tuple[GallerySummary, Path]],
) -> tuple[GalleryGridCell, ...]:
    """Build the indexed comparison grid from one or more summaries.

    Multi-summary dedup (Bloco 22e.2a default): pairs are sorted by
    ``GallerySummary.gallery_id`` ascending; last write per identity
    key ``(host, variant_family, variant_precision, aspect_ratio_name)``
    wins. V2 typical usage is 1 summary; V3 consolidation scenarios
    (cross-dispatch merge) get correct semantics free.

    Canonical iteration order matches :mod:`installer.benchmark.gallery`
    enumeration: ``VARIANT_CATALOG`` order × ``DEFAULT_ASPECT_RATIOS``
    order × hosts-sorted. Video variants are skipped (i2v is chained,
    not enumerated as a standalone cell).

    Calls :func:`_resolve_artifact_paths` per cell to populate
    ``artifact_status`` + ``t2i_relative_path`` + ``i2v_relative_path``
    based on actual on-disk artifact presence.
    """
    pairs = sorted(summary_path_pairs, key=lambda p: p[0].gallery_id)

    by_key: dict[tuple[str, str, str, str], tuple[GalleryCell, Path]] = {}
    for summary, summary_path in pairs:
        source_dir = summary_path.parent
        for cell in summary.cells:
            key = (
                cell.host,
                cell.variant_family,
                cell.variant_precision,
                cell.aspect_ratio_name,
            )
            by_key[key] = (cell, source_dir)

    aspect_by_name: dict[str, AspectRatio] = {
        a.name: a for a in DEFAULT_ASPECT_RATIOS
    }
    # Fallback aspect when summaries used custom aspects not in DEFAULT.
    for cell, _ in by_key.values():
        if cell.aspect_ratio_name not in aspect_by_name:
            # Synthesize an AspectRatio so downstream code has dims;
            # actual width/height are unknown without summary lookup,
            # so default to a square (consumer can re-resolve via
            # summary.aspect_ratios if needed).
            aspect_by_name[cell.aspect_ratio_name] = AspectRatio(
                cell.aspect_ratio_name, 0, 0,
            )

    hosts_present = sorted({k[0] for k in by_key})

    grid_cells: list[GalleryGridCell] = []
    for variant in VARIANT_CATALOG:
        if variant.is_video:
            continue
        for aspect in DEFAULT_ASPECT_RATIOS:
            for host in hosts_present:
                key = (host, variant.family, variant.precision, aspect.name)
                hit = by_key.get(key)
                if hit is None:
                    continue
                cell, source_dir = hit
                grid_cell = _gallery_cell_to_grid_cell(cell, aspect, source_dir)
                grid_cells.append(_resolve_artifact_paths(grid_cell, source_dir))

    return tuple(grid_cells)


def _gallery_cell_to_grid_cell(
    cell: GalleryCell,
    aspect: AspectRatio,
    source_dir: Path,
) -> GalleryGridCell:
    """Construct an unresolved :class:`GalleryGridCell` from a gallery cell.

    Artifact resolution (status + final relative paths) is performed
    by :func:`_resolve_artifact_paths` immediately after; this helper
    just carries over identity + telemetry + error_message and stamps
    placeholder ``artifact_status="missing"``.
    """
    return GalleryGridCell(
        host=cell.host,
        variant_family=cell.variant_family,
        variant_precision=cell.variant_precision,
        aspect_ratio=aspect,
        status=cell.status,
        t2i_wall_clock_s=cell.t2i_wall_clock_s,
        t2i_vram_peak_mib=cell.t2i_vram_peak_mib,
        t2i_ram_peak_mib=cell.t2i_ram_peak_mib,
        t2i_gpu_util_avg_pct=cell.t2i_gpu_util_avg_pct,
        t2i_gpu_power_avg_w=cell.t2i_gpu_power_avg_w,
        i2v_wall_clock_s=cell.i2v_wall_clock_s,
        i2v_vram_peak_mib=cell.i2v_vram_peak_mib,
        i2v_ram_peak_mib=cell.i2v_ram_peak_mib,
        error_message=cell.error_message,
        artifact_status="missing",
        summary_source_dir=source_dir,
        t2i_relative_path=cell.t2i_output_path,
        i2v_relative_path=cell.i2v_output_path,
    )


def _resolve_artifact_paths(
    cell: GalleryGridCell,
    summary_source_dir: Path,
) -> GalleryGridCell:
    """Resolve artifact existence + canonical relative paths.

    Reconstructs expected paths from cell identity using gallery's
    canonical output dir layout (output_dir/<host>/<family>_<precision>_
    <aspect>/t2i.png + i2v.webp) — robust against stale
    ``t2i_output_path`` / ``i2v_output_path`` values in the summary.

    Returns a new :class:`GalleryGridCell` with:

    - ``artifact_status``:
      - ``"both"`` if t2i.png AND i2v.webp present on disk
      - ``"t2i_only"`` if t2i.png present, i2v.webp absent
      - ``"missing"`` if t2i.png absent (cell failed t2i OR artifacts
        deleted post-dispatch)
    - ``t2i_relative_path`` / ``i2v_relative_path``: forward-slash
      relative-to-summary-source-dir strings (suitable for embedding
      in markdown after rebase to output dir), or ``None`` if the
      artifact is missing on disk.
    """
    cell_dirname = (
        f"{cell.variant_family}_{cell.variant_precision}_"
        f"{cell.aspect_ratio.name.replace(':', 'x')}"
    )
    cell_dir = summary_source_dir / cell.host / cell_dirname
    t2i_abs = cell_dir / "t2i.png"
    i2v_abs = cell_dir / "i2v.webp"

    t2i_exists = t2i_abs.is_file()
    i2v_exists = i2v_abs.is_file()

    if not t2i_exists:
        status: ArtifactStatus = "missing"
    elif i2v_exists:
        status = "both"
    else:
        status = "t2i_only"

    t2i_rel: str | None = (
        t2i_abs.relative_to(summary_source_dir).as_posix()
        if t2i_exists
        else None
    )
    i2v_rel: str | None = (
        i2v_abs.relative_to(summary_source_dir).as_posix()
        if i2v_exists
        else None
    )

    return replace(
        cell,
        artifact_status=status,
        t2i_relative_path=t2i_rel,
        i2v_relative_path=i2v_rel,
    )


def _apply_filters(
    cells: Iterable[GalleryGridCell],
    variant_filter: str | None,
    host_filter: str | None,
) -> tuple[GalleryGridCell, ...]:
    """Apply regex filters mirroring :func:`gallery._enumerate_combinations`.

    - ``variant_filter`` is applied as :func:`re.search` to
      ``f"{family}:{precision}"``.
    - ``host_filter`` is applied as :func:`re.search` to the host alias.

    Both ``None`` (default) returns input unchanged. Order is
    preserved (callers rely on :func:`_build_comparison_grid` canonical
    ordering). Empty result emits a WARN but does not raise — caller
    may legitimately want a zero-cell report (dry-run sweep).
    """
    result = list(cells)

    if variant_filter is not None:
        variant_re = re.compile(variant_filter)
        result = [
            c for c in result
            if variant_re.search(f"{c.variant_family}:{c.variant_precision}")
        ]

    if host_filter is not None:
        host_re = re.compile(host_filter)
        result = [c for c in result if host_re.search(c.host)]

    if not result:
        logger.warning(
            "_apply_filters: 0 cells after filtering "
            "(variant_filter=%r, host_filter=%r)",
            variant_filter, host_filter,
        )

    return tuple(result)


def _rebase_artifact_path(
    rel_path: str | None,
    source_dir: Path,
    output_dir: Path,
) -> str | None:
    """Convert summary-source-relative path to output-dir-relative.

    Markdown viewers resolve image links relative to the markdown
    file's directory. Artifacts live under ``summary_source_dir``,
    markdown lives in ``output_dir`` — compute relative path from
    one to the other via :func:`os.path.relpath`. Returns
    POSIX-style forward-slash path for cross-platform compatibility.
    """
    if rel_path is None:
        return None
    absolute = (source_dir / rel_path).resolve()
    relative = os.path.relpath(absolute, output_dir.resolve())
    return Path(relative).as_posix()


def _render_markdown_gallery(
    cells: tuple[GalleryGridCell, ...],
    output_path: Path,
    chart_names: list[str],
    insights: list[str],
    source_summaries: list[str],
    report_id: str,
) -> None:
    """Render the top-level visual gallery markdown to ``output_path``.

    Layout sections:
        1. Header: timestamp, report_id, source summaries, cell counts.
        2. Methodology paragraph.
        3. Insights bullets (from :func:`_derive_insights`).
        4. Status summary table.
        5. Chart embeds (relative ``charts/<name>.png`` paths).
        6. Gallery by variant — 3-column tables (host × aspect)
           with t2i.png embedded and i2v.webp as animated image
           (WEBP is image format that supports animation; ``<img>``
           tag renders inline in GitHub/VSCode markdown viewers).
        7. Errors / anomalies section (cells with status != success
           or i2v failures, forensic data preserved).

    Image paths are rebased via :func:`_rebase_artifact_path` so the
    markdown is portable as long as ``output_dir`` and
    ``summary_source_dir`` retain their relative position.
    """
    output_dir = output_path.parent
    lines: list[str] = []

    # 1. Header.
    lines.append("# Gallery Report V2")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(UTC).isoformat()}")
    lines.append(f"**Report ID:** `{report_id}`")
    lines.append("**Source summaries:**")
    for s in source_summaries:
        lines.append(f"- `{s}`")
    lines.append("")
    n_total = len(cells)
    n_both = sum(1 for c in cells if c.artifact_status == "both")
    n_t2i_only = sum(1 for c in cells if c.artifact_status == "t2i_only")
    n_missing = sum(1 for c in cells if c.artifact_status == "missing")
    lines.append(
        f"**Cells:** {n_total} total · {n_both} with i2v · "
        f"{n_t2i_only} t2i-only · {n_missing} missing on disk"
    )
    lines.append("")

    # 2. Methodology.
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "V2 gallery cross-product: 11 t2i variants × 3 hosts × 3 aspect "
        "ratios = 99 cells. 1 run per combo (vs DA-008 5-run mechanic in "
        "V1 sweep). Optional i2v chain via WAN 2.2 fp8/fp16 where viable "
        "(see débito #21 queue-poisoning notes for i2v gaps)."
    )
    lines.append("")

    # 3. Insights.
    lines.append("## Insights")
    lines.append("")
    for ins in insights:
        lines.append(f"- {ins}")
    lines.append("")

    # 4. Status summary.
    lines.append("## Status Summary")
    lines.append("")
    lines.append("| status | count |")
    lines.append("|---|---|")
    for status, n in Counter(c.status for c in cells).most_common():
        lines.append(f"| `{status}` | {n} |")
    lines.append("")

    # 5. Charts.
    if chart_names:
        lines.append("## Charts")
        lines.append("")
        for name in chart_names:
            pretty = name.replace("_", " ").title()
            lines.append(f"![{pretty}](charts/{name}.png)")
            lines.append("")

    # 6. Gallery by variant.
    lines.append("## Gallery — by variant")
    lines.append("")
    t2i_variants = [v for v in VARIANT_CATALOG if not v.is_video]
    for variant in t2i_variants:
        variant_cells = [
            c for c in cells
            if c.variant_family == variant.family
            and c.variant_precision == variant.precision
        ]
        if not variant_cells:
            continue
        lines.append(f"### {variant.family}:{variant.precision}")
        lines.append("")
        for aspect in DEFAULT_ASPECT_RATIOS:
            aspect_cells = [
                c for c in variant_cells
                if c.aspect_ratio.name == aspect.name
            ]
            if not aspect_cells:
                continue
            lines.append(
                f"#### Aspect {aspect.name} "
                f"({aspect.width}×{aspect.height})"
            )
            lines.append("")
            lines.append("| host | t2i | metrics | i2v |")
            lines.append("|---|---|---|---|")
            for host in HOST_ORDER:
                cell = next(
                    (c for c in aspect_cells if c.host == host), None,
                )
                if cell is None:
                    lines.append(f"| {host} | — | (cell missing) | — |")
                    continue
                t2i_md = "—"
                if cell.t2i_relative_path is not None:
                    rebased = _rebase_artifact_path(
                        cell.t2i_relative_path,
                        cell.summary_source_dir, output_dir,
                    )
                    if rebased is not None:
                        t2i_md = f"![]({rebased})"
                metrics_parts: list[str] = []
                if cell.t2i_wall_clock_s is not None:
                    metrics_parts.append(
                        _format_wall_clock(cell.t2i_wall_clock_s)
                    )
                if cell.t2i_vram_peak_mib is not None:
                    metrics_parts.append(
                        f"VRAM {cell.t2i_vram_peak_mib:.0f}MiB"
                    )
                if cell.t2i_ram_peak_mib is not None:
                    metrics_parts.append(
                        f"RAM {cell.t2i_ram_peak_mib:.0f}MiB"
                    )
                if cell.t2i_gpu_power_avg_w is not None:
                    metrics_parts.append(
                        f"{cell.t2i_gpu_power_avg_w:.0f}W"
                    )
                metrics_md = (
                    " · ".join(metrics_parts) if metrics_parts else "—"
                )
                i2v_md = "—"
                if cell.i2v_relative_path is not None:
                    rebased_i2v = _rebase_artifact_path(
                        cell.i2v_relative_path,
                        cell.summary_source_dir, output_dir,
                    )
                    if rebased_i2v is not None:
                        # WEBP is image format with animation support;
                        # <img> renders inline + auto-plays in GitHub/VSCode.
                        i2v_md = (
                            f'<img src="{rebased_i2v}" alt="i2v" width="300">'
                        )
                elif (
                    cell.error_message is not None
                    and "i2v" in cell.error_message
                ):
                    i2v_md = "! i2v_failed"
                lines.append(
                    f"| {host} | {t2i_md} | {metrics_md} | {i2v_md} |"
                )
            lines.append("")

    # 7. Errors / anomalies.
    error_cells = [
        c for c in cells
        if c.status != "success"
        or (c.error_message is not None and "i2v" in c.error_message)
    ]
    if error_cells:
        lines.append("## Errors / Anomalies")
        lines.append("")
        for c in error_cells:
            msg = (c.error_message or "(none)")[:200]
            lines.append(
                f"- `{c.host}/{c.variant_family}:{c.variant_precision}/"
                f"{c.aspect_ratio.name}` status=`{c.status}` — {msg}"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("rendered gallery markdown to %s", output_path)


# ============================================================================
# Chart renderers (stubs — Sub-tarefa 2 implements)
# ============================================================================

def _format_wall_clock(seconds: float) -> str:
    """Compact wall-clock annotation for heatmap cells."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _find_cell(
    cells: tuple[GalleryGridCell, ...],
    host: str,
    family: str,
    precision: str,
    aspect_name: str,
) -> GalleryGridCell | None:
    """Linear scan for a cell by composite key. O(N) — fine for N≤99."""
    for c in cells:
        if (
            c.host == host
            and c.variant_family == family
            and c.variant_precision == precision
            and c.aspect_ratio.name == aspect_name
        ):
            return c
    return None


def _render_chart_cross_cg_wallclock_matrix(
    cells: tuple[GalleryGridCell, ...],
    output_path: Path,
) -> None:
    """Heatmap: rows = (variant_family:precision × aspect), cols = host.

    Color = ``t2i_wall_clock_s`` log-scale (vmin=1, vmax=5000 covers
    the 4.4s–4029s observed range). Sequential ``viridis`` cmap;
    ``cmap.set_bad`` colors masked cells light gray. Per-cell annotation
    formatted as ``<N>s`` / ``<M>m<S>`` / ``<H>h<M>m`` for readability.
    Text contrast: white if value > median, black otherwise.
    """
    t2i_variants = [v for v in VARIANT_CATALOG if not v.is_video]

    row_labels: list[str] = []
    matrix_rows: list[list[float]] = []
    for variant in t2i_variants:
        for aspect in DEFAULT_ASPECT_RATIOS:
            row_labels.append(
                f"{variant.family}:{variant.precision} ({aspect.name})"
            )
            row: list[float] = []
            for host in HOST_ORDER:
                cell = _find_cell(
                    cells, host, variant.family, variant.precision, aspect.name,
                )
                if cell is None or cell.t2i_wall_clock_s is None:
                    row.append(float("nan"))
                else:
                    row.append(cell.t2i_wall_clock_s)
            matrix_rows.append(row)

    matrix = np.ma.masked_invalid(np.array(matrix_rows))

    fig, ax = plt.subplots(figsize=(11, 18))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#cccccc")
    norm = LogNorm(vmin=1, vmax=5000)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(HOST_ORDER)))
    ax.set_xticklabels(HOST_ORDER)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    flat_vals = matrix.compressed()
    median_val = float(np.median(flat_vals)) if flat_vals.size else 1.0
    for i in range(len(row_labels)):
        for j in range(len(HOST_ORDER)):
            v = matrix[i, j]
            if np.ma.is_masked(v):
                ax.text(
                    j, i, "N/A", ha="center", va="center",
                    color="#666666", fontsize=8,
                )
            else:
                color = "white" if float(v) > median_val else "black"
                ax.text(
                    j, i, _format_wall_clock(float(v)),
                    ha="center", va="center", color=color, fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="t2i wall-clock (s, log scale)")
    ax.set_title("Cross-CG t2i Wall-Clock Matrix")
    ax.set_xlabel("host")
    plt.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _render_chart_vram_ram_offload(
    cells: tuple[GalleryGridCell, ...],
    output_path: Path,
) -> None:
    """Scatter: x = vram_peak_mib, y = ram_peak_mib.

    Marker shape per host (o/s/^), color per variant family (6 tab10
    colors), point size proportional to ``t2i_wall_clock_s``. Diagonal
    ``y = 1.5 × x`` red dashed line marks the offload-dominated
    threshold (RAM peak > 1.5× VRAM peak — heuristic Bloco 22e.2a).
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    for c in cells:
        if c.t2i_vram_peak_mib is None or c.t2i_ram_peak_mib is None:
            continue
        size = max(20.0, min(500.0, (c.t2i_wall_clock_s or 0.0) * 0.5))
        ax.scatter(
            c.t2i_vram_peak_mib, c.t2i_ram_peak_mib,
            s=size,
            c=[FAMILY_COLORS.get(c.variant_family, (0.5, 0.5, 0.5, 1.0))],
            marker=HOST_MARKERS.get(c.host, "x"),
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Offload boundary y = 1.5x.
    xlim = ax.get_xlim()
    ax.plot(
        [xlim[0], xlim[1]],
        [1.5 * xlim[0], 1.5 * xlim[1]],
        "r--", alpha=0.5, label="RAM = 1.5 × VRAM (offload threshold)",
    )
    ax.set_xlim(xlim)

    # Legends.
    host_handles = [
        Line2D(
            [0], [0], marker=HOST_MARKERS[h],
            color="w", markerfacecolor="gray", markeredgecolor="black",
            markersize=10, label=h,
        )
        for h in HOST_ORDER
    ]
    family_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=FAMILY_COLORS[f], markeredgecolor="black",
            markersize=10, label=f,
        )
        for f in FAMILY_ORDER
    ]
    legend_host = ax.legend(
        handles=host_handles, loc="upper left", title="host (marker)",
    )
    ax.add_artist(legend_host)
    ax.legend(
        handles=family_handles, loc="lower right", title="family (color)",
    )

    ax.set_xlabel("t2i VRAM peak (MiB)")
    ax.set_ylabel("t2i RAM peak (MiB)")
    ax.set_title("VRAM × RAM Offload Signature (point size ∝ wall-clock)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _render_chart_gpu_power_efficiency(
    cells: tuple[GalleryGridCell, ...],
    output_path: Path,
) -> None:
    """Grouped bars: x = variant family:precision, y = joules/image.

    Joules/image = ``t2i_wall_clock_s × t2i_gpu_power_avg_w``,
    averaged across aspects per (variant, host). Lower = better.
    Y-axis log scale (range typically 1e3 J – 1e6 J across V2 cells).
    Bars colored by :data:`HOST_COLORS`.
    """
    # Aggregate per (variant_key, host) → mean joules-per-image.
    joules: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in cells:
        if c.t2i_wall_clock_s is None or c.t2i_gpu_power_avg_w is None:
            continue
        key = f"{c.variant_family}:{c.variant_precision}"
        joules[key][c.host].append(c.t2i_wall_clock_s * c.t2i_gpu_power_avg_w)

    variant_keys = [k for k in T2I_VARIANT_LABELS if k in joules]

    fig, ax = plt.subplots(figsize=(14, 8))
    n_hosts = len(HOST_ORDER)
    bar_width = 0.25
    x = np.arange(len(variant_keys))

    for i, host in enumerate(HOST_ORDER):
        ys: list[float] = []
        for key in variant_keys:
            vals = joules[key].get(host, [])
            ys.append(sum(vals) / len(vals) if vals else 0.0)
        ax.bar(
            x + i * bar_width, ys, bar_width,
            color=HOST_COLORS[host], label=host,
        )

    ax.set_yscale("log")
    ax.set_xticks(x + bar_width * (n_hosts - 1) / 2)
    ax.set_xticklabels(variant_keys, rotation=45, ha="right")
    ax.set_ylabel("Energy per image (joules, mean across aspects, log scale)")
    ax.set_title("GPU Energy per Image (lower = better)")
    ax.legend(title="host")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _render_chart_i2v_coverage(
    cells: tuple[GalleryGridCell, ...],
    output_path: Path,
) -> None:
    """Heatmap: rows = variant_family:precision, cols = host.

    Cell value = count (out of 3 aspects) where the cell has both
    t2i + i2v rendered (artifact_status="both"). Discrete colormap:
    0 = light gray, 1–2 = yellow gradient, 3 = green. Cells where
    at least one aspect had ``i2v_failed`` (status=success but
    ``"i2v"`` in ``error_message``) get a ``!`` warning marker
    appended to the annotation.
    """
    matrix = np.zeros((len(T2I_VARIANT_LABELS), len(HOST_ORDER)))
    failed_mark = np.zeros_like(matrix, dtype=bool)

    label_to_row = {label: i for i, label in enumerate(T2I_VARIANT_LABELS)}
    host_to_col = {host: i for i, host in enumerate(HOST_ORDER)}

    for c in cells:
        key = f"{c.variant_family}:{c.variant_precision}"
        if key not in label_to_row or c.host not in host_to_col:
            continue
        i = label_to_row[key]
        j = host_to_col[c.host]
        if c.artifact_status == "both":
            matrix[i, j] += 1
        elif (
            c.status == "success"
            and c.error_message is not None
            and "i2v" in c.error_message
        ):
            failed_mark[i, j] = True

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = ListedColormap(["#dddddd", "#fff3a0", "#ffea66", "#7ed957"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(HOST_ORDER)))
    ax.set_xticklabels(HOST_ORDER)
    ax.set_yticks(range(len(T2I_VARIANT_LABELS)))
    ax.set_yticklabels(T2I_VARIANT_LABELS)

    for i in range(len(T2I_VARIANT_LABELS)):
        for j in range(len(HOST_ORDER)):
            v = int(matrix[i, j])
            label = f"{v}/3"
            if failed_mark[i, j]:
                label += " !"
            ax.text(
                j, i, label, ha="center", va="center",
                color="black", fontsize=10,
            )

    fig.colorbar(
        im, ax=ax, ticks=[0, 1, 2, 3],
        label="cells with i2v output (of 3 aspects)",
    )
    ax.set_title(
        "i2v Coverage Matrix (! = at least one cell had i2v_failed)"
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


# ============================================================================
# Auto-derived insights (stubs — Sub-tarefa 2 implements)
# ============================================================================

def _derive_insights(cells: tuple[GalleryGridCell, ...]) -> list[str]:
    """Generate 5-10 narrative insight bullets from the cell statistics.

    All values are computed from ``cells`` — no hardcoding. Bullets
    cover: cross-host median t2i, fastest/slowest absolute, VRAM-
    offload regime classification, lowest joules/image, i2v coverage,
    status breakdown, largest cross-host delta per variant. Cells
    with ``None`` telemetry are skipped for the metric in question;
    bullets that can't be computed (e.g., zero cells with timing) are
    omitted rather than emitting "(N/A)".
    """
    insights: list[str] = []

    # 1. Per-host median t2i wall-clock.
    by_host_walls: dict[str, list[float]] = defaultdict(list)
    for c in cells:
        if c.t2i_wall_clock_s is not None:
            by_host_walls[c.host].append(c.t2i_wall_clock_s)
    medians: dict[str, float] = {
        h: median(walls) for h, walls in by_host_walls.items() if walls
    }
    if medians:
        sorted_hosts = sorted(medians.keys())
        parts = ", ".join(f"{h}={medians[h]:.1f}s" for h in sorted_hosts)
        insights.append(f"Median t2i wall-clock by host: {parts}")

    # 2. Pairwise speedup vs slowest host median.
    if len(medians) >= 2:
        slowest_host = max(medians, key=lambda h: medians[h])
        for h in sorted(medians.keys()):
            if h == slowest_host:
                continue
            ratio = medians[slowest_host] / medians[h]
            insights.append(
                f"{h} is {ratio:.2f}x faster (median t2i) than {slowest_host}"
            )

    # 3. Fastest + slowest absolute t2i cell.
    walls_with_cells = [
        (c, c.t2i_wall_clock_s) for c in cells if c.t2i_wall_clock_s is not None
    ]
    if walls_with_cells:
        fastest_c, fastest_w = min(walls_with_cells, key=lambda p: p[1])
        slowest_c, slowest_w = max(walls_with_cells, key=lambda p: p[1])
        insights.append(
            f"Fastest t2i: {fastest_c.host} {fastest_c.variant_family}:"
            f"{fastest_c.variant_precision} {fastest_c.aspect_ratio.name} "
            f"= {fastest_w:.1f}s"
        )
        insights.append(
            f"Slowest t2i: {slowest_c.host} {slowest_c.variant_family}:"
            f"{slowest_c.variant_precision} {slowest_c.aspect_ratio.name} "
            f"= {slowest_w:.1f}s ({slowest_w / fastest_w:.1f}x ratio)"
        )

    # 4. VRAM-offload regime classification (RAM > 1.5 * VRAM).
    offload_count = sum(
        1 for c in cells
        if c.t2i_ram_peak_mib is not None
        and c.t2i_vram_peak_mib is not None
        and c.t2i_vram_peak_mib > 0
        and c.t2i_ram_peak_mib > 1.5 * c.t2i_vram_peak_mib
    )
    total_with_mem = sum(
        1 for c in cells
        if c.t2i_ram_peak_mib is not None and c.t2i_vram_peak_mib is not None
    )
    if total_with_mem:
        insights.append(
            f"Offload-dominated regime (RAM > 1.5x VRAM): "
            f"{offload_count}/{total_with_mem} cells"
        )

    # 5. Lowest joules-per-image t2i (efficiency).
    energy = [
        (c, c.t2i_wall_clock_s * c.t2i_gpu_power_avg_w)
        for c in cells
        if c.t2i_wall_clock_s is not None
        and c.t2i_gpu_power_avg_w is not None
    ]
    if energy:
        eff_cell, joules = min(energy, key=lambda p: p[1])
        insights.append(
            f"Lowest joules/image (t2i): {eff_cell.host} "
            f"{eff_cell.variant_family}:{eff_cell.variant_precision} "
            f"{eff_cell.aspect_ratio.name} = {joules:.0f}J"
        )

    # 6. i2v coverage.
    i2v_ok = sum(1 for c in cells if c.artifact_status == "both")
    i2v_failed = sum(
        1 for c in cells
        if c.status == "success"
        and c.error_message is not None
        and "i2v" in c.error_message
    )
    if cells:
        pct = 100 * i2v_ok / len(cells)
        insights.append(
            f"i2v output coverage: {i2v_ok}/{len(cells)} cells "
            f"({pct:.0f}%); {i2v_failed} cells t2i_success + i2v_failed"
        )

    # 7. Status breakdown.
    by_status = Counter(c.status for c in cells)
    parts = ", ".join(f"{s}={n}" for s, n in by_status.most_common())
    insights.append(f"Status breakdown: {parts}")

    # 8. Largest cross-host t2i ratio per variant (fastest-host fastest
    # cell vs slowest-host fastest cell per variant).
    by_variant_host_best: dict[str, dict[str, float]] = defaultdict(dict)
    for c in cells:
        if c.t2i_wall_clock_s is None:
            continue
        key = f"{c.variant_family}:{c.variant_precision}"
        existing = by_variant_host_best[key].get(c.host)
        if existing is None or c.t2i_wall_clock_s < existing:
            by_variant_host_best[key][c.host] = c.t2i_wall_clock_s
    deltas: list[tuple[str, float]] = []
    for key, host_walls in by_variant_host_best.items():
        if len(host_walls) >= 2:
            mn = min(host_walls.values())
            mx = max(host_walls.values())
            if mn > 0:
                deltas.append((key, mx / mn))
    if deltas:
        deltas.sort(key=lambda p: -p[1])
        worst_variant, worst_ratio = deltas[0]
        insights.append(
            f"Largest cross-host t2i delta: {worst_variant} "
            f"= {worst_ratio:.1f}x range"
        )

    return insights


# ============================================================================
# Persistence (stubs — Sub-tarefa 2 implements)
# ============================================================================

def _save_data_json(
    report_summary: GalleryReportSummary,
    output_path: Path,
) -> None:
    """Serialize :class:`GalleryReportSummary` as pretty JSON (atomic).

    Mirrors :func:`installer.benchmark.gallery._save_summary_atomic`:
    write to ``<path>.tmp`` then :meth:`Path.replace` (atomic on
    POSIX + Windows since Python 3.3). A crash mid-render leaves the
    prior data file intact. Structured-data complement to the
    markdown — downstream tooling reads ``gallery_data.json`` rather
    than re-parsing the markdown.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(
            asdict(report_summary), f,
            indent=2, ensure_ascii=False, default=str,
        )
        f.write("\n")
    tmp_path.replace(output_path)
    logger.info("saved gallery data to %s", output_path)


# ============================================================================
# CLI
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for :func:`main` (ships full args in Sub-tarefa 1)."""
    parser = argparse.ArgumentParser(
        description=(
            "Gallery analysis + visual markdown report (V2). Ingests "
            "one or more gallery summary.json files + their per-cell "
            "PNG/WEBP artifacts; renders gallery_report.md (comparison "
            "grids with embedded images), 4 cross-cutting charts, and "
            "gallery_data.json for downstream tooling."
        ),
    )
    parser.add_argument(
        "--summary-glob",
        default="reports/gallery_*/summary.json",
        help=(
            "Glob pattern matching one or more gallery summary.json "
            "files. Default scans reports/gallery_*/summary.json. "
            "Multi-summary use is V3-reserved (consolidate dispatches)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory (default: "
            "reports/gallery_analysis/<UTC-timestamp>/)."
        ),
    )
    parser.add_argument(
        "--variant-filter",
        default=None,
        help=(
            "Optional regex on 'family:precision' "
            "(e.g. 'flux2:.*'); mirrors gallery's --variant-filter."
        ),
    )
    parser.add_argument(
        "--host-filter",
        default=None,
        help="Optional regex on host alias (e.g. 'cg-(4|5)090').",
    )
    parser.add_argument(
        "--include-charts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Render the 4 cross-cutting charts as PNGs and embed them "
            "in the markdown (default: True). Use --no-include-charts "
            "to produce a text-only report (faster, no matplotlib I/O)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover + load summaries, print the resolved plan + "
            "filtered cell count, exit without rendering."
        ),
    )
    return parser


_CHART_RENDERERS: dict[
    str, Callable[[tuple[GalleryGridCell, ...], Path], None]
] = {
    "cross_cg_wallclock_matrix": _render_chart_cross_cg_wallclock_matrix,
    "vram_ram_offload": _render_chart_vram_ram_offload,
    "gpu_power_efficiency": _render_chart_gpu_power_efficiency,
    "i2v_coverage": _render_chart_i2v_coverage,
}


def main() -> None:
    """End-to-end gallery report orchestrator (Bloco 22e Sub-tarefa 2b)."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    parser = _build_argparser()
    args = parser.parse_args()

    report_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(f"reports/gallery_analysis/{report_id}")
    )

    # 1. Discover + load summaries.
    paths = _discover_gallery_summaries(args.summary_glob)
    logger.info(
        "discovered %d summary file(s): %s",
        len(paths),
        [p.parent.name for p in paths],
    )

    pairs: list[tuple[GallerySummary, Path]] = []
    for p in paths:
        try:
            s = _load_gallery_summary(p)
        except GalleryReportConfigError as exc:
            logger.warning("skipping %s: %s", p, exc)
            continue
        pairs.append((s, p))

    if not pairs:
        raise GalleryReportConfigError(
            "no summaries loaded — all discovered files failed validation"
        )

    # 2. Build grid + filter.
    grid = _build_comparison_grid(pairs)
    grid = _apply_filters(grid, args.variant_filter, args.host_filter)
    logger.info(
        "comparison grid: %d cells (variant_filter=%r, host_filter=%r)",
        len(grid), args.variant_filter, args.host_filter,
    )

    # 3. Dry-run gate.
    if args.dry_run:
        print(f"\n=== GALLERY REPORT PLAN (dry-run, report_id={report_id}) ===")
        print(f"summary_glob:   {args.summary_glob}")
        print(f"summaries:      {len(pairs)} loaded")
        for s, p in pairs:
            print(f"  - {p} (gallery_id={s.gallery_id}, cells={len(s.cells)})")
        print(f"cells:          {len(grid)} (post-filter)")
        print(f"output_dir:     {output_dir}")
        print(f"include_charts: {args.include_charts}")
        return

    # 4. Derive insights.
    insights = _derive_insights(grid)

    # 5. mkdir output_dir + charts subdir.
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    if args.include_charts:
        charts_dir.mkdir(parents=True, exist_ok=True)

    # 6. Render charts (skip-and-continue per chart failure).
    rendered_charts: list[str] = []
    if args.include_charts:
        for name, fn in _CHART_RENDERERS.items():
            chart_path = charts_dir / f"{name}.png"
            try:
                fn(grid, chart_path)
                rendered_charts.append(name)
                logger.info(
                    "rendered chart %s.png (%d bytes)",
                    name, chart_path.stat().st_size,
                )
            except Exception as exc:  # noqa: BLE001 — skip-and-continue
                logger.warning(
                    "chart %s render failed: %r — skipping", name, exc,
                )

    # 7. Save data.json.
    report_summary = GalleryReportSummary(
        schema_version=1,
        report_id=report_id,
        config={
            "summary_glob": args.summary_glob,
            "output_dir": str(output_dir),
            "variant_filter": args.variant_filter,
            "host_filter": args.host_filter,
            "include_charts": args.include_charts,
        },
        source_summaries=tuple(str(p) for p in paths),
        cells=grid,
        rendered_charts=tuple(rendered_charts),
    )
    _save_data_json(report_summary, output_dir / "data.json")

    # 8. Render markdown.
    _render_markdown_gallery(
        grid,
        output_dir / "gallery_report.md",
        rendered_charts,
        insights,
        [str(p) for p in paths],
        report_id,
    )

    # 9. Final tally.
    md_path = output_dir / "gallery_report.md"
    data_path = output_dir / "data.json"
    print("\n=== GALLERY REPORT DONE ===", file=sys.stderr)
    print(f"output_dir:        {output_dir}", file=sys.stderr)
    print(f"cells:             {len(grid)}", file=sys.stderr)
    print(
        f"charts:            "
        f"{len(rendered_charts)}/{len(_CHART_RENDERERS)} rendered",
        file=sys.stderr,
    )
    print(
        f"gallery_report.md: {md_path.stat().st_size} bytes",
        file=sys.stderr,
    )
    print(
        f"data.json:         {data_path.stat().st_size} bytes",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
