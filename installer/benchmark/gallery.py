"""Benchmark gallery orchestrator — cross-product visual gallery (V2).

Single responsibility: orchestrate the V2 visual-gallery matrix
(host × model_variant × aspect_ratio) into a consolidated JSON. The
inner-loop execution (snapshot dispatch, workflow queue, history poll,
output download) is delegated to :mod:`installer.benchmark.runner`;
this module composes those primitives across a cross-product of
hosts, model variants, and aspect ratios.

V2 gallery semantics differ from V1 sweep (DA-008 mechanic):

- **1 run per combo**, not 5. Gallery focuses on visual comparison +
  capture timing/telemetry, not statistical rigor. Single ``seed=42``
  used across all combos.
- **3 aspect ratios** evaluated per (host, variant): 1:1 / 16:9 / 9:16.
- **Optional i2v chain** (default ON) — each t2i output may seed a
  chained i2v generation using the host-appropriate WAN variant
  (Sub-tarefa 2a :func:`_pick_i2v_variant`).
- **Expanded status taxonomy**: ``"success"`` | ``"oom_vram"`` |
  ``"oom_ram"`` | ``"timeout"`` | ``"error_other"`` | ``"skipped"``.
  Granular OOM classification differentiates VRAM vs RAM exhaustion
  for V2 offload-heavy workloads (Hunyuan 80 GiB RAM peak observed
  Bloco 22b.3 cg-4090 smoke).
- **Variant catalog hardcoded module-level** (:data:`VARIANT_CATALOG`)
  — variants carry workflow_name + unet_filename metadata that the
  models manifest does not, so a manifest-derived catalog is not
  feasible.

Bloco 22c Sub-tarefa 1 ships the dataclass/CLI scaffold only — the
helper function bodies raise :class:`NotImplementedError`. Sub-tarefa 2a
implements the pure-function plumbing (cross-product enumeration,
resolution/UNet injection, error classification, i2v variant picker);
Sub-tarefa 2b adds the I/O orchestration (``_execute_t2i`` +
``_execute_i2v_chain``) reusing
:func:`installer.benchmark.runner._spawn_snapshot_until_signal`,
:func:`installer.benchmark.runner._signal_snapshot_stop`,
:func:`installer.benchmark.runner._run_single`, and
:func:`installer.benchmark.interface.ComfyUIClient.upload_image`
(latter for i2v chain seeding via :class:`LoadImage`).

DA-007 dispatch SSH single-coordinator (reused from sweep.py).
DA-013 ComfyUI lifecycle via RDP (reinforced — V2 GGUF custom nodes
introduced new restart-required regression class Bloco 22b.3).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import gcd
from pathlib import Path
from typing import Any, Literal

from installer.benchmark import interface, runner

logger = logging.getLogger(__name__)


# Default stop-flag path on remote hosts (reused from runner.py to keep
# the snapshot-stop signaling contract in sync between sweep + gallery).
# Sub-tarefa 2b passes this to runner._signal_snapshot_stop().
_REMOTE_STOP_FLAG_DEFAULT: str = runner.REMOTE_STOP_FLAG_DEFAULT


# WAN canonical frame count (Bloco 22c default #2 — fixed cross-aspects
# for V2 simplicity; reused as length input on WanImageToVideo).
I2V_LENGTH_DEFAULT: int = 81


# Cross-thread lock guarding the shared cells list + atomic summary write
# (Bloco 22d.2.1 parallel refactor). Threads append to cells_shared and
# write summary.json under this lock; per-host I/O outside the lock runs
# concurrent across hosts.
_summary_lock = threading.Lock()


# ============================================================================
# Exceptions
# ============================================================================

class GalleryError(Exception):
    """Base class for all gallery module errors."""


class GalleryConfigError(GalleryError):
    """Raised when gallery CLI/configuration is invalid.

    Examples: ``--workflows-dir`` does not exist, ``--variant-filter``
    matches zero variants, malformed ``--aspect-ratios`` strings, etc.
    """


class GalleryExecutionError(GalleryError):
    """Raised when execution orchestration fails outside per-combo errors.

    Per-combo failures are captured as :class:`GalleryCell` with
    ``status != "success"`` (skip-and-continue, mirroring sweep.py).
    This exception is reserved for cross-cutting failures: pre-flight
    rejection of all hosts, output-dir creation failure, etc.
    """


# ============================================================================
# Type aliases
# ============================================================================

GalleryCellStatus = Literal[
    "success",
    "oom_vram",
    "oom_ram",
    "timeout",
    "error_other",
    "skipped",
]
"""Outcome label for a single (host, variant, aspect_ratio) combo.

V2 expansion of V1's ``"success" | "error" | "skipped"``. Granular
OOM classification (``"oom_vram"`` vs ``"oom_ram"``) and explicit
``"timeout"`` make visible the 3 distinct bottleneck regimes observed
in Bloco 22b.3 smokes (Hunyuan RAM-bound, FLUX.2 CPU-bound, Qwen
balanced). :func:`_classify_error` (Sub-tarefa 2a) maps exceptions
+ remote stderr signals to one of these labels.
"""


# ============================================================================
# Dataclasses (Bloco 22c spec)
# ============================================================================

@dataclass(frozen=True, slots=True)
class AspectRatio:
    """One aspect ratio enumerated per gallery cell.

    The ``name`` field is the human-readable label (e.g. ``"16:9"``)
    used in gallery presentation and as a key in the consolidated
    summary. ``width`` and ``height`` are injected at workflow build
    time via :func:`_inject_resolution` into the workflow's empty
    latent / image node (model-family-specific: ``EmptyLatentImage`` /
    ``EmptySD3LatentImage`` / ``EmptyFlux2LatentImage`` /
    ``EmptyHunyuanImageLatent`` / ``WanImageToVideo``).
    """

    name: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """One model variant enumerated per gallery cell.

    ``workflow_name`` is the basename of the JSON in
    ``installer/benchmark/workflows/`` used as the topology template;
    ``unet_filename`` is the diffusion-model filename injected via
    :func:`_inject_unet_name` (replaces the default committed in the
    workflow JSON). Multiple variants may share one ``workflow_name``
    when only the UNet filename differs (e.g. FLUX.2 Q4_K_M / Q8_0 /
    BF16 all reuse ``flux2_dev_gguf.json``).

    For video variants (``is_video=True``), :attr:`unet_filename_low`
    is the low-noise expert filename (WAN 2.2 dual-expert pattern;
    high-noise = :attr:`unet_filename`).
    """

    family: str
    precision: str
    workflow_name: str
    unet_filename: str
    is_video: bool = False
    unet_filename_low: str | None = None


@dataclass(frozen=True, slots=True)
class GalleryConfig:
    """Immutable configuration captured for the gallery summary header.

    Field order: required fields first, then defaulted (Python
    dataclass constraint). ``output_dir`` is required (resolved by
    :func:`main` before construction — default
    ``reports/gallery_<UTC-timestamp>`` is applied at the CLI layer).

    ``i2v_skip_hosts`` (Bloco 22d.2.1): per-host opt-out from the
    i2v chain even when ``i2v_chain`` is True globally. Use case:
    low-VRAM hosts (e.g. cg-3060 12 GiB) where WAN dual-expert
    offload is prohibitively slow / unstable. Hosts in this tuple
    still run t2i; only the i2v chain is skipped.
    """

    workflows_dir: Path
    hosts: tuple[str, ...]
    aspect_ratios: tuple[AspectRatio, ...]
    output_dir: Path
    seed: int = 42
    workflow_timeout_s: int = 7200
    i2v_chain: bool = True
    variant_filter: str | None = None
    host_filter: str | None = None
    i2v_skip_hosts: tuple[str, ...] = ()
    retry_statuses: tuple[GalleryCellStatus, ...] = ()


@dataclass(frozen=True, slots=True)
class GalleryCell:
    """Outcome of one (host, variant, aspect_ratio) combo execution.

    Telemetry fields are populated when the corresponding phase
    succeeded; on failure they are ``None`` and :attr:`error_message`
    carries the diagnostic. ``i2v_*`` fields remain ``None`` when
    :attr:`status` != ``"success"`` for the t2i phase, when
    :attr:`GalleryConfig.i2v_chain` is ``False``, or when no viable
    i2v variant exists for the host
    (:func:`_pick_i2v_variant` returns ``None``).

    Output paths are stored relative to :attr:`GalleryConfig.output_dir`
    for portability — gallery summaries remain valid after the output
    directory is renamed/moved.
    """

    host: str
    variant_family: str
    variant_precision: str
    aspect_ratio_name: str
    status: GalleryCellStatus
    t2i_wall_clock_s: float | None
    t2i_vram_peak_mib: float | None
    t2i_ram_peak_mib: float | None
    t2i_gpu_util_avg_pct: float | None
    t2i_gpu_power_avg_w: float | None
    t2i_output_path: str | None
    i2v_wall_clock_s: float | None
    i2v_vram_peak_mib: float | None
    i2v_ram_peak_mib: float | None
    i2v_output_path: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class GallerySummary:
    """Consolidated gallery output (single JSON file at gallery end).

    Mirrors :class:`installer.benchmark.sweep.SweepSummary` shape but
    swaps the ``results``/``skipped`` split for a flat ``cells`` tuple
    — gallery skip outcomes are encoded as ``GalleryCell`` with
    ``status="skipped"`` (consistent taxonomy, single iteration).

    ``aspect_ratios`` and ``variants`` are denormalized into the
    summary so downstream readers (gallery report generator, future
    analyze.py extension) have the cross-product axes available
    without re-parsing config.
    """

    schema_version: int
    gallery_id: str
    config: dict[str, Any]
    aspect_ratios: tuple[AspectRatio, ...]
    variants: tuple[ModelVariant, ...]
    cells: tuple[GalleryCell, ...]


# ============================================================================
# Module-level catalogs
# ============================================================================

DEFAULT_ASPECT_RATIOS: tuple[AspectRatio, ...] = (
    AspectRatio("1:1", 1024, 1024),
    AspectRatio("16:9", 1280, 720),
    AspectRatio("9:16", 720, 1280),
)
"""Default 3-aspect spread evaluated per (host, variant).

Sized for V2 visual gallery storytelling: square (1024²) preserves
parity with V2 workflow defaults (matches the smoke baselines from
Bloco 22b.3); 16:9 / 9:16 swap dimensions to show landscape vs
portrait composition behavior. CLI override accepts ``WxH`` strings.
"""


VARIANT_CATALOG: tuple[ModelVariant, ...] = (
    # V1 t2i archetypes (kept in V2 gallery for cross-generation comparison).
    ModelVariant("sdxl", "fp16", "sdxl_base.json", "sd_xl_base_1.0.safetensors"),
    ModelVariant("flux1", "fp8", "flux_dev_fp8.json", "flux1-dev-fp8.safetensors"),
    ModelVariant("flux1", "fp16", "flux_dev_fp16.json", "flux1-dev.safetensors"),
    ModelVariant(
        "qwen-image",
        "fp8",
        "qwen_image_fp8.json",
        "qwen_image_fp8_e4m3fn.safetensors",
    ),
    # V2 t2i — Bloco 22b.2 production workflows.
    ModelVariant("flux2", "q4km", "flux2_dev_gguf.json", "flux2-dev-Q4_K_M.gguf"),
    ModelVariant("flux2", "q8", "flux2_dev_gguf.json", "flux2-dev-Q8_0.gguf"),
    ModelVariant("flux2", "bf16", "flux2_dev_gguf.json", "flux2-dev-BF16.gguf"),
    ModelVariant(
        "qwen-2512",
        "fp8",
        "qwen_image_2512.json",
        "qwen_image_2512_fp8_e4m3fn.safetensors",
    ),
    ModelVariant(
        "qwen-2512",
        "bf16",
        "qwen_image_2512.json",
        "qwen_image_2512_bf16.safetensors",
    ),
    ModelVariant(
        "hunyuan-2.1",
        "bf16",
        "hunyuan_image_21.json",
        "hunyuanimage2.1_bf16.safetensors",
    ),
    ModelVariant(
        "hunyuan-2.1",
        "fp8",
        "hunyuan_image_21.json",
        "hunyuanimage2.1_fp8_e4m3fn.safetensors",
    ),
    # i2v — WAN 2.2 dual-expert (high-noise + low-noise UNets). Reuses
    # the V1 wan22_i2v_fp8.json topology as template; gallery.py swaps
    # both unet_name and unet_name_low at runtime via
    # _inject_unet_name() per variant precision.
    ModelVariant(
        "wan22",
        "fp8",
        "wan22_i2v_fp8.json",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        is_video=True,
        unet_filename_low="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ),
    ModelVariant(
        "wan22",
        "fp16",
        "wan22_i2v_fp8.json",
        "wan2.2_i2v_high_noise_14B_fp16.safetensors",
        is_video=True,
        unet_filename_low="wan2.2_i2v_low_noise_14B_fp16.safetensors",
    ),
)
"""Hardcoded variant catalog: 11 t2i + 2 i2v = 13 entries (Bloco 22c spec).

Order matters for deterministic enumeration: V1 → V2 t2i (smallest to
largest) → i2v. Variant filtering (``--variant-filter``) is regex over
``f"{family}:{precision}"`` and preserves catalog order.
"""


# ============================================================================
# Helpers (stubs — Sub-tarefa 2a/2b implements)
# ============================================================================

def _enumerate_combinations(
    config: GalleryConfig,
    variants: Iterable[ModelVariant],
) -> list[tuple[str, ModelVariant, AspectRatio]]:
    """Cross-product ``hosts × t2i_variants × aspect_ratios``.

    i2v variants (:attr:`ModelVariant.is_video` ``True``) are EXCLUDED
    from this enumeration — they are chained off successful t2i outputs
    via :func:`_execute_i2v_chain` (Sub-tarefa 2b) and chosen per-host
    via :func:`_pick_i2v_variant`, not enumerated as standalone combos.

    Iteration order: host-major (host A all combos, then host B, etc.)
    matching :mod:`installer.benchmark.sweep` for VRAM-cache continuity.
    Combos are returned sorted by ``(host, family, precision, aspect_name)``
    so output is deterministic across invocations.

    :attr:`GalleryConfig.variant_filter` is applied as ``re.search`` to
    ``f"{family}:{precision}"``; :attr:`GalleryConfig.host_filter` is
    applied as ``re.search`` to each host alias. Empty result emits a
    WARNING but does not raise — :func:`main` can serialize an empty
    :class:`GallerySummary` gracefully.
    """
    t2i_variants = [v for v in variants if not v.is_video]

    if config.variant_filter is not None:
        variant_re = re.compile(config.variant_filter)
        t2i_variants = [
            v for v in t2i_variants
            if variant_re.search(f"{v.family}:{v.precision}")
        ]

    hosts = list(config.hosts)
    if config.host_filter is not None:
        host_re = re.compile(config.host_filter)
        hosts = [h for h in hosts if host_re.search(h)]

    combos: list[tuple[str, ModelVariant, AspectRatio]] = [
        (host, variant, aspect)
        for host in hosts
        for variant in t2i_variants
        for aspect in config.aspect_ratios
    ]
    combos.sort(key=lambda c: (c[0], c[1].family, c[1].precision, c[2].name))

    if not combos:
        logger.warning(
            "_enumerate_combinations: 0 combos after filtering "
            "(hosts=%d, t2i_variants=%d, aspect_ratios=%d)",
            len(hosts), len(t2i_variants), len(config.aspect_ratios),
        )

    return combos


_RESOLUTION_BEARING_CLASS_TYPES: frozenset[str] = frozenset({
    "EmptyLatentImage",       # SDXL
    "EmptySD3LatentImage",    # Qwen-Image (V1 + V2 2512)
    "EmptyFlux2LatentImage",  # FLUX.2
    "EmptyHunyuanImageLatent",  # Hunyuan-Image 2.1
    "WanImageToVideo",        # WAN 2.2 i2v (also carries `length`)
    "Flux2Scheduler",         # FLUX.2 scheduler — width/height must sync
                              # with EmptyFlux2LatentImage (PARTE 3 spec)
})


def _inject_resolution(
    workflow: dict[str, Any],
    width: int,
    height: int,
    length: int | None = None,
) -> dict[str, Any]:
    """Inject ``width``/``height`` into ALL resolution-bearing nodes.

    Scans for nodes whose ``class_type`` is in
    :data:`_RESOLUTION_BEARING_CLASS_TYPES` and updates each node's
    ``inputs.width`` / ``inputs.height``. Multiple matches are
    expected for some workflows — FLUX.2 has BOTH
    ``EmptyFlux2LatentImage`` (latent shape) AND ``Flux2Scheduler``
    (sigmas-aware dimensions); both must stay in sync per Bloco 22c
    PARTE 3 spec.

    For ``WanImageToVideo`` (i2v), if ``length`` is provided it is
    also injected into ``inputs.length``. Other class types ignore
    the ``length`` parameter even when present.

    Returns a deep copy; the input dict is not mutated.

    Raises:
        GalleryExecutionError: zero matching nodes found.
    """
    result = deepcopy(workflow)
    matches: list[str] = []
    for node_id, node in result.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in _RESOLUTION_BEARING_CLASS_TYPES:
            continue
        inputs = node.setdefault("inputs", {})
        inputs["width"] = width
        inputs["height"] = height
        if class_type == "WanImageToVideo" and length is not None:
            inputs["length"] = length
        matches.append(node_id)

    if not matches:
        raise GalleryExecutionError(
            "No resolution-bearing node found in workflow (looked for "
            f"{sorted(_RESOLUTION_BEARING_CLASS_TYPES)})"
        )
    return result


_LOADER_FIELD_MAP: dict[str, str] = {
    "UNETLoader": "unet_name",
    "UnetLoaderGGUF": "unet_name",
    "CheckpointLoaderSimple": "ckpt_name",
}


def _inject_unet_name(
    workflow: dict[str, Any],
    unet_filename: str,
    unet_filename_low: str | None = None,
) -> dict[str, Any]:
    """Inject ``unet_filename`` into the workflow's UNet/ckpt loader node.

    Recognized loader classes (mapped to their filename input field
    via :data:`_LOADER_FIELD_MAP`):

    - ``UNETLoader`` → ``inputs.unet_name``
    - ``UnetLoaderGGUF`` → ``inputs.unet_name``
    - ``CheckpointLoaderSimple`` → ``inputs.ckpt_name``

    For non-video variants (``unet_filename_low`` is ``None``):
    exactly 1 loader expected; ``unet_filename`` overwrites its
    filename field.

    For video dual-expert variants (``unet_filename_low`` is not
    ``None``, e.g. WAN 2.2 i2v): exactly 2 loaders expected.
    Mapping high-noise vs low-noise:

    - First pass: inspect the loader's current filename for
      ``"high"`` / ``"low"`` substring (case-insensitive). When both
      assignments succeed and refer to distinct loaders, that mapping
      wins.
    - Fallback: sort loaders by node_id ascending; lowest = high,
      next = low (canonical convention in V1 workflow JSON).

    Returns a deep copy; the input dict is not mutated.

    Raises:
        GalleryExecutionError: zero loaders found; or single-expert
            requested but multiple loaders present (ambiguous); or
            dual-expert requested but loader count != 2.
    """
    result = deepcopy(workflow)

    loaders: list[tuple[str, str, dict[str, Any]]] = []
    for node_id, node in result.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in _LOADER_FIELD_MAP:
            continue
        inputs = node.setdefault("inputs", {})
        loaders.append((node_id, class_type, inputs))

    if not loaders:
        raise GalleryExecutionError(
            "No UNet/checkpoint loader node found in workflow"
        )

    if unet_filename_low is None:
        if len(loaders) > 1:
            raise GalleryExecutionError(
                f"workflow has {len(loaders)} loader nodes but "
                "unet_filename_low is None — ambiguous injection target"
            )
        _, class_type, inputs = loaders[0]
        inputs[_LOADER_FIELD_MAP[class_type]] = unet_filename
        return result

    # Dual-expert path.
    if len(loaders) != 2:
        raise GalleryExecutionError(
            f"unet_filename_low provided but workflow has {len(loaders)} "
            "loader(s); expected exactly 2 for dual-expert i2v"
        )

    high_idx: int | None = None
    low_idx: int | None = None
    for i, (_, class_type, inputs) in enumerate(loaders):
        field = _LOADER_FIELD_MAP[class_type]
        original = inputs.get(field, "")
        if not isinstance(original, str):
            continue
        original_lower = original.lower()
        if "high" in original_lower:
            high_idx = i
        elif "low" in original_lower:
            low_idx = i

    if high_idx is None or low_idx is None or high_idx == low_idx:
        # Fallback: node_id ascending convention.
        sorted_indices = sorted(range(len(loaders)), key=lambda i: loaders[i][0])
        high_idx = sorted_indices[0]
        low_idx = sorted_indices[1]

    _, high_ct, high_inputs = loaders[high_idx]
    _, low_ct, low_inputs = loaders[low_idx]
    high_inputs[_LOADER_FIELD_MAP[high_ct]] = unet_filename
    low_inputs[_LOADER_FIELD_MAP[low_ct]] = unet_filename_low
    return result


def _classify_error(exc: Exception, error_message: str) -> GalleryCellStatus:
    """Map exception + remote stderr/error string to :data:`GalleryCellStatus`.

    Heuristic cascade (first match wins):

    1. :class:`installer.benchmark.interface.ComfyUITimeoutError`
       instance → ``"timeout"``.
    2. ``type(exc).__name__ == "OutOfMemoryError"`` (catches
       ``torch.cuda.OutOfMemoryError`` without requiring a coordinator-
       side ``torch`` import) → ``"oom_vram"``.
    3. :class:`MemoryError` instance (Python builtin, host-side RAM
       exhaustion) OR substring ``"MemoryError"`` in ``error_message``
       → ``"oom_ram"``.
    4. Substring ``"out of memory"`` in ``error_message.lower()``
       qualified by:
       - ``"cuda"`` or ``"vram"`` → ``"oom_vram"``
       - ``"ram"`` or ``"system"`` → ``"oom_ram"``
    5. Substring ``"cuda"`` AND ``"alloc"`` in ``error_message.lower()``
       (CUDA allocator failure proxy) → ``"oom_vram"``.
    6. Fallback → ``"error_other"`` (any
       :class:`installer.benchmark.runner.RunnerError`,
       :class:`installer.benchmark.interface.ComfyUIError`, or
       uncategorized exception lands here — never raises).

    The substring heuristics are best-effort proxies; the
    ``"error_other"`` fallback keeps the function total.
    """
    if isinstance(exc, interface.ComfyUITimeoutError):
        return "timeout"

    if type(exc).__name__ == "OutOfMemoryError":
        return "oom_vram"

    if isinstance(exc, MemoryError) or "MemoryError" in error_message:
        return "oom_ram"

    msg_lower = error_message.lower()

    if "out of memory" in msg_lower:
        if "cuda" in msg_lower or "vram" in msg_lower:
            return "oom_vram"
        if "ram" in msg_lower or "system" in msg_lower:
            return "oom_ram"

    if "cuda" in msg_lower and "alloc" in msg_lower:
        return "oom_vram"

    return "error_other"


def _pick_i2v_variant(
    host: str,
    t2i_aspect_ratio: AspectRatio,
    variants: Iterable[ModelVariant],
) -> ModelVariant | None:
    """Pick the canonical i2v variant viable for ``host``.

    Default policy (highest precision viable per host):

    - ``cg-3060`` (RTX 3060 8 GB) → WAN 2.2 fp8 (only fits — fp16
      dual UNet ~57 GiB exceeds even with full RAM offload).
    - ``cg-4090`` (RTX 4090 24 GB) → WAN 2.2 fp8 (fp16 fits with
      heavy offload but timing penalty is severe — Bloco 22b.3
      Hunyuan smoke showed 78.5 GiB RAM peak under similar pressure).
    - ``cg-5090`` (RTX 5090 32 GB) → WAN 2.2 fp16 (light offload,
      best visual quality for gallery).
    - Unknown host → fp8 (universal-fit fallback).

    Returns ``None`` only when ``variants`` contains no
    ``is_video=True`` entries (e.g., a filtered catalog opted out
    of i2v).

    ``t2i_aspect_ratio`` is reserved for future V3 heuristics
    (e.g. aspect-specific i2v selection when memory pressure dictates);
    currently unused by the default picker.
    """
    _ = t2i_aspect_ratio  # reserved for V3; currently part of signature only

    video_variants = [v for v in variants if v.is_video]
    if not video_variants:
        return None

    target_precision = "fp16" if host == "cg-5090" else "fp8"

    for v in video_variants:
        if v.precision == target_precision:
            return v

    # Fallback chain: target → fp8 (universal-fit) → first available.
    for v in video_variants:
        if v.precision == "fp8":
            return v

    return video_variants[0]


def _execute_t2i(
    host: str,
    variant: ModelVariant,
    aspect_ratio: AspectRatio,
    seed: int,
    workflows_dir: Path,
    output_dir: Path,
    timeout_s: int,
    client: interface.ComfyUIClient | None = None,
) -> GalleryCell:
    """Execute one t2i combo, capturing telemetry + output path.

    Sequence:
        1. Read workflow JSON at ``workflows_dir / variant.workflow_name``;
           inject resolution (:func:`_inject_resolution`) and UNet
           filename (:func:`_inject_unet_name`).
        2. Write the modified workflow to
           ``run_dir / "workflow_t2i.json"`` — this is the payload
           passed to :func:`runner._run_single` (workflow_path-based
           interface; seed is injected by runner on read). Persisting
           the payload makes the exact request reproducible
           post-mortem.
        3. Spawn snapshot.py via
           :func:`runner._spawn_snapshot_until_signal`.
        4. :func:`runner._run_single` (no ckpt sanity check —
           gallery's UNETLoader/UnetLoaderGGUF/CheckpointLoaderSimple
           injection paths cover all V1+V2 variants and ComfyUI
           validates ``unet_name`` itself).
        5. Download outputs via :func:`runner._download_outputs`;
           canonicalize the first PNG/WEBP to ``t2i.png``.
        6. Persist snapshot dict to ``run_dir / "snapshot_t2i.json"``.

    Returns a :class:`GalleryCell` with ``status="success"`` + t2i
    telemetry on success, or with status classified by
    :func:`_classify_error` on failure. ``i2v_*`` fields are
    ``None``; :func:`_execute_i2v_chain` is responsible for merging
    those when chained by :func:`main`.

    If ``client`` is ``None``, a new :class:`interface.ComfyUIClient`
    is constructed; callers managing per-host client caches (e.g.
    :func:`main`) should pass the cached client for HTTP keep-alive.
    """
    run_dir = output_dir / host / _cell_id(variant, aspect_ratio)
    run_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        client = interface.ComfyUIClient(f"http://{host}:8188", timeout=timeout_s)

    workflow_src_path = workflows_dir / variant.workflow_name
    try:
        workflow = json.loads(workflow_src_path.read_text(encoding="utf-8"))
        workflow = _inject_resolution(
            workflow, aspect_ratio.width, aspect_ratio.height,
        )
        workflow = _inject_unet_name(
            workflow, variant.unet_filename, variant.unet_filename_low,
        )
    except (OSError, json.JSONDecodeError, GalleryExecutionError) as exc:
        msg = f"{type(exc).__name__}: {exc}"
        return _empty_cell(
            host, variant, aspect_ratio,
            _classify_error(exc, msg), msg,
        )

    workflow_payload_path = run_dir / "workflow_t2i.json"
    workflow_payload_path.write_text(
        json.dumps(workflow, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    snapshot_proc: Any = None
    try:
        snapshot_proc = runner._spawn_snapshot_until_signal(host)
        run_result = runner._run_single(
            client=client,
            workflow_path=workflow_payload_path,
            ckpt_filename=None,
            seed=seed,
            snapshot_proc=snapshot_proc,
            stop_flag_remote_path=_REMOTE_STOP_FLAG_DEFAULT,
            host=host,
            workflow_timeout=timeout_s,
        )
    except (runner.RunnerError, interface.ComfyUIError) as exc:
        _terminate_snapshot_silently(host, snapshot_proc)
        msg = f"{type(exc).__name__}: {exc}"
        return _empty_cell(
            host, variant, aspect_ratio,
            _classify_error(exc, msg), msg,
        )
    except Exception as exc:  # noqa: BLE001 — last-resort skip-and-continue
        _terminate_snapshot_silently(host, snapshot_proc)
        logger.warning(
            "unexpected exception in t2i %s/%s:%s/%s: %r",
            host, variant.family, variant.precision, aspect_ratio.name, exc,
        )
        msg = f"unexpected: {type(exc).__name__}: {exc}"
        return _empty_cell(
            host, variant, aspect_ratio,
            _classify_error(exc, msg), msg,
        )

    # Output download (best-effort — telemetry is the primary signal).
    try:
        runner._download_outputs(client, run_result.outputs, run_dir)
    except Exception as exc:  # noqa: BLE001 — telemetry preserved
        logger.warning(
            "t2i output download failed for %s/%s:%s/%s: %r — telemetry preserved",
            host, variant.family, variant.precision, aspect_ratio.name, exc,
        )

    # Persist snapshot dict for post-mortem inspection.
    (run_dir / "snapshot_t2i.json").write_text(
        json.dumps(run_result.snapshot, indent=2),
        encoding="utf-8",
    )

    canonical_t2i = _canonicalize_primary_output(
        run_dir, prefix="t2i", extensions=_PRIMARY_OUTPUT_EXT_T2I,
    )
    t2i_output_rel: str | None = (
        canonical_t2i.relative_to(output_dir).as_posix()
        if canonical_t2i is not None
        else None
    )

    snap = run_result.snapshot
    return GalleryCell(
        host=host,
        variant_family=variant.family,
        variant_precision=variant.precision,
        aspect_ratio_name=aspect_ratio.name,
        status="success",
        t2i_wall_clock_s=run_result.wallclock_seconds,
        t2i_vram_peak_mib=_to_float(snap.get("peak_vram_mb")),
        t2i_ram_peak_mib=_to_float(snap.get("ram_peak_mib")),
        t2i_gpu_util_avg_pct=_to_float(snap.get("gpu_avg_utilization_pct")),
        t2i_gpu_power_avg_w=_to_float(snap.get("gpu_avg_power_w")),
        t2i_output_path=t2i_output_rel,
        i2v_wall_clock_s=None,
        i2v_vram_peak_mib=None,
        i2v_ram_peak_mib=None,
        i2v_output_path=None,
        error_message=None,
    )


def _execute_i2v_chain(
    host: str,
    t2i_output_path: Path,
    aspect_ratio: AspectRatio,
    seed: int,
    workflows_dir: Path,
    output_dir: Path,
    timeout_s: int,
    variants: Iterable[ModelVariant],
    client: interface.ComfyUIClient | None = None,
) -> dict[str, Any]:
    """Chain an i2v generation off a successful t2i output.

    Returns a dict with field-keys that :func:`main` merges into the
    parent :class:`GalleryCell` via :func:`dataclasses.replace`:

    - ``i2v_wall_clock_s``, ``i2v_vram_peak_mib``, ``i2v_ram_peak_mib``,
      ``i2v_output_path``: populated on success; ``None`` on failure.
    - ``error_message``: ``None`` on success, ``"i2v_<status>: ..."``
      prefixed string on failure (preserves the t2i ``status="success"``
      while surfacing the i2v fault — gallery cells represent t2i
      primary content, i2v is bonus chain).

    Sequence:
        1. :func:`_pick_i2v_variant` selects host-appropriate WAN
           variant (cg-5090 → fp16; otherwise fp8). Returns dict with
           ``error_message="no_i2v_variant_viable"`` if no video
           variant matched.
        2. Upload ``t2i_output_path`` to the executor via
           :meth:`interface.ComfyUIClient.upload_image`; canonical
           server name is injected into the workflow's ``LoadImage``
           node via :func:`_inject_load_image`.
        3. :func:`_inject_resolution` (with ``length=I2V_LENGTH_DEFAULT``)
           + :func:`_inject_unet_name` (dual-expert high+low).
        4. Write payload to ``run_dir / "workflow_i2v.json"``; spawn
           snapshot; :func:`runner._run_single`.
        5. Download + canonicalize to ``i2v.webp``; persist
           ``snapshot_i2v.json``.
    """
    i2v_variant = _pick_i2v_variant(host, aspect_ratio, variants)
    if i2v_variant is None:
        return {
            "i2v_wall_clock_s": None,
            "i2v_vram_peak_mib": None,
            "i2v_ram_peak_mib": None,
            "i2v_output_path": None,
            "error_message": "no_i2v_variant_viable",
        }

    # Cell's run_dir is the parent of t2i_output_path. Reuse so
    # workflow_i2v.json + i2v.webp + snapshot_i2v.json colocate with
    # the t2i artifacts (per Bloco 22c output dir layout spec).
    run_dir = t2i_output_path.parent

    if client is None:
        client = interface.ComfyUIClient(f"http://{host}:8188", timeout=timeout_s)

    try:
        uploaded_name = client.upload_image(t2i_output_path)
    except (interface.ComfyUIError, OSError, ValueError) as exc:
        return {
            "i2v_wall_clock_s": None,
            "i2v_vram_peak_mib": None,
            "i2v_ram_peak_mib": None,
            "i2v_output_path": None,
            "error_message": f"i2v_upload_failed: {type(exc).__name__}: {exc}",
        }

    try:
        workflow = json.loads(
            (workflows_dir / i2v_variant.workflow_name).read_text(encoding="utf-8")
        )
        workflow = _inject_resolution(
            workflow, aspect_ratio.width, aspect_ratio.height,
            length=I2V_LENGTH_DEFAULT,
        )
        workflow = _inject_unet_name(
            workflow, i2v_variant.unet_filename, i2v_variant.unet_filename_low,
        )
        workflow = _inject_load_image(workflow, uploaded_name)
    except (OSError, json.JSONDecodeError, GalleryExecutionError) as exc:
        return {
            "i2v_wall_clock_s": None,
            "i2v_vram_peak_mib": None,
            "i2v_ram_peak_mib": None,
            "i2v_output_path": None,
            "error_message": f"i2v_inject_failed: {type(exc).__name__}: {exc}",
        }

    workflow_payload_path = run_dir / "workflow_i2v.json"
    workflow_payload_path.write_text(
        json.dumps(workflow, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    snapshot_proc: Any = None
    try:
        snapshot_proc = runner._spawn_snapshot_until_signal(host)
        run_result = runner._run_single(
            client=client,
            workflow_path=workflow_payload_path,
            ckpt_filename=None,
            seed=seed,
            snapshot_proc=snapshot_proc,
            stop_flag_remote_path=_REMOTE_STOP_FLAG_DEFAULT,
            host=host,
            workflow_timeout=timeout_s,
        )
    except (runner.RunnerError, interface.ComfyUIError) as exc:
        _terminate_snapshot_silently(host, snapshot_proc)
        msg = f"{type(exc).__name__}: {exc}"
        return {
            "i2v_wall_clock_s": None,
            "i2v_vram_peak_mib": None,
            "i2v_ram_peak_mib": None,
            "i2v_output_path": None,
            "error_message": f"i2v_{_classify_error(exc, msg)}: {msg}",
        }
    except Exception as exc:  # noqa: BLE001 — last-resort skip-and-continue
        _terminate_snapshot_silently(host, snapshot_proc)
        logger.warning(
            "unexpected i2v exception on %s %s:%s/%s: %r",
            host, i2v_variant.family, i2v_variant.precision,
            aspect_ratio.name, exc,
        )
        return {
            "i2v_wall_clock_s": None,
            "i2v_vram_peak_mib": None,
            "i2v_ram_peak_mib": None,
            "i2v_output_path": None,
            "error_message": (
                f"i2v_unexpected: {type(exc).__name__}: {exc}"
            ),
        }

    try:
        runner._download_outputs(client, run_result.outputs, run_dir)
    except Exception as exc:  # noqa: BLE001 — telemetry preserved
        logger.warning(
            "i2v output download failed on %s %s:%s/%s: %r — telemetry preserved",
            host, i2v_variant.family, i2v_variant.precision,
            aspect_ratio.name, exc,
        )

    (run_dir / "snapshot_i2v.json").write_text(
        json.dumps(run_result.snapshot, indent=2),
        encoding="utf-8",
    )

    canonical_i2v = _canonicalize_primary_output(
        run_dir, prefix="i2v", extensions=_PRIMARY_OUTPUT_EXT_I2V,
    )
    i2v_output_rel: str | None = (
        canonical_i2v.relative_to(output_dir).as_posix()
        if canonical_i2v is not None
        else None
    )

    snap = run_result.snapshot
    return {
        "i2v_wall_clock_s": run_result.wallclock_seconds,
        "i2v_vram_peak_mib": _to_float(snap.get("peak_vram_mb")),
        "i2v_ram_peak_mib": _to_float(snap.get("ram_peak_mib")),
        "i2v_output_path": i2v_output_rel,
        "error_message": None,
    }


def _to_float(value: Any) -> float | None:
    """Coerce a snapshot dict value to ``float``, preserving ``None``."""
    return float(value) if value is not None else None


def _cell_id(variant: ModelVariant, aspect_ratio: AspectRatio) -> str:
    """Filesystem-safe cell directory name (colon in aspect → 'x')."""
    return (
        f"{variant.family}_{variant.precision}_"
        f"{aspect_ratio.name.replace(':', 'x')}"
    )


def _empty_cell(
    host: str,
    variant: ModelVariant,
    aspect_ratio: AspectRatio,
    status: GalleryCellStatus,
    error_message: str | None,
) -> GalleryCell:
    """Build a :class:`GalleryCell` with all telemetry/output fields ``None``."""
    return GalleryCell(
        host=host,
        variant_family=variant.family,
        variant_precision=variant.precision,
        aspect_ratio_name=aspect_ratio.name,
        status=status,
        t2i_wall_clock_s=None,
        t2i_vram_peak_mib=None,
        t2i_ram_peak_mib=None,
        t2i_gpu_util_avg_pct=None,
        t2i_gpu_power_avg_w=None,
        t2i_output_path=None,
        i2v_wall_clock_s=None,
        i2v_vram_peak_mib=None,
        i2v_ram_peak_mib=None,
        i2v_output_path=None,
        error_message=error_message,
    )


def _terminate_snapshot_silently(host: str, snapshot_proc: Any) -> None:
    """Best-effort termination of a remote snapshot subprocess + flag drop.

    Mirrors :func:`installer.benchmark.sweep._terminate_snapshot_silently`.
    Duplicated rather than imported to keep gallery a sibling of sweep
    (no inter-orchestrator dependency).
    """
    if snapshot_proc is None:
        return
    with contextlib.suppress(Exception):
        runner._signal_snapshot_stop(host, _REMOTE_STOP_FLAG_DEFAULT)
    try:
        snapshot_proc.communicate(timeout=10)
    except Exception:  # noqa: BLE001 — fall back to kill on any communicate failure
        with contextlib.suppress(Exception):
            snapshot_proc.kill()


_PRIMARY_OUTPUT_EXT_T2I: tuple[str, ...] = (".png", ".webp", ".jpg")
_PRIMARY_OUTPUT_EXT_I2V: tuple[str, ...] = (".webp", ".mp4", ".png")


def _canonicalize_primary_output(
    run_dir: Path,
    prefix: str,
    extensions: tuple[str, ...],
) -> Path | None:
    """Rename the first downloaded output of recognized extension to ``<prefix><ext>``.

    Scans ``run_dir`` for each extension in priority order. The first
    file whose stem differs from ``prefix`` (avoids self-overwrite on
    re-run) is renamed to ``run_dir / f"{prefix}{ext}"``. Returns the
    canonical Path on success, ``None`` if no candidate was found.
    """
    for ext in extensions:
        candidates = sorted(
            p for p in run_dir.glob(f"*{ext}")
            if p.is_file() and p.stem != prefix
        )
        if not candidates:
            continue
        canonical = run_dir / f"{prefix}{ext}"
        if canonical.exists():
            canonical.unlink()
        candidates[0].rename(canonical)
        return canonical
    return None


def _inject_load_image(workflow: dict[str, Any], image_name: str) -> dict[str, Any]:
    """Inject ``image_name`` into the workflow's ``LoadImage`` node.

    The i2v chain uploads the t2i PNG via
    :meth:`interface.ComfyUIClient.upload_image` (server returns its
    canonical name); this helper points the WAN workflow's
    :class:`LoadImage` at that canonical name.

    Returns a deep copy; the input dict is not mutated.

    Raises:
        GalleryExecutionError: zero or multiple LoadImage nodes
            (ambiguous injection target).
    """
    result = deepcopy(workflow)
    matches: list[str] = []
    for node_id, node in result.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "LoadImage":
            continue
        node.setdefault("inputs", {})["image"] = image_name
        matches.append(node_id)
    if not matches:
        raise GalleryExecutionError("No LoadImage node found in i2v workflow")
    if len(matches) > 1:
        raise GalleryExecutionError(
            f"Multiple LoadImage nodes ({len(matches)}); ambiguous injection target"
        )
    return result


def _derive_aspect_name(width: int, height: int) -> str:
    """Compute a human-readable aspect ratio name from ``WxH`` (e.g. 1280x720 → '16:9')."""
    divisor = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _parse_aspect_spec(spec: str) -> AspectRatio:
    """Parse a ``WxH`` CLI argument into :class:`AspectRatio`.

    Auto-derives the aspect ``name`` via greatest common divisor
    (e.g. ``1280x720`` → ``"16:9"``).

    Raises:
        GalleryConfigError: ``spec`` does not match ``^\\d+x\\d+$`` or
            either dimension is non-positive.
    """
    m = _ASPECT_RATIO_RE.match(spec)
    if m is None:
        raise GalleryConfigError(
            f"--aspect-ratios {spec!r} does not match WxH pattern (e.g. 1024x1024)"
        )
    width = int(m.group(1))
    height = int(m.group(2))
    if width <= 0 or height <= 0:
        raise GalleryConfigError(
            f"--aspect-ratios {spec!r}: width and height must both be positive"
        )
    return AspectRatio(_derive_aspect_name(width, height), width, height)


def _print_dry_run_plan(
    gallery_id: str,
    config: GalleryConfig,
    combos: list[tuple[str, ModelVariant, AspectRatio]],
) -> None:
    """Print the cross-product execution plan to stdout (no remote I/O)."""
    t2i_count = sum(1 for v in VARIANT_CATALOG if not v.is_video)
    i2v_count = sum(1 for v in VARIANT_CATALOG if v.is_video)
    print(f"\n=== GALLERY PLAN (--dry-run, gallery_id={gallery_id}) ===")
    print(f"output_dir:     {config.output_dir}")
    print(f"hosts:          {list(config.hosts)}")
    print(f"aspect_ratios:  {[a.name for a in config.aspect_ratios]}")
    print(
        f"catalog:        {t2i_count} t2i variants + {i2v_count} i2v variants"
    )
    print(
        f"filters:        variant_filter={config.variant_filter!r} "
        f"host_filter={config.host_filter!r}"
    )
    print(f"i2v_chain:      {config.i2v_chain}")
    print(f"seed:           {config.seed}")
    print(f"timeout_s:      {config.workflow_timeout_s}")

    print(
        f"\n{'idx':<5} {'host':<10} {'variant':<24} {'aspect':<8} action"
    )
    print(f"{'-' * 5} {'-' * 10} {'-' * 24} {'-' * 8} {'-' * 16}")
    for i, (host, variant, aspect) in enumerate(combos):
        label = f"{variant.family}:{variant.precision}"
        action = "execute_t2i" + (" + i2v_chain" if config.i2v_chain else "")
        print(f"{i + 1:<5} {host:<10} {label:<24} {aspect.name:<8} {action}")
    print(f"\ntotal combos: {len(combos)}")


def _log_cell_result(idx: int, total: int, cell: GalleryCell) -> None:
    """Compact INFO line summarizing one cell outcome (post-execution)."""
    base = (
        f"[{idx}/{total}] {cell.host} "
        f"{cell.variant_family}:{cell.variant_precision} "
        f"{cell.aspect_ratio_name} status={cell.status}"
    )
    if cell.status == "success":
        wall = cell.t2i_wall_clock_s
        vram = cell.t2i_vram_peak_mib
        ram = cell.t2i_ram_peak_mib
        logger.info(
            "%s t2i_wall=%.2fs vram_peak=%.0fMiB ram_peak=%.0fMiB",
            base,
            wall if wall is not None else 0.0,
            vram if vram is not None else 0.0,
            ram if ram is not None else 0.0,
        )
        if cell.i2v_wall_clock_s is not None:
            logger.info(
                "  i2v: wall=%.2fs vram_peak=%.0fMiB ram_peak=%.0fMiB",
                cell.i2v_wall_clock_s,
                cell.i2v_vram_peak_mib if cell.i2v_vram_peak_mib is not None else 0.0,
                cell.i2v_ram_peak_mib if cell.i2v_ram_peak_mib is not None else 0.0,
            )
        elif cell.error_message is not None:
            logger.warning("  i2v failed: %s", cell.error_message)
    else:
        logger.warning("%s error=%s", base, cell.error_message)


def _print_summary_tally(
    cells: list[GalleryCell],
    wall_s: float,
    summary_path: Path,
    resume_skipped: int = 0,
) -> None:
    """Final tally on stderr (status counts + wall-clock + summary path).

    ``wall_s`` measures THIS invocation's runtime only;
    ``resume_skipped`` is the count of cells inherited from a prior
    interrupted run (added to the tally for honesty about gallery
    completeness vs invocation cost).
    """
    status_counts: Counter[str] = Counter(c.status for c in cells)
    print("\n=== GALLERY DONE ===", file=sys.stderr)
    print(f"total cells:                  {len(cells)}", file=sys.stderr)
    if resume_skipped > 0:
        print(
            f"resume_skipped (prior run):   {resume_skipped}",
            file=sys.stderr,
        )
    print(
        f"wall_clock (this invocation): {wall_s:.1f}s ({wall_s / 60:.1f}min)",
        file=sys.stderr,
    )
    print(f"summary:                      {summary_path}", file=sys.stderr)
    print("\nstatus counts:", file=sys.stderr)
    for status, count in status_counts.most_common():
        print(f"  {status:<15} {count}", file=sys.stderr)


def _save_summary(summary: GallerySummary, output_path: Path) -> None:
    """Serialize :class:`GallerySummary` as pretty JSON (indent=2, UTF-8).

    Mirrors :func:`installer.benchmark.sweep._save_summary` exactly:
    ``asdict`` for dataclass→dict, ``indent=2`` for human-readable
    diffs, ``ensure_ascii=False`` so non-ASCII workflow strings
    (Chinese negative prompts in WAN) survive without escape mangling,
    ``default=str`` to coerce Path (in ``GalleryConfig``) and any
    non-JSON-native primitive into its repr — robust against future
    schema additions.

    Creates parent directories as needed and appends a trailing
    newline for POSIX-friendly diffs.

    Note: callers in the per-cell incremental save loop should prefer
    :func:`_save_summary_atomic` for crash tolerance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    logger.info("saved gallery summary to %s", output_path)


def _save_summary_atomic(summary: GallerySummary, output_path: Path) -> None:
    """Atomic write to ``output_path`` via ``<path>.tmp`` + ``Path.replace``.

    Crash-tolerant variant of :func:`_save_summary` used by
    :func:`main`'s incremental save loop. Writes to a sibling
    ``.tmp`` file then atomically replaces the target —
    :meth:`Path.replace` is atomic on both POSIX and Windows
    (Python 3.3+). On interruption mid-write, the prior
    ``summary.json`` remains intact; on interruption between
    write and replace, the ``.tmp`` may linger but the target
    is unchanged.

    Logs at DEBUG (not INFO) — called potentially hundreds of
    times per sweep, INFO would drown the cell-completion logs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    tmp_path.replace(output_path)
    logger.debug("atomic-saved gallery summary to %s", output_path)


def _load_resume_state(
    summary_path: Path,
    retry_statuses: Iterable[GalleryCellStatus] = (),
) -> tuple[str | None, list[GalleryCell], set[tuple[str, str, str, str]]]:
    """Load resume state from an existing ``summary.json``.

    Returns ``(original_gallery_id, existing_cells, completed_keys)``
    where ``completed_keys`` are
    ``(host, variant_family, variant_precision, aspect_ratio_name)``
    tuples used by :func:`main` to filter the combo enumeration.
    All three are empty/``None`` when ``summary_path`` does not exist
    (resume effectively becomes a no-op, fresh dispatch).

    ``retry_statuses`` (Bloco 22d.2.4): cells in the loaded summary
    whose ``status`` is in this iterable are TREATED AS NOT-YET-DONE:

    - They are NOT added to ``completed_keys`` (so the combo is
      re-enumerated for execution).
    - They are NOT added to ``existing_cells`` (so the worker's
      replacement-by-append produces a single up-to-date entry per
      key, not duplicates).

    Default empty tuple = previous behavior: ALL keys regardless of
    status are blocked from re-execution.

    Raises:
        GalleryConfigError: file exists but is corrupted (JSON parse
            error, missing/invalid ``gallery_id``, non-list ``cells``,
            or cell shape incompatible with :class:`GalleryCell`).
    """
    if not summary_path.is_file():
        return None, [], set()

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GalleryConfigError(
            f"--resume: summary at {summary_path} is corrupted (JSON parse): {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise GalleryConfigError(
            f"--resume: summary at {summary_path} is not a JSON object"
        )

    original_id = data.get("gallery_id")
    if not isinstance(original_id, str) or not original_id:
        raise GalleryConfigError(
            f"--resume: summary at {summary_path} missing valid 'gallery_id'"
        )

    raw_cells = data.get("cells", [])
    if not isinstance(raw_cells, list):
        raise GalleryConfigError(
            f"--resume: summary at {summary_path} has non-list 'cells' field"
        )

    retry_set = set(retry_statuses)
    existing_cells: list[GalleryCell] = []
    completed_keys: set[tuple[str, str, str, str]] = set()
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise GalleryConfigError(
                f"--resume: summary at {summary_path} has non-object cell entry"
            )
        try:
            cell = GalleryCell(**raw)
        except TypeError as exc:
            raise GalleryConfigError(
                f"--resume: cell {raw.get('host')!r}/"
                f"{raw.get('variant_family')!r}:{raw.get('variant_precision')!r} "
                f"shape mismatch (schema_version drift?): {exc}"
            ) from exc
        if cell.status in retry_set:
            # Drop from resume state — combo will be re-enumerated and
            # the retry result replaces this stale cell.
            continue
        existing_cells.append(cell)
        completed_keys.add((
            cell.host,
            cell.variant_family,
            cell.variant_precision,
            cell.aspect_ratio_name,
        ))

    return original_id, existing_cells, completed_keys


# ============================================================================
# Per-host worker (Bloco 22d.2.1 parallel)
# ============================================================================

def _run_host_combos(
    host: str,
    host_combos: list[tuple[str, ModelVariant, AspectRatio]],
    config: GalleryConfig,
    gallery_id: str,
    aspect_ratios: tuple[AspectRatio, ...],
    cells_shared: list[GalleryCell],
    progress_counter: dict[str, int],
    summary_path: Path,
) -> None:
    """Execute all combos assigned to one host (one ThreadPoolExecutor worker).

    Each ThreadPoolExecutor worker owns its host completely:

    - A dedicated :class:`interface.ComfyUIClient` (requests.Session is
      NOT thread-safe per requests docs; per-host client avoids sharing).
    - Per-cell ``runner._spawn_snapshot_until_signal`` + downloads —
      the remote SSH process and ComfyUI server are also per-host.

    The ONLY shared mutable state is :data:`cells_shared` and the
    on-disk ``summary.json``; both are guarded by :data:`_summary_lock`.
    Logger calls are thread-safe (logging module guarantees) so they
    can fire outside the lock.

    On per-cell failure (RunnerError / ComfyUIError / unexpected),
    the worker continues with the next combo. On unrecoverable
    exceptions outside the per-combo try/except, the worker re-raises
    so :func:`main`'s ``future.result()`` surfaces the fault in logs.
    """
    client = interface.ComfyUIClient(
        f"http://{host}:8188", timeout=config.workflow_timeout_s,
    )

    for combo_host, variant, aspect_ratio in host_combos:
        cell_label = (
            f"{combo_host} {variant.family}:{variant.precision} "
            f"{aspect_ratio.name}"
        )
        logger.info("[start] %s — t2i", cell_label)
        cell = _execute_t2i(
            host=combo_host,
            variant=variant,
            aspect_ratio=aspect_ratio,
            seed=config.seed,
            workflows_dir=config.workflows_dir,
            output_dir=config.output_dir,
            timeout_s=config.workflow_timeout_s,
            client=client,
        )

        if (
            cell.status == "success"
            and config.i2v_chain
            and combo_host not in config.i2v_skip_hosts
            and cell.t2i_output_path is not None
        ):
            logger.info("[start] %s — i2v chain", cell_label)
            i2v_data = _execute_i2v_chain(
                host=combo_host,
                t2i_output_path=config.output_dir / cell.t2i_output_path,
                aspect_ratio=aspect_ratio,
                seed=config.seed,
                workflows_dir=config.workflows_dir,
                output_dir=config.output_dir,
                timeout_s=config.workflow_timeout_s,
                variants=VARIANT_CATALOG,
                client=client,
            )
            cell = replace(cell, **i2v_data)

        with _summary_lock:
            cells_shared.append(cell)
            progress_counter["done"] += 1
            done = progress_counter["done"]
            total = progress_counter["total"]
            _save_summary_atomic(
                GallerySummary(
                    schema_version=1,
                    gallery_id=gallery_id,
                    config=asdict(config),
                    aspect_ratios=aspect_ratios,
                    variants=VARIANT_CATALOG,
                    cells=tuple(cells_shared),
                ),
                summary_path,
            )
            _log_cell_result(done, total, cell)


# ============================================================================
# CLI
# ============================================================================

_ASPECT_RATIO_RE = re.compile(r"^(\d+)x(\d+)$")


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for :func:`main` (ships full args in Sub-tarefa 1)."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark gallery orchestrator (V2). Iterates the cross-"
            "product (host × variant × aspect_ratio) in host-major order, "
            "1 run per combo. Optional i2v chain off successful t2i "
            "outputs. Skips per-combo failures with classified status "
            "(oom_vram / oom_ram / timeout / error_other). Delegates "
            "inner-loop execution to runner.py primitives."
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
        "--aspect-ratios",
        nargs="+",
        default=None,
        help=(
            "Aspect ratios as WxH strings (e.g. 1024x1024 1280x720 720x1280). "
            "Default: 1024x1024 1280x720 720x1280 (square + 16:9 + 9:16). "
            "Names ('1:1', '16:9', '9:16') are auto-derived from W/H."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for all combos (V2 = 1-run, no seed_base). Default 42.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ./reports/gallery_<UTC-timestamp>).",
    )
    parser.add_argument(
        "--workflow-timeout-s",
        type=int,
        default=7200,
        help=(
            "Per-combo workflow timeout in seconds (default: 7200 = 2h). "
            "Generous to accommodate heavy offload on cg-3060."
        ),
    )
    parser.add_argument(
        "--i2v-chain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Chain an i2v generation off each successful t2i output "
            "(default: True). Use --no-i2v-chain to skip."
        ),
    )
    parser.add_argument(
        "--variant-filter",
        default=None,
        help=(
            "Optional regex; only variants whose 'family:precision' "
            "string matches are included (e.g. 'flux2:.*', '.*:fp16')."
        ),
    )
    parser.add_argument(
        "--host-filter",
        default=None,
        help="Optional regex; only hosts whose alias matches are included.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cross-product plan without running anything.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted gallery — loads cells from "
            "<output-dir>/summary.json and skips combos already present "
            "(regardless of their status; no per-cell retry). Pair with "
            "--output-dir pointing at the existing gallery directory; "
            "without that, the default --output-dir uses a new timestamp "
            "and resume becomes a no-op (fresh dispatch). Preserves the "
            "original gallery_id from the loaded summary."
        ),
    )
    parser.add_argument(
        "--no-i2v-on-hosts",
        nargs="*",
        default=[],
        metavar="HOST",
        help=(
            "Skip the i2v chain on these specific hosts (t2i still runs). "
            "Use for low-VRAM hosts where WAN dual-expert offload is "
            "prohibitively slow / unstable (e.g. --no-i2v-on-hosts "
            "cg-3060). Hosts not in this list run the full chain when "
            "--i2v-chain is True. No effect when --no-i2v-chain is set."
        ),
    )
    parser.add_argument(
        "--retry-statuses",
        nargs="*",
        default=[],
        choices=["timeout", "error_other", "oom_vram", "oom_ram", "skipped"],
        metavar="STATUS",
        help=(
            "Status values that should be RE-EXECUTED on --resume instead "
            "of skipped. Default: empty (skip ALL prior cells regardless "
            "of status — original resume semantics). Example: "
            "'--retry-statuses timeout error_other' re-runs cells that "
            "previously timed out or errored. 'success' is intentionally "
            "excluded from choices (no use case for redoing successes)."
        ),
    )
    return parser


def main() -> None:
    """End-to-end gallery orchestrator (Bloco 22c Sub-tarefa 2b).

    Sequence:
        1. Parse args; resolve ``gallery_id`` (UTC timestamp) and
           ``output_dir`` (defaults to ``reports/gallery_<id>``).
        2. Parse ``--aspect-ratios`` strings → tuple[:class:`AspectRatio`,
           ...] (with GCD-derived ``name``). Default
           :data:`DEFAULT_ASPECT_RATIOS` applies when absent.
        3. Build :class:`GalleryConfig`; enumerate combos via
           :func:`_enumerate_combinations`.
        4. If ``--dry-run``: print plan + exit before any remote I/O.
        5. Pre-flight health-check each unique host. Unhealthy hosts
           contribute ``status="skipped"`` cells for all their combos.
        6. Per-combo loop:
           a. :func:`_execute_t2i`.
           b. If ``status="success"`` and ``i2v_chain`` enabled, chain
              via :func:`_execute_i2v_chain`; merge i2v fields into
              the cell via :func:`dataclasses.replace`.
        7. Build :class:`GallerySummary`; save to
           ``output_dir/summary.json`` via :func:`_save_summary`.
        8. Print tabular status tally to stderr.

    Best-effort ``git pull`` per healthy host (mirrors sweep pattern)
    is intentionally omitted in V1 gallery — the host should already
    be on the right commit from sweep dispatch; gallery is a follow-up
    visual pass, not a fresh dispatch.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    parser = _build_argparser()
    args = parser.parse_args()

    gallery_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    workflows_dir = Path(args.workflows_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(f"reports/gallery_{gallery_id}")
    )

    # Parse --aspect-ratios (or fall back to defaults).
    aspect_ratios: tuple[AspectRatio, ...]
    if args.aspect_ratios:
        aspect_ratios = tuple(_parse_aspect_spec(s) for s in args.aspect_ratios)
    else:
        aspect_ratios = DEFAULT_ASPECT_RATIOS

    config = GalleryConfig(
        workflows_dir=workflows_dir,
        hosts=tuple(args.hosts),
        aspect_ratios=aspect_ratios,
        output_dir=output_dir,
        seed=args.seed,
        workflow_timeout_s=args.workflow_timeout_s,
        i2v_chain=args.i2v_chain,
        variant_filter=args.variant_filter,
        host_filter=args.host_filter,
        i2v_skip_hosts=tuple(args.no_i2v_on_hosts),
        retry_statuses=tuple(args.retry_statuses),
    )

    combos = _enumerate_combinations(config, VARIANT_CATALOG)

    summary_path = output_dir / "summary.json"

    # Resume gate: optionally preload cells from prior interrupted run.
    existing_cells: list[GalleryCell] = []
    completed_keys: set[tuple[str, str, str, str]] = set()
    if args.resume:
        resumed_id, existing_cells, completed_keys = _load_resume_state(
            summary_path, retry_statuses=config.retry_statuses,
        )
        if resumed_id is not None:
            gallery_id = resumed_id
            retry_note = (
                f" (retry_statuses={list(config.retry_statuses)})"
                if config.retry_statuses
                else ""
            )
            logger.info(
                "--resume: continuing gallery_id=%s with %d completed cells from %s%s",
                gallery_id, len(completed_keys), summary_path, retry_note,
            )
        else:
            logger.info(
                "--resume: no existing summary at %s — starting fresh dispatch",
                summary_path,
            )

    # Filter combos to skip cells already completed (empty set in non-resume).
    filtered_combos = [
        combo for combo in combos
        if (combo[0], combo[1].family, combo[1].precision, combo[2].name)
        not in completed_keys
    ]
    resume_skipped = len(combos) - len(filtered_combos)

    logger.info(
        "gallery_id=%s combos=%d (filtered=%d, resume_skipped=%d) hosts=%s output_dir=%s",
        gallery_id, len(combos), len(filtered_combos), resume_skipped,
        config.hosts, output_dir,
    )

    if args.dry_run:
        _print_dry_run_plan(gallery_id, config, combos)
        return

    # Pre-flight health checks (unique hosts only, order-preserving).
    # Sequential is fine — 3 hosts × is_alive(timeout=5) is ~3s total.
    unique_hosts = list(dict.fromkeys(c[0] for c in filtered_combos))
    unhealthy_hosts: set[str] = set()
    for host in unique_hosts:
        url = f"http://{host}:8188"
        probe = interface.ComfyUIClient(url, timeout=5)
        alive = False
        try:
            alive = probe.is_alive()
        except Exception as exc:  # noqa: BLE001 — probe failures are non-fatal
            logger.warning("host %s pre-flight raised: %r", host, exc)
        if alive:
            logger.info("  [PRE-FLIGHT] %s OK", host)
        else:
            unhealthy_hosts.add(host)
            logger.warning("  [PRE-FLIGHT] %s UNHEALTHY", host)

    # Cells list seeded with existing_cells preserved from --resume load
    # (empty list in non-resume runs). Threads append to this list under
    # _summary_lock; atomic save after each append keeps the on-disk
    # summary at most one cell behind in-memory state.
    cells_shared: list[GalleryCell] = list(existing_cells)
    overall_start = time.monotonic()

    # Append "skipped" cells for unhealthy hosts BEFORE spawning threads
    # (no remote I/O needed; doing this synchronously simplifies the
    # parallel loop body — workers only iterate healthy hosts).
    for host in unique_hosts:
        if host not in unhealthy_hosts:
            continue
        for combo_host, variant, aspect_ratio in filtered_combos:
            if combo_host != host:
                continue
            cells_shared.append(_empty_cell(
                combo_host, variant, aspect_ratio,
                "skipped", "host_unhealthy_pre_flight",
            ))
            logger.info(
                "[skip] %s %s:%s %s — host unhealthy",
                combo_host, variant.family, variant.precision,
                aspect_ratio.name,
            )
    if unhealthy_hosts:
        _save_summary_atomic(
            GallerySummary(
                schema_version=1,
                gallery_id=gallery_id,
                config=asdict(config),
                aspect_ratios=aspect_ratios,
                variants=VARIANT_CATALOG,
                cells=tuple(cells_shared),
            ),
            summary_path,
        )

    # Group remaining combos by HEALTHY host for ThreadPoolExecutor dispatch.
    combos_by_host: dict[str, list[tuple[str, ModelVariant, AspectRatio]]] = {}
    for combo in filtered_combos:
        if combo[0] in unhealthy_hosts:
            continue
        combos_by_host.setdefault(combo[0], []).append(combo)

    # Progress counter: shared dict guarded by _summary_lock for atomic
    # increment + read. Total = original combos (including resume-skipped),
    # done starts at the resume-loaded count so the [done/total] log
    # reflects "X of 99" cumulative gallery progress, not invocation-only.
    progress_counter: dict[str, int] = {
        "done": len(existing_cells),
        "total": len(combos),
    }

    if combos_by_host:
        logger.info(
            "parallel dispatch: spawning %d host worker thread(s) — %s",
            len(combos_by_host),
            {h: len(cs) for h, cs in combos_by_host.items()},
        )
        with ThreadPoolExecutor(
            max_workers=len(combos_by_host),
            thread_name_prefix="gallery-host",
        ) as executor:
            futures = {
                executor.submit(
                    _run_host_combos,
                    host=host,
                    host_combos=host_combos,
                    config=config,
                    gallery_id=gallery_id,
                    aspect_ratios=aspect_ratios,
                    cells_shared=cells_shared,
                    progress_counter=progress_counter,
                    summary_path=summary_path,
                ): host
                for host, host_combos in combos_by_host.items()
            }
            for future in as_completed(futures):
                host = futures[future]
                try:
                    future.result()
                    logger.info("host %s worker completed", host)
                except Exception:
                    logger.exception(
                        "host %s worker raised unhandled exception", host,
                    )
    else:
        logger.info(
            "no healthy combos to dispatch — skipping ThreadPoolExecutor",
        )

    overall_wall = time.monotonic() - overall_start

    # Final save — idempotent if workers already wrote latest state.
    summary = GallerySummary(
        schema_version=1,
        gallery_id=gallery_id,
        config=asdict(config),
        aspect_ratios=aspect_ratios,
        variants=VARIANT_CATALOG,
        cells=tuple(cells_shared),
    )
    _save_summary_atomic(summary, summary_path)
    logger.info("saved gallery summary to %s", summary_path)

    _print_summary_tally(cells_shared, overall_wall, summary_path, resume_skipped)


if __name__ == "__main__":
    main()
