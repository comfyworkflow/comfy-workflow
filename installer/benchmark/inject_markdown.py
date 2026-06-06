"""Convert Format A workflows to Format B with embedded Note nodes.

Reads ``installer/benchmark/workflows/<wf>.json`` (Format A — the API
shape used directly by ``/prompt`` and consumed by ``gallery.py`` /
``sweep.py`` / ``runner.py``), converts to Format B (the frontend
"saved workflow" shape), injects a ``Note`` node at the top-right of
the canvas with the workflow's user-facing markdown bula (the same
content rendered into the sidecar ``<wf>.md``), and writes the result
to ``installer/benchmark/workflows_distribute/<wf>.json``.

Two-version workflow distribution pattern (closes débito V2 #23):

- **Format A (internal)** — ``installer/benchmark/workflows/`` —
  consumed by ``/prompt`` directly. Never touched by this tool.
- **Format B (distribute)** — ``installer/benchmark/workflows_distribute/``
  — shipped to audience via ``install-X.bat``. Frontend-renderable
  with embedded Note bula. ComfyUI's frontend converts B → A on Run
  (stripping the Note), so user gets a working graph.

The converter relies on a cached ``/object_info`` snapshot at
``installer/benchmark/_object_info_schema.json``. Refresh manually by
running:

    curl -sS http://cg-5090:8188/object_info \\
      > installer/benchmark/_object_info_schema.json

…when the ComfyUI version on cg-5090 changes (rare). The schema is
used to (a) classify each input as widget vs socket, (b) order
``widgets_values`` per the schema's ``input_order`` declaration, and
(c) populate ``outputs[]`` with correct names and types.

ComfyUI's frontend injects a synthetic ``control_after_generate``
widget after every ``seed`` / ``noise_seed`` widget. The /object_info
endpoint does NOT report this widget (it's frontend-only convenience).
We re-insert ``"randomize"`` at the right slot to preserve schema
alignment.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from installer.benchmark.update_workflow_links import (
    DEFAULT_GITHUB_BASE_URL,
    HARDWARE_TIERS,
    INSTALL_VIDEO_LABEL,
    INSTALL_VIDEO_MAPPING,
    INSTALL_VIDEO_NUMBER,
    PILLAR_MAPPING,
    _build_readme_text,
    _extract_params,
    _placeholder_install_url,
    _placeholder_pillar_url,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_INPUT_DIR: Path = Path("installer/benchmark/workflows")
DEFAULT_OUTPUT_DIR: Path = Path("installer/benchmark/workflows_distribute")
DEFAULT_SCHEMA_PATH: Path = Path("installer/benchmark/_object_info_schema.json")
DEFAULT_MANIFEST_PATH: Path = Path("installer/benchmark/models_manifest.yaml")


def load_video_urls_from_manifest(
    manifest_path: Path,
) -> tuple[dict[str, str], dict[int, str]]:
    """Read videos: section + return (install_urls, pillar_urls).

    install_urls maps install-mini-series slug → URL; pillar_urls maps
    Pillar number → URL. Empty / missing values fall back to the
    placeholder URL builders so .md sidecars / Format B Notes always
    have a clickable target (defaults: repo URL with stable
    ``#install-<slug>-pending`` / ``#video-pillar-<N>-pending``
    fragment).
    """
    import yaml as _yaml
    data = _yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    videos = data.get("videos", {}) if isinstance(data, dict) else {}
    if not isinstance(videos, dict):
        videos = {}

    from installer.benchmark.update_workflow_links import (
        INSTALL_VIDEO_NUMBER,
        _videos_yaml_key_for_install_slug,
    )
    install_urls: dict[str, str] = {}
    for slug in INSTALL_VIDEO_NUMBER:
        key = _videos_yaml_key_for_install_slug(slug)
        url = videos.get(key, "")
        url = url.strip() if isinstance(url, str) else ""
        install_urls[slug] = url or _placeholder_install_url(
            slug, DEFAULT_GITHUB_BASE_URL,
        )

    pillar_urls: dict[int, str] = {}
    for n in (1, 2, 3, 4, 5):
        url = videos.get(f"pillar_{n}", "")
        url = url.strip() if isinstance(url, str) else ""
        pillar_urls[n] = url or _placeholder_pillar_url(
            n, DEFAULT_GITHUB_BASE_URL,
        )
    return install_urls, pillar_urls

# Grid layout: left-to-right by topological depth, vertical stack within column.
COL_WIDTH: int = 360
ROW_HEIGHT: int = 220
ORIGIN_X: int = 0
ORIGIN_Y: int = 0
NOTE_WIDTH: int = 480
NOTE_HEIGHT: int = 320

FORMAT_B_VERSION: float = 0.4

# Widgets the frontend inserts after these names (not reported by /object_info).
_FRONTEND_CONTROL_AFTER: frozenset[str] = frozenset({"seed", "noise_seed"})
_FRONTEND_CONTROL_DEFAULT: str = "randomize"


# ============================================================================
# Aspect variants (Phase 1.5b)
# ============================================================================
#
# A workflow base can ship multiple aspect ratio variants. Each variant
# overrides EmptyLatentImage dimensions and (HD only) injects a
# LatentUpscaleBy + KSampler refiner chain into the Format A workflow
# before the Format A → Format B conversion. The audience receives N
# files per install (one per variant), all visible under the same
# sidebar subfolder, with cross-references in each Note pointing to
# the siblings.
#
# Workflows not in ASPECT_VARIANTS keep the legacy single-output
# behavior (zero behavior change for flux / qwen / hunyuan / wan).
#
# HD pipeline parameters chosen via Phase 1.5 audit
# (internal_docs/quality_audit/20260514T165032Z/sdxl_phase1.5/):
# 25 steps primary + 15 steps refiner at denoise 0.45, LatentUpscaleBy
# 1.428 with nearest-exact upscale_method.


@dataclass(frozen=True)
class VariantSpec:
    """One aspect/HD variant of a workflow base.

    ``slug`` is appended to the workflow stem to form the output
    filename (slug=``""`` → base filename unchanged). ``width`` /
    ``height`` set the EmptyLatentImage dimensions (the NATIVE bucket
    for HD variants; the upscaled target is computed via ``scale_by``).
    """
    slug: str
    aspect_id: str
    aspect_label: str
    width: int
    height: int
    hd: bool
    scale_by: float = 1.0           # Only used when hd=True.
    refiner_steps: int = 15         # Only used when hd=True.
    refiner_denoise: float = 0.45   # Only used when hd=True.
    upscale_method: str = "nearest-exact"  # Only used when hd=True.


_SDXL_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        slug="",
        aspect_id="1x1",
        aspect_label="1:1 square (1024×1024 native)",
        width=1024,
        height=1024,
        hd=False,
    ),
    VariantSpec(
        slug="_landscape",
        aspect_id="16x9",
        aspect_label="16:9 landscape (1344×768 native bucket)",
        width=1344,
        height=768,
        hd=False,
    ),
    VariantSpec(
        slug="_portrait",
        aspect_id="9x16",
        aspect_label="9:16 portrait (768×1344 native bucket)",
        width=768,
        height=1344,
        hd=False,
    ),
    VariantSpec(
        slug="_landscape_hd",
        aspect_id="landscape_hd",
        aspect_label="16:9 HD (1344×768 native → ~1920×1097 via latent upscale)",
        width=1344,
        height=768,
        hd=True,
        scale_by=1.428,
    ),
    VariantSpec(
        slug="_portrait_hd",
        aspect_id="portrait_hd",
        aspect_label="9:16 HD (768×1344 native → ~1097×1920 via latent upscale)",
        width=768,
        height=1344,
        hd=True,
        scale_by=1.428,
    ),
)


_QWEN_2512_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        slug="_1x1",
        aspect_id="1x1",
        aspect_label="1:1 square (1328×1328 native bucket)",
        width=1328,
        height=1328,
        hd=False,
    ),
    VariantSpec(
        slug="_landscape",
        aspect_id="16x9",
        aspect_label="16:9 landscape (1664×928 native bucket)",
        width=1664,
        height=928,
        hd=False,
    ),
    VariantSpec(
        slug="_portrait",
        aspect_id="9x16",
        aspect_label="9:16 portrait (928×1664 native bucket)",
        width=928,
        height=1664,
        hd=False,
    ),
    VariantSpec(
        slug="_landscape_hd",
        aspect_id="landscape_hd",
        aspect_label="16:9 HD (1664×928 native → ~1926×1074 via latent upscale)",
        width=1664,
        height=928,
        hd=True,
        scale_by=1.157,
    ),
    VariantSpec(
        slug="_portrait_hd",
        aspect_id="portrait_hd",
        aspect_label="9:16 HD (928×1664 native → ~1080×1935 via latent upscale)",
        width=928,
        height=1664,
        hd=True,
        scale_by=1.163,
    ),
)


ASPECT_VARIANTS: dict[str, tuple[VariantSpec, ...]] = {
    "sdxl_base": _SDXL_VARIANTS,
    # qwen_2512_fp8 source SWAPPED Comfy-Org → unsloth. The Comfy-Org
    # official fp8 mis-dequantizes under the v0.19+ comfy-kitchen kernel
    # dispatcher; unsloth/Qwen-Image-2512-FP8 is compatible. Visual matrix
    # 14/14 PASS validated. See
    # internal_docs/fp8_noise_fix/REPORT_fp8_noise_root_cause.md.
    "qwen_2512_fp8": _QWEN_2512_VARIANTS,
    "qwen_2512_bf16": _QWEN_2512_VARIANTS,
    # FLUX no longer aspect-expands: each model variant (fp16 / fp8 /
    # schnell / Q8 / Q4) is its own Format A workflow with the correct
    # loader pre-wired. Aspect-ratio swap is documented in the Note
    # text per variant (edit EmptySD3LatentImage width/height) — FLUX
    # is natively resolution-flexible so no extra files needed.
    # Future: hunyuan / wan variants land here when their quality
    # audits clear.
}


# Per-workflow "extra Note text" appended to the audience-facing Note
# body. Each FLUX variant gets its own positioning + sibling cross-ref +
# Phase 1.5 timing block. Keys are stems (e.g. ``"flux_dev_fp8"``).
#
# Phase 1.5 numbers below are P1 (Mustang) × 1024² × seed 42 cells —
# i.e. the exact prompt/seed/resolution the shipped workflow runs out
# of the box, so the audience can reproduce the timing directly.
# Source: internal_docs/quality_audit/20260516T141731Z/flux_phase1_5/
# (commit e70572c).
_FLUX_SIBLINGS_GUIDE: str = (
    "\n"
    "🔀 Other FLUX variants in this install — open the matching workflow:\n"
    "- `flux_dev_fp16.json` — V1 fp16, max precision. ~22 GB model.\n"
    "- `flux_dev_fp8.json` — V2 fp8, default production. Best balance.\n"
    "- `flux_schnell_fp8.json` — V3 schnell, 4-step draft. Fastest.\n"
)

_FLUX_RESOLUTION_HINT: str = (
    "\n"
    "📐 Resolution — FLUX is natively flexible: edit `EmptySD3LatentImage` "
    "width/height for other aspects. 16:9 HD: 1920×1080. 9:16 HD: "
    "1080×1920. (Native HD beat SDXL-style upscale-then-refine on every "
    "GPU in Phase 1.5.)\n"
)


def _flux_timing_block(cg_3060_s: int, cg_4090_s: int, cg_5090_s: int) -> str:
    return (
        "\n"
        "📊 Phase 1.5 timing (1024² · same prompt + seed):\n"
        f"- RTX 3060 — {cg_3060_s} s\n"
        f"- RTX 4090 — {cg_4090_s} s\n"
        f"- RTX 5090 — {cg_5090_s} s\n"
    )


EXTRA_NOTE_TEXT: dict[str, str] = {
    "flux_dev_fp16": (
        "\n"
        "🎯 **V1 fp16 — max precision.** UNETLoader at `default` dtype, "
        "20 steps, guidance 3.5. ~22 GB model on disk; slower than fp8 "
        "with marginal visible quality lift. Ship this when you need "
        "the reference signal (eval, fine-tune ground truth).\n"
        + _FLUX_SIBLINGS_GUIDE
        + _FLUX_RESOLUTION_HINT
        + _flux_timing_block(96, 47, 26)
    ),
    "flux_dev_fp8": (
        "\n"
        "🎯 **V2 fp8 — default production.** UNETLoader at "
        "`fp8_e4m3fn` dtype, 20 steps, guidance 3.5. ~16 GB model. "
        "Best balance of VRAM / speed / quality per Phase 1.5 — this "
        "is the recommended out-of-the-box workflow.\n"
        + _FLUX_SIBLINGS_GUIDE
        + _FLUX_RESOLUTION_HINT
        + _flux_timing_block(84, 32, 21)
    ),
    "flux_schnell_fp8": (
        "\n"
        "🎯 **V3 schnell — 4-step draft.** UNETLoader at `fp8_e4m3fn` "
        "dtype, only 4 steps, FluxGuidance set to 0. ~16 GB model. "
        "Fastest FLUX variant (~2× dev speed); aesthetic differs from "
        "dev — better for ideation passes than final renders.\n"
        + _FLUX_SIBLINGS_GUIDE
        + _FLUX_RESOLUTION_HINT
        + _flux_timing_block(31, 16, 12)
    ),
    # Q4/Q8 GGUF EXTRA_NOTE_TEXT entries dropped — see install-flux1.bat
    # comment in generate_install_scripts.py (operator constraint
    # "no Q-quants"). ComfyUI-GGUF custom node also removed.
}


# Qwen-Image install (vídeo #4). Phase 1.5 audit (commit b313b96) drove
# the ship list down to 2 variants — V2 fp8 default + V3 Lightning
# 4-step — because V1 bf16, V4 GGUF Q8, V5 GGUF Q4, V6 base fp8 all
# failed the "real gain or mandatory fallback" rule:
#   - V1 bf16: 2× slower than V2 fp8 with invisible quality lift
#   - V4/V5 GGUF (base, not 2512): Q-dequant tax beats offload savings
#     on cg-3060 (215 s warm for Q8 vs 150 s for fp8); Q4 even slower
#     than Q8 (dequant inversion).
#   - V6 base fp8: same speed as V2 but inferior composition (2512 is
#     a meaningful checkpoint update, not just a re-quantization).
#
# Phase 1.5 timing below is warm-mean (mean of seeds 43+44) at 1024²
# with P1 Mustang. Source: internal_docs/quality_audit/20260519T020647Z/
# qwen_image_phase1_5/.
_QWEN_2512_SIBLINGS_GUIDE: str = (
    "\n"
    "🔀 Other Qwen-Image 2512 workflows in this install (10 total — fp8 + bf16 × 5 aspects):\n"
    "- `qwen_2512_fp8_1x1.json` (1328×1328 · ~19 GB model · fits 16 GB VRAM)\n"
    "- `qwen_2512_fp8_landscape.json` (1664×928 · fp8)\n"
    "- `qwen_2512_fp8_portrait.json` (928×1664 · fp8)\n"
    "- `qwen_2512_fp8_landscape_hd.json` (1664×928 → 1920×1080 upscale · fp8)\n"
    "- `qwen_2512_fp8_portrait_hd.json` (928×1664 → 1080×1920 upscale · fp8)\n"
    "- `qwen_2512_bf16_1x1.json` (1328×1328 · ~38 GB model · reference precision)\n"
    "- `qwen_2512_bf16_landscape.json` (1664×928 · bf16)\n"
    "- `qwen_2512_bf16_portrait.json` (928×1664 · bf16)\n"
    "- `qwen_2512_bf16_landscape_hd.json` (1664×928 → 1920×1080 upscale · bf16)\n"
    "- `qwen_2512_bf16_portrait_hd.json` (928×1664 → 1080×1920 upscale · bf16)\n"
)

_QWEN_2512_NATIVE_BUCKETS: str = (
    "\n"
    "📐 Qwen-Image 2512 native aspect buckets — edit `EmptySD3LatentImage` "
    "width/height to switch within a workflow:\n"
    "- 1:1   →  1328×1328\n"
    "- 16:9  →  1664× 928 (landscape)\n"
    "- 9:16  →   928×1664 (portrait)\n"
    "- 4:3   →  1472×1104\n"
    "- 3:4   →  1104×1472\n"
    "- 3:2   →  1584×1056\n"
    "- 2:3   →  1056×1584\n"
)

_QWEN_2512_LIGHTNING_HOWTO: str = (
    "\n"
    "⚡ **Lightning 4-step mode** — 4× faster than Quality (50 steps), "
    "slight quality trade-off. To enable:\n"
    "1. Add a `LoraLoaderModelOnly` node between `UNETLoader` and "
    "`ModelSamplingAuraFlow` (right-click canvas → Add Node → loaders → "
    "LoraLoaderModelOnly).\n"
    "2. Wire `UNETLoader.MODEL` → `LoraLoaderModelOnly.model` → "
    "`ModelSamplingAuraFlow.model`.\n"
    "3. Set `lora_name` = `Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors`, "
    "`strength_model` = `1.0`.\n"
    "4. On `KSampler`: set `steps` = `4`, `cfg` = `1.0` (keep `euler` / `simple`).\n"
    "5. Run.\n"
)

_QWEN_2512_NEGATIVE_BILINGUAL: str = (
    "\n"
    "🚫 **Negative prompt (Chinese default).** The `CLIPTextEncode (Negative Prompt)` "
    "node ships with the canonical 12-clause guard:\n"
    "`低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲`\n"
    "\n"
    "Equivalent in English (if you prefer to swap):\n"
    "`low resolution, low quality, distorted limbs, distorted fingers, "
    "oversaturated, waxy texture, no facial detail, overly smooth, "
    "AI-looking, chaotic composition, blurry text, distorted`\n"
)

_QWEN_2512_FP8_TRAP_HINT: str = (
    "\n"
    "⚠️ **fp8 loader detail**: this workflow uses `UNETLoader` with "
    "`weight_dtype: \"default\"` — NOT `fp8_e4m3fn_fast`. The fast-GEMM "
    "path produces pure noise for Qwen-Image fp8 (silent failure — no "
    "error, just unrecognizable output). Keep `default`.\n"
)


_QWEN_SIBLINGS_GUIDE: str = (
    "\n"
    "🔀 Other Qwen-Image variant in this install — open the matching workflow:\n"
    "- `qwen_image_2512.json` — V2 fp8, 20-step default. Best balance.\n"
    "- `qwen_image_2512_lightning4.json` — V3 Lightning, 4 steps. 7.9× faster.\n"
)

_QWEN_RESOLUTION_HINT: str = (
    "\n"
    "📐 Resolution — Qwen-Image-2512 is natively flexible: edit "
    "`EmptySD3LatentImage` width/height for other aspects. 16:9 HD: "
    "1920×1080. 9:16 HD: 1080×1920. (Phase 1.5: native HD renders clean "
    "on every GPU — no duplication artifacts, no upscale path needed.)\n"
)

_QWEN_FP8_TRAP_HINT: str = (
    "\n"
    "⚠️ Loader detail: this workflow uses `UNETLoader` with "
    "`weight_dtype: \"default\"` — NOT `fp8_e4m3fn_fast`. The fast-GEMM "
    "path produces pure noise for Qwen-Image fp8 (silent failure — no "
    "error, just unrecognizable output). Stay on `default`.\n"
)


def _qwen_timing_block(cg_3060_s: int, cg_4090_s: int, cg_5090_s: int) -> str:
    return (
        "\n"
        "📊 Phase 1.5 timing (1024² · same prompt + seed · warm-mean):\n"
        f"- RTX 3060 — {cg_3060_s} s\n"
        f"- RTX 4090 — {cg_4090_s} s\n"
        f"- RTX 5090 — {cg_5090_s} s\n"
    )


EXTRA_NOTE_TEXT.update({
    "qwen_2512_fp8": (
        "\n"
        "🎯 **Qwen-Image 2512 fp8 — production default.** UNETLoader "
        "loading `qwen-image-2512-fp8.safetensors` (~19 GB, unsloth build). "
        "50 steps, CFG 4.0, euler/simple, ModelSamplingAuraFlow shift 3.1. "
        "FULL bf16 CLIP (`qwen_2.5_vl_7b.safetensors`, 16.6 GB). Fits "
        "16 GB VRAM with room for offload — runs on RTX 4060 Ti / 4070 / "
        "3060 12 GB tier and up.\n"
        + _QWEN_2512_SIBLINGS_GUIDE
        + _QWEN_2512_NATIVE_BUCKETS
        + _QWEN_2512_LIGHTNING_HOWTO
        + _QWEN_2512_NEGATIVE_BILINGUAL
        + "\n"
        "⚠️ **fp8 source detail**: this install ships unsloth's "
        "`qwen-image-2512-fp8.safetensors`. Comfy-Org's official fp8 "
        "(filename `qwen_image_2512_fp8_e4m3fn.safetensors`) is BROKEN "
        "under the ComfyUI v0.19+ `comfy-kitchen` kernel dispatcher → "
        "silent pure-noise output. Upstream open issues #11665, #11662, "
        "#11255, #12648. Do NOT manually swap to the Comfy-Org file. The "
        "unsloth quantization is compatible with current dispatcher.\n"
    ),
    "qwen_2512_bf16": (
        "\n"
        "🎯 **Qwen-Image 2512 bf16 — reference precision.** UNETLoader at "
        "`default` dtype loading `qwen_image_2512_bf16.safetensors` "
        "(~38 GB). 50 steps, CFG 4.0, euler/simple, ModelSamplingAuraFlow "
        "shift 3.1. FULL bf16 CLIP (`qwen_2.5_vl_7b.safetensors`, 16.6 GB). "
        "No quantization losses — recommended when output precision "
        "matters (eval, hero shots, ground truth). Needs 24 GB+ VRAM for "
        "comfortable iteration; 16 GB works with aggressive offload.\n"
        + _QWEN_2512_SIBLINGS_GUIDE
        + _QWEN_2512_NATIVE_BUCKETS
        + _QWEN_2512_LIGHTNING_HOWTO
        + _QWEN_2512_NEGATIVE_BILINGUAL
        + "\n"
        "⚠️ **fp8 source detail (if you load a sibling `qwen_2512_fp8_*` "
        "workflow)**: this install ships the unsloth quantization "
        "(`qwen-image-2512-fp8.safetensors` — HYPHENS). Do NOT manually "
        "swap to Comfy-Org's official `qwen_image_2512_fp8_e4m3fn.safetensors` "
        "(UNDERSCORES) — that file is BROKEN on ComfyUI v0.19+ due to a "
        "kernel dispatcher regression (silent pure-noise output). Upstream "
        "open issues #11665, #11662, #11255, #12648.\n"
    ),
})


_HUNYUAN_21_SIBLINGS_GUIDE: str = (
    "\n"
    "🔀 Other Hunyuan-Image 2.1 variants in this install:\n"
    "- `hunyuan_image_21_fp8.json` — fp8 Quality, 50 steps, recommended default (16 GB VRAM)\n"
    "- `hunyuan_image_21_bf16.json` — bf16 Quality, 50 steps, reference precision (32 GB VRAM)\n"
    "- `hunyuan_image_21_distilled_fp8.json` — 8-step distilled fp8, ~5× faster for iteration\n"
)

_HUNYUAN_21_NATIVE_BUCKETS: str = (
    "\n"
    "📐 Hunyuan-Image 2.1 is NATIVE 2K — do NOT generate at 1K (artifacts).\n"
    "Edit `EmptySD3LatentImage` width/height for other aspects:\n"
    "- 1:1   →  2048×2048\n"
    "- 16:9  →  2560×1536\n"
    "- 9:16  →  1536×2560\n"
    "- 4:3   →  2304×1792\n"
    "- 3:4   →  1792×2304\n"
    "- 3:2   →  2304×1536\n"
    "- 2:3   →  1536×2304\n"
)

_HUNYUAN_21_CONFIG_NOTE: str = (
    "\n"
    "⚙️ **Canonical config** (research-validated):\n"
    "- Sampler: `euler` · Scheduler: **`normal`** (Karras causes blurry artifacts on this model)\n"
    "- `ModelSamplingSD3` shift: **5** for full Quality, **4** for distilled\n"
    "- CFG: **3.5** for Quality, **3.25** for distilled\n"
    "- Dual text encoder: `qwen_2.5_vl_7b` (MLLM) + `byt5_small_glyphxl_fp16` (glyph-aware)\n"
    "- Negative prompt: leave empty (model doesn't use classifier-free guidance like SDXL)\n"
)


EXTRA_NOTE_TEXT.update({
    "hunyuan_image_21_fp8": (
        "\n"
        "🎯 **Hunyuan-Image 2.1 fp8 — production default.** UNETLoader at "
        "`fp8_e4m3fn` dtype loading `hunyuanimage2.1_fp8_e4m3fn.safetensors` "
        "(~17.4 GB). 50 steps, CFG 3.5, euler/normal, ModelSamplingSD3 shift 5. "
        "Dual text encoder pipeline (Qwen-2.5-VL MLLM + ByT5 glyph). Strength: "
        "rendering legible text directly on objects (signs, mugs, posters).\n"
        + _HUNYUAN_21_SIBLINGS_GUIDE
        + _HUNYUAN_21_NATIVE_BUCKETS
        + _HUNYUAN_21_CONFIG_NOTE
    ),
    "hunyuan_image_21_bf16": (
        "\n"
        "🎯 **Hunyuan-Image 2.1 bf16 — reference precision.** UNETLoader at "
        "`default` dtype loading `hunyuanimage2.1_bf16.safetensors` "
        "(~34.9 GB). 50 steps, CFG 3.5, euler/normal, shift 5. Same config "
        "as fp8 with reference-precision UNET. Needs RTX 5090 (32 GB) with "
        "block-swap; lower-VRAM cards fall back to fp8.\n"
        + _HUNYUAN_21_SIBLINGS_GUIDE
        + _HUNYUAN_21_NATIVE_BUCKETS
        + _HUNYUAN_21_CONFIG_NOTE
    ),
    "hunyuan_image_21_distilled_fp8": (
        "\n"
        "🎯 **Hunyuan-Image 2.1 distilled fp8 — 8-step meanflow.** UNETLoader "
        "at `fp8_e4m3fn` dtype loading "
        "`hunyuanimage2.1_distilled_fp8_e4m3fn.safetensors` (~17.5 GB). "
        "**8 steps**, CFG 3.25, euler/normal, **shift 4** (NOT 5). ~5× "
        "faster than Quality on RTX 5090 (~10 s warm vs ~60 s). Use for "
        "rapid iteration; promote winners to Quality for final renders.\n"
        + _HUNYUAN_21_SIBLINGS_GUIDE
        + _HUNYUAN_21_NATIVE_BUCKETS
        + _HUNYUAN_21_CONFIG_NOTE
    ),
})


# HiDream-I1 install (vídeo #7 / Benchmark Pillar #5). 17B hybrid
# transformer + LLaMA 3.1 text encoder, three distillation tiers sharing
# one 4-way QuadrupleCLIPLoader. Native 1024×1024 (1 MP) — contrast
# Hunyuan-Image 2.1 (native 2K). Recipe + anti-patterns are
# peer-verified in C:\KB\models\hidream-i1.md (Phase 0+ unified
# benchmark 2026-05-29, fp8 all-tiers PASS on cg-5090 / cg-4090 /
# cg-3060). Timing below is the KB benchmark (1024² · seed 42 cold /
# seed 43 warm), the exact prompt/seed/res the shipped workflow runs.
_HIDREAM_I1_SIBLINGS_GUIDE: str = (
    "\n"
    "🔀 Other HiDream-I1 tiers in this install (eixo distillation — same model, fewer steps):\n"
    "- `hidream_i1_full_fp8.json` — full Quality, 50 steps, uni_pc/simple, CFG 5.0. Best fidelity.\n"
    "- `hidream_i1_dev_fp8.json` — dev, 28 steps, lcm/normal, CFG 1.0. Balanced.\n"
    "- `hidream_i1_fast_fp8.json` — fast, 16 steps, lcm/normal, CFG 1.0. Fastest.\n"
)

_HIDREAM_I1_NATIVE_BUCKETS: str = (
    "\n"
    "📐 HiDream-I1 is NATIVE 1 MP (1024×1024) — NOT 2K (contrast Hunyuan-Image 2.1).\n"
    "Edit `EmptySD3LatentImage` width/height for the supported native buckets:\n"
    "- 1:1        →  1024×1024\n"
    "- 9:16 tall  →   768×1360\n"
    "- 16:9 wide  →  1360× 768\n"
    "- 3:4        →   880×1168\n"
    "- 4:3        →  1168× 880\n"
    "- 3:2        →  1248× 832\n"
    "- 2:3        →   832×1248\n"
)

_HIDREAM_I1_CONFIG_NOTE: str = (
    "\n"
    "⚙️ **Canonical config** (peer-verified — see anti-patterns):\n"
    "- `QuadrupleCLIPLoader` slot order: 1=`clip_l_hidream` · 2=`clip_g_hidream` "
    "· 3=`t5xxl_fp8_e4m3fn_scaled` · 4=`llama_3.1_8b_instruct_fp8_scaled`\n"
    "- `UNETLoader` weight_dtype: **`default`** (fp8 single-file already quantized)\n"
    "- VAE: `ae.safetensors` (FLUX VAE, shared)\n"
    "- full tier scheduler: **`simple`** (NOT normal) · dev/fast: `normal`\n"
    "- dev/fast CFG: **1.0** (distilled — no classifier-free guidance)\n"
    "\n"
    "🚫 Anti-patterns (do NOT do these):\n"
    "- Do NOT swap t5xxl to the non-scaled file — the **_scaled** variant is required.\n"
    "- Do NOT add a `type` parameter to `QuadrupleCLIPLoader` (not a valid input).\n"
    "- No GGUF Q-quants; no sageattention/bitsandbytes/triton wrappers.\n"
)

_HIDREAM_I1_TIMING_NOTE: str = (
    "\n"
    "📊 KB benchmark (1024² · seed 42 cold / seed 43 warm · Phase 0+ 2026-05-29):\n"
    "- full — RTX 3060: 407s · RTX 4090: 81s · RTX 5090: 73s (cold)\n"
    "- dev  — RTX 3060: 130s · RTX 4090: 77s · RTX 5090: 44s (cold)\n"
    "- fast — RTX 3060:  89s · RTX 4090: 32s · RTX 5090: 40s (cold)\n"
    "(RTX 3060 12 GB pays cold-ish cost every run — model offloads to RAM, no resident warm path.)\n"
)


EXTRA_NOTE_TEXT.update({
    "hidream_i1_full_fp8": (
        "\n"
        "🎯 **HiDream-I1 full fp8 — Quality default.** `UNETLoader` at "
        "`default` dtype loading `hidream_i1_full_fp8.safetensors` (~17 GB). "
        "**50 steps, CFG 5.0, uni_pc / simple**, ModelSamplingSD3 shift 3.0. "
        "Negative prompt optional (empty is fine on full). Highest fidelity "
        "of the three tiers — use for final/hero renders.\n"
        + _HIDREAM_I1_SIBLINGS_GUIDE
        + _HIDREAM_I1_NATIVE_BUCKETS
        + _HIDREAM_I1_CONFIG_NOTE
        + _HIDREAM_I1_TIMING_NOTE
    ),
    "hidream_i1_dev_fp8": (
        "\n"
        "🎯 **HiDream-I1 dev fp8 — balanced distilled.** `UNETLoader` at "
        "`default` dtype loading `hidream_i1_dev_fp8.safetensors` (~17 GB). "
        "**28 steps, CFG 1.0, lcm / normal**, ModelSamplingSD3 shift **6.0** "
        "(NOT 3.0 — dev uses a higher shift). Distilled: leave negative "
        "prompt empty (CFG 1.0 disables guidance). Faster than full with "
        "close quality — good production middle ground.\n"
        + _HIDREAM_I1_SIBLINGS_GUIDE
        + _HIDREAM_I1_NATIVE_BUCKETS
        + _HIDREAM_I1_CONFIG_NOTE
        + _HIDREAM_I1_TIMING_NOTE
    ),
    "hidream_i1_fast_fp8": (
        "\n"
        "🎯 **HiDream-I1 fast fp8 — fastest distilled.** `UNETLoader` at "
        "`default` dtype loading `hidream_i1_fast_fp8.safetensors` (~17 GB). "
        "**16 steps, CFG 1.0, lcm / normal**, ModelSamplingSD3 shift 3.0. "
        "Distilled: leave negative prompt empty. Fewest steps — use for "
        "rapid iteration, promote winners to full for final renders.\n"
        + _HIDREAM_I1_SIBLINGS_GUIDE
        + _HIDREAM_I1_NATIVE_BUCKETS
        + _HIDREAM_I1_CONFIG_NOTE
        + _HIDREAM_I1_TIMING_NOTE
    ),
})


# Per-workflow distribute subfolder under ``workflows_distribute/``.
# Mirrors the audience-facing ComfyUI sidebar category exposed by the
# install scripts. Workflows not listed land at the flat root (legacy
# behavior preserved for not-yet-shipped models). Uses forward slashes
# so the same string concatenates cleanly into the GitHub raw URL the
# install .bat composes in ``ship_workflow``.
WORKFLOW_DIST_SUBDIR: dict[str, str] = {
    "sdxl_base": "Image/SDXL",
    "flux_dev_fp16": "Image/FLUX",
    "flux_dev_fp8": "Image/FLUX",
    "flux_schnell_fp8": "Image/FLUX",
    "qwen_2512_fp8": "Image/Qwen",
    "qwen_2512_bf16": "Image/Qwen",
    "hunyuan_image_21_fp8": "Image/Hunyuan",
    "hunyuan_image_21_bf16": "Image/Hunyuan",
    "hunyuan_image_21_distilled_fp8": "Image/Hunyuan",
    "hidream_i1_full_fp8": "Image/HiDream",
    "hidream_i1_dev_fp8": "Image/HiDream",
    "hidream_i1_fast_fp8": "Image/HiDream",
    "z_image_turbo_bf16": "Image/ZImage",
    "z_image_turbo_fp8": "Image/ZImage",
    "klein_4b_base_bf16": "Image/FLUX.2 Klein",
    "klein_4b_distilled_bf16": "Image/FLUX.2 Klein",
    "klein_4b_base_fp8": "Image/FLUX.2 Klein",
    "klein_4b_distilled_fp8": "Image/FLUX.2 Klein",
}


def _resolve_output_path(
    output_dir: Path, workflow_filename: str,
) -> Path:
    """Map a workflow filename to its Format B output path (with subdir)."""
    stem = workflow_filename.removesuffix(".json")
    # Aspect-variant slug strip: sibling files share the base's subdir.
    for base_stem in WORKFLOW_DIST_SUBDIR:
        if stem == base_stem or stem.startswith(base_stem + "_"):
            return output_dir / WORKFLOW_DIST_SUBDIR[base_stem] / workflow_filename
    return output_dir / workflow_filename


def _extra_workflow_notes(workflow_name: str) -> str:
    """Return the variant-swap guide text for a workflow stem, or ``""``."""
    stem = workflow_name.removesuffix(".json")
    # Strip aspect-variant slug so siblings share the same swap guide.
    for v_map_key in EXTRA_NOTE_TEXT:
        if stem == v_map_key or stem.startswith(v_map_key + "_"):
            return EXTRA_NOTE_TEXT[v_map_key]
    return ""


def variant_filenames_for(workflow_name: str) -> list[str]:
    """Public helper: return the list of variant output filenames for a base.

    For workflows in :data:`ASPECT_VARIANTS`, expands to per-variant
    filenames (``<stem><slug>.json``). For workflows not in the map,
    returns ``[workflow_name]`` unchanged — preserves legacy
    single-output behavior so the install-X.bat generator can call this
    uniformly without conditional logic.
    """
    stem = workflow_name.removesuffix(".json")
    variants = ASPECT_VARIANTS.get(stem)
    if variants is None:
        return [workflow_name]
    return [f"{stem}{v.slug}.json" for v in variants]


# ============================================================================
# Schema loading + helpers
# ============================================================================

def _load_schema(schema_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached ``/object_info`` schema (per class_type → input/output)."""
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"schema at {schema_path} not a JSON object (top-level)"
        )
    return data


def _is_connection_ref(value: Any) -> bool:
    """True iff ``value`` looks like a Format A connection ref
    ``[<node_id>, <slot_int>]``."""
    if not isinstance(value, list) or len(value) != 2:
        return False
    src, slot = value
    if not isinstance(slot, int):
        return False
    if isinstance(src, int):
        return True
    return bool(isinstance(src, str) and src.isdigit())


def _get_input_order(schema_entry: dict[str, Any]) -> list[str]:
    """Return ordered list of input names from schema (required + optional)."""
    order = schema_entry.get("input_order", {})
    if isinstance(order, dict):
        names: list[str] = []
        names.extend(order.get("required", []))
        names.extend(order.get("optional", []))
        if names:
            return names
    # Fallback: required dict insertion order, then optional dict order.
    inputs = schema_entry.get("input", {})
    if not isinstance(inputs, dict):
        return []
    out: list[str] = []
    for section in ("required", "optional"):
        section_data = inputs.get(section, {})
        if isinstance(section_data, dict):
            out.extend(section_data.keys())
    return out


_SOCKET_PRIMITIVE_TYPES: frozenset[str] = frozenset({
    "INT", "FLOAT", "STRING", "BOOLEAN",
})


def _classify_input(spec: Any) -> str:
    """Classify an input spec as 'widget' or 'socket'.

    ComfyUI ``/object_info`` reports each input as a tuple-like list:
    ``[type, options_dict?]``. ``type`` is either:

    - a string (e.g. ``"MODEL"``, ``"INT"``, ``"FLOAT"``, ``"STRING"``,
      ``"CONDITIONING"``, ``"LATENT"`` …)
    - a list — combo / enum (always widget)

    Widgets are primitive types or combos. Sockets are everything else.
    """
    if not isinstance(spec, list) or not spec:
        return "widget"
    type_def = spec[0]
    if isinstance(type_def, list):
        return "widget"
    if not isinstance(type_def, str):
        return "widget"
    if type_def in _SOCKET_PRIMITIVE_TYPES:
        return "widget"
    return "socket"


def _widget_default(spec: Any) -> Any:
    """Return the default value for a widget input spec, or ``""`` if none."""
    if isinstance(spec, list) and len(spec) >= 2 and isinstance(spec[1], dict):
        return spec[1].get("default", "")
    return ""


def _patch_control_after_generate(
    widget_names: list[str], widgets_values: list[Any],
) -> tuple[list[str], list[Any]]:
    """Insert ``"randomize"`` after each ``seed`` / ``noise_seed`` widget.

    ComfyUI's frontend treats ``control_after_generate`` as a synthetic
    widget inserted alongside seed-style integer widgets. It is NOT
    reported by ``/object_info`` but IS expected in ``widgets_values``
    when loading Format B. Returns updated (names, values) tuples.
    """
    patched_names: list[str] = []
    patched_values: list[Any] = []
    for name, value in zip(widget_names, widgets_values, strict=True):
        patched_names.append(name)
        patched_values.append(value)
        if name in _FRONTEND_CONTROL_AFTER:
            patched_names.append("control_after_generate")
            patched_values.append(_FRONTEND_CONTROL_DEFAULT)
    return patched_names, patched_values


# ============================================================================
# Format A → Format B converter
# ============================================================================

def convert_a_to_b(
    workflow_a: dict[str, Any],
    schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Convert a Format A workflow dict to a Format B workflow dict.

    Walks Format A's nodes, separates sockets (linked) from widgets
    (literal values) using ``schema``, generates Format B
    ``nodes[] / links[] / outputs[]``, and grid-lays the canvas.

    Connection types come from the source node's output type in
    ``schema``; outputs[] names + types come from ``schema``'s
    ``output`` / ``output_name`` lists.

    Nodes whose ``class_type`` is missing from ``schema`` get
    minimal placeholders (type ``"*"`` everywhere) — those classes
    will display but may not run if the frontend strict-checks types.
    """
    nodes_b: list[dict[str, Any]] = []
    links_b: list[list[Any]] = []
    next_link_id = 1

    # node_id → slot_idx → list of consuming link_ids
    consumers: dict[int, dict[int, list[int]]] = {}

    # ----- Stage 1: build node entries (no link wiring yet) -----
    for str_id, node_a in workflow_a.items():
        if not isinstance(node_a, dict):
            continue
        try:
            node_id = int(str_id)
        except ValueError:
            continue
        class_type = node_a.get("class_type")
        if not isinstance(class_type, str):
            continue
        inputs_a = node_a.get("inputs", {})
        if not isinstance(inputs_a, dict):
            inputs_a = {}

        schema_entry = schema.get(class_type, {})
        input_order = _get_input_order(schema_entry)
        required = schema_entry.get("input", {}).get("required", {}) or {}
        optional = schema_entry.get("input", {}).get("optional", {}) or {}
        merged_specs = {**required, **optional}

        socket_names: list[str] = []
        widget_names: list[str] = []
        for name in input_order:
            spec = merged_specs.get(name)
            if spec is None:
                continue
            if _classify_input(spec) == "socket":
                socket_names.append(name)
            else:
                widget_names.append(name)

        format_b_inputs: list[dict[str, Any]] = []
        for sname in socket_names:
            format_b_inputs.append(
                {"name": sname, "type": "*", "link": None}
            )

        widgets_values: list[Any] = []
        for wname in widget_names:
            if wname in inputs_a:
                widgets_values.append(inputs_a[wname])
            else:
                widgets_values.append(_widget_default(merged_specs.get(wname, [])))

        # Frontend synthetic widget injection (control_after_generate).
        widget_names, widgets_values = _patch_control_after_generate(
            widget_names, widgets_values,
        )

        nodes_b.append({
            "id": node_id,
            "type": class_type,
            "pos": [0, 0],
            "size": [320, max(140, 28 * (len(format_b_inputs) + len(widgets_values) + 2))],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": format_b_inputs,
            "outputs": [],  # filled in stage 3
            "properties": {"Node name for S&R": class_type},
            "widgets_values": widgets_values,
        })

    node_by_id: dict[int, dict[str, Any]] = {n["id"]: n for n in nodes_b}
    socket_names_by_node: dict[int, list[str]] = {
        n["id"]: [i["name"] for i in n["inputs"]] for n in nodes_b
    }

    # ----- Stage 2: walk Format A inputs to wire links -----
    for str_id, node_a in workflow_a.items():
        if not isinstance(node_a, dict):
            continue
        try:
            node_id = int(str_id)
        except ValueError:
            continue
        if node_id not in node_by_id:
            continue
        inputs_a = node_a.get("inputs", {})
        if not isinstance(inputs_a, dict):
            continue
        sockets = socket_names_by_node.get(node_id, [])
        for input_name, value in inputs_a.items():
            if not _is_connection_ref(value):
                continue
            src_raw, src_slot = value
            try:
                src_id = int(src_raw)
            except (TypeError, ValueError):
                continue
            try:
                tgt_socket_idx = sockets.index(input_name)
            except ValueError:
                # Schema doesn't list this input as a socket — skip wiring;
                # the connection won't render visually but it's preserved
                # in the original Format A used by the runner.
                logger.warning(
                    "schema mismatch: node %d (%s) input '%s' not in "
                    "schema sockets; link not wired in Format B",
                    node_id, node_by_id[node_id]["type"], input_name,
                )
                continue
            link_id = next_link_id
            next_link_id += 1
            links_b.append(
                [link_id, src_id, int(src_slot), node_id, tgt_socket_idx, "*"]
            )
            node_by_id[node_id]["inputs"][tgt_socket_idx]["link"] = link_id
            consumers.setdefault(src_id, {}).setdefault(int(src_slot), []).append(link_id)

    # ----- Stage 3: populate outputs[] using schema + consumers -----
    for node_b in nodes_b:
        node_id = node_b["id"]
        class_type = node_b["type"]
        schema_entry = schema.get(class_type, {})
        out_types_raw = schema_entry.get("output", []) or []
        out_names_raw = schema_entry.get("output_name", []) or out_types_raw
        used = consumers.get(node_id, {})
        max_idx = max(len(out_types_raw), (max(used.keys()) + 1) if used else 0)
        outputs: list[dict[str, Any]] = []
        for slot_idx in range(max_idx):
            otype: str = (
                str(out_types_raw[slot_idx])
                if slot_idx < len(out_types_raw) else "*"
            )
            oname: str = (
                str(out_names_raw[slot_idx])
                if slot_idx < len(out_names_raw) else otype
            )
            outputs.append({
                "name": oname,
                "type": otype,
                "links": used.get(slot_idx, []),
                "slot_index": slot_idx,
            })
        node_b["outputs"] = outputs

    # ----- Stage 4: refine link.type from source output type -----
    for link in links_b:
        src_id, src_slot = int(link[1]), int(link[2])
        src_node = node_by_id.get(src_id)
        if not src_node:
            continue
        if src_slot < len(src_node["outputs"]):
            link[5] = src_node["outputs"][src_slot]["type"]
        dst_id, dst_slot = int(link[3]), int(link[4])
        dst_node = node_by_id.get(dst_id)
        if dst_node and dst_slot < len(dst_node["inputs"]):
            dst_node["inputs"][dst_slot]["type"] = link[5]

    # ----- Stage 5: grid layout -----
    _layout_grid(nodes_b, links_b)

    return {
        "last_node_id": max((n["id"] for n in nodes_b), default=0),
        "last_link_id": next_link_id - 1,
        "nodes": nodes_b,
        "links": links_b,
        "groups": [],
        "config": {},
        "extra": {},
        "version": FORMAT_B_VERSION,
    }


def _layout_grid(
    nodes_b: list[dict[str, Any]],
    links_b: list[list[Any]],
) -> None:
    """Position nodes by topological depth (longest path from source).

    Source nodes (no incoming links) → column 0. Each downstream node
    sits at ``max(predecessor_depth) + 1``. Within a column, nodes
    stack vertically in ``node_id`` order.
    """
    if not nodes_b:
        return
    incoming: dict[int, set[int]] = {n["id"]: set() for n in nodes_b}
    for link in links_b:
        src = int(link[1])
        dst = int(link[3])
        if dst in incoming and src != dst:
            incoming[dst].add(src)

    depth: dict[int, int] = {}
    # Iterative fixed-point — handles DAGs cleanly; non-DAGs settle
    # at whatever depth they reach (cycles get 0 for any unresolved).
    for _ in range(len(nodes_b) + 1):
        changed = False
        for n in nodes_b:
            nid = n["id"]
            preds = incoming.get(nid, set())
            if not preds:
                new_d = 0
            else:
                resolved = [depth[p] for p in preds if p in depth]
                if len(resolved) != len(preds):
                    continue
                new_d = max(resolved) + 1
            if depth.get(nid) != new_d:
                depth[nid] = new_d
                changed = True
        if not changed:
            break

    for n in nodes_b:
        depth.setdefault(n["id"], 0)

    by_depth: dict[int, list[dict[str, Any]]] = {}
    for n in nodes_b:
        by_depth.setdefault(depth[n["id"]], []).append(n)

    for d, nodes_at_depth in by_depth.items():
        for idx, n in enumerate(sorted(nodes_at_depth, key=lambda x: int(x["id"]))):
            n["pos"] = [
                ORIGIN_X + d * COL_WIDTH,
                ORIGIN_Y + idx * ROW_HEIGHT,
            ]


# ============================================================================
# Note injection
# ============================================================================

def inject_note(
    workflow_b: dict[str, Any],
    note_markdown: str,
) -> dict[str, Any]:
    """Append a ``Note`` node at the top-right of the canvas.

    Note is a frontend-only node (no inputs, no outputs); ComfyUI's
    ``graphToPrompt`` strips it before submitting to ``/prompt``, so
    execution is unaffected. We position it just above and to the
    right of the rightmost executable node so it doesn't overlap.
    """
    max_x = max(
        (int(n["pos"][0]) for n in workflow_b["nodes"]),
        default=0,
    )
    note_id = int(workflow_b.get("last_node_id", 0)) + 1
    workflow_b["last_node_id"] = note_id
    workflow_b["nodes"].append({
        "id": note_id,
        "type": "Note",
        "pos": [max_x + COL_WIDTH, ORIGIN_Y - NOTE_HEIGHT - 40],
        "size": [NOTE_WIDTH, NOTE_HEIGHT],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": "About this workflow",
        "properties": {},
        "widgets_values": [note_markdown],
        "color": "#432",
        "bgcolor": "#653",
    })
    return workflow_b


# ============================================================================
# Per-workflow pipeline
# ============================================================================

def _atomic_write(target: Path, content: str) -> bool:
    """Write content atomically. Returns True iff disk content changed."""
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    if existing == content:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)
    return True


# ============================================================================
# Variant Format A transforms + Note builder
# ============================================================================

def _apply_variant_to_workflow_a(
    workflow_a: dict[str, Any], variant: VariantSpec,
) -> dict[str, Any]:
    """Return a deep copy of ``workflow_a`` with the variant applied.

    Transforms applied:

    - ``EmptyLatentImage`` widgets_values → variant's
      ``(width, height, batch_size=1)`` (native bucket dims; HD upscale
      runs DOWNSTREAM in latent space).
    - HD variants only: inject ``LatentUpscaleBy`` (id ``"10"``) +
      ``KSampler`` refiner (id ``"11"``) + rewire ``VAEDecode``
      (id ``"8"``) to consume the refiner's LATENT output.

    Assumes the base workflow has the canonical sdxl_base.json topology
    (CheckpointLoaderSimple ``"4"``, EmptyLatentImage ``"5"``,
    CLIPTextEncode ``"6"`` positive / ``"7"`` negative, KSampler ``"3"``,
    VAEDecode ``"8"``, SaveImage ``"9"``).
    """
    out = copy.deepcopy(workflow_a)

    # Resize the workflow's empty-latent node (works on any node ID).
    # FLUX uses EmptySD3LatentImage; SDXL uses EmptyLatentImage.
    _EMPTY_LATENT_CLASSES = {"EmptyLatentImage", "EmptySD3LatentImage"}
    for node in out.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") in _EMPTY_LATENT_CLASSES:
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                inputs["width"] = variant.width
                inputs["height"] = variant.height

    if not variant.hd:
        return out

    # HD pipeline injection. Find the primary KSampler and copy its
    # config + input wires so the refiner mirrors it. Topology-agnostic:
    # the refiner re-uses the primary KSampler's model/positive/negative
    # refs (works for SDXL CheckpointLoader at "4" AND Qwen split-loader
    # topology where model passes through ModelSamplingAuraFlow).
    primary_ksampler_id: str | None = None
    primary_sampler = "dpmpp_2m"
    primary_scheduler = "karras"
    primary_cfg = 7.0
    primary_seed = 42
    primary_model_ref: list[Any] | None = None
    primary_positive_ref: list[Any] | None = None
    primary_negative_ref: list[Any] | None = None
    for nid, node in out.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "KSampler":
            primary_ksampler_id = nid
            inputs = node.get("inputs", {}) or {}
            if isinstance(inputs, dict):
                primary_sampler = str(inputs.get("sampler_name", primary_sampler))
                primary_scheduler = str(inputs.get("scheduler", primary_scheduler))
                primary_cfg = float(inputs.get("cfg", primary_cfg))
                primary_seed = int(inputs.get("seed", primary_seed))
                m = inputs.get("model")
                p = inputs.get("positive")
                n = inputs.get("negative")
                if isinstance(m, list):
                    primary_model_ref = list(m)
                if isinstance(p, list):
                    primary_positive_ref = list(p)
                if isinstance(n, list):
                    primary_negative_ref = list(n)
            break
    if primary_ksampler_id is None:
        raise ValueError(
            "HD variant requires a KSampler node in the base workflow"
        )
    if primary_model_ref is None or primary_positive_ref is None or primary_negative_ref is None:
        raise ValueError(
            "HD variant requires the primary KSampler to have model/positive/negative wires"
        )

    # LatentUpscaleBy (id "10")
    out["10"] = {
        "class_type": "LatentUpscaleBy",
        "inputs": {
            "samples": [primary_ksampler_id, 0],
            "upscale_method": variant.upscale_method,
            "scale_by": variant.scale_by,
        },
    }
    # KSampler refiner (id "11") — re-uses primary's model/positive/negative wires.
    out["11"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": primary_seed,
            "steps": variant.refiner_steps,
            "cfg": primary_cfg,
            "sampler_name": primary_sampler,
            "scheduler": primary_scheduler,
            "denoise": variant.refiner_denoise,
            "model": primary_model_ref,
            "positive": primary_positive_ref,
            "negative": primary_negative_ref,
            "latent_image": ["10", 0],
        },
    }
    # Rewire VAEDecode (id "8") to refiner output.
    vae_decode = out.get("8")
    if isinstance(vae_decode, dict):
        vae_inputs = vae_decode.get("inputs")
        if isinstance(vae_inputs, dict):
            vae_inputs["samples"] = ["11", 0]
    return out


def _compose_variant_note_markdown(
    workflow_name: str,
    workflow_a_variant: dict[str, Any],
    variant: VariantSpec,
    siblings: tuple[VariantSpec, ...],
    install_urls: dict[str, str],
    pillar_urls: dict[int, str],
) -> str:
    """Compose the Note markdown for one variant, including sibling cross-refs.

    The Note body is structurally consistent with the .md sidecar
    ``_build_readme_text`` output (so audience-facing content stays
    aligned with the GitHub online sidecar), but adds:

    - aspect label in the H1 title
    - HD refiner step + denoise in the params section
    - a "Outras aspect ratios neste install" list pointing at the
      sibling filenames (current variant excluded)
    """
    install_entry = INSTALL_VIDEO_MAPPING[workflow_name]
    pillar_entry = PILLAR_MAPPING[workflow_name]
    hardware = HARDWARE_TIERS[workflow_name]
    base_display = install_entry["display_name"]
    install_slug: str = install_entry["install_slug"]
    install_video_num = INSTALL_VIDEO_NUMBER[install_slug]
    install_video_label = INSTALL_VIDEO_LABEL[install_slug]
    primary_pillar: int = pillar_entry["primary"]
    secondary_pillars: list[int] = pillar_entry["secondary"]

    install_url = install_urls.get(
        install_slug, _placeholder_install_url(install_slug, DEFAULT_GITHUB_BASE_URL),
    )
    primary_pillar_url = pillar_urls.get(
        primary_pillar, _placeholder_pillar_url(primary_pillar, DEFAULT_GITHUB_BASE_URL),
    )
    setup_base = f"{DEFAULT_GITHUB_BASE_URL}/tree/main/setup-windows"

    # Params: extract from the VARIANT workflow (HD-modified or native).
    params = _extract_params(workflow_a_variant)

    base_stem = workflow_name.removesuffix(".json")

    lines: list[str] = []
    lines.append(f"# {base_display} — {variant.aspect_label}")
    lines.append("")
    lines.append(
        f"📺 Install video #{install_video_num}: "
        f"[{install_video_label}]({install_url})"
    )
    lines.append(
        f"🔬 Benchmark Pillar #{primary_pillar}: "
        f"[Pillar #{primary_pillar}]({primary_pillar_url})"
    )
    for sp in secondary_pillars:
        sp_url = pillar_urls.get(
            sp, _placeholder_pillar_url(sp, DEFAULT_GITHUB_BASE_URL),
        )
        lines.append(f"🔬 Also in Pillar #{sp}: [Pillar #{sp}]({sp_url})")
    lines.append("")

    lines.append(f"📥 Install: [{setup_base}]({setup_base})")
    for script in install_entry["scripts"]:
        lines.append(f"- `{script}`")
    lines.append("")

    # Config block — variant-specific.
    lines.append("⚙️ Config:")
    if variant.hd:
        target_w = int(round(variant.width * variant.scale_by))
        target_h = int(round(variant.height * variant.scale_by))
        lines.append(
            f"- Resolution: {variant.width}×{variant.height} native "
            f"→ ~{target_w}×{target_h} after latent upscale"
        )
    else:
        lines.append(f"- Resolution: {variant.width}×{variant.height}")
    sampler = params.get("sampler")
    scheduler = params.get("scheduler")
    if sampler and scheduler:
        lines.append(f"- Sampler: `{sampler}` / `{scheduler}`")
    elif sampler:
        lines.append(f"- Sampler: `{sampler}`")
    bits: list[str] = []
    if params.get("steps") is not None:
        bits.append(f"{params['steps']} steps")
    if params.get("cfg") is not None:
        bits.append(f"CFG {params['cfg']}")
    elif params.get("guidance") is not None:
        bits.append(f"guidance {params['guidance']}")
    if bits:
        lines.append(f"- {' · '.join(bits)}")
    if variant.hd:
        lines.append(
            f"- Refiner: {variant.refiner_steps} steps, denoise "
            f"{variant.refiner_denoise}, `{variant.upscale_method}` upscale"
        )
    lines.append("")

    # Cross-reference siblings (excluding self).
    other_siblings = [v for v in siblings if v.slug != variant.slug]
    if other_siblings:
        lines.append("📐 Other aspect ratios in this install:")
        for s in other_siblings:
            sibling_filename = f"{base_stem}{s.slug}.json"
            lines.append(f"- `{sibling_filename}` — {s.aspect_label}")
        lines.append("")

    lines.append(
        f"💾 Minimum hardware: {hardware['ram']} RAM · {hardware['vram']} VRAM"
    )

    # Optional: per-workflow swap guide (e.g. FLUX 5-variant pack).
    extra = _extra_workflow_notes(workflow_name)
    if extra:
        lines.append(extra)

    return "\n".join(lines)


def _compose_note_markdown(
    workflow_name: str,
    workflow_a: dict[str, Any],
    install_urls: dict[str, str],
    pillar_urls: dict[int, str],
) -> str:
    """Build the Note's markdown body using the same composer as the .md sidecar.

    Appends the optional per-workflow ``EXTRA_NOTE_TEXT`` block (e.g. the
    FLUX variant positioning + Phase 1.5 timing) so single-output
    workflows get the same audience-facing context that variant-aware
    bases get via :func:`_compose_variant_note_markdown`.
    """
    install_entry = INSTALL_VIDEO_MAPPING[workflow_name]
    pillar_entry = PILLAR_MAPPING[workflow_name]
    hardware = HARDWARE_TIERS[workflow_name]
    params = _extract_params(workflow_a)
    setup_base = f"{DEFAULT_GITHUB_BASE_URL}/tree/main/setup-windows"
    body = _build_readme_text(
        workflow_name=workflow_name,
        install_entry=install_entry,
        pillar_entry=pillar_entry,
        hardware=hardware,
        params=params,
        install_urls=install_urls,
        pillar_urls=pillar_urls,
        github_base=DEFAULT_GITHUB_BASE_URL,
        setup_base=setup_base,
    )
    extra = _extra_workflow_notes(workflow_name)
    if extra:
        body = body + extra
    return body


def build_one(
    workflow_name: str,
    input_dir: Path,
    output_dir: Path,
    schema: dict[str, dict[str, Any]],
    install_urls: dict[str, str],
    pillar_urls: dict[int, str],
    dry_run: bool,
) -> tuple[bool, Path, dict[str, Any]]:
    """Convert + inject one workflow. Returns (written, output_path, format_b)."""
    in_path = input_dir / workflow_name
    workflow_a = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(workflow_a, dict):
        raise ValueError(f"{in_path}: top-level is not a JSON object")

    note_md = _compose_note_markdown(
        workflow_name, workflow_a, install_urls, pillar_urls,
    )
    workflow_b = convert_a_to_b(workflow_a, schema)
    workflow_b = inject_note(workflow_b, note_md)

    out_path = _resolve_output_path(output_dir, workflow_name)
    new_text = json.dumps(workflow_b, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        existing = out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
        return (existing != new_text), out_path, workflow_b
    changed = _atomic_write(out_path, new_text)
    return changed, out_path, workflow_b


def build_one_variant(
    workflow_name: str,
    variant: VariantSpec,
    siblings: tuple[VariantSpec, ...],
    input_dir: Path,
    output_dir: Path,
    schema: dict[str, dict[str, Any]],
    install_urls: dict[str, str],
    pillar_urls: dict[int, str],
    dry_run: bool,
) -> tuple[bool, Path, dict[str, Any]]:
    """Build one aspect/HD variant of a workflow base.

    Output filename: ``<stem><variant.slug>.json`` under
    ``output_dir``. Returns ``(changed, output_path, format_b)``.

    Pipeline: load Format A base → apply variant
    (:func:`_apply_variant_to_workflow_a`) → compose variant Note
    markdown with sibling cross-refs → convert variant Format A to
    Format B → inject Note → atomic write.
    """
    in_path = input_dir / workflow_name
    base_a = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(base_a, dict):
        raise ValueError(f"{in_path}: top-level is not a JSON object")

    variant_a = _apply_variant_to_workflow_a(base_a, variant)
    note_md = _compose_variant_note_markdown(
        workflow_name, variant_a, variant, siblings,
        install_urls, pillar_urls,
    )
    workflow_b = convert_a_to_b(variant_a, schema)
    workflow_b = inject_note(workflow_b, note_md)

    stem = workflow_name.removesuffix(".json")
    variant_filename = f"{stem}{variant.slug}.json"
    out_path = _resolve_output_path(output_dir, variant_filename)
    new_text = json.dumps(workflow_b, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        existing = (
            out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
        )
        return (existing != new_text), out_path, workflow_b
    changed = _atomic_write(out_path, new_text)
    return changed, out_path, workflow_b


# ============================================================================
# CLI
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Format A workflows in installer/benchmark/workflows/ "
            "into Format B with embedded Note bula at "
            "installer/benchmark/workflows_distribute/. Idempotent: re-run "
            "with same inputs = zero on-disk diff."
        ),
    )
    parser.add_argument(
        "--workflow",
        help=(
            "Workflow name (with or without .json extension), e.g. "
            "'sdxl_base' or 'sdxl_base.json'. Use --all to process every "
            "workflow in INSTALL_VIDEO_MAPPING."
        ),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process every workflow registered in INSTALL_VIDEO_MAPPING.",
    )
    parser.add_argument(
        "--input-dir", default=str(DEFAULT_INPUT_DIR),
        help=f"Format A workflows source dir (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Format B distribute output dir (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--schema", default=str(DEFAULT_SCHEMA_PATH),
        help=(
            f"Cached /object_info JSON file (default: {DEFAULT_SCHEMA_PATH}). "
            "Refresh manually via: curl -sS http://<host>:8188/object_info > "
            f"{DEFAULT_SCHEMA_PATH}"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing files.",
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST_PATH),
        help=(
            "Path to models_manifest.yaml — read for the videos: section "
            f"(default: {DEFAULT_MANIFEST_PATH}). URLs are no longer CLI "
            "flags on this tool; the single source of truth is the YAML, "
            "updated via update_workflow_links.py --install-<slug>-url "
            "and --pillar-N-url flags."
        ),
    )
    return parser


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = _build_argparser().parse_args()

    if not args.all and not args.workflow:
        raise SystemExit("must specify --workflow <name> or --all")

    schema_path = Path(args.schema)
    if not schema_path.is_file():
        raise SystemExit(
            f"schema cache not found: {schema_path}. "
            "Run: curl -sS http://cg-5090:8188/object_info > "
            f"{schema_path}"
        )
    schema = _load_schema(schema_path)
    logger.info("loaded schema %s (%d node types)", schema_path, len(schema))

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    install_urls, pillar_urls = load_video_urls_from_manifest(manifest_path)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = list(INSTALL_VIDEO_MAPPING.keys())
    else:
        wf_name = args.workflow
        if not wf_name.endswith(".json"):
            wf_name = wf_name + ".json"
        if wf_name not in INSTALL_VIDEO_MAPPING:
            raise SystemExit(
                f"unknown workflow: {wf_name}. Known: "
                f"{sorted(INSTALL_VIDEO_MAPPING.keys())}"
            )
        targets = [wf_name]

    n_changed = 0
    n_idempotent = 0
    for wf_name in targets:
        in_path = input_dir / wf_name
        if not in_path.is_file():
            logger.warning("workflow source missing, skipping: %s", in_path)
            continue

        stem = Path(wf_name).stem
        # Multi-variant base: build each aspect/HD variant separately.
        if stem in ASPECT_VARIANTS:
            variants = ASPECT_VARIANTS[stem]
            for variant in variants:
                changed, out_path, workflow_b = build_one_variant(
                    workflow_name=wf_name,
                    variant=variant,
                    siblings=variants,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    schema=schema,
                    install_urls=install_urls,
                    pillar_urls=pillar_urls,
                    dry_run=args.dry_run,
                )
                n_nodes = len(workflow_b["nodes"])
                n_links = len(workflow_b["links"])
                note_present = any(
                    n["type"] == "Note" for n in workflow_b["nodes"]
                )
                prefix = "[DRY-RUN] " if args.dry_run else ""
                if changed:
                    n_changed += 1
                    logger.info(
                        "%sWROTE %s (variant=%s nodes=%d links=%d note=%s)",
                        prefix, out_path, variant.aspect_id,
                        n_nodes, n_links, note_present,
                    )
                else:
                    n_idempotent += 1
                    logger.info("[IDEMPOTENT] %s", out_path)
            continue

        # Single-output legacy path (unchanged behavior for non-variant
        # workflows: flux / qwen / hunyuan / wan).
        changed, out_path, workflow_b = build_one(
            workflow_name=wf_name,
            input_dir=input_dir,
            output_dir=output_dir,
            schema=schema,
            install_urls=install_urls,
            pillar_urls=pillar_urls,
            dry_run=args.dry_run,
        )
        n_nodes = len(workflow_b["nodes"])
        n_links = len(workflow_b["links"])
        note_present = any(n["type"] == "Note" for n in workflow_b["nodes"])
        prefix = "[DRY-RUN] " if args.dry_run else ""
        if changed:
            n_changed += 1
            logger.info(
                "%sWROTE %s (nodes=%d, links=%d, note=%s)",
                prefix, out_path, n_nodes, n_links, note_present,
            )
        else:
            n_idempotent += 1
            logger.info("[IDEMPOTENT] %s", out_path)

    logger.info(
        "=== %s === changed=%d, idempotent=%d, total=%d",
        "DRY-RUN SUMMARY" if args.dry_run else "DONE",
        n_changed, n_idempotent, n_changed + n_idempotent,
    )


if __name__ == "__main__":
    main()
