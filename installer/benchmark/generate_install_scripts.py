"""Generate per-Pillar / per-model installer ``.bat`` scripts from manifest.

Reads :data:`installer/benchmark/models_manifest.yaml` plus the
:data:`SCRIPT_DEFS` mapping below and writes 7 ``install-*.bat`` scripts
plus one ``setup-sdxl.bat`` master orchestrator into ``setup-windows/``.

Each ``install-X.bat``:

- Verifies the ComfyUI base install at ``C:\\ComfyUI_windows_portable\\``
  before doing anything (friendly error if missing).
- Downloads model files via ``curl`` with byte-exact size verification.
  Idempotent: re-runs skip files already present with matching size.
- Clones any required custom nodes (idempotent: skips if folder exists).
- Ships the workflow JSON + sidecar ``.md`` from
  ``raw.githubusercontent.com``.

``setup-sdxl.bat`` is a thin orchestrator that runs
``01-install-base.bat`` then ``install-sdxl.bat`` — i.e. the fresh
end-to-end SDXL setup wired up for the "Install SDXL" install-mini-
series video. Supports ``--unattended`` flag (or
``COMFY_NONINTERACTIVE`` env var) for SSH dispatch (DA-013 compatible —
never launches ComfyUI interactively).

Note: "Pillar" in this project refers to the **Benchmark Pillar** mini-
series (cross-model editorial videos, mapped per workflow in
``update_workflow_links.py::PILLAR_MAPPING``); it is **not** the same
thing as the install-mini-series videos. The orchestrator name is keyed
to the workflow / model installed, not to a Pillar number.

Atomic write semantics: each ``.bat`` is written to ``<path>.tmp`` then
renamed via :meth:`Path.replace`, so a crash mid-write leaves the prior
version intact. Idempotent: re-running with the same manifest produces
zero diff if the on-disk script already matches the proposed content.

DO NOT EDIT GENERATED .bat FILES BY HAND. Re-generate via::

    python -m installer.benchmark.generate_install_scripts
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from installer.benchmark.inject_markdown import variant_filenames_for

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MANIFEST: Path = Path("installer/benchmark/models_manifest.yaml")
DEFAULT_SETUP_DIR: Path = Path("setup-windows")
DEFAULT_TEMPLATE_DIR: Path = Path("installer/benchmark/templates")
DEFAULT_RAW_BASE: str = (
    "https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main"
)

# Rendered when a videos: slug is empty or missing. Audience-facing text
# (every install bat is in English per the channel principle).
_COMING_SOON: str = "(Coming soon - watch the repo for updates)"

# Matches {{<KEY>_URL}} and {{<KEY>_URL_OR_COMING_SOON}} placeholders.
# The slug body (lowercased) must match a key in the manifest videos:
# section — e.g. INSTALL_BASE → "install_base", PILLAR_1 → "pillar_1".
# Both forms resolve identically: URL when set, _COMING_SOON when empty;
# the suffix is purely a stylistic hint for the human editor.
_VIDEO_PLACEHOLDER_RE: re.Pattern[str] = re.compile(
    r"\{\{([A-Z][A-Z0-9_]*)_URL(?:_OR_COMING_SOON)?\}\}",
)


@dataclass(frozen=True)
class SuccessBlockMeta:
    """Metadata for the SUCCESS block emitted at the end of an install .bat.

    Each install script ends with an "X install complete!" block that
    confirms what landed on disk, points the audience at the matching
    tutorial video, and previews the next video in the series — see the
    self-explanatory bats sub-tarefa. URL fields reference videos:
    slugs in the manifest; ``_render_success_block`` emits placeholders
    that get substituted by ``_substitute_video_placeholders`` at the
    end of the .bat composition pipeline.
    """
    title: str                                  # "FLUX install complete!"
    installed_summary: tuple[str, ...]          # "Models installed: ..." lines
    sidebar_summary: tuple[str, ...]            # "Workflows in ComfyUI sidebar:" lines
    current_video_slug: str                     # videos: key — e.g. "install_sdxl"
    current_video_label: str                    # "Video #2 - SDXL install + benchmark"
    next_video_slug: str = ""                   # videos: key for next-in-series, "" => skip
    next_video_label: str = ""                  # human label for next video
    extra_after_installed: tuple[str, ...] = () # e.g. "Default recommendation: ..."


@dataclass(frozen=True)
class ScriptDef:
    """Per-script metadata. ``models`` references manifest entries by name.

    ``category`` is the ComfyUI sidebar subfolder under
    ``Comfy Workflow/`` where the distributed workflow JSON lands —
    e.g. ``"Image\\SDXL"`` for SDXL t2i, ``"Image\\FLUX"`` for FLUX t2i,
    ``"Video"`` for i2v workflows. Use backslashes — the generated .bat
    converts to forward slashes for the GitHub raw URL inside
    ``ship_workflow``.

    ``cleanup_glob`` is a Windows glob (e.g. ``"sdxl_*.json"``) of
    flat-layout workflow files to delete from the legacy
    ``Comfy Workflow\\Image\\`` folder before shipping the new
    subfolder layout. Empty string skips the cleanup step (legacy
    flat scripts that never moved into a subfolder).

    ``success_meta`` drives the audience-facing SUCCESS block emitted
    at the end of the .bat — confirms the install, points at the
    tutorial video, previews the next video in the series.
    """
    display_name: str
    pillars: tuple[int, ...]
    models: tuple[str, ...]
    custom_nodes: tuple[tuple[str, str], ...]
    workflows: tuple[str, ...]
    ram_min: str
    vram_min: str
    category: str
    success_meta: SuccessBlockMeta
    cleanup_glob: str = ""


# Mirrors PILLAR_MAPPING / HARDWARE_TIERS in update_workflow_links.py —
# kept in sync manually; the two surfaces will consolidate in a follow-up
# (débito V2 #24 candidate: shared metadata source of truth).
SCRIPT_DEFS: dict[str, ScriptDef] = {
    "install-sdxl.bat": ScriptDef(
        display_name="SDXL Base 1.0",
        pillars=(1,),
        models=("sdxl_base_1.0",),
        custom_nodes=(),
        workflows=("sdxl_base.json",),
        ram_min="16 GB",
        vram_min="8 GB",
        category="Image\\SDXL",
        cleanup_glob="sdxl_*.json",
        success_meta=SuccessBlockMeta(
            title="SDXL install complete!",
            installed_summary=(
                "Model installed: SDXL Base 1.0",
            ),
            sidebar_summary=(
                "Workflows in ComfyUI sidebar:",
                "  Comfy Workflow > Image > SDXL > (5 aspect variants)",
            ),
            current_video_slug="install_sdxl",
            current_video_label="Video #2 - SDXL install + benchmark",
            next_video_slug="install_flux1",
            next_video_label="Video #3 - FLUX",
        ),
    ),
    "install-flux1.bat": ScriptDef(
        # Phase 1.5 audit (e70572c): ship 5 separate variant workflows
        # (one per loader/dtype combo) instead of an aggregate. Each
        # variant opens-and-runs with the correct loader pre-wired —
        # no node swaps for the audience. Q8/Q4 use UnetLoaderGGUF
        # (requires the city96 ComfyUI-GGUF custom node).
        display_name="FLUX.1 (5 variants pre-wired)",
        pillars=(2, 1),
        models=(
            "flux_dev_fp8",
            "flux_dev_fp16",
            "flux_schnell_fp8",
            "flux_dev_Q8_gguf",
            "flux_dev_Q4_gguf",
            "flux_shared_encoders",
            "flux_shared_vae",
        ),
        custom_nodes=(
            ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
        ),
        workflows=(
            "flux_dev_fp16.json",
            "flux_dev_fp8.json",
            "flux_schnell_fp8.json",
            "flux_dev_Q8.json",
            "flux_dev_Q4.json",
        ),
        ram_min="32 GB",
        vram_min="10 GB",
        category="Image\\FLUX",
        cleanup_glob="flux_*.json",
        success_meta=SuccessBlockMeta(
            title="FLUX install complete!",
            installed_summary=(
                "Models installed: fp16, fp8, schnell, Q8, Q4 (5 variants)",
            ),
            sidebar_summary=(
                "Workflows in ComfyUI sidebar:",
                "  Comfy Workflow > Image > FLUX > (5 variants)",
            ),
            current_video_slug="install_flux1",
            current_video_label="Video #3 - FLUX install + benchmark",
            next_video_slug="install_qwen_image",
            next_video_label="Video #4 - Qwen-Image",
            extra_after_installed=(
                "Default recommendation: flux_dev_fp8 (best balance for most GPUs)",
            ),
        ),
    ),
    "install-qwen-image.bat": ScriptDef(
        # Phase 1.5 audit (b313b96) ship list collapsed to 2 variants:
        # V2 fp8 default (the only fp8 dtype that produces clean output —
        # NOT fp8_e4m3fn_fast) + V3 Lightning 4-step (7.9× speedup,
        # equalizes 3060→5090 spread). V1 bf16 / V4 Q8 GGUF / V5 Q4 GGUF
        # / V6 base fp8 all dropped under the "real gain or mandatory
        # fallback" rule — see internal_docs/quality_audit/20260519T020647Z/
        # qwen_image_phase1_5/REPORT_phase1.5.md.
        display_name="Qwen-Image 2512 (V2 fp8 + V3 Lightning)",
        pillars=(3,),
        models=(
            "qwen_image_2512_fp8",
            "qwen_lightning_lora_2512_4step",
            "qwen_shared_encoders",
            "qwen_shared_vae",
        ),
        custom_nodes=(),
        workflows=(
            "qwen_image_2512.json",
            "qwen_image_2512_lightning4.json",
        ),
        ram_min="32 GB",
        vram_min="12 GB",
        category="Image\\Qwen",
        cleanup_glob="qwen_*.json",
        success_meta=SuccessBlockMeta(
            title="Qwen-Image install complete!",
            installed_summary=(
                "Models installed: 2512 fp8 + Lightning 4-step LoRA",
            ),
            sidebar_summary=(
                "Workflows in ComfyUI sidebar:",
                "  Comfy Workflow > Image > Qwen > (2 variants)",
            ),
            current_video_slug="install_qwen_image",
            current_video_label="Video #4 - Qwen-Image install + benchmark",
            next_video_slug="install_hunyuan_21",
            next_video_label="Video #5 - Hunyuan-Image 2.1",
            extra_after_installed=(
                "Default recommendation: qwen_image_2512.json (V2 fp8)",
                "For fast drafts: qwen_image_2512_lightning4.json (4 steps, 7.9x faster)",
            ),
        ),
    ),
    # Other install scripts (flux2 / hunyuan-21 / wan22) are intentionally
    # absent from this map — they are withheld from the public
    # setup-windows/ tree until each model family's release video drops.
    # Definitions stay in the internal working copy and rotate back in
    # here on each video launch.
}


# ============================================================================
# Manifest loading
# ============================================================================

def _load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Read ``models_manifest.yaml`` and index entries by ``name``."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(
            f"manifest {manifest_path} missing top-level 'models' list"
        )
    models = data["models"]
    if not isinstance(models, list):
        raise ValueError(
            f"manifest 'models' must be a list, got {type(models).__name__}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(
                f"manifest model entry missing 'name': {entry!r}"
            )
        indexed[entry["name"]] = entry
    return indexed


def _load_videos(manifest_path: Path) -> dict[str, str]:
    """Read the manifest ``videos:`` section as a slug→URL dict.

    Returns an empty dict if the section is missing (legacy manifest).
    Non-string values are coerced to empty strings — empty strings
    propagate through ``_substitute_video_placeholders`` as
    "Coming soon - watch the repo for updates".
    """
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    raw = data.get("videos", {}) or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for slug, url in raw.items():
        if not isinstance(slug, str):
            continue
        out[slug] = url if isinstance(url, str) else ""
    return out


def _gib(total_bytes: int) -> str:
    """Format bytes as ``X.XX`` GiB (two decimals)."""
    return f"{total_bytes / (1024 ** 3):.2f}"


# ============================================================================
# .bat composition
# ============================================================================

# Header + subroutines + main prologue. Format placeholders are
# ``{display_name}``, ``{pillar_summary}``, ``{model_names}``,
# ``{ram_min}``, ``{vram_min}``, ``{total_gib}``, ``{repo_raw}``.
_BAT_HEADER_FMT = """\
@echo off
REM ============================================================================
REM Comfy Workflow - {display_name}
REM ============================================================================
REM Generated by installer/benchmark/generate_install_scripts.py
REM DO NOT EDIT BY HAND. Re-generate via:
REM   python -m installer.benchmark.generate_install_scripts
REM ============================================================================
REM Pillars: {pillar_summary}
REM Models (from models_manifest.yaml): {model_names}
REM Hardware minimo: {ram_min} RAM, {vram_min} VRAM
REM Total download (this script): ~{total_gib} GiB
REM ============================================================================
setlocal EnableDelayedExpansion

set "REPO_RAW={repo_raw}"
set "CATEGORY={category}"

REM ----- Pre-req: ComfyUI base install must exist -----
REM The detailed ERROR block below is for users who downloaded this
REM .bat without running 01-install-base.bat first. YouTube channel
REM link restrictions (DA: external-channel links banned) mean the
REM audience cannot navigate via playlists — so this block embeds the
REM repo URL, the file name to fetch, and the Video #1 tutorial URL
REM directly. {{INSTALL_BASE_URL}} is substituted at generation time
REM from the manifest videos: section (empty => "Coming soon").
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\main.py" (
    echo.
    echo =========================================================================
    echo   ERROR: ComfyUI base install not found
    echo =========================================================================
    echo.
    echo   This install needs ComfyUI base installed FIRST.
    echo   Step 1 of the series - you skipped it.
    echo.
    echo   How to fix:
    echo.
    echo    1. Open:           https://github.com/comfyworkflow/comfy-workflow
    echo    2. Click folder:   setup-windows
    echo    3. Download:       01-install-base.bat
    echo    4. Run it          ^(installs ComfyUI portable + essential nodes^)
    echo    5. Come back and run this script again
    echo.
    echo   Tutorial video ^(Video #1 - ComfyUI base^):
    echo   {{{{INSTALL_BASE_URL}}}}
    echo.
    echo =========================================================================
    if not defined COMFY_NONINTERACTIVE pause
    exit /b 1
)

goto :main

REM ============================================================================
REM Subroutine: download_or_skip
REM   %1 = url
REM   %2 = absolute target path
REM   %3 = expected size in bytes (string compare; no int32 limit)
REM ============================================================================
:download_or_skip
set "URL=%~1"
set "TARGET=%~2"
set "EXPECTED=%~3"
if exist "!TARGET!" (
    for %%A in ("!TARGET!") do set "ACTUAL=%%~zA"
    if "!ACTUAL!"=="!EXPECTED!" (
        echo   Skip ^(size match^): !TARGET!
        goto :eof
    )
    echo   Re-downloading ^(size mismatch !ACTUAL! != !EXPECTED!^): !TARGET!
    del "!TARGET!" >nul 2>&1
)
echo   Downloading: !URL!
curl.exe -L --fail --progress-bar --retry 3 --retry-delay 5 -o "!TARGET!" "!URL!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: download failed for !URL!
    exit /b 1
)
for %%A in ("!TARGET!") do set "ACTUAL=%%~zA"
if not "!ACTUAL!"=="!EXPECTED!" (
    echo   ERROR: post-download size mismatch ^(!ACTUAL! != !EXPECTED!^): !TARGET!
    exit /b 1
)
echo   OK ^(verified !EXPECTED! bytes^): !TARGET!
goto :eof

REM ============================================================================
REM Subroutine: ensure_custom_node
REM   %1 = folder name (e.g. ComfyUI-GGUF)
REM   %2 = git repo URL
REM ============================================================================
:ensure_custom_node
set "NODE_NAME=%~1"
set "NODE_DIR=C:\\ComfyUI_windows_portable\\ComfyUI\\custom_nodes\\!NODE_NAME!"
set "REPO=%~2"
if exist "!NODE_DIR!" (
    echo   Custom node already present: !NODE_NAME!
    goto :eof
)
echo   Cloning custom node: !NODE_NAME!
git clone "!REPO!" "!NODE_DIR!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: git clone failed for !NODE_NAME!
    exit /b 1
)
goto :eof

REM ============================================================================
REM Subroutine: ship_workflow
REM   %1 = workflow filename (e.g. sdxl_base.json)
REM Downloads the Format B distribute version from
REM "workflows_distribute/!CATEGORY_URL!/!WF_JSON!" (URL form: forward
REM slashes) into the matching local sidebar subfolder
REM "Comfy Workflow\\!CATEGORY!\\" (Windows form: backslashes). The
REM repo source path and the user-visible category share the same
REM hierarchy, so audience and repo browsers see the same structure.
REM The .md sidecar is NOT shipped — its content is embedded as a Note
REM node inside the Format B workflow, so the audience sees the bula
REM directly on the canvas (sidebar Workflows -> Comfy Workflow ->
REM !CATEGORY! -> click).
REM ============================================================================
:ship_workflow
set "WF_JSON=%~1"
set "DEST_BASE=C:\\ComfyUI_windows_portable\\ComfyUI\\user\\default\\workflows\\Comfy Workflow"
set "DEST_DIR=!DEST_BASE!\\!CATEGORY!"
set "CATEGORY_URL=!CATEGORY:\\=/!"
if not exist "!DEST_DIR!" mkdir "!DEST_DIR!"
echo   Shipping workflow: !WF_JSON! ^(category: !CATEGORY!^)
curl.exe -L --fail --silent --retry 3 --retry-delay 5 -o "!DEST_DIR!\\!WF_JSON!" "!REPO_RAW!/installer/benchmark/workflows_distribute/!CATEGORY_URL!/!WF_JSON!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: download failed for !WF_JSON!
    exit /b 1
)
goto :eof

REM ============================================================================
REM Subroutine: cleanup_flat_workflows
REM   %1 = glob pattern (e.g. sdxl_*.json)
REM Removes flat-layout workflow files from the legacy
REM "Comfy Workflow\\Image\\" folder. The new layout lives under
REM "Comfy Workflow\\Image\\<FAMILY>\\" so installs from this script
REM will populate the family subfolder; this subroutine prunes
REM stale flat siblings left over from prior installs to avoid
REM duplicate workflows showing up in the audience's sidebar.
REM ============================================================================
:cleanup_flat_workflows
set "PATTERN=%~1"
set "FLAT_DIR=C:\\ComfyUI_windows_portable\\ComfyUI\\user\\default\\workflows\\Comfy Workflow\\Image"
if not exist "!FLAT_DIR!" goto :eof
for %%F in ("!FLAT_DIR!\\!PATTERN!") do (
    if exist "%%F" (
        echo   Removing legacy flat workflow: %%~nxF
        del "%%F" >nul 2>&1
    )
)
goto :eof

:main

echo.
echo ========================================================
echo  Comfy Workflow - {display_name}
echo ========================================================
echo  Pillars: {pillar_summary}
echo  Hardware minimo: {ram_min} RAM, {vram_min} VRAM
echo  Total download: ~{total_gib} GiB
echo ========================================================
echo.

REM ----- Pre-create model folders (idempotent) -----
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\checkpoints"      mkdir "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\checkpoints"
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusion_models" mkdir "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\diffusion_models"
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\text_encoders"    mkdir "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\text_encoders"
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\vae"              mkdir "C:\\ComfyUI_windows_portable\\ComfyUI\\models\\vae"
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\user\\default\\workflows\\Comfy Workflow" mkdir "C:\\ComfyUI_windows_portable\\ComfyUI\\user\\default\\workflows\\Comfy Workflow"

"""

# Per-script SUCCESS block at the .bat tail. Composed from the
# ScriptDef.success_meta fields by :func:`_render_success_block` — the
# audience-facing "X install complete!" panel that confirms what
# landed on disk, points at the tutorial video, and previews the next
# video in the series. {{<SLUG>_URL}} placeholders are substituted at
# generation time from the manifest videos: section (empty =>
# "Coming soon - watch the repo for updates").
def _cmd_escape(text: str) -> str:
    """Escape characters that cmd.exe ``echo`` would consume.

    - ``>`` and ``<`` are redirection operators — escape as ``^>`` / ``^<``.
    - ``!`` is delayed-expansion when ``setlocal EnableDelayedExpansion``
      is active (every bat we generate enables it) — escape as ``^^!``.
    - ``(`` / ``)`` inside an echo body need not be escaped since echo
      isn't a parser-sensitive command for parens; but author-supplied
      strings already use ``^(`` / ``^)`` where they want literals.
    """
    return text.replace(">", "^>").replace("<", "^<").replace("!", "^^!")


def _render_success_block(meta: SuccessBlockMeta) -> str:
    """Compose the SUCCESS block at the .bat tail from ScriptDef metadata."""
    lines: list[str] = [
        "",
        "echo.",
        "echo =========================================================================",
        f"echo   {_cmd_escape(meta.title)}",
        "echo =========================================================================",
        "echo.",
    ]
    for line in meta.installed_summary:
        lines.append(f"echo   {_cmd_escape(line)}")
    if meta.sidebar_summary:
        if meta.installed_summary:
            lines.append("echo.")
        for line in meta.sidebar_summary:
            lines.append(f"echo   {_cmd_escape(line)}")
    if meta.extra_after_installed:
        lines.append("echo.")
        for line in meta.extra_after_installed:
            lines.append(f"echo   {_cmd_escape(line)}")
    lines.append("echo.")
    lines.append(
        f"echo   Tutorial video ^({_cmd_escape(meta.current_video_label)}^):"
    )
    lines.append(
        f"echo   {{{{{meta.current_video_slug.upper()}_URL}}}}"
    )
    if meta.next_video_slug:
        lines.append("echo.")
        lines.append(
            f"echo   Next in series ^({_cmd_escape(meta.next_video_label)}^):"
        )
        lines.append(
            f"echo   {{{{{meta.next_video_slug.upper()}_URL_OR_COMING_SOON}}}}"
        )
    lines.append("echo.")
    lines.append("echo   Repo ^(all installs^):")
    lines.append("echo   https://github.com/comfyworkflow/comfy-workflow")
    lines.append("echo.")
    lines.append(
        "echo ========================================================================="
    )
    lines.append("echo.")
    lines.append("if not defined COMFY_NONINTERACTIVE pause")
    lines.append("endlocal")
    lines.append("exit /b 0")
    return "\n".join(lines) + "\n"


def _substitute_video_placeholders(
    text: str, videos: dict[str, str],
) -> str:
    """Replace ``{{<SLUG>_URL}}`` / ``{{<SLUG>_URL_OR_COMING_SOON}}`` in ``text``.

    Looks up the lowercased slug in the manifest ``videos:`` section.
    Empty / missing values render as
    ``(Coming soon - watch the repo for updates)`` so the .bat never
    emits a blank URL line.
    """
    def repl(match: re.Match[str]) -> str:
        slug = match.group(1).lower()
        url = videos.get(slug, "")
        if isinstance(url, str):
            url = url.strip()
        else:
            url = ""
        return url if url else _COMING_SOON
    return _VIDEO_PLACEHOLDER_RE.sub(repl, text)


def _build_bat(
    script_name: str,
    script_def: ScriptDef,
    manifest: dict[str, dict[str, Any]],
    repo_raw: str,
) -> str:
    """Compose the full ``.bat`` content for one ``install-X.bat`` script."""
    total_bytes = 0
    file_lines: list[str] = []
    for model_name in script_def.models:
        if model_name not in manifest:
            raise ValueError(
                f"{script_name}: model '{model_name}' not in manifest"
            )
        entry = manifest[model_name]
        for f in entry.get("files", []):
            size = int(f["size_bytes"])
            total_bytes += size
            win_path = f["path"].replace("/", "\\")
            file_lines.append(
                f'call :download_or_skip '
                f'"{f["url"]}" '
                f'"C:\\ComfyUI_windows_portable\\ComfyUI\\models\\{win_path}" '
                f'{size}'
            )

    total_steps = (
        1
        + (1 if script_def.custom_nodes else 0)
        + (1 if script_def.cleanup_glob else 0)
        + 1
    )
    pillar_summary = ", ".join(f"#{p}" for p in script_def.pillars)
    model_names = ", ".join(script_def.models)
    total_gib = _gib(total_bytes)

    parts: list[str] = [
        _BAT_HEADER_FMT.format(
            display_name=script_def.display_name,
            pillar_summary=pillar_summary,
            model_names=model_names,
            ram_min=script_def.ram_min,
            vram_min=script_def.vram_min,
            total_gib=total_gib,
            repo_raw=repo_raw,
            category=script_def.category,
        ),
    ]

    step = 1
    parts.append(
        f"echo [{step}/{total_steps}] Downloading model files "
        f"(total: ~{total_gib} GiB)...\n"
    )
    parts.extend(f"{line}\n" for line in file_lines)

    if script_def.custom_nodes:
        step += 1
        parts.append(
            f"\necho [{step}/{total_steps}] Installing custom nodes...\n"
        )
        for folder, repo in script_def.custom_nodes:
            parts.append(f'call :ensure_custom_node "{folder}" "{repo}"\n')

    if script_def.cleanup_glob:
        step += 1
        parts.append(
            f"\necho [{step}/{total_steps}] Pruning legacy flat-layout "
            f"workflow files ({script_def.cleanup_glob})...\n"
        )
        parts.append(
            f'call :cleanup_flat_workflows "{script_def.cleanup_glob}"\n'
        )

    step += 1
    # Expand each base workflow into per-variant filenames (see
    # inject_markdown.ASPECT_VARIANTS). Non-variant workflows keep their
    # single name; multi-variant bases expand to N files.
    expanded_workflows: list[str] = []
    for wf in script_def.workflows:
        expanded_workflows.extend(variant_filenames_for(wf))
    parts.append(
        f"\necho [{step}/{total_steps}] Shipping {len(expanded_workflows)} "
        f"workflow file(s)...\n"
    )
    for wf in expanded_workflows:
        parts.append(f'call :ship_workflow "{wf}"\n')

    parts.append(_render_success_block(script_def.success_meta))

    return "".join(parts)


# 01-install-base.bat template processing. Source lives under
# installer/benchmark/templates/01-install-base.bat (hand-edited;
# carries the same {{<SLUG>_URL}} placeholders as install-X.bat
# success blocks). The generator copies the template into
# setup-windows/ with placeholders substituted from videos:.
_BASE_TEMPLATE_NAME: str = "01-install-base.bat"


def _build_base_bat(template_dir: Path) -> str:
    """Read the 01-install-base.bat template (placeholders unresolved)."""
    template_path = template_dir / _BASE_TEMPLATE_NAME
    if not template_path.is_file():
        raise SystemExit(
            f"template not found: {template_path}. "
            "Expected the hand-edited 01-install-base.bat source under "
            f"{template_dir}."
        )
    return template_path.read_text(encoding="utf-8")


# ============================================================================
# Atomic write
# ============================================================================

def _atomic_write(target: Path, content: str) -> bool:
    """Write ``content`` to ``target`` atomically. Returns ``True`` on change."""
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    if existing == content:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(target)
    return True


# ============================================================================
# CLI
# ============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-Pillar install .bat scripts from "
            "models_manifest.yaml. Idempotent: re-runs produce zero diff "
            "when manifest + SCRIPT_DEFS are unchanged."
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"path to models manifest (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--setup-dir",
        default=str(DEFAULT_SETUP_DIR),
        help=f"output dir (default: {DEFAULT_SETUP_DIR}).",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help=(
            "hand-edited .bat templates dir "
            f"(default: {DEFAULT_TEMPLATE_DIR}). "
            "Currently holds 01-install-base.bat — generator substitutes "
            "{{<SLUG>_URL}} placeholders from manifest videos: and writes "
            "to setup-dir."
        ),
    )
    parser.add_argument(
        "--repo-raw-base",
        default=DEFAULT_RAW_BASE,
        help=(
            "base URL for workflow JSON / sidecar downloads "
            "(default: raw.githubusercontent.com/comfyworkflow/...)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report proposed changes without writing files.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = _build_argparser().parse_args()

    manifest_path = Path(args.manifest)
    setup_dir = Path(args.setup_dir)
    template_dir = Path(args.template_dir)
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not setup_dir.is_dir():
        raise SystemExit(f"--setup-dir does not exist: {setup_dir}")
    if not template_dir.is_dir():
        raise SystemExit(f"--template-dir does not exist: {template_dir}")

    manifest = _load_manifest(manifest_path)
    videos = _load_videos(manifest_path)
    set_video_slugs = sorted(s for s, u in videos.items() if u)
    logger.info(
        "loaded manifest %s (%d model entries, %d/%d video slugs set: %s)",
        manifest_path, len(manifest),
        len(set_video_slugs), len(videos),
        ", ".join(set_video_slugs) if set_video_slugs else "(none)",
    )

    n_written = 0
    n_idempotent = 0

    def _emit(target: Path, content: str) -> None:
        nonlocal n_written, n_idempotent
        line_count = len(content.splitlines())
        if args.dry_run:
            existing = (
                target.read_text(encoding="utf-8")
                if target.is_file() else ""
            )
            if existing == content:
                logger.info(
                    "[DRY-RUN/IDEMPOTENT] %s (%d lines)",
                    target.name, line_count,
                )
                n_idempotent += 1
            else:
                tag = "UPDATE" if existing else "CREATE"
                logger.info(
                    "[DRY-RUN/%s] %s (%d lines)", tag, target.name, line_count,
                )
                n_written += 1
            return
        if _atomic_write(target, content):
            logger.info("WROTE %s (%d lines)", target, line_count)
            n_written += 1
        else:
            logger.info("[IDEMPOTENT] %s", target)
            n_idempotent += 1

    for script_name, script_def in SCRIPT_DEFS.items():
        content = _build_bat(
            script_name=script_name,
            script_def=script_def,
            manifest=manifest,
            repo_raw=args.repo_raw_base,
        )
        content = _substitute_video_placeholders(content, videos)
        _emit(setup_dir / script_name, content)

    # 01-install-base.bat: read template, substitute placeholders, emit.
    base_content = _build_base_bat(template_dir)
    base_content = _substitute_video_placeholders(base_content, videos)
    _emit(setup_dir / _BASE_TEMPLATE_NAME, base_content)

    summary_tag = "DRY-RUN SUMMARY" if args.dry_run else "DONE"
    logger.info(
        "=== %s === written=%d, idempotent=%d, total=%d",
        summary_tag, n_written, n_idempotent, n_written + n_idempotent,
    )


if __name__ == "__main__":
    main()
