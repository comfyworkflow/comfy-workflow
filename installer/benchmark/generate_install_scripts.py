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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_MANIFEST: Path = Path("installer/benchmark/models_manifest.yaml")
DEFAULT_SETUP_DIR: Path = Path("setup-windows")
DEFAULT_RAW_BASE: str = (
    "https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main"
)


@dataclass(frozen=True)
class ScriptDef:
    """Per-script metadata. ``models`` references manifest entries by name.

    ``category`` is the ComfyUI sidebar subfolder under
    ``Comfy Workflow/`` where the distributed workflow JSON lands —
    e.g. ``"Image"`` for t2i workflows, ``"Video"`` for i2v workflows.
    """
    display_name: str
    pillars: tuple[int, ...]
    models: tuple[str, ...]
    custom_nodes: tuple[tuple[str, str], ...]
    workflows: tuple[str, ...]
    ram_min: str
    vram_min: str
    category: str


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
        category="Image",
    ),
    "install-flux1.bat": ScriptDef(
        display_name="FLUX.1 dev (fp8 + fp16)",
        pillars=(4, 1),
        models=(
            "flux_dev_fp8",
            "flux_dev_fp16",
            "flux_shared_encoders",
            "flux_shared_vae",
        ),
        custom_nodes=(),
        workflows=("flux_dev_fp8.json", "flux_dev_fp16.json"),
        ram_min="64 GB",
        vram_min="24 GB",
        category="Image",
    ),
    "install-flux2.bat": ScriptDef(
        display_name="FLUX.2 dev GGUF (Q4_K_M)",
        pillars=(2, 4),
        models=(
            "flux_2_dev_q4km",
            "flux_2_shared_encoder",
            "flux_2_shared_vae",
        ),
        custom_nodes=(
            ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF.git"),
        ),
        workflows=("flux2_dev_gguf.json",),
        ram_min="48 GB",
        vram_min="12 GB",
        category="Image",
    ),
    "install-qwen-image.bat": ScriptDef(
        display_name="Qwen-Image fp8",
        pillars=(1,),
        models=(
            "qwen_image_fp8",
            "qwen_shared_encoders",
            "qwen_shared_vae",
        ),
        custom_nodes=(),
        workflows=("qwen_image_fp8.json",),
        ram_min="64 GB",
        vram_min="12 GB",
        category="Image",
    ),
    "install-qwen-2512.bat": ScriptDef(
        display_name="Qwen-Image 2512 fp8",
        pillars=(2,),
        models=(
            "qwen_image_2512_fp8",
            "qwen_shared_encoders",
            "qwen_shared_vae",
        ),
        custom_nodes=(),
        workflows=("qwen_image_2512.json",),
        ram_min="64 GB",
        vram_min="12 GB",
        category="Image",
    ),
    "install-hunyuan-21.bat": ScriptDef(
        display_name="Hunyuan-Image 2.1 bf16",
        pillars=(2, 3),
        models=(
            "hunyuan_image_21_bf16",
            "hunyuan_shared_encoders",
            "hunyuan_shared_vae",
        ),
        custom_nodes=(),
        workflows=("hunyuan_image_21.json",),
        ram_min="96 GB",
        vram_min="16 GB",
        category="Image",
    ),
    "install-wan22.bat": ScriptDef(
        display_name="WAN 2.2 i2v fp8 dual-expert",
        pillars=(5,),
        models=(
            "wan22_i2v_14b_fp8",
            "wan_shared_encoder",
            "wan_shared_vae",
        ),
        custom_nodes=(),
        workflows=("wan22_i2v_fp8.json",),
        ram_min="96 GB",
        vram_min="16 GB",
        category="Video",
    ),
}

ORCHESTRATOR_NAME: str = "setup-sdxl.bat"


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
if not exist "C:\\ComfyUI_windows_portable\\ComfyUI\\main.py" (
    echo.
    echo ERROR: ComfyUI base install not found.
    echo.
    echo   Expected at: C:\\ComfyUI_windows_portable\\ComfyUI\\main.py
    echo.
    echo Run 01-install-base.bat first, then re-run this script.
    echo.
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
REM Downloads the Format B distribute version into the
REM "Comfy Workflow\\!CATEGORY!\\" sidebar subfolder. The .md sidecar is
REM NOT shipped — its content is embedded as a Note node inside the
REM Format B workflow, so the audience sees the bula directly on the
REM canvas (sidebar Workflows -> Comfy Workflow -> !CATEGORY! -> click).
REM ============================================================================
:ship_workflow
set "WF_JSON=%~1"
set "DEST_BASE=C:\\ComfyUI_windows_portable\\ComfyUI\\user\\default\\workflows\\Comfy Workflow"
set "DEST_DIR=!DEST_BASE!\\!CATEGORY!"
if not exist "!DEST_DIR!" mkdir "!DEST_DIR!"
echo   Shipping workflow: !WF_JSON! ^(category: !CATEGORY!^)
curl.exe -L --fail --silent --retry 3 --retry-delay 5 -o "!DEST_DIR!\\!WF_JSON!" "!REPO_RAW!/installer/benchmark/workflows_distribute/!WF_JSON!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: download failed for !WF_JSON!
    exit /b 1
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

# Footer: completion banner + next steps. ``{display_name}`` and
# ``{workflow_hints}`` are substituted.
_BAT_FOOTER_FMT = """
echo.
echo ========================================================
echo  DONE - {display_name}
echo ========================================================
echo.
echo  Next steps:
echo    1. Run C:\\ComfyUI_windows_portable\\run_nvidia_gpu.bat
echo    2. Open ComfyUI in browser, load workflow:
{workflow_hints}
echo.
if not defined COMFY_NONINTERACTIVE pause
endlocal
exit /b 0
"""


def _bat_workflow_hint(wf: str) -> str:
    return f"echo        ComfyUI -^> Workflows -^> {wf}"


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

    total_steps = 1 + (1 if script_def.custom_nodes else 0) + 1
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

    step += 1
    parts.append(
        f"\necho [{step}/{total_steps}] Shipping workflow files...\n"
    )
    for wf in script_def.workflows:
        parts.append(f'call :ship_workflow "{wf}"\n')

    workflow_hints = "\n".join(
        _bat_workflow_hint(wf) for wf in script_def.workflows
    )
    parts.append(
        _BAT_FOOTER_FMT.format(
            display_name=script_def.display_name,
            workflow_hints=workflow_hints,
        )
    )

    return "".join(parts)


_ORCHESTRATOR_TEMPLATE = """\
@echo off
REM ============================================================================
REM Comfy Workflow - SDXL fresh setup (Install SDXL mini-series)
REM ============================================================================
REM Generated by installer/benchmark/generate_install_scripts.py
REM DO NOT EDIT BY HAND. Re-generate via:
REM   python -m installer.benchmark.generate_install_scripts
REM ============================================================================
REM Master orchestrator: runs 01-install-base.bat + install-sdxl.bat in
REM sequence to deliver a fresh SDXL-ready ComfyUI install in one shot.
REM For SSH dispatch, set COMFY_NONINTERACTIVE=1 (or pass --unattended /
REM -u as first arg) to suppress pauses and the FIRST LAUNCH manual step
REM in 01-install-base.bat.
REM
REM Note: "Install SDXL" is a video from the install mini-series, which
REM is distinct from the Benchmark Pillar mini-series. The sidecar
REM README next to each workflow JSON tracks both cross-references.
REM ============================================================================
setlocal EnableDelayedExpansion

REM ----- Parse --unattended / -u flag (sets COMFY_NONINTERACTIVE) -----
if /i "%~1"=="--unattended" set "COMFY_NONINTERACTIVE=1"
if /i "%~1"=="-u" set "COMFY_NONINTERACTIVE=1"

REM ----- Locate script directory (so we can call sibling .bat files) -----
set "SCRIPT_DIR=%~dp0"
if "!SCRIPT_DIR:~-1!"=="\\" set "SCRIPT_DIR=!SCRIPT_DIR:~0,-1!"

echo.
echo ========================================================
echo  Comfy Workflow - SDXL fresh setup (SDXL Base 1.0)
echo ========================================================
echo.
echo  This will run, in sequence:
echo    [1/2] 01-install-base.bat  (ComfyUI portable + essential nodes)
echo    [2/2] install-sdxl.bat     (SDXL 1.0 model + workflow)
echo.
echo  Hardware minimo (SDXL): 16 GB RAM, 8 GB VRAM
echo  Total download: ~10 GiB (3 GiB ComfyUI portable + 7 GiB SDXL)
echo.
if defined COMFY_NONINTERACTIVE (
    echo  Running unattended ^(COMFY_NONINTERACTIVE=1^).
) else (
    echo  Press any key to continue, or close window to abort.
    pause
)

echo.
echo === [1/2] Running 01-install-base.bat ===
echo.
if not exist "!SCRIPT_DIR!\\01-install-base.bat" (
    echo ERROR: 01-install-base.bat not found next to setup-sdxl.bat.
    echo Expected: !SCRIPT_DIR!\\01-install-base.bat
    if not defined COMFY_NONINTERACTIVE pause
    exit /b 1
)
call "!SCRIPT_DIR!\\01-install-base.bat"
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: 01-install-base.bat failed ^(exit !ERRORLEVEL!^).
    if not defined COMFY_NONINTERACTIVE pause
    exit /b 1
)

echo.
echo === [2/2] Running install-sdxl.bat ===
echo.
if not exist "!SCRIPT_DIR!\\install-sdxl.bat" (
    echo ERROR: install-sdxl.bat not found next to setup-sdxl.bat.
    echo Expected: !SCRIPT_DIR!\\install-sdxl.bat
    if not defined COMFY_NONINTERACTIVE pause
    exit /b 1
)
call "!SCRIPT_DIR!\\install-sdxl.bat"
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo ERROR: install-sdxl.bat failed ^(exit !ERRORLEVEL!^).
    if not defined COMFY_NONINTERACTIVE pause
    exit /b 1
)

echo.
echo ========================================================
echo  SDXL SETUP COMPLETE
echo ========================================================
echo.
echo  Installed:
echo    - ComfyUI Portable (Python 3.13 + CUDA 13.0)
echo    - Essential custom nodes: Manager, rgthree, Crystools
echo    - SDXL Base 1.0 model
echo    - sdxl_base.json workflow + sidecar README
echo.
echo  Next steps:
echo    1. Run C:\\ComfyUI_windows_portable\\run_nvidia_gpu.bat
echo    2. Open workflow: ComfyUI -^> Workflows -^> sdxl_base.json
echo.
if not defined COMFY_NONINTERACTIVE pause
endlocal
exit /b 0
"""


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
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not setup_dir.is_dir():
        raise SystemExit(f"--setup-dir does not exist: {setup_dir}")

    manifest = _load_manifest(manifest_path)
    logger.info(
        "loaded manifest %s (%d model entries)",
        manifest_path, len(manifest),
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
        _emit(setup_dir / script_name, content)

    _emit(setup_dir / ORCHESTRATOR_NAME, _ORCHESTRATOR_TEMPLATE)

    summary_tag = "DRY-RUN SUMMARY" if args.dry_run else "DONE"
    logger.info(
        "=== %s === written=%d, idempotent=%d, total=%d",
        summary_tag, n_written, n_idempotent, n_written + n_idempotent,
    )


if __name__ == "__main__":
    main()
