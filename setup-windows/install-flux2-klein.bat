@echo off
REM ============================================================================
REM Comfy Workflow - FLUX.2 Klein 4B (base + distilled, bf16 + fp8)
REM ============================================================================
REM Pillars: #7
REM Models: klein_4b_base_bf16, klein_4b_distilled_bf16, klein_4b_base_fp8,
REM         klein_4b_distilled_fp8, klein_shared_encoder, flux_2_shared_vae
REM Hardware minimo: 32 GB RAM, 8 GB VRAM (fp8); 12 GB VRAM (bf16)
REM Total download (this script): ~32.04 GiB
REM ============================================================================
setlocal EnableDelayedExpansion

set "REPO_RAW=https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main"
set "CATEGORY=Image\Klein"

REM ----- Pre-req: ComfyUI base install must exist -----
if not exist "C:\ComfyUI_windows_portable\ComfyUI\main.py" (
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
    echo   https://youtu.be/PZmJqxP5ajs
    echo.
    echo =========================================================================
    if not defined COMFY_NONINTERACTIVE pause
    exit /b 1
)

goto :main

REM ============================================================================
REM Subroutine: download_or_skip
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
REM Subroutine: ship_workflow
REM ============================================================================
:ship_workflow
set "WF_JSON=%~1"
set "DEST_BASE=C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow"
set "DEST_DIR=!DEST_BASE!\!CATEGORY!"
set "CATEGORY_URL=!CATEGORY:\=/!"
if not exist "!DEST_DIR!" mkdir "!DEST_DIR!"
echo   Shipping workflow: !WF_JSON! ^(category: !CATEGORY!^)
curl.exe -L --fail --silent --retry 3 --retry-delay 5 -o "!DEST_DIR!\!WF_JSON!" "!REPO_RAW!/installer/benchmark/workflows_distribute/!CATEGORY_URL!/!WF_JSON!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: download failed for !WF_JSON!
    exit /b 1
)
goto :eof

REM ============================================================================
REM Subroutine: cleanup_flat_workflows
REM ============================================================================
:cleanup_flat_workflows
set "PATTERN=%~1"
set "FLAT_DIR=C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow\Image"
if not exist "!FLAT_DIR!" goto :eof
for %%F in ("!FLAT_DIR!\!PATTERN!") do (
    if exist "%%F" (
        echo   Removing legacy flat workflow: %%~nxF
        del "%%F" >nul 2>&1
    )
)
goto :eof

:main

echo.
echo ========================================================
echo  Comfy Workflow - FLUX.2 Klein 4B (base + distilled, bf16 + fp8)
echo ========================================================
echo  Pillars: #7
echo  Hardware minimo: 32 GB RAM, 8 GB VRAM (fp8) / 12 GB VRAM (bf16)
echo  Total download: ~32.04 GiB
echo ========================================================
echo.

REM ----- Pre-create model folders (idempotent) -----
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models" mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders"    mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\vae"              mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\vae"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow" mkdir "C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow"

echo [1/3] Downloading model files (total: ~32.04 GiB)...
call :download_or_skip "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\flux-2-klein-base-4b.safetensors" 7751105712
call :download_or_skip "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\flux-2-klein-4b.safetensors" 7751105712
call :download_or_skip "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\flux-2-klein-base-4b-fp8.safetensors" 4089498488
call :download_or_skip "https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\flux-2-klein-4b-fp8.safetensors" 4070624520
call :download_or_skip "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders\qwen_3_4b.safetensors" 8044982048
call :download_or_skip "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\vae\flux2-vae.safetensors" 336213556

echo [2/3] Pruning legacy flat-layout workflow files (klein_*.json)...
call :cleanup_flat_workflows "klein_*.json"

echo [3/3] Shipping 4 workflow file(s)...
call :ship_workflow "klein_4b_base_bf16.json"
call :ship_workflow "klein_4b_distilled_bf16.json"
call :ship_workflow "klein_4b_base_fp8.json"
call :ship_workflow "klein_4b_distilled_fp8.json"

echo.
echo =========================================================================
echo   FLUX.2 Klein 4B install complete^^!
echo =========================================================================
echo.
echo   Models installed: base bf16 (7.75 GB) + distilled bf16 (7.75 GB) + base fp8 (4.07 GB) + distilled fp8 (4.07 GB) + Qwen3-4B encoder (8.04 GB) + FLUX2 VAE (0.3 GB) - total ~32.0 GB
echo.
echo   Workflows in ComfyUI sidebar:
echo     Comfy Workflow ^> Image ^> Klein ^> (base + distilled x bf16 + fp8)
echo.
echo   Default recommendation: klein_4b_distilled_fp8.json (4 steps, CFG 1.0, ~1.2s warm on RTX 4090)
echo   For reference precision: klein_4b_base_bf16.json (20 steps, CFG 5.0)
echo   BFL Apache-2.0 weights. Native 1024x1024 + supports non-square buckets up to 1344x768.
echo.
echo   Tutorial video ^(Video #9 - FLUX.2 Klein install + benchmark^):
echo   (Coming soon - watch the repo for updates)
echo.
echo   Repo ^(all installs^):
echo   https://github.com/comfyworkflow/comfy-workflow
echo.
echo =========================================================================
echo.
if not defined COMFY_NONINTERACTIVE pause
endlocal
exit /b 0
