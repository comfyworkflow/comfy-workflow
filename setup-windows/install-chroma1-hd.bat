@echo off
REM ============================================================================
REM Comfy Workflow - Chroma1-HD (HD fp8 + HD bf16 + Flash)
REM ============================================================================
REM Pillars: Chroma1-HD family
REM Models: Chroma1-HD fp8 (scaled), Chroma1-HD bf16, Chroma1-Flash,
REM         + FLAN-T5-XXL text encoder (required) + shared FLUX VAE.
REM Hardware minimum: 32 GB RAM, 12 GB VRAM (fp8 - no offload needed);
REM                   bf16/Flash on 12 GB cards run via RAM offload (64 GB RAM recommended)
REM Total download (this script): ~47.3 GiB
REM ============================================================================
setlocal EnableDelayedExpansion

set "REPO_RAW=https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main"
set "CATEGORY=Image\Chroma1-HD"

REM Failure tracking - populated by subroutines for soft failures
set "BAT_FAILURES="
set /a BAT_FAIL_COUNT=0

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
REM Subroutine: download_or_skip URL TARGET EXPECTED_BYTES
REM Returns exit 1 on failure (caller halts via || goto :critical_halt)
REM ============================================================================
:download_or_skip
set "URL=%~1"
set "TARGET=%~2"
set "EXPECTED=%~3"
if exist "!TARGET!" (
    for %%A in ("!TARGET!") do set "ACTUAL=%%~zA"
    if "!ACTUAL!"=="!EXPECTED!" (
        echo   Skip ^(size match^): !TARGET!
        exit /b 0
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
exit /b 0

REM ============================================================================
REM Subroutine: ship_workflow WF_JSON  (production: curl from REPO_RAW)
REM SOFT-tracks failure (workflow file is small; user can retry or fetch manually)
REM ============================================================================
:ship_workflow
set "WF_JSON=%~1"
set "DEST_BASE=C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow"
set "DEST_DIR=!DEST_BASE!\!CATEGORY!"
set "CATEGORY_URL=!CATEGORY:\=/!"
set "CATEGORY_URL=!CATEGORY_URL: =%%20!"
if not exist "!DEST_DIR!" mkdir "!DEST_DIR!"
echo   Shipping workflow: !WF_JSON! ^(category: !CATEGORY!^)
curl.exe -L --fail --silent --retry 3 --retry-delay 5 -o "!DEST_DIR!\!WF_JSON!" "!REPO_RAW!/installer/benchmark/workflows_distribute/!CATEGORY_URL!/!WF_JSON!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: download failed for !WF_JSON!
    set "BAT_FAILURES=!BAT_FAILURES! [ship:!WF_JSON!]"
    set /a BAT_FAIL_COUNT+=1
    exit /b 0
)
echo   OK: shipped !WF_JSON!
exit /b 0

:main

echo.
echo ========================================================
echo  Comfy Workflow - Chroma1-HD
echo ========================================================
echo  Models: Chroma1-HD fp8 + Chroma1-HD bf16 + Chroma1-Flash
echo          + FLAN-T5-XXL encoder ^(required^) + FLUX VAE
echo  Hardware min: 32 GB RAM, 12 GB VRAM ^(fp8 fits 12 GB without offload^)
echo                bf16/Flash on 12 GB cards offload to RAM - 64 GB RAM recommended
echo  Total download: ~47.3 GiB
echo ========================================================
echo.

REM ----- Pre-create model folders (idempotent) -----
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models" mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders"    mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\vae"              mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\vae"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow" mkdir "C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow"

echo [1/5] Downloading Chroma1-HD fp8 - production default ^(9.19 GiB^)...
call :download_or_skip "https://huggingface.co/silveroxides/Chroma1-HD-fp8-scaled/resolve/main/Chroma1-HD-fp8_scaled_defaultloader_hybrid_large_rev2.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\Chroma1-HD-fp8_scaled_defaultloader_hybrid_large_rev2.safetensors" 9193371409 || goto :critical_halt

echo [2/5] Downloading Chroma1-HD bf16 - reference precision ^(16.6 GiB^)...
call :download_or_skip "https://huggingface.co/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\Chroma1-HD.safetensors" 17800038288 || goto :critical_halt

echo [3/5] Downloading Chroma1-Flash - 8-step speed variant ^(16.6 GiB^)...
call :download_or_skip "https://huggingface.co/lodestones/Chroma1-Flash/resolve/main/Chroma1-HD-Flash.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\Chroma1-HD-Flash.safetensors" 17800038288 || goto :critical_halt

echo [4/5] Downloading FLAN-T5-XXL text encoder + FLUX VAE ^(5.6 GiB^)...
call :download_or_skip "https://huggingface.co/silveroxides/flan-t5-xxl-encoder-only/resolve/main/flan-t5-xxl_float8_e4m3fn_scaled_stochastic.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\text_encoders\flan-t5-xxl_float8_e4m3fn_scaled_stochastic.safetensors" 5684140184 || goto :critical_halt
call :download_or_skip "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\vae\ae.safetensors" 335304388 || goto :critical_halt

echo [5/5] Shipping 3 workflow file^(s^)...
call :ship_workflow "chroma1_hd_fp8.json"
call :ship_workflow "chroma1_hd_bf16.json"
call :ship_workflow "chroma1_flash.json"

goto :final_status

:critical_halt
echo.
echo =========================================================================
echo   CRITICAL FAILURE - install HALTED
echo =========================================================================
echo   A required model download failed. Without it the workflows cannot
echo   run, so the install stopped.
echo   Check the last error above ^(usually network or HF availability^),
echo   fix it, then re-run this script. Earlier successful downloads will be
echo   skipped automatically ^(byte-size match^).
echo =========================================================================
set "BAT_FAILURES=!BAT_FAILURES! [critical-step]"
set /a BAT_FAIL_COUNT+=1
goto :final_status

:final_status
echo.
if !BAT_FAIL_COUNT! EQU 0 (
    echo =========================================================================
    echo   Chroma1-HD install complete - all steps OK.
    echo =========================================================================
    echo.
    echo   Models installed: Chroma1-HD fp8 ^(9.19 GB^) + Chroma1-HD bf16 ^(17.80 GB^) + Chroma1-Flash ^(17.80 GB^) + FLAN-T5-XXL encoder ^(5.68 GB^) + FLUX VAE ^(0.34 GB^) - total ~50.8 GB
    echo.
    echo   Workflows in ComfyUI sidebar:
    echo     Comfy Workflow ^> Image ^> Chroma1-HD ^> ^(HD fp8 + HD bf16 + Flash^)
    echo.
    echo   No custom nodes needed - everything runs on ComfyUI core nodes.
    echo.
    echo   Default recommendation: chroma1_hd_fp8.json ^(26 steps, CFG 3.8 - fits 12 GB VRAM without offload, peak ~10 GB^)
    echo   For reference precision: chroma1_hd_bf16.json ^(same recipe, 16-bit weights^)
    echo   For speed: chroma1_flash.json ^(8-step heun, CFG 1.0^)
    echo.
    echo   IMPORTANT: all three workflows use the FLAN-T5-XXL text encoder
    echo   installed by this script. Do NOT swap it for the regular t5xxl -
    echo   the FLAN build renders text in images dramatically better.
    echo.
    echo   Model licenses: Chroma1-HD, Chroma1-Flash, the fp8 build and the
    echo   FLAN-T5-XXL encoder are all Apache-2.0 ^(check each model card^).
    echo.
    echo   Performance ^(1152x1152, warm, this recipe^):
    echo     RTX 5090:  fp8 ~16s  /  bf16 ~26s  /  Flash ~8s
    echo     RTX 4090:  fp8 ~22s  /  bf16 ~32s  /  Flash ~10s
    echo     RTX 3060:  fp8 ~3m21s / bf16 ~3m30s / Flash ~62s
    echo   VRAM: fp8 runs on a 12 GB card with NO offload ^(peak ~10 GB^).
    echo   bf16 and Flash on 12 GB cards offload weights to system RAM -
    echo   they work ^(tested with 64 GB RAM^) but are slower and need
    echo   plenty of free RAM. On 16 GB+ VRAM cards everything fits.
    echo.
    echo   Tutorial video ^(Chroma1-HD install + benchmark^):
    echo   ^(Coming soon - watch the repo for updates^)
    echo.
    echo   Repo ^(all installs^):
    echo   https://github.com/comfyworkflow/comfy-workflow
    echo.
    echo =========================================================================
    if not defined COMFY_NONINTERACTIVE pause
    endlocal
    exit /b 0
) else (
    echo =========================================================================
    echo   Chroma1-HD install FINISHED WITH !BAT_FAIL_COUNT! FAILURE^(S^):
    echo     !BAT_FAILURES!
    echo =========================================================================
    echo.
    echo   The install did NOT fully succeed. What this means per failure type:
    echo.
    echo     [ship:NAME.json] - workflow JSON was not downloaded from the public repo.
    echo                        The workflow will be missing from the ComfyUI sidebar.
    echo                        Fix: re-run this script ^(other downloads will be skipped^).
    echo.
    echo     [critical-step] - a model download failed. The install was halted.
    echo                        Fix: address the upstream error printed above, then re-run.
    echo.
    echo =========================================================================
    if not defined COMFY_NONINTERACTIVE pause
    endlocal
    exit /b 1
)
