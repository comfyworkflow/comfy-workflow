@echo off
REM ============================================================================
REM Comfy Workflow - SDXL Finetunes Pack (Juggernaut, RealVis, Pony, NoobAI)
REM ============================================================================
REM Pillars: SDXL family
REM Models: Juggernaut XL Ragnarok, RealVisXL V5.0, Pony Diffusion V6 XL,
REM         NoobAI XL V-Pred 1.0, + shared sdxl_vae_fp16_fix and upscalers.
REM Hardware minimum: 16 GB RAM, 8 GB VRAM (base, no hi-res); 12 GB VRAM (hi-res 1.5x + ADetailer fits tight)
REM Total download (this script): ~28.5 GiB
REM ============================================================================
setlocal EnableDelayedExpansion

set "REPO_RAW=https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main"
set "CATEGORY=Image\SDXL Finetunes"

REM Failure tracking — populated by subroutines for soft failures
set "BAT_FAILURES="
set /a BAT_FAIL_COUNT=0

REM ----- Pre-req 1: ComfyUI base install must exist -----
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

REM ----- Pre-req 2: git must be installed (for custom node clones) -----
where git >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo =========================================================================
    echo   ERROR: git not installed
    echo =========================================================================
    echo.
    echo   This install uses git to install ComfyUI custom nodes ^(Impact Pack,
    echo   ControlNet Aux^). Git for Windows is free and quick to install.
    echo.
    echo   How to fix:
    echo.
    echo    1. Open:    https://git-scm.com/download/win
    echo    2. Download and install with defaults
    echo    3. Restart this script
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
REM Subroutine: install_custom_node NODE_NAME REPO_URL
REM Returns exit 1 on git clone failure (caller halts).
REM Pip dep failures are SOFT-tracked (not fatal — node may still partial-load).
REM ============================================================================
:install_custom_node
set "NODE_NAME=%~1"
set "NODE_URL=%~2"
set "NODES_DIR=C:\ComfyUI_windows_portable\ComfyUI\custom_nodes"
if exist "!NODES_DIR!\!NODE_NAME!\" (
    echo   Skip ^(already installed^): !NODE_NAME!
    goto :install_node_deps
)
echo   git clone: !NODE_NAME!
git clone --depth=1 "!NODE_URL!" "!NODES_DIR!\!NODE_NAME!"
if !ERRORLEVEL! NEQ 0 (
    echo   ERROR: git clone failed for !NODE_NAME!
    exit /b 1
)
:install_node_deps
if not exist "!NODES_DIR!\!NODE_NAME!\requirements.txt" exit /b 0
echo   Installing pip dependencies for !NODE_NAME! ^(--no-cache-dir^)
"C:\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install --no-cache-dir --no-warn-script-location -q -r "!NODES_DIR!\!NODE_NAME!\requirements.txt"
if !ERRORLEVEL! EQU 0 exit /b 0
echo   pip --no-cache-dir failed, retrying with --user fallback
"C:\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install --no-cache-dir --user --no-warn-script-location -q -r "!NODES_DIR!\!NODE_NAME!\requirements.txt"
if !ERRORLEVEL! EQU 0 exit /b 0
echo   ERROR: pip deps FAILED for !NODE_NAME! after --user fallback ^(FaceDetailer/ADetailer may not load^)
set "BAT_FAILURES=!BAT_FAILURES! [pip:!NODE_NAME!]"
set /a BAT_FAIL_COUNT+=1
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

REM ============================================================================
REM Subroutine: cleanup_flat_workflows PATTERN
REM ============================================================================
:cleanup_flat_workflows
set "PATTERN=%~1"
set "FLAT_DIR=C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow\Image"
if not exist "!FLAT_DIR!" exit /b 0
for %%F in ("!FLAT_DIR!\!PATTERN!") do (
    if exist "%%F" (
        echo   Removing legacy flat workflow: %%~nxF
        del "%%F" >nul 2>&1
    )
)
exit /b 0

:main

echo.
echo ========================================================
echo  Comfy Workflow - SDXL Finetunes Pack
echo ========================================================
echo  Models: Juggernaut Ragnarok + RealVis V5.0 + Pony V6 + NoobAI V-Pred
echo  Hardware min: 16 GB RAM, 8 GB VRAM ^(base^) / 12 GB VRAM ^(hi-res 1.5x + ADetailer - tight on 12 GB^)
echo  Total download: ~28.5 GiB
echo ========================================================
echo.

REM ----- Pre-create model folders (idempotent) -----
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\checkpoints"     mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\checkpoints"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\vae"             mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\vae"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\upscale_models"  mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\upscale_models"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\ultralytics\bbox" mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\ultralytics\bbox"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\models\sams"            mkdir "C:\ComfyUI_windows_portable\ComfyUI\models\sams"
if not exist "C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow" mkdir "C:\ComfyUI_windows_portable\ComfyUI\user\default\workflows\Comfy Workflow"

echo [1/5] Downloading 4 SDXL finetunes ^(~27.7 GiB^)...
call :download_or_skip "https://huggingface.co/prolapse/xl4supir/resolve/main/juggernautXL_ragnarok.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\juggernautXL_ragnarok.safetensors" 7105350162 || goto :critical_halt
call :download_or_skip "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\RealVisXL_V5.0_fp16.safetensors" 6938065488 || goto :critical_halt
call :download_or_skip "https://huggingface.co/AstraliteHeart/pony-diffusion-v6/resolve/main/v6.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\ponyDiffusionV6XL_v6.safetensors" 6938041050 || goto :critical_halt
call :download_or_skip "https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/resolve/main/NoobAI-XL-Vpred-v1.0.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\checkpoints\NoobAI-XL-Vpred-v1.0.safetensors" 7105350110 || goto :critical_halt

echo [2/5] Downloading shared SDXL VAE ^(0.32 GiB^)...
call :download_or_skip "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors" "C:\ComfyUI_windows_portable\ComfyUI\models\vae\sdxl_vae_fp16_fix.safetensors" 334641162 || goto :critical_halt

echo [3/5] Downloading hi-res upscale models ^(0.13 GiB^)...
call :download_or_skip "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth" "C:\ComfyUI_windows_portable\ComfyUI\models\upscale_models\4x-UltraSharp.pth" 66961958 || goto :critical_halt
call :download_or_skip "https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth" "C:\ComfyUI_windows_portable\ComfyUI\models\upscale_models\4x_NMKD-Siax_200k.pth" 66957746 || goto :critical_halt

echo [4a/5] Downloading face/hand detection models ^(0.07 GiB^)...
call :download_or_skip "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" "C:\ComfyUI_windows_portable\ComfyUI\models\ultralytics\bbox\face_yolov8m.pt" 52026019 || goto :critical_halt
call :download_or_skip "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt" "C:\ComfyUI_windows_portable\ComfyUI\models\ultralytics\bbox\hand_yolov8s.pt" 22507643 || goto :critical_halt

echo [4b/5] Downloading SAM mask model ^(0.36 GiB^)...
call :download_or_skip "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" "C:\ComfyUI_windows_portable\ComfyUI\models\sams\sam_vit_b_01ec64.pth" 375042383 || goto :critical_halt

echo [4c/5] Installing custom nodes ^(Impact Pack + Subpack + ControlNet Aux^)...
call :install_custom_node "ComfyUI-Impact-Pack" "https://github.com/ltdrdata/ComfyUI-Impact-Pack" || goto :critical_halt
call :install_custom_node "ComfyUI-Impact-Subpack" "https://github.com/ltdrdata/ComfyUI-Impact-Subpack" || goto :critical_halt
call :install_custom_node "comfyui_controlnet_aux" "https://github.com/Fannovel16/comfyui_controlnet_aux" || goto :critical_halt

echo [5/5] Pruning legacy flat-layout workflow files ^(sdxl_*.json^)...
call :cleanup_flat_workflows "sdxl_*.json"

echo [5/5] Shipping 4 workflow file^(s^)...
call :ship_workflow "juggernaut_xl_ragnarok.json"
call :ship_workflow "realvis_xl_v5.json"
call :ship_workflow "pony_v6_xl.json"
call :ship_workflow "noobai_xl_vpred.json"

goto :final_status

:critical_halt
echo.
echo =========================================================================
echo   CRITICAL FAILURE - install HALTED
echo =========================================================================
echo   A required step failed ^(model download or custom-node git clone^).
echo   Without it the workflows cannot run, so the install stopped.
echo   Check the last error above ^(usually network, HF availability, or git auth^),
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
    echo   SDXL Finetunes install complete - all steps OK.
    echo =========================================================================
    echo.
    echo   Models installed: Juggernaut Ragnarok ^(6.62 GB^) + RealVisXL V5.0 ^(6.46 GB^) + Pony V6 ^(6.46 GB^) + NoobAI V-Pred ^(6.62 GB^) + shared SDXL VAE ^(0.32 GB^) + upscalers ^(0.13 GB^) + face/hand detectors + SAM - total ~28.5 GB
    echo.
    echo   IMPORTANT: ComfyUI needs to RESTART to load the new custom nodes.
    echo   Close ComfyUI and re-open it before using the workflows.
    echo.
    echo   Workflows in ComfyUI sidebar:
    echo     Comfy Workflow ^> Image ^> SDXL Finetunes ^> ^(4 finetunes - one each^)
    echo.
    echo   Each workflow is MAX-quality showcase recipe ^(hi-res fix 1.5x + ADetailer + finetune-specific tags^).
    echo.
    echo   Default recommendation: juggernaut_xl_ragnarok.json ^(photoreal, ~15-20s on RTX 4090^)
    echo   For anime: noobai_xl_vpred.json ^(vPred plumbing pre-wired^)
    echo   For stylized fantasy: pony_v6_xl.json ^(score tags + source/rating steering^)
    echo.
    echo   Model licenses ^(check each model card for full terms^):
    echo     Juggernaut XL Ragnarok ^(RunDiffusion^):     CreativeML Open RAIL-M + RunDiffusion commercial addendum
    echo     RealVisXL V5.0 ^(SG161222^):                  CreativeML Open RAIL++-M ^(openrail++^)
    echo     Pony Diffusion V6 XL ^(PurpleSmartAI^):       Fair AI Public License 1.0-SD ^(modified^)
    echo     NoobAI XL V-Pred 1.0 ^(Laxhar Dream Lab^):   Fair AI Public License 1.0-SD ^(+ Laxhar addendum^)
    echo   Pony + NoobAI have commercial-use restrictions — read the model card before commercial use.
    echo.
    echo   Performance on RTX 3060 ^(12 GB^):
    echo     Base render only ^(no hi-res^):           ~26-45 seconds per image
    echo     Full workflow ^(hi-res 1.5x + ADetailer^): ~67 seconds per image ^(peak ~11.5 GB VRAM, fits 12 GB^)
    echo     Headroom is tight on 12 GB - close other GPU apps for headroom.
    echo.
    echo   Tutorial video ^(SDXL Finetunes Pack install + benchmark^):
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
    echo   SDXL Finetunes install FINISHED WITH !BAT_FAIL_COUNT! FAILURE^(S^):
    echo     !BAT_FAILURES!
    echo =========================================================================
    echo.
    echo   The install did NOT fully succeed. What this means per failure type:
    echo.
    echo     [pip:NODE_NAME] - python deps did not install for that custom node.
    echo                       Impact-Pack/Subpack/ControlNet Aux are REQUIRED for FaceDetailer.
    echo                       Without them the workflows will error at the FaceDetailer node.
    echo                       Fix: re-run this script, OR manually install:
    echo                       cd C:\ComfyUI_windows_portable\python_embeded
    echo                       python.exe -m pip install --no-cache-dir --user -r ^<custom_nodes\NODE_NAME\requirements.txt^>
    echo.
    echo     [ship:NAME.json] - workflow JSON was not downloaded from the public repo.
    echo                        The workflow will be missing from the ComfyUI sidebar.
    echo                        Fix: re-run this script ^(other downloads will be skipped^).
    echo.
    echo     [critical-step] - download or git clone failed. The install was halted.
    echo                        Fix: address the upstream error printed above, then re-run.
    echo.
    echo =========================================================================
    if not defined COMFY_NONINTERACTIVE pause
    endlocal
    exit /b 1
)
