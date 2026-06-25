@echo off
setlocal enabledelayedexpansion
title CW20 - Installer (downloads everything, 1 click)

REM ============================================================
REM  The ONLY file you download. It fetches everything else
REM  (workflows, config, captions, model) and installs OneTrainer.
REM  >>> Set the RAW line below to the repo's raw base (no trailing /) <<<
REM      e.g.: https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main/CW20
REM ============================================================
set "RAW=https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/cw20-install-kit/CW20"

set "COMFY=C:\ComfyUI_windows_portable\ComfyUI"
set "CKPT=%COMFY%\models\checkpoints"
set "ROOT=%USERPROFILE%\Downloads\CW20_SDXL_LoRA"

echo.
echo === [1/6] Checking Python + ComfyUI ===
python --version || ( echo ERROR: install Python 3.10-3.13 and tick "Add Python to PATH". & pause & exit /b 1 )
if not exist "%CKPT%" ( echo ERROR: ComfyUI not found at C:\ComfyUI_windows_portable ^(video #1^). & pause & exit /b 1 )
if not exist "%COMFY%\models\loras" mkdir "%COMFY%\models\loras"

echo.
echo === [2/6] Creating work folders ===
for %%D in (raw_gen dataset workspace output) do mkdir "%ROOT%\%%D" 2>nul
echo [OK] %ROOT%

echo.
echo === [3/6] Downloading the workflows into ComfyUI (Workflows menu -^> Comfy Workflow -^> CW20) ===
set "WF=%COMFY%\user\default\workflows\Comfy Workflow\CW20"
mkdir "%WF%" 2>nul
curl -L -o "%WF%\1 - Generate images.json" "%RAW%/workflow_1_generate.json" || goto :dlerr
curl -L -o "%WF%\2 - Use LoRA.json"        "%RAW%/workflow_2_use_lora.json" || goto :dlerr

echo.
echo === [4/6] Downloading helpers (config + captions + guide) ===
curl -L -o "%ROOT%\CW20_SDXL_config.template.json" "%RAW%/CW20_SDXL_config.template.json" || goto :dlerr
curl -L -o "%ROOT%\dataset\simple_captions.bat"    "%RAW%/simple_captions.bat"            || goto :dlerr
curl -L -o "%ROOT%\README.md"                      "%RAW%/README.md"                      || goto :dlerr

echo.
echo === [5/6] Model (Juggernaut XL v9, ~7.1 GB if missing) ===
set "JUG=Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
if exist "%CKPT%\%JUG%" ( echo [OK] model already present. ) else (
  curl -L -o "%CKPT%\%JUG%" https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors || goto :dlerr
)

echo.
echo === [6/6] OneTrainer (installs ONCE in Downloads\OneTrainer, reused next time) ===
set "OT=%USERPROFILE%\Downloads\OneTrainer"
if exist "%OT%\start-ui.bat" goto :ot_ready
echo Installing OneTrainer ^(venv + torch + deps, ~10-20 min^)...
set "ZIP=%USERPROFILE%\Downloads\onetrainer.zip"
curl -L -o "%ZIP%" https://github.com/Nerogar/OneTrainer/archive/3e3b3e8f.zip || goto :dlerr
"%SystemRoot%\System32\tar.exe" -xf "%ZIP%" -C "%USERPROFILE%\Downloads"
if exist "%OT%" rmdir /s /q "%OT%"
for /d %%D in ("%USERPROFILE%\Downloads\OneTrainer-*") do move "%%D" "%OT%" >nul
del "%ZIP%" >nul 2>&1
if not exist "%OT%\install.bat" ( echo ERROR: OneTrainer extraction failed. & pause & exit /b 1 )
set "PIP_NO_CACHE_DIR=1"
pushd "%OT%"
call install.bat
popd
goto :ot_done
:ot_ready
echo [OK] OneTrainer already installed at %OT% - skipping its install.
:ot_done

REM --- generate the CW20 preset into OneTrainer (always; data stays in CW20_SDXL_LoRA) ---
if not exist "%OT%\training_presets" mkdir "%OT%\training_presets"
powershell -NoProfile -Command "$r='%ROOT%' -replace '\\','\\'; $t=(Get-Content -Raw -LiteralPath '%ROOT%\CW20_SDXL_config.template.json') -replace '__ROOT__',$r; [System.IO.File]::WriteAllText('%OT%\training_presets\CW20_SDXL_config.json',$t)"

echo.
echo **** ALL DONE. ****
echo - Workflows in ComfyUI: Workflows menu -^> Comfy Workflow -^> CW20.
echo - Training: %OT%\start-ui.bat  (load the CW20_SDXL_config preset).
echo - Your data + guide (README): %ROOT%
pause
endlocal
exit /b 0

:dlerr
echo.
echo ERROR downloading a file. Check: (1) your internet, (2) the RAW line at the top of this .bat.
pause
endlocal
exit /b 1
