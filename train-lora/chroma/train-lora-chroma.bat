@echo off
setlocal enabledelayedexpansion
title Train LoRA Chroma - Installer (downloads everything, 1 click)

REM ============================================================
REM  The ONLY file you download. It fetches everything else
REM  (workflows, config, caption helper, models) and installs
REM  OneTrainer - or REUSES it if you have it from the SDXL video.
REM  Tutorial video: coming with the premiere
REM  >>> Set the RAW line below to the repo's raw base (no trailing /) <<<
REM      e.g.: https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/main/train-lora/chroma
REM ============================================================
set "RAW=https://raw.githubusercontent.com/comfyworkflow/comfy-workflow/cw21-kit/train-lora/chroma"

set "COMFY=C:\ComfyUI_windows_portable\ComfyUI"
set "ROOT=%USERPROFILE%\Downloads\train-lora-chroma"

echo.
echo === [1/6] Checking Python + ComfyUI ===
python --version || ( echo ERROR: install Python 3.10-3.13 and tick "Add Python to PATH". & pause & exit /b 1 )
if not exist "%COMFY%\main.py" ( echo ERROR: ComfyUI not found at C:\ComfyUI_windows_portable ^(video #1^). & pause & exit /b 1 )
for %%D in ("%COMFY%\models\diffusion_models" "%COMFY%\models\text_encoders" "%COMFY%\models\vae" "%COMFY%\models\loras") do if not exist %%D mkdir %%D

echo.
echo === [2/6] Creating work folders ===
for %%D in (raw_gen dataset workspace output) do mkdir "%ROOT%\%%D" 2>nul
echo [OK] %ROOT%

echo.
echo === [3/6] Downloading the workflows into ComfyUI (Workflows menu -^> Comfy Workflow -^> LoRA -^> Chroma) ===
set "WF=%COMFY%\user\default\workflows\Comfy Workflow\LoRA\Chroma"
mkdir "%WF%" 2>nul
curl -L -o "%WF%\1 - Generate images.json" "%RAW%/workflow_1_generate.json" || goto :dlerr
curl -L -o "%WF%\2 - Use LoRA.json"        "%RAW%/workflow_2_use_lora.json" || goto :dlerr

echo.
echo === [4/6] Downloading helpers (config + caption helper + guide) ===
curl -L -o "%ROOT%\config.template.json"    "%RAW%/config.template.json"   || goto :dlerr
curl -L -o "%ROOT%\concepts.template.json"  "%RAW%/concepts.template.json" || goto :dlerr
curl -L -o "%ROOT%\dataset\nl_captions.py"  "%RAW%/nl_captions.py"         || goto :dlerr
curl -L -o "%ROOT%\dataset\nl_captions.bat" "%RAW%/nl_captions.bat"        || goto :dlerr
curl -L -o "%ROOT%\README.md"               "%RAW%/README.md"              || goto :dlerr

echo.
echo === [5/6] Models for ComfyUI (Chroma1-HD ~17.8 GB + T5 encoder ~5.7 GB + VAE, skipped if you have them) ===
call :fetch "https://huggingface.co/lodestones/Chroma1-HD/resolve/main/Chroma1-HD.safetensors" "%COMFY%\models\diffusion_models\Chroma1-HD.safetensors" 17800038288 || goto :dlerr
call :fetch "https://huggingface.co/silveroxides/flan-t5-xxl-encoder-only/resolve/main/flan-t5-xxl_float8_e4m3fn_scaled_stochastic.safetensors" "%COMFY%\models\text_encoders\flan-t5-xxl_float8_e4m3fn_scaled_stochastic.safetensors" 5684140184 || goto :dlerr
call :fetch "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/resolve/main/split_files/vae/ae.safetensors" "%COMFY%\models\vae\ae.safetensors" 335304388 || goto :dlerr

echo.
echo === [6/6] OneTrainer (reused from the SDXL video if you already have it) ===
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
echo [OK] OneTrainer already installed at %OT% - reusing it ^(nothing to reinstall^).
:ot_done

REM --- generate the preset with the logged-in user's paths (validated JSON before writing) ---
if not exist "%OT%\training_presets" mkdir "%OT%\training_presets"
powershell -NoProfile -Command "$r='%ROOT%' -replace '\\','\\'; $t=(Get-Content -Raw -LiteralPath '%ROOT%\config.template.json') -replace '__ROOT__',$r; $null=$t|ConvertFrom-Json; [System.IO.File]::WriteAllText('%OT%\training_presets\train-lora-chroma.json',$t)"
if not exist "%OT%\training_presets\train-lora-chroma.json" echo WARNING: could not write the training preset. In OneTrainer, configure the settings by hand from the README.

REM --- link the dataset: Chroma-specific concepts file (does NOT touch your SDXL concepts.json) ---
if not exist "%OT%\training_concepts" mkdir "%OT%\training_concepts"
powershell -NoProfile -Command "$r='%ROOT%' -replace '\\','\\'; $t=(Get-Content -Raw -LiteralPath '%ROOT%\concepts.template.json') -replace '__ROOT__',$r; $null=$t|ConvertFrom-Json; [System.IO.File]::WriteAllText('%OT%\training_concepts\train-lora-chroma-concepts.json',$t)"
if not exist "%OT%\training_concepts\concepts.json" copy /y "%OT%\training_concepts\train-lora-chroma-concepts.json" "%OT%\training_concepts\concepts.json" >nul
if not exist "%OT%\training_concepts\train-lora-chroma-concepts.json" echo WARNING: could not auto-link the dataset. In OneTrainer: concepts tab -^> Add Concept -^> Path = "%ROOT%\dataset".

echo.
echo **** ALL DONE. ****
echo - Workflows in ComfyUI: Workflows menu -^> Comfy Workflow -^> LoRA -^> Chroma.
echo - Captions: put your images in %ROOT%\dataset and run nl_captions.bat there.
echo - Training: %OT%\start-ui.bat  (load the train-lora-chroma preset).
echo - Your data + guide (README): %ROOT%
pause
endlocal
exit /b 0

:fetch
set "F_URL=%~1"
set "F_TGT=%~2"
set "F_EXP=%~3"
if exist "%F_TGT%" (
  for %%A in ("%F_TGT%") do set "F_ACT=%%~zA"
  if "!F_ACT!"=="%F_EXP%" ( echo [OK] already present: %~nx2 & exit /b 0 )
  echo Re-downloading ^(size mismatch^): %~nx2
  del "%F_TGT%" >nul 2>&1
)
echo Downloading: %~nx2
curl -L --fail --retry 3 --retry-delay 5 -o "%F_TGT%" "%F_URL%" || exit /b 1
for %%A in ("%F_TGT%") do set "F_ACT=%%~zA"
if not "!F_ACT!"=="%F_EXP%" ( echo ERROR: size mismatch after download: %~nx2 & exit /b 1 )
echo [OK] verified: %~nx2
exit /b 0

:dlerr
echo.
echo ERROR downloading a file. Check: (1) your internet, (2) the RAW line at the top of this .bat.
pause
endlocal
exit /b 1
