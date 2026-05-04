@echo off
setlocal EnableDelayedExpansion

REM ==========================================================
REM Comfy Workflow - ComfyUI Portable Installer
REM ==========================================================
REM Installs or updates the Comfy Workflow ComfyUI Portable
REM environment on a Windows 10 / Windows 11 machine with an
REM NVIDIA GPU (RTX 20-series and above).
REM
REM When you run this script, it detects whether ComfyUI is
REM already installed at C:\ComfyUI_windows_portable\ and lets
REM you choose:
REM
REM   [U] UPDATE - keep your existing install, just refresh the
REM                3 essential custom nodes and the requirements.
REM                Your models and workflows are preserved.
REM   [F] FRESH  - back up the old folder to .OLD-<timestamp>\
REM                and install everything from scratch.
REM   [C] CANCEL - exit without changing anything.
REM
REM If no install is found, the script proceeds in Fresh mode
REM automatically (no prompt).
REM
REM This script does NOT modify any folder on your system
REM other than C:\ComfyUI_windows_portable\ and the global
REM Python 3.13 site-packages.
REM
REM Note: winget package installs may show UAC prompts.
REM
REM Tech debt note: in Update mode the "Locate 7z.exe" check
REM still runs but the variable is never used. Cheap, no-op,
REM left in place in case a future Update mode applies a .7z
REM delta patch instead of skipping download entirely.
REM ==========================================================

goto :main

REM ==========================================================
REM Subroutine: winget_install
REM   %1 = package id
REM   %2 = friendly name
REM   %3 = step number string (e.g. "1/12")
REM ==========================================================
:winget_install
echo.
echo [Step  %~3] Installing %~2 via winget (id: %~1)...
winget list --id %~1 --exact --source winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Already installed, skipping.
    goto :eof
)
winget install --id %~1 --exact --silent --accept-package-agreements --accept-source-agreements
REM Note: winget exit codes vary; we re-validate after refreshing PATH.
goto :eof

REM ==========================================================
REM Subroutine: ensure_node
REM   %1 = folder name (e.g. "ComfyUI-Manager")
REM   %2 = git repo URL
REM
REM Clones if missing, otherwise tries a fast-forward-only pull.
REM --ff-only means a viewer with local edits will get a clean
REM "not possible to fast-forward" error instead of a merge
REM conflict; we just warn and continue with the existing copy.
REM
REM Structure: early-return via `goto :eof` to avoid nesting the
REM errorlevel check inside an `else (...)` block (which would
REM evaluate ERRORLEVEL at parse-time, not run-time, and produce
REM false positives or silent misses).
REM ==========================================================
:ensure_node
if not exist "%~1" (
    echo   Cloning %~1...
    git clone "%~2"
    goto :eof
)
echo   Updating %~1 ^(git pull --ff-only^)...
git -C "%~1" pull --ff-only
if !ERRORLEVEL! NEQ 0 (
    echo   WARNING: git pull failed for %~1. Continuing with existing version.
    echo   ^(This usually means local changes -- fix manually if needed.^)
)
goto :eof

REM ==========================================================
REM Main
REM ==========================================================
:main

echo.
echo ========================================================
echo  Comfy Workflow - ComfyUI Portable Installer
echo ========================================================
echo.

REM ----- Check NVIDIA GPU -----
echo [Check 1/3] Looking for NVIDIA GPU...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :nvidia_not_found
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto :nvidia_not_found
echo   NVIDIA GPU detected.
goto :nvidia_ok

:nvidia_not_found
echo.
echo ERROR: NVIDIA GPU not detected.
echo.
echo ComfyUI Portable (CUDA 13.0 build) requires an NVIDIA GPU
echo with recent drivers. If you have an NVIDIA GPU, install or
echo update drivers from:
echo   https://www.nvidia.com/Download/index.aspx
echo and re-run this script.
echo.
echo Aborting installation.
pause
exit /b 1

:nvidia_ok

REM ----- Check WinGet -----
echo [Check 2/3] Looking for winget...
where winget >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: winget not found.
    echo.
    echo winget is built into Windows 10 ^(version 1809+^) and Windows 11.
    echo If it is missing, install "App Installer" from the Microsoft Store.
    echo.
    pause
    exit /b 1
)
echo   winget is available.

REM ==========================================================
REM Check 3/3 - Existing install detection + mode selection
REM ==========================================================
REM Note: this block uses !VAR! (delayed expansion) for any
REM variable set inside the block (EXISTING_COMMIT, USER_CHOICE,
REM INSTALL_MODE), because cmd expands %VAR% at parse time --
REM before the block runs -- which would yield empty values.
echo [Check 3/3] Looking for existing ComfyUI install...
set "INSTALL_MODE="
if exist "C:\ComfyUI_windows_portable\ComfyUI\main.py" (
    REM Try to read the current ComfyUI commit hash for context.
    set "EXISTING_COMMIT=unknown"
    for /f "delims=" %%i in ('git -C "C:\ComfyUI_windows_portable\ComfyUI" rev-parse --short HEAD 2^>nul') do set "EXISTING_COMMIT=%%i"

    echo.
    echo ========================================================
    echo  EXISTING INSTALL DETECTED
    echo ========================================================
    echo.
    echo  C:\ComfyUI_windows_portable\ already exists.
    echo  ComfyUI commit: !EXISTING_COMMIT!
    echo.
    echo  What do you want to do?
    echo.
    echo    [U] UPDATE  - Keep existing ComfyUI, update the 3
    echo                  essential custom nodes ^(Manager, rgthree,
    echo                  Crystools^) and install any missing pieces.
    echo                  Your models and workflows are preserved.
    echo                  RECOMMENDED for most users.
    echo.
    echo    [F] FRESH   - Back up existing folder to
    echo                  C:\ComfyUI_windows_portable.OLD-^<timestamp^>\
    echo                  then install everything from scratch.
    echo                  Use this only if your install is broken.
    echo                  WARNING: any models/workflows in the
    echo                  current folder will be in the backup,
    echo                  not in the new install.
    echo.
    echo    [C] CANCEL  - Exit without making any changes.
    echo.
    set "USER_CHOICE="
    set /p "USER_CHOICE=Choice [U/F/C]: "
    if /i "!USER_CHOICE!"=="U" set "INSTALL_MODE=update"
    if /i "!USER_CHOICE!"=="F" set "INSTALL_MODE=fresh"
    if /i "!USER_CHOICE!"=="C" (
        echo.
        echo Cancelled by user. No changes were made.
        echo.
        pause
        exit /b 0
    )
    if not defined INSTALL_MODE (
        echo.
        echo ERROR: Invalid choice "!USER_CHOICE!". Please type U, F, or C.
        pause
        exit /b 1
    )
) else (
    if exist "C:\ComfyUI_windows_portable" (
        echo.
        echo NOTE: C:\ComfyUI_windows_portable\ exists but does not look
        echo like a valid ComfyUI install ^(missing ComfyUI\main.py^).
        echo Proceeding in Fresh mode -- will back up that folder before
        echo installing.
    ) else (
        echo   No existing install found.
    )
    set "INSTALL_MODE=fresh"
)
echo   Install mode: !INSTALL_MODE!

echo.
echo All checks passed.
echo.

REM ==========================================================
REM Steps 1-4 / 12 - Install prerequisites via winget
REM ==========================================================
call :winget_install "7zip.7zip"            "7-Zip"       "1/12"
call :winget_install "Git.Git"              "Git"         "2/12"
call :winget_install "Gyan.FFmpeg"          "FFmpeg"      "3/12"
call :winget_install "Python.Python.3.13"   "Python 3.13" "4/12"

REM ==========================================================
REM Refresh PATH from registry so newly-installed tools become
REM visible to this shell without reopening the terminal
REM ==========================================================
echo.
echo Refreshing PATH from registry...
for /f "delims=" %%i in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')"') do set "PATH=%%i"

REM ==========================================================
REM Step 5/12 - Validate Python resolution (must NOT be embedded)
REM ==========================================================
echo.
echo [Step  5/12] Validating that 'python' resolves to global Python 3.13...
where python > "%TEMP%\python_path.txt" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: 'python' command not found after winget install.
    echo Try closing this window, opening a new cmd, and re-running.
    echo.
    pause
    exit /b 1
)
findstr /i "python_embeded" "%TEMP%\python_path.txt" >nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ERROR: 'python' resolves to a ComfyUI embedded Python:
    type "%TEMP%\python_path.txt"
    echo.
    echo Remove 'python_embeded' from your PATH variable, then
    echo re-run this script. Embedded Python is not meant to be
    echo on PATH.
    echo.
    pause
    exit /b 1
)
del "%TEMP%\python_path.txt" >nul 2>&1
echo   Python resolution OK.

REM ==========================================================
REM Phase A: backup (Fresh mode only, when folder exists)
REM ==========================================================
REM Note: this block uses !TIMESTAMP! (delayed expansion) and
REM !ERRORLEVEL! for variables set inside the block.
set "TIMESTAMP="
if /i "%INSTALL_MODE%"=="fresh" (
    if exist "C:\ComfyUI_windows_portable" (
        for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd-HHmmss"') do set "TIMESTAMP=%%i"
        if not defined TIMESTAMP (
            echo.
            echo ERROR: Could not generate timestamp via PowerShell.
            echo Refusing to continue without a unique backup name
            echo ^(would risk overwriting an earlier backup^).
            pause
            exit /b 1
        )

        echo.
        echo ========================================================
        echo  PHASE A - Backup existing install
        echo ========================================================
        echo.

        echo Stopping python.exe processes inside C:\ComfyUI_windows_portable\ ...
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -like 'C:\ComfyUI_windows_portable*' } | ForEach-Object { Write-Host ('  Stopping PID {0}: {1}' -f $_.Id, $_.Path); Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"

        echo Renaming C:\ComfyUI_windows_portable\ to .OLD-!TIMESTAMP!\ ...
        ren "C:\ComfyUI_windows_portable" "ComfyUI_windows_portable.OLD-!TIMESTAMP!"
        if !ERRORLEVEL! NEQ 0 (
            echo.
            echo ERROR: Could not rename the folder.
            echo Some files may still be locked. Close any program using
            echo files inside C:\ComfyUI_windows_portable\ and try again.
            pause
            exit /b 1
        )
        echo Done. Backup created at: C:\ComfyUI_windows_portable.OLD-!TIMESTAMP!\
        echo.
        timeout /t 3 /nobreak >nul
    )
)

REM ==========================================================
REM Locate 7z.exe (winget does NOT add it to PATH)
REM Only required if we will extract (Fresh mode).
REM ==========================================================
set "SEVENZIP="
if exist "%ProgramFiles%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles%\7-Zip\7z.exe"
if not defined SEVENZIP if exist "%ProgramFiles(x86)%\7-Zip\7z.exe" set "SEVENZIP=%ProgramFiles(x86)%\7-Zip\7z.exe"
if not defined SEVENZIP if exist "%LocalAppData%\Programs\7-Zip\7z.exe" set "SEVENZIP=%LocalAppData%\Programs\7-Zip\7z.exe"
if /i "%INSTALL_MODE%"=="fresh" (
    if not defined SEVENZIP (
        echo.
        echo ERROR: 7z.exe not found after winget install.
        echo Expected at: "%ProgramFiles%\7-Zip\7z.exe"
        echo.
        pause
        exit /b 1
    )
    echo   Found 7z at: %SEVENZIP%
)

REM ==========================================================
REM Steps 6-7 / 12 - Download + Extract (Fresh mode only)
REM ==========================================================
set "COMFY_URL=https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
set "COMFY_ARCHIVE=%TEMP%\ComfyUI_windows_portable_nvidia.7z"

if /i "%INSTALL_MODE%"=="update" goto :skip_download

echo.
echo [Step  6/12] Downloading ComfyUI Portable (~3 GB)...
echo   URL: %COMFY_URL%
echo   This may take several minutes depending on your connection.
echo   curl will retry up to 3 times with 5s delay on failure.
echo.
curl.exe -L --fail --progress-bar --retry 3 --retry-delay 5 -o "%COMFY_ARCHIVE%" "%COMFY_URL%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Download failed after retries.
    echo Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

REM ----- Sanity check: archive must be at least ~1 GB -----
REM We compare in MB (not bytes) because cmd treats numbers as
REM signed int32 (max ~2.14 billion); 3 GB = 3,221,225,472 bytes
REM would overflow and break the LSS comparison.
REM Note: the for /f below is a single long line on purpose --
REM line-continuation with ^ is unreliable inside for /f ('...').
set "ARCHIVE_MB="
for /f "delims=" %%i in ('powershell -NoProfile -Command "[int]((Get-Item '%COMFY_ARCHIVE%').Length / 1MB)"') do set "ARCHIVE_MB=%%i"
if not defined ARCHIVE_MB (
    echo.
    echo ERROR: Could not determine archive size after download.
    echo.
    pause
    exit /b 1
)
if %ARCHIVE_MB% LSS 1000 (
    echo.
    echo ERROR: Downloaded archive is only %ARCHIVE_MB% MB, expected ~3000 MB.
    echo Download may have been truncated or corrupted.
    del "%COMFY_ARCHIVE%" >nul 2>&1
    echo.
    pause
    exit /b 1
)
echo   Download complete. Archive size: %ARCHIVE_MB% MB.

echo.
echo [Step  7/12] Extracting ComfyUI Portable to C:\ ...
"%SEVENZIP%" x "%COMFY_ARCHIVE%" -oC:\ -y
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Extraction failed.
    echo Archive preserved at: %COMFY_ARCHIVE%
    echo.
    pause
    exit /b 1
)
if not exist "C:\ComfyUI_windows_portable\run_nvidia_gpu.bat" (
    echo.
    echo ERROR: Extraction completed but C:\ComfyUI_windows_portable\run_nvidia_gpu.bat
    echo was not found. The archive may be corrupt or have a different layout.
    echo Archive preserved at: %COMFY_ARCHIVE%
    echo.
    pause
    exit /b 1
)
echo   Extraction OK.

REM Note: archive is preserved at %COMFY_ARCHIVE% until the very end of
REM the script, so any failure between here and the final cleanup lets
REM you retry without re-downloading 3 GB.
goto :after_download

:skip_download
echo.
echo [Steps 6-7/12] Skipped (Update mode -- ComfyUI already exists).

:after_download

REM ==========================================================
REM From here on, work inside C:\ComfyUI_windows_portable\
REM ==========================================================
cd /d C:\ComfyUI_windows_portable

REM ==========================================================
REM Step 8/12 - Pre-create model folders (idempotent)
REM ==========================================================
echo.
echo [Step  8/12] Pre-creating model folders...
if not exist "ComfyUI\custom_nodes"             mkdir "ComfyUI\custom_nodes"
if not exist "ComfyUI\models\text_encoders"     mkdir "ComfyUI\models\text_encoders"
if not exist "ComfyUI\models\diffusion_models"  mkdir "ComfyUI\models\diffusion_models"
if not exist "ComfyUI\models\vae"               mkdir "ComfyUI\models\vae"
if not exist "ComfyUI\models\loras"             mkdir "ComfyUI\models\loras"
if not exist "ComfyUI\models\checkpoints"       mkdir "ComfyUI\models\checkpoints"
if not exist "ComfyUI\models\clip_vision"       mkdir "ComfyUI\models\clip_vision"
if not exist "ComfyUI\models\controlnet"        mkdir "ComfyUI\models\controlnet"
if not exist "ComfyUI\models\upscale_models"    mkdir "ComfyUI\models\upscale_models"
echo   Done.

REM ==========================================================
REM Step 9/12 - First launch (Fresh mode only)
REM ==========================================================
if /i "%INSTALL_MODE%"=="update" goto :skip_first_launch

echo.
echo ========================================================
echo  [Step  9/12] FIRST LAUNCH - MANUAL VALIDATION
echo ========================================================
echo.
echo  ComfyUI will start in a NEW WINDOW in 5 seconds.
echo.
echo  WHAT TO DO:
echo    1. Wait 30-90 seconds for first-time setup.
echo    2. Your browser opens automatically with the workflow editor.
echo    3. CLOSE the browser tab.
echo    4. CLOSE the ComfyUI cmd window.
echo    5. RETURN to THIS window and press any key.
echo.
echo  IMPORTANT: do NOT press any key here until ComfyUI has
echo  started successfully and you have closed both the browser
echo  tab and the ComfyUI cmd window. Otherwise pip install
echo  steps will fail because python.exe is still locking files.
echo.
echo  If ComfyUI does not open after 2 minutes, press Ctrl+C
echo  to abort and check NVIDIA drivers.
echo.
echo ========================================================
echo.
timeout /t 5 /nobreak >nul

start "ComfyUI First Launch" "C:\ComfyUI_windows_portable\run_nvidia_gpu.bat"

echo.
echo ComfyUI is now starting in a new window.
echo Loading takes 30-90 seconds on first launch...
echo.
timeout /t 10 /nobreak >nul

echo Press any key HERE only AFTER you have:
echo   - Seen the ComfyUI workflow editor in your browser
echo   - Closed the browser tab
echo   - Closed the ComfyUI cmd window
echo.
pause

REM ----- Belt and suspenders: kill any python.exe still running -----
echo.
echo Stopping any python.exe still running inside ComfyUI_windows_portable...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object { $_.Path -like 'C:\ComfyUI_windows_portable*' } | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"
REM Give Windows 2s to release file handles before pip starts
timeout /t 2 /nobreak >nul
goto :after_first_launch

:skip_first_launch
echo.
echo [Step  9/12] Skipped (Update mode -- ComfyUI already initialized).

:after_first_launch

REM ==========================================================
REM Step 10/12 - Upgrade pip + install/update essential custom nodes
REM ==========================================================
echo.
echo [Step 10/12] Upgrading pip in embedded Python...
python_embeded\python.exe -s -m pip install -U pip setuptools wheel

echo.
echo [Step 10/12] Installing/updating essential custom nodes...
cd /d C:\ComfyUI_windows_portable\ComfyUI\custom_nodes

call :ensure_node "ComfyUI-Manager"   "https://github.com/ltdrdata/ComfyUI-Manager.git"
call :ensure_node "rgthree-comfy"     "https://github.com/rgthree/rgthree-comfy.git"
call :ensure_node "ComfyUI-Crystools" "https://github.com/crystian/ComfyUI-Crystools.git"

cd /d C:\ComfyUI_windows_portable

REM ==========================================================
REM Step 11/12 - Install/update custom node Python requirements
REM ==========================================================
echo.
echo [Step 11/12] Installing custom node requirements...

if exist "ComfyUI\custom_nodes\ComfyUI-Manager\requirements.txt" ^
    python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\ComfyUI-Manager\requirements.txt

if exist "ComfyUI\custom_nodes\rgthree-comfy\requirements.txt" ^
    python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\rgthree-comfy\requirements.txt

if exist "ComfyUI\custom_nodes\ComfyUI-Crystools\requirements.txt" ^
    python_embeded\python.exe -s -m pip install -r ComfyUI\custom_nodes\ComfyUI-Crystools\requirements.txt

REM ==========================================================
REM Step 12/12 - Install requests + pillow in global Python 3.13
REM ==========================================================
echo.
echo [Step 12/12] Installing requests + pillow in global Python 3.13...
python -m pip install --upgrade requests pillow
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not install requests/pillow into global Python.
    echo You can run this manually later:
    echo   python -m pip install requests pillow
)

REM ==========================================================
REM Final cleanup: remove archive only after everything succeeded
REM ==========================================================
if exist "%COMFY_ARCHIVE%" del "%COMFY_ARCHIVE%" >nul 2>&1

REM ==========================================================
REM Manual configuration step (Node ID)
REM ==========================================================
echo.
echo ========================================================
echo  MANUAL STEP - Enable Node ID Badge
echo ========================================================
echo.
echo  Comfy Workflow channel workflows rely on stable node IDs.
echo  Enable "Node ID Badge" in your ComfyUI settings:
echo.
echo    1. Run C:\ComfyUI_windows_portable\run_nvidia_gpu.bat
echo    2. Once ComfyUI loads in your browser, click the gear
echo       icon (Settings) on the top-right.
echo    3. Search for "Node ID Badge Mode" (or just "Node ID").
echo    4. Set it to "Show All".
echo    5. Close settings. Done.
echo.
echo  This is a one-time step. Settings persist across launches.
echo.
echo ========================================================

REM ==========================================================
REM Summary (mode-dependent)
REM ==========================================================
echo.
echo ========================================================
if /i "%INSTALL_MODE%"=="update" (
    echo  UPDATE COMPLETE
) else (
    echo  INSTALLATION COMPLETE
)
echo ========================================================
echo.
if /i "%INSTALL_MODE%"=="update" (
    echo  Updated:
    echo    - 3 essential custom nodes ^(Manager, rgthree, Crystools^)
    echo    - Custom node Python requirements
    echo    - Global Python packages: requests, pillow
    echo.
    echo  Preserved:
    echo    - Your existing ComfyUI install at C:\ComfyUI_windows_portable\
    echo    - All your models, workflows, and other custom nodes
) else (
    echo  Installed:
    echo    - 7-Zip, Git, FFmpeg, Python 3.13 ^(winget^)
    echo    - ComfyUI Portable ^(Python 3.13 + CUDA 13.0^) at:
    echo        C:\ComfyUI_windows_portable\
    echo    - Custom nodes: ComfyUI-Manager, rgthree-comfy, ComfyUI-Crystools
    echo    - Global Python packages: requests, pillow
    if defined TIMESTAMP (
        echo.
        echo  Old install was backed up to:
        echo    C:\ComfyUI_windows_portable.OLD-!TIMESTAMP!\
    )
)
echo.
echo  Next steps:
echo    1. Run C:\ComfyUI_windows_portable\run_nvidia_gpu.bat
echo    2. Enable "Node ID Badge Mode" in settings ^(see above^)
echo    3. Watch the channel for workflows!
echo.
echo ========================================================
echo.
pause
endlocal
exit /b 0