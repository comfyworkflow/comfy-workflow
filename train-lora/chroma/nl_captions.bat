@echo off
setlocal
title Chroma captions - guided helper

REM ============================================================
REM  Guided natural-language captions for Chroma LoRA training.
REM  Put this .bat (and nl_captions.py) INSIDE your dataset
REM  folder and double-click it. It asks a couple of questions
REM  per photo and writes one rich caption .txt per image.
REM ============================================================

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.10-3.13 and tick "Add Python to PATH".
  pause & exit /b 1
)
python "%~dp0nl_captions.py"
echo.
pause
endlocal
