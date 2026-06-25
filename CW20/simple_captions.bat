@echo off
setlocal
title CW20 - simple captions

REM ============================================================
REM  Creates a .txt with a SIMPLE caption for every image in THIS folder.
REM  Caption = your trigger word + your subject class, the same on all.
REM  Put this .bat INSIDE your dataset folder and run it.
REM  Press Enter on both questions to use the demo character (Juno).
REM ============================================================

set "TRIGGER=cmfychar"
set "CLASS=1girl, solo"

echo.
echo Leave blank and press Enter to keep the value in [brackets].
set /p "TRIGGER=Your trigger word (a made-up word) [%TRIGGER%]: "
set /p "CLASS=What it is - 1girl / 1boy / robot / dog / a sword... [%CLASS%]: "
set "CAP=%TRIGGER%, %CLASS%"

set /a N=0
for %%F in (*.png *.jpg *.jpeg) do (
  >"%%~nF.txt" echo %CAP%
  set /a N+=1
)

echo.
echo Done: %N% caption file(s) created with "%CAP%".
pause
endlocal
