@echo off
setlocal
title CW20 - simple captions

REM ============================================================
REM  Creates a .txt with the SIMPLE caption for every image here.
REM  Caption = trigger + class, the same on all (the default method).
REM  Put this .bat INSIDE your dataset folder and run it.
REM ============================================================

set "CAP=cmfychar, 1girl, solo"

set /a N=0
for %%F in (*.png *.jpg *.jpeg) do (
  >"%%~nF.txt" echo %CAP%
  set /a N+=1
)

echo.
echo Done: %N% caption file(s) created with "%CAP%".
echo (For your own character, change "1girl" to its class: 1boy, robot, dog, sword...)
pause
endlocal
