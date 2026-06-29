@echo off
setlocal
title CW20 - simple captions

REM ============================================================
REM  Writes the SIMPLE caption (trigger + class, same on all)
REM  to a .txt next to every image in THIS folder.
REM  Beginner default. Put the .bat in your dataset folder, run it.
REM ============================================================

set "TRIGGER=cmfychar"
set "CLASS=1girl, solo"

echo.
echo This writes the SAME caption to every image here (the simple default).
echo Press Enter to keep the value in [brackets], or type your own.
echo.
set /p "TRIGGER=Your trigger word (a made-up word) [%TRIGGER%]: "
set /p "CLASS=What it is - 1girl / 1boy / robot / dog / a sword... [%CLASS%]: "
set "CAP=%TRIGGER%, %CLASS%"

echo.
echo Every image will be captioned:  %CAP%
set /p "OK=Correct? [Y/n]: "
if /i "%OK%"=="n" goto :cancelled

if not exist "*.txt" goto :write
echo.
echo NOTE: some .txt caption files already exist in this folder.
set /p "OW=Overwrite ALL of them with the simple caption? [y/N]: "
if /i not "%OW%"=="y" goto :cancelled

:write
set /a N=0
for %%F in (*.png *.jpg *.jpeg) do (
  >"%%~nF.txt" echo %CAP%
  set /a N+=1
)

echo.
echo Done: %N% caption file(s) written with "%CAP%".
echo.
echo These are SIMPLE captions - the same line on every image. It is the
echo beginner default and works for one character. For more control (freely
echo change outfits/scenes), edit each .txt to describe what VARIES, or use
echo OneTrainer's built-in WD14 tagger. The video shows both.
echo.
pause
endlocal
goto :eof

:cancelled
echo.
echo Cancelled - nothing was written.
echo.
pause
endlocal