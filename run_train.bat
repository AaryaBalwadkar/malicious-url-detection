@echo off
REM Double-click executable launcher: full training pipeline
REM Uses local .venv if present, else system python
title Malicious URL Detection - Train
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  echo Using .venv Python...
  .venv\Scripts\python.exe src\train.py
) else (
  echo .venv not found, using system python...
  python src\train.py
)
echo.
echo Done. Results in results\ , figures\ , models\
pause
