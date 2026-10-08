@echo off
REM Double-click executable launcher: demo prediction with best model
REM Usage: run_demo.bat [ModelName] [RowNumber]
REM Example: run_demo.bat XGBoost 0
title Malicious URL Detection - Demo Predict
cd /d "%~dp0"
set MODEL=%1
set ROW=%2
if "%MODEL%"=="" set MODEL=XGBoost
if "%ROW%"=="" set ROW=0
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe src\demo_predict.py --model %MODEL% --row %ROW%
) else (
  python src\demo_predict.py --model %MODEL% --row %ROW%
)
echo.
pause
