@echo off
REM plot_final_metrics.bat
REM Run from project root.
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create/activate venv first.
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\plot_final_metrics.py
