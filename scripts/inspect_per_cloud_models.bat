\
@echo off
REM inspect_per_cloud_models.bat
REM Run from project root: D:\Research_AFO-ZT_MultiCloud
REM Generates human-readable inspection outputs for baseline_per_cloud*.pkl

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Please create/activate your venv first.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python scripts\inspect_pickles.py
echo.
echo Done. Open outputs\models\baseline_per_cloud*_preview.json OR *_globals.json and *_pickle_ops_head.txt in VS Code.
