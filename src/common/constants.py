from __future__ import annotations

from pathlib import Path

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMAS_DIR = DATA_DIR / "schemas"

DEFAULT_RAW_CSV = RAW_DIR / "afozt_multicloud_logs_50k.csv"
DEFAULT_UNIFIED_CSV = PROCESSED_DIR / "unified_telemetry.csv"
