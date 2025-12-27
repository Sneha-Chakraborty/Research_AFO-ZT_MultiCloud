from __future__ import annotations

from pathlib import Path

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMAS_DIR = DATA_DIR / "schemas"

DEFAULT_RAW_CSV = RAW_DIR / "afozt_multicloud_logs_50k.csv"
DEFAULT_UNIFIED_CSV = PROCESSED_DIR / "unified_telemetry.csv"

# Outputs
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

DEFAULT_TRUST_GRAPH_DB = MODELS_DIR / "trust_graph.sqlite"
