from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `from src...` imports when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import DEFAULT_RAW_CSV, DEFAULT_UNIFIED_CSV
from src.common.logging_setup import setup_logger
from src.ingestion.normalize import normalize_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.2: Normalize raw multi-cloud logs into unified telemetry schema.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_RAW_CSV), help="Path to raw CSV.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_UNIFIED_CSV), help="Path to output unified CSV.")
    args = parser.parse_args()

    logger = setup_logger("normalize")
    out_path, n = normalize_csv(args.input, args.output)
    logger.info(f"✅ Wrote unified telemetry CSV: {out_path} (rows={n})")


if __name__ == "__main__":
    main()
