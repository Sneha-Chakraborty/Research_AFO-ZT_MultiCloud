from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `from src...` imports when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import DEFAULT_TRUST_GRAPH_DB, DEFAULT_UNIFIED_CSV
from src.common.logging_setup import setup_logger
from src.trust_graph.builder_sqlite import build_or_update_trust_graph_sqlite


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.3: Build/update the Cross-Cloud Trust Graph (SQLite).")
    parser.add_argument("--input", type=str, default=str(DEFAULT_UNIFIED_CSV), help="Path to unified telemetry CSV.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_TRUST_GRAPH_DB), help="Path to SQLite trust graph DB.")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV chunksize for stable memory usage.")
    parser.add_argument("--full-rebuild", action="store_true", help="Rebuild from scratch (disables incremental filter).")
    parser.add_argument("--reset-db", action="store_true", help="Delete DB file and rebuild.")
    args = parser.parse_args()

    logger = setup_logger("build_trust_graph")
    res = build_or_update_trust_graph_sqlite(
        unified_csv=Path(args.input),
        db_path=Path(args.db),
        chunksize=int(args.chunksize),
        incremental=not args.full_rebuild,
        reset_db=bool(args.reset_db),
    )
    logger.info(f"Done. DB at: {res.db_path}")


if __name__ == "__main__":
    main()
