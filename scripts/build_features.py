from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import DEFAULT_FEATURES_CSV, DEFAULT_FEATURES_PARQUET, DEFAULT_TRUST_GRAPH_DB, DEFAULT_UNIFIED_CSV
from src.features.feature_engineering import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.4: Build tabular + graph features.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_UNIFIED_CSV), help="Path to unified telemetry CSV.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_TRUST_GRAPH_DB), help="Path to trust_graph.sqlite")
    parser.add_argument("--out-csv", type=str, default=str(DEFAULT_FEATURES_CSV), help="Output features CSV")
    parser.add_argument("--out-parquet", type=str, default=str(DEFAULT_FEATURES_PARQUET), help="Output features parquet")
    parser.add_argument("--window-minutes", type=int, default=30, help="Window size for rolling features")
    parser.add_argument("--blast-k", type=int, default=3, help="Blast radius hop limit K")
    parser.add_argument("--max-hops-sensitive", type=int, default=6, help="Max hops for shortest path to sensitive")
    args = parser.parse_args()

    build_features(
        unified_csv=Path(args.input),
        trust_graph_db=Path(args.db),
        out_csv=Path(args.out_csv),
        out_parquet=Path(args.out_parquet),
        window_minutes=int(args.window_minutes),
        blast_k=int(args.blast_k),
        max_hops_sensitive=int(args.max_hops_sensitive),
        include_graph_static=True,
    )


if __name__ == "__main__":
    main()
