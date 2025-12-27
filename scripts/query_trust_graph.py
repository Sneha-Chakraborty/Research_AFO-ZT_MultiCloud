from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import DEFAULT_TRUST_GRAPH_DB
from src.trust_graph.sqlite_store import TrustGraphSQLite
from src.trust_graph.queries import find_cross_cloud_pivots, top_risky_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick queries over the SQLite trust graph (Step 2.3).")
    parser.add_argument("--db", type=str, default=str(DEFAULT_TRUST_GRAPH_DB), help="Path to trust_graph.sqlite")
    parser.add_argument("--principal", type=str, default="", help="principal_id to summarize cross-cloud pivots")
    parser.add_argument("--top-edges", type=int, default=0, help="Print top-N risky edges")
    args = parser.parse_args()

    store = TrustGraphSQLite(Path(args.db), read_only=True)
    try:
        print("DB stats:", store.get_stats())

        if args.principal:
            piv = find_cross_cloud_pivots(store, args.principal)
            print("\nCross-cloud pivot summary:")
            print(piv)

        if args.top_edges and args.top_edges > 0:
            rows = top_risky_edges(store, limit=int(args.top_edges))
            print("\nTop risky edges:")
            for r in rows:
                print(r["cloud_provider"], r["edge_type"], r["src_id"], "->", r["dst_id"], "deny=", r["deny_count"], "attack=", r["attack_count"], "bytes_sum=", r["bytes_out_sum"])
    finally:
        store.close()


if __name__ == "__main__":
    main()
