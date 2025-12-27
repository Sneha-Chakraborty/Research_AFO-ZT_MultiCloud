from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.common.logging_setup import setup_logger
from src.trust_graph.mapper import build_graph_items_from_row
from src.trust_graph.sqlite_store import TrustGraphSQLite


@dataclass
class BuildResult:
    rows_seen: int
    rows_ingested: int
    max_ts_ingested: str
    db_path: Path


def _valid_ts(s: str) -> bool:
    return bool(s) and s != "NaT"


def build_or_update_trust_graph_sqlite(
    unified_csv: Path,
    db_path: Path,
    *,
    chunksize: int = 50_000,
    incremental: bool = True,
    reset_db: bool = False,
) -> BuildResult:
    """Build or incrementally update the Cross-Cloud Trust Graph (Step 2.3).

    - Reads Step 2.2 output: unified telemetry CSV
    - Writes SQLite graph at outputs/models/trust_graph.sqlite (default)
    - If incremental=True: only ingest rows with ts > last_ingested_ts stored in DB meta
    """
    logger = setup_logger("trust_graph")

    unified_csv = Path(unified_csv)
    db_path = Path(db_path)

    if reset_db and db_path.exists():
        logger.warning(f"Resetting trust graph DB: {db_path}")
        db_path.unlink()

    store = TrustGraphSQLite(db_path=db_path, read_only=False)
    try:
        last_ts = store.get_meta("last_ingested_ts", default="")
        if not incremental:
            last_ts = ""

        rows_seen = 0
        rows_ingested = 0
        max_ts = last_ts or ""

        # Chunked read keeps memory stable and is "research-grade" reproducible.
        for chunk in pd.read_csv(unified_csv, chunksize=chunksize):
            rows_seen += len(chunk)
            # Ensure ts column exists (Step 2.2 guarantees it)
            if "ts" not in chunk.columns:
                raise ValueError("Unified telemetry CSV missing 'ts' column. Run Step 2.2 first.")

            if incremental and last_ts:
                # 'YYYY-MM-DD HH:MM:SS' compares lexicographically
                chunk = chunk[chunk["ts"].astype(str) > str(last_ts)]
                if chunk.empty:
                    continue

            # Convert NaNs for stable row access; itertuples is fast
            nodes_batch = []
            edges_batch = []

            for row in chunk.itertuples(index=False):
                nodes, edges = build_graph_items_from_row(row)
                nodes_batch.extend(nodes)
                edges_batch.extend(edges)

                # update max_ts
                ts = getattr(row, "ts", "")
                if _valid_ts(str(ts)) and str(ts) > str(max_ts):
                    max_ts = str(ts)

            if edges_batch or nodes_batch:
                store.ingest(nodes_batch, edges_batch)
                rows_ingested += len(chunk)

        if incremental and max_ts:
            store.set_meta("last_ingested_ts", max_ts)

        stats = store.get_stats()
        logger.info(
            f"✅ Trust graph updated: db={db_path} | rows_seen={rows_seen} rows_ingested={rows_ingested} "
            f"| nodes={stats['nodes']} edges_agg={stats['edges_agg']} edge_events={stats['edge_events']} "
            f"| last_ingested_ts={max_ts or '(none)'}"
        )

        return BuildResult(rows_seen=rows_seen, rows_ingested=rows_ingested, max_ts_ingested=max_ts, db_path=db_path)
    finally:
        store.close()
