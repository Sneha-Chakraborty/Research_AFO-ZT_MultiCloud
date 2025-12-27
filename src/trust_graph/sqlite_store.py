from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class NodeUpsert:
    node_id: str
    node_type: str
    ts: str
    attrs: Dict[str, Any]


@dataclass(frozen=True)
class EdgeEvent:
    ts: str
    cloud_provider: str
    tenant_id: str
    region: str
    edge_type: str
    src_id: str
    dst_id: str
    sensitivity: str
    bytes_out: float
    access_result: str
    action: str
    api: str
    operation: str
    label_attack: int
    attack_type: str


class TrustGraphSQLite:
    """SQLite-backed trust graph store.

    Design goals:
    - Research-grade reproducibility: schema is explicit, deterministic node/edge ids.
    - Scalable on laptops: append-only edge_events + aggregated edges_agg.
    - Incremental updates: meta table stores last_ingested_ts.
    """

    def __init__(self, db_path: Path, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.conn = self._connect()
        self._apply_pragmas()
        self._create_schema()

    # -------------------------
    # Connection / schema
    # -------------------------
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # If a previous run left WAL/SHM files owned by another user, SQLite may behave as read-only.
        # Best-effort cleanup avoids confusing "readonly database" errors.
        if not self.read_only:
            for suf in ("-wal", "-shm"):
                side = Path(str(self.db_path) + suf)
                try:
                    if side.exists() and not os.access(side, os.W_OK):
                        side.unlink()
                except Exception:
                    pass


        if self.read_only:
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(self.db_path)

        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _apply_pragmas(self) -> None:
        # Pragmas tuned for "write-many, read-many" on a single user machine.
        # WAL improves concurrent read while ingesting; NORMAL is a safe speed/robustness tradeoff.
        cur = self.conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            # If WAL is not available (e.g., restrictive FS), fall back gracefully.
            cur.execute("PRAGMA journal_mode=DELETE;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA cache_size=-200000;")  # ~200k pages in memory (negative => KiB)
        cur.execute("PRAGMA foreign_keys=ON;")
        self.conn.commit()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()

        # Meta
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            """
        )

        # Nodes (deduplicated)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
              node_id TEXT PRIMARY KEY,
              node_type TEXT NOT NULL,
              first_seen TEXT,
              last_seen TEXT,
              attrs_json TEXT
            );
            """
        )

        # Edge events (append-only, "each edge stores timestamp")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edge_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT NOT NULL,
              cloud_provider TEXT NOT NULL,
              tenant_id TEXT,
              region TEXT,
              edge_type TEXT NOT NULL,
              src_id TEXT NOT NULL,
              dst_id TEXT NOT NULL,
              sensitivity TEXT,
              bytes_out REAL,
              access_result TEXT,
              action TEXT,
              api TEXT,
              operation TEXT,
              label_attack INTEGER,
              attack_type TEXT
            );
            """
        )

        # Aggregated edges for fast queries
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges_agg (
              edge_key TEXT PRIMARY KEY,
              src_id TEXT NOT NULL,
              dst_id TEXT NOT NULL,
              edge_type TEXT NOT NULL,
              cloud_provider TEXT NOT NULL,
              first_seen TEXT,
              last_seen TEXT,
              event_count INTEGER NOT NULL DEFAULT 0,
              permit_count INTEGER NOT NULL DEFAULT 0,
              deny_count INTEGER NOT NULL DEFAULT 0,
              restrict_count INTEGER NOT NULL DEFAULT 0,
              attack_count INTEGER NOT NULL DEFAULT 0,
              bytes_out_sum REAL NOT NULL DEFAULT 0.0,
              attrs_json TEXT
            );
            """
        )

        # Indexes (critical for "research grade" query speed)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edge_events_ts ON edge_events(ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edge_events_src ON edge_events(src_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edge_events_dst ON edge_events(dst_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edge_events_type ON edge_events(edge_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_agg_src ON edges_agg(src_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_agg_dst ON edges_agg(dst_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_agg_type ON edges_agg(edge_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_agg_last_seen ON edges_agg(last_seen);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_agg_cloud ON edges_agg(cloud_provider);")

        self.conn.commit()

    # -------------------------
    # Meta helpers
    # -------------------------
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self.conn.cursor()
        row = cur.execute("SELECT value FROM meta WHERE key = ?;", (key,)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        if self.read_only:
            raise RuntimeError("DB is read-only; cannot set meta.")
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value;
            """,
            (key, value),
        )
        self.conn.commit()

    # -------------------------
    # Upserts / ingest
    # -------------------------
    @staticmethod
    def _edge_key(src_id: str, dst_id: str, edge_type: str, cloud_provider: str) -> str:
        # Deterministic aggregation key
        return f"{src_id}::{edge_type}::{dst_id}::{cloud_provider}"

    def upsert_nodes(self, nodes: Sequence[NodeUpsert]) -> None:
        if self.read_only:
            raise RuntimeError("DB is read-only; cannot upsert nodes.")
        if not nodes:
            return

        rows = [
            (
                n.node_id,
                n.node_type,
                n.ts,
                n.ts,
                json.dumps(n.attrs, ensure_ascii=False, separators=(",", ":")),
            )
            for n in nodes
        ]

        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO nodes(node_id, node_type, first_seen, last_seen, attrs_json)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              last_seen = CASE WHEN excluded.last_seen > nodes.last_seen THEN excluded.last_seen ELSE nodes.last_seen END,
              attrs_json = excluded.attrs_json;
            """,
            rows,
        )

    def insert_edge_events(self, events: Sequence[EdgeEvent]) -> None:
        if self.read_only:
            raise RuntimeError("DB is read-only; cannot insert edge events.")
        if not events:
            return

        rows = [
            (
                e.ts,
                e.cloud_provider,
                e.tenant_id,
                e.region,
                e.edge_type,
                e.src_id,
                e.dst_id,
                e.sensitivity,
                float(e.bytes_out) if e.bytes_out is not None else 0.0,
                e.access_result,
                e.action,
                e.api,
                e.operation,
                int(e.label_attack) if e.label_attack is not None else 0,
                e.attack_type,
            )
            for e in events
        ]

        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO edge_events(
              ts, cloud_provider, tenant_id, region, edge_type, src_id, dst_id,
              sensitivity, bytes_out, access_result, action, api, operation,
              label_attack, attack_type
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            rows,
        )

    def upsert_edges_agg(self, events: Sequence[EdgeEvent]) -> None:
        """Update aggregated edges based on event batch."""
        if self.read_only:
            raise RuntimeError("DB is read-only; cannot upsert edges.")
        if not events:
            return

        # Aggregate in Python first to reduce SQL churn
        agg: Dict[str, Dict[str, Any]] = {}
        for e in events:
            ek = self._edge_key(e.src_id, e.dst_id, e.edge_type, e.cloud_provider)
            if ek not in agg:
                agg[ek] = {
                    "edge_key": ek,
                    "src_id": e.src_id,
                    "dst_id": e.dst_id,
                    "edge_type": e.edge_type,
                    "cloud_provider": e.cloud_provider,
                    "first_seen": e.ts,
                    "last_seen": e.ts,
                    "event_count": 0,
                    "permit_count": 0,
                    "deny_count": 0,
                    "restrict_count": 0,
                    "attack_count": 0,
                    "bytes_out_sum": 0.0,
                    # store last-seen attributes (compact)
                    "attrs": {
                        "tenant_id": e.tenant_id,
                        "region": e.region,
                        "sensitivity": e.sensitivity,
                        "action": e.action,
                        "api": e.api,
                        "operation": e.operation,
                    },
                }

            a = agg[ek]
            a["event_count"] += 1
            a["bytes_out_sum"] += float(e.bytes_out) if e.bytes_out is not None else 0.0
            if (e.access_result or "").lower() == "permit":
                a["permit_count"] += 1
            elif (e.access_result or "").lower() == "deny":
                a["deny_count"] += 1
            elif (e.access_result or "").lower() == "restrict":
                a["restrict_count"] += 1

            if int(e.label_attack or 0) == 1:
                a["attack_count"] += 1

            if e.ts < a["first_seen"]:
                a["first_seen"] = e.ts
            if e.ts > a["last_seen"]:
                a["last_seen"] = e.ts
                a["attrs"] = {
                    "tenant_id": e.tenant_id,
                    "region": e.region,
                    "sensitivity": e.sensitivity,
                    "action": e.action,
                    "api": e.api,
                    "operation": e.operation,
                }

        rows = []
        for ek, a in agg.items():
            rows.append(
                (
                    a["edge_key"],
                    a["src_id"],
                    a["dst_id"],
                    a["edge_type"],
                    a["cloud_provider"],
                    a["first_seen"],
                    a["last_seen"],
                    a["event_count"],
                    a["permit_count"],
                    a["deny_count"],
                    a["restrict_count"],
                    a["attack_count"],
                    a["bytes_out_sum"],
                    json.dumps(a["attrs"], ensure_ascii=False, separators=(",", ":")),
                )
            )

        cur = self.conn.cursor()
        cur.executemany(
            """
            INSERT INTO edges_agg(
              edge_key, src_id, dst_id, edge_type, cloud_provider,
              first_seen, last_seen, event_count,
              permit_count, deny_count, restrict_count, attack_count,
              bytes_out_sum, attrs_json
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(edge_key) DO UPDATE SET
              last_seen = CASE WHEN excluded.last_seen > edges_agg.last_seen THEN excluded.last_seen ELSE edges_agg.last_seen END,
              first_seen = CASE WHEN excluded.first_seen < edges_agg.first_seen THEN excluded.first_seen ELSE edges_agg.first_seen END,
              event_count = edges_agg.event_count + excluded.event_count,
              permit_count = edges_agg.permit_count + excluded.permit_count,
              deny_count = edges_agg.deny_count + excluded.deny_count,
              restrict_count = edges_agg.restrict_count + excluded.restrict_count,
              attack_count = edges_agg.attack_count + excluded.attack_count,
              bytes_out_sum = edges_agg.bytes_out_sum + excluded.bytes_out_sum,
              attrs_json = excluded.attrs_json;
            """,
            rows,
        )

    def ingest(self, nodes: Sequence[NodeUpsert], edge_events: Sequence[EdgeEvent]) -> None:
        """Ingest nodes + edge events as one transaction."""
        if self.read_only:
            raise RuntimeError("DB is read-only; cannot ingest.")
        cur = self.conn.cursor()
        try:
            cur.execute("BEGIN;")
            self.upsert_nodes(nodes)
            self.insert_edge_events(edge_events)
            self.upsert_edges_agg(edge_events)
            cur.execute("COMMIT;")
        except Exception:
            cur.execute("ROLLBACK;")
            raise

    # -------------------------
    # Read APIs (for queries)
    # -------------------------
    def iter_edges_agg_from_src(self, src_id: str, edge_type: Optional[str] = None) -> Iterator[sqlite3.Row]:
        cur = self.conn.cursor()
        if edge_type is None:
            rows = cur.execute(
                "SELECT * FROM edges_agg WHERE src_id = ?;",
                (src_id,),
            )
        else:
            rows = cur.execute(
                "SELECT * FROM edges_agg WHERE src_id = ? AND edge_type = ?;",
                (src_id, edge_type),
            )
        for r in rows:
            yield r

    def iter_neighbors(self, node_id: str) -> Iterator[str]:
        cur = self.conn.cursor()
        rows = cur.execute("SELECT dst_id FROM edges_agg WHERE src_id = ?;", (node_id,))
        for r in rows:
            yield str(r["dst_id"])

    def get_stats(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        n_nodes = cur.execute("SELECT COUNT(1) AS c FROM nodes;").fetchone()["c"]
        n_events = cur.execute("SELECT COUNT(1) AS c FROM edge_events;").fetchone()["c"]
        n_edges = cur.execute("SELECT COUNT(1) AS c FROM edges_agg;").fetchone()["c"]
        return {"nodes": int(n_nodes), "edge_events": int(n_events), "edges_agg": int(n_edges)}