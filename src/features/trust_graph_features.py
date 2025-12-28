from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.trust_graph.sqlite_store import TrustGraphSQLite


SENSITIVE_LABELS_DEFAULT = {"high", "critical", "secret"}


@dataclass(frozen=True)
class GraphStaticFeatures:
    shortest_to_sensitive_hops: int  # -1 if unreachable
    blast_radius_sensitive_k: int    # count of sensitive nodes within K hops (undirected BFS)


class TrustGraphFeatureComputer:
    """Compute graph-derived features from the SQLite trust graph.

    Notes
    - Uses UNDIRECTED neighborhood queries to allow navigation across structural edges
    (tenant<->cloud<->region<->resource), since structural edges are stored as directed.
    - Caches neighbor lists to keep CPU/runtime reasonable.
    """

    def __init__(
        self,
        db_path: str,
        *,
        sensitivity_labels: Optional[Set[str]] = None,
        max_neighbor_cache: int = 200_000,
    ) -> None:
        self.store = TrustGraphSQLite(db_path=db_path, read_only=True)  # type: ignore[arg-type]
        self.sensitivity_labels = {s.lower() for s in (sensitivity_labels or SENSITIVE_LABELS_DEFAULT)}
        self._neighbor_cache: Dict[str, List[str]] = {}
        self._max_neighbor_cache = int(max_neighbor_cache)
        self._sensitive_nodes = self._load_sensitive_resource_nodes()

    def close(self) -> None:
        self.store.close()

    def _load_sensitive_resource_nodes(self) -> Set[str]:
        cur = self.store.conn.cursor()
        sens: Set[str] = set()
        for r in cur.execute("SELECT node_id, attrs_json FROM nodes WHERE node_type='resource';"):
            attrs = r["attrs_json"] or "{}"
            try:
                a = json.loads(attrs)
            except Exception:
                a = {}
            label = str(a.get("sensitivity", "")).lower()
            if label in self.sensitivity_labels:
                sens.add(str(r["node_id"]))
        return sens

    def neighbors_undirected(self, node_id: str) -> List[str]:
        """Return undirected neighbors by reading both directions from edges_agg."""
        if node_id in self._neighbor_cache:
            return self._neighbor_cache[node_id]

        cur = self.store.conn.cursor()
        out: List[str] = []
        # forward neighbors
        for r in cur.execute("SELECT dst_id FROM edges_agg WHERE src_id=?;", (node_id,)):
            out.append(str(r["dst_id"]))
        # reverse neighbors
        for r in cur.execute("SELECT src_id FROM edges_agg WHERE dst_id=?;", (node_id,)):
            out.append(str(r["src_id"]))

        # lightweight cache management
        if len(self._neighbor_cache) < self._max_neighbor_cache:
            self._neighbor_cache[node_id] = out
        return out

    def shortest_path_to_sensitive(
        self,
        start_node: str,
        *,
        max_hops: int = 6,
    ) -> int:
        """Return shortest hop distance from start_node to any sensitive resource; -1 if none within max_hops."""
        if not self._sensitive_nodes:
            return -1

        q = deque([start_node])
        dist: Dict[str, int] = {start_node: 0}

        while q:
            u = q.popleft()
            d = dist[u]
            if d > max_hops:
                continue
            if u in self._sensitive_nodes and u != start_node:
                return d

            for v in self.neighbors_undirected(u):
                if v not in dist:
                    dist[v] = d + 1
                    if dist[v] <= max_hops:
                        q.append(v)
        return -1

    def blast_radius_sensitive(
        self,
        start_node: str,
        *,
        k_hops: int = 3,
        max_nodes: int = 50_000,
    ) -> int:
        """Count sensitive resources reachable within k_hops (undirected)."""
        if not self._sensitive_nodes:
            return 0

        q = deque([start_node])
        dist: Dict[str, int] = {start_node: 0}
        count = 0

        while q:
            u = q.popleft()
            d = dist[u]
            if d > k_hops:
                continue
            if u in self._sensitive_nodes and u != start_node:
                count += 1

            if len(dist) > max_nodes:
                # safety valve for pathological graphs
                break

            for v in self.neighbors_undirected(u):
                if v not in dist:
                    dist[v] = d + 1
                    if dist[v] <= k_hops:
                        q.append(v)

        return count

    def static_features_for_principal(
        self,
        principal_id_raw: str,
        *,
        max_hops: int = 6,
        k_hops: int = 3,
    ) -> GraphStaticFeatures:
        start = f"principal:{principal_id_raw}"
        return GraphStaticFeatures(
            shortest_to_sensitive_hops=self.shortest_path_to_sensitive(start, max_hops=max_hops),
            blast_radius_sensitive_k=self.blast_radius_sensitive(start, k_hops=k_hops),
        )
