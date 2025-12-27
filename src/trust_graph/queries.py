from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.trust_graph.sqlite_store import TrustGraphSQLite


@dataclass
class PivotFinding:
    principal_node: str
    clouds: List[str]
    unique_resources: int
    unique_tenants: int


def find_cross_cloud_pivots(store: TrustGraphSQLite, principal_id_raw: str) -> PivotFinding:
    """Return a simple cross-cloud pivot summary for one principal."""
    principal_node = f"principal:{principal_id_raw}"
    clouds: Set[str] = set()
    resources: Set[str] = set()
    tenants: Set[str] = set()

    cur = store.conn.cursor()
    # Look at aggregated edges from principal to resource (accessed), and principal to tenant/cloud (connected-to)
    for r in cur.execute(
        """
        SELECT edge_type, dst_id, cloud_provider
        FROM edges_agg
        WHERE src_id = ? AND (edge_type = 'accessed' OR edge_type = 'connected-to');
        """,
        (principal_node,),
    ):
        clouds.add(str(r["cloud_provider"]))
        dst = str(r["dst_id"])
        if dst.startswith("resource:"):
            resources.add(dst)
        if dst.startswith("tenant:"):
            tenants.add(dst)

    return PivotFinding(
        principal_node=principal_node,
        clouds=sorted(clouds),
        unique_resources=len(resources),
        unique_tenants=len(tenants),
    )


def top_risky_edges(store: TrustGraphSQLite, limit: int = 20) -> List[Dict[str, object]]:
    """Rank edges by a simple risk proxy for demo/research plots.

    risk_proxy = deny_count + 2*attack_count + log1p(bytes_out_sum/1e6)
    """
    cur = store.conn.cursor()
    rows = cur.execute(
        """
        SELECT
          src_id, dst_id, edge_type, cloud_provider,
          deny_count, attack_count, bytes_out_sum,
          event_count, last_seen, attrs_json
        FROM edges_agg
        ORDER BY (deny_count + 2*attack_count + (CASE WHEN bytes_out_sum > 0 THEN (ln(1 + bytes_out_sum / 1000000.0)) ELSE 0 END)) DESC
        LIMIT ?;
        """,
        (int(limit),),
    ).fetchall()

    out: List[Dict[str, object]] = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    return out


def shortest_path_to_sensitive_resource(
    store: TrustGraphSQLite,
    start_node: str,
    *,
    max_hops: int = 6,
    sensitivity_labels: Optional[Set[str]] = None,
) -> Optional[List[str]]:
    """BFS over edges_agg to find a path to any sensitive resource.

    This avoids heavy dependencies (NetworkX) while remaining reproducible.
    """
    if sensitivity_labels is None:
        sensitivity_labels = {"high", "critical", "secret"}

    cur = store.conn.cursor()

    # Preload sensitive resource nodes
    sens_resources: Set[str] = set()
    for r in cur.execute("SELECT node_id, attrs_json FROM nodes WHERE node_type = 'resource';"):
        attrs = r["attrs_json"] or "{}"
        try:
            import json as _json

            a = _json.loads(attrs)
        except Exception:
            a = {}
        label = str(a.get("sensitivity", "")).lower()
        if label in sensitivity_labels:
            sens_resources.add(str(r["node_id"]))

    # BFS
    q = deque([start_node])
    parent: Dict[str, str] = {start_node: ""}
    depth: Dict[str, int] = {start_node: 0}

    while q:
        u = q.popleft()
        d = depth[u]
        if d > max_hops:
            continue
        if u in sens_resources and u != start_node:
            # reconstruct
            path = [u]
            while parent[path[-1]]:
                path.append(parent[path[-1]])
            return list(reversed(path))

        for v in store.iter_neighbors(u):
            if v not in parent:
                parent[v] = u
                depth[v] = d + 1
                q.append(v)

    return None
