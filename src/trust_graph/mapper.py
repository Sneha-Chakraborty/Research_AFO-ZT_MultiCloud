from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.trust_graph.sqlite_store import EdgeEvent, NodeUpsert


def _s(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x)


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def _i(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return int(x)
    except Exception:
        return default


def node_id(node_type: str, raw_id: str) -> str:
    return f"{node_type}:{raw_id}"


def build_graph_items_from_row(row: Any) -> Tuple[List[NodeUpsert], List[EdgeEvent]]:
    """Map one unified telemetry row -> nodes + edge events.

    Required in doc (Step 2.3):
    - Nodes: principal, role, device, workload, resource, cloud tenant, region
    - Edges: accessed, assumed-role, token-issued, called-api, data-read, connected-to
    - Each edge stores: timestamp, cloud_provider, sensitivity, bytes_out
    """
    ts = _s(getattr(row, "ts", ""))
    cloud = _s(getattr(row, "cloud_provider", ""))
    tenant_id_raw = _s(getattr(row, "tenant_id", ""))
    region_raw = _s(getattr(row, "region", ""))

    principal_raw = _s(getattr(row, "principal_id", ""))
    principal_type = _s(getattr(row, "principal_type", "")) or "unknown"
    role_raw = _s(getattr(row, "role", ""))
    device_raw = _s(getattr(row, "device_id", ""))
    workload_raw = _s(getattr(row, "workload_id", ""))
    resource_raw = _s(getattr(row, "resource_id", ""))
    resource_type = _s(getattr(row, "resource_type", ""))
    sensitivity = _s(getattr(row, "resource_sensitivity", ""))
    session_raw = _s(getattr(row, "session_id", ""))
    token_raw = _s(getattr(row, "token_id", ""))
    token_scope = _s(getattr(row, "token_scope", ""))
    action = _s(getattr(row, "action", ""))
    api = _s(getattr(row, "api", ""))
    operation = _s(getattr(row, "operation", ""))
    access_result = _s(getattr(row, "access_result", ""))
    bytes_out = _f(getattr(row, "bytes_out", 0.0), 0.0)
    label_attack = _i(getattr(row, "label_attack", 0), 0)
    attack_type = _s(getattr(row, "attack_type", ""))

    # Stable IDs (include cloud namespace where appropriate)
    cloud_id = node_id("cloud", cloud) if cloud else ""
    tenant_id = node_id("tenant", f"{cloud}:{tenant_id_raw}" if cloud and tenant_id_raw else tenant_id_raw) if tenant_id_raw else ""
    region_id = node_id("region", f"{cloud}:{region_raw}" if cloud and region_raw else region_raw) if region_raw else ""

    principal_id = node_id("principal", principal_raw) if principal_raw else ""
    role_id = node_id("role", role_raw) if role_raw else ""
    device_id = node_id("device", device_raw) if device_raw else ""
    workload_id = node_id("workload", workload_raw) if workload_raw else ""
    resource_id = node_id("resource", resource_raw) if resource_raw else ""
    session_id = node_id("session", session_raw) if session_raw else ""
    token_id = node_id("token", token_raw) if token_raw else ""
    api_id = node_id("api", api) if api else ""

    nodes: List[NodeUpsert] = []
    edges: List[EdgeEvent] = []

    def add_node(nid: str, ntype: str, attrs: Dict[str, Any]) -> None:
        if not nid:
            return
        nodes.append(NodeUpsert(node_id=nid, node_type=ntype, ts=ts, attrs=attrs))

    def add_edge(edge_type: str, src: str, dst: str, extra_sensitivity: str = "", extra_bytes_out: float = 0.0) -> None:
        if not (src and dst):
            return
        edges.append(
            EdgeEvent(
                ts=ts,
                cloud_provider=cloud or "unknown",
                tenant_id=tenant_id_raw,
                region=region_raw,
                edge_type=edge_type,
                src_id=src,
                dst_id=dst,
                sensitivity=extra_sensitivity,
                bytes_out=extra_bytes_out,
                access_result=access_result,
                action=action,
                api=api,
                operation=operation,
                label_attack=label_attack,
                attack_type=attack_type,
            )
        )

    # Nodes
    if cloud_id:
        add_node(cloud_id, "cloud", {"cloud_provider": cloud})
    if tenant_id:
        add_node(tenant_id, "tenant", {"tenant_id": tenant_id_raw, "cloud_provider": cloud})
        if cloud_id:
            add_edge("connected-to", tenant_id, cloud_id)
    if region_id:
        add_node(region_id, "region", {"region": region_raw, "cloud_provider": cloud})
        if cloud_id:
            add_edge("connected-to", region_id, cloud_id)

    if principal_id:
        add_node(principal_id, "principal", {"principal_id": principal_raw, "principal_type": principal_type})
        if tenant_id:
            add_edge("connected-to", principal_id, tenant_id)
        elif cloud_id:
            add_edge("connected-to", principal_id, cloud_id)

    if role_id:
        add_node(role_id, "role", {"role": role_raw})
        if principal_id:
            add_edge("assumed-role", principal_id, role_id)

    if device_id:
        add_node(device_id, "device", {"device_id": device_raw, "posture_score": getattr(row, "posture_score", None)})
        if principal_id:
            add_edge("connected-to", principal_id, device_id)

    if workload_id:
        add_node(workload_id, "workload", {"workload_id": workload_raw})
        if principal_id:
            add_edge("connected-to", principal_id, workload_id)

    if resource_id:
        add_node(resource_id, "resource", {"resource_id": resource_raw, "resource_type": resource_type, "sensitivity": sensitivity})
        if region_id:
            add_edge("connected-to", resource_id, region_id)

    if session_id:
        add_node(session_id, "session", {"session_id": session_raw})
        if principal_id:
            add_edge("connected-to", principal_id, session_id)

    if token_id:
        add_node(token_id, "token", {"token_id": token_raw, "token_scope": token_scope})
        # token-issued: usually when token appears; connect principal/session to token
        if session_id:
            add_edge("token-issued", session_id, token_id)
        elif principal_id:
            add_edge("token-issued", principal_id, token_id)

    if api_id:
        add_node(api_id, "api", {"api": api})
        if principal_id:
            add_edge("called-api", principal_id, api_id)

    # Core access edges
    if principal_id and resource_id:
        add_edge("accessed", principal_id, resource_id, extra_sensitivity=sensitivity, extra_bytes_out=bytes_out)

        # data-read is a more specific edge type (use action + bytes_out heuristic)
        if action.lower() in {"read", "download", "list"} or bytes_out > 0:
            add_edge("data-read", principal_id, resource_id, extra_sensitivity=sensitivity, extra_bytes_out=bytes_out)

    return nodes, edges
