# Cross-Cloud Trust Graph (Step 2.3) — SQLite Schema

This prototype maintains a **live cross-cloud trust graph** using SQLite for scalability and reproducibility.

## Why SQLite?

- Handles 50k+ events on a laptop without in-memory graph blowups
- Supports incremental updates (append new events)
- Enables query + analytics without heavy dependencies

---

## Tables

### 1) `nodes`
Deduplicated graph nodes.

| Column | Type | Meaning |
|---|---|---|
| node_id | TEXT (PK) | Deterministic id: `type:value` (e.g., `principal:u_102`) |
| node_type | TEXT | principal / role / device / workload / resource / tenant / region / cloud / session / token / api |
| first_seen | TEXT | first timestamp observed |
| last_seen | TEXT | last timestamp observed |
| attrs_json | TEXT | JSON attributes (type-specific) |

### 2) `edge_events`
**Append-only** edge occurrences — each edge stores timestamp (research requirement).

| Column | Type | Meaning |
|---|---|---|
| event_id | INTEGER (PK) | auto id |
| ts | TEXT | event timestamp |
| cloud_provider | TEXT | aws / azure / gcp |
| tenant_id | TEXT | tenant |
| region | TEXT | region |
| edge_type | TEXT | accessed / assumed-role / token-issued / called-api / data-read / connected-to |
| src_id, dst_id | TEXT | node ids |
| sensitivity | TEXT | resource sensitivity (if applicable) |
| bytes_out | REAL | exfil proxy |
| access_result | TEXT | permit/deny/restrict |
| action/api/operation | TEXT | normalized action details |
| label_attack | INTEGER | 0/1 |
| attack_type | TEXT | label |

### 3) `edges_agg`
Aggregated edges (fast queries).

Unique key: `(src_id, edge_type, dst_id, cloud_provider)` stored as `edge_key`.

Stores:
- counts: permit/deny/restrict, attack_count
- bytes_out_sum
- first_seen / last_seen
- compact last-seen attributes

### 4) `meta`
Stores incremental ingestion state.
- `last_ingested_ts`

---

## Deterministic Node IDs

- `principal:{principal_id}`
- `role:{role}`
- `device:{device_id}`
- `workload:{workload_id}`
- `resource:{resource_id}`
- `tenant:{cloud}:{tenant_id}`
- `region:{cloud}:{region}`
- `cloud:{cloud_provider}`
- `session:{session_id}`
- `token:{token_id}`
- `api:{api}`

---

## How to Build / Update

- Build or update (incremental by default):
  `python scripts/build_trust_graph.py`

- Full rebuild:
  `python scripts/build_trust_graph.py --reset-db`
