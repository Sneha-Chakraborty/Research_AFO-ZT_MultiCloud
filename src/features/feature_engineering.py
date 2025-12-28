from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.logging_setup import setup_logger
from src.features.trust_graph_features import TrustGraphFeatureComputer


@dataclass
class FeatureBuildResult:
    rows: int
    out_csv: Optional[Path]
    out_parquet: Optional[Path]


def _scope_width(scope: object) -> int:
    if scope is None or (isinstance(scope, float) and np.isnan(scope)):  # type: ignore[arg-type]
        return 0
    s = str(scope).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return 0
    # split on common separators
    for sep in ["|", ",", ";", " "]:
        if sep in s:
            parts = [p for p in s.split(sep) if p.strip()]
            return len(parts)
    return 1


def _is_privileged_role(role: object) -> int:
    if role is None or (isinstance(role, float) and np.isnan(role)):  # type: ignore[arg-type]
        return 0
    s = str(role).lower()
    keys = ["admin", "owner", "root", "superuser", "priv", "security", "sudo"]
    return 1 if any(k in s for k in keys) else 0


def build_features(
    unified_csv: str | Path,
    trust_graph_db: str | Path,
    *,
    out_csv: str | Path | None = None,
    out_parquet: str | Path | None = None,
    window_minutes: int = 30,
    ewma_alpha: float = 0.2,
    max_hops_sensitive: int = 6,
    blast_k: int = 3,
    include_graph_static: bool = True,
) -> FeatureBuildResult:
    """Step 2.4 — Feature engineering (tabular + graph).

    Inputs:
      - unified telemetry CSV from Step 2.2
      - trust_graph.sqlite from Step 2.3

    Outputs:
      - features.csv (optional)
      - features.parquet (optional; falls back gracefully if pyarrow isn't installed)
    """
    logger = setup_logger("features")
    unified_csv = Path(unified_csv)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet is not None:
        out_parquet = Path(out_parquet)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(unified_csv)

    # Time ordering is important for rolling features
    df["ts_dt"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values("ts_dt").reset_index(drop=True)

    # Basic numeric cleaning
    for c in ["bytes_out", "latency_ms", "token_age", "posture_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ---- Tabular features (cheap) ----
    df["f_token_scope_width"] = df["token_scope"].apply(_scope_width) if "token_scope" in df.columns else 0
    df["f_token_age"] = df["token_age"] if "token_age" in df.columns else 0.0
    df["f_bytes_out_log1p"] = np.log1p(df["bytes_out"]) if "bytes_out" in df.columns else 0.0
    df["f_is_privileged_role"] = df["role"].apply(_is_privileged_role) if "role" in df.columns else 0

    # Prefer raw passthrough if available (from Step 2.2)
    if "raw_login_failures_5m" in df.columns:
        df["f_failed_logins_5m"] = pd.to_numeric(df["raw_login_failures_5m"], errors="coerce").fillna(0.0)
    else:
        # fallback: treat deny as failure proxy
        df["f_failed_logins_5m"] = (df.get("access_result", "").astype(str).str.lower() == "deny").astype(int)

    # Geo switch / "impossible travel" proxy (no coordinates available)
    df["f_geo_switch"] = 0
    df["f_fast_geo_switch"] = 0

    # Device posture drift via EWMA deviation
    df["f_posture_drift"] = 0.0

    # Bytes-out spike via EWMA z-like score per principal
    df["f_bytes_spike_score"] = 0.0

    # New resource edges in sliding window (principal -> resource)
    df["g_is_new_resource_edge"] = 0
    df["g_new_resource_count_win"] = 0
    df["g_new_resource_rate_win"] = 0.0

    # Cross-cloud hops per session (distinct clouds seen so far)
    df["g_cross_cloud_hops_session"] = 0

    # ---- Rolling state caches ----
    last_geo: Dict[str, str] = {}
    last_ts: Dict[str, pd.Timestamp] = {}
    posture_ewma: Dict[str, float] = {}
    bytes_ewma: Dict[str, float] = {}
    bytes_ewvar: Dict[str, float] = {}
    # session cloud sets
    sess_clouds: Dict[str, set] = {}
    # principal -> set of seen resources
    seen_res: Dict[str, set] = {}
    # principal -> deque of timestamps for new resource edges within window
    from collections import deque
    new_res_times: Dict[str, deque] = {}

    window = pd.Timedelta(minutes=int(window_minutes))

    for i, row in df.iterrows():
        pid = str(row.get("principal_id", "unknown"))
        did = str(row.get("device_id", "unknown"))
        sid = str(row.get("session_id", "unknown"))
        geo = str(row.get("geo", "unknown"))
        cloud = str(row.get("cloud_provider", "unknown"))
        ts = row.get("ts_dt")
        if pd.isna(ts):
            ts = None  # type: ignore[assignment]

        # geo switch features
        prev_geo = last_geo.get(pid)
        df.at[i, "f_geo_switch"] = 1 if (prev_geo is not None and geo != prev_geo) else 0
        if prev_geo is not None and geo != prev_geo and ts is not None:
            prev_t = last_ts.get(pid)
            if prev_t is not None and (ts - prev_t) <= pd.Timedelta(minutes=60):
                df.at[i, "f_fast_geo_switch"] = 1
        last_geo[pid] = geo
        if ts is not None:
            last_ts[pid] = ts

        # posture drift (per device)
        posture = float(row.get("posture_score", 0.0))
        pe = posture_ewma.get(did, posture)
        pe_new = ewma_alpha * posture + (1 - ewma_alpha) * pe
        posture_ewma[did] = pe_new
        df.at[i, "f_posture_drift"] = float(abs(posture - pe_new))

        # bytes spike (per principal)
        b = float(row.get("bytes_out", 0.0))
        be = bytes_ewma.get(pid, b)
        be_new = ewma_alpha * b + (1 - ewma_alpha) * be
        bytes_ewma[pid] = be_new
        # EW variance
        var = bytes_ewvar.get(pid, 0.0)
        diff = b - be_new
        var_new = ewma_alpha * (diff * diff) + (1 - ewma_alpha) * var
        bytes_ewvar[pid] = var_new
        denom = np.sqrt(var_new) + 1e-6
        df.at[i, "f_bytes_spike_score"] = float(diff / denom)

        # cross-cloud hops per session
        if sid not in sess_clouds:
            sess_clouds[sid] = set()
        sess_clouds[sid].add(cloud)
        df.at[i, "g_cross_cloud_hops_session"] = max(0, len(sess_clouds[sid]) - 1)

        # new resource edges and rate in window
        rid = str(row.get("resource_id", ""))
        if pid not in seen_res:
            seen_res[pid] = set()
            new_res_times[pid] = deque()

        is_new_edge = 0
        if rid and rid.lower() not in {"nan", "none", "unknown"}:
            if rid not in seen_res[pid]:
                is_new_edge = 1
                seen_res[pid].add(rid)
                if ts is not None:
                    new_res_times[pid].append(ts)

        df.at[i, "g_is_new_resource_edge"] = is_new_edge

        if ts is not None:
            dq = new_res_times[pid]
            while dq and (ts - dq[0]) > window:
                dq.popleft()
            df.at[i, "g_new_resource_count_win"] = len(dq)
            # normalize by total events for principal in same window (approx via last_ts diff)
            # simple proxy: count_win / window_minutes
            df.at[i, "g_new_resource_rate_win"] = float(len(dq) / max(1.0, window_minutes))

    # ---- Graph static features (shortest to sensitive, blast radius) ----
    if include_graph_static:
        logger.info("Computing static graph features (shortest path to sensitive, blast radius)...")
        gcomp = TrustGraphFeatureComputer(str(trust_graph_db))
        try:
            shortest_list: List[int] = []
            blast_list: List[int] = []
            cache_static: Dict[str, Tuple[int, int]] = {}

            for pid in df["principal_id"].astype(str).tolist():
                if pid in cache_static:
                    sh, bl = cache_static[pid]
                else:
                    sf = gcomp.static_features_for_principal(
                        pid, max_hops=max_hops_sensitive, k_hops=blast_k
                    )
                    sh, bl = sf.shortest_to_sensitive_hops, sf.blast_radius_sensitive_k
                    cache_static[pid] = (sh, bl)
                shortest_list.append(sh)
                blast_list.append(bl)

            df["g_shortest_to_sensitive_hops"] = shortest_list
            df["g_blast_radius_sensitive_k"] = blast_list
        finally:
            gcomp.close()
    else:
        df["g_shortest_to_sensitive_hops"] = -1
        df["g_blast_radius_sensitive_k"] = 0

    # ---- Final selection (keep labels + identifiers for downstream) ----
    keep_cols = [
        "ts",
        "principal_id",
        "principal_type",
        "role",
        "device_id",
        "workload_id",
        "session_id",
        "token_id",
        "cloud_provider",
        "tenant_id",
        "region",
        "resource_id",
        "resource_type",
        "resource_sensitivity",
        "action",
        "api",
        "operation",
        "access_result",
        "latency_ms",
        "bytes_out",
        "label_attack",
        "attack_type",
    ]
    feat_cols = [
        "f_failed_logins_5m",
        "f_geo_switch",
        "f_fast_geo_switch",
        "f_posture_drift",
        "f_token_age",
        "f_token_scope_width",
        "f_bytes_out_log1p",
        "f_bytes_spike_score",
        "f_is_privileged_role",
        "g_cross_cloud_hops_session",
        "g_is_new_resource_edge",
        "g_new_resource_count_win",
        "g_new_resource_rate_win",
        "g_shortest_to_sensitive_hops",
        "g_blast_radius_sensitive_k",
    ]

    # include passthrough raw_* signals if present (helps baselines too)
    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    final_cols = [c for c in keep_cols if c in df.columns] + raw_cols + feat_cols
    out_df = df[final_cols].copy()

    # write outputs
    out_csv_path = None
    out_parquet_path = None

    if out_csv is not None:
        out_df.to_csv(out_csv, index=False)
        out_csv_path = Path(out_csv)
        logger.info(f"✅ Wrote features CSV: {out_csv_path} (rows={len(out_df)})")

    if out_parquet is not None:
        try:
            out_df.to_parquet(out_parquet, index=False)
            out_parquet_path = Path(out_parquet)
            logger.info(f"✅ Wrote features Parquet: {out_parquet_path} (rows={len(out_df)})")
        except Exception as e:
            logger.warning(f"Could not write parquet (install pyarrow). Falling back to CSV only. Reason: {e}")

    return FeatureBuildResult(rows=len(out_df), out_csv=out_csv_path, out_parquet=out_parquet_path)
