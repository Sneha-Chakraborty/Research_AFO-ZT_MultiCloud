from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.utils import parse_ts, synthetic_ipv4, stable_hash_to_int, to_bool01


UNIFIED_COLUMNS: List[str] = [
    # time
    "ts",
    # who
    "principal_id",
    "principal_type",
    "role",
    "mfa_used",
    # what
    "action",
    "api",
    "operation",
    "resource_id",
    "resource_type",
    "resource_sensitivity",
    # where
    "cloud_provider",
    "tenant_id",
    "region",
    "ip",
    "geo",
    # device/workload
    "device_id",
    "posture_score",
    "workload_id",
    # session
    "session_id",
    "token_id",
    "token_scope",
    "token_age",
    # outcomes
    "access_result",
    "latency_ms",
    "bytes_out",
    # truth
    "label_attack",
    "attack_type",
]


RAW_PASSTHROUGH = [
    "login_failures_5m",
    "api_calls_5m",
    "time_anomaly",
    "privilege_escalation",
    "prev_cloud",
    "cross_cloud_hop",
    "device_os",
    "device_compliance",
    "auth_method",
    "risk_truth",
    "access_granted",
    "label_malicious",
    "cloud",
    "cloud_account",
    "country",
    "user_id",
]


def _norm_cloud(x: object) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    return {"aws": "aws", "azure": "azure", "gcp": "gcp"}.get(s, s)


def _posture_score_from_compliance(x: object) -> float:
    """Map device compliance / posture input to a 0..1 score.

    Supports:
    - numeric scores already in [0,1]
    - boolean-ish or string labels (COMPLIANT/NONCOMPLIANT)
    """
    if pd.isna(x):
        return 0.5
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = float(x)
            # if it's already a score (common in the synthetic dataset)
            if 0.0 <= v <= 1.0:
                return float(np.clip(v, 0.0, 1.0))
            # if it's a percentage-like value
            if 1.0 < v <= 100.0:
                return float(np.clip(v / 100.0, 0.0, 1.0))
        except Exception:
            pass
        return 0.5

    s = str(x).strip().lower()
    if s in {"compliant", "true", "1", "yes"}:
        return 0.9
    if s in {"noncompliant", "non-compliant", "false", "0", "no"}:
        return 0.2
    return 0.5
    s = str(x).strip().lower()
    # map common values
    if s in {"compliant", "true", "1", "yes"}:
        return 0.9
    if s in {"noncompliant", "non-compliant", "false", "0", "no"}:
        return 0.2
    return 0.5


def _normalize_action(action: object) -> str:
    if pd.isna(action):
        return "unknown"
    s = str(action).strip().lower()
    return s


def _derive_api(resource_type: object, action: object) -> str:
    rt = "unknown" if pd.isna(resource_type) else str(resource_type).strip().lower()
    ac = _normalize_action(action)
    # cloud-agnostic label
    return f"{rt}:{ac}"


def _derive_operation(action: object) -> str:
    return _normalize_action(action)


def _derive_token_id(principal_id: object, session_id: object) -> str:
    n = stable_hash_to_int(principal_id, session_id, mod=10**10)
    return f"tok_{n:010d}"


def _derive_workload_id(principal_id: object, cloud_provider: object) -> str:
    n = stable_hash_to_int(principal_id, cloud_provider, mod=10**8)
    return f"wk_{n:08d}"


def _derive_latency_ms(ts: pd.Series, bytes_out: pd.Series, principal_id: pd.Series, action: pd.Series) -> pd.Series:
    """Deterministic synthetic latency so later steps can compare response time."""
    base = np.log1p(bytes_out.fillna(0).astype(float)) * 12.0
    # add deterministic jitter 0..50ms based on ids
    jitter = [
        stable_hash_to_int(t, b, p, a, mod=5000) / 100.0
        for t, b, p, a in zip(ts.astype(str), bytes_out.fillna(0), principal_id.astype(str), action.astype(str))
    ]
    return (base + np.array(jitter)).astype(float)


def normalize_to_unified_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw logs into a unified telemetry schema.

    Works with the synthetic dataset produced in Step 2.1, but is designed to be tolerant
    if some optional fields are missing.
    """
    df = df.copy()

    # Timestamp
    if "ts" in df.columns:
        df["ts"] = parse_ts(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    elif "timestamp" in df.columns:
        df["ts"] = parse_ts(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["ts"] = pd.NaT

    # Cloud / tenant
    if "cloud_provider" in df.columns:
        df["cloud_provider"] = df["cloud_provider"].map(_norm_cloud)
    elif "cloud" in df.columns:
        df["cloud_provider"] = df["cloud"].map(_norm_cloud)
    else:
        df["cloud_provider"] = "unknown"

    if "tenant_id" not in df.columns:
        if "cloud_account" in df.columns:
            df["tenant_id"] = df["cloud_account"].astype(str)
        elif "tenant" in df.columns:
            df["tenant_id"] = df["tenant"].astype(str)
        else:
            df["tenant_id"] = "unknown"

    # geo / region / ip
    if "geo" not in df.columns:
        if "country" in df.columns:
            df["geo"] = df["country"].astype(str)
        else:
            df["geo"] = "unknown"

    if "ip" not in df.columns:
        # deterministic private IP from principal + device + tenant
        pid = df["user_id"] if "user_id" in df.columns else df.get("principal_id", "unknown")
        did = df["device_id"] if "device_id" in df.columns else "unknown"
        tid = df["tenant_id"]
        df["ip"] = [synthetic_ipv4(p, d, t) for p, d, t in zip(pid.astype(str), did.astype(str), tid.astype(str))]

    # Who
    if "principal_id" not in df.columns:
        if "user_id" in df.columns:
            df["principal_id"] = df["user_id"].astype(str)
        else:
            df["principal_id"] = "unknown"

    if "principal_type" not in df.columns:
        df["principal_type"] = "user"

    if "role" not in df.columns:
        df["role"] = "unknown"

    if "mfa_used" in df.columns:
        df["mfa_used"] = df["mfa_used"].apply(to_bool01).astype(int)
    else:
        df["mfa_used"] = 0

    # What
    if "action" not in df.columns:
        df["action"] = "unknown"
    df["action"] = df["action"].apply(_normalize_action)

    if "resource_id" not in df.columns:
        df["resource_id"] = "unknown"

    if "resource_type" not in df.columns:
        df["resource_type"] = "unknown"

    if "resource_sensitivity" not in df.columns:
        df["resource_sensitivity"] = "UNKNOWN"

    if "api" not in df.columns:
        df["api"] = [_derive_api(rt, ac) for rt, ac in zip(df["resource_type"], df["action"])]

    if "operation" not in df.columns:
        df["operation"] = df["action"].apply(_derive_operation)

    # Device / workload
    if "device_id" not in df.columns:
        df["device_id"] = "unknown"

    if "posture_score" not in df.columns:
        if "device_compliance" in df.columns:
            df["posture_score"] = df["device_compliance"].apply(_posture_score_from_compliance).astype(float)
        else:
            df["posture_score"] = 0.5

    if "workload_id" not in df.columns:
        df["workload_id"] = [_derive_workload_id(pid, cp) for pid, cp in zip(df["principal_id"], df["cloud_provider"])]

    # Session
    if "session_id" not in df.columns:
        df["session_id"] = "unknown"

    if "token_scope" not in df.columns:
        df["token_scope"] = "UNKNOWN"

    if "token_age" not in df.columns:
        if "token_age_s" in df.columns:
            df["token_age"] = pd.to_numeric(df["token_age_s"], errors="coerce")
        else:
            df["token_age"] = np.nan

    if "token_id" not in df.columns:
        df["token_id"] = [_derive_token_id(pid, sid) for pid, sid in zip(df["principal_id"], df["session_id"])]

    # Outcomes
    if "access_result" not in df.columns:
        if "access_granted" in df.columns:
            df["access_result"] = ["permit" if int(x) == 1 else "deny" for x in df["access_granted"].fillna(0)]
        else:
            df["access_result"] = "unknown"

    if "bytes_out" not in df.columns:
        df["bytes_out"] = 0.0
    df["bytes_out"] = pd.to_numeric(df["bytes_out"], errors="coerce").fillna(0.0)

    if "latency_ms" not in df.columns:
        df["latency_ms"] = _derive_latency_ms(df["ts"], df["bytes_out"], df["principal_id"], df["action"])

    # Truth
    if "label_attack" not in df.columns:
        if "label_malicious" in df.columns:
            df["label_attack"] = df["label_malicious"].astype(int)
        else:
            df["label_attack"] = 0

    if "attack_type" not in df.columns:
        df["attack_type"] = df.get("attack_type", "UNKNOWN")

    # Region
    if "region" not in df.columns:
        df["region"] = "unknown"
    df["region"] = df["region"].astype(str)

    # Keep passthrough fields (prefixed) if they exist
    passthrough = {}
    for c in RAW_PASSTHROUGH:
        if c in df.columns:
            passthrough[f"raw_{c}"] = df[c]

    unified = df[UNIFIED_COLUMNS].copy()
    for k, v in passthrough.items():
        unified[k] = v

    return unified


def normalize_csv(input_csv: str | Path, output_csv: str | Path) -> Tuple[Path, int]:
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    out = normalize_to_unified_schema(df)
    out.to_csv(output_csv, index=False)
    return output_csv, len(out)
