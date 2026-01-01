from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


_DECISION_SEVERITY = {
    "allow": 0,
    "stepup": 1,
    "restrict": 2,
    "deny": 3,
    "block": 3,
}


def decision_severity(decision: Any) -> int:
    s = str(decision).strip().lower()
    return int(_DECISION_SEVERITY.get(s, 0))


def cross_cloud_attack_detection_rate(
    df: pd.DataFrame,
    *,
    decision_col: str = "decision_tuned",
    xcloud_col_candidates: Tuple[str, ...] = (
        "g_cross_cloud_hops_session",
        "raw_cross_cloud_hop",
        "cross_cloud_hops",
        "is_cross_cloud_event",
        "is_cross_cloud",
    ),
    cloud_col_candidates: Tuple[str, ...] = ("cloud_provider", "cloud", "provider"),
    key_col_candidates: Tuple[str, ...] = ("session_id", "raw_session_id", "trace_id", "principal_id", "user_id"),
) -> Dict[str, float]:
    """Cross-cloud pivot detection rate.

    Why this exists:
      Some score CSVs (especially AFO-ZT outputs) may not carry the raw generator
      flag columns used to mark cross-cloud pivots. This function therefore:

      1) Uses an explicit cross-cloud flag/hops column if present (fast path).
      2) Otherwise derives cross-cloud events by grouping a key (prefer session_id)
         and checking if the same key touches >1 cloud providers.

    Metric:
      - xcloud_events = number of events that belong to a cross-cloud group
      - xcloud_attacks = number of attack events among those
      - xcloud_flagged_tpr = among xcloud_attacks, fraction flagged (stepup/restrict/deny)
      - xcloud_hard_tpr = among xcloud_attacks, fraction hard (restrict/deny)
    """
    if "label_attack" not in df.columns:
        return {"xcloud_events": 0.0, "xcloud_attacks": 0.0, "xcloud_flagged_tpr": 0.0, "xcloud_hard_tpr": 0.0}

    # 1) Fast path: explicit cross-cloud marker exists
    xcol = None
    for c in xcloud_col_candidates:
        if c in df.columns:
            xcol = c
            break

    is_x: Optional[pd.Series] = None
    if xcol is not None:
        x = pd.to_numeric(df[xcol], errors="coerce").fillna(0)
        # accept booleans/0-1/integers; treat >0.5 as True
        is_x = x.astype(float) > 0.5

    # 2) Derive: group-by key touches multiple clouds
    if is_x is None:
        cloud_col = next((c for c in cloud_col_candidates if c in df.columns), None)
        key_col = next((c for c in key_col_candidates if c in df.columns), None)

        if cloud_col is None or key_col is None:
            return {"xcloud_events": 0.0, "xcloud_attacks": 0.0, "xcloud_flagged_tpr": 0.0, "xcloud_hard_tpr": 0.0}

        tmp = df[[key_col, cloud_col]].copy()
        tmp[key_col] = tmp[key_col].fillna("unknown").astype(str)
        tmp[cloud_col] = tmp[cloud_col].fillna("unknown").astype(str)

        n_clouds = tmp.groupby(key_col)[cloud_col].transform("nunique")
        is_x = n_clouds > 1

    y = df["label_attack"].fillna(0).astype(int)
    is_attack = y == 1

    d = df.get(decision_col, "").fillna("").astype(str).str.lower()
    flagged = d.isin(["stepup", "restrict", "deny"])
    hard = d.isin(["restrict", "deny"])

    x_attacks = is_x & is_attack
    denom = int(x_attacks.sum())
    return {
        "xcloud_events": float(is_x.sum()),
        "xcloud_attacks": float(denom),
        "xcloud_flagged_tpr": float(flagged[x_attacks].mean()) if denom > 0 else 0.0,
        "xcloud_hard_tpr": float(hard[x_attacks].mean()) if denom > 0 else 0.0,
    }

def blast_radius_reduction(
    df: pd.DataFrame,
    *,
    decision_col: str = "decision_tuned",
    blast_col: str = "g_blast_radius_sensitive_k",
    blast_col_candidates: Tuple[str, ...] = (
        "g_blast_radius_sensitive_k",
        "g_sensitive_blast_k",
        "g_sensitive_reach_k",
        "g_sensitive_neighbor_count",
        "g_sensitive_neighbors_k",
    ),
    pivot_col_candidates: Tuple[str, ...] = (
        "g_pivot_score",
        "g_graph_pivot_score",
        "graph_pivot_score",
        "pivot_score",
    ),
) -> Dict[str, float]:
    """Graph-based containment proxy: blast radius reduction.

    Some score files do not carry an explicit blast radius feature. We therefore:
      1) Prefer an explicit blast feature (k-hop sensitive blast / reach).
      2) Else fall back to a proxy based on pivot score (higher pivot score => larger blast).

    We apply conservative reduction factors by decision severity and report (attacks only):
      - mean_blast_before
      - mean_blast_after
      - blast_radius_reduction_pct = 100*(1 - after/before)
    """
    if "label_attack" not in df.columns:
        return {
            "mean_blast_before": float("nan"),
            "mean_blast_after": float("nan"),
            "blast_radius_reduction_pct": float("nan"),
        }

    # Pick an available blast feature
    bcol = None
    if blast_col in df.columns:
        bcol = blast_col
    else:
        bcol = next((c for c in blast_col_candidates if c in df.columns), None)

    attacks = df.loc[df["label_attack"].fillna(0).astype(int) == 1].copy()
    if attacks.empty:
        return {"mean_blast_before": 0.0, "mean_blast_after": 0.0, "blast_radius_reduction_pct": 0.0}

    if bcol is not None:
        blast = pd.to_numeric(attacks[bcol], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    else:
        # Proxy: pivot score -> blast estimate in "reachable sensitive nodes" units (rough, but comparable)
        pcol = next((c for c in pivot_col_candidates if c in attacks.columns), None)
        if pcol is None:
            return {
                "mean_blast_before": float("nan"),
                "mean_blast_after": float("nan"),
                "blast_radius_reduction_pct": float("nan"),
            }
        piv = pd.to_numeric(attacks[pcol], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
        # Scale pivot to a plausible blast count range [1..~21]
        blast = 1.0 + (piv.clip(upper=1.0) * 20.0)

    dec = attacks.get(decision_col, "").fillna("").astype(str).str.lower()

    # Reduction factors (interpretable, conservative)
    red = dec.map({"allow": 0.00, "stepup": 0.10, "restrict": 0.45, "deny": 0.75}).fillna(0.00).astype(float)
    after = blast * (1.0 - red)

    before_mean = float(blast.mean()) if len(blast) else 0.0
    after_mean = float(after.mean()) if len(after) else 0.0
    reduction = 0.0 if before_mean <= 0 else 100.0 * (1.0 - (after_mean / before_mean))

    return {
        "mean_blast_before": before_mean,
        "mean_blast_after": after_mean,
        "blast_radius_reduction_pct": float(reduction),
    }

def policy_consistency_variance(
    df: pd.DataFrame,
    *,
    decision_col: str = "decision_tuned",
    key_col: str = "principal_id",
    cloud_col: str = "cloud_provider",
    min_events_per_key: int = 10,
) -> Dict[str, float]:
    """Policy consistency across clouds (variance of outcomes).

    We compute a severity score per decision and measure how much the *same principal*
    receives different average severities across clouds. Lower is better (more consistent).
    """
    if key_col not in df.columns or cloud_col not in df.columns:
        return {"policy_consistency_var": float("nan"), "deny_rate_var": float("nan")}

    tmp = df[[key_col, cloud_col, decision_col]].copy()
    tmp[key_col] = tmp[key_col].fillna("unknown").astype(str)
    tmp[cloud_col] = tmp[cloud_col].fillna("unknown").astype(str)
    tmp["_sev"] = tmp[decision_col].apply(decision_severity).astype(float)
    tmp["_deny"] = tmp[decision_col].fillna("").astype(str).str.lower().isin(["deny"]).astype(float)

    # Principal must have multiple clouds to be meaningful
    counts = tmp.groupby(key_col).size()
    eligible = counts[counts >= min_events_per_key].index
    tmp = tmp[tmp[key_col].isin(eligible)]
    if tmp.empty:
        return {"policy_consistency_var": 0.0, "deny_rate_var": 0.0}

    # per (principal, cloud) means
    pc = tmp.groupby([key_col, cloud_col], dropna=False).agg(sev_mean=("_sev", "mean"), deny_rate=("_deny", "mean")).reset_index()

    # variance across clouds per principal
    sev_var = pc.groupby(key_col)["sev_mean"].var(ddof=0).fillna(0.0)
    deny_var = pc.groupby(key_col)["deny_rate"].var(ddof=0).fillna(0.0)

    return {
        "policy_consistency_var": float(sev_var.mean()),
        "deny_rate_var": float(deny_var.mean()),
    }


def rollout_safety_metrics_from_json(path: str) -> Dict[str, float]:
    """Extract rollout safety metrics from rollout_metrics.json (Step 2.7)."""
    try:
        import json as _json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return {"canary_false_deny_rate": float("nan"), "rollout_false_deny_rate": float("nan")}
        raw = _json.loads(p.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {"canary_false_deny_rate": float("nan"), "rollout_false_deny_rate": float("nan")}

    def _get(stage: str, key: str) -> float:
        try:
            return float(raw.get(stage, {}).get(key, float("nan")))
        except Exception:
            return float("nan")

    return {
        "canary_false_deny_rate": _get("canary", "false_deny_rate"),
        "rollout_false_deny_rate": _get("rollout", "false_deny_rate"),
        "canary_mfa_friction_rate": _get("canary", "mfa_friction_rate"),
        "rollout_mfa_friction_rate": _get("rollout", "mfa_friction_rate"),
        "canary_latency_p95_ms": _get("canary", "latency_p95"),
        "rollout_latency_p95_ms": _get("rollout", "latency_p95"),
        "final_stage_budgets_ok": float(raw.get("final_stage_budgets_ok", float("nan"))),
    }
