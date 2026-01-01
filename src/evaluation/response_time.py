from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.afozt.orchestrator import OrchestratorConfig, SimulatedOrchestrator
from src.afozt.policy_compiler import PolicyCompiler
from src.afozt.policy_intent import IntentDecision
from src.afozt.safe_rollout import policy_compiler_config_from_thresholds_yaml
from src.evaluation.metrics import latency_summary


def _to_intent_decision(x: Any) -> IntentDecision:
    s = str(x).strip().lower()
    if s == "stepup":
        return IntentDecision.STEPUP
    if s == "restrict":
        return IntentDecision.RESTRICT
    if s in {"deny", "block"}:
        return IntentDecision.DENY
    return IntentDecision.ALLOW


@dataclass
class LatencyProfile:
    """Extra processing delay added before actions execute (seconds)."""
    processing_delay_s: float = 0.0


def load_latency_profiles(path: str) -> Dict[str, LatencyProfile]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles", {}) or {}
    out: Dict[str, LatencyProfile] = {}
    for name, cfg in profiles.items():
        out[str(name)] = LatencyProfile(processing_delay_s=float(cfg.get("processing_delay_s", 0.0)))
    return out


def compute_response_times(
    df: pd.DataFrame,
    *,
    thresholds_yaml: str = "configs/thresholds.yaml",
    decision_col: str = "decision_tuned",
    only_non_allow: bool = True,
    extra_processing_delay_s: float = 0.0,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute response-time stats (MTTR proxy) for a decision stream.

    This is an offline simulator:
      - Forces intent decision from decision_col (no re-thresholding).
      - Compiles SOAR plans with PolicyCompiler.
      - Sums per-action latencies from SimulatedOrchestrator.
      - Adds extra_processing_delay_s to reflect engine overhead.

    Returns:
      - response_times_df: one row per executed intent with response_time_s
      - stats: latency_summary over response_time_s
    """
    if decision_col not in df.columns:
        raise KeyError(f"Missing decision column: {decision_col}")

    work = df.copy()
    if max_rows is not None and max_rows > 0:
        work = work.iloc[: int(max_rows)].copy()

    # Select which events trigger an "incident response" timing.
    d = work[decision_col].fillna("").astype(str).str.lower()
    if only_non_allow:
        work = work.loc[~d.eq("allow")].copy()
    if work.empty:
        rt = pd.DataFrame(columns=["intent_id", "decision", "response_time_s", "total_action_latency_s", "n_actions"])
        return rt, {"overall": latency_summary([])}

    cfg = policy_compiler_config_from_thresholds_yaml(thresholds_yaml)
    compiler = PolicyCompiler(cfg)
    orch = SimulatedOrchestrator(OrchestratorConfig())

    rows: List[Dict[str, Any]] = []
    for i, row in work.iterrows():
        intent_id = str(row.get("intent_id", f"evt_{i}"))

        intent = compiler.build_intent(
            intent_id=intent_id,
            risk=float(row.get("risk", 0.0)),
            confidence=float(row.get("confidence", 1.0)),
            rationale_tags=str(row.get("rationale_tags", "")).split(";") if pd.notna(row.get("rationale_tags", "")) else [],
            rationale_text=str(row.get("rationale_text", "")) if pd.notna(row.get("rationale_text", "")) else "",
            ts=str(row.get("ts", "")) if pd.notna(row.get("ts", "")) else None,
            principal_id=str(row.get("principal_id", "")) if pd.notna(row.get("principal_id", "")) else None,
            principal_type=str(row.get("principal_type", "")) if pd.notna(row.get("principal_type", "")) else None,
            role=str(row.get("role", "")) if pd.notna(row.get("role", "")) else None,
            session_id=str(row.get("session_id", "")) if pd.notna(row.get("session_id", "")) else None,
            token_id=str(row.get("token_id", "")) if pd.notna(row.get("token_id", "")) else None,
            device_id=str(row.get("device_id", "")) if pd.notna(row.get("device_id", "")) else None,
            workload_id=str(row.get("workload_id", "")) if pd.notna(row.get("workload_id", "")) else None,
            cloud_provider=str(row.get("cloud_provider", "")) if pd.notna(row.get("cloud_provider", "")) else None,
            tenant_id=str(row.get("tenant_id", "")) if pd.notna(row.get("tenant_id", "")) else None,
            region=str(row.get("region", "")) if pd.notna(row.get("region", "")) else None,
            resource_id=str(row.get("resource_id", "")) if pd.notna(row.get("resource_id", "")) else None,
            resource_type=str(row.get("resource_type", "")) if pd.notna(row.get("resource_type", "")) else None,
            resource_sensitivity=str(row.get("resource_sensitivity", "")) if pd.notna(row.get("resource_sensitivity", "")) else None,
        )

        # Force decision from the stream (do not re-threshold)
        forced = row.get(decision_col, row.get("effective_decision", None))
        intent.decision = _to_intent_decision(forced)

    plan = compiler.compile_plan(intent)
    lat_fn = getattr(orch, "action_latency", None)
    if callable(lat_fn):
        total_lat = float(sum(lat_fn(a.action_type) for a in plan.actions))
    else:
        # Backward-compat fallback for older orchestrator versions
        lat2 = getattr(orch, "_latency_for", None)
        if callable(lat2):
            total_lat = float(sum(lat2(a) for a in plan.actions))
        else:
            total_lat = 0.0
    
    response_time = float(extra_processing_delay_s + total_lat)

    rows.append(
        {
            "intent_id": intent_id,
            "decision": str(forced),
            "response_time_s": response_time,
            "total_action_latency_s": total_lat,
            "n_actions": int(len(plan.actions)),
        }
    )

    rt_df = pd.DataFrame(rows)
    stats = {"overall": latency_summary(rt_df["response_time_s"].tolist())}
    return rt_df, stats
