from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.afozt.orchestrator import OrchestratorConfig, SimulatedOrchestrator
from src.afozt.policy_compiler import PolicyCompiler
from src.afozt.policy_intent import IntentDecision
from src.afozt.safe_rollout import policy_compiler_config_from_thresholds_yaml
from src.common.constants import RESULTS_DIR
from src.common.logging_setup import setup_logger
from src.evaluation.latency_simulator import response_times_from_exec_events, latency_stats


def _as_dt(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    s = str(ts).strip()
    if not s or s == "nan":
        return None
    # Accept 'YYYY-MM-DD HH:MM:SS' and ISO
    try:
        return pd.to_datetime(s, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _to_intent_decision(d: Any) -> IntentDecision:
    s = str(d).strip().lower()
    if s in {"allow", "permit"}:
        return IntentDecision.ALLOW
    if s in {"stepup", "step-up", "mfa"}:
        return IntentDecision.STEPUP
    if s in {"restrict"}:
        return IntentDecision.RESTRICT
    if s in {"deny", "block"}:
        return IntentDecision.DENY
    return IntentDecision.ALLOW


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2.8 — Simulated orchestration + SOAR actions with response-time stats.")
    ap.add_argument("--decisions", default=str(RESULTS_DIR / "rollout_decisions.csv"), help="Rollout decisions CSV (from Step 2.7).")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml", help="Thresholds YAML (from tune_thresholds.py).")
    ap.add_argument("--out_exec", default=str(RESULTS_DIR / "action_executions_rollout.jsonl"), help="JSONL of executed actions.")
    ap.add_argument("--out_event_latency", default=str(RESULTS_DIR / "response_times.csv"), help="Per-intent response time CSV.")
    ap.add_argument("--out_stats", default=str(RESULTS_DIR / "latency_stats.json"), help="Latency summary JSON.")
    ap.add_argument("--include_shadow", action="store_true", help="Also execute actions for SHADOW rows (hypothetical). Default: enforced only.")
    ap.add_argument("--processing_delay_ms", type=float, default=0.0, help="Extra detection processing delay before actions start.")
    args = ap.parse_args()

    logger = setup_logger("run_orchestration", RESULTS_DIR / "run_orchestration.log")

    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions file: {decisions_path.resolve()}")

    df = pd.read_csv(decisions_path)
    logger.info(f"Loaded: {decisions_path} rows={len(df)} cols={len(df.columns)}")

    # Choose which rows to execute
    if args.include_shadow:
        df_exec = df.copy()
        df_exec["__execute__"] = True
    else:
        if "enforced" not in df.columns:
            raise ValueError("Expected column 'enforced' in rollout decisions (run Step 2.7 first).")
        df_exec = df[df["enforced"] == True].copy()  # noqa: E712
        df_exec["__execute__"] = True

    logger.info(f"Rows selected for execution: {len(df_exec)} (include_shadow={args.include_shadow})")

    # Build compiler using tuned thresholds (if present)
    cfg = policy_compiler_config_from_thresholds_yaml(args.thresholds)
    compiler = PolicyCompiler(cfg)
    orch = SimulatedOrchestrator(OrchestratorConfig())

    exec_events: List[Dict[str, Any]] = []

    # For joining later
    keep_meta_cols = [c for c in ["intent_id", "ts", "stage", "effective_decision", "intent_decision", "decision_tuned", "cloud_provider"] if c in df_exec.columns]
    meta_rows: List[Dict[str, Any]] = []

    started = datetime.utcnow()
    for idx, row in df_exec.iterrows():
        intent_id = str(row.get("intent_id", f"evt_{idx}"))

        # Build intent from row context
        intent = compiler.build_intent(
            intent_id=intent_id,
            risk=float(row.get("risk", 0.0)),
            confidence=float(row.get("confidence", 0.0)),
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

        # Force decision to match rollout effective decision (so actions align with Step 2.7)
        forced = row.get("effective_decision", row.get("decision_tuned", row.get("intent_decision", None)))
        intent.decision = _to_intent_decision(forced)

        plan = compiler.compile_plan(intent)

        start_time = _as_dt(row.get("ts"))
        if start_time is None:
            start_time = datetime.utcnow()
        if args.processing_delay_ms and args.processing_delay_ms > 0:
            start_time = start_time + pd.Timedelta(milliseconds=float(args.processing_delay_ms)).to_pytimedelta()

        events = orch.execute_plan(plan, start_time=start_time)
        # Attach detect_ts and stage for easier attribution
        stage = row.get("stage", None)
        for e in events:
            e["detect_ts"] = start_time.isoformat() + "Z"
            if stage is not None:
                e["stage"] = str(stage)
            e["effective_decision"] = str(forced) if forced is not None else None
        exec_events.extend(events)

        meta = {c: row.get(c, None) for c in keep_meta_cols}
        meta_rows.append(meta)

    out_exec = Path(args.out_exec)
    out_exec.parent.mkdir(parents=True, exist_ok=True)
    with out_exec.open("w", encoding="utf-8") as f:
        for e in exec_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info(f"✅ Wrote action execution log: {out_exec} events={len(exec_events)}")

    # Compute per-intent response time from executions
    df_latency = response_times_from_exec_events(exec_events)

    # Merge rollout metadata (stage, decision) using intent_id
    if meta_rows and not df_latency.empty:
        df_meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["intent_id"]) if "intent_id" in df_latency.columns else pd.DataFrame(meta_rows)
        df_latency = df_latency.merge(df_meta, on="intent_id", how="left")

    out_event_latency = Path(args.out_event_latency)
    out_event_latency.parent.mkdir(parents=True, exist_ok=True)
    df_latency.to_csv(out_event_latency, index=False)
    logger.info(f"✅ Wrote per-intent response times: {out_event_latency} rows={len(df_latency)}")

    # Summaries
    # Summaries
    overall = latency_stats(df_latency).get("overall", {})

    stage_breakdown: Dict[str, Any] = {}
    if "stage" in df_latency.columns:
        stage_breakdown = latency_stats(df_latency, group_cols=["stage"]).get("by_stage", {})

    decision_breakdown: Dict[str, Any] = {}
    if "effective_decision" in df_latency.columns:
        decision_breakdown = latency_stats(df_latency, group_cols=["effective_decision"]).get("by_effective_decision", {})

    stats = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "execution_events": len(exec_events),
        "intents_executed": int(df_latency["intent_id"].nunique()) if not df_latency.empty and "intent_id" in df_latency.columns else 0,
        "overall": overall,
        "by_stage": stage_breakdown,
        "by_effective_decision": decision_breakdown,
    }
    out_stats = Path(args.out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"✅ Wrote latency summary: {out_stats}")

    ended = datetime.utcnow()
    logger.info(f"Done (Step 2.8). elapsed_s={(ended-started).total_seconds():.2f}")


if __name__ == "__main__":
    main()
