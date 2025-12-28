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

from src.afozt.orchestrator import SimulatedOrchestrator
from src.afozt.policy_compiler import PolicyCompiler, PolicyCompilerConfig
from src.common.constants import RESULTS_DIR
from src.common.logging_setup import setup_logger


def _split_tags(s: Any) -> List[str]:
    if s is None:
        return []
    txt = str(s)
    if not txt or txt == "nan":
        return []
    parts = [p.strip() for p in txt.split(";")]
    return [p for p in parts if p]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.6: Compile policy intents into enforceable cloud actions.")
    parser.add_argument("--scores", type=str, default=str(RESULTS_DIR / "afozt_scores.csv"), help="Step 2.5 scored CSV.")
    parser.add_argument("--actions-out", type=str, default=str(RESULTS_DIR / "policy_actions.csv"), help="Flattened actions CSV.")
    parser.add_argument(
        "--exec-out",
        type=str,
        default=str(RESULTS_DIR / "action_executions.jsonl"),
        help="Simulated execution log (JSONL).",
    )
    parser.add_argument("--max-events", type=int, default=0, help="If >0, only process first N events.")
    parser.add_argument("--execute", action="store_true", help="Also simulate executing the compiled action plans.")
    args = parser.parse_args()

    logger = setup_logger("policy_compiler")

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Scores file not found: {scores_path}. Run Step 2.5 first (scripts/run_afozt.py)."
        )

    df = pd.read_csv(scores_path)
    if args.max_events and args.max_events > 0:
        df = df.iloc[: args.max_events].copy()

    logger.info(f"Loaded scores: {scores_path} (rows={len(df)}, cols={len(df.columns)})")

    compiler = PolicyCompiler(PolicyCompilerConfig())
    orch = SimulatedOrchestrator()

    flat_rows: List[Dict[str, Any]] = []
    exec_events: List[Dict[str, Any]] = []

    started = datetime.utcnow()

    for i, row in df.iterrows():
        risk = float(row.get("risk", 0.0))
        conf = float(row.get("confidence", 0.0))

        intent_id = f"evt_{i}"  # stable, deterministic

        intent = compiler.build_intent(
            intent_id,
            risk=risk,
            confidence=conf,
            rationale_tags=_split_tags(row.get("rationale_tags", "")),
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
            hints={"scored_decision": str(row.get("decision", ""))},
        )

        plan = compiler.compile_plan(intent)
        flat_rows.extend(plan.to_flat_rows())

        if args.execute:
            exec_events.extend(orch.execute_plan(plan))

    actions_out = Path(args.actions_out)
    actions_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat_rows).to_csv(actions_out, index=False)
    logger.info(f"✅ Wrote policy actions: {actions_out} (rows={len(flat_rows)})")

    if args.execute:
        exec_out = Path(args.exec_out)
        exec_out.parent.mkdir(parents=True, exist_ok=True)
        with exec_out.open("w", encoding="utf-8") as f:
            for e in exec_events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        summ = SimulatedOrchestrator.summarize(exec_events)
        logger.info(f"✅ Wrote execution log: {exec_out} (events={len(exec_events)})")
        logger.info(f"Execution summary: {summ}")

    ended = datetime.utcnow()
    logger.info(f"Done (Step 2.6). elapsed_s={(ended-started).total_seconds():.2f}")


if __name__ == "__main__":
    main()
