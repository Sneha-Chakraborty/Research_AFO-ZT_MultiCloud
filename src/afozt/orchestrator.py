from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.afozt.policy_intent import ActionPlan, ActionType, CloudAction


@dataclass
class OrchestratorConfig:
    """Simulated execution latencies (seconds) for Step 2.6/2.8."""

    # Identity controls
    step_up_mfa_s: float = 0.2
    shorten_session_s: float = 0.15
    restrict_scope_s: float = 0.25

    # Containment
    isolate_segment_s: float = 1.0
    revoke_token_s: float = 0.6
    quarantine_endpoint_s: float = 2.0

    # Coordination
    alert_s: float = 0.1
    soar_playbook_s: float = 1.5
    deny_s: float = 0.05


class SimulatedOrchestrator:
    """Simulate executing cloud actions and emit an execution log.

    Why this exists already in Step 2.6:
    - You can validate your policy mapping without real cloud credentials.
    - Later (Step 2.8), you can swap this with real adapters.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.cfg = config or OrchestratorConfig()


def action_latency(self, action_type: ActionType | str) -> float:
    """Return simulated latency (seconds) for a single action type.

    This is used by evaluation/response_time.py to estimate response times
    without executing orchestration.
    """
    # Normalize to ActionType
    t: ActionType
    try:
        if isinstance(action_type, ActionType):
            t = action_type
        else:
            s = str(action_type)
            try:
                t = ActionType(s)          # matches enum values like "deny"
            except Exception:
                t = ActionType[s]          # matches enum names like "DENY"
    except Exception:
        return 0.05

    if t == ActionType.STEP_UP_MFA:
        return float(self.cfg.step_up_mfa_s)
    if t == ActionType.SHORTEN_SESSION:
        return float(self.cfg.shorten_session_s)
    if t == ActionType.RESTRICT_TOKEN_SCOPE:
        return float(self.cfg.restrict_scope_s)
    if t == ActionType.ISOLATE_SEGMENT:
        return float(self.cfg.isolate_segment_s)
    if t == ActionType.REVOKE_TOKEN:
        return float(self.cfg.revoke_token_s)
    if t == ActionType.QUARANTINE_ENDPOINT:
        return float(self.cfg.quarantine_endpoint_s)
    if t == ActionType.ALERT:
        return float(self.cfg.alert_s)
    if t == ActionType.SOAR_PLAYBOOK:
        return float(self.cfg.soar_playbook_s)
    if t == ActionType.DENY:
        return float(self.cfg.deny_s)
    if t == ActionType.ALLOW:
        return 0.0
    return 0.05
    def _latency_for(self, a: CloudAction) -> float:
        t = a.action_type
        if t == ActionType.STEP_UP_MFA:
            return self.cfg.step_up_mfa_s
        if t == ActionType.SHORTEN_SESSION:
            return self.cfg.shorten_session_s
        if t == ActionType.RESTRICT_TOKEN_SCOPE:
            return self.cfg.restrict_scope_s
        if t == ActionType.ISOLATE_SEGMENT:
            return self.cfg.isolate_segment_s
        if t == ActionType.REVOKE_TOKEN:
            return self.cfg.revoke_token_s
        if t == ActionType.QUARANTINE_ENDPOINT:
            return self.cfg.quarantine_endpoint_s
        if t == ActionType.ALERT:
            return self.cfg.alert_s
        if t == ActionType.SOAR_PLAYBOOK:
            return self.cfg.soar_playbook_s
        if t == ActionType.DENY:
            return self.cfg.deny_s
        return 0.05

    def execute_plan(
        self,
        plan: ActionPlan,
        *,
        start_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Execute actions sequentially (simulated).

        Returns a list of action execution events (dicts) suitable for JSONL/CSV.
        """
        t0 = start_time or datetime.utcnow()
        now = t0
        events: List[Dict[str, Any]] = []

        for idx, a in enumerate(plan.actions):
            dt = self._latency_for(a)
            # In a real system, you would call the cloud adapter here.
            # For now, we assume success.
            started = now
            finished = now + timedelta(seconds=float(dt))
            now = finished

            events.append(
                {
                    "intent_id": plan.intent.intent_id,
                    "action_index": idx,
                    "provider": a.provider,
                    "action_type": a.action_type.value,
                    "description": a.description,
                    "parameters": a.parameters,
                    "started_at": started.isoformat() + "Z",
                    "finished_at": finished.isoformat() + "Z",
                    "latency_s": float(dt),
                    "status": "success",
                }
            )

        return events

    @staticmethod
    def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return {"actions": 0, "total_latency_s": 0.0}
        total = float(sum(float(e.get("latency_s", 0.0)) for e in events))
        return {
            "actions": len(events),
            "total_latency_s": total,
            "providers": sorted(list({str(e.get("provider")) for e in events})),
            "action_types": sorted(list({str(e.get("action_type")) for e in events})),
        }
