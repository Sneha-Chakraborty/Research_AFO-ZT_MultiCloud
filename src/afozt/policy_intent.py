from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentDecision(str, Enum):
    ALLOW = "allow"
    STEPUP = "stepup"
    RESTRICT = "restrict"
    DENY = "deny"


class ActionType(str, Enum):
    # Identity / access
    ALLOW = "allow"
    STEP_UP_MFA = "step_up_mfa"
    SHORTEN_SESSION = "shorten_session"
    RESTRICT_TOKEN_SCOPE = "restrict_token_scope"

    # Containment / response
    ISOLATE_SEGMENT = "isolate_segment"
    ALERT = "alert"
    DENY = "deny"
    REVOKE_TOKEN = "revoke_token"
    QUARANTINE_ENDPOINT = "quarantine_endpoint"
    SOAR_PLAYBOOK = "soar_playbook"


@dataclass
class PolicyIntent:
    """Cloud-agnostic intent produced from (risk, confidence, rationale).

    Step 2.6 goal: convert Step 2.5 scoring outputs into an enforceable,
    cloud-adapted action plan.
    """

    intent_id: str
    ts: Optional[str] = None

    # Identity/session context
    principal_id: Optional[str] = None
    principal_type: Optional[str] = None
    role: Optional[str] = None
    session_id: Optional[str] = None
    token_id: Optional[str] = None
    device_id: Optional[str] = None
    workload_id: Optional[str] = None

    # Target context
    cloud_provider: Optional[str] = None
    tenant_id: Optional[str] = None
    region: Optional[str] = None
    resource_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_sensitivity: Optional[str] = None

    # Decision inputs
    risk: float = 0.0
    confidence: float = 0.0
    decision: IntentDecision = IntentDecision.ALLOW

    # Human outputs
    rationale_tags: List[str] = field(default_factory=list)
    rationale_text: str = ""

    # Optional additional policy hints (e.g., budgets, cohort for canary)
    hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudAction:
    """A concrete provider-specific action generated from an intent."""

    provider: str
    action_type: ActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionPlan:
    """Cloud-adapted plan: one intent -> many concrete actions."""

    intent: PolicyIntent
    actions: List[CloudAction] = field(default_factory=list)

    def to_flat_rows(self) -> List[Dict[str, Any]]:
        """Flatten for CSV/JSONL export."""
        rows: List[Dict[str, Any]] = []
        for a in self.actions:
            rows.append(
                {
                    "intent_id": self.intent.intent_id,
                    "ts": self.intent.ts,
                    "principal_id": self.intent.principal_id,
                    "session_id": self.intent.session_id,
                    "token_id": self.intent.token_id,
                    "device_id": self.intent.device_id,
                    "cloud_provider": self.intent.cloud_provider,
                    "tenant_id": self.intent.tenant_id,
                    "region": self.intent.region,
                    "resource_id": self.intent.resource_id,
                    "risk": self.intent.risk,
                    "confidence": self.intent.confidence,
                    "decision": self.intent.decision.value,
                    "rationale_tags": ";".join(self.intent.rationale_tags),
                    "action_provider": a.provider,
                    "action_type": a.action_type.value,
                    "action_description": a.description,
                    "action_parameters": str(a.parameters),
                }
            )
        return rows
