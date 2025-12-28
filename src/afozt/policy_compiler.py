from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.afozt.policy_intent import ActionPlan, ActionType, CloudAction, IntentDecision, PolicyIntent


@dataclass
class PolicyCompilerConfig:
    """Thresholds for Step 2.6 policy-intent compilation.

    Default thresholds follow the mapping in the implementation plan:
    - risk < 0.30 -> allow
    - 0.30..0.60 -> step-up (MFA, shorten session, restrict scope)
    - 0.60..0.80 -> restrict + contain + alert
    - >=0.80 AND confident -> deny + revoke + quarantine + playbook
    """

    t_allow: float = 0.30
    t_stepup: float = 0.60
    t_restrict: float = 0.80
    min_conf_for_hard_actions: float = 0.60

    # Action parameters (used only for simulated adapters)
    short_session_minutes: int = 15
    restrict_scope: str = "least_privilege"
    isolate_policy: str = "microsegment"


class PolicyCompiler:
    """Compile Step 2.5 scores into cloud-agnostic PolicyIntent and provider actions."""

    def __init__(self, config: Optional[PolicyCompilerConfig] = None) -> None:
        self.cfg = config or PolicyCompilerConfig()

    def decision_from_risk(self, risk: float, confidence: float) -> IntentDecision:
        r = float(risk)
        c = float(confidence)
        if r < self.cfg.t_allow:
            return IntentDecision.ALLOW
        if r < self.cfg.t_stepup:
            return IntentDecision.STEPUP
        if r < self.cfg.t_restrict:
            return IntentDecision.RESTRICT

        # Hard deny only when confidence is sufficient; otherwise restrict
        if c >= self.cfg.min_conf_for_hard_actions:
            return IntentDecision.DENY
        return IntentDecision.RESTRICT

    def build_intent(
        self,
        intent_id: str,
        *,
        risk: float,
        confidence: float,
        rationale_tags: Optional[List[str]] = None,
        rationale_text: str = "",
        ts: Optional[str] = None,
        principal_id: Optional[str] = None,
        principal_type: Optional[str] = None,
        role: Optional[str] = None,
        session_id: Optional[str] = None,
        token_id: Optional[str] = None,
        device_id: Optional[str] = None,
        workload_id: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        tenant_id: Optional[str] = None,
        region: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_sensitivity: Optional[str] = None,
        hints: Optional[Dict[str, Any]] = None,
    ) -> PolicyIntent:
        decision = self.decision_from_risk(risk, confidence)
        tags = rationale_tags or []
        return PolicyIntent(
            intent_id=intent_id,
            ts=ts,
            principal_id=principal_id,
            principal_type=principal_type,
            role=role,
            session_id=session_id,
            token_id=token_id,
            device_id=device_id,
            workload_id=workload_id,
            cloud_provider=cloud_provider,
            tenant_id=tenant_id,
            region=region,
            resource_id=resource_id,
            resource_type=resource_type,
            resource_sensitivity=resource_sensitivity,
            risk=float(risk),
            confidence=float(confidence),
            decision=decision,
            rationale_tags=tags,
            rationale_text=rationale_text or "",
            hints=hints or {},
        )

    # ---- Provider-specific adapter strings (simulated, but realistic) ----

    def _aws_action(self, action_type: ActionType, intent: PolicyIntent) -> CloudAction:
        p: Dict[str, Any] = {
            "principal_id": intent.principal_id,
            "role": intent.role,
            "session_id": intent.session_id,
            "token_id": intent.token_id,
            "resource_id": intent.resource_id,
            "tenant_id": intent.tenant_id,
            "region": intent.region,
            "device_id": intent.device_id,
        }

        if action_type == ActionType.STEP_UP_MFA:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Enforce MFA step-up via IdP/Conditional Access for this principal",
                parameters={**p, "mfa_required": True},
            )
        if action_type == ActionType.SHORTEN_SESSION:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Issue short-lived session/token (reduce TTL)",
                parameters={**p, "session_ttl_minutes": self.cfg.short_session_minutes},
            )
        if action_type == ActionType.RESTRICT_TOKEN_SCOPE:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Apply least-privilege session policy / restrict token scope",
                parameters={**p, "scope_policy": self.cfg.restrict_scope},
            )
        if action_type == ActionType.ISOLATE_SEGMENT:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Tighten security group / NACL (micro-segmentation) for workload/device",
                parameters={**p, "isolation": self.cfg.isolate_policy},
            )
        if action_type == ActionType.REVOKE_TOKEN:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Revoke session (disable access key / revoke refresh where applicable)",
                parameters=p,
            )
        if action_type == ActionType.QUARANTINE_ENDPOINT:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Quarantine endpoint via EDR/SSM integration",
                parameters=p,
            )
        if action_type == ActionType.ALERT:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Create security alert (SOC) with context and rationale",
                parameters={**p, "rationale": intent.rationale_text},
            )
        if action_type == ActionType.SOAR_PLAYBOOK:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Trigger SOAR playbook: containment + ticket + evidence capture",
                parameters={**p, "playbook": "containment_high_risk"},
            )
        if action_type == ActionType.DENY:
            return CloudAction(
                provider="aws",
                action_type=action_type,
                description="Deny access at PDP/PEP (block request)",
                parameters=p,
            )
        # allow
        return CloudAction(provider="aws", action_type=ActionType.ALLOW, description="Allow (log decision)", parameters=p)

    def _azure_action(self, action_type: ActionType, intent: PolicyIntent) -> CloudAction:
        p: Dict[str, Any] = {
            "principal_id": intent.principal_id,
            "role": intent.role,
            "session_id": intent.session_id,
            "token_id": intent.token_id,
            "resource_id": intent.resource_id,
            "tenant_id": intent.tenant_id,
            "region": intent.region,
            "device_id": intent.device_id,
        }
        if action_type == ActionType.STEP_UP_MFA:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Enforce Conditional Access step-up MFA for this user/service principal",
                parameters={**p, "mfa_required": True},
            )
        if action_type == ActionType.SHORTEN_SESSION:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Shorten token/session lifetime (CA session controls)",
                parameters={**p, "session_ttl_minutes": self.cfg.short_session_minutes},
            )
        if action_type == ActionType.RESTRICT_TOKEN_SCOPE:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Limit token scopes / enforce least privilege via policy",
                parameters={**p, "scope_policy": self.cfg.restrict_scope},
            )
        if action_type == ActionType.ISOLATE_SEGMENT:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Isolate via NSG/micro-segmentation; restrict egress",
                parameters={**p, "isolation": self.cfg.isolate_policy},
            )
        if action_type == ActionType.REVOKE_TOKEN:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Revoke refresh tokens / sign-in sessions (Entra ID)",
                parameters=p,
            )
        if action_type == ActionType.QUARANTINE_ENDPOINT:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Quarantine endpoint via Defender for Endpoint",
                parameters=p,
            )
        if action_type == ActionType.ALERT:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Create security alert (Sentinel) with context",
                parameters={**p, "rationale": intent.rationale_text},
            )
        if action_type == ActionType.SOAR_PLAYBOOK:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Trigger SOAR playbook (Sentinel/Logic Apps)",
                parameters={**p, "playbook": "containment_high_risk"},
            )
        if action_type == ActionType.DENY:
            return CloudAction(
                provider="azure",
                action_type=action_type,
                description="Deny access (Conditional Access / policy enforcement)",
                parameters=p,
            )
        return CloudAction(provider="azure", action_type=ActionType.ALLOW, description="Allow (log decision)", parameters=p)

    def _gcp_action(self, action_type: ActionType, intent: PolicyIntent) -> CloudAction:
        p: Dict[str, Any] = {
            "principal_id": intent.principal_id,
            "role": intent.role,
            "session_id": intent.session_id,
            "token_id": intent.token_id,
            "resource_id": intent.resource_id,
            "tenant_id": intent.tenant_id,
            "region": intent.region,
            "device_id": intent.device_id,
        }
        if action_type == ActionType.STEP_UP_MFA:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Require MFA step-up (Cloud Identity / context-aware access)",
                parameters={**p, "mfa_required": True},
            )
        if action_type == ActionType.SHORTEN_SESSION:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Shorten token/session TTL (context-aware/session controls)",
                parameters={**p, "session_ttl_minutes": self.cfg.short_session_minutes},
            )
        if action_type == ActionType.RESTRICT_TOKEN_SCOPE:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Restrict OAuth scopes / reduce permissions (least privilege)",
                parameters={**p, "scope_policy": self.cfg.restrict_scope},
            )
        if action_type == ActionType.ISOLATE_SEGMENT:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Isolate via VPC firewall rules / micro-segmentation",
                parameters={**p, "isolation": self.cfg.isolate_policy},
            )
        if action_type == ActionType.REVOKE_TOKEN:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Revoke access token / disable service account key",
                parameters=p,
            )
        if action_type == ActionType.QUARANTINE_ENDPOINT:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Quarantine endpoint via EDR integration",
                parameters=p,
            )
        if action_type == ActionType.ALERT:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Create security alert (Chronicle/SCC) with context",
                parameters={**p, "rationale": intent.rationale_text},
            )
        if action_type == ActionType.SOAR_PLAYBOOK:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Trigger SOAR playbook (SCC/Chronicle)",
                parameters={**p, "playbook": "containment_high_risk"},
            )
        if action_type == ActionType.DENY:
            return CloudAction(
                provider="gcp",
                action_type=action_type,
                description="Deny access (policy enforcement)",
                parameters=p,
            )
        return CloudAction(provider="gcp", action_type=ActionType.ALLOW, description="Allow (log decision)", parameters=p)

    def compile_plan(self, intent: PolicyIntent) -> ActionPlan:
        """Compile a single intent into provider-specific actions.

        We keep actions deterministic and composable so the same intent can be
        adapted to any cloud provider (or multiple providers in later steps).
        """

        provider = (intent.cloud_provider or "").lower() or "aws"

        # Choose cloud adapter for the *current* event's provider.
        adapter = {
            "aws": self._aws_action,
            "azure": self._azure_action,
            "gcp": self._gcp_action,
        }.get(provider, self._aws_action)

        actions: List[CloudAction] = []

        if intent.decision == IntentDecision.ALLOW:
            actions.append(adapter(ActionType.ALLOW, intent))

        elif intent.decision == IntentDecision.STEPUP:
            actions.append(adapter(ActionType.STEP_UP_MFA, intent))
            actions.append(adapter(ActionType.SHORTEN_SESSION, intent))
            actions.append(adapter(ActionType.RESTRICT_TOKEN_SCOPE, intent))
            # Add an informational alert if cross-cloud hop detected
            if "cross_cloud_hop" in set(intent.rationale_tags):
                actions.append(adapter(ActionType.ALERT, intent))

        elif intent.decision == IntentDecision.RESTRICT:
            actions.append(adapter(ActionType.RESTRICT_TOKEN_SCOPE, intent))
            actions.append(adapter(ActionType.ISOLATE_SEGMENT, intent))
            actions.append(adapter(ActionType.ALERT, intent))

        else:  # DENY
            actions.append(adapter(ActionType.DENY, intent))
            actions.append(adapter(ActionType.REVOKE_TOKEN, intent))
            # Quarantine only if we have a device_id/workload_id signal
            if (intent.device_id and intent.device_id.strip()) or (intent.workload_id and intent.workload_id.strip()):
                actions.append(adapter(ActionType.QUARANTINE_ENDPOINT, intent))
            actions.append(adapter(ActionType.SOAR_PLAYBOOK, intent))

        return ActionPlan(intent=intent, actions=actions)
