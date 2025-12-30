from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.afozt.policy_intent import IntentDecision


@dataclass
class StaticRulesConfig:
    """
    Static RBAC/ABAC-like baseline (no ML).

    Philosophy:
    - Uses only *direct* attributes (role, sensitivity, MFA, device posture) + a few obvious telemetry flags.
    - Produces rigid outcomes (allow/stepup/restrict/deny) that are typically less adaptive than AFO-ZT.

    Output contract:
    - risk in [0,1]
    - confidence in [0,1]
    - decision (and decision_tuned) in {"allow","stepup","restrict","deny"}
    - rationale_tags and rationale_text for transparency
    """

    # Device posture thresholds
    posture_restrict: float = 0.40
    posture_stepup: float = 0.60

    # Login failure thresholds (5-minute window)
    login_fail_stepup: int = 6
    login_fail_restrict: int = 10

    # Bytes-out quantiles for exfil heuristics (tuned to this synthetic dataset scale)
    bytes_p95: float = 33000.0
    bytes_p99: float = 100000.0


_SENS_RISK = {"LOW": 0.05, "MED": 0.15, "HIGH": 0.30, "CRITICAL": 0.45}
_ROLE_RISK = {"admin": 0.05, "service": 0.10, "engineer": 0.15, "employee": 0.22, "contractor": 0.28}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_sens(v: object) -> str:
    s = str(v).strip().upper()
    if s in _SENS_RISK:
        return s
    return "LOW"


def _norm_role(v: object) -> str:
    s = str(v).strip().lower()
    if s in _ROLE_RISK:
        return s
    return "employee"


class StaticRulesBaseline:
    def __init__(self, cfg: StaticRulesConfig | None = None) -> None:
        self.cfg = cfg or StaticRulesConfig()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply static rules to produce risk/confidence/decision + rationale.

        Expects (best-effort) columns from unified telemetry:
        - role, resource_sensitivity, mfa_used, posture_score, action, token_scope
        - raw_login_failures_5m, raw_time_anomaly, raw_privilege_escalation, raw_cross_cloud_hop, bytes_out
        - label_attack, attack_type (for evaluation)
        """
        out = df.copy()

        # Ensure required columns exist (best-effort defaults)
        for col, default in [
            ("role", "employee"),
            ("resource_sensitivity", "LOW"),
            ("mfa_used", 0),
            ("posture_score", 1.0),
            ("action", ""),
            ("token_scope", "NARROW"),
            ("raw_login_failures_5m", 0),
            ("raw_time_anomaly", 0),
            ("raw_privilege_escalation", 0),
            ("raw_cross_cloud_hop", 0),
            ("bytes_out", 0),
        ]:
            if col not in out.columns:
                out[col] = default

        # Normalize key fields
        role = out["role"].map(_norm_role)
        sens = out["resource_sensitivity"].map(_norm_sens)
        mfa = out["mfa_used"].fillna(0).astype(int).clip(0, 1)
        posture = pd.to_numeric(out["posture_score"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        action = out["action"].fillna("").astype(str).str.lower()
        token_scope = out["token_scope"].fillna("NARROW").astype(str).str.upper()
        login_f = pd.to_numeric(out["raw_login_failures_5m"], errors="coerce").fillna(0).astype(int)
        time_anom = pd.to_numeric(out["raw_time_anomaly"], errors="coerce").fillna(0).astype(int)
        priv_esc = pd.to_numeric(out["raw_privilege_escalation"], errors="coerce").fillna(0).astype(int)
        xcloud = pd.to_numeric(out["raw_cross_cloud_hop"], errors="coerce").fillna(0).astype(int)
        bytes_out = pd.to_numeric(out["bytes_out"], errors="coerce").fillna(0).astype(float)

        # ---------- Risk aggregation (interpretable) ----------
        base = sens.map(_SENS_RISK).astype(float) + role.map(_ROLE_RISK).astype(float)

        # MFA missing increases risk for anything above LOW sensitivity
        mfa_pen = ((mfa == 0) & sens.isin(["MED", "HIGH", "CRITICAL"])).astype(float) * 0.18

        # Device posture drift
        posture_pen = np.where(posture < self.cfg.posture_restrict, 0.28, np.where(posture < self.cfg.posture_stepup, 0.14, 0.0))

        # Simple telemetry flags (still "static rules", no learning)
        login_pen = np.where(login_f >= self.cfg.login_fail_restrict, 0.22, np.where(login_f >= self.cfg.login_fail_stepup, 0.12, 0.0))
        time_pen = (time_anom == 1).astype(float) * 0.22
        priv_pen = (priv_esc == 1).astype(float) * 0.30
        xcloud_pen = (xcloud == 1).astype(float) * 0.16

        # Exfil heuristic (bytes_out)
        exfil_pen = np.where(bytes_out >= self.cfg.bytes_p99, 0.32, np.where(bytes_out >= self.cfg.bytes_p95, 0.16, 0.0))

        # High-risk actions by non-admins
        highrisk_act = action.isin(["delete", "admin_change"]).astype(int)
        act_pen = np.where((highrisk_act == 1) & (role != "admin"), 0.38, np.where(highrisk_act == 1, 0.08, 0.0))

        # Wide token scopes add risk unless admin/service
        scope_pen = np.where((token_scope == "WIDE") & (~role.isin(["admin", "service"])), 0.10, 0.0)

        s_rule = base + mfa_pen + posture_pen + login_pen + time_pen + priv_pen + xcloud_pen + exfil_pen + act_pen + scope_pen
        s_rule = s_rule.clip(0.0, 1.0)

        # ---------- Decision rules (RBAC/ABAC style) ----------
        decisions: List[str] = []
        confidences: List[float] = []
        tags_list: List[str] = []
        texts: List[str] = []

        for i in range(len(out)):
            tags: List[str] = []

            r = role.iat[i]
            s = sens.iat[i]
            mf = int(mfa.iat[i])
            ps = float(posture.iat[i])
            lf = int(login_f.iat[i])
            ta = int(time_anom.iat[i])
            pe = int(priv_esc.iat[i])
            xc = int(xcloud.iat[i])
            bo = float(bytes_out.iat[i])
            act = action.iat[i]
            sc = token_scope.iat[i]

            # Hard policy-like rules
            hard_deny = False
            hard_restrict = False
            need_stepup = False

            # Privileged operations by non-admin
            if act in ("admin_change", "delete") and r != "admin":
                hard_deny = True
                tags.append("rbac_privileged_op_non_admin")

            # CRITICAL resources: only admin/service; others need MFA at minimum
            if s == "CRITICAL" and r not in ("admin", "service"):
                tags.append("abac_critical_resource")
                if mf == 0 or ps < 0.50:
                    hard_deny = True
                    tags.append("abac_critical_no_mfa_or_low_posture")
                else:
                    hard_restrict = True
                    tags.append("abac_critical_restrict_non_privileged")

            # Privilege escalation indicator
            if pe == 1:
                tags.append("signal_privilege_escalation")
                if s in ("HIGH", "CRITICAL"):
                    hard_restrict = True
                else:
                    need_stepup = True

            # Impossible travel / time anomaly
            if ta == 1:
                tags.append("signal_impossible_travel")
                hard_restrict = True

            # Too many failures
            if lf >= self.cfg.login_fail_restrict:
                tags.append("signal_bruteforce_high")
                hard_restrict = True
            elif lf >= self.cfg.login_fail_stepup:
                tags.append("signal_bruteforce_medium")
                need_stepup = True

            # Device posture
            if ps < self.cfg.posture_restrict:
                tags.append("device_low_posture")
                hard_restrict = True
            elif ps < self.cfg.posture_stepup:
                tags.append("device_medium_posture")
                need_stepup = True

            # MFA missing for sensitive access
            if mf == 0 and s in ("HIGH", "CRITICAL"):
                tags.append("mfa_missing_sensitive")
                need_stepup = True

            # Cross-cloud hop: restrict for normal users (baseline lacks graph correlation)
            if xc == 1 and r not in ("admin", "service"):
                tags.append("signal_cross_cloud_hop")
                need_stepup = True

            # Exfil bytes
            if bo >= self.cfg.bytes_p99:
                tags.append("signal_exfil_spike")
                hard_restrict = True
            elif bo >= self.cfg.bytes_p95:
                tags.append("signal_exfil_medium")
                need_stepup = True

            # Wide scope token + sensitive
            if sc == "WIDE" and s in ("HIGH", "CRITICAL") and r not in ("admin", "service"):
                tags.append("token_scope_wide_sensitive")
                need_stepup = True

            # Resolve final decision
            if hard_deny:
                dec = IntentDecision.DENY.value
                conf = 0.90
            elif hard_restrict:
                dec = IntentDecision.RESTRICT.value
                conf = 0.80
            elif need_stepup:
                dec = IntentDecision.STEPUP.value
                conf = 0.65
            else:
                dec = IntentDecision.ALLOW.value
                conf = 0.60

            decisions.append(dec)
            confidences.append(float(conf))
            tags_list.append(";".join(sorted(set(tags))) if tags else "")
            texts.append("Static rules triggered: " + (", ".join(sorted(set(tags))) if tags else "none"))

        out["s_rule"] = s_rule.astype(float)
        out["risk"] = out["s_rule"].astype(float)  # baseline has only rule-score
        out["confidence"] = pd.Series(confidences, index=out.index).astype(float).clip(0.0, 1.0)
        out["decision"] = pd.Series(decisions, index=out.index).astype(str)
        out["decision_tuned"] = out["decision"]  # static baseline doesn't tune thresholds
        out["rationale_tags"] = pd.Series(tags_list, index=out.index).astype(str)
        out["rationale_text"] = pd.Series(texts, index=out.index).astype(str)

        return out
