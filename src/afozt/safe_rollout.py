from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.afozt.policy_compiler import PolicyCompiler, PolicyCompilerConfig
from src.common.logging_setup import setup_logger


class RolloutStage(str, Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    ENFORCE = "enforce"
    ROLLOUT = "rollout"


@dataclass(frozen=True)
class RolloutBudgets:
    """Safety budgets that can trigger rollback."""
    false_deny_budget: float = 0.02
    mfa_friction_budget: float = 0.10
    latency_p95_budget_ms: float = 600.0
    error_rate_budget: float = 0.01
    analyst_override_budget: float = 0.10


@dataclass
class RolloutConfig:
    """Step 2.7 — Safe rollout controller: shadow → canary → enforce → rollout.

    Model:
    - Shadow: compute intent decisions but don't enforce; evaluate KPIs hypothetically.
    - Canary: enforce on a small deterministic cohort.
    - Enforce: enforce on a selected scope (cloud/tenant/region).
    - Rollout: enforce globally.

    Rollback triggers:
    false-deny, MFA friction, simulated latency/error, analyst override rate.
    """

    # Stage windows
    shadow_days: int = 7
    canary_days: int = 3
    enforce_days: int = 3

    # Canary selection
    canary_fraction: float = 0.05
    cohort_key: str = "session_id"
    seed: int = 42

    # Enforce scope (for ENFORCE stage)
    enforce_cloud_provider: Optional[str] = "aws"
    enforce_tenant_id: Optional[str] = None
    enforce_region: Optional[str] = None

    # Override model
    override_conf_threshold: float = 0.55

    # IMPORTANT: must be default_factory (no mutable defaults in dataclasses)
    budgets: RolloutBudgets = field(default_factory=RolloutBudgets)

    @staticmethod
    def from_yaml(path: str) -> "RolloutConfig":
        """Load rollout config from YAML.

        Supports either:
        - budgets keys at top-level
        - or nested under "budgets:"
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        raw_budgets = raw.get("budgets", raw) or {}

        budgets = RolloutBudgets(
            false_deny_budget=float(raw_budgets.get("false_deny_budget", 0.02)),
            mfa_friction_budget=float(raw_budgets.get("mfa_friction_budget", 0.10)),
            latency_p95_budget_ms=float(raw_budgets.get("latency_p95_budget_ms", 600.0)),
            error_rate_budget=float(raw_budgets.get("error_rate_budget", 0.01)),
            analyst_override_budget=float(raw_budgets.get("analyst_override_budget", 0.10)),
        )

        return RolloutConfig(
            shadow_days=int(raw.get("shadow_days", 7)),
            canary_days=int(raw.get("canary_days", 3)),
            enforce_days=int(raw.get("enforce_days", 3)),
            canary_fraction=float(raw.get("canary_fraction", 0.05)),
            cohort_key=str(raw.get("cohort_key", "session_id")),
            seed=int(raw.get("seed", 42)),
            enforce_cloud_provider=raw.get("enforce_cloud_provider", raw.get("enforce_cloud", "aws")),
            enforce_tenant_id=raw.get("enforce_tenant_id", raw.get("enforce_tenant")),
            enforce_region=raw.get("enforce_region", None),
            override_conf_threshold=float(raw.get("override_conf_threshold", 0.55)),
            budgets=budgets,
        )


def policy_compiler_config_from_thresholds_yaml(path: str) -> PolicyCompilerConfig:
    """Map configs/thresholds.yaml into PolicyCompilerConfig (optional)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PolicyCompilerConfig(
        t_allow=float(raw.get("allow_lt", 0.30)),
        t_stepup=float(raw.get("stepup_lt", 0.60)),
        t_restrict=float(raw.get("restrict_lt", 0.80)),
        min_conf_for_hard_actions=float(raw.get("min_conf_for_hard_actions", 0.60)),
        short_session_minutes=int(raw.get("short_session_minutes", 15)),
        restrict_scope=str(raw.get("restrict_scope", "least_privilege")),
        isolate_policy=str(raw.get("isolate_policy", "microsegment")),
    )


def _stable_hash_to_unit_interval(text: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return x / float(16**8 - 1)


def _pick_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["ts", "timestamp", "event_time", "time"]:
        if c in df.columns:
            return c
    return None


def _parse_ts_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", utc=False)
    if out.notna().any():
        return out

    # numeric epoch seconds/ms fallback
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().any():
        med = float(x.dropna().median())
        unit = "ms" if med > 1e12 else "s"
        return pd.to_datetime(x, unit=unit, errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def _select_scope_mask(
    df: pd.DataFrame,
    *,
    cloud_provider: Optional[str],
    tenant_id: Optional[str],
    region: Optional[str],
) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if cloud_provider and "cloud_provider" in df.columns:
        m &= df["cloud_provider"].astype(str).str.lower().eq(str(cloud_provider).lower())
    if tenant_id and "tenant_id" in df.columns:
        m &= df["tenant_id"].astype(str).eq(str(tenant_id))
    if region and "region" in df.columns:
        m &= df["region"].astype(str).eq(str(region))
    return m


def _simulated_policy_overhead_ms(decision: str) -> float:
    if decision == "allow":
        return 0.0
    if decision == "stepup":
        return 150.0
    if decision == "restrict":
        return 300.0
    if decision == "deny":
        return 50.0
    return 0.0


def _simulated_policy_error_prob(decision: str) -> float:
    if decision == "allow":
        return 0.0002
    if decision == "stepup":
        return 0.0030
    if decision == "restrict":
        return 0.0050
    if decision == "deny":
        return 0.0
    return 0.0002


def compute_rollout_kpis(
    df: pd.DataFrame,
    *,
    effective_decision_col: str,
    enforced_col: str,
    label_attack_col: str = "label_attack",
    confidence_col: str = "confidence",
    latency_col: str = "latency_ms",
    override_conf_threshold: float = 0.55,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute KPIs for a given slice (typically one stage window)."""
    if df.empty:
        return {
            "n_events": 0,
            "enforced_fraction": 0.0,
            "false_deny_rate": 0.0,
            "mfa_friction_rate": 0.0,
            "latency_p95_ms": 0.0,
            "policy_error_rate": 0.0,
            "analyst_override_rate": 0.0,
        }

    enforced = df[enforced_col].astype(bool)
    enforced_df = df.loc[enforced].copy()
    n = int(len(df))
    n_enf = int(len(enforced_df))

    if n_enf == 0:
        return {
            "n_events": n,
            "enforced_fraction": 0.0,
            "false_deny_rate": 0.0,
            "mfa_friction_rate": 0.0,
            "latency_p95_ms": 0.0,
            "policy_error_rate": 0.0,
            "analyst_override_rate": 0.0,
        }

    # false-deny: benign events that were denied
    if label_attack_col in enforced_df.columns:
        benign = (pd.to_numeric(enforced_df[label_attack_col], errors="coerce").fillna(0).astype(int) == 0)
    else:
        benign = pd.Series(True, index=enforced_df.index)

    eff = enforced_df[effective_decision_col].astype(str)

    false_deny = benign & eff.eq("deny")
    false_deny_rate = float(false_deny.mean())

    # MFA friction
    mfa_friction_rate = float(eff.eq("stepup").mean())

    # Latency p95 (FIXED: if latency_col missing, create a per-row Series)
    if latency_col in enforced_df.columns:
        base_latency = pd.to_numeric(enforced_df[latency_col], errors="coerce").fillna(100.0).astype(float)
    else:
        base_latency = pd.Series(100.0, index=enforced_df.index, dtype=float)

    overhead = eff.map(_simulated_policy_overhead_ms).astype(float)
    latency_effective = base_latency + overhead
    latency_p95 = float(np.percentile(latency_effective.to_numpy(), 95))

    # Simulated policy error rate (unexpected errors)
    probs = eff.map(_simulated_policy_error_prob).astype(float).to_numpy()
    draws = np.array([_stable_hash_to_unit_interval(str(idx), seed=seed) for idx in enforced_df.index])
    policy_error_rate = float((draws < probs).mean())

    # Analyst override: severe action but low confidence
    conf = pd.to_numeric(enforced_df.get(confidence_col, 1.0), errors="coerce")
    if not isinstance(conf, pd.Series):
        conf = pd.Series(conf, index=enforced_df.index, dtype=float)
    conf = conf.fillna(1.0).astype(float)

    severe = eff.isin(["restrict", "deny"])
    overrides = severe & (conf < float(override_conf_threshold))
    analyst_override_rate = float(overrides.mean())

    return {
        "n_events": n,
        "enforced_fraction": float(n_enf / n) if n else 0.0,
        "false_deny_rate": false_deny_rate,
        "mfa_friction_rate": mfa_friction_rate,
        "latency_p95_ms": latency_p95,
        "policy_error_rate": policy_error_rate,
        "analyst_override_rate": analyst_override_rate,
    }


def budgets_ok(kpis: Dict[str, float], budgets: RolloutBudgets) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if kpis["false_deny_rate"] > budgets.false_deny_budget:
        reasons.append(f"false-deny {kpis['false_deny_rate']:.4f} > {budgets.false_deny_budget:.4f}")
    if kpis["mfa_friction_rate"] > budgets.mfa_friction_budget:
        reasons.append(f"MFA friction {kpis['mfa_friction_rate']:.4f} > {budgets.mfa_friction_budget:.4f}")
    if kpis["latency_p95_ms"] > budgets.latency_p95_budget_ms:
        reasons.append(f"p95 latency {kpis['latency_p95_ms']:.1f}ms > {budgets.latency_p95_budget_ms:.1f}ms")
    if kpis["policy_error_rate"] > budgets.error_rate_budget:
        reasons.append(f"policy error {kpis['policy_error_rate']:.4f} > {budgets.error_rate_budget:.4f}")
    if kpis["analyst_override_rate"] > budgets.analyst_override_budget:
        reasons.append(f"override {kpis['analyst_override_rate']:.4f} > {budgets.analyst_override_budget:.4f}")
    return (len(reasons) == 0, reasons)


class SafeRolloutController:
    """Offline simulator for Step 2.7 safe rollout controller."""

    def __init__(self, cfg: RolloutConfig, policy_cfg: Optional[PolicyCompilerConfig] = None) -> None:
        self.cfg = cfg
        self.policy_compiler = PolicyCompiler(policy_cfg or PolicyCompilerConfig())
        self.logger = setup_logger("safe_rollout")

    def _parse_tags(self, x: Any) -> List[str]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, list):
            return [str(t) for t in x]
        s = str(x).strip()
        if not s:
            return []

        # JSON list string?
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(t) for t in arr]
            except Exception:
                # fallback: simple parsing
                s2 = s.strip("[]").replace("'", "").replace('"', "")
                return [p.strip() for p in s2.split(",") if p.strip()]

        if "|" in s:
            return [p.strip() for p in s.split("|") if p.strip()]
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]

    def _add_intent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Pre-create columns for speed/stability
        out["intent_id"] = ""
        out["intent_decision"] = ""
        out["action_count"] = 0
        out["action_types"] = ""

        for i, row in out.iterrows():
            intent_id = f"intent_{i}"
            intent = self.policy_compiler.build_intent(
                intent_id,
                risk=float(row.get("risk", 0.0)),
                confidence=float(row.get("confidence", 1.0)),
                rationale_tags=self._parse_tags(row.get("rationale_tags", [])),
                rationale_text=str(row.get("rationale_text", "")),
                ts=str(row.get("ts", "")) if "ts" in out.columns else None,
                principal_id=str(row.get("principal_id", "")) if "principal_id" in out.columns else None,
                principal_type=str(row.get("principal_type", "")) if "principal_type" in out.columns else None,
                role=str(row.get("role", "")) if "role" in out.columns else None,
                session_id=str(row.get("session_id", "")) if "session_id" in out.columns else None,
                token_id=str(row.get("token_id", "")) if "token_id" in out.columns else None,
                device_id=str(row.get("device_id", "")) if "device_id" in out.columns else None,
                workload_id=str(row.get("workload_id", "")) if "workload_id" in out.columns else None,
                cloud_provider=str(row.get("cloud_provider", "")) if "cloud_provider" in out.columns else None,
                tenant_id=str(row.get("tenant_id", "")) if "tenant_id" in out.columns else None,
                region=str(row.get("region", "")) if "region" in out.columns else None,
                resource_id=str(row.get("resource_id", "")) if "resource_id" in out.columns else None,
                resource_type=str(row.get("resource_type", "")) if "resource_type" in out.columns else None,
                resource_sensitivity=str(row.get("resource_sensitivity", "")) if "resource_sensitivity" in out.columns else None,
                hints=None,
            )
            plan = self.policy_compiler.compile_plan(intent)

            out.at[i, "intent_id"] = intent.intent_id
            out.at[i, "intent_decision"] = intent.decision.value
            out.at[i, "action_count"] = len(plan.actions)
            out.at[i, "action_types"] = "|".join([a.action_type.value for a in plan.actions])

        return out

    def _canary_mask(self, df: pd.DataFrame) -> pd.Series:
        key = self.cfg.cohort_key
        series = df[key].astype(str).fillna("") if key in df.columns else df.index.astype(str)
        vals = series.map(lambda x: _stable_hash_to_unit_interval(x, seed=self.cfg.seed))
        return vals < float(self.cfg.canary_fraction)

    def run(self, scored_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Run the rollout simulation over the scored dataset."""
        df = scored_df.copy()

        # Sort by timestamp if possible
        ts_col = _pick_timestamp_column(df)
        if ts_col:
            df["_ts_parsed"] = _parse_ts_series(df[ts_col])
            df = df.sort_values(by="_ts_parsed", kind="stable")
        else:
            df["_ts_parsed"] = pd.NaT

        df = df.reset_index(drop=True)
        df = self._add_intent_columns(df)

        enforce_scope = _select_scope_mask(
            df,
            cloud_provider=self.cfg.enforce_cloud_provider,
            tenant_id=self.cfg.enforce_tenant_id,
            region=self.cfg.enforce_region,
        )
        canary_mask = self._canary_mask(df)

        stage_order = [RolloutStage.SHADOW, RolloutStage.CANARY, RolloutStage.ENFORCE, RolloutStage.ROLLOUT]
        stage_days_map = {
            RolloutStage.SHADOW: self.cfg.shadow_days,
            RolloutStage.CANARY: self.cfg.canary_days,
            RolloutStage.ENFORCE: self.cfg.enforce_days,
            RolloutStage.ROLLOUT: 10_000,  # rest of dataset
        }

        def _slice_window(start_idx: int, stage_days: int) -> Tuple[pd.DataFrame, int]:
            if start_idx >= len(df):
                return df.iloc[0:0].copy(), start_idx

            if df["_ts_parsed"].notna().any():
                start_t = df.loc[start_idx, "_ts_parsed"]
                if pd.isna(start_t):
                    end_idx = min(len(df), start_idx + stage_days * 2000)
                    return df.iloc[start_idx:end_idx].copy(), end_idx

                end_t = start_t + pd.Timedelta(days=int(stage_days))
                sub = df.iloc[start_idx:]
                mask = sub["_ts_parsed"].isna() | (sub["_ts_parsed"] < end_t)
                win = sub.loc[mask]
                end_idx = int(win.index.max() + 1) if len(win) else start_idx
                return df.iloc[start_idx:end_idx].copy(), end_idx

            end_idx = min(len(df), start_idx + stage_days * 2000)
            return df.iloc[start_idx:end_idx].copy(), end_idx

        timeline_rows: List[Dict[str, Any]] = []
        all_decisions: List[pd.DataFrame] = []
        metrics: Dict[str, Any] = {}

        cursor = 0
        current_stage_idx = 0

        while cursor < len(df) and current_stage_idx < len(stage_order):
            stage = stage_order[current_stage_idx]
            days = stage_days_map[stage]

            win, next_cursor = _slice_window(cursor, days)
            if stage == RolloutStage.ROLLOUT:
                win = df.iloc[cursor:].copy()
                next_cursor = len(df)

            if win.empty:
                break

            idxs = win.index
            enforced = pd.Series(False, index=idxs)

            if stage == RolloutStage.SHADOW:
                enforced[:] = False
            elif stage == RolloutStage.CANARY:
                enforced[:] = canary_mask.loc[idxs].fillna(False).to_numpy()
            elif stage == RolloutStage.ENFORCE:
                enforced[:] = enforce_scope.loc[idxs].fillna(False).to_numpy()
            elif stage == RolloutStage.ROLLOUT:
                enforced[:] = True

            win["stage"] = stage.value
            win["enforced"] = enforced.astype(bool)

            # Effective decision: enforced intent decision, else allow (shadow behavior)
            win["effective_decision"] = np.where(win["enforced"], win["intent_decision"], "allow")

            # KPI evaluation: in SHADOW evaluate hypothetically (treat as enforced)
            eval_df = win.copy()
            if stage == RolloutStage.SHADOW:
                eval_df["enforced"] = True
                eval_df["effective_decision"] = eval_df["intent_decision"]

            kpis = compute_rollout_kpis(
                eval_df,
                effective_decision_col="effective_decision",
                enforced_col="enforced",
                override_conf_threshold=self.cfg.override_conf_threshold,
                seed=self.cfg.seed,
            )
            ok, reasons = budgets_ok(kpis, self.cfg.budgets)

            timeline_rows.append(
                {
                    "stage": stage.value,
                    "start_index": int(cursor),
                    "end_index": int(next_cursor),
                    "n_events": int(len(win)),
                    "budgets_ok": bool(ok),
                    "reasons": "; ".join(reasons),
                }
            )
            metrics[stage.value] = kpis
            all_decisions.append(win)

            if ok:
                cursor = next_cursor
                current_stage_idx += 1
            else:
                self.logger.warning(f"Rollback triggered in stage={stage.value}: {reasons}")
                rest = df.iloc[next_cursor:].copy()
                if not rest.empty:
                    rest["stage"] = RolloutStage.SHADOW.value
                    rest["enforced"] = False
                    rest["effective_decision"] = "allow"
                    all_decisions.append(rest)
                break

        timeline_df = pd.DataFrame(timeline_rows)
        decisions_df = pd.concat(all_decisions, ignore_index=True) if all_decisions else df.iloc[0:0].copy()

        decisions_df = decisions_df.drop(columns=["_ts_parsed"], errors="ignore")
        return timeline_df, decisions_df, metrics
