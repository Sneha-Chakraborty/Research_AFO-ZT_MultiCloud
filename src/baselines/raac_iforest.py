from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import IsolationForest

from src.afozt.policy_intent import IntentDecision


def _to_float(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.fillna(default).astype(float)


def _to_int01(series: pd.Series, default: int = 0) -> pd.Series:
    # Accept booleans, 0/1, 'true/false', etc.
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    s = s.fillna(default)
    out = s.astype(str).str.strip().str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0})
    return out.fillna(pd.to_numeric(s, errors="coerce")).fillna(default).astype(int).clip(0, 1)


def _scope_width(scope: object) -> int:
    if scope is None or (isinstance(scope, float) and np.isnan(scope)):  # type: ignore[arg-type]
        return 0
    s = str(scope).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return 0
    # handle comma/space/semicolon separated scopes
    parts = [p for p in re.split(r"[;, \t]+", s) if p]
    return len(parts)


def _sens_score(sens: object) -> float:
    s = "" if sens is None else str(sens).strip().lower()
    if s in {"critical", "secret"}:
        return 1.0
    if s in {"high"}:
        return 0.75
    if s in {"medium", "med"}:
        return 0.45
    if s in {"low"}:
        return 0.15
    # unknown
    return 0.25


@dataclass
class RaacIForestConfig:
    """Base-paper style baseline: RAAC + Isolation Forest + SOAR mapping.

    - Computes contextual risk (interpretable, threshold-style)
    - Computes behavioral risk via Isolation Forest anomaly score
    - Fuses them into a single RAAC risk score
    - Maps risk -> 4-class ZT decision (allow/stepup/restrict/deny)

    Important: This baseline intentionally does NOT use trust-graph features or rollout logic.
    """

    # Risk fusion
    alpha_context: float = 0.55
    beta_behavior: float = 0.45

    # RAAC thresholds (4-class mapping)
    t_allow: float = 0.35
    t_stepup: float = 0.55
    t_deny: float = 0.75

    # Contextual risk weights
    w_posture: float = 0.30
    w_mfa_missing: float = 0.20
    w_time_anomaly: float = 0.15
    w_priv_escalation: float = 0.20
    w_sensitive_resource: float = 0.15

    posture_low: float = 0.50  # below this, posture contributes risk

    # Isolation Forest
    iforest_n_estimators: int = 160
    iforest_contamination: float = 0.06
    iforest_max_samples: str | int = "auto"
    random_state: int = 42

    # Robust scaling for anomaly -> [0,1]
    anom_q_low: float = 0.05
    anom_q_high: float = 0.95

    # Feature columns used for Isolation Forest (must be numeric)
    # If missing, filled with 0.
    iforest_feature_cols: Tuple[str, ...] = (
        "raw_login_failures_5m",
        "raw_api_calls_5m",
        "bytes_out",
        "token_age",
        "posture_score",
        "latency_ms",
    )


class RaacIForestBaseline:
    def __init__(self, cfg: RaacIForestConfig | None = None) -> None:
        self.cfg = cfg or RaacIForestConfig()
        # runtime learned
        self.model: Optional[IsolationForest] = None
        self._anom_low: float = 0.0
        self._anom_high: float = 1.0
        self._feature_cols: Tuple[str, ...] = tuple(self.cfg.iforest_feature_cols)

    @property
    def artifact(self) -> dict:
        return {
            "baseline": "raac_iforest",
            "config": asdict(self.cfg),
            "feature_cols": list(self._feature_cols),
            "derived": {"anom_low": float(self._anom_low), "anom_high": float(self._anom_high)},
            "model": self.model,
        }

    def fit(self, df_train: pd.DataFrame) -> "RaacIForestBaseline":
        """Fit Isolation Forest on (mostly) benign behaviour."""
        X = self._make_X(df_train)
        # try to fit mostly on benign if available
        if "label_attack" in df_train.columns:
            y = _to_int01(df_train["label_attack"], default=0)
            benign_mask = y == 0
            if benign_mask.sum() >= max(200, int(0.1 * len(df_train))):
                X_fit = X.loc[benign_mask.values]
            else:
                X_fit = X
        else:
            X_fit = X

        self.model = IsolationForest(
            n_estimators=int(self.cfg.iforest_n_estimators),
            contamination=float(self.cfg.iforest_contamination),
            max_samples=self.cfg.iforest_max_samples,
            random_state=int(self.cfg.random_state),
        )
        self.model.fit(X_fit)

        # robust scaling params from train scores
        raw = self._raw_anom_score(X_fit)
        ql = float(np.quantile(raw, self.cfg.anom_q_low)) if len(raw) else 0.0
        qh = float(np.quantile(raw, self.cfg.anom_q_high)) if len(raw) else 1.0
        if qh <= ql:
            qh = ql + 1e-6
        self._anom_low, self._anom_high = ql, qh
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score dataframe and return decisions and rationale.

        Expects unified telemetry columns (best effort):
        - posture_score, mfa_used, resource_sensitivity
        - raw_time_anomaly, raw_privilege_escalation
        - label_attack, attack_type (for evaluation)
        """
        out = df.copy()

        # Ensure columns exist
        for col, default in [
            ("posture_score", 1.0),
            ("mfa_used", 0),
            ("resource_sensitivity", "LOW"),
            ("raw_time_anomaly", 0),
            ("raw_privilege_escalation", 0),
            ("token_scope", ""),
            ("token_age", 0),
            ("bytes_out", 0),
            ("raw_login_failures_5m", 0),
            ("raw_api_calls_5m", 0),
            ("latency_ms", 0),
            ("action", ""),
            ("role", "employee"),
        ]:
            if col not in out.columns:
                out[col] = default

        # Fit lazily if needed (use current df)
        if self.model is None:
            self.fit(out)

        # Contextual risk
        posture = _to_float(out["posture_score"], default=1.0).clip(0.0, 1.0)
        posture_risk = (self.cfg.posture_low - posture) / max(self.cfg.posture_low, 1e-6)
        posture_risk = posture_risk.clip(0.0, 1.0)

        mfa_missing = 1 - _to_int01(out["mfa_used"], default=0)  # 1 => risk
        time_anom = _to_int01(out["raw_time_anomaly"], default=0)
        priv_esc = _to_int01(out["raw_privilege_escalation"], default=0)

        sens = out["resource_sensitivity"].apply(_sens_score).astype(float)

        s_ctx = (
            self.cfg.w_posture * posture_risk
            + self.cfg.w_mfa_missing * mfa_missing
            + self.cfg.w_time_anomaly * time_anom
            + self.cfg.w_priv_escalation * priv_esc
            + self.cfg.w_sensitive_resource * sens
        ).clip(0.0, 1.0)

        # Behavioral risk via Isolation Forest
        X = self._make_X(out)
        raw_anom = self._raw_anom_score(X)
        s_anom = (raw_anom - self._anom_low) / max(self._anom_high - self._anom_low, 1e-6)
        s_anom = np.clip(s_anom, 0.0, 1.0)

        # RAAC fusion
        alpha = float(self.cfg.alpha_context)
        beta = float(self.cfg.beta_behavior)
        denom = alpha + beta
        if denom <= 0:
            alpha, beta, denom = 0.5, 0.5, 1.0
        alpha /= denom
        beta /= denom
        risk = (alpha * s_ctx + beta * s_anom).clip(0.0, 1.0)

        # Confidence = agreement between context & behavior, plus stronger signal if risk extreme
        agree = 1.0 - np.abs(s_ctx - s_anom)
        conf = (0.75 * agree + 0.25 * (1.0 - np.abs(risk - 0.5) * 2)).clip(0.0, 1.0)

        # Decision mapping
        t_allow, t_stepup, t_deny = float(self.cfg.t_allow), float(self.cfg.t_stepup), float(self.cfg.t_deny)

        decision = np.full(len(out), IntentDecision.ALLOW.value, dtype=object)
        decision[(risk >= t_allow) & (risk < t_stepup)] = IntentDecision.STEPUP.value
        decision[(risk >= t_stepup) & (risk < t_deny)] = IntentDecision.RESTRICT.value
        decision[risk >= t_deny] = IntentDecision.DENY.value

        out["s_ctx"] = s_ctx.astype(float)
        out["s_anom"] = pd.Series(s_anom, index=out.index).astype(float)
        out["risk"] = pd.Series(risk, index=out.index).astype(float)
        out["confidence"] = pd.Series(conf, index=out.index).astype(float)

        out["decision"] = pd.Series(decision, index=out.index).astype(str)
        out["decision_tuned"] = out["decision"]  # no tuned thresholds for baseline

        # Rationale tags
        tags_list: List[str] = []
        texts: List[str] = []
        for i in range(len(out)):
            tags: List[str] = []
            if s_ctx.iloc[i] >= 0.6:
                tags.append("raac_context_high")
            if s_anom[i] >= 0.6:
                tags.append("iforest_anomaly_high")
            if mfa_missing.iloc[i] == 1:
                tags.append("mfa_missing")
            if time_anom.iloc[i] == 1:
                tags.append("time_anomaly")
            if priv_esc.iloc[i] == 1:
                tags.append("privilege_escalation")
            if sens.iloc[i] >= 0.75:
                tags.append("sensitive_resource")

            if not tags:
                tags = ["normal_behavior"]
            tags_list.append("|".join(tags))

            # Short human text
            txt_parts = []
            if "raac_context_high" in tags:
                txt_parts.append("Contextual risk is high (posture/MFA/time/privilege/sensitivity).")
            if "iforest_anomaly_high" in tags:
                txt_parts.append("Behavior deviates from baseline (Isolation Forest anomaly).")
            if "mfa_missing" in tags:
                txt_parts.append("MFA not used.")
            if "time_anomaly" in tags:
                txt_parts.append("Unusual access time.")
            if "privilege_escalation" in tags:
                txt_parts.append("Privilege escalation signal.")
            if "sensitive_resource" in tags:
                txt_parts.append("Sensitive resource access.")
            if not txt_parts:
                txt_parts.append("No significant risk indicators.")
            texts.append(" ".join(txt_parts))

        out["rationale_tags"] = pd.Series(tags_list, index=out.index).astype(str)
        out["rationale_text"] = pd.Series(texts, index=out.index).astype(str)

        return out

    def _make_X(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure columns exist
        X = pd.DataFrame(index=df.index)
        for c in self._feature_cols:
            if c not in df.columns:
                X[c] = 0.0
            else:
                X[c] = _to_float(df[c], default=0.0)
        # guard inf
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return X

    def _raw_anom_score(self, X: pd.DataFrame) -> np.ndarray:
        # IsolationForest decision_function: higher => more normal
        # We'll invert so higher => more anomalous
        assert self.model is not None
        normality = self.model.decision_function(X)  # shape (n,)
        return (-normality).astype(float)
