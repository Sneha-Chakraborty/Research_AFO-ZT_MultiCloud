from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.afozt.calibration import QuantileCalibrator
from src.common.logging_setup import setup_logger


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class ScoreOutputs:
    # component scores (0..1)
    s_anom: np.ndarray
    s_rule: np.ndarray
    s_graph: np.ndarray
    # fused (0..1)
    risk: np.ndarray
    confidence: np.ndarray
    # human outputs
    rationale_tags: List[str]
    rationale_text: List[str]
    decision: List[str]


@dataclass
class BrainConfig:
    # IsolationForest
    iforest_n_estimators: int = 200
    iforest_max_samples: int | float = 0.8
    iforest_contamination: float = 0.02
    iforest_random_state: int = 42

    # Fusion weights
    w_anom: float = 1.20
    w_rule: float = 1.00
    w_graph: float = 1.15
    bias: float = -0.40

    # Decision thresholds (Step 2.6 will refine; Step 2.5 provides a draft)
    t_allow: float = 0.30
    t_stepup: float = 0.60
    t_restrict: float = 0.80
    t_deny: float = 0.90
    min_conf_for_hard_actions: float = 0.60

    # Rationale thresholds
    thr_failed_logins_5m: float = 3.0
    thr_bytes_spike: float = 2.5
    thr_posture_drift: float = 0.25
    thr_new_resource_rate: float = 1.0
    thr_shortest_sensitive_hops: int = 2
    thr_blast_radius_k: int = 15

    # Which columns to use for anomaly model
    anomaly_feature_cols: Optional[List[str]] = None


class UnifiedDecisionBrain:
    """Step 2.5: Risk/Trust scoring + Confidence + Rationale Output.

    Research-grade properties:
    - Hybrid score fusion: anomaly + rules + graph signals
    - Calibrated anomaly score to [0,1] using empirical quantiles
    - Deterministic, reproducible (fixed random_state)
    - Produces rationale tags and short rationale text per event
    """

    def __init__(self, config: Optional[BrainConfig] = None) -> None:
        self.cfg = config or BrainConfig()
        self.logger = setup_logger("afozt_brain")
        self.scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5, 95))
        self.iforest = IsolationForest(
            n_estimators=self.cfg.iforest_n_estimators,
            max_samples=self.cfg.iforest_max_samples,
            contamination=self.cfg.iforest_contamination,
            random_state=self.cfg.iforest_random_state,
            n_jobs=-1,
        )
        self.calibrator = QuantileCalibrator()
        self.feature_cols_: List[str] = []

    @staticmethod
    def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
        if col not in df.columns:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float).to_numpy()

    @staticmethod
    def _safe_str(df: pd.DataFrame, col: str, default: str = "") -> np.ndarray:
        if col not in df.columns:
            return np.full(len(df), default, dtype=object)
        return df[col].astype(str).fillna(default).to_numpy()

    def _default_anomaly_cols(self, df: pd.DataFrame) -> List[str]:
        # Use the Step 2.4 engineered features if present. Keep it stable & CPU-friendly.
        candidates = [
            "f_token_scope_width",
            "f_token_age",
            "f_bytes_out_log1p",
            "f_failed_logins_5m",
            "f_geo_switch",
            "f_fast_geo_switch",
            "f_posture_drift",
            "f_bytes_spike_score",
            "f_is_privileged_role",
            "g_cross_cloud_hops_session",
            "g_is_new_resource_edge",
            "g_new_resource_count_win",
            "g_new_resource_rate_win",
            "g_shortest_to_sensitive_hops",
            "g_blast_radius_sensitive_k",
        ]
        cols = [c for c in candidates if c in df.columns]
        # Fallback: any numeric columns that start with f_/g_
        if not cols:
            cols = [c for c in df.columns if (c.startswith("f_") or c.startswith("g_"))]
        return cols

    def fit(self, df_train: pd.DataFrame) -> None:
        """Fit anomaly model + calibrator.

        Best practice: train on benign-only events if labels exist.
        """
        df = df_train.copy()

        # If truth labels exist, fit on benign to improve precision.
        if "label_attack" in df.columns:
            benign = df[pd.to_numeric(df["label_attack"], errors="coerce").fillna(0).astype(int) == 0]
            if len(benign) >= max(500, int(0.1 * len(df))):
                df = benign

        self.feature_cols_ = self.cfg.anomaly_feature_cols or self._default_anomaly_cols(df)
        if not self.feature_cols_:
            raise ValueError("No anomaly feature columns found. Ensure Step 2.4 features exist.")

        X = df[self.feature_cols_].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        Xs = self.scaler.fit_transform(X)
        self.iforest.fit(Xs)

        # IsolationForest: higher score_samples => more normal. We want anomaly magnitude.
        raw_anom = (-self.iforest.score_samples(Xs)).astype(float)
        self.calibrator.fit(raw_anom)
        self.logger.info(
            f"✅ Fitted UnifiedDecisionBrain: train_rows={len(df)} cols={len(self.feature_cols_)} "
            f"contamination={self.cfg.iforest_contamination}"
        )

    def _score_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols_].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        Xs = self.scaler.transform(X)
        raw_anom = (-self.iforest.score_samples(Xs)).astype(float)
        s = self.calibrator.transform(raw_anom)
        return s

    def _score_rules(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        """Interpretable rule score in [0,1] plus rule tags per row."""
        n = len(df)
        tags: List[List[str]] = [[] for _ in range(n)]

        failed = self._safe_num(df, "f_failed_logins_5m", 0.0)
        fast_geo = self._safe_num(df, "f_fast_geo_switch", 0.0)
        posture = self._safe_num(df, "f_posture_drift", 0.0)
        bytes_spike = self._safe_num(df, "f_bytes_spike_score", 0.0)
        priv = self._safe_num(df, "f_is_privileged_role", 0.0)
        sens = self._safe_str(df, "resource_sensitivity", "")

        # Normalize each risk hint to [0,1] with simple saturating transforms.
        s_failed = np.clip(failed / max(1.0, self.cfg.thr_failed_logins_5m), 0.0, 1.0)
        s_geo = np.clip(fast_geo, 0.0, 1.0)
        s_posture = np.clip(posture / max(1e-6, self.cfg.thr_posture_drift), 0.0, 1.0)
        s_bytes = np.clip(bytes_spike / max(1e-6, self.cfg.thr_bytes_spike), 0.0, 1.0)

        sens_low = np.char.lower(sens.astype(str))
        is_sensitive = np.isin(sens_low, np.array(["high", "critical", "secret"], dtype=object)).astype(float)
        s_priv = np.clip(priv, 0.0, 1.0) * (0.5 + 0.5 * is_sensitive)

        # Weighted interpretable blend (kept stable; tune later)
        score = (
            0.30 * s_failed
            + 0.25 * s_geo
            + 0.20 * s_bytes
            + 0.15 * s_posture
            + 0.10 * s_priv
        )
        score = np.clip(score, 0.0, 1.0)

        for i in range(n):
            if failed[i] >= self.cfg.thr_failed_logins_5m:
                tags[i].append("failed_logins_spike")
            if fast_geo[i] >= 1:
                tags[i].append("fast_geo_switch")
            if bytes_spike[i] >= self.cfg.thr_bytes_spike:
                tags[i].append("bytes_exfil_spike")
            if posture[i] >= self.cfg.thr_posture_drift:
                tags[i].append("device_posture_drift")
            if priv[i] >= 1 and is_sensitive[i] >= 1:
                tags[i].append("privileged_sensitive_access")

        return score, tags

    def _score_graph(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        n = len(df)
        tags: List[List[str]] = [[] for _ in range(n)]

        hops = self._safe_num(df, "g_cross_cloud_hops_session", 0.0)
        new_edge = self._safe_num(df, "g_is_new_resource_edge", 0.0)
        new_rate = self._safe_num(df, "g_new_resource_rate_win", 0.0)
        shortest = self._safe_num(df, "g_shortest_to_sensitive_hops", -1.0)
        blast = self._safe_num(df, "g_blast_radius_sensitive_k", 0.0)

        # Convert to normalized risks
        s_hops = np.clip(hops / 2.0, 0.0, 1.0)  # 0..2 hops maps to 0..1
        s_new_edge = np.clip(new_edge, 0.0, 1.0)
        s_new_rate = np.clip(new_rate / max(1e-6, self.cfg.thr_new_resource_rate), 0.0, 1.0)

        # shortest: smaller hops -> higher risk; -1 means unknown => low
        s_short = np.where(shortest < 0, 0.0, np.clip((self.cfg.thr_shortest_sensitive_hops - shortest + 1) / 3.0, 0.0, 1.0))
        s_blast = np.clip(blast / max(1.0, float(self.cfg.thr_blast_radius_k)), 0.0, 1.0)

        score = np.clip(
            0.30 * s_hops + 0.25 * s_new_edge + 0.20 * s_new_rate + 0.15 * s_short + 0.10 * s_blast,
            0.0,
            1.0,
        )

        for i in range(n):
            if hops[i] >= 1:
                tags[i].append("cross_cloud_hop")
            if new_edge[i] >= 1:
                tags[i].append("new_resource_edge")
            if new_rate[i] >= self.cfg.thr_new_resource_rate:
                tags[i].append("new_edge_rate_spike")
            if shortest[i] >= 0 and shortest[i] <= self.cfg.thr_shortest_sensitive_hops:
                tags[i].append("close_to_sensitive_resource")
            if blast[i] >= self.cfg.thr_blast_radius_k:
                tags[i].append("large_blast_radius")

        return score, tags

    def _confidence(self, s_anom: np.ndarray, s_rule: np.ndarray, s_graph: np.ndarray) -> np.ndarray:
        S = np.vstack([s_anom, s_rule, s_graph]).T  # n x 3
        mean = S.mean(axis=1)
        std = S.std(axis=1)

        # Agreement: std=0 => 1; std~0.5 => 0
        conf_agree = np.clip(1.0 - (std / 0.5), 0.0, 1.0)

        # Strength: mean near 0 or 1 => 1; near 0.5 => 0
        conf_strength = np.clip(1.0 - 4.0 * mean * (1.0 - mean), 0.0, 1.0)

        return np.clip(0.55 * conf_agree + 0.45 * conf_strength, 0.0, 1.0)

    def _decision(self, risk: np.ndarray, conf: np.ndarray) -> List[str]:
        out: List[str] = []
        for r, c in zip(risk.tolist(), conf.tolist()):
            if r < self.cfg.t_allow:
                out.append("allow")
            elif r < self.cfg.t_stepup:
                out.append("stepup")
            elif r < self.cfg.t_restrict:
                out.append("restrict")
            else:
                # hard deny only when confident; otherwise restrict
                if r >= self.cfg.t_deny and c >= self.cfg.min_conf_for_hard_actions:
                    out.append("deny")
                else:
                    out.append("restrict")
        return out

    def score(self, df: pd.DataFrame) -> ScoreOutputs:
        if not self.feature_cols_:
            raise RuntimeError("UnifiedDecisionBrain is not fit yet. Call fit(train_df) first.")

        df2 = df.copy()
        s_anom = self._score_anomaly(df2)
        s_rule, tags_rule = self._score_rules(df2)
        s_graph, tags_graph = self._score_graph(df2)

        z = (
            self.cfg.w_anom * s_anom
            + self.cfg.w_rule * s_rule
            + self.cfg.w_graph * s_graph
            + self.cfg.bias
        )
        risk = sigmoid(z)

        conf = self._confidence(s_anom, s_rule, s_graph)
        decision = self._decision(risk, conf)

        rationale_tags: List[str] = []
        rationale_text: List[str] = []

        for i in range(len(df2)):
            # component contributions for "why"
            contrib = [
                ("anomaly", self.cfg.w_anom * float(s_anom[i])),
                ("rules", self.cfg.w_rule * float(s_rule[i])),
                ("graph", self.cfg.w_graph * float(s_graph[i])),
            ]
            contrib.sort(key=lambda x: x[1], reverse=True)
            top_comp = [c[0] for c in contrib[:2]]

            tags = []
            tags.extend(tags_rule[i])
            tags.extend(tags_graph[i])

            # Deduplicate while preserving order
            seen = set()
            tags_unique = []
            for t in tags:
                if t not in seen:
                    seen.add(t)
                    tags_unique.append(t)

            # Build short text
            if tags_unique:
                txt = f"Top factors: {', '.join(top_comp)}; signals: {', '.join(tags_unique[:4])}"
            else:
                txt = f"Top factors: {', '.join(top_comp)}"

            rationale_tags.append(";".join(tags_unique))
            rationale_text.append(txt)

        return ScoreOutputs(
            s_anom=s_anom,
            s_rule=s_rule,
            s_graph=s_graph,
            risk=risk,
            confidence=conf,
            rationale_tags=rationale_tags,
            rationale_text=rationale_text,
            decision=decision,
        )

    def to_artifact(self) -> Dict[str, object]:
        """Serialize minimal model state for reproducibility."""
        return {
            "cfg": self.cfg.__dict__,
            "feature_cols": self.feature_cols_,
            "scaler": self.scaler,
            "iforest": self.iforest,
            "calibrator": self.calibrator,
        }

    @staticmethod
    def from_artifact(artifact: Dict[str, object]) -> "UnifiedDecisionBrain":
        cfg = BrainConfig(**artifact["cfg"])
        brain = UnifiedDecisionBrain(cfg)
        brain.feature_cols_ = list(artifact["feature_cols"])
        brain.scaler = artifact["scaler"]  # type: ignore[assignment]
        brain.iforest = artifact["iforest"]  # type: ignore[assignment]
        brain.calibrator = artifact["calibrator"]  # type: ignore[assignment]
        return brain
