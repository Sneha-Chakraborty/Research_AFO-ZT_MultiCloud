from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.afozt.policy_intent import IntentDecision


@dataclass
class SiemRulesConfig:
    """SIEM-like baseline: threshold rules + simple correlation.

    This baseline mimics typical SIEM behaviour:
    - coarse threshold rules (login failures, bytes_out spike, API burst, etc.)
    - correlation in a short time window per principal/session

    It is intentionally *less context-aware* than AFO-ZT (no trust-graph reasoning, no safe rollout).

    Output contract matches other models:
      - risk in [0,1]
      - confidence in [0,1]
      - decision/decision_tuned in {allow, stepup, restrict, deny}
      - rationale_tags and rationale_text

    Notes:
      - "manual delay" is not applied in decisions; simulate it in Step 2.8 by running
        scripts/run_orchestration.py with a positive --processing_delay_ms.
    """

    # Base thresholds
    login_fail_alert: int = 5
    login_fail_high: int = 10

    api_calls_alert: int = 60
    api_calls_high: int = 120

    posture_low: float = 0.35
    posture_medium: float = 0.55

    # If set to None, the baseline auto-derives from data quantiles
    bytes_p95: float | None = None
    bytes_p99: float | None = None

    # Correlation windows (minutes)
    window_minutes: int = 15
    # How many different rule hits in the window triggers escalation
    correlation_hits_to_escalate: int = 3


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _norm_sens(v: object) -> str:
    s = str(v).strip().upper()
    if s in {"LOW", "MED", "HIGH", "CRITICAL"}:
        return s
    return "LOW"


def _as_dt(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, errors="coerce")
    # ensure no timezone for simple comparisons
    try:
        return dt.dt.tz_localize(None)  # type: ignore[attr-defined]
    except Exception:
        return dt


class SiemRulesBaseline:
    def __init__(self, cfg: SiemRulesConfig | None = None) -> None:
        self.cfg = cfg or SiemRulesConfig()
        self._derived: Dict[str, float] = {}

    def _derive_thresholds(self, df: pd.DataFrame) -> None:
        # bytes_out thresholds are often set as percentile-based SIEM rules
        bytes_out = pd.to_numeric(df.get("bytes_out", 0.0), errors="coerce").fillna(0.0).astype(float)
        if self.cfg.bytes_p95 is None:
            self._derived["bytes_p95"] = float(np.percentile(bytes_out.to_numpy(), 95))
        else:
            self._derived["bytes_p95"] = float(self.cfg.bytes_p95)

        if self.cfg.bytes_p99 is None:
            self._derived["bytes_p99"] = float(np.percentile(bytes_out.to_numpy(), 99))
        else:
            self._derived["bytes_p99"] = float(self.cfg.bytes_p99)

        # guard against degenerate data
        self._derived["bytes_p95"] = max(1.0, self._derived["bytes_p95"])
        self._derived["bytes_p99"] = max(self._derived["bytes_p95"], self._derived["bytes_p99"])

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # best-effort defaults
        for col, default in [
            ("ts", ""),
            ("principal_id", "unknown"),
            ("session_id", "unknown"),
            ("role", "employee"),
            ("resource_sensitivity", "LOW"),
            ("mfa_used", 0),
            ("posture_score", 1.0),
            ("token_scope", "NARROW"),
            ("bytes_out", 0.0),
            ("raw_login_failures_5m", 0),
            ("raw_api_calls_5m", 0),
            ("raw_time_anomaly", 0),
            ("raw_privilege_escalation", 0),
            ("raw_cross_cloud_hop", 0),
        ]:
            if col not in out.columns:
                out[col] = default

        # derive thresholds from the current run (SIEM playbooks often use static percentiles)
        self._derive_thresholds(out)
        b95 = float(self._derived["bytes_p95"])
        b99 = float(self._derived["bytes_p99"])

        # normalize inputs
        dt = _as_dt(out["ts"])
        pid = out["principal_id"].fillna("unknown").astype(str)
        sid = out["session_id"].fillna("unknown").astype(str)
        sens = out["resource_sensitivity"].map(_norm_sens)
        mfa = out["mfa_used"].fillna(0).astype(int).clip(0, 1)
        posture = pd.to_numeric(out["posture_score"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        scope = out["token_scope"].fillna("NARROW").astype(str).str.upper()

        login_f = pd.to_numeric(out["raw_login_failures_5m"], errors="coerce").fillna(0).astype(int)
        api_c = pd.to_numeric(out["raw_api_calls_5m"], errors="coerce").fillna(0).astype(int)
        time_anom = pd.to_numeric(out["raw_time_anomaly"], errors="coerce").fillna(0).astype(int)
        priv_esc = pd.to_numeric(out["raw_privilege_escalation"], errors="coerce").fillna(0).astype(int)
        xcloud = pd.to_numeric(out["raw_cross_cloud_hop"], errors="coerce").fillna(0).astype(int)
        bytes_out = pd.to_numeric(out["bytes_out"], errors="coerce").fillna(0.0).astype(float)

        # -----------------
        # Rule hits per row
        # -----------------
        tags: List[List[str]] = [[] for _ in range(len(out))]
        base_sev = np.zeros(len(out), dtype=int)  # 0..3

        def _add(i: int, tag: str, sev: int) -> None:
            tags[i].append(tag)
            base_sev[i] = int(max(base_sev[i], sev))

        for i in range(len(out)):
            # brute force
            if login_f.iat[i] >= self.cfg.login_fail_high:
                _add(i, "siem_bruteforce_high", 2)
            elif login_f.iat[i] >= self.cfg.login_fail_alert:
                _add(i, "siem_bruteforce_alert", 1)

            # impossible travel / time anomaly
            if time_anom.iat[i] == 1:
                _add(i, "siem_impossible_travel", 2)

            # privilege escalation
            if priv_esc.iat[i] == 1:
                _add(i, "siem_privilege_escalation", 3 if sens.iat[i] in {"HIGH", "CRITICAL"} else 2)

            # API burst
            if api_c.iat[i] >= self.cfg.api_calls_high:
                _add(i, "siem_api_burst_high", 2)
            elif api_c.iat[i] >= self.cfg.api_calls_alert:
                _add(i, "siem_api_burst_alert", 1)

            # Exfil spike
            if bytes_out.iat[i] >= b99:
                _add(i, "siem_exfil_spike_high", 3)
            elif bytes_out.iat[i] >= b95:
                _add(i, "siem_exfil_spike_medium", 2)

            # Cross-cloud hop
            if xcloud.iat[i] == 1:
                _add(i, "siem_cross_cloud_hop", 1)

            # Posture
            if posture.iat[i] < self.cfg.posture_low:
                _add(i, "siem_low_posture", 2)
            elif posture.iat[i] < self.cfg.posture_medium:
                _add(i, "siem_medium_posture", 1)

            # MFA missing for sensitive resources
            if mfa.iat[i] == 0 and sens.iat[i] in {"HIGH", "CRITICAL"}:
                _add(i, "siem_mfa_missing_sensitive", 1)

            # Wide scope token on sensitive resource
            if scope.iat[i] == "WIDE" and sens.iat[i] in {"HIGH", "CRITICAL"}:
                _add(i, "siem_wide_scope_sensitive", 2)

        # -----------------
        # Correlation rules
        # -----------------
        corr_sev = base_sev.copy()
        window = pd.Timedelta(minutes=int(self.cfg.window_minutes))

        # We correlate per principal_id (SIEM correlation searches are often identity-centric)
        df_idx = pd.DataFrame({"i": np.arange(len(out)), "principal_id": pid, "session_id": sid, "dt": dt})
        df_idx = df_idx.sort_values(["principal_id", "dt"], kind="mergesort")

        # Precompute booleans for faster scanning
        is_bruteforce = np.array([("siem_bruteforce_alert" in t or "siem_bruteforce_high" in t) for t in tags], dtype=bool)
        is_time = np.array([("siem_impossible_travel" in t) for t in tags], dtype=bool)
        is_priv = np.array([("siem_privilege_escalation" in t) for t in tags], dtype=bool)
        is_exfil_high = np.array([("siem_exfil_spike_high" in t) for t in tags], dtype=bool)
        is_exfil_med = np.array([("siem_exfil_spike_medium" in t) for t in tags], dtype=bool)
        is_xcloud = np.array([("siem_cross_cloud_hop" in t) for t in tags], dtype=bool)

        # A lightweight rolling-window scan (two pointers) per principal
        for principal, g in df_idx.groupby("principal_id", sort=False):
            idxs = g["i"].to_numpy(dtype=int)
            dts = g["dt"].to_numpy()

            left = 0
            # maintain window stats
            cnt_hits = 0
            cnt_bruteforce = 0
            cnt_time = 0
            cnt_priv = 0
            cnt_exfil_high = 0
            cnt_exfil_med = 0
            cnt_xcloud = 0

            # helper to compute "hit" count: number of distinct rules in the row (rough proxy for alert richness)
            row_hit_counts = np.array([len(set(tags[i])) for i in idxs], dtype=int)

            # We'll approximate rolling distinct hit count by using row_hit_counts sums.
            # It's coarse but matches SIEM-ish correlation behaviour (more signals => higher severity).
            for right in range(len(idxs)):
                i_r = idxs[right]

                # add right
                cnt_hits += int(row_hit_counts[right] > 0)
                cnt_bruteforce += int(is_bruteforce[i_r])
                cnt_time += int(is_time[i_r])
                cnt_priv += int(is_priv[i_r])
                cnt_exfil_high += int(is_exfil_high[i_r])
                cnt_exfil_med += int(is_exfil_med[i_r])
                cnt_xcloud += int(is_xcloud[i_r])

                # shrink left to satisfy window
                while left <= right and (pd.Timestamp(dts[right]) - pd.Timestamp(dts[left])) > window:
                    i_l = idxs[left]
                    cnt_hits -= int(row_hit_counts[left] > 0)
                    cnt_bruteforce -= int(is_bruteforce[i_l])
                    cnt_time -= int(is_time[i_l])
                    cnt_priv -= int(is_priv[i_l])
                    cnt_exfil_high -= int(is_exfil_high[i_l])
                    cnt_exfil_med -= int(is_exfil_med[i_l])
                    cnt_xcloud -= int(is_xcloud[i_l])
                    left += 1

                # Apply correlations at current row i_r
                # C1: brute force + impossible travel in window => account takeover suspicion
                if cnt_bruteforce > 0 and cnt_time > 0:
                    corr_sev[i_r] = max(corr_sev[i_r], 3)
                    tags[i_r].append("corr_ato_bruteforce_plus_impossible_travel")

                # C2: privilege escalation + sensitive access => high severity
                if cnt_priv > 0 and sens.iat[i_r] in {"HIGH", "CRITICAL"}:
                    corr_sev[i_r] = max(corr_sev[i_r], 3)
                    tags[i_r].append("corr_privilege_escalation_on_sensitive")

                # C3: cross-cloud hop + exfil spike => likely cross-cloud campaign
                if cnt_xcloud > 0 and (cnt_exfil_high > 0 or cnt_exfil_med >= 2):
                    corr_sev[i_r] = max(corr_sev[i_r], 3)
                    tags[i_r].append("corr_cross_cloud_plus_exfil")

                # C4: lots of signals in short window => escalation
                if cnt_hits >= self.cfg.correlation_hits_to_escalate:
                    corr_sev[i_r] = max(corr_sev[i_r], 2)
                    tags[i_r].append("corr_multi_signal_escalation")

        # -----------------
        # Map severity -> risk/decision
        # -----------------
        decisions: List[str] = []
        confidences: List[float] = []
        risks: List[float] = []
        tags_list: List[str] = []
        texts: List[str] = []

        for i in range(len(out)):
            sev = int(corr_sev[i])
            uniq_tags = sorted(set(tags[i]))

            if sev >= 3:
                dec = IntentDecision.DENY.value
                risk = 0.92
                conf = 0.70
            elif sev == 2:
                dec = IntentDecision.RESTRICT.value
                risk = 0.75
                conf = 0.62
            elif sev == 1:
                dec = IntentDecision.STEPUP.value
                risk = 0.50
                conf = 0.55
            else:
                dec = IntentDecision.ALLOW.value
                risk = 0.20
                conf = 0.50

            # small adjustments (more tags => more confidence)
            conf = _clip01(conf + 0.05 * min(4, len(uniq_tags)))
            risk = _clip01(risk + 0.02 * min(6, len(uniq_tags)))

            decisions.append(dec)
            confidences.append(conf)
            risks.append(risk)
            tags_list.append(";".join(uniq_tags) if uniq_tags else "")
            texts.append("SIEM rules triggered: " + (", ".join(uniq_tags) if uniq_tags else "none"))

        out["siem_severity"] = corr_sev.astype(int)
        out["siem_bytes_p95"] = float(b95)
        out["siem_bytes_p99"] = float(b99)
        out["risk"] = pd.Series(risks, index=out.index).astype(float)
        out["confidence"] = pd.Series(confidences, index=out.index).astype(float).clip(0.0, 1.0)
        out["decision"] = pd.Series(decisions, index=out.index).astype(str)
        out["decision_tuned"] = out["decision"]  # no tuning stage in this baseline
        out["rationale_tags"] = pd.Series(tags_list, index=out.index).astype(str)
        out["rationale_text"] = pd.Series(texts, index=out.index).astype(str)

        return out

    def artifact(self) -> Dict[str, object]:
        return {
            "baseline": "siem_rules",
            "config": self.cfg.__dict__,
            "derived": self._derived,
        }
