from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.evaluation.metrics import latency_summary


def _parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "nan":
        return None
    # Accept either ISO '...Z' or pandas-friendly strings
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
        return datetime.fromisoformat(s.replace(" ", "T")).replace(tzinfo=None)
    except Exception:
        try:
            return pd.to_datetime(s, errors="coerce").to_pydatetime()  # type: ignore[attr-defined]
        except Exception:
            return None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def response_times_from_exec_events(exec_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute per-intent response time from action execution events."""
    if not exec_events:
        return pd.DataFrame(columns=["intent_id", "detect_ts", "finished_ts", "response_time_s", "total_action_latency_s", "action_events"])
    rows = []
    # group by intent_id
    by_intent: Dict[str, List[Dict[str, Any]]] = {}
    for e in exec_events:
        iid = str(e.get("intent_id", ""))
        by_intent.setdefault(iid, []).append(e)

    for iid, evs in by_intent.items():
        detect = _parse_dt(evs[0].get("detect_ts") or evs[0].get("started_at"))
        finished = max((_parse_dt(e.get("finished_at")) for e in evs if e.get("finished_at") is not None), default=None)
        total_lat = float(sum(float(e.get("latency_s", 0.0)) for e in evs))
        resp = 0.0
        if detect and finished:
            resp = float((finished - detect).total_seconds())
        rows.append(
            {
                "intent_id": iid,
                "detect_ts": detect.isoformat(sep=" ") if detect else None,
                "finished_ts": finished.isoformat(sep=" ") if finished else None,
                "response_time_s": resp,
                "total_action_latency_s": total_lat,
                "action_events": len(evs),
            }
        )
    return pd.DataFrame(rows)


def latency_stats(df_event_latency: pd.DataFrame, group_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    group_cols = group_cols or []
    stats: Dict[str, Any] = {"overall": latency_summary(df_event_latency.get("response_time_s", []))}
    if not group_cols:
        return stats

    grouped = df_event_latency.groupby(group_cols, dropna=False)
    by_group: Dict[str, Any] = {}
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        name = "|".join([str(k) for k in key])
        by_group[name] = latency_summary(g["response_time_s"].tolist())
    stats["by_" + "_".join(group_cols)] = by_group
    return stats
