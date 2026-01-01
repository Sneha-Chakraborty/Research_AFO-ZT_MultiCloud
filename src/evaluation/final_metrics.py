from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.evaluation.metrics import summarize_scores
from src.evaluation.multicloud_metrics import (
    blast_radius_reduction,
    cross_cloud_attack_detection_rate,
    policy_consistency_variance,
    rollout_safety_metrics_from_json,
)
from src.evaluation.response_time import compute_response_times, load_latency_profiles


@dataclass
class EngineRun:
    name: str
    scores_csv: Path
    decision_col: str = "decision_tuned"
    # Optional: for AFO-ZT safe rollout KPIs
    rollout_metrics_json: Optional[Path] = None


def compute_engine_metrics(
    run: EngineRun,
    *,
    thresholds_yaml: str = "configs/thresholds.yaml",
    latency_profiles_yaml: str = "configs/latency_profiles.yaml",
    only_non_allow_for_latency: bool = True,
    max_latency_rows: Optional[int] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(run.scores_csv)
    base = summarize_scores(df, decision_col=run.decision_col)

    # Latency / MTTR proxy
    profiles = load_latency_profiles(latency_profiles_yaml)
    prof = profiles.get(run.name, profiles.get(run.name.replace("-", "_"), None))
    extra_delay = float(prof.processing_delay_s) if prof is not None else 0.0

    rt_df, rt_stats = compute_response_times(
        df,
        thresholds_yaml=thresholds_yaml,
        decision_col=run.decision_col,
        only_non_allow=only_non_allow_for_latency,
        extra_processing_delay_s=extra_delay,
        max_rows=max_latency_rows,
    )
    base["avg_response_time_s"] = float(rt_stats["overall"]["mean_s"])
    base["mttr_proxy_s"] = float(rt_stats["overall"]["mean_s"])
    base["p95_response_time_s"] = float(rt_stats["overall"]["p95_s"])
    base["latency_count"] = float(rt_stats["overall"]["count"])

    # Multi-cloud novelty metrics
    base.update(cross_cloud_attack_detection_rate(df, decision_col=run.decision_col))
    base.update(blast_radius_reduction(df, decision_col=run.decision_col))
    base.update(policy_consistency_variance(df, decision_col=run.decision_col))

    # Rollout safety metrics (AFO-ZT only, others -> NaN)
    if run.rollout_metrics_json is not None:
        base.update(rollout_safety_metrics_from_json(str(run.rollout_metrics_json)))
    else:
        base.update(rollout_safety_metrics_from_json("__missing__"))

    return base


def write_final_metrics(
    runs: List[EngineRun],
    *,
    out_csv: Path,
    out_json: Path,
    thresholds_yaml: str = "configs/thresholds.yaml",
    latency_profiles_yaml: str = "configs/latency_profiles.yaml",
    only_non_allow_for_latency: bool = True,
    max_latency_rows: Optional[int] = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for r in runs:
        m = compute_engine_metrics(
            r,
            thresholds_yaml=thresholds_yaml,
            latency_profiles_yaml=latency_profiles_yaml,
            only_non_allow_for_latency=only_non_allow_for_latency,
            max_latency_rows=max_latency_rows,
        )
        m["engine"] = r.name
        rows.append(m)

    df = pd.DataFrame(rows)
    # nice ordering
    first = ["engine", "rows", "attacks", "benign"]
    rest = [c for c in df.columns if c not in first]
    df = df[first + rest]

    df.to_csv(out_csv, index=False)

    # also keep structured json for programmatic checks
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
