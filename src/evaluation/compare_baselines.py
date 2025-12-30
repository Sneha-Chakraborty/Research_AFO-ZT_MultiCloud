from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.evaluation.metrics import summarize_scores


@dataclass
class BaselineRun:
    name: str
    scores_csv: Path
    decision_col: str = "decision_tuned"


def compare_runs(runs: List[BaselineRun]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for r in runs:
        df = pd.read_csv(r.scores_csv)
        metrics = summarize_scores(df, decision_col=r.decision_col)
        rows.append({"model": r.name, **metrics})
    return pd.DataFrame(rows)


def write_outputs(df_summary: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(out_csv, index=False)


def write_confusions(runs: List[BaselineRun], out_json: Path) -> None:
    """
    Optional: store decision counts for quick sanity checks.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    conf: Dict[str, Dict[str, int]] = {}
    for r in runs:
        df = pd.read_csv(r.scores_csv)
        col = r.decision_col
        conf[r.name] = df[col].fillna("").astype(str).value_counts().to_dict()
    out_json.write_text(json.dumps(conf, indent=2), encoding="utf-8")
