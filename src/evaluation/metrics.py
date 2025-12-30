from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


@dataclass
class Confusion:
    TP: int
    FP: int
    TN: int
    FN: int

    def as_dict(self) -> Dict[str, int]:
        return {"TP": self.TP, "FP": self.FP, "TN": self.TN, "FN": self.FN}


def binary_confusion(y_true: Iterable[int], y_pred: Iterable[int]) -> Confusion:
    """Compute confusion matrix for 0/1 labels."""
    yt = np.asarray(list(y_true)).astype(int)
    yp = np.asarray(list(y_pred)).astype(int)

    TP = int(((yp == 1) & (yt == 1)).sum())
    TN = int(((yp == 0) & (yt == 0)).sum())
    FP = int(((yp == 1) & (yt == 0)).sum())
    FN = int(((yp == 0) & (yt == 1)).sum())
    return Confusion(TP=TP, FP=FP, TN=TN, FN=FN)


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, Any]:
    """Precision/recall/FPR/accuracy/F1 for binary 0/1 predictions."""
    c = binary_confusion(y_true, y_pred)
    precision = safe_div(c.TP, c.TP + c.FP)
    recall = safe_div(c.TP, c.TP + c.FN)
    fpr = safe_div(c.FP, c.FP + c.TN)
    accuracy = safe_div(c.TP + c.TN, c.TP + c.TN + c.FP + c.FN)
    f1 = safe_div(2 * precision * recall, precision + recall)
    out: Dict[str, Any] = dict(c.as_dict())
    out.update(
        {
            "precision": float(precision),
            "recall": float(recall),
            "fpr": float(fpr),
            "accuracy": float(accuracy),
            "f1": float(f1),
        }
    )
    return out


def percentile(x: Iterable[float], q: float) -> float:
    xs = [float(v) for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def latency_summary(latencies_s: Iterable[float]) -> Dict[str, Any]:
    xs = [float(v) for v in latencies_s if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not xs:
        return {
            "count": 0,
            "mean_s": 0.0,
            "p50_s": 0.0,
            "p95_s": 0.0,
            "p99_s": 0.0,
            "max_s": 0.0,
        }
    arr = np.asarray(xs, dtype=float)
    return {
        "count": int(arr.size),
        "mean_s": float(arr.mean()),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "p99_s": float(np.percentile(arr, 99)),
        "max_s": float(arr.max()),
    }


def hard_flag_from_decision(decision: Any) -> int:
    """Map decisions to a 0/1 'flagged' outcome for detection metrics.

    By default: restrict/deny => 1, else 0.
    """
    if decision is None:
        return 0
    d = str(decision).strip().lower()
    return 1 if d in {"restrict", "deny"} else 0
