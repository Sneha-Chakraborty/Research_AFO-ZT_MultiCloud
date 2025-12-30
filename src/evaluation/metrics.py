from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute basic binary classification metrics for 0/1 series.
    Returns accuracy, precision, recall, fpr, f1 and confusion counts.
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = _safe_div(TP + TN, TP + TN + FP + FN)
    prec = _safe_div(TP, TP + FP)
    rec = _safe_div(TP, TP + FN)
    fpr = _safe_div(FP, FP + TN)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "fpr": float(fpr),
        "f1": float(f1),
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }


def decision_flags(df: pd.DataFrame, decision_col: str = "decision_tuned") -> Tuple[pd.Series, pd.Series]:
    """
    Convert 4-class ZT decisions into:
      - flag_pred: 1 if flagged (stepup/restrict/deny), else 0
      - hard_pred: 1 if hard action (restrict/deny), else 0
    """
    if decision_col not in df.columns:
        raise KeyError(f"Missing decision column: {decision_col}")

    dec = df[decision_col].fillna("").astype(str)
    flag_pred = dec.isin(["stepup", "restrict", "deny"]).astype(int)
    hard_pred = dec.isin(["restrict", "deny"]).astype(int)
    return flag_pred, hard_pred


def access_decision_precision(df: pd.DataFrame, decision_col: str = "decision_tuned") -> float:
    """
    Access Decision Precision (for deny/stepup/restrict):
    Among all flagged decisions, what fraction were truly attacks?

    If there are no flagged decisions, returns 0.0.
    """
    if "label_attack" not in df.columns:
        raise KeyError("Missing label_attack column for evaluation.")
    y = df["label_attack"].astype(int)
    flag_pred, _ = decision_flags(df, decision_col=decision_col)

    flagged = flag_pred == 1
    return float((y[flagged] == 1).mean()) if int(flagged.sum()) > 0 else 0.0


def summarize_scores(df: pd.DataFrame, decision_col: str = "decision_tuned") -> Dict[str, float]:
    """
    Summary metrics used consistently across baselines + AFO-ZT.
    """
    if "label_attack" not in df.columns:
        raise KeyError("Missing label_attack column for evaluation.")
    y = df["label_attack"].astype(int)

    flag_pred, hard_pred = decision_flags(df, decision_col=decision_col)
    m_flag = binary_metrics(y, flag_pred)
    m_hard = binary_metrics(y, hard_pred)

    return {
        "rows": float(len(df)),
        "attacks": float((y == 1).sum()),
        "benign": float((y == 0).sum()),
        # Detection (flagged)
        "det_accuracy": m_flag["accuracy"],
        "det_precision": m_flag["precision"],
        "det_recall": m_flag["recall"],
        "det_fpr": m_flag["fpr"],
        "det_f1": m_flag["f1"],
        # Hard actions
        "hard_accuracy": m_hard["accuracy"],
        "hard_precision": m_hard["precision"],
        "hard_recall": m_hard["recall"],
        "hard_fpr": m_hard["fpr"],
        "hard_f1": m_hard["f1"],
        # ZT-specific
        "access_decision_precision": access_decision_precision(df, decision_col=decision_col),
        "flagged_rate": float(flag_pred.mean()),
        "hard_rate": float(hard_pred.mean()),
    }
