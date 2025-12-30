from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    # y_true, y_pred are 0/1
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = _safe_div(TP + TN, TP + TN + FP + FN)
    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)
    fpr = _safe_div(FP, FP + TN)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
    }


def overall_from_tuned(scores_csv: Path) -> None:
    df = pd.read_csv(scores_csv)
    if "label_attack" not in df.columns:
        raise ValueError("scores file must contain label_attack")
    if "decision_tuned" not in df.columns:
        raise ValueError("scores file must contain decision_tuned (run tune_thresholds.py first)")

    y = df["label_attack"].astype(int)

    # Detection = any non-allow action
    pred_flag = df["decision_tuned"].isin(["stepup", "restrict", "deny"]).astype(int)
    m_det = _binary_metrics(y, pred_flag)

    # Access decision = hard actions only
    pred_hard = df["decision_tuned"].isin(["restrict", "deny"]).astype(int)
    m_hard = _binary_metrics(y, pred_hard)

    print("\n=== OVERALL (from outputs/results/afozt_scores_with_tuned_decisions.csv) ===")
    print(f"Rows: {len(df)} | Attacks: {int((y==1).sum())} | Benign: {int((y==0).sum())}")

    print("\nDetection (flagged = stepup/restrict/deny)")
    print(f"  Accuracy:  {m_det['accuracy']:.4f}")
    print(f"  Precision: {m_det['precision']:.4f}")
    print(f"  Recall:    {m_det['recall']:.4f}")
    print(f"  FPR:       {m_det['fpr']:.4f}")
    print(f"  TP={m_det['TP']} FP={m_det['FP']} TN={m_det['TN']} FN={m_det['FN']}")

    print("\nAccess decisions (hard = restrict/deny)  ← closest to “access-decision precision”")
    print(f"  Accuracy:  {m_hard['accuracy']:.4f}")
    print(f"  Precision: {m_hard['precision']:.4f}")
    print(f"  Recall:    {m_hard['recall']:.4f}")
    print(f"  FPR:       {m_hard['fpr']:.4f}")
    print(f"  TP={m_hard['TP']} FP={m_hard['FP']} TN={m_hard['TN']} FN={m_hard['FN']}")

    # Mix
    print("\nDecision mix (tuned):")
    print(df["decision_tuned"].value_counts(dropna=False).to_string())


def stagewise_from_rollout(rollout_csv: Path) -> None:
    df = pd.read_csv(rollout_csv)

    required = {"stage", "enforced", "effective_decision", "label_attack"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rollout_decisions.csv missing columns: {sorted(missing)}")

    df = df[df["enforced"].astype(bool)].copy()
    if df.empty:
        print("\n=== STAGE-WISE (rollout_decisions.csv) ===")
        print("No enforced rows found. (Did rollout run with enforcement enabled?)")
        return

    print("\n=== STAGE-WISE (from outputs/results/rollout_decisions.csv, enforced only) ===")
    for stage, g in df.groupby("stage", sort=False):
        y = g["label_attack"].astype(int)
        pred_hard = g["effective_decision"].isin(["restrict", "deny"]).astype(int)
        m = _binary_metrics(y, pred_hard)

        print(f"\nStage: {stage} | rows={len(g)} | attacks={int((y==1).sum())} | benign={int((y==0).sum())}")
        print("Access decisions (hard = restrict/deny)")
        print(f"  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  FPR: {m['fpr']:.4f}  Accuracy: {m['accuracy']:.4f}")
        print(f"  TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")

    # Also show per-stage decision mix for effective_decision
    print("\nEffective decision mix per stage (enforced only):")
    for stage, g in df.groupby("stage", sort=False):
        print(f"\nStage: {stage}")
        print(g["effective_decision"].value_counts(dropna=False).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Report AFO-ZT precision/accuracy/FPR overall + stage-wise.")
    ap.add_argument(
        "--scores",
        default="outputs/results/afozt_scores_with_tuned_decisions.csv",
        help="CSV produced by tune_thresholds.py",
    )
    ap.add_argument(
        "--rollout",
        default="outputs/results/rollout_decisions.csv",
        help="CSV produced by Step 2.7 safe rollout",
    )
    args = ap.parse_args()

    scores_csv = Path(args.scores)
    rollout_csv = Path(args.rollout)

    if scores_csv.exists():
        overall_from_tuned(scores_csv)
    else:
        print(f"⚠️ Missing overall scores file: {scores_csv} (run tune_thresholds.py)")

    if rollout_csv.exists():
        stagewise_from_rollout(rollout_csv)
    else:
        print(f"⚠️ Missing rollout decisions file: {rollout_csv} (run Step 2.7)")


if __name__ == "__main__":
    main()
