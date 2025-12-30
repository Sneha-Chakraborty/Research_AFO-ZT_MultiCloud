from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/results/afozt_scores.csv")
    ap.add_argument("--fpr_restrict", type=float, default=0.02, help="Benign fraction allowed to become restrict/deny.")
    ap.add_argument("--fpr_deny", type=float, default=0.005, help="Benign fraction allowed to become deny.")
    ap.add_argument("--mfa_rate", type=float, default=0.10, help="Overall stepup target (approx, using benign quantile).")
    ap.add_argument("--write", default="configs/thresholds.yaml", help="Where to write thresholds YAML.")
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    if "risk" not in df.columns or "label_attack" not in df.columns:
        raise ValueError("scores file must contain columns: risk, label_attack")

    benign = df[df["label_attack"] == 0].copy()
    attack = df[df["label_attack"] == 1].copy()

    if benign.empty or attack.empty:
        raise ValueError("Need both benign and attack rows to tune thresholds.")

    # Pick thresholds from BENIGN distribution (so we can control false actions)
    # restrict threshold = (1 - fpr_restrict) quantile of benign risk
    t_restrict = float(benign["risk"].quantile(1.0 - args.fpr_restrict))
    t_deny = float(benign["risk"].quantile(1.0 - args.fpr_deny))

    # Stepup threshold: use benign quantile so roughly top mfa_rate of benign get stepup-or-worse
    # (You can tweak later; this is a solid starting point.)
    t_stepup = float(benign["risk"].quantile(1.0 - args.mfa_rate))

    # Allow threshold: below stepup threshold is allow
    t_allow = t_stepup

    # Evaluate implied decisions
    def decide(r: float) -> str:
        if r >= t_deny:
            return "deny"
        if r >= t_restrict:
            return "restrict"
        if r >= t_stepup:
            return "stepup"
        return "allow"

    df["decision_tuned"] = df["risk"].map(decide)

    # Metrics
    pred_severe = df["decision_tuned"].isin(["restrict", "deny"])
    pred_deny = df["decision_tuned"].eq("deny")
    y = df["label_attack"].astype(int)

    tpr_severe = float((pred_severe & (y == 1)).sum() / max(1, (y == 1).sum()))
    fpr_severe = float((pred_severe & (y == 0)).sum() / max(1, (y == 0).sum()))
    tpr_deny = float((pred_deny & (y == 1)).sum() / max(1, (y == 1).sum()))
    fpr_deny = float((pred_deny & (y == 0)).sum() / max(1, (y == 0).sum()))

    stepup_rate = float(df["decision_tuned"].eq("stepup").mean())
    restrict_rate = float(df["decision_tuned"].eq("restrict").mean())
    deny_rate = float(df["decision_tuned"].eq("deny").mean())
    allow_rate = float(df["decision_tuned"].eq("allow").mean())

    print("\nTUNED THRESHOLDS (from benign quantiles)")
    print(f"allow_lt  = {t_allow:.6f}")
    print(f"stepup_lt = {t_stepup:.6f}")
    print(f"restrict_lt = {t_restrict:.6f}")
    print(f"deny_lt   = {t_deny:.6f}")

    print("\nDECISION MIX (overall)")
    print(f"allow={allow_rate:.3f} stepup={stepup_rate:.3f} restrict={restrict_rate:.3f} deny={deny_rate:.3f}")

    print("\nSEVERE (restrict/deny) detection metrics")
    print(f"TPR={tpr_severe:.3f}  FPR={fpr_severe:.3f}")

    print("\nDENY-only detection metrics")
    print(f"TPR={tpr_deny:.3f}  FPR={fpr_deny:.3f}")

    # Write YAML thresholds for Step 2.6/2.7 policy compiler
    out_path = Path(args.write)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                f"allow_lt: {t_allow:.6f}",
                f"stepup_lt: {t_stepup:.6f}",
                f"restrict_lt: {t_restrict:.6f}",
                f"deny_lt: {t_deny:.6f}",
                "",
                "# Confidence gating (tune later; your confidence median ~0.27)",
                "min_conf_for_hard_actions: 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n✅ Wrote tuned thresholds to: {out_path}")

    # Optional: write a preview CSV
    preview_path = Path("outputs/results/afozt_scores_with_tuned_decisions.csv")
    df.to_csv(preview_path, index=False)
    print(f"✅ Wrote preview decisions to: {preview_path}")


if __name__ == "__main__":
    main()
