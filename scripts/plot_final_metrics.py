"""
plot_final_metrics.py
---------------------
Generates publication-friendly PNG plots comparing AFO-ZT vs baselines.

Expects:
  outputs/results/final_metrics_summary.csv

Produces:
  outputs/figures/*.png
  outputs/figures/leaderboard.md

Usage (from project root):
  python scripts/plot_final_metrics.py

Notes:
  - Uses matplotlib only (no seaborn).
  - Saves one figure per metric (no subplots) for easy insertion into paper.
  - If some metrics are missing in the CSV, it skips those plots.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "outputs" / "results"
FIGS = REPO_ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def _guess_engine_col(df: pd.DataFrame) -> str:
    for c in ["engine", "model", "system", "name", "baseline"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _safe_float_series(df: pd.DataFrame, col: str):
    return pd.to_numeric(df[col], errors="coerce")


def _plot_bar(df: pd.DataFrame, engine_col: str, metric_col: str, title: str, ylabel: str, higher_is_better: bool):
    engines = df[engine_col].astype(str).tolist()
    values = _safe_float_series(df, metric_col).tolist()

    fig = plt.figure()
    ax = plt.gca()
    bars = ax.bar(engines, values)

    # Hatch AFO-ZT bar if present (no custom colors)
    for i, e in enumerate(engines):
        if e.lower() in {"afozt", "afo-zt", "afo_zt", "afozt_unified"}:
            bars[i].set_hatch("///")
            ax.text(i, values[i], "  AFO-ZT", rotation=90, va="bottom")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(engines)))
    ax.set_xticklabels(engines, rotation=25, ha="right")

    # Mark best
    best_idx = None
    best_val = None
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if best_val is None:
            best_val = v
            best_idx = i
        else:
            if higher_is_better and v > best_val:
                best_val, best_idx = v, i
            if (not higher_is_better) and v < best_val:
                best_val, best_idx = v, i
    if best_idx is not None:
        ax.text(best_idx, values[best_idx], "  ★ best", rotation=90, va="bottom")

    fig.tight_layout()
    out = FIGS / f"metric_{metric_col}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    path = RESULTS / "final_metrics_summary.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python scripts/run_final_metrics.py first.")

    df = pd.read_csv(path)
    engine_col = _guess_engine_col(df)
    df[engine_col] = df[engine_col].astype(str)

    metric_specs = [
        # Common metrics
        ("det_accuracy", "Detection Accuracy", "Accuracy", True),
        ("det_precision", "Detection Precision", "Precision", True),
        ("det_recall", "Detection Recall", "Recall", True),
        ("det_f1", "Detection F1", "F1", True),
        ("det_fpr", "Detection False Positive Rate", "FPR", False),
        ("access_decision_precision", "Access Decision Precision", "Precision", True),
        ("avg_response_time_s", "Average Response Time", "Seconds", False),
        ("mttr_proxy_s", "MTTR Proxy (detect→action)", "Seconds", False),

        # AFO-ZT novelty metrics
        ("cross_cloud_attack_detection_rate", "Cross-cloud Attack Detection Rate", "Rate", True),
        ("blast_radius_reduction", "Blast Radius Reduction (graph containment)", "Reduction", True),
        ("policy_consistency_var", "Policy Consistency Variance Across Clouds", "Variance", False),
        ("canary_false_deny_rate", "Rollout Safety: Canary False-Deny Rate", "Rate", False),
        ("rollout_false_deny_rate", "Rollout Safety: Global False-Deny Rate", "Rate", False),
    ]

    written = []
    for col, title, ylabel, hib in metric_specs:
        if col in df.columns:
            out = _plot_bar(df, engine_col, col, title, ylabel, hib)
            written.append(str(out.relative_to(REPO_ROOT)))

    # Leaderboard markdown (best per metric)
    md = []
    md.append("# Final Metrics Leaderboard\n\n")
    md.append(f"Source: `{path.relative_to(REPO_ROOT)}`\n\n")
    md.append("## Plots generated\n")
    for p in written:
        md.append(f"- `{p}`\n")

    md.append("\n## Best per metric\n")
    for col, title, _, hib in metric_specs:
        if col not in df.columns:
            continue
        tmp = df[[engine_col, col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()
        if len(tmp) == 0:
            continue
        tmp = tmp.sort_values(col, ascending=not hib)
        best = tmp.iloc[0]
        md.append(f"- **{title}**: `{best[engine_col]}` = `{best[col]}`\n")

    (FIGS / "leaderboard.md").write_text("".join(md), encoding="utf-8")

    print("Wrote plots to:", FIGS)
    print("Wrote:", FIGS / "leaderboard.md")


if __name__ == "__main__":
    main()
