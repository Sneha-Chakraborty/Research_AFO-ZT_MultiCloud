from __future__ import annotations

from pathlib import Path
import pandas as pd


# --- Config ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = REPO_ROOT / "outputs" / "results" / "final_metrics_summary.csv"

# Engine name variants you might have in your summary file
AFOZT_NAMES = {"afozt", "afo-zt", "afo_zt", "afozt_unified", "afozt (unified)"}

# Metrics we care about + whether higher is better
# (Add/remove columns here if your summary uses different names)
METRICS = {
    # Common metrics
    "det_accuracy": True,
    "det_precision": True,
    "det_recall": True,
    "det_f1": True,
    "det_fpr": False,
    "access_decision_precision": True,
    "avg_response_time_s": False,
    "mttr_proxy_s": False,

    # Optional “hard action” metrics if present
    "hard_accuracy": True,
    "hard_precision": True,
    "hard_fpr": False,
    "hard_rate": False,  # lower is better (less disruption) IF you want; change to True if you want more blocking

    # Multi-cloud novelty metrics (if present)
    "cross_cloud_attack_detection_rate": True,
    "blast_radius_reduction": True,
    "policy_consistency_var": False,
    "canary_false_deny_rate": False,
    "rollout_false_deny_rate": False,
}

# If you want to treat hard_rate as neutral (not “best”), delete it from METRICS.


def guess_engine_col(df: pd.DataFrame) -> str:
    for c in ["engine", "model", "system", "name", "baseline"]:
        if c in df.columns:
            return c
    return df.columns[0]


def is_seconds(metric: str) -> bool:
    return metric.endswith("_s") or "time" in metric or "mttr" in metric


def is_ratio_like(metric: str) -> bool:
    # treat as 0..1 fraction -> percent formatting
    keys = ["accuracy", "precision", "recall", "f1", "fpr", "rate", "reduction"]
    return any(k in metric for k in keys) and not is_seconds(metric) and "var" not in metric


def fmt(metric: str, v: float) -> str:
    if pd.isna(v):
        return "NA"
    if is_seconds(metric):
        return f"{float(v):.3f} s"
    if is_ratio_like(metric):
        x = float(v)
        # If already in 0..100 range, keep as percent; else convert 0..1 → %
        if x <= 1.0:
            x *= 100.0
        return f"{x:.2f}%"
    # variance / counts / misc
    return f"{float(v):.6g}"


def main():
    if not SUMMARY_CSV.exists():
        raise SystemExit(
            f"Missing {SUMMARY_CSV}\n"
            f"Run: python scripts/run_final_metrics.py --only-non-allow-latency --write-latencies\n"
            f"then rerun this script."
        )

    df = pd.read_csv(SUMMARY_CSV)
    engine_col = guess_engine_col(df)
    df[engine_col] = df[engine_col].astype(str)

    # Find AFO-ZT row
    def norm_name(s: str) -> str:
        return s.strip().lower()

    df["_engine_norm"] = df[engine_col].map(norm_name)
    afo_mask = df["_engine_norm"].isin(AFOZT_NAMES)

    if afo_mask.sum() == 0:
        raise SystemExit(
            f"Could not find AFO-ZT row. I searched for: {sorted(AFOZT_NAMES)}\n"
            f"Engines present: {df[engine_col].tolist()}"
        )

    # If multiple AFO-ZT rows exist, take the first
    afo_row = df[afo_mask].iloc[0]
    afo_name = afo_row[engine_col]

    # Coerce all metric columns numeric where possible
    for col in METRICS.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    best_lines = []
    missing = []

    for metric, higher_is_better in METRICS.items():
        if metric not in df.columns:
            missing.append(metric)
            continue

        series = df[[engine_col, metric]].dropna()
        if series.empty:
            continue

        # Determine best row (ties allowed)
        sorted_series = series.sort_values(metric, ascending=not higher_is_better)
        best_val = float(sorted_series.iloc[0][metric])

        # Check if AFO-ZT is among best (tie included)
        afo_val = float(afo_row[metric]) if not pd.isna(afo_row[metric]) else None
        if afo_val is None:
            continue

        # Consider numeric tie tolerance
        tol = 1e-12
        is_best = (abs(afo_val - best_val) <= tol)

        if is_best:
            # next best baseline (excluding AFO-ZT)
            others = sorted_series[sorted_series[engine_col] != afo_name]
            next_best = None
            if not others.empty:
                next_best = others.iloc[0]
            margin = None
            if next_best is not None:
                nb_val = float(next_best[metric])
                margin = (afo_val - nb_val) if higher_is_better else (nb_val - afo_val)

            best_lines.append((metric, afo_val, next_best, margin, higher_is_better))

    # Print
    print("\n=== Metrics where AFO-ZT is BEST vs all baselines ===\n")
    print(f"AFO-ZT engine name detected as: {afo_name}\n")

    if not best_lines:
        print("No metrics found where AFO-ZT is strictly best (or CSV missing those columns).")
        return

    # Sort by metric name for readability
    for metric, afo_val, next_best, margin, hib in sorted(best_lines, key=lambda x: x[0]):
        line = f"- {metric}: AFO-ZT = {fmt(metric, afo_val)}"
        if next_best is not None:
            nb_name = str(next_best[engine_col])
            nb_val = float(next_best[metric])
            line += f" | next best = {nb_name} ({fmt(metric, nb_val)})"
            if margin is not None and not pd.isna(margin):
                # margin formatting: same unit style as metric
                line += f" | margin = {fmt(metric, margin) if not is_seconds(metric) else f'{margin:.3f} s'}"
        print(line)

    if missing:
        print("\n[Note] These metrics were not found in final_metrics_summary.csv (ok if you didn’t compute them):")
        print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
