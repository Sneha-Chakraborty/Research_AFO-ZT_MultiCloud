from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "results"


DEFAULT_TARGETS = {
    "detection_accuracy_min": 0.964,      # 96.4%
    "false_positive_rate_max": 0.031,     # 3.1%
    "avg_response_time_s_max": 2.3,
    "access_decision_precision_min": 0.946,  # 94.6%
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Check AFO-ZT targets from final_metrics_summary.csv")
    ap.add_argument("--summary", default=str(RESULTS_DIR / "final_metrics_summary.csv"))
    ap.add_argument("--engine", default="afozt")
    ap.add_argument("--out", default=str(RESULTS_DIR / "final_targets_check.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    row = df[df["engine"] == args.engine]
    if row.empty:
        raise ValueError(f"Engine not found in summary: {args.engine}")

    r = row.iloc[0].to_dict()

    # Use hard metrics (restrict/deny) for detection-like reporting
    det_acc = float(r.get("hard_accuracy", 0.0))
    fpr = float(r.get("hard_fpr", 1.0))
    avg_rt = float(r.get("avg_response_time_s", 9999.0))
    adp = float(r.get("access_decision_precision", 0.0))

    checks = {
        "detection_accuracy": {"value": det_acc, "target": f">= {DEFAULT_TARGETS['detection_accuracy_min']}", "pass": det_acc >= DEFAULT_TARGETS["detection_accuracy_min"]},
        "false_positive_rate": {"value": fpr, "target": f"<= {DEFAULT_TARGETS['false_positive_rate_max']}", "pass": fpr <= DEFAULT_TARGETS["false_positive_rate_max"]},
        "avg_response_time_s": {"value": avg_rt, "target": f"<= {DEFAULT_TARGETS['avg_response_time_s_max']}", "pass": avg_rt <= DEFAULT_TARGETS["avg_response_time_s_max"]},
        "access_decision_precision": {"value": adp, "target": f">= {DEFAULT_TARGETS['access_decision_precision_min']}", "pass": adp >= DEFAULT_TARGETS["access_decision_precision_min"]},
    }

    overall = all(v["pass"] for v in checks.values())

    out = {
        "engine": args.engine,
        "overall_pass": overall,
        "checks": checks,
        "note": "Targets are evaluated on HARD decisions (restrict/deny) for accuracy/FPR; adjust if you prefer flagged metrics.",
    }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("PASS" if overall else "FAIL")
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()
