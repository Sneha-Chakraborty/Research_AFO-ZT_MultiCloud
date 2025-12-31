from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import RESULTS_DIR
from src.evaluation.metrics import summarize_scores


TARGETS = {
    "det_accuracy_min": 0.964,            # >96.4%
    "det_fpr_max": 0.031,                 # <3.1%
    # NOTE: this is "access decision precision" (when it steps-up/restricts/denies, how often justified)
    "access_decision_precision_min": 0.946,  # >94.6%
    # Latency: uses orchestration response-time mean (seconds)
    "response_time_mean_s_max": 2.3,
}


def main() -> None:
    scores_path = RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv"
    latency_stats_path = RESULTS_DIR / "latency_stats.json"

    report: Dict[str, Any] = {"targets": TARGETS, "passed": True, "checks": {}}

    if scores_path.exists():
        df = pd.read_csv(scores_path)
        m = summarize_scores(df, decision_col="decision_tuned")

        # Backward-compatible alias (so older tooling still works)
        # "access_decision_precision" is the meaningful metric here.
        if "decision_precision" not in m and "access_decision_precision" in m:
            m["decision_precision"] = m["access_decision_precision"]

        report["metrics"] = m

        def _check(name: str, ok: bool, value: Any) -> None:
            report["checks"][name] = {"ok": bool(ok), "value": value}
            report["passed"] = bool(report["passed"] and ok)

        det_acc = float(m.get("det_accuracy", 0.0))
        det_fpr = float(m.get("det_fpr", 1.0))

        access_prec = float(m.get("access_decision_precision", m.get("decision_precision", 0.0)))
        hard_prec = float(m.get("hard_precision", 0.0))  # extra visibility (restrict/deny only)

        _check("det_accuracy", det_acc >= TARGETS["det_accuracy_min"], det_acc)
        _check("det_fpr", det_fpr <= TARGETS["det_fpr_max"], det_fpr)

        # Main target check: access decision precision
        _check(
            "access_decision_precision",
            access_prec >= TARGETS["access_decision_precision_min"],
            access_prec,
        )

        # Informational (does not affect PASS/FAIL)
        report["checks"]["hard_precision_info"] = {"ok": None, "value": hard_prec}

    else:
        report["passed"] = False
        report["checks"]["scores_file"] = {"ok": False, "value": f"Missing: {scores_path}"}

    if latency_stats_path.exists():
        try:
            stats = json.loads(latency_stats_path.read_text(encoding="utf-8"))
            report["latency_stats"] = stats

            overall = stats.get("overall", {})
            mean_s = float(overall.get("mean_s", stats.get("mean_s", 0.0)))

            ok = mean_s <= TARGETS["response_time_mean_s_max"]
            report["checks"]["response_time_mean_s"] = {"ok": ok, "value": mean_s}
            report["passed"] = bool(report["passed"] and ok)
        except Exception as e:
            report["checks"]["latency_stats"] = {"ok": False, "value": str(e)}
            report["passed"] = False
    else:
        report["checks"]["latency_stats"] = {"ok": False, "value": f"Missing: {latency_stats_path}"}
        report["passed"] = False

    out = RESULTS_DIR / "afozt_target_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    print("PASS" if report["passed"] else "FAIL")


if __name__ == "__main__":
    main()
