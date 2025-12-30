from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import RESULTS_DIR
from src.common.logging_setup import setup_logger
from src.common.utils import ensure_dirs
from src.evaluation.compare_baselines import BaselineRun, compare_runs, write_confusions, write_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baseline runs + AFO-ZT outputs (Step 3).")
    ap.add_argument("--static-scores", default=str(RESULTS_DIR / "baseline_static_scores.csv"))
    ap.add_argument("--siem-scores", default=str(RESULTS_DIR / "baseline_siem_scores.csv"))
    ap.add_argument("--raac-scores", default=str(RESULTS_DIR / "baseline_raac_iforest_scores.csv"))
    ap.add_argument("--afozt-scores", default=str(RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv"))
    ap.add_argument("--out-summary", default=str(RESULTS_DIR / "metrics_summary_step3.csv"))
    ap.add_argument("--out-confusions", default=str(RESULTS_DIR / "confusion_matrices_step3.json"))
    args = ap.parse_args()

    ensure_dirs()
    logger = setup_logger("run_all_and_compare")

    runs = []

    static_path = Path(args.static_scores)
    if static_path.exists():
        runs.append(BaselineRun(name="baseline_static", scores_csv=static_path))
    else:
        logger.warning(
            f"Missing static baseline scores: {static_path} "
            f"(run: python scripts/run_baselines.py --baseline static)"
        )

    siem_path = Path(args.siem_scores)
    if siem_path.exists():
        runs.append(BaselineRun(name="baseline_siem", scores_csv=siem_path))
    else:
        logger.warning(
            f"Missing SIEM baseline scores: {siem_path} "
            f"(run: python scripts/run_baselines.py --baseline siem)"
        )

    raac_path = Path(args.raac_scores)
    if raac_path.exists():
        runs.append(BaselineRun(name="baseline_raac_iforest", scores_csv=raac_path))
    else:
        logger.warning(
            f"Missing RAAC+IForest baseline scores: {raac_path} "
            f"(run: python scripts/run_baselines.py --baseline raac_iforest)"
        )

    afozt_path = Path(args.afozt_scores)
    if afozt_path.exists():
        runs.append(BaselineRun(name="afozt", scores_csv=afozt_path))
    else:
        logger.warning(
            f"Missing AFO-ZT scores: {afozt_path} "
            f"(run: python scripts/run_afozt.py, then scripts/report_precision.py)"
        )

    if not runs:
        raise SystemExit("No runs found. Generate at least one scores CSV first.")

    out_summary = Path(args.out_summary)
    out_conf = Path(args.out_confusions)

    df_summary = compare_runs(runs)
    write_outputs(df_summary, out_summary)
    write_confusions(runs, out_conf)

    logger.info(f"Wrote: {out_summary}")
    logger.info(f"Wrote: {out_conf}")


if __name__ == "__main__":
    main()
