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
    ap = argparse.ArgumentParser(description="Run comparison table across models (Step 3: starts with static baseline).")
    ap.add_argument("--static-scores", default=str(RESULTS_DIR / "baseline_static_scores.csv"))
    ap.add_argument("--afozt-scores", default=str(RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv"))
    ap.add_argument("--out-csv", default=str(RESULTS_DIR / "metrics_summary_step3.csv"))
    ap.add_argument("--out-confusions", default=str(RESULTS_DIR / "confusion_matrices_step3.json"))
    args = ap.parse_args()

    ensure_dirs()
    logger = setup_logger("run_all_and_compare")

    runs = []
    static_path = Path(args.static_scores)
    if static_path.exists():
        runs.append(BaselineRun(name="baseline_static", scores_csv=static_path))
    else:
        logger.warning(f"Missing static baseline scores: {static_path} (run: python scripts/run_baselines.py --baseline static)")

    afozt_path = Path(args.afozt_scores)
    if afozt_path.exists():
        runs.append(BaselineRun(name="afozt", scores_csv=afozt_path))
    else:
        logger.warning(f"Missing AFO-ZT tuned scores: {afozt_path} (run: python scripts/run_afozt.py then python scripts/tune_thresholds.py)")

    if not runs:
        raise FileNotFoundError("No score files found to compare.")

    df_summary = compare_runs(runs)
    out_csv = Path(args.out_csv)
    write_outputs(df_summary, out_csv)
    write_confusions(runs, Path(args.out_confusions))

    logger.info(f"Wrote: {out_csv}")
    logger.info(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
