from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.afozt.safe_rollout import (
    RolloutConfig,
    SafeRolloutController,
    policy_compiler_config_from_thresholds_yaml,
)
from src.common.constants import RESULTS_DIR
from src.common.logging_setup import setup_logger


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.7: Safe rollout controller (shadow → canary → enforce → rollout).")
    parser.add_argument(
        "--scores",
        type=str,
        default=str(RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv"),
        help="Path to Step 2.5 scores CSV.",
    )
    parser.add_argument(
        "--rollout-config",
        type=str,
        default=str(REPO_ROOT / "configs" / "rollout.yaml"),
        help="Path to rollout.yaml.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=str(REPO_ROOT / "configs" / "thresholds.yaml"),
        help="Path to thresholds.yaml (used to align policy compiler thresholds).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="rollout",
        help="Output prefix (writes CSV/JSON into outputs/results).",
    )
    args = parser.parse_args()

    ensure_dirs()
    logger = setup_logger("run_safe_rollout")

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}. Run Step 2.5 first: python scripts/run_afozt.py")

    df = pd.read_csv(scores_path)
    logger.info(f"Loaded scores: {scores_path} (rows={len(df)}, cols={len(df.columns)})")

    rollout_cfg = RolloutConfig.from_yaml(args.rollout_config)
    policy_cfg = policy_compiler_config_from_thresholds_yaml(args.thresholds)

    controller = SafeRolloutController(rollout_cfg, policy_cfg=policy_cfg)
    timeline_df, decisions_df, metrics = controller.run(df)

    timeline_out = RESULTS_DIR / f"{args.out_prefix}_timeline.csv"
    decisions_out = RESULTS_DIR / f"{args.out_prefix}_decisions.csv"
    metrics_out = RESULTS_DIR / f"{args.out_prefix}_metrics.json"

    timeline_df.to_csv(timeline_out, index=False)
    decisions_df.to_csv(decisions_out, index=False)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✅ Wrote rollout timeline: {timeline_out}")
    logger.info(f"✅ Wrote rollout decisions: {decisions_out}")
    logger.info(f"✅ Wrote rollout metrics: {metrics_out}")

    # Print a tiny summary
    if not timeline_df.empty:
        last = timeline_df.iloc[-1].to_dict()
        logger.info(f"Final stage attempted: {last.get('stage')} (budgets_ok={last.get('budgets_ok')})")
        if last.get("reasons"):
            logger.info(f"Reasons: {last.get('reasons')}")


if __name__ == "__main__":
    main()
