from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.logging_setup import setup_logger
from src.evaluation.final_metrics import EngineRun, compute_engine_metrics
from src.evaluation.response_time import compute_response_times, load_latency_profiles


RESULTS_DIR = REPO_ROOT / "outputs" / "results"


def _default_runs() -> List[EngineRun]:
    return [
        EngineRun(name="baseline_static", scores_csv=RESULTS_DIR / "baseline_static_scores.csv"),
        EngineRun(name="baseline_siem", scores_csv=RESULTS_DIR / "baseline_siem_scores.csv"),
        EngineRun(name="baseline_raac_iforest", scores_csv=RESULTS_DIR / "baseline_raac_iforest_scores.csv"),
        EngineRun(name="baseline_per_cloud_brains", scores_csv=RESULTS_DIR / "baseline_per_cloud_scores.csv"),
        EngineRun(
            name="afozt",
            scores_csv=RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv",
            rollout_metrics_json=RESULTS_DIR / "rollout_metrics.json",
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Final step — compute strong/common metrics + multi-cloud novelty metrics across all baselines and AFO-ZT.")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml", help="Thresholds YAML (used for plan compilation + consistency).")
    ap.add_argument("--latency-profiles", default="configs/latency_profiles.yaml", help="Extra processing delays per engine.")
    ap.add_argument("--out-summary", default=str(RESULTS_DIR / "final_metrics_summary.csv"), help="Output CSV summary.")
    ap.add_argument("--out-json", default=str(RESULTS_DIR / "final_metrics_rows.json"), help="Output JSON rows.")
    ap.add_argument("--only-non-allow-latency", action="store_true", help="Compute response time/MTTR only on stepup/restrict/deny rows (recommended).")
    ap.add_argument("--write-latencies", action="store_true", help="Also write per-engine response_times_*.csv and latency_stats_*.json to outputs/results.")
    ap.add_argument("--max-latency-rows", type=int, default=0, help="Optional cap to speed up latency sim (0 = no cap).")
    args = ap.parse_args()

    logger = setup_logger("final_metrics", RESULTS_DIR / "final_metrics.log")

    out_summary = Path(args.out_summary)
    out_json = Path(args.out_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    runs = _default_runs()
    usable: List[EngineRun] = []
    for r in runs:
        if r.scores_csv.exists():
            usable.append(r)
        else:
            logger.warning(f"Missing scores for {r.name}: {r.scores_csv} (run the corresponding baseline/AFO-ZT pipeline first)")

    if not usable:
        raise FileNotFoundError("No score files found. Run baselines and AFO-ZT first (see README).")

    max_rows = args.max_latency_rows if args.max_latency_rows and args.max_latency_rows > 0 else None
    rows = []
    for r in usable:
        m = compute_engine_metrics(
            r,
            thresholds_yaml=args.thresholds,
            latency_profiles_yaml=args.latency_profiles,
            only_non_allow_for_latency=bool(args.only_non_allow_latency),
            max_latency_rows=max_rows,
        )
        m["engine"] = r.name
        rows.append(m)

        if args.write_latencies:
            df = pd.read_csv(r.scores_csv)
            profiles = load_latency_profiles(args.latency_profiles)
            extra = float(profiles.get(r.name).processing_delay_s) if r.name in profiles else 0.0
            rt_df, stats = compute_response_times(
                df,
                thresholds_yaml=args.thresholds,
                decision_col=r.decision_col,
                only_non_allow=bool(args.only_non_allow_latency),
                extra_processing_delay_s=extra,
                max_rows=max_rows,
            )
            rt_out = RESULTS_DIR / f"response_times_{r.name}.csv"
            st_out = RESULTS_DIR / f"latency_stats_{r.name}.json"
            rt_df.to_csv(rt_out, index=False)
            st_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            logger.info(f"Wrote: {rt_out}")
            logger.info(f"Wrote: {st_out}")

    df_out = pd.DataFrame(rows)
    first = ["engine", "rows", "attacks", "benign"]
    rest = [c for c in df_out.columns if c not in first]
    df_out = df_out[first + rest]
    df_out.to_csv(out_summary, index=False)
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    logger.info(f"Wrote: {out_summary}")
    logger.info(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
