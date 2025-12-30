from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.siem_rules import SiemRulesBaseline, SiemRulesConfig
from src.baselines.static_rules import StaticRulesBaseline, StaticRulesConfig
from src.common.constants import DEFAULT_UNIFIED_CSV, MODELS_DIR, RESULTS_DIR
from src.common.logging_setup import setup_logger
from src.common.utils import ensure_dirs


def run_static(in_csv: Path, out_csv: Path, model_out: Path) -> None:
    df = pd.read_csv(in_csv)
    baseline = StaticRulesBaseline(StaticRulesConfig())

    scored = baseline.score(df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"baseline": "static_rules", "config": StaticRulesConfig().__dict__}, model_out)


def run_siem(in_csv: Path, out_csv: Path, model_out: Path) -> None:
    df = pd.read_csv(in_csv)
    baseline = SiemRulesBaseline(SiemRulesConfig())

    scored = baseline.score(df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_csv, index=False)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline.artifact(), model_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: Run baselines on unified telemetry.")
    ap.add_argument(
        "--baseline",
        choices=["static", "siem"],
        default="static",
        help="Which baseline to run.",
    )
    ap.add_argument("--in", dest="in_csv", default=str(DEFAULT_UNIFIED_CSV), help="Input unified telemetry CSV.")
    ap.add_argument(
        "--out",
        dest="out_csv",
        default=None,
        help="Output scores CSV (default depends on baseline).",
    )
    ap.add_argument(
        "--model-out",
        default=None,
        help="Where to store baseline artifact (default depends on baseline).",
    )
    args = ap.parse_args()

    ensure_dirs()
    logger = setup_logger("run_baselines")

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}. Run Step 2.2 normalize first.")

    # baseline-specific defaults
    if args.baseline == "static":
        out_csv = Path(args.out_csv) if args.out_csv else (RESULTS_DIR / "baseline_static_scores.csv")
        model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / "baseline_static.pkl")
    else:
        out_csv = Path(args.out_csv) if args.out_csv else (RESULTS_DIR / "baseline_siem_scores.csv")
        model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / "baseline_siem.pkl")

    logger.info(f"Running baseline={args.baseline} on {in_csv} -> {out_csv}")

    if args.baseline == "static":
        run_static(in_csv, out_csv, model_out)
    elif args.baseline == "siem":
        run_siem(in_csv, out_csv, model_out)
    else:
        raise ValueError(f"Unsupported baseline: {args.baseline}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
