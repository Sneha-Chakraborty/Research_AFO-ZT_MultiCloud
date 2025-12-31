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
from src.baselines.raac_iforest import RaacIForestBaseline, RaacIForestConfig
from src.baselines.per_cloud_brains import PerCloudBrainsBaseline, PerCloudBrainsConfig
from src.common.constants import DEFAULT_UNIFIED_CSV, MODELS_DIR, RESULTS_DIR, PROCESSED_DIR
from src.common.logging_setup import setup_logger
from src.common.utils import ensure_dirs


def run_static(in_csv: Path, out_csv: Path, model_out: Path) -> None:
    df = pd.read_csv(in_csv)
    baseline = StaticRulesBaseline(StaticRulesConfig())
    scores = baseline.score(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_csv, index=False)
    joblib.dump({"baseline": "static_rules", "config": baseline.cfg.__dict__}, model_out)


def run_siem(in_csv: Path, out_csv: Path, model_out: Path) -> None:
    df = pd.read_csv(in_csv)
    baseline = SiemRulesBaseline(SiemRulesConfig())
    scores = baseline.score(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_csv, index=False)
    joblib.dump({"baseline": "siem_rules", "config": baseline.cfg.__dict__}, model_out)


def run_raac_iforest(in_csv: Path, out_csv: Path, model_out: Path, *, train_csv: Path | None = None) -> None:
    df = pd.read_csv(in_csv)

    # Best-effort: if train split exists, fit on it; else fit on full df
    df_train = None
    if train_csv is not None and train_csv.exists():
        df_train = pd.read_csv(train_csv)
    else:
        default_train = PROCESSED_DIR / "train.csv"
        if default_train.exists():
            df_train = pd.read_csv(default_train)

    baseline = RaacIForestBaseline(RaacIForestConfig())
    if df_train is not None:
        baseline.fit(df_train)

    scores = baseline.score(df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_csv, index=False)

    joblib.dump(baseline.artifact, model_out)


def run_per_cloud(in_csv: Path, out_csv: Path, model_out: Path, *, train_csv: Path | None = None) -> None:
    """Step 4 baseline: per-cloud brains (separate analytics per provider)."""
    df = pd.read_csv(in_csv)

    df_train = None
    if train_csv is not None and train_csv.exists():
        df_train = pd.read_csv(train_csv)
    else:
        default_train = PROCESSED_DIR / "train.csv"
        if default_train.exists():
            df_train = pd.read_csv(default_train)

    baseline = PerCloudBrainsBaseline(PerCloudBrainsConfig())
    baseline.fit(df_train if df_train is not None else df)
    scores = baseline.score(df, df_train=df_train)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_csv, index=False)

    # Main artifact
    joblib.dump(baseline.artifact, model_out)

    # Convenience: dump provider-specific artifacts too
    try:
        for provider, model in baseline.models.items():
            joblib.dump(model.artifact, MODELS_DIR / f"baseline_per_cloud_{provider}.pkl")
        if baseline.global_model is not None:
            joblib.dump(baseline.global_model.artifact, MODELS_DIR / "baseline_per_cloud_global_fallback.pkl")
    except Exception:
        # Provider dumps are best-effort; don't fail the whole run.
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baselines (Step 3 + Step 4).")
    ap.add_argument(
        "--baseline",
        choices=["static", "siem", "raac_iforest", "per_cloud"],
        default="static",
        help="Which baseline to run.",
    )
    ap.add_argument("--in", dest="in_csv", default=str(DEFAULT_UNIFIED_CSV), help="Input unified telemetry CSV.")
    ap.add_argument("--train", dest="train_csv", default=None, help="Optional train CSV for ML baselines.")
    ap.add_argument("--out", dest="out_csv", default=None, help="Output scores CSV (default depends on baseline).")
    ap.add_argument("--model-out", default=None, help="Where to store baseline artifact (default depends on baseline).")
    args = ap.parse_args()

    ensure_dirs()
    logger = setup_logger("run_baselines")

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}. Run Step 2.2 normalize first.")

    train_csv = Path(args.train_csv) if args.train_csv else None

    # baseline-specific defaults
    if args.baseline == "static":
        out_csv = Path(args.out_csv) if args.out_csv else (RESULTS_DIR / "baseline_static_scores.csv")
        model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / "baseline_static.pkl")
    elif args.baseline == "siem":
        out_csv = Path(args.out_csv) if args.out_csv else (RESULTS_DIR / "baseline_siem_scores.csv")
        model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / "baseline_siem.pkl")
    elif args.baseline == "raac_iforest":
        out_csv = Path(args.out_csv) if args.out_csv else (RESULTS_DIR / "baseline_raac_iforest_scores.csv")
        model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / "baseline_raac_iforest.pkl")
    else:
        out_csv = Path(args.out_csv) if args.out_csv else (RESULTS_DIR / "baseline_per_cloud_scores.csv")
        model_out = Path(args.model_out) if args.model_out else (MODELS_DIR / "baseline_per_cloud.pkl")

    logger.info(f"Running baseline={args.baseline} on {in_csv} -> {out_csv}")

    if args.baseline == "static":
        run_static(in_csv, out_csv, model_out)
    elif args.baseline == "siem":
        run_siem(in_csv, out_csv, model_out)
    elif args.baseline == "raac_iforest":
        run_raac_iforest(in_csv, out_csv, model_out, train_csv=train_csv)
    elif args.baseline == "per_cloud":
        run_per_cloud(in_csv, out_csv, model_out, train_csv=train_csv)
    else:
        raise ValueError(f"Unsupported baseline: {args.baseline}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
