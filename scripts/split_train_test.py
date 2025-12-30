from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import DEFAULT_UNIFIED_CSV, PROCESSED_DIR
from src.common.logging_setup import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser(description="Split unified telemetry into time-ordered train/test CSV.")
    ap.add_argument("--in", dest="in_csv", default=str(DEFAULT_UNIFIED_CSV))
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--train-out", default=str(PROCESSED_DIR / "train.csv"))
    ap.add_argument("--test-out", default=str(PROCESSED_DIR / "test.csv"))
    args = ap.parse_args()

    logger = setup_logger("split_train_test")

    in_csv = Path(args.in_csv)
    df = pd.read_csv(in_csv)
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)

    n = len(df)
    k = int(n * float(args.train_frac))
    train = df.iloc[:k].copy()
    test = df.iloc[k:].copy()

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_out, index=False)
    Path(args.test_out).parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(args.test_out, index=False)

    logger.info(f"Wrote train={len(train)} -> {args.train_out}")
    logger.info(f"Wrote test={len(test)} -> {args.test_out}")


if __name__ == "__main__":
    main()
