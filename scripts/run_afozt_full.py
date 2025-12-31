from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

# Allow `from src...` imports when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.constants import (
    DEFAULT_RAW_CSV,
    DEFAULT_UNIFIED_CSV,
    DEFAULT_TRUST_GRAPH_DB,
    DEFAULT_FEATURES_CSV,
    DEFAULT_FEATURES_PARQUET,
    RESULTS_DIR,
)
from src.common.logging_setup import setup_logger
from src.common.utils import ensure_dirs
from src.evaluation.metrics import summarize_scores


def _run(cmd: list[str], logger_name: str = "run") -> None:
    """Run a child python script with inherited stdout/stderr."""
    logger = setup_logger(logger_name)
    logger.info("▶ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """
    Step 5 — Your AFO-ZT (unified brain) end-to-end runner.

    Runs (best-effort):
      Step 2.2 Normalize -> Step 2.3 Trust graph -> Step 2.4 Features ->
      Step 2.5 Unified brain scoring -> Step 2.6 Threshold tuning ->
      Step 2.7 Safe rollout -> Step 2.8 Orchestration + latency.

    Notes:
      - This is CPU-friendly and intended for your offline synthetic dataset workflow.
      - Outputs are written under outputs/results and configs/.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DEFAULT_RAW_CSV), help="Raw CSV (Step 2.1 output).")
    ap.add_argument("--unified", default=str(DEFAULT_UNIFIED_CSV), help="Unified telemetry CSV (Step 2.2 output).")
    ap.add_argument("--db", default=str(DEFAULT_TRUST_GRAPH_DB), help="Trust graph sqlite db (Step 2.3 output).")
    ap.add_argument("--features", default="", help="Features file. If empty, uses processed/features.parquet/csv.")
    ap.add_argument("--train-frac", type=float, default=0.7, help="Time-ordered training fraction.")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild of normalize/graph/features even if files exist.")

    # Threshold tuning knobs (Step 2.6)
    ap.add_argument("--fpr-restrict", type=float, default=0.02)
    ap.add_argument("--fpr-deny", type=float, default=0.005)
    ap.add_argument("--mfa-rate", type=float, default=0.10)

    # Orchestration knobs (Step 2.8)
    ap.add_argument("--processing-delay-ms", type=float, default=0.0, help="Extra detection processing delay before actions start.")
    ap.add_argument("--include-shadow", action="store_true", help="Also execute actions for shadow rows (hypothetical).")

    args = ap.parse_args()
    ensure_dirs()
    logger = setup_logger("run_afozt_full")

    raw = Path(args.raw)
    unified = Path(args.unified)
    db = Path(args.db)

    # Step 2.2 — Normalize
    if args.rebuild or not unified.exists():
        _run([sys.executable, str(REPO_ROOT / "scripts" / "normalize_logs.py"), "--input", str(raw), "--output", str(unified)], "normalize")

    # Step 2.3 — Trust graph build
    if args.rebuild or not db.exists():
        _run([sys.executable, str(REPO_ROOT / "scripts" / "build_trust_graph.py"), "--input", str(unified), "--db", str(db), "--reset_db"], "trust_graph")

    # Step 2.4 — Features
    feat = Path(args.features) if args.features else (DEFAULT_FEATURES_PARQUET if DEFAULT_FEATURES_PARQUET.exists() else DEFAULT_FEATURES_CSV)
    if args.rebuild or not feat.exists():
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_features.py"),
                "--input",
                str(unified),
                "--db",
                str(db),
                "--out_csv",
                str(DEFAULT_FEATURES_CSV),
                "--out_parquet",
                str(DEFAULT_FEATURES_PARQUET),
            ],
            "features",
        )
        feat = DEFAULT_FEATURES_PARQUET if DEFAULT_FEATURES_PARQUET.exists() else DEFAULT_FEATURES_CSV

    # Step 2.5 — Unified brain scoring
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_afozt.py"),
            "--features",
            str(feat),
            "--train-frac",
            str(args.train_frac),
        ],
        "afozt",
    )

    # Step 2.6 — Tune thresholds
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tune_thresholds.py"),
            "--scores",
            str(RESULTS_DIR / "afozt_scores.csv"),
            "--fpr_restrict",
            str(args.fpr_restrict),
            "--fpr_deny",
            str(args.fpr_deny),
            "--mfa_rate",
            str(args.mfa_rate),
            "--write",
            str(REPO_ROOT / "configs" / "thresholds.yaml"),
        ],
        "thresholds",
    )

    # Step 2.7 — Safe rollout
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_safe_rollout.py"),
            "--scores",
            str(RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv"),
            "--thresholds",
            str(REPO_ROOT / "configs" / "thresholds.yaml"),
            "--rollout-config",
            str(REPO_ROOT / "configs" / "rollout.yaml"),
            "--out-prefix",
            "rollout",
        ],
        "rollout",
    )

    # Step 2.8 — Orchestration
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_orchestration.py"),
        "--decisions",
        str(RESULTS_DIR / "rollout_decisions.csv"),
        "--thresholds",
        str(REPO_ROOT / "configs" / "thresholds.yaml"),
        "--processing_delay_ms",
        str(args.processing_delay_ms),
    ]
    if args.include_shadow:
        cmd.append("--include_shadow")
    _run(cmd, "orchestration")

    # Final summary
    out: Dict[str, Any] = {}
    try:
        df_scores = pd.read_csv(RESULTS_DIR / "afozt_scores_with_tuned_decisions.csv")
        out["scores_metrics"] = summarize_scores(df_scores, decision_col="decision_tuned")
    except Exception as e:
        out["scores_metrics_error"] = str(e)

    try:
        stats_path = RESULTS_DIR / "latency_stats.json"
        if stats_path.exists():
            out["latency_stats"] = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception as e:
        out["latency_stats_error"] = str(e)

    out_path = RESULTS_DIR / "afozt_full_run_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()
