from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.afozt.decision_brain import BrainConfig, UnifiedDecisionBrain
from src.common.constants import (
    DEFAULT_FEATURES_CSV,
    DEFAULT_FEATURES_PARQUET,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.common.logging_setup import setup_logger


def _load_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _time_split(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    if "ts" in d.columns:
        d["ts_dt"] = pd.to_datetime(d["ts"], errors="coerce")
        d = d.sort_values("ts_dt").reset_index(drop=True)
    n = len(d)
    cut = int(train_frac * n)
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2.5: AFO-ZT unified brain risk scoring with confidence + rationale.")
    parser.add_argument("--features", type=str, default="", help="Path to features parquet/csv. Default: processed/features.parquet (if exists) else features.csv")
    parser.add_argument("--model-out", type=str, default=str(MODELS_DIR / "afozt_unified.pkl"), help="Where to save trained brain artifact.")
    parser.add_argument("--scores-out", type=str, default=str(RESULTS_DIR / "afozt_scores.csv"), help="Where to write scored events.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Time-ordered train fraction.")
    parser.add_argument("--no-train", action="store_true", help="Skip training and load model from --model-out.")
    args = parser.parse_args()

    logger = setup_logger("run_afozt")

    # choose default features file
    feat_path: Optional[Path] = None
    if args.features:
        feat_path = Path(args.features)
    else:
        if Path(DEFAULT_FEATURES_PARQUET).exists():
            feat_path = Path(DEFAULT_FEATURES_PARQUET)
        else:
            feat_path = Path(DEFAULT_FEATURES_CSV)

    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}. Run Step 2.4 first (scripts/build_features.py).")

    df = _load_features(feat_path)
    logger.info(f"Loaded features: {feat_path} (rows={len(df)}, cols={len(df.columns)})")

    model_out = Path(args.model_out)
    scores_out = Path(args.scores_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    scores_out.parent.mkdir(parents=True, exist_ok=True)

    if args.no_train and model_out.exists():
        artifact = joblib.load(model_out)
        brain = UnifiedDecisionBrain.from_artifact(artifact)
        logger.info(f"Loaded model artifact: {model_out}")
        df_train, df_test = _time_split(df, train_frac=args.train_frac)
    else:
        df_train, df_test = _time_split(df, train_frac=args.train_frac)
        brain = UnifiedDecisionBrain(BrainConfig())
        brain.fit(df_train)
        joblib.dump(brain.to_artifact(), model_out)
        logger.info(f"✅ Saved model artifact: {model_out}")

    # Score the full dataset (train+test) for downstream policy/rollout steps.
    out = brain.score(df)

    keep_cols = [c for c in [
        "ts", "principal_id", "principal_type", "role", "device_id", "workload_id", "session_id",
        "token_id", "cloud_provider", "tenant_id", "region", "resource_id", "resource_type",
        "resource_sensitivity", "action", "api", "operation", "access_result", "latency_ms", "bytes_out",
        "label_attack", "attack_type"
    ] if c in df.columns]

    scored = df[keep_cols].copy()
    scored["s_anom"] = out.s_anom
    scored["s_rule"] = out.s_rule
    scored["s_graph"] = out.s_graph
    scored["risk"] = out.risk
    scored["confidence"] = out.confidence
    scored["decision"] = out.decision
    scored["rationale_tags"] = out.rationale_tags
    scored["rationale_text"] = out.rationale_text

    scored.to_csv(scores_out, index=False)
    logger.info(f"✅ Wrote Step 2.5 scored outputs: {scores_out} (rows={len(scored)})")

    # Quick sanity stats
    if "label_attack" in scored.columns:
        y = pd.to_numeric(scored["label_attack"], errors="coerce").fillna(0).astype(int)
        # Simple: how many attacks got restrict/deny?
        hit = scored["decision"].isin(["restrict", "deny"]).astype(int)
        tpr = float((hit[y == 1].sum() / max(1, (y == 1).sum())))
        fpr = float((hit[y == 0].sum() / max(1, (y == 0).sum())))
        logger.info(f"Sanity: restrict-or-deny TPR={tpr:.3f} FPR={fpr:.3f}")

    logger.info("Done (Step 2.5). Next: Step 2.6 policy intent compiler.")


if __name__ == "__main__":
    main()
