from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, List

import pandas as pd

from src.baselines.raac_iforest import RaacIForestBaseline, RaacIForestConfig


@dataclass
class PerCloudBrainsConfig:
    """Per-cloud brains baseline (separate analytics per provider).

    This baseline simulates a common multi-cloud anti-pattern:
    each cloud provider runs its own analytics & thresholds, with no
    unified cross-cloud context.

    Compared to AFO-ZT (unified telemetry + trust graph + safe rollout), this baseline:
    - misses cross-cloud correlations (hop sequences, lateral movement across providers)
    - produces inconsistent decisions across clouds (separate models)
    - often increases friction (more step-ups/restricts) to compensate

    Output contract (same as other baselines):
    - risk in [0,1]
    - confidence in [0,1]
    - decision & decision_tuned in {allow,stepup,restrict,deny}
    - rationale_tags & rationale_text
    """

    # Column used to split providers
    provider_col: str = "cloud_provider"

    # Minimum rows needed to fit a provider-specific model.
    # If a provider has fewer rows, fallback to the global model.
    min_train_rows: int = 400

    # Underlying per-cloud model: RAAC + Isolation Forest (base-paper style),
    # but trained separately per provider.
    base_cfg: RaacIForestConfig = field(
        default_factory=lambda: RaacIForestConfig(
            # slightly more sensitive than the global RAAC baseline to reflect
            # teams over-compensating when they lack cross-cloud signals.
            iforest_contamination=0.08,
            t_allow=0.30,
            t_stepup=0.52,
            t_deny=0.74,
        )
    )


class PerCloudBrainsBaseline:
    """Train one RAAC+IForest model per cloud provider, then score using that provider's model."""

    def __init__(self, cfg: Optional[PerCloudBrainsConfig] = None) -> None:
        self.cfg = cfg or PerCloudBrainsConfig()
        self.models: Dict[str, RaacIForestBaseline] = {}
        self.global_model: Optional[RaacIForestBaseline] = None

    @property
    def artifact(self) -> dict:
        return {
            "baseline": "per_cloud_brains",
            "config": {
                **asdict(self.cfg),
                # dataclasses can't deep-serialize sklearn models; keep the base config separately
                "base_cfg": asdict(self.cfg.base_cfg),
            },
            "providers": sorted(self.models.keys()),
            "models": {k: v.artifact for k, v in self.models.items()},
            "global_model": self.global_model.artifact if self.global_model is not None else None,
        }

    def fit(self, df_train: pd.DataFrame) -> "PerCloudBrainsBaseline":
        """Fit per-provider models (and a global fallback model)."""
        df_train = df_train.copy()
        if self.cfg.provider_col not in df_train.columns:
            df_train[self.cfg.provider_col] = "unknown"

        # global fallback (ensures scoring always works)
        self.global_model = RaacIForestBaseline(self.cfg.base_cfg)
        self.global_model.fit(df_train)

        # per-provider models
        for provider, g in df_train.groupby(df_train[self.cfg.provider_col].astype(str).str.lower(), dropna=False):
            g = g.copy()
            if len(g) < int(self.cfg.min_train_rows):
                # too little data -> keep only global model for this provider
                continue
            m = RaacIForestBaseline(self.cfg.base_cfg)
            m.fit(g)
            self.models[str(provider)] = m

        return self

    def score(self, df: pd.DataFrame, *, df_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Score a dataframe using provider-specific models.

        If not fitted yet:
        - fit using df_train if provided
        - otherwise fit on the scoring df (best-effort)
        """
        out = df.copy()
        if self.cfg.provider_col not in out.columns:
            out[self.cfg.provider_col] = "unknown"

        if self.global_model is None:
            self.fit(df_train if df_train is not None else out)

        parts: List[pd.DataFrame] = []

        # group-by preserves order only within groups; we'll re-sort by index later
        providers = out[self.cfg.provider_col].astype(str).str.lower()
        for provider, idx in providers.groupby(providers).groups.items():
            g = out.loc[idx].copy()

            # pick provider model if available, else fallback to global
            m = self.models.get(str(provider))
            if m is None:
                # If we have enough rows for this provider in the scoring df, build a model on-the-fly.
                if len(g) >= int(self.cfg.min_train_rows):
                    m = RaacIForestBaseline(self.cfg.base_cfg)
                    m.fit(g)
                    self.models[str(provider)] = m
                else:
                    m = self.global_model

            scored = m.score(g)
            scored["percloud_provider"] = str(provider)
            scored["percloud_model"] = "provider" if str(provider) in self.models else "global_fallback"
            parts.append(scored)

        merged = pd.concat(parts, axis=0).sort_index()

        # Add baseline-specific rationale hint
        if "rationale_tags" in merged.columns:
            merged["rationale_tags"] = merged["rationale_tags"].fillna("").astype(str) + "|per_cloud_brain"
        else:
            merged["rationale_tags"] = "per_cloud_brain"

        if "rationale_text" in merged.columns:
            merged["rationale_text"] = merged["rationale_text"].fillna("").astype(str) + " (per-cloud model; no cross-cloud context)"
        else:
            merged["rationale_text"] = "Per-cloud model; no cross-cloud context."

        return merged
