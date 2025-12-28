from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantileCalibrator:
    """Map raw anomaly magnitudes to calibrated [0,1] via empirical CDF.

    This keeps Step 2.5 explainable and robust across datasets, without requiring labels.

    fit(): stores sorted scores from training.
    transform(): returns percentile rank in [0,1].
    """

    sorted_scores_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray) -> None:
        s = np.asarray(scores, dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            self.sorted_scores_ = np.array([0.0], dtype=float)
            return
        self.sorted_scores_ = np.sort(s)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.sorted_scores_ is None:
            raise RuntimeError("QuantileCalibrator not fit yet.")
        ref = self.sorted_scores_
        s = np.asarray(scores, dtype=float)
        # rank = number of ref <= s
        ranks = np.searchsorted(ref, s, side="right").astype(float)
        pct = ranks / float(max(1, ref.size))
        return np.clip(pct, 0.0, 1.0)
