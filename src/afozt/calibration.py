from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantileCalibrator:
    """Calibrate a 1D array to [0, 1] using an empirical CDF.

    Used in Step 2.5 to map IsolationForest anomaly magnitudes (unbounded)
    into a stable probability-like score.
    """

    ref_sorted_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0], dtype=float)
        self.ref_sorted_ = np.sort(arr)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.ref_sorted_ is None:
            raise RuntimeError("QuantileCalibrator is not fit yet.")
        ref = self.ref_sorted_
        arr = np.asarray(x, dtype=float)
        out = np.zeros_like(arr, dtype=float)
        mask = np.isfinite(arr)
        if mask.any():
            idx = np.searchsorted(ref, arr[mask], side="right")
            out[mask] = idx / float(max(1, ref.size))
        # Non-finite values map to low score
        out[~mask] = 0.0
        return np.clip(out, 0.0, 1.0)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
