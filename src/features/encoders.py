from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class EncodedFeatures:
    X: np.ndarray
    feature_names: List[str]
    y: Optional[np.ndarray]


def encode_features(
    df: pd.DataFrame,
    *,
    label_col: str = "label_attack",
    categorical: Optional[List[str]] = None,
    numeric: Optional[List[str]] = None,
) -> EncodedFeatures:
    """Utility for later steps: transform a features dataframe into (X, y).

    This is intentionally simple (research prototype):
    - OneHot for categorical columns
    - StandardScaler for numeric columns
    """
    if categorical is None:
        categorical = [c for c in df.columns if df[c].dtype == "object"]
    if numeric is None:
        numeric = [c for c in df.columns if c not in categorical and c != label_col]

    y = None
    if label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()

    # categorical
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(df[categorical].fillna("unknown")) if categorical else np.zeros((len(df), 0))
    cat_names = ohe.get_feature_names_out(categorical).tolist() if categorical else []

    # numeric
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numeric].fillna(0.0)) if numeric else np.zeros((len(df), 0))
    num_names = list(numeric) if numeric else []

    X = np.hstack([X_num, X_cat])
    names = num_names + cat_names
    return EncodedFeatures(X=X, feature_names=names, y=y)
