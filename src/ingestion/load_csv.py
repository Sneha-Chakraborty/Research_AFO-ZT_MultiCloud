from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Load raw multi-cloud log CSV."""
    path = Path(path)
    df = pd.read_csv(path)
    return df
