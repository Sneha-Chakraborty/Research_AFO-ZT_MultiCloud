from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from src.ingestion.normalize import UNIFIED_COLUMNS


def validate_unified_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)
