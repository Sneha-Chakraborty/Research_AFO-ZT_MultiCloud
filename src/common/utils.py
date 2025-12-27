from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional

import pandas as pd


def stable_hash_to_int(*parts: Any, mod: int = 2**31 - 1) -> int:
    """Deterministic hash -> int for synthetic field generation."""
    s = "|".join("" if p is None else str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def synthetic_ipv4(*parts: Any) -> str:
    """Create a stable, private-range IPv4 address (10.x.x.x) from identifiers."""
    n = stable_hash_to_int(*parts, mod=256**3)
    b1 = (n // (256**2)) % 256
    b2 = (n // 256) % 256
    b3 = n % 256
    # Avoid 0/255 edge bytes for nicer display
    b1 = 1 + (b1 % 254)
    b2 = 1 + (b2 % 254)
    b3 = 1 + (b3 % 254)
    return f"10.{b1}.{b2}.{b3}"


def to_bool01(x: Any) -> int:
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float)):
        return 1 if float(x) >= 1 else 0
    s = str(x).strip().lower()
    return 1 if s in {"1", "true", "t", "yes", "y"} else 0


def parse_ts(series: pd.Series) -> pd.Series:
    """Parse timestamps to pandas datetime64[ns] (UTC-naive)."""
    # The synthetic dataset uses 'YYYY-MM-DD HH:MM:SS'
    return pd.to_datetime(series, errors="coerce")
