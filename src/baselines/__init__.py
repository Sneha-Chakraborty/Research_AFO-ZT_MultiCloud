"""Baselines package (Step 3).

Implemented:
- Static RBAC/ABAC-like rules baseline
- SIEM threshold/correlation rules baseline
- Base-paper-style RAAC + Isolation Forest baseline (with SOAR-style decision mapping)

Each baseline reads the same unified telemetry CSV and outputs the same decision fields,
so evaluation stays apples-to-apples.
"""

from .static_rules import StaticRulesBaseline, StaticRulesConfig
from .siem_rules import SiemRulesBaseline, SiemRulesConfig
from .raac_iforest import RaacIForestBaseline, RaacIForestConfig

__all__ = [
    "StaticRulesBaseline",
    "StaticRulesConfig",
    "SiemRulesBaseline",
    "SiemRulesConfig",
    "RaacIForestBaseline",
    "RaacIForestConfig",
]
