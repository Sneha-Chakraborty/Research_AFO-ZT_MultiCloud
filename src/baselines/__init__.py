"""
Baselines package.

Step 3 adds the first baseline: StaticRulesBaseline (RBAC/ABAC-like).
Other baselines (SIEM rules, RAAC+IForest, per-cloud brains) are implemented in later steps.
"""
from .static_rules import StaticRulesBaseline, StaticRulesConfig

__all__ = ["StaticRulesBaseline", "StaticRulesConfig"]
