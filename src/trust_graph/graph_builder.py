from __future__ import annotations

# Backward-compatible wrapper for earlier filenames.
from src.trust_graph.builder_sqlite import BuildResult, build_or_update_trust_graph_sqlite

__all__ = ["BuildResult", "build_or_update_trust_graph_sqlite"]
