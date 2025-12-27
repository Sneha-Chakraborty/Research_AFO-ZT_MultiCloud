from __future__ import annotations

# The SQLite builder already supports incremental updates via meta(last_ingested_ts).
from src.trust_graph.builder_sqlite import BuildResult, build_or_update_trust_graph_sqlite

__all__ = ["BuildResult", "build_or_update_trust_graph_sqlite"]
