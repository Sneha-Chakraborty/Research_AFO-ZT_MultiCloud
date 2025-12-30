"""
inspect_siem_pickle.py
Load a .pkl (pickle) model/artifact and write a readable JSON preview + quick summary.

Usage:
  python scripts/inspect_siem_pickle.py outputs/models/baseline_siem.pkl
  python scripts/inspect_siem_pickle.py outputs/models/baseline_siem.pkl --max-items 50
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any


def _safe_slim(obj: Any, max_items: int = 50, _depth: int = 0, _max_depth: int = 6) -> Any:
    """
    Convert arbitrary Python objects into JSON-serializable structures (best-effort),
    truncating large containers to keep preview readable.
    """
    if _depth >= _max_depth:
        return f"<max_depth_reached type={type(obj).__name__}>"

    # Primitive JSON types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Bytes
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"

    # Dict-like
    if isinstance(obj, dict):
        items = list(obj.items())
        slim = {}
        for i, (k, v) in enumerate(items[:max_items]):
            slim[str(k)] = _safe_slim(v, max_items=max_items, _depth=_depth + 1)
        if len(items) > max_items:
            slim["__truncated__"] = f"{len(items) - max_items} more keys omitted"
        return slim

    # List/Tuple/Set
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        slim_list = [_safe_slim(v, max_items=max_items, _depth=_depth + 1) for v in seq[:max_items]]
        if len(seq) > max_items:
            slim_list.append(f"<truncated: {len(seq) - max_items} more items>")
        return slim_list

    # Numpy/pandas (optional)
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "preview": _safe_slim(obj.flatten()[: min(max_items, obj.size)].tolist(),
                                     max_items=max_items, _depth=_depth + 1),
            }
    except Exception:
        pass

    # Fallback: object with __dict__
    if hasattr(obj, "__dict__"):
        return {
            "__type__": type(obj).__name__,
            "__module__": getattr(type(obj), "__module__", ""),
            "attrs": _safe_slim(vars(obj), max_items=max_items, _depth=_depth + 1),
        }

    # Final fallback
    return f"<unserializable type={type(obj).__name__}: {str(obj)[:200]}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to .pkl file (e.g., outputs/models/baseline_siem.pkl)")
    ap.add_argument("--max-items", type=int, default=50, help="Max items to show per container")
    args = ap.parse_args()

    pkl_path = args.path
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    size = os.path.getsize(pkl_path)
    print(f"[inspect_siem_pickle] Loading: {pkl_path}")
    print(f"[inspect_siem_pickle] Size: {size} bytes")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    print(f"[inspect_siem_pickle] Loaded OK. Type: {type(obj)}")

    preview = _safe_slim(obj, max_items=args.max_items)

    out_json = os.path.splitext(pkl_path)[0] + "_preview.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print(f"[inspect_siem_pickle] Wrote preview: {out_json}")
    print("[inspect_siem_pickle] Open that JSON file in VS Code to view contents.")


if __name__ == "__main__":
    main()
