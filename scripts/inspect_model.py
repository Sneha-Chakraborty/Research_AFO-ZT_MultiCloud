from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _summarize(obj: object) -> dict:
    """Best-effort summary for joblib/pickle artifacts."""
    out: dict = {
        "artifact_type": str(type(obj)),
    }
    if isinstance(obj, dict):
        out["keys"] = sorted(list(obj.keys()))
        # shallow preview of common fields
        for k in ["baseline", "config", "derived", "cfg", "feature_cols"]:
            if k in obj:
                v = obj.get(k)
                if isinstance(v, list):
                    out[k] = {"count": len(v), "preview": v[:30]}
                else:
                    out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a .pkl/.joblib artifact and write a small JSON manifest.")
    ap.add_argument("--path", default="outputs/models/afozt_unified.pkl", help="Path to a joblib/pickle artifact.")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <path>.manifest.json)")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    artifact = joblib.load(p)
    manifest = _summarize(artifact)

    out = Path(args.out) if args.out else p.with_suffix(p.suffix + ".manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
