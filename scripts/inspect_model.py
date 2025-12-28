from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = Path("outputs/models/afozt_unified.pkl")
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

    artifact = joblib.load(p)

    manifest = {
        "artifact_type": str(type(artifact)),
        "artifact_keys": sorted(list(artifact.keys())) if isinstance(artifact, dict) else None,
    }

    if isinstance(artifact, dict):
        cfg = artifact.get("cfg", {})
        manifest["cfg"] = cfg

        cols = artifact.get("feature_cols", [])
        manifest["feature_cols_count"] = len(cols) if isinstance(cols, list) else None
        manifest["feature_cols_preview"] = cols[:30] if isinstance(cols, list) else None

        iforest = artifact.get("iforest", None)
        if iforest is not None and hasattr(iforest, "get_params"):
            manifest["iforest_params"] = iforest.get_params()

        calibrator = artifact.get("calibrator", None)
        if calibrator is not None:
            manifest["calibrator_class"] = str(type(calibrator))

    out = p.with_suffix(".manifest.json")
    out.write_text(json.dumps(manifest, indent=2))
    print("Wrote:", out)


if __name__ == "__main__":
    main()
