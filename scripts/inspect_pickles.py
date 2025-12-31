\
"""
inspect_pickles.py (v5)
-----------------------
Like v4, but improves globals extraction:
- Captures BOTH GLOBAL and STACK_GLOBAL references.

Why your *_globals.json might be empty in v4:
- Many pickles (protocol 4/5) use STACK_GLOBAL instead of GLOBAL, so v4 missed them.
- Or the pickle contains only builtins (dict/list/int/float/str) so there are no globals to report.

Outputs per pickle:
- <name>_summary.txt                 (always)
- <name>_preview.json                (only if unpickling succeeds)
- <name>_globals.json                (always; extracted from pickle stream)
- <name>_pickle_ops_head.txt         (always; first N pickle ops)
- <name>_inspect_report.txt          (always; environment + errors)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import types
from io import BytesIO
from typing import Any, Iterable, List, Dict, Tuple

import pickletools


# --- Ensure local project modules are importable (root + src) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


DEFAULT_PATHS = [
    r"outputs/models/baseline_per_cloud.pkl",
    r"outputs/models/baseline_per_cloud_aws.pkl",
    r"outputs/models/baseline_per_cloud_azure.pkl",
    r"outputs/models/baseline_per_cloud_gcp.pkl",
    r"outputs/models/baseline_per_cloud_global_fallback.pkl",
]


# ---------- Stub modules to make unpickling work ----------
class _PlaceholderBase:
    def __init__(self, *args, **kwargs):
        pass


def install_stub_module(module_name: str) -> None:
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)

    def __getattr__(name: str):
        return type(name, (_PlaceholderBase,), {})
    stub.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[module_name] = stub


def ensure_unpickle_deps(extra: List[str] | None = None) -> None:
    must_exist_or_stub = ["config"]
    if extra:
        must_exist_or_stub.extend(extra)
    for m in must_exist_or_stub:
        try:
            importlib.import_module(m)
        except Exception:
            install_stub_module(m)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "config":
            ensure_unpickle_deps()
        return super().find_class(module, name)


# ---------- JSON preview helpers ----------
def safe_slim(obj: Any, max_items: int = 80, depth: int = 0, max_depth: int = 7) -> Any:
    if depth >= max_depth:
        return f"<max_depth_reached type={type(obj).__name__}>"

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (bytes, bytearray)):
        return {"__type__": "bytes", "len": len(obj)}

    if isinstance(obj, dict):
        items = list(obj.items())
        out = {}
        for k, v in items[:max_items]:
            out[str(k)] = safe_slim(v, max_items=max_items, depth=depth + 1, max_depth=max_depth)
        if len(items) > max_items:
            out["__truncated__"] = f"{len(items) - max_items} more keys omitted"
        return out

    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        out_list = [safe_slim(v, max_items=max_items, depth=depth + 1, max_depth=max_depth) for v in seq[:max_items]]
        if len(seq) > max_items:
            out_list.append(f"<truncated: {len(seq) - max_items} more items>")
        return out_list

    # Optional numpy support
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.ndarray):
            flat = obj.ravel()
            preview = flat[: min(max_items, flat.size)].tolist()
            return {
                "__type__": "ndarray",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "preview": safe_slim(preview, max_items=max_items, depth=depth + 1, max_depth=max_depth),
            }
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        return {
            "__type__": type(obj).__name__,
            "__module__": getattr(type(obj), "__module__", ""),
            "attrs": safe_slim(vars(obj), max_items=max_items, depth=depth + 1, max_depth=max_depth),
        }

    return f"<unserializable type={type(obj).__name__}: {str(obj)[:200]}>"


def summarize(obj: Any) -> str:
    lines = [f"Type: {type(obj)}"]
    if isinstance(obj, dict):
        keys = list(obj.keys())
        lines.append(f"Dict keys ({len(keys)}): {keys[:60]}")
    else:
        try:
            attrs = [a for a in dir(obj) if not a.startswith('_')]
            lines.append(f"Public attrs (sample): {attrs[:80]}")
        except Exception:
            pass
    return "\n".join(lines) + "\n"


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------- Fallback: pickle disassembly inspection ----------
_STRING_OPS = {
    "BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8",
    "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
}


def extract_globals_and_ops(pkl_bytes: bytes, max_ops: int = 8000) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Parse pickle opcodes without unpickling:
    - ops_lines: textual disassembly head
    - globals_list: module/class references extracted from GLOBAL or STACK_GLOBAL patterns

    STACK_GLOBAL heuristic:
      Many protocol 4/5 pickles push module + name as strings, then emit STACK_GLOBAL.
      We record the last two string pushes seen immediately before STACK_GLOBAL.
    """
    ops_lines: List[str] = []
    globals_list: List[Dict[str, str]] = []
    seen = set()

    last_strings: List[str] = []  # keep a short rolling window of recent string args

    bio = BytesIO(pkl_bytes)
    for i, (op, arg, pos) in enumerate(pickletools.genops(bio)):
        if i >= max_ops:
            ops_lines.append(f"... <truncated after {max_ops} ops>")
            break

        arg_s = ""
        if arg is not None:
            try:
                arg_s = repr(arg)
            except Exception:
                arg_s = "<unrepr>"
        ops_lines.append(f"{pos:08x}  {op.name:<16} {arg_s}")

        # Track recent strings for STACK_GLOBAL heuristic
        if op.name in _STRING_OPS and isinstance(arg, (str, bytes, bytearray)):
            if isinstance(arg, (bytes, bytearray)):
                try:
                    s = arg.decode("utf-8", errors="replace")
                except Exception:
                    s = "<bytes>"
            else:
                s = arg
            last_strings.append(s)
            if len(last_strings) > 8:
                last_strings = last_strings[-8:]

        # GLOBAL opcode: arg usually "module name"
        if op.name == "GLOBAL" and isinstance(arg, str):
            parts = arg.strip().split()
            if len(parts) >= 2:
                mod, name = parts[0], parts[1]
                key = (mod, name)
                if key not in seen:
                    seen.add(key)
                    globals_list.append({"module": mod, "name": name})

        # STACK_GLOBAL opcode (protocol 4/5)
        if op.name == "STACK_GLOBAL":
            if len(last_strings) >= 2:
                mod, name = last_strings[-2], last_strings[-1]
                key = (mod, name)
                if key not in seen:
                    seen.add(key)
                    globals_list.append({"module": mod, "name": name})

    return ops_lines, globals_list


def write_env_report(path_txt: str, pkl_path: str, error: str | None) -> None:
    import platform
    lines = []
    lines.append(f"pickle_path: {pkl_path}")
    lines.append(f"python: {sys.version.replace(os.linesep, ' ')}")
    lines.append(f"platform: {platform.platform()}")
    for lib in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            m = importlib.import_module(lib)
            ver = getattr(m, "__version__", "unknown")
            lines.append(f"{lib}: {ver}")
        except Exception as e:
            lines.append(f"{lib}: not importable ({e})")
    if error:
        lines.append("")
        lines.append("ERROR:")
        lines.append(error)
    ensure_parent(path_txt)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_one(path: str, max_items: int, max_depth: int, max_ops: int) -> None:
    if not os.path.exists(path):
        print(f"[SKIP] Not found: {path}")
        return

    size = os.path.getsize(path)
    print(f"[LOAD] {path} ({size} bytes)")

    base = os.path.splitext(path)[0]
    summary_path = base + "_summary.txt"
    preview_path = base + "_preview.json"
    globals_path = base + "_globals.json"
    ops_path = base + "_pickle_ops_head.txt"
    report_path = base + "_inspect_report.txt"

    with open(path, "rb") as f:
        pkl_bytes = f.read()

    # Always write globals + ops
    ops_lines, globals_list = extract_globals_and_ops(pkl_bytes, max_ops=max_ops)
    ensure_parent(globals_path)
    with open(globals_path, "w", encoding="utf-8") as f:
        json.dump({"globals": globals_list}, f, indent=2, ensure_ascii=False)

    ensure_parent(ops_path)
    with open(ops_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ops_lines) + "\n")

    error_text = None
    try:
        ensure_unpickle_deps()
        obj = SafeUnpickler(BytesIO(pkl_bytes)).load()

        ensure_parent(summary_path)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summarize(obj))

        preview = safe_slim(obj, max_items=max_items, max_depth=max_depth)
        ensure_parent(preview_path)
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(preview, f, indent=2, ensure_ascii=False)

        print(f"[OK] Wrote: {preview_path}")
        print(f"[OK] Wrote: {summary_path}")

    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        ensure_parent(summary_path)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Unpickling failed; wrote fallback inspection files instead.\n")
            f.write(f"Error: {error_text}\n")
            f.write(f"See: {globals_path}\n")
            f.write(f"See: {ops_path}\n")
        print(f"[WARN] Unpickling failed: {error_text}")
        print(f"[OK] Wrote fallback: {globals_path}")
        print(f"[OK] Wrote fallback: {ops_path}")
        print(f"[OK] Wrote: {summary_path}")

    write_env_report(report_path, pkl_path=path, error=error_text)
    print(f"[OK] Wrote: {report_path}")


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Pickle paths. If empty, uses default per-cloud baseline paths.")
    ap.add_argument("--max-items", type=int, default=80, help="Max items per container in JSON preview.")
    ap.add_argument("--max-depth", type=int, default=7, help="Max recursion depth for JSON preview.")
    ap.add_argument("--max-ops", type=int, default=8000, help="How many pickle ops to write in the disassembly head.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    paths = args.paths if args.paths else DEFAULT_PATHS
    for p in paths:
        process_one(p, max_items=args.max_items, max_depth=args.max_depth, max_ops=args.max_ops)

    print("\nDone. Open *_preview.json if present; otherwise open *_globals.json and *_pickle_ops_head.txt in VS Code.")


if __name__ == "__main__":
    main()
