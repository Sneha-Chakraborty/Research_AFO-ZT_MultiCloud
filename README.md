# Research_AFO-ZT_MultiCloud (Simulation Prototype)

This repo contains a **CPU-friendly simulation** + baselines + **AFO‑ZT unified brain** for multi‑cloud (AWS/Azure/GCP).
It generates synthetic access events, scores them, compiles policy intents, runs a safe rollout (shadow→canary→enforce→rollout),
executes simulated SOAR actions, and then computes a **final metrics summary** + plots.

## Quick start (Windows)

### 1) Create venv + install
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2) Generate dataset + features (only once)
```bat
python scripts\generate_data.py
python scripts\normalize_logs.py
python scripts\build_features.py
python scripts\build_trust_graph.py
```

### 3) Run baselines (Static, SIEM, RAAC+IForest, Per-cloud brains)
```bat
python scripts\run_baselines.py
```

### 4) Run AFO‑ZT unified brain (full pipeline: fit → score → tune → rollout → orchestration)
```bat
python scripts\run_afozt_full.py --mfa-rate 0.003 --fpr-restrict 0.001 --fpr-deny 0.0005
```

### 5) Compute final metrics (all engines)
```bat
python scripts\run_final_metrics.py --only-non-allow-latency --write-latencies
```

Outputs:
- `outputs/results/final_metrics_summary.csv` (main comparison table)
- `outputs/results/final_metrics_rows.json` (full per-engine JSON)
- `outputs/results/final_metrics_*.json` (per-engine detailed metrics)
- `outputs/results/latency_stats_*.json` (per-engine latency summary, if enabled)

### 6) Plot graphs (PNG)
```bat
python scripts\plot_final_metrics.py
```

This writes PNGs into `outputs/plots/` and a compact CSV you can paste into your paper.

## Notes

- **Cross-cloud detection rate** now uses a robust fallback: if the score file does not contain the generator’s cross‑cloud flag,
  the metric is derived from grouping by session/principal and checking if that entity touched **>1 cloud provider**.
- **Blast radius reduction** uses a blast feature when present; otherwise it uses a conservative pivot‑score proxy.

