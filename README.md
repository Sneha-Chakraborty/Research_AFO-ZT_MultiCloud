# AFO-ZT Prototype (Multi-cloud)

## Step 2.2 — Normalize into a Unified Telemetry Schema

Input (raw):
- `data/raw/afozt_multicloud_logs_50k.csv`

Output (unified):
- `data/processed/unified_telemetry.csv`

Run:
```bash
python scripts/normalize_logs.py
```

Custom paths:
```bash
python scripts/normalize_logs.py --input data/raw/afozt_multicloud_logs_50k.csv --output data/processed/unified_telemetry.csv
```

## Step 3 — Baselines

### 1) Static rules (RBAC/ABAC-like)

```bash
python scripts/run_baselines.py --baseline static
```

Outputs:
- `outputs/results/baseline_static_scores.csv`
- `outputs/models/baseline_static.pkl`

### 2) SIEM rules (threshold + correlation)

```bash
python scripts/run_baselines.py --baseline siem
```

Outputs:
- `outputs/results/baseline_siem_scores.csv`
- `outputs/models/baseline_siem.pkl`

### 3) Base-paper style (RAAC + Isolation Forest + SOAR mapping)

```bash
# optionally create a train/test split first
python scripts/split_train_test.py

python scripts/run_baselines.py --baseline raac_iforest
```

Outputs:
- `outputs/results/baseline_raac_iforest_scores.csv`
- `outputs/models/baseline_raac_iforest.pkl`

### 4) Per-cloud brains (separate analytics per provider)

```bash
python scripts/run_baselines.py --baseline per_cloud
```

Outputs:
- `outputs/results/baseline_per_cloud_scores.csv`
- `outputs/models/baseline_per_cloud.pkl`
- (best-effort) `outputs/models/baseline_per_cloud_<provider>.pkl`

### Compare baselines vs AFO-ZT

```bash
python scripts/run_all_and_compare.py
```

Outputs:
- `outputs/results/metrics_summary_step4.csv`
- `outputs/results/confusion_matrices_step4.json`


## Step 5 — Your AFO-ZT (Unified Brain) end-to-end

One command (runs Steps 2.2 → 2.8 in order):

```bash
python scripts/run_afozt_full.py
```

Force rebuild of intermediate artifacts (normalize/graph/features):

```bash
python scripts/run_afozt_full.py --rebuild
```

Key outputs (generated/overwritten):

- `data/processed/unified_telemetry.csv`
- `outputs/models/trust_graph.sqlite`
- `data/processed/features.parquet` (and `features.csv`)
- `outputs/results/afozt_scores.csv` (Step 2.5)
- `configs/thresholds.yaml` + `outputs/results/afozt_scores_with_tuned_decisions.csv` (Step 2.6)
- `outputs/results/rollout_timeline.csv`, `outputs/results/rollout_decisions.csv`, `outputs/results/rollout_metrics.json` (Step 2.7)
- `outputs/results/action_executions_rollout.jsonl`, `outputs/results/response_times.csv`, `outputs/results/latency_stats.json` (Step 2.8)
- `outputs/results/afozt_full_run_summary.json` (final combined summary)

Optional: check your requested metric targets:

```bash
python scripts/check_afozt_targets.py
```

Outputs:
- `outputs/results/afozt_target_check.json`
