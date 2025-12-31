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
