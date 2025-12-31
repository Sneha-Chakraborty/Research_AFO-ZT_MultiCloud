# Baselines

The baselines are intentionally designed to be **reasonable** but **limited** compared to AFO-ZT.
They all read the same unified telemetry (`data/processed/unified_telemetry.csv`) and output a scored CSV under `outputs/results/`.

## Why AFO-ZT should beat these baselines (realistically)

AFO-ZT adds capabilities that these baselines do **not** have:
- **Trust-graph correlation** (edges and cross-cloud lateral movement signals)
- **Closed-loop rollout safety** (shadow → canary → enforce → rollout with guardrail budgets)
- **Unified policy intent compilation** across providers (PDP/PEP orchestration model)

Baselines below are missing one or more of these, leading to either:
- higher false positives (more unnecessary MFA/blocks), or
- higher false negatives (missing correlated attacks), or
- inconsistent actions across clouds.

## How to run

### Static rules (RBAC/ABAC-like)
```bash
python scripts/run_baselines.py --baseline static
```

### SIEM rules (threshold + correlation)
```bash
python scripts/run_baselines.py --baseline siem
```

### Base-paper style (RAAC + Isolation Forest + SOAR mapping)
```bash
# recommended
python scripts/split_train_test.py

python scripts/run_baselines.py --baseline raac_iforest
```

### Per-cloud brains (separate analytics per provider)
```bash
python scripts/run_baselines.py --baseline per_cloud
```

### Compare everything
```bash
python scripts/run_all_and_compare.py
```

Outputs:
- `outputs/results/metrics_summary_step4.csv`
- `outputs/results/confusion_matrices_step4.json`
