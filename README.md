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
