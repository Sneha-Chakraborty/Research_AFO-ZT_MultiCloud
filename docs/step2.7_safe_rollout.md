# Step 2.7 — Safe Rollout Controller (Shadow → Canary → Enforce → Rollout)

This step adds an **offline simulator** that shows how AFO‑ZT can safely enable policy enforcement in stages,
while watching *safety budgets* (rollback triggers).

## What it does

Stages:

1. **Shadow**: compute intents/actions but do **not** enforce; log hypothetical impact.
2. **Canary**: enforce on a small deterministic cohort (e.g., 5% of sessions).
3. **Enforce**: enforce broadly inside a chosen scope (e.g., one cloud provider / tenant).
4. **Rollout**: enforce globally.

Rollback triggers (budgets) computed from the scored logs:

- False-deny rate
- MFA friction rate
- Simulated p95 latency impact
- Simulated policy-induced unexpected error rate
- Analyst override rate (severe action + low confidence)

> Note: latency/error/override are deterministic simulations suitable for offline demo;
> later steps can replace them with real service SLO telemetry.

## How to run

Prereq: you already ran **Step 2.5** to generate scores:

```bash
python scripts/run_afozt.py
```

Now run the rollout controller:

```bash
python scripts/run_safe_rollout.py
```

Outputs written to `outputs/results/`:

- `rollout_timeline.csv`
- `rollout_decisions.csv`
- `rollout_metrics.json`

## Config

Edit `configs/rollout.yaml` to control:

- window sizes (`shadow_days`, `canary_days`, `enforce_days`)
- canary fraction and hashing key
- enforcement scope (cloud/tenant/region)
- budgets (rollback triggers)
