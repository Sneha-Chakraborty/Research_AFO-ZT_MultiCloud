# Unified Telemetry Schema (Step 2.2)

Goal: Convert cloud-specific events into one common record format so *all* baselines and AFO-ZT read the same input.

## Core fields (single-brain schema)

### who
- `principal_id` (string): user/service identity id
- `principal_type` (string): e.g., `user`, `service`
- `role` (string): effective role
- `mfa_used` (int 0/1): whether MFA was used (if applicable)

### what
- `action` (string): high-level action (e.g., `login`, `read`, `write`, `assume_role`)
- `api` (string): normalized API label (cloud-agnostic)
- `operation` (string): normalized operation label
- `resource_id` (string)
- `resource_type` (string)
- `resource_sensitivity` (string): e.g., `LOW`, `MEDIUM`, `HIGH`

### where
- `cloud_provider` (string): `aws` / `azure` / `gcp`
- `tenant_id` (string): account/subscription/project id
- `region` (string)
- `ip` (string): IPv4 (synthetic if not present)
- `geo` (string): country/geo label

### device / workload
- `device_id` (string)
- `posture_score` (float 0..1): derived from compliance/posture
- `workload_id` (string): optional (synthetic if not present)

### session
- `session_id` (string)
- `token_id` (string): synthetic if not present
- `token_scope` (string): e.g., `NARROW`, `MEDIUM`, `WIDE`
- `token_age` (float): seconds (or similar)

### outcomes
- `access_result` (string): `permit` / `deny` / `restrict`
- `latency_ms` (float): synthetic if not present
- `bytes_out` (float)

### truth
- `label_attack` (int 0/1)
- `attack_type` (string)

## Notes

- The prototype also preserves a few original columns as `raw_*` fields so later steps (features/graph) can reuse them.
