# Final Metrics Leaderboard

Source: `outputs\results\final_metrics_summary.csv`

## Plots generated
- `outputs\figures\metric_det_accuracy.png`
- `outputs\figures\metric_det_precision.png`
- `outputs\figures\metric_det_recall.png`
- `outputs\figures\metric_det_f1.png`
- `outputs\figures\metric_det_fpr.png`
- `outputs\figures\metric_access_decision_precision.png`
- `outputs\figures\metric_avg_response_time_s.png`
- `outputs\figures\metric_mttr_proxy_s.png`
- `outputs\figures\metric_policy_consistency_var.png`
- `outputs\figures\metric_canary_false_deny_rate.png`
- `outputs\figures\metric_rollout_false_deny_rate.png`

## Best per metric
- **Detection Accuracy**: `afozt` = `0.98688`
- **Detection Precision**: `afozt` = `0.9463061690784464`
- **Detection Recall**: `baseline_siem` = `1.0`
- **Detection F1**: `afozt` = `0.883398506932101`
- **Detection False Positive Rate**: `afozt` = `0.003`
- **Access Decision Precision**: `afozt` = `0.9463061690784464`
- **Average Response Time**: `afozt` = `0.2`
- **MTTR Proxy (detect→action)**: `afozt` = `0.2`
- **Policy Consistency Variance Across Clouds**: `baseline_siem` = `0.0329548141438872`
- **Rollout Safety: Canary False-Deny Rate**: `afozt` = `0.002004008016032`
- **Rollout Safety: Global False-Deny Rate**: `afozt` = `0.0005641748942172`
