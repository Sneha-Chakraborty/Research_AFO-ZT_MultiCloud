-- SQLite
SELECT src_id, edge_type, dst_id, cloud_provider, deny_count, attack_count, bytes_out_sum, event_count, last_seen
FROM edges_agg
WHERE edge_type IN ('accessed','data-read','called-api','token-issued','assumed-role')
ORDER BY (attack_count*10 + deny_count + (bytes_out_sum/1000000.0)) DESC
LIMIT 10;
