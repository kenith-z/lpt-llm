# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 25.64 |
| decode_tokens_per_sec | 6.92 |
| first_token_latency_ms | 154.0619 |
| peak_memory_bytes | 2900479488 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 24 |
| expected_retnet_layer_state_count | 24 |
| retnet_enabled_layer_count | 24 |
| retnet_assist_layers | `all_layers` |
| retnet_assist_selected_layers | `[]` |
| retnet_adapter_rank | 16 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 14.644650 |
| retnet_q_adapter_delta_norm_mean | 5.674315 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.720449 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 20.975037 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 77.9882 | 5 |
| 1 | 7.0643 | 5 |
| 2 | 6.6578 | 5 |
| 3 | 6.7591 | 5 |
| 4 | 6.7028 | 5 |
| 5 | 6.5732 | 5 |
| 6 | 6.6155 | 5 |
| 7 | 6.4800 | 5 |
| 8 | 6.8378 | 5 |
| 9 | 6.8777 | 5 |
| 10 | 6.3008 | 5 |
| 11 | 6.3434 | 5 |
| 12 | 5.7343 | 5 |
| 13 | 5.7867 | 5 |
| 14 | 5.5961 | 5 |
| 15 | 5.8429 | 5 |
| 16 | 5.5924 | 5 |
| 17 | 5.8733 | 5 |
| 18 | 5.5357 | 5 |
| 19 | 5.6646 | 5 |
| 20 | 5.9919 | 5 |
| 21 | 5.6406 | 5 |
| 22 | 5.5264 | 5 |
| 23 | 5.5418 | 5 |
