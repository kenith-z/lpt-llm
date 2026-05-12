# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 26.85 |
| decode_tokens_per_sec | 8.03 |
| first_token_latency_ms | 135.3145 |
| peak_memory_bytes | 2900451840 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 6 |
| expected_retnet_layer_state_count | 6 |
| retnet_enabled_layer_count | 6 |
| retnet_assist_layers | `selected_layers` |
| retnet_assist_selected_layers | `[2, 6, 10, 14, 18, 22]` |
| retnet_adapter_rank | 16 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 13.283762 |
| retnet_q_adapter_delta_norm_mean | 5.532413 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.697026 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 20.808015 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 72.9620 | 5 |
| 1 | 5.1332 | 5 |
| 2 | 11.0951 | 5 |
| 3 | 5.2068 | 5 |
| 4 | 5.1392 | 5 |
| 5 | 5.1043 | 5 |
| 6 | 6.0391 | 5 |
| 7 | 4.9659 | 5 |
| 8 | 4.9401 | 5 |
| 9 | 4.9528 | 5 |
| 10 | 6.2534 | 5 |
| 11 | 4.7599 | 5 |
| 12 | 4.9590 | 5 |
| 13 | 5.0856 | 5 |
| 14 | 6.7184 | 5 |
| 15 | 5.0691 | 5 |
| 16 | 5.1357 | 5 |
| 17 | 5.1796 | 5 |
| 18 | 6.3067 | 5 |
| 19 | 5.3333 | 5 |
| 20 | 5.2248 | 5 |
| 21 | 5.2852 | 5 |
| 22 | 6.2617 | 5 |
| 23 | 4.6806 | 5 |
