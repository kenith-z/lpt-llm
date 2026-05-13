# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 30.27 |
| decode_tokens_per_sec | 10.88 |
| first_token_latency_ms | 81.9902 |
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
| retnet_summary_norm_mean | 12.140842 |
| retnet_q_adapter_delta_norm_mean | 3.503829 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| retnet_context_adapter_delta_norm_mean | 0.000000 |
| retnet_alpha_context_mean | 0.00000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.730056 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 20.077140 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 75.8921 | 5 |
| 1 | 4.1557 | 5 |
| 2 | 3.3744 | 5 |
| 3 | 3.4325 | 5 |
| 4 | 3.4583 | 5 |
| 5 | 3.5883 | 5 |
| 6 | 3.6987 | 5 |
| 7 | 4.0178 | 5 |
| 8 | 3.9184 | 5 |
| 9 | 3.4937 | 5 |
| 10 | 3.3881 | 5 |
| 11 | 3.3072 | 5 |
| 12 | 3.2974 | 5 |
| 13 | 3.5187 | 5 |
| 14 | 4.2541 | 5 |
| 15 | 5.1004 | 5 |
| 16 | 4.4862 | 5 |
| 17 | 4.0450 | 5 |
| 18 | 3.5633 | 5 |
| 19 | 3.2000 | 5 |
| 20 | 3.2175 | 5 |
| 21 | 3.6998 | 5 |
| 22 | 3.5156 | 5 |
| 23 | 3.8054 | 5 |
