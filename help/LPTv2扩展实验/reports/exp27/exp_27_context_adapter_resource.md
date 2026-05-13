# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 27.77 |
| decode_tokens_per_sec | 9.89 |
| first_token_latency_ms | 101.9064 |
| peak_memory_bytes | 2900514816 |
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
| retnet_summary_norm_mean | 11.162801 |
| retnet_q_adapter_delta_norm_mean | 3.947989 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| retnet_context_adapter_delta_norm_mean | 12.037177 |
| retnet_alpha_context_mean | -0.00005293 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.802298 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 18.589082 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 73.3328 | 5 |
| 1 | 5.1800 | 5 |
| 2 | 5.0654 | 5 |
| 3 | 5.0555 | 5 |
| 4 | 5.1374 | 5 |
| 5 | 4.5959 | 5 |
| 6 | 4.4248 | 5 |
| 7 | 4.3658 | 5 |
| 8 | 4.9976 | 5 |
| 9 | 5.4164 | 5 |
| 10 | 5.1137 | 5 |
| 11 | 5.0891 | 5 |
| 12 | 4.6075 | 5 |
| 13 | 3.5389 | 5 |
| 14 | 3.4500 | 5 |
| 15 | 4.0657 | 5 |
| 16 | 4.1606 | 5 |
| 17 | 4.6513 | 5 |
| 18 | 4.9377 | 5 |
| 19 | 4.6308 | 5 |
| 20 | 4.1798 | 5 |
| 21 | 3.9688 | 5 |
| 22 | 3.9678 | 5 |
| 23 | 3.5914 | 5 |
