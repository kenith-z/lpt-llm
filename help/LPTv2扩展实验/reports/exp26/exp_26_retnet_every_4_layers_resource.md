# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 27.56 |
| decode_tokens_per_sec | 7.49 |
| first_token_latency_ms | 135.1777 |
| peak_memory_bytes | 2900451840 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 6 |
| expected_retnet_layer_state_count | 6 |
| retnet_enabled_layer_count | 6 |
| retnet_assist_layers | `every_4_layers` |
| retnet_assist_selected_layers | `[]` |
| retnet_adapter_rank | 16 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 9.335006 |
| retnet_q_adapter_delta_norm_mean | 2.759270 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.732880 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 21.941067 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 76.4336 | 5 |
| 1 | 5.8897 | 5 |
| 2 | 5.5701 | 5 |
| 3 | 5.5222 | 5 |
| 4 | 6.8018 | 5 |
| 5 | 5.6407 | 5 |
| 6 | 5.8329 | 5 |
| 7 | 5.5769 | 5 |
| 8 | 6.5075 | 5 |
| 9 | 5.2808 | 5 |
| 10 | 5.5725 | 5 |
| 11 | 5.3949 | 5 |
| 12 | 6.6731 | 5 |
| 13 | 5.2893 | 5 |
| 14 | 5.0809 | 5 |
| 15 | 4.9863 | 5 |
| 16 | 6.5866 | 5 |
| 17 | 5.2836 | 5 |
| 18 | 5.1323 | 5 |
| 19 | 5.2821 | 5 |
| 20 | 6.2745 | 5 |
| 21 | 5.2412 | 5 |
| 22 | 5.2209 | 5 |
| 23 | 5.1487 | 5 |
