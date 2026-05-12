# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 25.03 |
| decode_tokens_per_sec | 11.50 |
| first_token_latency_ms | 94.7398 |
| peak_memory_bytes | 2900514304 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 24 |
| expected_retnet_layer_state_count | 24 |
| retnet_enabled_layer_count | 24 |
| retnet_assist_layers | `all_layers` |
| retnet_assist_selected_layers | `[]` |
| retnet_adapter_rank | 32 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 10.893974 |
| retnet_q_adapter_delta_norm_mean | 2.808356 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.754476 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 20.947058 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 77.4321 | 5 |
| 1 | 5.5682 | 5 |
| 2 | 4.8557 | 5 |
| 3 | 4.9821 | 5 |
| 4 | 4.7215 | 5 |
| 5 | 4.2978 | 5 |
| 6 | 4.2175 | 5 |
| 7 | 4.2336 | 5 |
| 8 | 4.7401 | 5 |
| 9 | 4.2328 | 5 |
| 10 | 4.1124 | 5 |
| 11 | 3.8977 | 5 |
| 12 | 4.1259 | 5 |
| 13 | 3.9587 | 5 |
| 14 | 3.9231 | 5 |
| 15 | 3.9611 | 5 |
| 16 | 3.8255 | 5 |
| 17 | 3.8425 | 5 |
| 18 | 4.5720 | 5 |
| 19 | 4.0951 | 5 |
| 20 | 3.8759 | 5 |
| 21 | 3.8363 | 5 |
| 22 | 4.4743 | 5 |
| 23 | 4.5126 | 5 |
