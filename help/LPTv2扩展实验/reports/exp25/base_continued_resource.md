# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 29.18 |
| decode_tokens_per_sec | 10.42 |
| first_token_latency_ms | 81.0512 |
| peak_memory_bytes | 2900479488 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 24 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 13.066739 |
| retnet_q_adapter_delta_norm_mean | 5.711489 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.746317 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 21.326418 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 77.7282 | 5 |
| 1 | 4.3959 | 5 |
| 2 | 3.4914 | 5 |
| 3 | 3.4140 | 5 |
| 4 | 3.3919 | 5 |
| 5 | 3.9259 | 5 |
| 6 | 4.7452 | 5 |
| 7 | 4.3006 | 5 |
| 8 | 4.0024 | 5 |
| 9 | 3.7873 | 5 |
| 10 | 4.0072 | 5 |
| 11 | 3.8960 | 5 |
| 12 | 3.9115 | 5 |
| 13 | 3.9378 | 5 |
| 14 | 3.8521 | 5 |
| 15 | 3.8499 | 5 |
| 16 | 3.8019 | 5 |
| 17 | 3.9897 | 5 |
| 18 | 4.1296 | 5 |
| 19 | 4.8544 | 5 |
| 20 | 4.5335 | 5 |
| 21 | 4.4258 | 5 |
| 22 | 4.3480 | 5 |
| 23 | 4.3543 | 5 |
