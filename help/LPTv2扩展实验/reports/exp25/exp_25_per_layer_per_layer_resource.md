# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 31.18 |
| decode_tokens_per_sec | 11.00 |
| first_token_latency_ms | 80.3940 |
| peak_memory_bytes | 2904495104 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| retnet_state_slot_count | 24 |
| retnet_layer_state_count | 24 |
| retnet_parameter_sharing | `per_layer` |
| retnet_state_sharing | `per_layer` |
| retnet_summary_norm_mean | 11.956980 |
| retnet_q_adapter_delta_norm_mean | 4.515781 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.821304 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 19.762910 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 71.2692 | 5 |
| 1 | 3.6103 | 5 |
| 2 | 4.1408 | 5 |
| 3 | 4.0622 | 5 |
| 4 | 4.2998 | 5 |
| 5 | 4.6134 | 5 |
| 6 | 4.5260 | 5 |
| 7 | 4.0251 | 5 |
| 8 | 3.9370 | 5 |
| 9 | 3.9755 | 5 |
| 10 | 3.8154 | 5 |
| 11 | 3.9409 | 5 |
| 12 | 4.1373 | 5 |
| 13 | 3.9067 | 5 |
| 14 | 3.5119 | 5 |
| 15 | 3.2736 | 5 |
| 16 | 3.2022 | 5 |
| 17 | 3.1909 | 5 |
| 18 | 3.2076 | 5 |
| 19 | 3.1927 | 5 |
| 20 | 3.2250 | 5 |
| 21 | 3.1742 | 5 |
| 22 | 3.2054 | 5 |
| 23 | 3.1882 | 5 |
