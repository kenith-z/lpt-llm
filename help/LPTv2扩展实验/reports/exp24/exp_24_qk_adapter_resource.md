# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 26.94 |
| decode_tokens_per_sec | 10.02 |
| first_token_latency_ms | 109.4150 |
| peak_memory_bytes | 2901550080 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| retnet_summary_norm_mean | 13.136007 |
| retnet_q_adapter_delta_norm_mean | 4.756325 |
| retnet_k_adapter_delta_norm_mean | 4.086860 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.738551 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 21.143420 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 79.8406 | 5 |
| 1 | 4.4182 | 5 |
| 2 | 5.0026 | 5 |
| 3 | 4.9982 | 5 |
| 4 | 4.6727 | 5 |
| 5 | 4.2583 | 5 |
| 6 | 4.2338 | 5 |
| 7 | 4.9317 | 5 |
| 8 | 4.9422 | 5 |
| 9 | 4.1954 | 5 |
| 10 | 4.4913 | 5 |
| 11 | 3.9963 | 5 |
| 12 | 4.4577 | 5 |
| 13 | 4.5472 | 5 |
| 14 | 4.1481 | 5 |
| 15 | 3.7014 | 5 |
| 16 | 3.4469 | 5 |
| 17 | 3.4469 | 5 |
| 18 | 3.4353 | 5 |
| 19 | 3.9911 | 5 |
| 20 | 4.8536 | 5 |
| 21 | 4.6821 | 5 |
| 22 | 4.4982 | 5 |
| 23 | 3.6590 | 5 |
