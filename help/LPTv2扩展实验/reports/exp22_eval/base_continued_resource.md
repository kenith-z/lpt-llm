# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 4.95 |
| decode_tokens_per_sec | 6.20 |
| first_token_latency_ms | 155.1942 |
| peak_memory_bytes | 2901292032 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.654800 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 25.736296 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 494.6797 | 5 |
| 1 | 85.2406 | 5 |
| 2 | 9.1108 | 5 |
| 3 | 8.2230 | 5 |
| 4 | 6.0094 | 5 |
| 5 | 6.2175 | 5 |
| 6 | 7.7545 | 5 |
| 7 | 6.2747 | 5 |
| 8 | 6.9409 | 5 |
| 9 | 7.6401 | 5 |
| 10 | 7.9368 | 5 |
| 11 | 8.0277 | 5 |
| 12 | 6.9451 | 5 |
| 13 | 7.7451 | 5 |
| 14 | 6.4743 | 5 |
| 15 | 5.6724 | 5 |
| 16 | 6.5240 | 5 |
| 17 | 6.5494 | 5 |
| 18 | 5.7028 | 5 |
| 19 | 5.8942 | 5 |
| 20 | 6.0962 | 5 |
| 21 | 6.5370 | 5 |
| 22 | 6.8630 | 5 |
| 23 | 7.1200 | 5 |
