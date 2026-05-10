# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 29.40 |
| decode_tokens_per_sec | 9.29 |
| first_token_latency_ms | 120.8980 |
| peak_memory_bytes | 2901310464 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 1536 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.673609 |
| xlstm_adapter_delta_norm_mean | 2.636715 |
| router_entropy_mean | 0.710727 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 19.299075 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 76.5019 | 5 |
| 1 | 4.9568 | 5 |
| 2 | 5.7892 | 5 |
| 3 | 4.8363 | 5 |
| 4 | 5.3294 | 5 |
| 5 | 4.0292 | 5 |
| 6 | 4.7762 | 5 |
| 7 | 4.6700 | 5 |
| 8 | 6.0537 | 5 |
| 9 | 4.3493 | 5 |
| 10 | 3.9499 | 5 |
| 11 | 3.3112 | 5 |
| 12 | 3.7900 | 5 |
| 13 | 3.3086 | 5 |
| 14 | 4.7417 | 5 |
| 15 | 3.8593 | 5 |
| 16 | 4.5263 | 5 |
| 17 | 3.7781 | 5 |
| 18 | 4.5973 | 5 |
| 19 | 3.7522 | 5 |
| 20 | 4.4655 | 5 |
| 21 | 3.7676 | 5 |
| 22 | 4.7849 | 5 |
| 23 | 4.4072 | 5 |
