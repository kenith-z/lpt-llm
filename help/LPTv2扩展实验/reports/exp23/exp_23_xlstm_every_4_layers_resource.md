# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 30.40 |
| decode_tokens_per_sec | 10.29 |
| first_token_latency_ms | 96.1035 |
| peak_memory_bytes | 2901301248 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 768 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.342065 |
| xlstm_adapter_delta_norm_mean | 1.953257 |
| router_entropy_mean | 0.830289 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 19.833432 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 73.2474 | 5 |
| 1 | 4.1323 | 5 |
| 2 | 4.0402 | 5 |
| 3 | 4.0158 | 5 |
| 4 | 4.7536 | 5 |
| 5 | 3.9664 | 5 |
| 6 | 3.9736 | 5 |
| 7 | 3.9829 | 5 |
| 8 | 4.6778 | 5 |
| 9 | 4.0082 | 5 |
| 10 | 4.0121 | 5 |
| 11 | 3.9517 | 5 |
| 12 | 4.1162 | 5 |
| 13 | 3.3113 | 5 |
| 14 | 3.2259 | 5 |
| 15 | 3.2455 | 5 |
| 16 | 4.3954 | 5 |
| 17 | 3.7873 | 5 |
| 18 | 3.6358 | 5 |
| 19 | 3.6893 | 5 |
| 20 | 4.4911 | 5 |
| 21 | 3.7799 | 5 |
| 22 | 3.7443 | 5 |
| 23 | 3.7562 | 5 |
