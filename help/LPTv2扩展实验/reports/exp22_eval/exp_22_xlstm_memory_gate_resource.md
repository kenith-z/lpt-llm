# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 22.99 |
| decode_tokens_per_sec | 6.15 |
| first_token_latency_ms | 166.2500 |
| peak_memory_bytes | 2905535488 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 3072 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.332007 |
| xlstm_adapter_delta_norm_mean | 2.340869 |
| router_entropy_mean | 0.536397 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 31.485127 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 91.1488 | 5 |
| 1 | 6.6800 | 5 |
| 2 | 7.5727 | 5 |
| 3 | 6.6471 | 5 |
| 4 | 7.3830 | 5 |
| 5 | 6.7086 | 5 |
| 6 | 6.8334 | 5 |
| 7 | 7.0921 | 5 |
| 8 | 6.7383 | 5 |
| 9 | 6.9963 | 5 |
| 10 | 6.9878 | 5 |
| 11 | 6.9790 | 5 |
| 12 | 6.3664 | 5 |
| 13 | 6.8159 | 5 |
| 14 | 7.4964 | 5 |
| 15 | 7.3189 | 5 |
| 16 | 6.8212 | 5 |
| 17 | 6.9860 | 5 |
| 18 | 7.3779 | 5 |
| 19 | 5.9968 | 5 |
| 20 | 6.3384 | 5 |
| 21 | 6.8533 | 5 |
| 22 | 6.8770 | 5 |
| 23 | 5.8433 | 5 |
