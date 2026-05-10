# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 26.61 |
| decode_tokens_per_sec | 7.39 |
| first_token_latency_ms | 157.5929 |
| peak_memory_bytes | 2901292032 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.815915 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 17.953770 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 74.0890 | 5 |
| 1 | 6.0370 | 5 |
| 2 | 5.6157 | 5 |
| 3 | 5.4476 | 5 |
| 4 | 5.5087 | 5 |
| 5 | 5.8273 | 5 |
| 6 | 5.7307 | 5 |
| 7 | 6.0858 | 5 |
| 8 | 5.8653 | 5 |
| 9 | 5.9010 | 5 |
| 10 | 5.8552 | 5 |
| 11 | 6.4299 | 5 |
| 12 | 6.4176 | 5 |
| 13 | 6.6516 | 5 |
| 14 | 5.9506 | 5 |
| 15 | 5.6689 | 5 |
| 16 | 5.6081 | 5 |
| 17 | 5.6159 | 5 |
| 18 | 5.7841 | 5 |
| 19 | 6.1224 | 5 |
| 20 | 6.3421 | 5 |
| 21 | 5.8219 | 5 |
| 22 | 5.6322 | 5 |
| 23 | 5.6921 | 5 |
