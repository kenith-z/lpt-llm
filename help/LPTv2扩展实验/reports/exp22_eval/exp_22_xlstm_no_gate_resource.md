# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 21.99 |
| decode_tokens_per_sec | 6.10 |
| first_token_latency_ms | 160.0793 |
| peak_memory_bytes | 2901328896 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 3072 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.581707 |
| xlstm_adapter_delta_norm_mean | 2.627198 |
| router_entropy_mean | 0.788188 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 29.911964 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 97.4979 | 5 |
| 1 | 6.9223 | 5 |
| 2 | 7.4691 | 5 |
| 3 | 7.0951 | 5 |
| 4 | 7.3067 | 5 |
| 5 | 6.9541 | 5 |
| 6 | 7.0306 | 5 |
| 7 | 6.8589 | 5 |
| 8 | 6.2898 | 5 |
| 9 | 7.0046 | 5 |
| 10 | 6.2695 | 5 |
| 11 | 6.5045 | 5 |
| 12 | 8.2893 | 5 |
| 13 | 6.4810 | 5 |
| 14 | 6.7818 | 5 |
| 15 | 7.3255 | 5 |
| 16 | 7.0479 | 5 |
| 17 | 7.2898 | 5 |
| 18 | 7.5343 | 5 |
| 19 | 6.8569 | 5 |
| 20 | 6.7976 | 5 |
| 21 | 7.2237 | 5 |
| 22 | 6.8367 | 5 |
| 23 | 6.4352 | 5 |
