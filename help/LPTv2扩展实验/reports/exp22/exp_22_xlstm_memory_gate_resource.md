# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 24.99 |
| decode_tokens_per_sec | 5.98 |
| first_token_latency_ms | 174.9470 |
| peak_memory_bytes | 2905535488 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 3072 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.285014 |
| xlstm_adapter_delta_norm_mean | 2.170423 |
| router_entropy_mean | 0.752006 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 19.070584 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 78.3020 | 5 |
| 1 | 7.8177 | 5 |
| 2 | 7.9345 | 5 |
| 3 | 7.7053 | 5 |
| 4 | 7.6838 | 5 |
| 5 | 8.0613 | 5 |
| 6 | 7.8131 | 5 |
| 7 | 7.7684 | 5 |
| 8 | 7.5134 | 5 |
| 9 | 7.3737 | 5 |
| 10 | 7.6355 | 5 |
| 11 | 7.3264 | 5 |
| 12 | 7.3014 | 5 |
| 13 | 7.1021 | 5 |
| 14 | 6.6791 | 5 |
| 15 | 6.3471 | 5 |
| 16 | 6.7448 | 5 |
| 17 | 6.6652 | 5 |
| 18 | 6.4561 | 5 |
| 19 | 6.3861 | 5 |
| 20 | 6.9005 | 5 |
| 21 | 6.9591 | 5 |
| 22 | 6.4154 | 5 |
| 23 | 6.7117 | 5 |
