# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 22.81 |
| decode_tokens_per_sec | 5.71 |
| first_token_latency_ms | 174.3182 |
| peak_memory_bytes | 2901328896 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 3072 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.590821 |
| xlstm_adapter_delta_norm_mean | 2.501962 |
| router_entropy_mean | 0.782566 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 18.385893 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 89.5923 | 5 |
| 1 | 7.9092 | 5 |
| 2 | 7.6816 | 5 |
| 3 | 7.5664 | 5 |
| 4 | 7.7139 | 5 |
| 5 | 7.4254 | 5 |
| 6 | 7.4286 | 5 |
| 7 | 7.6762 | 5 |
| 8 | 7.7244 | 5 |
| 9 | 7.5543 | 5 |
| 10 | 7.3957 | 5 |
| 11 | 7.1251 | 5 |
| 12 | 7.2888 | 5 |
| 13 | 7.2127 | 5 |
| 14 | 7.7122 | 5 |
| 15 | 7.3565 | 5 |
| 16 | 7.2889 | 5 |
| 17 | 7.0870 | 5 |
| 18 | 7.0736 | 5 |
| 19 | 7.6072 | 5 |
| 20 | 7.4352 | 5 |
| 21 | 7.3237 | 5 |
| 22 | 7.1067 | 5 |
| 23 | 7.1407 | 5 |
