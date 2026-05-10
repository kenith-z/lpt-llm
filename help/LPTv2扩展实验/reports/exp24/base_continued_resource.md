# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 29.93 |
| decode_tokens_per_sec | 8.50 |
| first_token_latency_ms | 82.0268 |
| peak_memory_bytes | 2901292032 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| retnet_summary_norm_mean | 11.945847 |
| retnet_q_adapter_delta_norm_mean | 4.330010 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.814088 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 17.888150 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 79.0290 | 5 |
| 1 | 4.5673 | 5 |
| 2 | 4.6422 | 5 |
| 3 | 4.8063 | 5 |
| 4 | 5.2077 | 5 |
| 5 | 4.7788 | 5 |
| 6 | 4.7043 | 5 |
| 7 | 4.6478 | 5 |
| 8 | 4.7222 | 5 |
| 9 | 4.5900 | 5 |
| 10 | 4.5264 | 5 |
| 11 | 4.4928 | 5 |
| 12 | 4.6074 | 5 |
| 13 | 4.5487 | 5 |
| 14 | 4.4613 | 5 |
| 15 | 4.4301 | 5 |
| 16 | 4.4003 | 5 |
| 17 | 4.5453 | 5 |
| 18 | 4.5232 | 5 |
| 19 | 4.6215 | 5 |
| 20 | 4.9744 | 5 |
| 21 | 4.4877 | 5 |
| 22 | 4.4241 | 5 |
| 23 | 4.5125 | 5 |
