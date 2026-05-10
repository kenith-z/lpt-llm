# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 29.56 |
| decode_tokens_per_sec | 8.29 |
| first_token_latency_ms | 94.2449 |
| peak_memory_bytes | 2901328896 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 3072 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 2.591157 |
| xlstm_adapter_delta_norm_mean | 2.478241 |
| router_entropy_mean | 0.796813 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 18.636315 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 78.3766 | 5 |
| 1 | 4.9457 | 5 |
| 2 | 4.7126 | 5 |
| 3 | 4.6606 | 5 |
| 4 | 4.7423 | 5 |
| 5 | 4.0384 | 5 |
| 6 | 4.5287 | 5 |
| 7 | 4.6931 | 5 |
| 8 | 4.7551 | 5 |
| 9 | 5.3235 | 5 |
| 10 | 5.2746 | 5 |
| 11 | 5.2232 | 5 |
| 12 | 5.1507 | 5 |
| 13 | 5.1030 | 5 |
| 14 | 4.6413 | 5 |
| 15 | 4.5109 | 5 |
| 16 | 4.9731 | 5 |
| 17 | 5.3026 | 5 |
| 18 | 5.2847 | 5 |
| 19 | 5.2944 | 5 |
| 20 | 4.5835 | 5 |
| 21 | 4.5334 | 5 |
| 22 | 4.4810 | 5 |
| 23 | 4.4902 | 5 |
