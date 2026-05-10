# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 28.72 |
| decode_tokens_per_sec | 10.72 |
| first_token_latency_ms | 98.1295 |
| peak_memory_bytes | 2901301248 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 768 |
| xlstm_effective_beta_mean | 0.00010028 |
| xlstm_memory_norm_mean | 4.414667 |
| xlstm_adapter_delta_norm_mean | 6.507231 |
| router_entropy_mean | 0.779805 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 21.527145 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 69.4837 | 5 |
| 1 | 4.3655 | 5 |
| 2 | 4.0311 | 5 |
| 3 | 3.6285 | 5 |
| 4 | 3.4312 | 5 |
| 5 | 3.4001 | 5 |
| 6 | 3.3570 | 5 |
| 7 | 3.3670 | 5 |
| 8 | 3.8077 | 5 |
| 9 | 4.0536 | 5 |
| 10 | 4.8101 | 5 |
| 11 | 4.5846 | 5 |
| 12 | 4.4841 | 5 |
| 13 | 4.4220 | 5 |
| 14 | 3.8675 | 5 |
| 15 | 3.2094 | 5 |
| 16 | 3.8710 | 5 |
| 17 | 3.8527 | 5 |
| 18 | 10.4733 | 5 |
| 19 | 4.5859 | 5 |
| 20 | 3.8455 | 5 |
| 21 | 3.7564 | 5 |
| 22 | 3.7354 | 5 |
| 23 | 3.7894 | 5 |
