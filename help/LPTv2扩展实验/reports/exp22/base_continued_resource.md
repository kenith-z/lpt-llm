# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 28.69 |
| decode_tokens_per_sec | 7.09 |
| first_token_latency_ms | 189.5173 |
| peak_memory_bytes | 2901292032 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.696305 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 19.102977 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 71.1290 | 5 |
| 1 | 6.8853 | 5 |
| 2 | 6.7860 | 5 |
| 3 | 6.5488 | 5 |
| 4 | 5.9455 | 5 |
| 5 | 5.8397 | 5 |
| 6 | 5.7798 | 5 |
| 7 | 6.1905 | 5 |
| 8 | 6.3926 | 5 |
| 9 | 6.5144 | 5 |
| 10 | 6.2673 | 5 |
| 11 | 6.2054 | 5 |
| 12 | 6.1193 | 5 |
| 13 | 6.1615 | 5 |
| 14 | 5.9484 | 5 |
| 15 | 6.0381 | 5 |
| 16 | 5.8980 | 5 |
| 17 | 5.7954 | 5 |
| 18 | 5.8335 | 5 |
| 19 | 5.9607 | 5 |
| 20 | 5.1892 | 5 |
| 21 | 5.0690 | 5 |
| 22 | 5.1793 | 5 |
| 23 | 5.1674 | 5 |
