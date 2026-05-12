# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 30.61 |
| decode_tokens_per_sec | 11.98 |
| first_token_latency_ms | 81.4780 |
| peak_memory_bytes | 2900479488 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 3072 |
| retnet_state_slot_count | 24 |
| retnet_layer_state_count | 24 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `per_layer` |
| retnet_summary_norm_mean | 10.316284 |
| retnet_q_adapter_delta_norm_mean | 3.331058 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.772738 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 18.023885 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 72.4265 | 5 |
| 1 | 3.5282 | 5 |
| 2 | 3.3425 | 5 |
| 3 | 3.4156 | 5 |
| 4 | 3.4074 | 5 |
| 5 | 3.3975 | 5 |
| 6 | 3.4226 | 5 |
| 7 | 3.3936 | 5 |
| 8 | 3.2958 | 5 |
| 9 | 3.5607 | 5 |
| 10 | 3.7444 | 5 |
| 11 | 4.0213 | 5 |
| 12 | 3.9156 | 5 |
| 13 | 3.7511 | 5 |
| 14 | 3.8947 | 5 |
| 15 | 3.9118 | 5 |
| 16 | 3.8068 | 5 |
| 17 | 3.9082 | 5 |
| 18 | 4.4050 | 5 |
| 19 | 4.3906 | 5 |
| 20 | 3.8796 | 5 |
| 21 | 3.1974 | 5 |
| 22 | 3.1917 | 5 |
| 23 | 3.2060 | 5 |
