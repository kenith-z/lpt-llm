# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 25.90 |
| decode_tokens_per_sec | 11.88 |
| first_token_latency_ms | 79.9611 |
| peak_memory_bytes | 2901352448 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 24 |
| retnet_parameter_sharing | `group` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 12.683956 |
| retnet_q_adapter_delta_norm_mean | 4.772937 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.790981 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 19.546822 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 77.3257 | 5 |
| 1 | 4.3246 | 5 |
| 2 | 4.2691 | 5 |
| 3 | 4.3646 | 5 |
| 4 | 4.2820 | 5 |
| 5 | 3.4653 | 5 |
| 6 | 3.4626 | 5 |
| 7 | 3.3743 | 5 |
| 8 | 3.6896 | 5 |
| 9 | 4.1032 | 5 |
| 10 | 4.3485 | 5 |
| 11 | 4.5615 | 5 |
| 12 | 4.5832 | 5 |
| 13 | 4.4614 | 5 |
| 14 | 4.4187 | 5 |
| 15 | 4.5062 | 5 |
| 16 | 4.3556 | 5 |
| 17 | 3.8647 | 5 |
| 18 | 3.8462 | 5 |
| 19 | 3.7755 | 5 |
| 20 | 3.7454 | 5 |
| 21 | 3.7862 | 5 |
| 22 | 3.6558 | 5 |
| 23 | 3.2213 | 5 |
