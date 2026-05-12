# LPT v2 Resource Report

- profile: `checkpoint`
- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 27.81 |
| decode_tokens_per_sec | 8.40 |
| first_token_latency_ms | 123.9048 |
| peak_memory_bytes | 2900461056 |
| paged_kv_page_bytes | 491520 |
| retnet_state_bytes | 768 |
| retnet_state_slot_count | 6 |
| retnet_layer_state_count | 12 |
| expected_retnet_layer_state_count | 12 |
| retnet_enabled_layer_count | 12 |
| retnet_assist_layers | `every_2_layers` |
| retnet_assist_selected_layers | `[]` |
| retnet_adapter_rank | 16 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_summary_norm_mean | 11.401059 |
| retnet_q_adapter_delta_norm_mean | 3.326281 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_memory_state_bytes | 0 |
| xlstm_effective_beta_mean | 0.00000000 |
| xlstm_memory_norm_mean | 0.000000 |
| xlstm_adapter_delta_norm_mean | 0.000000 |
| router_entropy_mean | 0.745050 |
| load_balance_loss_mean | 3.000000 |
| router_z_loss_mean | 21.334541 |

## Layer Time

| layer | mean_ms | calls |
|---:|---:|---:|
| 0 | 73.8868 | 5 |
| 1 | 4.8449 | 5 |
| 2 | 5.4306 | 5 |
| 3 | 4.9217 | 5 |
| 4 | 6.1793 | 5 |
| 5 | 5.1367 | 5 |
| 6 | 6.2064 | 5 |
| 7 | 5.0899 | 5 |
| 8 | 5.9972 | 5 |
| 9 | 4.7132 | 5 |
| 10 | 5.7233 | 5 |
| 11 | 4.7485 | 5 |
| 12 | 6.0210 | 5 |
| 13 | 4.9218 | 5 |
| 14 | 6.5339 | 5 |
| 15 | 5.4430 | 5 |
| 16 | 6.1796 | 5 |
| 17 | 4.2639 | 5 |
| 18 | 5.0768 | 5 |
| 19 | 4.1272 | 5 |
| 20 | 5.3071 | 5 |
| 21 | 4.1836 | 5 |
| 22 | 5.1844 | 5 |
| 23 | 4.2454 | 5 |
