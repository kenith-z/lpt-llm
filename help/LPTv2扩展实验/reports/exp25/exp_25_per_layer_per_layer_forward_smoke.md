# LPT v2 Forward Smoke

- checkpoint: `artifacts\lpt_v2\experiments_exp25\exp_25_per_layer_per_layer\checkpoints\latest\model.pt`
- device: `cuda:0`
- dtype: `float16`
- use_kv_cache: `False`

| metric | value |
|---|---:|
| forward_ok | True |
| logits_finite | True |
| logits_shape | `[1, 32, 129280]` |
| loss | 11.769736 |
| ppl | 129280.053809 |
| state_count | 24 |
| attention_state_count | 0 |
| retnet_state_count | 24 |
| retnet_layer_state_count | 24 |
| retnet_parameter_sharing | `per_layer` |
| retnet_state_sharing | `per_layer` |
| retnet_sharing_group_size | 4 |
| retnet_assist_mode | `q_adapter` |
| retnet_adapter_target | `['q']` |
| retnet_k_adapter_enabled | False |
| retnet_q_adapter_delta_norm_mean | 4.461261 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| xlstm_state_count | 0 |
| expected_xlstm_state_count | 0 |
| paged_kv_page_count | 0 |
