# LPT v2 Forward Smoke

- checkpoint: `artifacts\lpt_v2\experiments_exp27\base_continued\checkpoints\latest\model.pt`
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
| retnet_state_count | 6 |
| retnet_layer_state_count | 24 |
| expected_retnet_layer_state_count | 24 |
| retnet_assist_layers | `all_layers` |
| retnet_assist_selected_layers | `[]` |
| retnet_adapter_rank | 16 |
| retnet_parameter_sharing | `global` |
| retnet_state_sharing | `group` |
| retnet_sharing_group_size | 4 |
| retnet_assist_mode | `q_adapter` |
| retnet_adapter_target | `['q']` |
| retnet_k_adapter_enabled | False |
| retnet_context_adapter_enabled | False |
| retnet_q_adapter_delta_norm_mean | 3.564406 |
| retnet_k_adapter_delta_norm_mean | 0.000000 |
| retnet_context_adapter_delta_norm_mean | 0.000000 |
| retnet_alpha_context_mean | 0.00000000 |
| xlstm_state_count | 0 |
| expected_xlstm_state_count | 0 |
| paged_kv_page_count | 0 |
