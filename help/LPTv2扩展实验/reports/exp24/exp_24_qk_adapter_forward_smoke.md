# LPT v2 Forward Smoke

- checkpoint: `artifacts\lpt_v2\experiments_exp24\exp_24_qk_adapter\checkpoints\latest\model.pt`
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
| retnet_assist_mode | `qk_adapter` |
| retnet_adapter_target | `['q', 'k']` |
| retnet_k_adapter_enabled | True |
| retnet_q_adapter_delta_norm_mean | 4.551141 |
| retnet_k_adapter_delta_norm_mean | 3.779367 |
| xlstm_state_count | 0 |
| expected_xlstm_state_count | 0 |
| paged_kv_page_count | 0 |
