# LPT v2 Forward Smoke

- checkpoint: `artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_all_layers\checkpoints\latest\model.pt`
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
| xlstm_state_count | 24 |
| expected_xlstm_state_count | 24 |
| paged_kv_page_count | 0 |
