# LPT v2 xLSTMAssist Report

- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_token_count | 8 |
| decode_token_count | 9 |
| decay_count | 0 |
| boundary_reset_count | 1 |
| special_token_reset_count | 1 |
| session_reset_count | 2 |
| effective_beta | 0.00010028 |
| memory_norm | 1.380770 |
| adapter_delta_norm | 0.859500 |
| memory_vs_eval_switch_logit_delta_l2 | 0.136503 |
| router_entropy | 1.666870 |

decision: `close_or_debug`

xLSTMAssist 机制观测未全部通过，应检查状态池、reset 触发或 adapter。
