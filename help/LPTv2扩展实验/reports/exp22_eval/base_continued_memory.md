# LPT v2 xLSTMAssist Report

- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`

| metric | value |
|---|---:|
| prefill_token_count | None |
| decode_token_count | None |
| decay_count | None |
| boundary_reset_count | None |
| special_token_reset_count | None |
| special_token_reset_configured | False |
| session_reset_count | None |
| effective_beta | 0.00000000 |
| memory_norm | 0.000000 |
| adapter_delta_norm | 0.000000 |
| memory_vs_eval_switch_logit_delta_l2 | 0.000000 |
| router_entropy | 1.707333 |

decision: `close_or_debug`

xLSTMAssist 机制观测未全部通过，应检查状态池、reset 触发或 adapter。
