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
| special_token_reset_configured | False |
| session_reset_count | 2 |
| effective_beta | 0.00010028 |
| memory_norm | 1.536531 |
| adapter_delta_norm | 1.058695 |
| memory_vs_eval_switch_logit_delta_l2 | 0.161677 |
| router_entropy | 1.698190 |

decision: `admit_instrumentation_only`

xLSTMAssist 状态连续性、boundary/session reset 和输入 adapter 均形成可观测机制；当前配置未启用 special token reset，不将该项作为失败条件。
