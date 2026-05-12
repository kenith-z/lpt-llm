# LPT v2 Long Context Admission

- preset: `lpt_v2_small_base`
- device: `cuda:0`
- dtype: `bfloat16`
- sequence_length: `4100`
- attention_window_size: `2048`
- needle_depth: `0.0`
- checkpoint: `artifacts\lpt_v2\experiments_exp26\exp_26_retnet_rank32\checkpoints\latest\model.pt`
- training_stage: `exp_26_retnet_rank32`
- global_step: `1164`

| metric | assist | no_assist | delta |
|---|---:|---:|---:|
| needle_rank | 25257 | n/a | n/a |
| needle_logprob | -24.4114 | n/a | n/a |
| long_text_ppl | 485165195.41 | n/a | n/a |

decision: `admit_checkpoint_path`

已加载真实 v2 checkpoint 完成长上下文前向、PPL 与状态池准入；质量结论需结合独立验证集。
