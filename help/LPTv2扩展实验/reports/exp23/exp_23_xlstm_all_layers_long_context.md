# LPT v2 Long Context Admission

- preset: `checkpoint`
- device: `cuda:0`
- dtype: `bfloat16`
- sequence_length: `2052`
- attention_window_size: `2048`
- checkpoint: `artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_all_layers\checkpoints\latest\model.pt`
- training_stage: `exp_23_xlstm_all_layers`
- global_step: `1164`

| metric | assist | no_assist | delta |
|---|---:|---:|---:|
| needle_rank | 23362 | n/a | n/a |
| needle_logprob | -23.8479 | n/a | n/a |
| long_text_ppl | 485165195.41 | n/a | n/a |

decision: `close_or_debug`

checkpoint 可加载但长上下文状态未跨越局部窗口，应检查配置或输入长度。
