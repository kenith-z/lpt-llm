# LongRoPE2 候选因子 Sweep 报告

## Checkpoint
- checkpoint_path: artifacts\lpt_native_v1\text_pretrain\checkpoints\latest.pth
- training_stage: text_pretrain
- source_manifest: data\manifests\text_pretrain.json

## Runtime
- device: cuda
- cache_strategy: session_rebuild
- total_latency_sec: 3.358512

## Candidates
| name | status | source | factor_mode | min_factor | max_factor | needle_exact | retrieval_exact | ppl | latency_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | ok | checkpoint:model_config | uniform | 16.0 | 16.0 | 0 | 0 | 128:1.548e+23 | 1.872555 |
| bootstrap | ok | bootstrap:sequence_length=9120 | uniform | 16.0 | 16.0 | 0 | 0 | 128:1.548e+23 | 1.485321 |
