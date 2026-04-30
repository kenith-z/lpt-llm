# LongRoPE2 候选因子 Sweep 报告

## Checkpoint
- checkpoint_path: artifacts\lpt_ds_v1\text_pretrain\checkpoints\latest.pth
- training_stage: text_pretrain
- source_manifest: data\manifests\text_pretrain.json
- tokenizer_path: lpt_model\ds_tokenizer
- tokenizer_vocab_size: 129280
- tokenizer_bos/eos/pad: <｜begin▁of▁sentence｜> / <｜end▁of▁sentence｜> / <｜▁pad▁｜>

## Runtime
- device: cuda
- cache_strategy: session_rebuild
- total_latency_sec: 3.130174

## Candidates
| name | status | source | factor_mode | min_factor | max_factor | needle_exact | retrieval_exact | ppl | latency_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | ok | checkpoint:model_config | uniform | 16.0 | 16.0 | 0 | 0 | 128:1.503e+20 | 1.790895 |
| bootstrap | ok | bootstrap:sequence_length=8771 | uniform | 16.0 | 16.0 | 0 | 0 | 128:1.503e+20 | 1.319567 |
