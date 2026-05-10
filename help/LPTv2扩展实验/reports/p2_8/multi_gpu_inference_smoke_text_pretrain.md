# LPT v2 P2-8 Multi-GPU Inference Smoke

- success: `True`
- execution_mode: `model_parallel`
- checkpoint: `artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt`
- prompt_length: `32`
- generated_tokens: `1`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 35.347 |
| decode_tokens_per_sec | 7.961 |
| first_decode_ms | 125.612 |
| layer_state_count | 24 |
| paged_kv_allocated_page_count | 24 |

## Device Map

```text
execution_mode=model_parallel
primary_device=cuda:0
device_map_source=auto
visible_cuda_devices=[{"logical": "cuda:0", "visible": "0", "memory_gib": 16.0, "name": "NVIDIA GeForce RTX 4060 Ti"}, {"logical": "cuda:1", "visible": "1", "memory_gib": 15.93, "name": "NVIDIA GeForce RTX 5060 Ti"}]
layers 0 -> cuda:0
layers 1 -> cuda:1
layers 2 -> cuda:0
layers 3 -> cuda:1
layers 4 -> cuda:0
layers 5 -> cuda:1
layers 6 -> cuda:0
layers 7 -> cuda:1
layers 8 -> cuda:0
layers 9 -> cuda:1
layers 10 -> cuda:0
layers 11 -> cuda:1
layers 12 -> cuda:0
layers 13 -> cuda:1
layers 14 -> cuda:0
layers 15 -> cuda:1
layers 16 -> cuda:0
layers 17 -> cuda:1
layers 18 -> cuda:0
layers 19 -> cuda:1
layers 20 -> cuda:0
layers 21 -> cuda:1
layers 22 -> cuda:0
layers 23 -> cuda:1
estimated_device_mib={"cuda:0": 1528.81, "cuda:1": 1276.17}
```
