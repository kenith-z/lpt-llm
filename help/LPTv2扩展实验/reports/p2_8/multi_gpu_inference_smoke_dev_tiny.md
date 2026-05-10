# LPT v2 P2-8 Multi-GPU Inference Smoke

- success: `True`
- execution_mode: `model_parallel`
- checkpoint: `None`
- prompt_length: `96`
- generated_tokens: `2`

| metric | value |
|---|---:|
| prefill_tokens_per_sec | 112.737 |
| decode_tokens_per_sec | 24.469 |
| first_decode_ms | 64.032 |
| layer_state_count | 4 |
| paged_kv_allocated_page_count | 4 |

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
estimated_device_mib={"cuda:0": 6.05, "cuda:1": 5.95}
```
