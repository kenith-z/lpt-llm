# LPT v2 P2-8 Model Parallel Train Smoke

- success: `True`
- execution_mode: `model_parallel`
- checkpoint: `None`
- batch_size: `1`
- sequence_length: `16`
- steps: `1`

| metric | value |
|---|---:|
| last_loss | 248.88763427734375 |
| grad_norm | 93.16970421494646 |
| elapsed_seconds | 0.9346506000001682 |

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
estimated_device_mib={"cuda:0": 22.3, "cuda:1": 21.89}
```
