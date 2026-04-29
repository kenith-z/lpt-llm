
# 20260426 Sequence Packing GPU 基准实验报告

## 1. 实验目的

本次实验用于验证两件事：

1. 将训练态和推理态的 `rope_cache_max_sequence_length` 拆开后，训练显存预算是否真正从长推理缓存中回收。
2. 在真实 GPU 训练下，`sequence packing` / token-based batching 是否继续带来可观的吞吐提升和显存收益。

## 2. 实验背景

在拆分前，模型只有一套共享 `rope_cache`，训练与推理都绑定到同一个上限。虽然训练实际只使用 `768` 长度上下文，但如果推理侧上限配置为 `32768`，训练阶段仍会长期承担长缓存带来的额外显存开销。

为避免这个问题，当前代码已经改为：

- 训练态使用 `GlobalConfig.train_rope_cache_max_sequence_length`
- 推理态使用 `GlobalConfig.inference_rope_cache_max_sequence_length`
- 模型内部按运行场景惰性构建并选择 RoPE cache，而不是启动时一次性把长推理缓存也建出来

相关实现位置：

- [lpt_config/config.py](<F:\yuan\pyPro\moe_llm\lpt_config\config.py:31>)
- [lpt_model/model.py](<F:\yuan\pyPro\moe_llm\lpt_model\model.py:1110>)
- [lpt_training/train.py](<F:\yuan\pyPro\moe_llm\lpt_training\train.py:166>)
- [lpt_inference/session.py](<F:\yuan\pyPro\moe_llm\lpt_inference\session.py:208>)

## 3. 实验环境

- 日期：2026-04-26
- 操作系统：Windows 10
- Python 环境：项目 `.venv`
- GPU：`NVIDIA GeForce RTX 5060 Ti`
- CUDA 可见显存上限：约 `15.93 GiB`
- 数据集 manifest：`data/manifests/text_pretrain.json`
- 实际有效样本源：`paper_dev`
- 有效样本数：`500`

说明：

- 用户侧提到“16G 独显 + 共享显存合计 31.9G”，但本次 PyTorch/CUDA 基准过程中，实际可稳定利用的硬上限仍表现为约 `15.93 GiB` 的独显容量。

## 4. 实验方法

### 4.1 数据与设置

- 训练最大序列长度：`768`
- 训练态 RoPE cache 上限：`768`
- 推理态 RoPE cache 上限：`32768`
- warmup steps：`5`
- measured steps：`40`
- 基准对象：真实前向 + 反向 + `optimizer.step()`

### 4.2 基准模式

对以下场景分别测量：

1. `batch_size=16`，`packing=off/on`
2. `batch_size=32`，`packing=off`
3. `batch_size=32`，`packing=on`

`bs32` 采用拆分运行：

- 先单独跑 `packing=off`
- 再单独跑 `packing=on`

这样即使 `packing=off` 出现 OOM，也不会影响 `packing=on` 的有效结果。

### 4.3 使用命令

```powershell
.\.venv\Scripts\python.exe tools/benchmark_sequence_packing.py `
  --manifest data/manifests/text_pretrain.json `
  --manifest-kind text `
  --batch-size 16 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode both `
  --output-json .tmp_benchmarks/sequence_packing_text_pretrain_bs16_runtime_split.json
```

```powershell
.\.venv\Scripts\python.exe tools/benchmark_sequence_packing.py `
  --manifest data/manifests/text_pretrain.json `
  --manifest-kind text `
  --batch-size 32 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode off `
  --output-json .tmp_benchmarks/sequence_packing_text_pretrain_bs32_runtime_split_off.json
```

```powershell
.\.venv\Scripts\python.exe tools/benchmark_sequence_packing.py `
  --manifest data/manifests/text_pretrain.json `
  --manifest-kind text `
  --batch-size 32 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode on `
  --output-json .tmp_benchmarks/sequence_packing_text_pretrain_bs32_runtime_split_on.json
```

### 4.4 补充实验：`batch_size=4`，对比 `train_max_sequence_length=768/7680`

为补充观察“小 batch 下 packing 的收益形态”，额外增加两组实验：

1. `batch_size=4`，`train_max_sequence_length=768`
2. `batch_size=4`，`train_max_sequence_length=7680`

补充命令如下：

```powershell
.\.venv\Scripts\python.exe tools/benchmark_sequence_packing.py `
  --manifest data/manifests/text_pretrain.json `
  --manifest-kind text `
  --batch-size 4 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode both `
  --train-max-sequence-length 768 `
  --output-json .tmp_benchmarks/sequence_packing_text_pretrain_bs4_trainlen768.json
```

```powershell
.\.venv\Scripts\python.exe tools/benchmark_sequence_packing.py `
  --manifest data/manifests/text_pretrain.json `
  --manifest-kind text `
  --batch-size 4 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode both `
  --train-max-sequence-length 7680 `
  --output-json .tmp_benchmarks/sequence_packing_text_pretrain_bs4_trainlen7680.json
```

补充说明：

- 这两组实验中，如果未额外手动指定 `train_rope_cache_max_sequence_length`，benchmark 脚本会自动令它跟随 `train_max_sequence_length` 一起变化。
- 当前 `text_pretrain` 有效样本在这两个长度上限下的真实长度分布完全一致：
  - `count=500`
  - `mean=130.826`
  - `p95=183`
  - `max=322`

这意味着：

- `768 -> 7680` 不会改变这批样本的截断行为
- 如果两组结果非常接近，这是符合数据分布预期的，不是实验失效

## 5. 基准结果

### 5.1 `batch_size=16`

| 模式 | wall-clock | active tokens/s | token utilization | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| packing off | 36.420481s | 2254.006 | 0.663230 | 13.923 GB | 28.006 GB |
| packing on | 30.477734s | 2693.409 | 0.857954 | 8.846 GB | 26.461 GB |

对比：

- 吞吐提升：`1.1949x`
- wall-clock 改善：`1.1950x`
- token 利用率提升：`+0.1947`
- 峰值 allocated 显存下降：约 `5.08 GB`

### 5.2 `batch_size=32`

#### packing off

结果：失败，发生 CUDA OOM。

关键报错信息：

- 当时 PyTorch 已分配显存：`15.63 GiB`
- 额外申请失败大小：`4.56 GiB`

结论：

- 即使训练态和推理态的 RoPE cache 已经拆开，未开启 packing 的 `bs32` 仍超出这张卡的训练承载能力。

#### packing on

| 模式 | wall-clock | active tokens/s | token utilization | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| packing on | 72.861788s | 2257.384 | 0.867129 | 14.252 GB | 27.916 GB |

补充观察：

- `bs32 + packing` 可以稳定完成训练基准
- 但其 `active tokens/s` 与 `bs16 + packing` 基本持平，没有进一步带来吞吐提升

### 5.3 `batch_size=4`，`train_max_sequence_length=768`

| 模式 | wall-clock | active tokens/s | token utilization | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| packing off | 22.314629s | 920.338 | 0.781707 | 3.965 GB | 10.512 GB |
| packing on | 22.604716s | 920.958 | 0.993225 | 3.278 GB | 7.852 GB |

对比：

- 吞吐提升：`1.0007x`
- wall-clock 变化：`0.9872x`
- token 利用率提升：`+0.2115`
- 峰值 allocated 显存下降：约 `0.69 GB`

### 5.4 `batch_size=4`，`train_max_sequence_length=7680`

| 模式 | wall-clock | active tokens/s | token utilization | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| packing off | 22.384506s | 917.465 | 0.781707 | 3.965 GB | 10.512 GB |
| packing on | 22.476845s | 926.198 | 0.993225 | 3.278 GB | 7.852 GB |

对比：

- 吞吐提升：`1.0095x`
- wall-clock 变化：`0.9959x`
- token 利用率提升：`+0.2115`
- 峰值 allocated 显存下降：约 `0.69 GB`

## 6. 结果分析

### 6.1 训练/推理 RoPE cache 拆分是有效的

本次基准已不再依赖“在 benchmark 脚本里临时覆盖全局 `rope_cache_max_sequence_length`”的旧做法，而是直接验证代码改造后的默认运行时行为：

- 训练态只构建并使用 `768` 长度的训练 cache
- 推理态需要长上下文时，才惰性构建 `32768` 长度的推理 cache

这说明训练显存预算已经从长推理缓存中回收出来。

### 6.2 `sequence packing` 在真实 GPU 上收益明确

在 `bs16` 下，packing 同时改善了：

- 吞吐
- token 利用率
- 显存峰值

这说明第 6 点不是只有“理论上减少 padding”，而是在真实训练路径里实际生效。

### 6.3 `bs32` 的意义主要是容量边界，而不是更高吞吐

结果显示：

- `bs32 + no packing`：不可运行
- `bs32 + packing`：可运行
- `bs32 + packing`：吞吐没有明显超过 `bs16 + packing`

因此在当前数据分布和这张 `RTX 5060 Ti 16GB` 上，`bs32` 更像是“packing 带来的容量扩展边界”，而不是新的性能甜点。

### 6.4 当前最实用的训练配置

综合吞吐、稳定性和显存占用，当前最实用的选择是：

- `batch_size=16`
- `sequence_packing_enabled=True`

### 6.5 为什么 `bs=4, train_max_sequence_length=768/7680` 两组结果几乎相同

这部分结果接近，原因很直接：

- 当前参与实验的 `paper_dev` 样本最大只有 `322` token
- 因此无论训练最大长度设为 `768` 还是 `7680`，都不会触发额外截断，也不会增加单样本有效 token 数

这说明本次补充实验主要回答的是：

- 在“小 batch、样本本身不长”的场景下，packing 还能带来什么

结论是：

- token 利用率和显存占用仍然明显改善
- 但吞吐提升已经接近于零

这通常意味着在 `bs=4` 这种较小批次下，训练瓶颈更多来自：

- kernel launch / step 固定开销
- optimizer / scheduler 开销
- 小 batch 下算子利用率不足

而不是 padding 浪费本身。

## 7. 产物记录

本次实验结果见附录。

说明：

- `sequence_packing_text_pretrain_bs32_runtime_split_off.json` 未生成有效结果文件，因为该进程在基准过程中 OOM 退出。

## 8. 相关验证

代码改动后已额外通过：

- `python -m compileall lpt_config lpt_model lpt_inference lpt_training tools/benchmark_sequence_packing.py tests/test_model_behavior.py tests/test_inference.py tests/test_training_recipe.py`
- `.\.venv\Scripts\python.exe -m unittest tests.test_model_behavior`
- `.\.venv\Scripts\python.exe -m unittest tests.test_inference`
- `.\.venv\Scripts\python.exe -m unittest tests.test_training_recipe`

其中新增/更新的关键测试包括：

- [tests/test_model_behavior.py](<F:\yuan\pyPro\moe_llm\tests\test_model_behavior.py:259>)
- [tests/test_inference.py](<F:\yuan\pyPro\moe_llm\tests\test_inference.py:327>)
- [tests/test_training_recipe.py](<F:\yuan\pyPro\moe_llm\tests\test_training_recipe.py:190>)

## 9. 实验结论

训练态/推理态 RoPE cache 拆分已经有效落地，`sequence packing` 在真实 GPU 训练中继续带来显著收益，并把 `bs32` 从“不可运行”推进到了“可运行但不更快”的状态；当前这张卡上更合理的默认训练点仍是 `bs16 + packing`。

---

## 附录：基准原始 JSON 数据文件

### A.1 sequence_packing_text_pretrain_bs4_trainlen768.json

```json
{
  "config": {
    "batch_size": 4,
    "warmup_steps": 5,
    "measured_steps": 40,
    "packing_mode": "both",
    "train_max_sequence_length": 768,
    "train_rope_cache_max_sequence_length": 768,
    "inference_rope_cache_max_sequence_length": 32768
  },
  "benchmark": [
    {
      "sequence_packing_enabled": false,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 4,
      "raw_samples": 160,
      "packed_rows": 160,
      "active_tokens": 20537,
      "padded_tokens": 26272,
      "token_utilization": 0.781707,
      "wall_clock_seconds": 22.314629,
      "avg_step_ms": 557.866,
      "active_tokens_per_sec": 920.338,
      "padded_tokens_per_sec": 1177.344,
      "raw_samples_per_sec": 7.17,
      "peak_memory_allocated_gb": 3.965,
      "peak_memory_reserved_gb": 10.512,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    },
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 4,
      "raw_samples": 160,
      "packed_rows": 40,
      "active_tokens": 20818,
      "padded_tokens": 20960,
      "token_utilization": 0.993225,
      "wall_clock_seconds": 22.604716,
      "avg_step_ms": 565.118,
      "active_tokens_per_sec": 920.958,
      "padded_tokens_per_sec": 927.24,
      "raw_samples_per_sec": 7.078,
      "peak_memory_allocated_gb": 3.278,
      "peak_memory_reserved_gb": 7.852,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ],
  "comparison": {
    "active_tokens_per_sec_speedup": 1.0007,
    "token_utilization_gain": 0.2115,
    "wall_clock_ratio": 0.9872
  }
}
```

### A.2 sequence_packing_text_pretrain_bs4_trainlen7680.json

```json
{
  "config": {
    "batch_size": 4,
    "warmup_steps": 5,
    "measured_steps": 40,
    "packing_mode": "both",
    "train_max_sequence_length": 7680,
    "train_rope_cache_max_sequence_length": 7680,
    "inference_rope_cache_max_sequence_length": 32768
  },
  "benchmark": [
    {
      "sequence_packing_enabled": false,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 4,
      "raw_samples": 160,
      "packed_rows": 160,
      "active_tokens": 20537,
      "padded_tokens": 26272,
      "token_utilization": 0.781707,
      "wall_clock_seconds": 22.384506,
      "avg_step_ms": 559.613,
      "active_tokens_per_sec": 917.465,
      "padded_tokens_per_sec": 1173.669,
      "raw_samples_per_sec": 7.148,
      "peak_memory_allocated_gb": 3.965,
      "peak_memory_reserved_gb": 10.512,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    },
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 4,
      "raw_samples": 160,
      "packed_rows": 40,
      "active_tokens": 20818,
      "padded_tokens": 20960,
      "token_utilization": 0.993225,
      "wall_clock_seconds": 22.476845,
      "avg_step_ms": 561.921,
      "active_tokens_per_sec": 926.198,
      "padded_tokens_per_sec": 932.515,
      "raw_samples_per_sec": 7.118,
      "peak_memory_allocated_gb": 3.278,
      "peak_memory_reserved_gb": 7.852,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ],
  "comparison": {
    "active_tokens_per_sec_speedup": 1.0095,
    "token_utilization_gain": 0.2115,
    "wall_clock_ratio": 0.9959
  }
}
```

### A.3 sequence_packing_text_pretrain_bs8_current_rope.json

```json
{
  "benchmark": [
    {
      "sequence_packing_enabled": false,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 8,
      "raw_samples": 320,
      "packed_rows": 320,
      "active_tokens": 42101,
      "padded_tokens": 60416,
      "token_utilization": 0.696852,
      "wall_clock_seconds": 27.463382,
      "avg_step_ms": 686.585,
      "active_tokens_per_sec": 1532.987,
      "padded_tokens_per_sec": 2199.875,
      "raw_samples_per_sec": 11.652,
      "peak_memory_allocated_gb": 7.788,
      "peak_memory_reserved_gb": 21.426,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    },
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 8,
      "raw_samples": 320,
      "packed_rows": 80,
      "active_tokens": 41913,
      "padded_tokens": 56416,
      "token_utilization": 0.742928,
      "wall_clock_seconds": 24.210709,
      "avg_step_ms": 605.268,
      "active_tokens_per_sec": 1731.176,
      "padded_tokens_per_sec": 2330.209,
      "raw_samples_per_sec": 13.217,
      "peak_memory_allocated_gb": 5.208,
      "peak_memory_reserved_gb": 11.016,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ],
  "comparison": {
    "active_tokens_per_sec_speedup": 1.1293,
    "token_utilization_gain": 0.0461,
    "wall_clock_ratio": 1.1343
  }
}
```

### A.4 sequence_packing_text_pretrain_bs8_current_rope_seq.json

```json
{
  "benchmark": [
    {
      "sequence_packing_enabled": false,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 8,
      "raw_samples": 320,
      "packed_rows": 320,
      "active_tokens": 42101,
      "padded_tokens": 60416,
      "token_utilization": 0.696852,
      "wall_clock_seconds": 25.355408,
      "avg_step_ms": 633.885,
      "active_tokens_per_sec": 1660.435,
      "padded_tokens_per_sec": 2382.766,
      "raw_samples_per_sec": 12.621,
      "peak_memory_allocated_gb": 7.788,
      "peak_memory_reserved_gb": 22.217,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    },
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 8,
      "raw_samples": 320,
      "packed_rows": 80,
      "active_tokens": 41913,
      "padded_tokens": 56416,
      "token_utilization": 0.742928,
      "wall_clock_seconds": 23.911934,
      "avg_step_ms": 597.798,
      "active_tokens_per_sec": 1752.807,
      "padded_tokens_per_sec": 2359.324,
      "raw_samples_per_sec": 13.382,
      "peak_memory_allocated_gb": 5.208,
      "peak_memory_reserved_gb": 11.016,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ],
  "comparison": {
    "active_tokens_per_sec_speedup": 1.0556,
    "token_utilization_gain": 0.0461,
    "wall_clock_ratio": 1.0604
  }
}
```

### A.5 sequence_packing_text_pretrain_bs16_runtime_split.json

```json
{
  "config": {
    "batch_size": 16,
    "warmup_steps": 5,
    "measured_steps": 40,
    "packing_mode": "both",
    "train_rope_cache_max_sequence_length": 768,
    "inference_rope_cache_max_sequence_length": 32768
  },
  "benchmark": [
    {
      "sequence_packing_enabled": false,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 16,
      "raw_samples": 628,
      "packed_rows": 628,
      "active_tokens": 82092,
      "padded_tokens": 123776,
      "token_utilization": 0.66323,
      "wall_clock_seconds": 36.420481,
      "avg_step_ms": 910.512,
      "active_tokens_per_sec": 2254.006,
      "padded_tokens_per_sec": 3398.527,
      "raw_samples_per_sec": 17.243,
      "peak_memory_allocated_gb": 13.923,
      "peak_memory_reserved_gb": 28.006,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    },
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 16,
      "raw_samples": 628,
      "packed_rows": 131,
      "active_tokens": 82089,
      "padded_tokens": 95680,
      "token_utilization": 0.857954,
      "wall_clock_seconds": 30.477734,
      "avg_step_ms": 761.943,
      "active_tokens_per_sec": 2693.409,
      "padded_tokens_per_sec": 3139.341,
      "raw_samples_per_sec": 20.605,
      "peak_memory_allocated_gb": 8.846,
      "peak_memory_reserved_gb": 26.461,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ],
  "comparison": {
    "active_tokens_per_sec_speedup": 1.1949,
    "token_utilization_gain": 0.1947,
    "wall_clock_ratio": 1.195
  }
}
```

### A.6 sequence_packing_text_pretrain_bs16_train_rope_seq.json

```json
{
  "benchmark": [
    {
      "sequence_packing_enabled": false,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 16,
      "raw_samples": 628,
      "packed_rows": 628,
      "active_tokens": 82092,
      "padded_tokens": 123776,
      "token_utilization": 0.66323,
      "wall_clock_seconds": 35.981722,
      "avg_step_ms": 899.543,
      "active_tokens_per_sec": 2281.492,
      "padded_tokens_per_sec": 3439.969,
      "raw_samples_per_sec": 17.453,
      "peak_memory_allocated_gb": 13.923,
      "peak_memory_reserved_gb": 28.006,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    },
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 16,
      "raw_samples": 628,
      "packed_rows": 131,
      "active_tokens": 82089,
      "padded_tokens": 95680,
      "token_utilization": 0.857954,
      "wall_clock_seconds": 30.494985,
      "avg_step_ms": 762.375,
      "active_tokens_per_sec": 2691.885,
      "padded_tokens_per_sec": 3137.565,
      "raw_samples_per_sec": 20.594,
      "peak_memory_allocated_gb": 8.846,
      "peak_memory_reserved_gb": 26.461,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ],
  "comparison": {
    "active_tokens_per_sec_speedup": 1.1799,
    "token_utilization_gain": 0.1947,
    "wall_clock_ratio": 1.1799
  }
}
```

### A.7 sequence_packing_text_pretrain_bs32_runtime_split_on.json

```json
{
  "config": {
    "batch_size": 32,
    "warmup_steps": 5,
    "measured_steps": 40,
    "packing_mode": "on",
    "train_rope_cache_max_sequence_length": 768,
    "inference_rope_cache_max_sequence_length": 32768
  },
  "benchmark": [
    {
      "sequence_packing_enabled": true,
      "warmup_steps": 5,
      "measured_steps": 40,
      "batch_size": 32,
      "raw_samples": 1256,
      "packed_rows": 254,
      "active_tokens": 164477,
      "padded_tokens": 189680,
      "token_utilization": 0.867129,
      "wall_clock_seconds": 72.861788,
      "avg_step_ms": 1821.545,
      "active_tokens_per_sec": 2257.384,
      "padded_tokens_per_sec": 2603.285,
      "raw_samples_per_sec": 17.238,
      "peak_memory_allocated_gb": 14.252,
      "peak_memory_reserved_gb": 27.916,
      "device_name": "NVIDIA GeForce RTX 5060 Ti",
      "manifest_path": "data\\manifests\\text_pretrain.json"
    }
  ]
}
```
