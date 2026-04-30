# GLM5.1 及 DS4Tokenizer 基准对比实验报告

## 摘要

本实验比较 GLM5.1 tokenizer 与 DS4Tokenizer（本项目 `ds_tokenizer`）在同一批 1-11 号专升本教材语料上的分词规模、训练吞吐、显存占用与 LongRoPE2 候选因子评测表现。实验使用 416 条结构化 text JSONL 样本作为统一材料，并在同一张 `NVIDIA GeForce RTX 5060 Ti` 上执行 sequence packing 训练基准与 LongRoPE2 factor sweep smoke 评测。

结果显示，DS4Tokenizer 的总 token 数较 GLM5.1 降低 5.55%，超过 7680 token 的样本数量由 10 条降至 5 条。在 `batch_size=4, train_max_sequence_length=768` 的训练基准中，DS4Tokenizer 相对 GLM5.1 的 active tokens/s 提升约 9.3%-10.2%，峰值 allocated 显存降低约 1.34 GB。在 `train_max_sequence_length=7680` 的长窗口基准中，两种 tokenizer 在 `batch_size=4` 下均发生 OOM；在 `batch_size=1` 下均可运行，DS4Tokenizer 的 wall-clock 更短且峰值 allocated 显存更低。LongRoPE2 factor sweep 在 `text_pretrain` checkpoint 上已完成同阶段对比，DS4Tokenizer 对应 checkpoint 在 smoke 设置下获得更低的 PPL(128)，但生成型 needle 与 retrieval 精确匹配率均为 0.0，说明该结果只能作为链路与相对基线参考，不能作为充分的长上下文能力结论。

## 1. 实验目的

本实验旨在回答以下问题：

1. 在同一语料上，GLM5.1 tokenizer 与 DS4Tokenizer 的 token 规模差异是否显著。
2. tokenizer 切换是否改变 sequence packing 训练基准中的吞吐与显存表现。
3. 在长窗口训练边界下，较小词表的 DS4Tokenizer 是否带来可观察的显存收益。
4. 在已有 GLM5.1 与 DS 同阶段 `text_pretrain` checkpoint 条件下，LongRoPE2 factor sweep 是否可以完成可比评测。

## 2. 实验材料与环境

### 2.1 数据材料

实验语料为 1-11 号专升本教材结构化 text JSONL，共 416 条样本。各文件均来自 `data/structured/`，并复制到独立素材目录。

此前 token 计数实验结果如下：

| 指标 | GLM5.1 | DS4Tokenizer | 变化 |
|---|---:|---:|---:|
| 样本数 | 416 | 416 | 0 |
| 总 token 数 | 2,357,420 | 2,226,598 | -130,822 |
| 总 token 数变化率 | - | - | -5.55% |
| 超过 7680 token 的样本数 | 10 | 5 | -5 |
| 最大正文 token 数 | 9,120 | 8,770 | -350 |

### 2.2 Tokenizer 元数据

| 项目 | GLM5.1 | DS4Tokenizer |
|---|---:|---:|
| tokenizer 路径 | `.tmp_moe_llm_GLM/lpt_model/glm_tokenizer` | `lpt_model/ds_tokenizer` |
| vocab size | 154856 | 129280 |
| tokenizer_config SHA256 | `BE52009AC92A886B146B51D3F4E17F45CF449A3A596B3CBA2F8D81A93589B191` | `789E16A9396DC44A7D0EAF8627DBFDFD9F583F49EC82FF543ECC6A1A11DC8049` |
| BOS | `None` | `<｜begin▁of▁sentence｜>` / id `0` |
| EOS | `<|endoftext|>` / id `154820` | `<｜end▁of▁sentence｜>` / id `1` |
| PAD | `<|endoftext|>` / id `154820` | `<｜▁pad▁｜>` / id `2` |
| chat template version | `lpt-native-v1` | `lpt-ds-v1` |

GLM5.1 使用同一 token 作为 EOS 与 PAD。DS4Tokenizer 对 BOS、EOS、PAD 分别使用独立特殊 token 与独立 id。该差异要求训练、评测、checkpoint 与报告均显式绑定 tokenizer 元数据，否则不同 tokenizer 的实验结果不可审计。

### 2.3 运行环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 10 / PowerShell 7 |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| CUDA 可见设备数 | 1 |
| 可见显存 | 约 15.93 GiB |
| Python 环境 | 项目 `.venv` |
| 评测模型阶段 | `text_pretrain` |

## 3. 方法

### 3.1 Token 计数

使用同一 JSONL 输入，分别加载 GLM5.1 tokenizer 与 DS4Tokenizer，对每条样本的正文进行编码并统计 token 数。统计结果用于评估 tokenizer 对上下文长度预算、截断风险和训练窗口设置的影响。

### 3.2 Sequence Packing 训练基准

训练基准使用 `tools/benchmark_sequence_packing.py`。本实验保留两个代表性设置：

1. `batch_size=4, train_max_sequence_length=768`：用于比较常规训练窗口下的吞吐、显存与 packing 效果。
2. `batch_size=1, train_max_sequence_length=7680`：用于比较长窗口可运行边界。

此外，对 `batch_size=4, train_max_sequence_length=7680` 进行容量边界验证，记录两种 tokenizer 在 packing 开关下的 OOM 状态。

### 3.3 LongRoPE2 Factor Sweep

用户已在 `.tmp_moe_llm_GLM` 中运行 `main-pretrain.py` 并生成 GLM5.1 `text_pretrain` checkpoint。检查结果表明，GLM5.1 与 DS 均具备 `text_pretrain` 阶段 checkpoint，因此可以进行同阶段 LongRoPE2 factor sweep 对比。

评测设置如下：

| 参数 | 值 |
|---|---|
| model | `text_pretrain` |
| needle_lengths | 128 |
| needle_depths | 0.5 |
| retrieval_lengths | 128 |
| ppl_lengths | 128 |
| ppl_max_windows | 1 |
| max_generation_tokens | 4 |
| candidates | `current`, `bootstrap` |

该设置为 smoke 级评测，主要用于验证候选因子链路、checkpoint 可加载性和短窗口相对指标，不用于判定完整长上下文能力。

## 4. 实验结果

### 4.1 Sequence Packing：`batch_size=4, train_max_sequence_length=768`

| tokenizer | packing | active tokens/s | raw samples/s | token utilization | wall-clock | peak allocated | peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| GLM5.1 | off | 4026.443 | 5.265 | 0.995760 | 30.388858s | 8.849 GB | 11.896 GB |
| GLM5.1 | on | 4121.049 | 5.389 | 0.995760 | 29.691224s | 8.845 GB | 10.217 GB |
| DS4Tokenizer | off | 4399.553 | 5.753 | 0.995711 | 27.810324s | 7.511 GB | 10.066 GB |
| DS4Tokenizer | on | 4541.998 | 5.940 | 0.995711 | 26.938143s | 7.508 GB | 8.703 GB |

在该配置下，DS4Tokenizer 相对 GLM5.1 的 active tokens/s 提升如下：

| packing | DS4Tokenizer / GLM5.1 |
|---|---:|
| off | 1.0927x |
| on | 1.1021x |

峰值 allocated 显存方面，DS4Tokenizer 在 packing off 与 packing on 下均较 GLM5.1 低约 1.34 GB。由于 1-11 教材在 768 窗口下大量样本被截断到接近定长，两种 tokenizer 的 token utilization 均接近 1.0，因此 packing 的主要效果表现为轻微吞吐提升和 peak reserved 下降，而不是显著提升填充利用率。

### 4.2 长窗口容量边界：`batch_size=4, train_max_sequence_length=7680`

| tokenizer | packing | 结果 | 关键 OOM 信息 |
|---|---|---|---|
| GLM5.1 | off | OOM | 额外申请 15.28 GiB 失败 |
| GLM5.1 | on | OOM | 额外申请 15.28 GiB 失败 |
| DS4Tokenizer | off | OOM | 额外申请 6.10 GiB 失败 |
| DS4Tokenizer | on | OOM | 额外申请 6.10 GiB 失败 |

该结果表明，在长窗口训练中，sequence packing 不能消除最终语言模型 logits 与 cross entropy 的主要显存项。DS4Tokenizer 由于词表更小，额外申请量显著低于 GLM5.1，但在 16 GB 级 GPU 上仍不足以支撑 `batch_size=4, train_max_sequence_length=7680`。

### 4.3 长窗口可运行基线：`batch_size=1, train_max_sequence_length=7680`

| tokenizer | active tokens/s | raw samples/s | active tokens | wall-clock | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| GLM5.1 | 717.062 | 0.130 | 220,056 | 306.885393s | 17.118 GB | 23.631 GB |
| DS4Tokenizer | 728.487 | 0.141 | 206,949 | 284.080749s | 14.265 GB | 29.766 GB |

在 `batch_size=1` 下，两种 tokenizer 均可完成长窗口训练基准。DS4Tokenizer 相对 GLM5.1 的 active tokens/s 提升约 1.6%，raw samples/s 提升约 8.5%，wall-clock 缩短约 7.4%，峰值 allocated 显存降低约 2.85 GB。考虑到两种 tokenizer 的 token 切分数量不同，长窗口基线应同时考察 raw samples/s、wall-clock 与显存指标。

### 4.4 LongRoPE2 Factor Sweep

本轮新增 GLM5.1 `text_pretrain` checkpoint 后，先前“无法严格复跑 GLM5.1 sweep”的限制已经解除。两侧 checkpoint 元数据如下：

| 项目 | GLM5.1 | DS4Tokenizer |
|---|---|---|
| checkpoint | `.tmp_moe_llm_GLM/artifacts/lpt_native_v1/text_pretrain/checkpoints/latest.pth` | `artifacts/lpt_ds_v1/text_pretrain/checkpoints/latest.pth` |
| training_stage | `text_pretrain` | `text_pretrain` |
| training_mode | `full` | `full` |
| checkpoint_schema_version | 1 | 1 |
| model_config_schema_version | 1 | 1 |
| longrope2_factor_max_sequence_length | 9120 | 8771 |

Sweep 结果如下：

| tokenizer | candidate | factor | needle exact | retrieval exact | PPL(128) | latency |
|---|---|---:|---:|---:|---:|---:|
| GLM5.1 | current | 16x | 0.0 | 0.0 | 1.5481e23 | 1.872555s |
| GLM5.1 | bootstrap | 16x | 0.0 | 0.0 | 1.5481e23 | 1.485321s |
| DS4Tokenizer | current | 16x | 0.0 | 0.0 | 1.5034e20 | 1.790895s |
| DS4Tokenizer | bootstrap | 16x | 0.0 | 0.0 | 1.5034e20 | 1.319567s |

两侧 current 与 bootstrap 均为 16x，因此同一 tokenizer 内两个 candidate 的 PPL 相同符合预期。DS4Tokenizer 对应 checkpoint 的 PPL(128) 明显低于 GLM5.1 对应 checkpoint，但由于本评测为 smoke 设置，且两个 checkpoint 来自不同 tokenizer 与不同训练产物，该结果应解释为“同阶段短窗口评测下的当前基线差异”，不应外推为充分的模型质量结论。

## 5. 讨论

### 5.1 Tokenizer 切换对训练效率的影响

DS4Tokenizer 的词表规模小于 GLM5.1，且在 1-11 教材上产生更少 token。这一变化同时影响两个方面：一是同一文本的序列长度压力降低，二是最终 logits 的词表维度下降。实验中 DS4Tokenizer 在 768 训练窗口下具有更高吞吐和更低峰值 allocated 显存，符合上述机制预期。

### 5.2 Sequence Packing 收益受样本长度分布约束

在 1-11 教材上，`train_max_sequence_length=768` 会使大量样本接近定长，因此未 packing 时的 token utilization 已接近 1.0。此时 packing 对填充利用率的提升空间很小，收益主要体现为执行层面的轻微吞吐改善与 reserved 显存下降。该结论与短样本数据集上的 packing 收益不同，说明 packing 基线必须绑定具体数据分布。

### 5.3 长窗口训练的主要瓶颈

`train_max_sequence_length=7680` 下，`batch_size=4` 对两种 tokenizer 均不可运行。OOM 发生在前向损失计算或反向传播阶段，主要与 logits / cross entropy 的大张量有关。DS4Tokenizer 的较小 vocab 能缓解该问题，但不能独立解决长窗口训练显存瓶颈。若要稳定训练 7680 长窗口，需要进一步引入更小 batch、梯度累积、分块 loss、词表并行、activation checkpointing 或分布式训练策略。

### 5.4 LongRoPE2 Sweep 的解释边界

本次 sweep 已补齐 GLM5.1 text_pretrain checkpoint 的可运行基线，并实现同阶段对比。然而评测长度仅为 128，候选因子均为 16x，生成型任务精确匹配率均为 0.0。因此该实验主要证明评测链路、checkpoint 绑定和 tokenizer 元数据追踪已经可用；后续若要评价 LongRoPE2 因子质量，应扩展到 2K、4K、8K 窗口，并增加候选因子多样性。

## 6. 结论

1. DS4Tokenizer 在 1-11 教材上相对 GLM5.1 减少 5.55% 的 token 总量，并将超过 7680 token 的样本数从 10 条降低到 5 条，降低了长上下文训练和评测中的长度压力。
2. 在 `batch_size=4, train_max_sequence_length=768` 的真实 GPU 训练基准中，DS4Tokenizer 的 active tokens/s 比 GLM5.1 高约 9.3%-10.2%，峰值 allocated 显存低约 1.34 GB。
3. 在 `batch_size=4, train_max_sequence_length=7680` 下，两种 tokenizer 均发生 OOM。DS4Tokenizer 具有更低的显存申请压力，但不足以单独支撑 16 GB GPU 上的该配置。
4. 在 `batch_size=1, train_max_sequence_length=7680` 下，两种 tokenizer 均可运行；DS4Tokenizer 在 wall-clock、raw samples/s 和峰值 allocated 显存上优于 GLM5.1。
5. 用户新增的 GLM5.1 `text_pretrain` checkpoint 已满足 LongRoPE2 sweep 的可运行条件，先前“GLM5.1 sweep 缺 checkpoint”的限制已经解除。当前同阶段 smoke sweep 中，DS4Tokenizer 对应 checkpoint 的 PPL(128) 低于 GLM5.1，但该结果仍需在更长窗口和更丰富候选因子上复核。

综合来看，DS4Tokenizer 在本项目当前语料与模型配置下具有明确的工程收益：更低 token 数、更小词表、更低训练显存和更高常规窗口吞吐。长窗口训练瓶颈仍需通过训练策略和执行层优化解决，而不能仅依赖 tokenizer 切换。
