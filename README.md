# lpt-llm

`lpt-llm` 是一个面向研究与工程验证的原生 LLM 项目。模型名为 **LPT**，英文全称 **Ling Pre-trained Transformer**，中文名 **灵预**。

当前工程已经从早期原型推进为可训练、可评测、可推理的研究型骨架：包含 Hybrid `Attention + RetNet` 主干、LongRoPE2 长上下文位置编码、三阶段训练工作流、严格 checkpoint schema、结构化数据管线、sequence packing、长上下文评测和基础单机多卡推理执行层。

## 当前状态

已具备：

- Hybrid `Attention + RetNet` 模型主干。
- `text_pretrain / chat_sft / chat_lora` 三阶段训练工作流。
- DS4Tokenizer 默认接入，运行时 tokenizer 固定为 `lpt_model/ds_tokenizer`。
- DS chat template：`lpt-ds-v1`。
- LongRoPE2 训练/推理 mixed embedding 策略。
- 严格 checkpoint schema 与可序列化 `ModelConfig`。
- 结构化 `text / chat` JSONL schema、manifest 和流式加载。
- 严格样本边界的 sequence packing。
- `InferenceSession / CacheManager` 推理缓存接口。
- 长上下文评测与 LongRoPE2 factor sweep。
- `CUDA_VISIBLE_DEVICES` 感知、逻辑 GPU 发现、推理侧层级 `model_parallel`。

仍未完成：

- 训练侧 FSDP2 / 分布式 optimizer 与 checkpoint。
- 推理侧算子级 Tensor Parallelism。
- 服务化推理入口与 OpenAI / Anthropic 兼容 API。
- Paged KV Cache、continuous batching、FlashAttention2 等生产推理优化。
- 完整实验治理闭环。

## 模型概览

默认 `ModelConfig` 位于 `lpt_config/model_config.py`：

| 配置项 | 默认值 |
|---|---:|
| `num_layers` | 16（8 组 RetNet/Attention，即 2n） |
| `num_heads` | 8 |
| `num_kv_heads` | 2 |
| `cla_share_every_n_layers` | 2 |
| `head_dim` | 64 |
| `hidden_size` | 512 |
| block 类型 | RetNet 与 Attention 交替 |
| `original_max_len` | 2048 |
| `longrope2_target_length` | 跟随推理最大长度配置 |
| LongRoPE2 训练/推理 embedding | `mixed` |

### 当前模型结构

LPT 当前是 decoder-only 语言模型，主干由 `RetNet` 与 `Attention` 两类 sequence mixer 交替组成：

```text
input_ids
  -> token_embedding
  -> 2n x TransformerBlock
       -> RMSNorm
       -> sequence mixer: RetNet / Attention 交替
       -> residual
       -> RMSNorm
       -> SwiGLU FFN
       -> residual
  -> final RMSNorm
  -> tied lm_head
  -> logits
```

默认层序为 `RetNet, Attention` 交替重复 n 组；当前默认 n=8，共 16 层。每个 `TransformerBlock` 只负责统一的归一化、残差和 FFN 包装，序列混合能力由 `layer_block_types` 指定的 mixer 决定。

- `Attention` 层使用 GQA：`num_heads=8`、`num_kv_heads=2`、`head_dim=64`，每 4 个 query head 共享 1 组 KV head；优先使用 PyTorch SDPA 原生 `enable_gqa`，不支持时回退到手工扩展 KV。
- Attention KV Cache 同时使用 CLA（跨层 Attention 状态复用）：默认 `cla_share_every_n_layers=2`，每 2 个 Attention 层共享一个 KV 状态槽位；共享组内首个 Attention 层写入 KV，后续 Attention 层以只读方式复用该 KV。RetNet 层不参与 CLA，仍独立保存 retention state。
- `RetNet` 层使用多尺度 retention：默认 `retnet_value_factor=1`、`retnet_gate_fn=swish`、`retnet_chunk_size=256`，位置部分使用 XPOS 相对位置编码，支持并行、chunkwise prefill 和递归增量解码三种表示。
- FFN 使用 `SwiGLU`，中间维度按 `8 * hidden_size / 3` 计算后对齐到 256；输入 embedding 与输出 `lm_head` 权重共享。
- 层状态统一抽象为 `LayerState`：Attention 层保存经过 GQA 和 CLA 压缩后的 KV 状态，RetNet 层保存 retention state，供 prefill、增量推理和 `InferenceSession / CacheManager` 复用。

**注：模型当前结构并非最终结构。**

## Tokenizer

当前分支默认 tokenizer：DeepSeek-V4-Pro

```text
lpt_model/ds_tokenizer
```

关键特殊 token：

| token | 字符串 | id |
|---|---|---:|
| BOS | `<｜begin▁of▁sentence｜>` | 0 |
| EOS | `<｜end▁of▁sentence｜>` | 1 |
| PAD | `<｜▁pad▁｜>` | 2 |

旧 GLM5.1 tokenizer 仅作为对比实验和历史基线保留，不再作为当前训练/推理默认路径。跨 tokenizer 比较必须同时绑定 tokenizer 路径、vocab size、特殊 token、chat template、checkpoint 和实验输出。

相关报告：

- `help/GLM5.1及DS4的Tokenizer基准对比实验/GLM5.1及DS4Tokenizer基准对比实验报告.md`

## 目录结构

```text
.
├── main.py                         # 统一推理入口
├── main-pretrain.py                # text pretrain 训练入口
├── main-sft.py                     # chat SFT 训练入口
├── main-LoRA.py                    # chat LoRA 训练入口
├── lpt_config/                     # 全局配置、训练 recipe、ModelConfig
├── lpt_model/                      # LPT 模型、LongRoPE2、tokenizer 资源
├── lpt_protocol/                   # chat template 与训练片段渲染
├── lpt_data/                       # JSONL schema、manifest、流式数据加载
├── lpt_training/                   # batch 构造、sequence packing、训练循环、checkpoint
├── lpt_workflows/                  # 三阶段训练和推理加载流程
├── lpt_inference/                  # 推理、InferenceSession、可视化
├── lpt_lora/                       # LoRA adapter 与工作流
├── lpt_runtime/                    # 执行配置、device map、model parallel
├── lpt_evaluation/                 # 长上下文评测和 LongRoPE2 sweep
├── tools/                          # 数据转换、评测、benchmark 工具
├── tests/                          # 单元测试与回归测试
├── data/                           # 本地数据，默认不纳入版本控制
├── artifacts/                      # 训练产物，默认不纳入版本控制
└── help/                           # 项目任务、命令、实验报告
```

## 环境准备

项目基础依赖包括：

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install torch transformers tqdm matplotlib pillow tensorboard
```

如需使用 CSV / Parquet 数据转换工具，再按需安装：

```powershell
python -m pip install pandas pyarrow
```

Windows 下建议使用 PowerShell 7，并以 UTF-8 输出运行命令。

## 数据格式

训练数据通过 `data/manifests/*.json` 进入数据管线，manifest 指向 `data/structured/*.jsonl`。

`text` 样本示例：

```json
{"type": "text", "text": "待训练文本", "source": "example"}
```

`chat` 样本示例：

```json
{
  "type": "chat",
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，我是灵预。"}
  ],
  "source": "example"
}
```

可用转换工具：

```powershell
python tools/convert_txt_text_dataset.py --input data/raw/sample.txt --output data/structured/sample.text.jsonl --source-name sample
python tools/convert_raw_text_jsonl.py --input data/raw/sample.jsonl --output data/structured/sample.text.jsonl --source-name sample
python tools/convert_instruction_chat_jsonl.py --input data/raw/chat.jsonl --sft-output data/structured/chat.sft.jsonl --lora-output data/structured/chat.lora.jsonl --source-name chat
```

## 训练

### Text Pretrain

```powershell
python main-pretrain.py --manifest data/manifests/text_pretrain.json
```

### Chat SFT

```powershell
python main-sft.py --manifest data/manifests/chat_sft.json
```

如果存在 `text_pretrain` checkpoint，SFT 会优先从该 checkpoint 初始化；否则从随机初始化模型开始。

### Chat LoRA

```powershell
python main-LoRA.py --manifest data/manifests/chat_lora.json --base-source text_pretrain
```

`--base-source` 支持：

- `text_pretrain`
- `chat_sft`

### LongRoPE2 训练参数示例

```powershell
python main-sft.py `
  --manifest data/manifests/chat_sft.json `
  --train-max-sequence-length 7680 `
  --longrope2-target-window 32768 `
  --longrope2-window-lengths 2048,7680 `
  --longrope2-window-weights 0.5,0.5
```

已有完整续训状态时，模型结构以 checkpoint 为准，命令行 LongRoPE2 结构参数不会静默覆盖 checkpoint。

## Artifacts

当前 DS 分支训练产物根目录：

```text
artifacts/lpt_ds_v1/
```

典型结构：

```text
artifacts/lpt_ds_v1/
├── text_pretrain/
│   ├── checkpoints/latest.pth
│   ├── weights/model_weights.pth
│   ├── config/model_config.json
│   └── logs/
├── chat_sft/
└── chat_lora/from_text_pretrain/
```

checkpoint 严格要求包含：

- `checkpoint_schema_version`
- `model_config_schema_version`
- `model_config`
- `model_architecture_metadata`
- `training_stage`
- `source_manifest`
- tokenizer 与 chat template 元数据

旧 schema checkpoint 不会被兼容加载。

## 推理

统一入口：

```powershell
python main.py --model chat_sft --execution-mode single
```

可选模型：

- `text_pretrain` / `text_base`
- `chat_sft`
- `chat_lora` / `lora`

LoRA 推理：

```powershell
python main.py --model chat_lora --lora-base-source text_pretrain --execution-mode single
```

单机多卡推理：

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
python main.py --model chat_sft --execution-mode model_parallel --device-map auto
```

注意：`CUDA_VISIBLE_DEVICES=2,3` 后，程序内部看到的是逻辑设备 `cuda:0 / cuda:1`，手工 `device_map` 中应写逻辑设备号，不应写物理设备号。

## 评测与基准

### 长上下文评测

```powershell
python tools/evaluate_long_context.py `
  --model chat_sft `
  --text-manifest data/manifests/text_pretrain.json `
  --cache-strategy session_rebuild `
  --output-format both
```

评测覆盖：

- needle-in-a-haystack
- 长文本 PPL
- 长上下文 QA / retrieval

### LongRoPE2 候选因子 sweep

```powershell
python tools/evaluate_longrope2_factor_sweep.py `
  --model chat_sft `
  --uniform-factor uniform_8x=8 `
  --output-format both
```

sweep 会临时覆盖内存中的 `ModelConfig.longrope2_long_factors`，不会写回 checkpoint。

### Sequence Packing GPU 基准

```powershell
python tools/benchmark_sequence_packing.py `
  --manifest data/manifests/text_pretrain.json `
  --manifest-kind text `
  --packing-mode both
```

### Tokenizer JSONL 对比

```powershell
python tests/test_tokenizer_jsonl_token_counts.py data/structured/sample.text.jsonl
```

该脚本用于诊断同一 JSONL 在 GLM5.1 tokenizer 与 DS4Tokenizer 下的 token 数差异。

## 测试

全量测试：

```powershell
python -m unittest discover -s tests
```

常用拆分：

```powershell
python -m unittest tests.test_runtime_execution tests.test_inference tests.test_model_behavior tests.test_model_config tests.test_checkpoint_schema
python -m unittest tests.test_training_recipe
python -m unittest tests.test_data_pipeline
python -m unittest tests.test_long_context_eval tests.test_longrope2_factor_sweep
```

基础编译检查：

```powershell
python -m compileall main.py main-pretrain.py main-sft.py main-LoRA.py lpt_config lpt_runtime lpt_data lpt_model lpt_training lpt_inference lpt_lora lpt_protocol lpt_evaluation lpt_workflows tools tests
```

Windows 本机如果在同一 Python 进程内混跑训练测试与 `pyarrow` 数据管线测试时遇到原生扩展 access violation，可按模块拆开运行。

## 重要文档

- `help/命令.md`：正式命令维护入口。
- `help/GLM5.1及DS4的Tokenizer基准对比实验/GLM5.1及DS4Tokenizer基准对比实验报告.md`：标准化 tokenizer 基准对比实验报告。

## 开发约束

- 默认 Python 3，UTF-8 无 BOM。
- 路径处理优先使用 `pathlib.Path`。
- 不为旧 checkpoint schema 保留无意义兼容分支。
- 训练产物、数据和 help 材料通常是本地文件，不默认进入版本控制。
- 新增 CLI 参数时，同步更新 `help/命令.md`。
- 新增实验或 benchmark 时，必须绑定 tokenizer、manifest、checkpoint、模型配置和输出报告。

## 项目定位

`lpt-llm` 当前是一个以 LPT（Ling Pre-trained Transformer，灵预）为核心的 LLM 研究工程。它适合用于验证模型结构、长上下文策略、训练 recipe、tokenizer 切换、评测流程和执行层设计；它还不是完整生产级推理服务或分布式训练系统。
