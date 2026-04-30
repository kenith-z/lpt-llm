# GLM5.1 及 DS4Tokenizer 基准对比实验附录

## A. 素材目录

本实验的独立素材目录为：

```text
help/GLM5.1及DS4Tokenizer基准对比实验素材/
```

该目录保存实验报告所依据的源报告、数据样本、manifest、实验输出、脚本快照、tokenizer 配置和 checkpoint 元数据摘要。目录中未复制 `.pth` 权重文件；原因是实验报告用于独立审阅而非复现训练或推理，且权重文件体积较大。对应 checkpoint 的路径、大小、mtime、SHA256 与关键字段已记录在素材目录的 `checkpoint_metadata/text_pretrain_checkpoint_summary.json`。

## B. 素材文件清单与用途

| 素材目录文件 | 来源 | 用途 |
|---|---|---|
| `source_reports/20260429Tokenizer切换基准实验报告.md` | `help/20260429Tokenizer切换基准实验报告.md` | 提供 1-11 教材 token 统计基线。 |
| `source_reports/20260501Tokenizer切换同源1-11基准对比实验报告.md` | `help/20260501Tokenizer切换同源1-11基准对比实验报告.md` | 提供同源 1-11 sequence packing 初步对比。 |
| `data/textbooks_1_11/*.jsonl` | `data/structured/*.jsonl` | 本实验使用的 11 个结构化教材文件。 |
| `manifests/ds_textbooks_1_11_manifest.json` | `.tmp_ds_benchmarks/textbooks_1_11_manifest.json` | DS 侧 1-11 教材输入清单。 |
| `manifests/glm_textbooks_1_11_manifest.json` | `.tmp_moe_llm_GLM/.tmp_glm_benchmarks/textbooks_1_11_manifest.json` | GLM5.1 侧 1-11 教材输入清单。 |
| `benchmark_outputs/sequence_packing/sequence_packing_ds_textbooks_1_11_bs4_trainlen768.json` | `.tmp_ds_benchmarks/...` | DS 侧 768 窗口 sequence packing 结果。 |
| `benchmark_outputs/sequence_packing/sequence_packing_glm_textbooks_1_11_bs4_trainlen768.json` | `.tmp_moe_llm_GLM/.tmp_glm_benchmarks/...` | GLM5.1 侧 768 窗口 sequence packing 结果。 |
| `benchmark_outputs/sequence_packing/sequence_packing_ds_textbooks_1_11_bs1_trainlen7680_off.json` | `.tmp_ds_benchmarks/...` | DS 侧 7680 长窗口 `bs1` 可运行基线。 |
| `benchmark_outputs/sequence_packing/sequence_packing_glm_textbooks_1_11_bs1_trainlen7680_off.json` | `.tmp_moe_llm_GLM/.tmp_glm_benchmarks/...` | GLM5.1 侧 7680 长窗口 `bs1` 可运行基线。 |
| `benchmark_outputs/longrope2_sweep/ds_text_pretrain_longrope2_factor_sweep.json/.md` | `.tmp_ds_benchmarks/longrope2_factor_sweep_ds_textbooks_1_11_text_pretrain_smoke/` | DS `text_pretrain` LongRoPE2 sweep 结果。 |
| `benchmark_outputs/longrope2_sweep/glm_text_pretrain_longrope2_factor_sweep.json/.md` | `.tmp_moe_llm_GLM/.tmp_glm_benchmarks/longrope2_factor_sweep_glm_textbooks_1_11_text_pretrain_smoke/` | GLM5.1 `text_pretrain` LongRoPE2 sweep 结果。 |
| `tokenizer_metadata/ds_tokenizer/tokenizer_config.json` | `lpt_model/ds_tokenizer/tokenizer_config.json` | DS tokenizer 配置快照。 |
| `tokenizer_metadata/glm_tokenizer/tokenizer_config.json` | `.tmp_moe_llm_GLM/lpt_model/glm_tokenizer/tokenizer_config.json` | GLM5.1 tokenizer 配置快照。 |
| `scripts/current_ds/*.py` | 当前 DS 分支相关脚本 | 保存 DS 侧 benchmark、sweep 与 tokenizer 对比脚本快照。 |
| `scripts/glm5_1/*.py` | `.tmp_moe_llm_GLM` 旧分支相关脚本 | 保存 GLM5.1 侧 benchmark 与 sweep 脚本快照。 |
| `checkpoint_metadata/text_pretrain_checkpoint_summary.json` | 由两侧 checkpoint 提取 | 保存 checkpoint schema、训练阶段、tokenizer 元数据、LongRoPE2 配置摘要和 SHA256。 |
| `final_documents/GLM5.1及DS4Tokenizer基准对比实验报告.md` | `help/GLM5.1及DS4Tokenizer基准对比实验报告.md` | 最终实验报告副本。 |
| `final_documents/GLM5.1及DS4Tokenizer基准对比实验附录.md` | `help/GLM5.1及DS4Tokenizer基准对比实验附录.md` | 本附录副本。 |
| `material_file_manifest.json` | 素材目录自动生成 | 保存素材目录内文件相对路径、大小和时间戳。 |

## C. Checkpoint 状态判定

GLM5.1 旧版本目录中已存在以下 checkpoint：

```text
.tmp_moe_llm_GLM/artifacts/lpt_native_v1/text_pretrain/checkpoints/latest.pth
```

该 checkpoint 的关键字段：

| 字段 | 值 |
|---|---|
| `training_stage` | `text_pretrain` |
| `training_mode` | `full` |
| `checkpoint_schema_version` | `1` |
| `model_config_schema_version` | `1` |
| `tokenizer_category` | `lpt_model\glm_tokenizer` |
| `chat_template_version` | `lpt-native-v1` |
| `longrope2_factor_max_sequence_length` | `9120` |

DS 当前分支存在同阶段 checkpoint：

```text
artifacts/lpt_ds_v1/text_pretrain/checkpoints/latest.pth
```

该 checkpoint 的关键字段：

| 字段 | 值 |
|---|---|
| `training_stage` | `text_pretrain` |
| `training_mode` | `full` |
| `checkpoint_schema_version` | `1` |
| `model_config_schema_version` | `1` |
| `tokenizer_category` | `lpt_model\ds_tokenizer` |
| `chat_template_version` | `lpt-ds-v1` |
| `longrope2_factor_max_sequence_length` | `8771` |

结论：GLM5.1 与 DS 均具备 `text_pretrain` 阶段 checkpoint，可用于同阶段 LongRoPE2 factor sweep 对比。

## D. 执行命令

### D.1 DS sequence packing：`bs4, train_len=768`

```powershell
.\.venv\Scripts\python.exe tools\benchmark_sequence_packing.py `
  --manifest .tmp_ds_benchmarks/textbooks_1_11_manifest.json `
  --manifest-kind text `
  --batch-size 4 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode both `
  --train-max-sequence-length 768 `
  --output-json .tmp_ds_benchmarks/sequence_packing_ds_textbooks_1_11_bs4_trainlen768.json
```

### D.2 GLM5.1 sequence packing：`bs4, train_len=768`

```powershell
cd .tmp_moe_llm_GLM
..\.venv\Scripts\python.exe tools\benchmark_sequence_packing.py `
  --manifest .tmp_glm_benchmarks/textbooks_1_11_manifest.json `
  --manifest-kind text `
  --batch-size 4 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode both `
  --train-max-sequence-length 768 `
  --output-json .tmp_glm_benchmarks/sequence_packing_glm_textbooks_1_11_bs4_trainlen768.json
```

### D.3 DS 长窗口可运行基线：`bs1, train_len=7680`

```powershell
.\.venv\Scripts\python.exe tools\benchmark_sequence_packing.py `
  --manifest .tmp_ds_benchmarks/textbooks_1_11_manifest.json `
  --manifest-kind text `
  --batch-size 1 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode off `
  --train-max-sequence-length 7680 `
  --output-json .tmp_ds_benchmarks/sequence_packing_ds_textbooks_1_11_bs1_trainlen7680_off.json
```

### D.4 GLM5.1 长窗口可运行基线：`bs1, train_len=7680`

```powershell
cd .tmp_moe_llm_GLM
..\.venv\Scripts\python.exe tools\benchmark_sequence_packing.py `
  --manifest .tmp_glm_benchmarks/textbooks_1_11_manifest.json `
  --manifest-kind text `
  --batch-size 1 `
  --warmup-steps 5 `
  --measured-steps 40 `
  --packing-mode off `
  --train-max-sequence-length 7680 `
  --output-json .tmp_glm_benchmarks/sequence_packing_glm_textbooks_1_11_bs1_trainlen7680_off.json
```

### D.5 DS LongRoPE2 factor sweep

```powershell
.\.venv\Scripts\python.exe tools\evaluate_longrope2_factor_sweep.py `
  --model text_pretrain `
  --text-manifest .tmp_ds_benchmarks/textbooks_1_11_manifest.json `
  --needle-lengths 128 `
  --needle-depths 0.5 `
  --retrieval-lengths 128 `
  --ppl-lengths 128 `
  --ppl-max-windows 1 `
  --max-generation-tokens 4 `
  --output-dir .tmp_ds_benchmarks/longrope2_factor_sweep_ds_textbooks_1_11_text_pretrain_smoke `
  --output-format both
```

### D.6 GLM5.1 LongRoPE2 factor sweep

```powershell
cd .tmp_moe_llm_GLM
..\.venv\Scripts\python.exe tools\evaluate_longrope2_factor_sweep.py `
  --model text_pretrain `
  --text-manifest .tmp_glm_benchmarks/textbooks_1_11_manifest.json `
  --needle-lengths 128 `
  --needle-depths 0.5 `
  --retrieval-lengths 128 `
  --ppl-lengths 128 `
  --ppl-max-windows 1 `
  --max-generation-tokens 4 `
  --output-dir .tmp_glm_benchmarks/longrope2_factor_sweep_glm_textbooks_1_11_text_pretrain_smoke `
  --output-format both
```

## E. OOM 边界记录

| tokenizer | 命令配置 | packing | 结果 | 记录 |
|---|---|---|---|---|
| DS4Tokenizer | `batch_size=4, train_max_sequence_length=7680` | off | OOM | 额外申请 `6.10 GiB` 失败。 |
| DS4Tokenizer | `batch_size=4, train_max_sequence_length=7680` | on | OOM | 额外申请 `6.10 GiB` 失败。 |
| GLM5.1 | `batch_size=4, train_max_sequence_length=7680` | off | OOM | 额外申请 `15.28 GiB` 失败。 |
| GLM5.1 | `batch_size=4, train_max_sequence_length=7680` | on | OOM | 额外申请 `15.28 GiB` 失败。 |

OOM 发生位置位于语言模型损失计算或反向传播阶段，主要与长序列下 logits / cross entropy 张量规模相关。

## F. 附加说明

1. GLM5.1 旧版 `tools/evaluate_longrope2_factor_sweep.py` 的 `--uniform-factor` 参数在 append 默认值为 tuple 时存在解析缺陷。本实验为保证 GLM5.1 与 DS 命令一致，未使用额外 `uniform_8x` 候选，仅评估 `current` 与 `bootstrap`。
2. 当前 LongRoPE2 sweep 为 smoke 级设置，窗口长度为 128，生成长度为 4，不代表完整长上下文能力评测。
3. `help/20260430Tokenizer切换后基准对比实验报告.md` 是基于 `paper_dev` 的阶段性报告；本实验报告以 1-11 号专升本教材同源材料为主。
