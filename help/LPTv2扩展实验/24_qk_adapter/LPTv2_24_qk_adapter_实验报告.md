# LPT v2 第 24 项 RetNetAssist Q/K Adapter 实验报告

## 摘要

第 24 项评估 RetNetAssist 从 `q_adapter` 扩展到 `qk_adapter` 后，K 注入是否能在 sliding window 场景下带来稳定收益，并且不造成不可接受的显存、吞吐和缓存成本。

两个分支均已完成 1164 step chat SFT 训练，并完成 checkpoint validate、真实 checkpoint forward smoke、长上下文代理评测和资源报告。当前结论：`exp_24_qk_adapter` 的 eval loss 优于 `base_continued`，且 K adapter delta norm 可观测；但长上下文代理 loss 与 needle logprob 退化，router z-loss 明显升高，因此第 24 项暂不进入主干或组合实验。

## 1. 实验目的

1. 对比 Q-only RetNetAssist 与 Q/K RetNetAssist 的训练收敛和 eval loss。
2. 验证 K adapter 只作用于新生成的 key，不写入 RetNet state、不替换 Paged KV、不调制 value。
3. 量化 K adapter 带来的额外参数、显存峰值、吞吐、RetNet state bytes 与 adapter delta norm。

## 2. 实验材料与环境

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/model.pt` |
| base source manifest | `data/manifests/text_pretrain.json` |
| branch training manifest | `data/manifests/chat_sft.json` |
| eval manifest | `data/manifests/chat_sft_eval_exp24.json` |
| tokenizer | `lpt_model/ds_tokenizer` |
| 训练工作流 | chat SFT |
| 报告输出目录 | `help/LPTv2扩展实验/reports/exp24/` |

## 3. 方法

### 3.1 初始化命令

```powershell
.\.venv\Scripts\python.exe tools\init_lpt_v2_exp24_branches.py --base-checkpoint artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt --output-root artifacts\lpt_v2\experiments_exp24
```

初始化脚本会创建两个分支：

| 分支 | 作用 | 只允许变化的变量 |
|---|---|---|
| `base_continued` | 同等预算继续训练对照；由于共同基座必须是 Q-only RetNetAssist，该分支同时作为 Q-only baseline | 无 |
| `exp_24_qk_adapter` | Q/K adapter 实验分支 | `retnet_assist_mode="qk_adapter"`、`retnet_adapter_target=("q","k")`、`retnet_k_adapter_enabled=true` |

### 3.2 训练命令

统一使用 chat SFT 工作流和 `data/manifests/chat_sft.json`。沿用第 23 项省空间策略：只保留最终 `checkpoints/latest/model.pt`、`trainer_state.json`、`checkpoint_manifest.json`、`metrics.jsonl` 和 config；不保存 optimizer、scheduler、TensorBoard、best checkpoint 和额外 inference weights。

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued --init-checkpoint artifacts\lpt_v2\experiments_exp24\base_continued\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp24\base_continued --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp24.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp24_base_continued --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_24_qk_adapter --init-checkpoint artifacts\lpt_v2\experiments_exp24\exp_24_qk_adapter\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp24\exp_24_qk_adapter --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp24.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp24_qk_adapter --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

训练控制变量必须与第 23 项一致：同一个 `text_pretrain` 基座、同一个 tokenizer、同一个 `chat_sft` manifest、同一个 eval manifest、同一个 seed、batch size、learning rate、eval interval、LongRoPE2 设置和 dtype。

### 3.3 训练后评测命令

训练完成后每个分支先跑只读 forward smoke，确认 logits finite、shape 正确、RetNet state 数量正确，并记录 Q/K adapter delta norm。

```powershell
.\.venv\Scripts\python.exe tools\run_lpt_v2_forward_smoke.py --checkpoint artifacts\lpt_v2\experiments_exp24\<branch>\checkpoints\latest\model.pt --seq-len 32 --batch-size 1 --device auto --dtype auto --output-json help\LPTv2扩展实验\reports\exp24\<branch>_forward_smoke.json --output-md help\LPTv2扩展实验\reports\exp24\<branch>_forward_smoke.md
```

正式评测命令：

```powershell
.\.venv\Scripts\python.exe tools\validate_lpt_v2_checkpoint.py --checkpoint artifacts\lpt_v2\experiments_exp24\<branch>\checkpoints\latest\model.pt --map-location cpu
.\.venv\Scripts\python.exe tools\run_lpt_v2_long_context_eval.py --checkpoint artifacts\lpt_v2\experiments_exp24\<branch>\checkpoints\latest\model.pt --seq-len 2052 --attention-window-size 2048 --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp24\<branch>_long_context.json --output-md help\LPTv2扩展实验\reports\exp24\<branch>_long_context.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_resource_report.py --checkpoint artifacts\lpt_v2\experiments_exp24\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp24\<branch>_resource.json --output-md help\LPTv2扩展实验\reports\exp24\<branch>_resource.md
```

## 4. 数据项覆盖要求

| 数据项 | 来源 | 状态 |
|---|---|---|
| 训练 loss / final trainer loss | `metrics.jsonl`、`trainer_state.json` | 已完成 |
| eval loss / PPL | `metrics.jsonl` eval 行 | 已完成 |
| tokens_seen / samples_seen | `trainer_state.json`、`metrics.jsonl` | 已完成 |
| 训练吞吐、序列长度、CUDA 峰值 | `metrics.jsonl` | 已完成 |
| checkpoint schema | `validate_lpt_v2_checkpoint.py` | 已完成 |
| forward smoke | `reports/exp24/*_forward_smoke.json/md` | 已完成 |
| RetNet Q/K adapter norm | forward smoke、resource report | 已完成 |
| 长上下文代理指标 | `reports/exp24/*_long_context.json/md` | 已完成 |
| 资源指标 | `reports/exp24/*_resource.json/md` | 已完成 |

## 5. 结果记录

### 5.1 Checkpoint Validate

两个分支均通过 schema v2 校验：

```text
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
```

### 5.2 训练与 Eval

| 分支 | final trainer loss | final train loss | latest eval loss | best eval loss | best eval step | latest eval PPL | tokens_seen | avg tokens/s | max seq len | max train CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 17.224331 | 26.495504 | 26.448839 | 26.448839 | 1100 | 3.066090e11 | 435335 | 361.104 | 2328 | 16472.054 |
| `exp_24_qk_adapter` | 17.284222 | 26.532171 | 26.258096 | 26.258096 | 1100 | 2.533647e11 | 435335 | 349.346 | 2328 | 16506.062 |

`exp_24_qk_adapter` 的 best/latest eval loss 比 `base_continued` 低 `0.190743`，但训练吞吐低约 `3.26%`，训练 CUDA 峰值高约 `34.009 MiB`。

### 5.3 Forward Smoke

| 分支 | forward_ok | logits_shape | loss | PPL | mode | target | RetNet states | Q adapter norm | K adapter norm |
|---|---|---|---:|---:|---|---|---:|---:|---:|
| `base_continued` | true | `1x32x129280` | 11.769736 | 129280.054 | `q_adapter` | `q` | 24 | 4.290049 | 0.000000 |
| `exp_24_qk_adapter` | true | `1x32x129280` | 11.769736 | 129280.054 | `qk_adapter` | `q,k` | 24 | 4.551141 | 3.779367 |

Forward smoke 确认 Q/K 分支的 K adapter 路径已真实启用，且不会改变 logits shape 或 state 数量。

### 5.4 长上下文代理

| 分支 | status | long loss | PPL | needle rank | needle logprob | RetNet tokens | Paged KV window | Q adapter norm | K adapter norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | `close_or_debug` | 46.433018 | 4.851652e8 | 28100 | -25.523956 | 2048 | 2048 | 0.238505 | 0.000000 |
| `exp_24_qk_adapter` | `close_or_debug` | 47.057213 | 4.851652e8 | 28103 | -25.784645 | 2048 | 2048 | 0.286381 | 0.220732 |

长上下文代理没有给出通过信号：两个分支均为 `close_or_debug`，且 Q/K 分支 long loss 更高、needle logprob 更低、needle rank 略差。当前不能证明 K 注入在 sliding window 外信息保留上有稳定收益。

### 5.5 资源与机制指标

| 分支 | prefill tok/s | decode tok/s | first token ms | peak MiB | RetNet state bytes | RetNet summary norm | Q adapter norm | K adapter norm | router entropy | load balance loss | router z-loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 29.928 | 8.496 | 82.027 | 2766.888 | 3072 | 11.945847 | 4.330010 | 0.000000 | 0.814088 | 3.000 | 17.888150 |
| `exp_24_qk_adapter` | 26.938 | 10.023 | 109.415 | 2767.134 | 3072 | 13.136007 | 4.756325 | 4.086860 | 0.738551 | 3.000 | 21.143420 |

资源侧显存差异很小，K adapter 不增加 RetNet runtime state bytes；但 Q/K 分支 prefill 吞吐下降约 `9.99%`，首 token 延迟增加约 `27.388 ms`，router entropy 下降且 z-loss 上升明显，需要视为稳定性风险。

## 6. 结论

第 24 项结论为：`exp_24_qk_adapter` 暂不进入主干或组合实验。

证据：

1. 正向证据：eval loss 从 `26.448839` 降到 `26.258096`，K adapter norm 在 forward smoke 和资源报告中均可观测，说明机制路径有效。
2. 负向证据：长上下文代理从 `46.433018` 退化到 `47.057213`，needle logprob 从 `-25.523956` 退化到 `-25.784645`，两个分支仍为 `close_or_debug`。
3. 成本风险：训练吞吐下降约 `3.26%`，资源评测 prefill 吞吐下降约 `9.99%`，router z-loss 从 `17.888150` 升到 `21.143420`。

当前只保留 `qk_adapter` 作为归档对照。若后续要重新打开，需要补更强长上下文 eval manifest 或专门的 sliding-window 检索任务，并要求长上下文代理不退化后再考虑组合实验。

## 7. 附录

- 初始化脚本：`tools/init_lpt_v2_exp24_branches.py`
- eval manifest：`data/manifests/chat_sft_eval_exp24.json`
- 训练产物：`artifacts/lpt_v2/experiments_exp24/...`
- 报告产物：`help/LPTv2扩展实验/reports/exp24/...`
