# LPTv2 第 27 项 RetNetContextAdapter 实验报告

## 摘要

第 27 项已完成两分支 1164 step chat SFT 训练和训练后评测。修订后结论：`RetNetContextAdapter` 设为长上下文主候选，进入组合实验；若组合实验相对无 ContextAdapter 对照不劣化，则直接进入主干。

结论修订说明：初始自动结论为“暂不进入默认主干或默认组合实验，仅保留为长上下文专项候选”。人工复核后认为长上下文收益是必要条件，本轮 eval manifest 样本量较小，`0.283578` 的 eval loss 上升不足以抵消明确的长上下文改善，因此改为主候选进入组合实验。

## 1. 实验目的

1. 对比无 ContextAdapter 的 Q-only RetNetAssist 与启用 RetNetContextAdapter 后的 chat SFT 收敛、eval loss 和 PPL。
2. 验证 ContextAdapter 只复用既有 `SharedRetNetAssist` 的 `summary_sequence`，不新增状态池、不写入 Paged KV、不替换 K/V cache。
3. 量化 ContextAdapter 的额外参数、训练/推理显存、吞吐、`context_adapter_delta_norm` 和 `alpha_context`。

## 2. 实验材料与环境

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/model.pt` |
| base source manifest | `data/manifests/text_pretrain.json` |
| branch training manifest | `data/manifests/chat_sft.json` |
| eval manifest | `data/manifests/chat_sft_eval_exp27.json` |
| tokenizer | `lpt_model/ds_tokenizer` |
| 训练工作流 | chat SFT |
| ContextAdapter 注入位置 | Attention 输出投影之后，随 block 残差进入 `x = x + attn_out` |
| 报告输出目录 | `help/LPTv2扩展实验/reports/exp27/` |

## 3. 方法

### 3.1 初始化命令

```powershell
.\.venv\Scripts\python.exe tools\init_lpt_v2_exp27_branches.py --base-checkpoint artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt --output-root artifacts\lpt_v2\experiments_exp27
```

| 分支 | 作用 | 只允许变化的变量 |
|---|---|---|
| `base_continued` | 无 ContextAdapter 的继续训练对照 | 无 |
| `exp_27_context_adapter` | 启用 RetNetContextAdapter | `retnet_context_adapter_enabled=true`、`retnet_context_adapter_alpha=1e-4` |

初始化脚本要求共同基座必须是 Q-only RetNetAssist，且不能启用 K adapter、ContextAdapter 或 xLSTMAssist。`exp_27_context_adapter` 会复用基座可匹配权重；新增的 context adapter down/up projection 与 FP32 `alpha_context` 使用新初始化。

### 3.2 训练命令

两个分支统一使用 chat SFT 工作流、`data/manifests/chat_sft.json` 和 `data/manifests/chat_sft_eval_exp27.json`。训练保留最终 `checkpoints/latest/model.pt`、`trainer_state.json`、`checkpoint_manifest.json`、`metrics.jsonl` 和 config；关闭 optimizer、scheduler、TensorBoard、best checkpoint 和额外 inference weights。

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued --init-checkpoint artifacts\lpt_v2\experiments_exp27\base_continued\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp27\base_continued --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp27.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp27_base_continued --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_27_context_adapter --init-checkpoint artifacts\lpt_v2\experiments_exp27\exp_27_context_adapter\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp27\exp_27_context_adapter --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp27.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp27_context_adapter --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

### 3.3 训练后评测命令

训练后每个分支执行 checkpoint validate、forward smoke、4100 长上下文代理和资源报告。长上下文使用 `seq-len=4100`，确保 RetNetAssist token count 大于 2048 局部窗口。

```powershell
.\.venv\Scripts\python.exe tools\validate_lpt_v2_checkpoint.py --checkpoint artifacts\lpt_v2\experiments_exp27\<branch>\checkpoints\latest\model.pt --map-location cpu
.\.venv\Scripts\python.exe tools\run_lpt_v2_forward_smoke.py --checkpoint artifacts\lpt_v2\experiments_exp27\<branch>\checkpoints\latest\model.pt --seq-len 32 --batch-size 1 --device auto --dtype auto --output-json help\LPTv2扩展实验\reports\exp27\<branch>_forward_smoke.json --output-md help\LPTv2扩展实验\reports\exp27\<branch>_forward_smoke.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_long_context_eval.py --checkpoint artifacts\lpt_v2\experiments_exp27\<branch>\checkpoints\latest\model.pt --seq-len 4100 --attention-window-size 2048 --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp27\<branch>_long_context.json --output-md help\LPTv2扩展实验\reports\exp27\<branch>_long_context.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_resource_report.py --checkpoint artifacts\lpt_v2\experiments_exp27\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp27\<branch>_resource.json --output-md help\LPTv2扩展实验\reports\exp27\<branch>_resource.md
```

## 4. 数据项覆盖

| 数据项 | 来源 | 状态 |
|---|---|---|
| 分支初始化报告 | `artifacts/lpt_v2/experiments_exp27/*/init/init_report.json` | 已完成 |
| 训练 loss / final trainer loss | `metrics.jsonl`、`trainer_state.json` | 已完成 |
| eval loss / PPL | `metrics.jsonl` eval 行 | 已完成 |
| tokens_seen / samples_seen | `trainer_state.json`、`metrics.jsonl` | 已完成 |
| 训练吞吐、序列长度、CUDA 峰值 | `metrics.jsonl` | 已完成 |
| checkpoint schema | `validate_lpt_v2_checkpoint.py` | 已完成 |
| forward smoke | `reports/exp27/*_forward_smoke.json/md` | 已完成 |
| RetNet ContextAdapter norm / alpha | forward smoke、resource report、long context report | 已完成 |
| 长上下文代理指标 | `reports/exp27/*_long_context.json/md` | 已完成 |
| 资源指标 | `reports/exp27/*_resource.json/md` | 已完成 |

## 5. 结果记录

### 5.1 Checkpoint Validate

两个分支均通过 schema v2 校验：

```text
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
```

### 5.2 训练与 Eval

| 分支 | final trainer loss | latest/best eval loss | best eval step | latest eval PPL | tokens_seen | avg tokens/s | max seq len | max train CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 17.260975 | 24.727376 | 1100 | 5.482302e10 | 435335 | 343.175 | 2328 | 16468.799 |
| `exp_27_context_adapter` | 17.211117 | 25.010954 | 1100 | 7.279795e10 | 435335 | 314.688 | 2328 | 16523.202 |

`exp_27_context_adapter` 的 final trainer loss 略低，但 eval loss 比 `base_continued` 高 `0.283578`，eval PPL 更高；训练吞吐下降约 `8.30%`，训练 CUDA 峰值增加约 `54.403 MiB`。

### 5.3 Forward Smoke

| 分支 | forward_ok | loss | PPL | RetNet layer states | RetNet state slots | context enabled | Q adapter norm | context norm | alpha_context |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| `base_continued` | true | 11.769736 | 129280.053809 | 24 | 6 | false | 3.564406 | 0.000000 | 0.00000000 |
| `exp_27_context_adapter` | true | 11.769736 | 129280.053809 | 24 | 6 | true | 3.909823 | 11.030073 | -0.00005287 |

Forward smoke 确认 ContextAdapter 路径真实启用，`context_adapter_delta_norm` 可观测，且没有改变 logits shape、RetNet layer state 数量或 state slot 数量。`alpha_context` 训练后变为负值，说明该分支学到的是沿 context delta 的反向残差修正；当前实现只约束初始值非负，不对训练后的 scale 做 clamp。

### 5.4 长上下文代理

| 分支 | status | long loss | needle rank | needle logprob | RetNet tokens | Paged KV window | Q adapter norm | context norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | `admit_checkpoint_path` | 53.270287 | 25133 | -24.275009 | 4096 | 2048 | 1.370587 | 0.000000 |
| `exp_27_context_adapter` | `admit_checkpoint_path` | 49.252171 | 23507 | -23.369598 | 4096 | 2048 | 1.663666 | 4.672080 |

两个分支均跨过 2048 窗口并通过机制准入。ContextAdapter 分支 long loss 降低 `4.018116`，needle rank 提升 `1626` 名，needle logprob 提升 `0.905411`，说明它对当前长上下文代理有明确正向信号。

### 5.5 资源与机制指标

| 分支 | prefill tok/s | decode tok/s | first token ms | peak MiB | RetNet state bytes | context norm | router entropy | router z-loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 30.272 | 10.885 | 81.990 | 2766.113 | 768 | 0.000000 | 0.730056 | 20.077140 |
| `exp_27_context_adapter` | 27.766 | 9.891 | 101.906 | 2766.146 | 768 | 12.037177 | 0.802298 | 18.589082 |

ContextAdapter 不增加 RetNet runtime state bytes，资源评测显存峰值几乎不变；但 prefill 吞吐下降约 `8.28%`，decode 吞吐下降约 `9.13%`，首 token 延迟增加约 `19.916 ms`。Router entropy 更高、z-loss 更低，是稳定性侧的正向信号，但不足以抵消 eval 退化和吞吐成本。

### 5.6 参数与状态成本

| 分支 | total params | active params/token | RetNet core params | RetNet Q adapter params | RetNet context adapter params | adapter params | estimated FP32 state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 1,308,907,521 | 478,435,329 | 69,632 | 17,409 | 0 | 17,409 | 1,536 |
| `exp_27_context_adapter` | 1,308,924,930 | 478,452,738 | 69,632 | 17,409 | 17,409 | 34,818 | 1,536 |

在当前 `global/group` 共享策略下，ContextAdapter 只增加一组低秩 adapter 参数，共 `17,409` 个参数；不增加状态槽，也不增加 Paged KV。

## 6. 讨论

ContextAdapter 的机制是有效的：forward、long context 和 resource 报告都能观测到 context delta；长上下文代理也给出明显收益，尤其是 long loss、needle rank 和 needle logprob 同时改善。

`data/manifests/chat_sft_eval_exp27.json` 上 eval loss 从 `24.727376` 升到 `25.010954`，需要作为组合实验风险项保留，但该 eval 口径样本量较小，不能单独否定 ContextAdapter。当前更合理的解释是：ContextAdapter 对窗口外摘要利用有明确帮助，但需要在组合候选和更大 eval 口径下确认它不会造成实质通用能力退化。训练后 `alpha_context` 为负值，说明模型学到的是沿 context delta 的反向残差修正；这不是阻塞项，但组合实验需要继续观测 scale 稳定性。

资源侧成本可控但不可忽略：参数只增加 `17,409`，state bytes 不变；但训练吞吐和资源评测 prefill/decode 都下降约 8% 到 9%，首 token 延迟增加约 20 ms。该成本低于当前组合实验可接受的 10% 观察线，但最终是否默认启用必须由组合实验确认。

## 7. 结论

第 27 项结论修订为：`RetNetContextAdapter` 设为长上下文主候选，进入组合实验；组合实验效果不劣化则直接进入主干。

原自动结论为：`RetNetContextAdapter` 暂不进入默认主干或默认组合实验，仅保留为长上下文专项候选。人工介入后，基于“长上下文效果好是必要条件，小样本 eval loss 小幅上升不足以直接否决机制”的判断，将其调整为主候选。

证据：

1. 长上下文收益：4100 长上下文代理 long loss 从 `53.270287` 降到 `49.252171`，needle rank 和 logprob 均改善，ContextAdapter norm 可观测。
2. 状态成本：RetNet runtime state bytes 不增加，Paged KV 不受影响，额外参数仅 `17,409`。
3. 风险项：chat SFT eval loss 从 `24.727376` 升到 `25.010954`，训练吞吐下降约 `8.30%`，资源评测 prefill/decode 吞吐下降约 `8.28%/9.13%`，需要在组合实验中复核。

组合实验准入规则：

1. 组合分支应包含第 25 项 `global/per_layer`、第 26 项 `every_4_layers/rank16` 和第 27 项 `RetNetContextAdapter`。
2. 必须设置无 ContextAdapter 的组合对照。
3. 若 ContextAdapter 组合分支在长上下文指标上不劣化，且通用 eval、吞吐、显存不出现实质劣化，则直接进入主干。
4. 若只保留长上下文收益但通用 eval 或资源成本明显退化，则降级为长上下文专项开关。

## 8. 附录

- 初始化脚本：`tools/init_lpt_v2_exp27_branches.py`
- eval manifest：`data/manifests/chat_sft_eval_exp27.json`
- 训练产物：`artifacts/lpt_v2/experiments_exp27/...`
- 报告产物：`help/LPTv2扩展实验/reports/exp27/...`
- 训练后报告：
  - `help/LPTv2扩展实验/reports/exp27/*_forward_smoke.json`
  - `help/LPTv2扩展实验/reports/exp27/*_long_context.json`
  - `help/LPTv2扩展实验/reports/exp27/*_resource.json`
