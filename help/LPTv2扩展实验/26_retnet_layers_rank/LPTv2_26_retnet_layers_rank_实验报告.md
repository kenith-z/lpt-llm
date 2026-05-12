# LPT v2 第 26 项 RetNetAssist 启用层与 rank 实验报告

## 摘要

第 26 项已完成五分支 1164 step chat SFT 训练和训练后评测。当前定型口径调整为：`every_4_layers/rank16` 作为后续组合实验主候选，因为它的 4100 长上下文代理 loss 最低、训练吞吐最高且训练显存峰值最低；`every_2_layers/rank16` 虽然 eval loss 最低，但长上下文代理退化，保留为质量对照/备选；`rank32` 未带来 eval 收益，不进入默认主干。

需要保留的限制：本项为单项归因，仍保持基座 `global/group` 共享策略，没有混入第 25 项的 `global/per_layer` 候选。后续组合实验优先验证 `global/per_layer + every_4_layers`，并保留 `global/per_layer + every_2_layers` 作为质量对照。

## 1. 实验目的

1. 比较 `all_layers`、`every_2_layers`、`every_4_layers` 与 selected offset 的 RetNetAssist 启用层密度。
2. 比较 `rank16` 与 `rank32` 的 adapter 容量收益和成本。
3. 每个实验分支只改变一个维度：层策略或 rank，不同时改变共享策略、Q/K adapter、ContextAdapter 或 xLSTMAssist。

## 2. 实验材料与环境

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/model.pt` |
| base source manifest | `data/manifests/text_pretrain.json` |
| branch training manifest | `data/manifests/chat_sft.json` |
| eval manifest | `data/manifests/chat_sft_eval_exp26.json` |
| tokenizer | `lpt_model/ds_tokenizer` |
| 训练工作流 | chat SFT |
| RetNet 共享策略 | 保持基座 `global/group` |
| 报告输出目录 | `help/LPTv2扩展实验/reports/exp26/` |

## 3. 方法

### 3.1 初始化命令

```powershell
.\.venv\Scripts\python.exe tools\init_lpt_v2_exp26_branches.py --base-checkpoint artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt --output-root artifacts\lpt_v2\experiments_exp26
```

| 分支 | RetNet 层策略 | rank | 只允许变化的变量 |
|---|---|---:|---|
| `base_continued` | `all_layers` | 16 | 无 |
| `exp_26_retnet_every_2_layers` | `every_2_layers` | 16 | `retnet_assist_layers` |
| `exp_26_retnet_every_4_layers` | `every_4_layers` | 16 | `retnet_assist_layers` |
| `exp_26_retnet_selected_offset_layers` | `selected_layers=[2,6,10,14,18,22]` | 16 | `retnet_assist_layers` / `retnet_assist_selected_layers` |
| `exp_26_retnet_rank32` | `all_layers` | 32 | `retnet_adapter_rank` |

`rank32` 分支跳过 shape 不匹配的 adapter projection 权重，只复用 Attention、MoE、RetNet core、alpha 等可匹配权重；新 rank projection 保持新初始化。

### 3.2 训练命令

五个分支统一使用 chat SFT 工作流、`data/manifests/chat_sft.json` 和 `data/manifests/chat_sft_eval_exp26.json`。训练保留最终 `checkpoints/latest/model.pt`、`trainer_state.json`、`checkpoint_manifest.json`、`metrics.jsonl` 和 config；关闭 optimizer、scheduler、TensorBoard、best checkpoint 和额外 inference weights。

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued --init-checkpoint artifacts\lpt_v2\experiments_exp26\base_continued\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp26\base_continued --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp26.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp26_base_continued --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

其它分支只替换 `--stage`、`--init-checkpoint`、`--artifact-dir` 和 `--run-id` 为对应分支名。完整训练命令已在本报告历史版本中记录，训练产物位于 `artifacts/lpt_v2/experiments_exp26/...`。

### 3.3 训练后评测命令

```powershell
.\.venv\Scripts\python.exe tools\validate_lpt_v2_checkpoint.py --checkpoint artifacts\lpt_v2\experiments_exp26\<branch>\checkpoints\latest\model.pt --map-location cpu
.\.venv\Scripts\python.exe tools\run_lpt_v2_forward_smoke.py --checkpoint artifacts\lpt_v2\experiments_exp26\<branch>\checkpoints\latest\model.pt --seq-len 32 --batch-size 1 --device auto --dtype auto --output-json help\LPTv2扩展实验\reports\exp26\<branch>_forward_smoke.json --output-md help\LPTv2扩展实验\reports\exp26\<branch>_forward_smoke.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_long_context_eval.py --checkpoint artifacts\lpt_v2\experiments_exp26\<branch>\checkpoints\latest\model.pt --seq-len 4100 --attention-window-size 2048 --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp26\<branch>_long_context.json --output-md help\LPTv2扩展实验\reports\exp26\<branch>_long_context.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_resource_report.py --checkpoint artifacts\lpt_v2\experiments_exp26\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp26\<branch>_resource.json --output-md help\LPTv2扩展实验\reports\exp26\<branch>_resource.md
```

## 4. 数据项覆盖

| 数据项 | 来源 | 状态 |
|---|---|---|
| 分支初始化报告 | `artifacts/lpt_v2/experiments_exp26/*/init/init_report.json` | 已完成 |
| 训练 loss / final trainer loss | `metrics.jsonl`、`trainer_state.json` | 已完成 |
| eval loss / PPL | `metrics.jsonl` eval 行 | 已完成 |
| tokens_seen / samples_seen | `trainer_state.json`、`metrics.jsonl` | 已完成 |
| 训练吞吐、序列长度、CUDA 峰值 | `metrics.jsonl` | 已完成 |
| checkpoint schema | `validate_lpt_v2_checkpoint.py` | 已完成 |
| forward smoke | `reports/exp26/*_forward_smoke.json/md` | 已完成 |
| RetNet 启用层、rank、state slot | init report、forward smoke、resource report | 已完成 |
| 长上下文代理指标 | `reports/exp26/*_long_context.json/md` | 已完成 |
| 资源指标 | `reports/exp26/*_resource.json/md` | 已完成 |

## 5. 结果记录

### 5.1 Checkpoint Validate

五个分支均通过 schema v2 校验：

```text
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
```

### 5.2 训练与 Eval

| 分支 | final trainer loss | final train loss | best/latest eval loss | best eval step | latest eval PPL | tokens_seen | avg tokens/s | max seq len | max train CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 17.052820 | 26.748331 | 25.721639 | 1100 | 1.481719e11 | 435335 | 333.988 | 2328 | 16482.040 |
| `exp_26_retnet_every_2_layers` | 17.283241 | 26.391188 | 25.552126 | 1100 | 1.250683e11 | 435335 | 334.850 | 2328 | 16370.945 |
| `exp_26_retnet_every_4_layers` | 17.387251 | 26.842936 | 25.958100 | 1100 | 1.876980e11 | 435335 | 364.046 | 2328 | 16247.045 |
| `exp_26_retnet_selected_offset_layers` | 17.194378 | 26.666399 | 25.747802 | 1100 | 1.520997e11 | 435335 | 335.239 | 2328 | 16294.607 |
| `exp_26_retnet_rank32` | 17.276749 | 26.745516 | 25.796208 | 1100 | 1.596433e11 | 435335 | 331.106 | 2328 | 16450.687 |

`every_2_layers` 的 eval loss 最低，比 `base_continued` 低 `0.169513`。`rank32` 比 baseline 高 `0.074569`，没有证明额外 adapter 容量有效。

### 5.3 Forward Smoke

| 分支 | forward_ok | RetNet layer states | expected layer states | RetNet state slots | layers | rank | Q adapter norm |
|---|---|---:|---:|---:|---|---:|---:|
| `base_continued` | true | 24 | 24 | 6 | `all_layers` | 16 | 4.191429 |
| `exp_26_retnet_every_2_layers` | true | 12 | 12 | 6 | `every_2_layers` | 16 | 2.661732 |
| `exp_26_retnet_every_4_layers` | true | 6 | 6 | 6 | `every_4_layers` | 16 | 3.274777 |
| `exp_26_retnet_selected_offset_layers` | true | 6 | 6 | 6 | `selected_layers` | 16 | 5.045404 |
| `exp_26_retnet_rank32` | true | 24 | 24 | 6 | `all_layers` | 32 | 2.518423 |

Forward smoke 确认第 26 项的层策略真实生效：未启用层不再产生 RetNet layer state；state slot 仍为 6，是因为本项保持 `global/group` 状态共享，6 个 layer group 均被覆盖。

### 5.4 长上下文代理

| 分支 | status | long loss | needle rank | needle logprob | RetNet tokens | Paged KV window | first RetNet layer | enabled layers | Q adapter norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | `admit_checkpoint_path` | 50.436684 | 24694 | -23.801624 | 4096 | 2048 | 0 | 24 | 2.109570 |
| `exp_26_retnet_every_2_layers` | `admit_checkpoint_path` | 52.446457 | 26333 | -24.956926 | 4096 | 2048 | 0 | 12 | 1.355735 |
| `exp_26_retnet_every_4_layers` | `admit_checkpoint_path` | 48.870083 | 25287 | -24.632858 | 4096 | 2048 | 0 | 6 | 1.418979 |
| `exp_26_retnet_selected_offset_layers` | `admit_checkpoint_path` | 49.288731 | 26069 | -24.696518 | 4096 | 2048 | 2 | 6 | 2.320045 |
| `exp_26_retnet_rank32` | `admit_checkpoint_path` | 50.267555 | 25257 | -24.411352 | 4096 | 2048 | 0 | 24 | 1.206580 |

五个分支均跨过 2048 窗口并通过机制准入。质量代理没有单一完全胜者：`every_4_layers` 的 long loss 最低，baseline 的 needle rank/logprob 最好，`every_2_layers` 在长上下文代理上明显退化。

### 5.5 资源与机制指标

| 分支 | prefill tok/s | decode tok/s | first token ms | peak MiB | RetNet state bytes | RetNet slots | enabled layers | layer states | rank | router entropy | router z-loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 25.636 | 6.919 | 154.062 | 2766.113 | 768 | 6 | 24 | 24 | 16 | 0.720449 | 20.975037 |
| `exp_26_retnet_every_2_layers` | 27.812 | 8.398 | 123.905 | 2766.095 | 768 | 6 | 12 | 12 | 16 | 0.745050 | 21.334541 |
| `exp_26_retnet_every_4_layers` | 27.563 | 7.486 | 135.178 | 2766.086 | 768 | 6 | 6 | 6 | 16 | 0.732880 | 21.941067 |
| `exp_26_retnet_selected_offset_layers` | 26.851 | 8.025 | 135.314 | 2766.086 | 768 | 6 | 6 | 6 | 16 | 0.697026 | 20.808015 |
| `exp_26_retnet_rank32` | 25.027 | 11.501 | 94.740 | 2766.146 | 768 | 6 | 24 | 24 | 32 | 0.754476 | 20.947058 |

稀疏层策略减少了 RetNet 前向调用数，但由于本项保持 `global/group` 参数和状态共享，state bytes 与 state slots 没有下降。训练显存峰值随层稀疏下降，`every_4_layers` 最低。

### 5.6 参数与状态成本

| 分支 | total params | active params/token | RetNet core params | RetNet Q adapter params | adapter params | estimated FP32 state bytes |
|---|---:|---:|---:|---:|---:|---:|
| `base_continued` | 1,441,290,241 | 610,818,049 | 69,632 | 17,409 | 17,409 | 1,536 |
| `exp_26_retnet_every_2_layers` | 1,441,290,241 | 610,818,049 | 69,632 | 17,409 | 17,409 | 1,536 |
| `exp_26_retnet_every_4_layers` | 1,441,290,241 | 610,818,049 | 69,632 | 17,409 | 17,409 | 1,536 |
| `exp_26_retnet_selected_offset_layers` | 1,441,290,241 | 610,818,049 | 69,632 | 17,409 | 17,409 | 1,536 |
| `exp_26_retnet_rank32` | 1,441,307,649 | 610,835,457 | 69,632 | 34,817 | 34,817 | 1,536 |

在 `global/group` 共享策略下，层稀疏不减少物理参数，只减少运行时 RetNetAssist 调用。`rank32` 将 RetNet Q adapter 参数翻倍，但没有获得 eval loss 收益。

## 6. 讨论

`every_2_layers` 是本轮质量对照/备选：它以同等物理参数取得最低 eval loss，同时训练显存峰值低于 baseline。但它的长上下文代理表现最差，说明减少到 12 个启用层后，短期 chat SFT eval 与长上下文代理之间出现分歧。因此它不作为第 26 项主候选，只作为组合实验中的质量侧参照。

`every_4_layers` 是第 26 项主候选：它只启用 6 层，训练吞吐最高、训练显存峰值最低、4100 long loss 最低。虽然 eval loss 劣于 baseline 和 `every_2_layers`，但当前第 26 项的决策优先考虑长上下文代理与低计算成本，因此将其作为后续组合实验的主线入口。

`selected_offset_layers` 证明相同 6 层数量下层位置会影响结果：相对 `every_4_layers`，它 eval loss 更好、router z-loss 更低，但 long loss 略差。这个分支不进入默认主干，但可作为后续层位置搜索的参考。

`rank32` 不进入默认主干：它增加 RetNet adapter 参数，decode 与首 token 指标较好，但 eval loss 没有优于 baseline，long loss 也未超过 `every_4_layers`。

## 7. 结论

第 26 项结论为：`every_4_layers/rank16` 作为后续组合实验主候选，`every_2_layers/rank16` 保留为质量对照/备选，`rank32` 不进入默认主干。

进入后续组合实验的建议：

1. 主候选：`global/per_layer`（第 25 项）+ `every_4_layers/rank16`。
2. 质量对照/备选：`global/per_layer`（第 25 项）+ `every_2_layers/rank16`。
3. 暂不纳入：`rank32`、`selected_offset_layers`。

保留限制：第 26 项是在 `global/group` 共享策略下完成的单项归因，和第 25 项的最优共享策略存在交互风险，最终主干仍需组合实验确认。

## 8. 附录

- 初始化脚本：`tools/init_lpt_v2_exp26_branches.py`
- eval manifest：`data/manifests/chat_sft_eval_exp26.json`
- 训练产物：`artifacts/lpt_v2/experiments_exp26/...`
- 报告产物：`help/LPTv2扩展实验/reports/exp26/...`
- 训练后报告：
  - `help/LPTv2扩展实验/reports/exp26/*_forward_smoke.json`
  - `help/LPTv2扩展实验/reports/exp26/*_long_context.json`
  - `help/LPTv2扩展实验/reports/exp26/*_resource.json`
