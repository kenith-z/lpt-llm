# LPT v2 第 25 项 RetNetAssist 共享策略实验报告

## 摘要

第 25 项已完成四分支 1164 step chat SFT 训练和训练后评测。当前结论：`global parameter + per_layer state` 的 eval loss 最低、参数成本与 `global/group` 相同、运行态 state 成本绝对值很小，建议作为后续组合实验的 RetNetAssist 共享策略候选；`global/group` 保留为低状态成本 fallback；`group/group` 和 `per_layer/per_layer` 不进入默认主干。

需要保留的限制：4100 长上下文代理均已跨过 2048 窗口并通过机制准入，但它仍是代理任务，不足以单独证明长上下文质量定型。

## 1. 实验目的

1. 比较 RetNetAssist 参数 `global / group / per_layer` 共享对 chat SFT 收敛、eval loss、长上下文代理和资源成本的影响。
2. 比较 RetNetAssist 状态 `group / per_layer` 共享对状态语义、state bytes 和长上下文代理的影响。
3. 保持 Q-only RetNetAssist，不启用 K adapter、ContextAdapter 或 xLSTMAssist，避免污染第 25 项归因。

## 2. 实验材料与环境

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/model.pt` |
| base source manifest | `data/manifests/text_pretrain.json` |
| branch training manifest | `data/manifests/chat_sft.json` |
| eval manifest | `data/manifests/chat_sft_eval_exp25.json` |
| tokenizer | `lpt_model/ds_tokenizer` |
| 训练工作流 | chat SFT |
| RetNet group size | 连续 4 层一组 |
| 报告输出目录 | `help/LPTv2扩展实验/reports/exp25/` |

## 3. 方法

### 3.1 初始化命令

```powershell
.\.venv\Scripts\python.exe tools\init_lpt_v2_exp25_branches.py --base-checkpoint artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt --output-root artifacts\lpt_v2\experiments_exp25
```

| 分支 | 参数共享 | 状态共享 | 作用 |
|---|---|---|---|
| `base_continued` | `global` | `group` | 低成本共享策略基线 |
| `exp_25_global_per_layer` | `global` | `per_layer` | 隔离状态共享影响 |
| `exp_25_group_group` | `group` | `group` | 每 4 层共享一组参数和状态 |
| `exp_25_per_layer_per_layer` | `per_layer` | `per_layer` | 参数与状态容量上限对照 |

### 3.2 训练命令

四个分支统一使用 chat SFT 工作流、`data/manifests/chat_sft.json` 和 `data/manifests/chat_sft_eval_exp25.json`。训练保留最终 `checkpoints/latest/model.pt`、`trainer_state.json`、`checkpoint_manifest.json`、`metrics.jsonl` 和 config；关闭 optimizer、scheduler、TensorBoard、best checkpoint 和额外 inference weights。

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued --init-checkpoint artifacts\lpt_v2\experiments_exp25\base_continued\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp25\base_continued --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp25.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp25_base_continued --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

其它分支只替换 `--stage`、`--init-checkpoint`、`--artifact-dir` 和 `--run-id` 为对应分支名。

### 3.3 训练后评测命令

训练后对每个分支执行 checkpoint validate、forward smoke、长上下文代理和资源报告。长上下文使用 `seq-len=4100`，确保 RetNetAssist token count 大于 2048 的局部窗口。

```powershell
.\.venv\Scripts\python.exe tools\validate_lpt_v2_checkpoint.py --checkpoint artifacts\lpt_v2\experiments_exp25\<branch>\checkpoints\latest\model.pt --map-location cpu
.\.venv\Scripts\python.exe tools\run_lpt_v2_forward_smoke.py --checkpoint artifacts\lpt_v2\experiments_exp25\<branch>\checkpoints\latest\model.pt --seq-len 32 --batch-size 1 --device auto --dtype auto --output-json help\LPTv2扩展实验\reports\exp25\<branch>_forward_smoke.json --output-md help\LPTv2扩展实验\reports\exp25\<branch>_forward_smoke.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_long_context_eval.py --checkpoint artifacts\lpt_v2\experiments_exp25\<branch>\checkpoints\latest\model.pt --seq-len 4100 --attention-window-size 2048 --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp25\<branch>_long_context.json --output-md help\LPTv2扩展实验\reports\exp25\<branch>_long_context.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_resource_report.py --checkpoint artifacts\lpt_v2\experiments_exp25\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp25\<branch>_resource.json --output-md help\LPTv2扩展实验\reports\exp25\<branch>_resource.md
```

## 4. 数据项覆盖

| 数据项 | 来源 | 状态 |
|---|---|---|
| 训练 loss / final trainer loss | `metrics.jsonl`、`trainer_state.json` | 已完成 |
| eval loss / PPL | `metrics.jsonl` eval 行 | 已完成 |
| tokens_seen / samples_seen | `trainer_state.json`、`metrics.jsonl` | 已完成 |
| 训练吞吐、序列长度、CUDA 峰值 | `metrics.jsonl` | 已完成 |
| checkpoint schema | `validate_lpt_v2_checkpoint.py` | 已完成 |
| forward smoke | `reports/exp25/*_forward_smoke.json/md` | 已完成 |
| RetNet 参数共享/状态共享/slot count | init report、forward smoke、resource report | 已完成 |
| 长上下文代理指标 | `reports/exp25/*_long_context.json/md` | 已完成 |
| 资源指标 | `reports/exp25/*_resource.json/md` | 已完成 |

## 5. 结果记录

### 5.1 Checkpoint Validate

四个分支均通过 schema v2 校验：

```text
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
```

### 5.2 训练与 Eval

| 分支 | final trainer loss | final train loss | best/latest eval loss | best eval step | latest eval PPL | tokens_seen | avg tokens/s | max seq len | max train CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 17.405348 | 26.709335 | 26.799543 | 1100 | 4.354053e11 | 435335 | 328.249 | 2328 | 16468.716 |
| `exp_25_global_per_layer` | 17.168850 | 26.462921 | 26.443517 | 1100 | 3.049817e11 | 435335 | 331.854 | 2328 | 16450.840 |
| `exp_25_group_group` | 17.116564 | 26.590338 | 26.497069 | 1100 | 3.217592e11 | 435335 | 354.142 | 2328 | 16430.989 |
| `exp_25_per_layer_per_layer` | 17.478184 | 26.802734 | 26.762245 | 1100 | 4.194646e11 | 435335 | 311.681 | 2328 | 16442.688 |

`exp_25_global_per_layer` 的 eval loss 最低，比 `base_continued` 低 `0.356026`；`exp_25_group_group` 次之，低 `0.302474`；`per_layer/per_layer` 只有很小收益，且成本最高。

### 5.3 Forward Smoke

| 分支 | forward_ok | RetNet state slots | RetNet layer states | 参数共享 | 状态共享 | Q adapter norm |
|---|---|---:|---:|---|---|---:|
| `base_continued` | true | 6 | 24 | `global` | `group` | 4.887850 |
| `exp_25_global_per_layer` | true | 24 | 24 | `global` | `per_layer` | 3.294047 |
| `exp_25_group_group` | true | 6 | 24 | `group` | `group` | 4.287733 |
| `exp_25_per_layer_per_layer` | true | 24 | 24 | `per_layer` | `per_layer` | 4.461261 |

Forward smoke 确认共享策略实际生效：`group` 状态只有 6 个 state slot，但每层仍有可绑定的 RetNet layer state；`per_layer` 状态有 24 个 state slot。

### 5.4 长上下文代理

| 分支 | status | long loss | needle rank | needle logprob | RetNet tokens | Paged KV window | Q adapter norm |
|---|---|---:|---:|---:|---:|---:|---:|
| `base_continued` | `admit_checkpoint_path` | 50.007080 | 26016 | -24.809299 | 4096 | 2048 | 0.261567 |
| `exp_25_global_per_layer` | `admit_checkpoint_path` | 50.715603 | 26855 | -25.117964 | 4096 | 2048 | 0.233971 |
| `exp_25_group_group` | `admit_checkpoint_path` | 50.973766 | 24956 | -23.902933 | 4096 | 2048 | 0.328280 |
| `exp_25_per_layer_per_layer` | `admit_checkpoint_path` | 52.477898 | 28452 | -25.669361 | 4096 | 2048 | 0.586605 |

4100 长上下文补跑后，四个分支均跨过 2048 窗口，机制准入通过。质量侧没有单一完全胜者：`base_continued` 的 long loss 最低，`exp_25_group_group` 的 needle rank/logprob 最好，`exp_25_global_per_layer` 在长上下文代理上没有超过低成本 baseline。

### 5.5 资源与机制指标

| 分支 | prefill tok/s | decode tok/s | first token ms | peak MiB | RetNet state bytes | RetNet slots | router entropy | router z-loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 29.183 | 10.418 | 81.051 | 2766.113 | 768 | 6 | 0.746317 | 21.326418 |
| `exp_25_global_per_layer` | 30.614 | 11.977 | 81.478 | 2766.113 | 3072 | 24 | 0.772738 | 18.023885 |
| `exp_25_group_group` | 25.898 | 11.885 | 79.961 | 2766.945 | 768 | 6 | 0.790981 | 19.546822 |
| `exp_25_per_layer_per_layer` | 31.176 | 11.003 | 80.394 | 2769.942 | 3072 | 24 | 0.821304 | 19.762910 |

`global/per_layer` 的 runtime state 从 768 bytes 增加到 3072 bytes，但绝对成本很小；同时 decode tok/s 和 router z-loss 最优。`group/group` 保持 768 bytes state，但 resource prefill 最低。

### 5.6 参数与状态成本

| 分支 | total params | active params/token | RetNet core params | RetNet Q adapter params | adapter params | estimated FP32 state bytes |
|---|---:|---:|---:|---:|---:|---:|
| `base_continued` | 1,441,290,241 | 610,818,049 | 69,632 | 17,409 | 17,409 | 1,536 |
| `exp_25_global_per_layer` | 1,441,290,241 | 610,818,049 | 69,632 | 17,409 | 17,409 | 6,144 |
| `exp_25_group_group` | 1,441,725,446 | 611,253,254 | 417,792 | 104,454 | 104,454 | 1,536 |
| `exp_25_per_layer_per_layer` | 1,443,292,184 | 612,819,992 | 1,671,168 | 417,816 | 417,816 | 6,144 |

`global/per_layer` 不增加 RetNet 参数，只增加 state slot；`group/group` 和 `per_layer/per_layer` 都增加物理参数，并没有带来足以抵消成本的稳定收益。

## 6. 讨论

`global/per_layer` 是当前最稳的质量/成本折中：它保持全局共享参数，因此物理参数与 baseline 相同；per-layer state 避免同组层复用同一个历史摘要，eval loss 明显优于 `global/group`。state bytes 增加 4 倍，但从 768 bytes 到 3072 bytes，实际成本可以忽略。

`group/group` 在 eval loss 上优于 `global/group`，并且长上下文 needle 指标最好，但它增加 6 倍 RetNet core/Q adapter 参数，resource prefill 明显下降，收益不够稳定。

`per_layer/per_layer` 是容量上限分支，但 eval loss、长上下文代理和资源指标都没有证明额外参数值得保留，应作为负向对照归档。

长上下文代理只能说明机制跨窗可用，不能单独证明最终长上下文质量。第 25 项结论应进入后续组合实验，与第 26 项启用层/rank 和第 27 项 ContextAdapter 一起确认。

## 7. 结论

第 25 项结论为：`global parameter + per_layer state` 进入后续组合实验候选。

证据：

1. 质量收益：eval loss 从 `26.799543` 降到 `26.443517`，为四分支最低。
2. 参数成本：RetNet core 和 Q adapter 参数与 `global/group` 相同，没有额外物理参数。
3. 状态成本：runtime RetNet state bytes 从 `768` 到 `3072`，绝对值很小。
4. 稳定性：forward smoke 全部通过，`global/per_layer` 的 router z-loss 最低、decode tok/s 最高。

保留限制：`global/per_layer` 没有在 4100 长上下文代理中赢过低成本 baseline，因此第 25 项不单独作为长上下文定型充分证据。`global/group` 继续作为低成本 fallback；`group/group` 和 `per_layer/per_layer` 不进入默认主干。

## 8. 附录

- 初始化脚本：`tools/init_lpt_v2_exp25_branches.py`
- eval manifest：`data/manifests/chat_sft_eval_exp25.json`
- 训练产物：`artifacts/lpt_v2/experiments_exp25/...`
- 报告产物：`help/LPTv2扩展实验/reports/exp25/...`
- 训练后报告：
  - `help/LPTv2扩展实验/reports/exp25/*_forward_smoke.json`
  - `help/LPTv2扩展实验/reports/exp25/*_long_context.json`
  - `help/LPTv2扩展实验/reports/exp25/*_resource.json`
