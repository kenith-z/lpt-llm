# LPT v2 第 22 项 xLSTMAssist Memory Gate 实验报告

## 摘要

第 22 项已完成首轮三分支 chat SFT 单项对比，并完成带独立 eval manifest 的完整 1164 step 补充实验：`base_continued`、`exp_22_xlstm_no_gate`、`exp_22_xlstm_memory_gate` 都从同一个 `text_pretrain` 基座初始化，控制变量一致。

当前结论：放弃当前输入 Memory Gate 方案，暂不默认启用，也不进入组合实验。完整补充实验中 `base_continued` 的 eval loss 最低，no-gate 次之，Memory Gate 最高；Memory Gate 虽然继续降低 adapter 扰动，但没有带来可验证质量收益，资源侧收益不足以抵消 eval 退化。

## 1. 实验目的

本实验回答以下问题：

1. 在 `xLSTMAssist` 启用后，输入记忆门控 `gate_m = sigmoid(W_gate(h_ffn))` 是否比无 gate 的 xLSTM adapter 更稳定。
2. Memory Gate 是否改善训练收敛、状态追踪、长上下文或 PPL 代理指标。
3. Memory Gate 是否造成 MoE router collapse、expert load imbalance、吞吐下降或显存峰值不可接受。

本实验不回答输出 gate 是否进入主干。输出 gate 必须作为后续独立子实验处理。

## 2. 实验材料与环境

### 2.1 基座 checkpoint

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/model.pt` |
| trainer state | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/trainer_state.json` |
| base source manifest | `data/manifests/text_pretrain.json` |
| branch training manifest | `data/manifests/chat_sft.json` |
| tokenizer | `lpt_model/ds_tokenizer` |
| tokenizer config sha256 | `e4e7a46cd993c8e7f2422f868c87b41ad16c4ef270ec1417199d4414aff0549b` |
| tokenizer vocab size | `129280` |
| base global_step | `2680` |
| base tokens_seen | `5031895` |
| base checkpoint dtype | `bfloat16` |

### 2.2 实验分支

| 分支 | 作用 | 初始化 checkpoint |
|---|---|---|
| `base_continued` | 同等预算继续训练对照 | `artifacts/lpt_v2/experiments_exp22/base_continued/init/model.pt` |
| `exp_22_xlstm_no_gate` | xLSTM 无 gate 对照 | `artifacts/lpt_v2/experiments_exp22/exp_22_xlstm_no_gate/init/model.pt` |
| `exp_22_xlstm_memory_gate` | Memory Gate 实验分支 | `artifacts/lpt_v2/experiments_exp22/exp_22_xlstm_memory_gate/init/model.pt` |

初始化记录：`artifacts/lpt_v2/experiments_exp22/exp22_init_summary.json`。

Memory Gate 初始化策略：新增 24 个 `memory_gate` 模块，`weight=0`、`bias=2.0`，让 gate 初始接近常开；missing keys 仅为新增 gate 权重和 bias，unexpected keys 为空。

### 2.3 运行环境

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 10 / PowerShell 7 |
| Python 环境 | 项目 `.venv` |
| 训练 workflow | `tools/train_lpt_v2_experiment_branch.py`，统一 chat SFT |
| 训练数据 | `data/manifests/chat_sft.json` |
| 训练 batch size | `1` |
| 训练 learning rate | `3e-4` |
| 训练 seed | `20260503` |
| 训练 checkpoint 间隔 | `latest_save_interval=10` |
| 评测 device / dtype | `cuda:0` / `bfloat16` |

## 3. 方法

### 3.1 分支初始化

```powershell
.\.venv\Scripts\python.exe tools\init_lpt_v2_exp22_branches.py --base-checkpoint artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt --output-root artifacts\lpt_v2\experiments_exp22
```

配置 diff：

| 分支 | 配置变化 |
|---|---|
| `base_continued` | 无结构变化 |
| `exp_22_xlstm_no_gate` | `xlstm_memory_enabled=true`、`xlstm_memory_layers="all_layers"`、`moe_router_input_mode="memory_augmented_input"`、`xlstm_memory_gate_enabled=false` |
| `exp_22_xlstm_memory_gate` | 在 no-gate 分支基础上启用 `xlstm_memory_gate_enabled=true`、`xlstm_memory_gate_mode="input_conditioned_eval"` |

### 3.2 训练命令

三条分支统一使用 chat SFT 工作流和 `data/manifests/chat_sft.json`。

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued --init-checkpoint artifacts\lpt_v2\experiments_exp22\base_continued\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp22\base_continued --manifest data\manifests\chat_sft.json --batch-size 1 --learning-rate 3e-4 --latest-save-interval 10
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_22_xlstm_no_gate --init-checkpoint artifacts\lpt_v2\experiments_exp22\exp_22_xlstm_no_gate\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp22\exp_22_xlstm_no_gate --manifest data\manifests\chat_sft.json --batch-size 1 --learning-rate 3e-4 --latest-save-interval 10
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_22_xlstm_memory_gate --init-checkpoint artifacts\lpt_v2\experiments_exp22\exp_22_xlstm_memory_gate\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp22\exp_22_xlstm_memory_gate --manifest data\manifests\chat_sft.json --batch-size 1 --learning-rate 3e-4 --latest-save-interval 10
```

### 3.3 评测命令

每个分支都执行 checkpoint 严格校验、长上下文、资源和 xLSTM 专项评测。长上下文评测使用 `seq_len=2052`、`attention_window_size=2048`。

```powershell
.\.venv\Scripts\python.exe tools\validate_lpt_v2_checkpoint.py --checkpoint artifacts\lpt_v2\experiments_exp22\<branch>\checkpoints\latest\model.pt --map-location cpu
.\.venv\Scripts\python.exe tools\run_lpt_v2_long_context_eval.py --checkpoint artifacts\lpt_v2\experiments_exp22\<branch>\checkpoints\latest\model.pt --seq-len 2052 --attention-window-size 2048 --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp22\<branch>_long_context.json --output-md help\LPTv2扩展实验\reports\exp22\<branch>_long_context.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_resource_report.py --checkpoint artifacts\lpt_v2\experiments_exp22\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp22\<branch>_resource.json --output-md help\LPTv2扩展实验\reports\exp22\<branch>_resource.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_memory_eval.py --checkpoint artifacts\lpt_v2\experiments_exp22\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp22\<branch>_memory.json --output-md help\LPTv2扩展实验\reports\exp22\<branch>_memory.md
```

## 4. 实验结果

三条分支的 `latest` checkpoint 均通过严格校验：

```text
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
```

### 4.1 训练曲线

| 分支 | final trainer loss | metrics avg loss | metrics min loss | tokens_seen | avg tokens/s | max train CUDA peak MiB | max seq len |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 17.216202 | 26.273387 | 11.207788 | 435335 | 227.765882 | 16447.648438 | 2328 |
| `exp_22_xlstm_no_gate` | 17.134064 | 26.262895 | 11.112613 | 435335 | 208.835579 | 16705.648926 | 2328 |
| `exp_22_xlstm_memory_gate` | 17.104897 | 26.278669 | 11.382931 | 435335 | 224.302324 | 16761.914062 | 2328 |

训练观察：

- gate 分支 final trainer loss 最低，比 no-gate 低 0.029167，比 base-continued 低 0.111305。
- metrics 全程均值没有支持 gate 明显更优，no-gate 的 avg loss 略低。
- no-gate 的训练吞吐下降明显；gate 分支吞吐接近 base-continued。
- gate 分支训练显存峰值最高，比 base-continued 高约 314 MiB，比 no-gate 高约 56 MiB，当前仍在可控范围内。

### 4.2 长上下文 checkpoint 准入

| 分支 | status | long loss | capped PPL | needle rank | code loss | format loss | RetNet tokens | Paged KV window |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | `close_or_debug` | 46.529285 | 485165195.410 | 30850 | 46.646477 | 46.858002 | 2048 | 2048 |
| `exp_22_xlstm_no_gate` | `close_or_debug` | 46.720619 | 485165195.410 | 21404 | 46.766716 | 47.020016 | 2048 | 2048 |
| `exp_22_xlstm_memory_gate` | `close_or_debug` | 46.470715 | 485165195.410 | 28764 | 46.550091 | 46.681988 | 2048 | 2048 |

长上下文观察：

- gate 分支在 long/code/format 三个 loss 代理指标上最低，但差距很小，不能单独作为质量收益证据。
- no-gate 的 needle rank 最好，gate 优于 base 但弱于 no-gate，长上下文代理指标呈混合结果。
- 三个报告均为 `close_or_debug`，因为当前状态计数显示 `RetNet tokens == Paged KV window == 2048`，没有形成“状态跨出局部窗口”的准入证据。该结果更像评测或状态计数边界问题，不支持对 Memory Gate 下定型结论。

### 4.3 资源指标

| 分支 | prefill tokens/s | decode tokens/s | first token ms | eval peak MiB | Paged KV KiB | RetNet state bytes | xLSTM state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 28.687346 | 7.092729 | 189.517300 | 2766.887695 | 480.000000 | 3072 | 0 |
| `exp_22_xlstm_no_gate` | 22.809491 | 5.705837 | 174.318200 | 2766.922852 | 480.000000 | 3072 | 3072 |
| `exp_22_xlstm_memory_gate` | 24.990098 | 5.984514 | 174.947000 | 2770.934570 | 480.000000 | 3072 | 3072 |

资源观察：

- gate 相比 no-gate，prefill 吞吐提升约 9.6%，decode 吞吐提升约 4.9%。
- gate 相比 base-continued，prefill 吞吐仍低约 12.9%，decode 吞吐低约 15.6%。
- eval peak memory 三分支接近，gate 仅比 no-gate 高约 4 MiB；训练峰值差异更明显，但仍未触发 OOM。

### 4.4 MoE 与 xLSTM 机制指标

| 分支 | router entropy | load balance loss | router z_loss | resource memory norm | resource adapter delta norm |
|---|---:|---:|---:|---:|---:|
| `base_continued` | 0.696305 | 3.000000 | 19.102977 | 0.000000 | 0.000000 |
| `exp_22_xlstm_no_gate` | 0.782566 | 3.000000 | 18.385893 | 2.590821 | 2.501962 |
| `exp_22_xlstm_memory_gate` | 0.752006 | 3.000000 | 19.070584 | 2.285014 | 2.170423 |

| 分支 | memory eval status | effective beta | memory norm | adapter delta norm | logit delta L2 | memory router entropy | boundary reset | special token reset | session reset |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | `close_or_debug` | 0.00000000 | 0.000000 | 0.000000 | 0.000000 | 1.701244 | 不适用 | 不适用 | 不适用 |
| `exp_22_xlstm_no_gate` | `close_or_debug` | 0.00010028 | 1.466848 | 0.894902 | 0.163304 | 1.733724 | 1 | 1 | 2 |
| `exp_22_xlstm_memory_gate` | `close_or_debug` | 0.00010028 | 1.380770 | 0.859500 | 0.136503 | 1.666870 | 1 | 1 | 2 |

机制观察：

- gate 分支的 memory norm、adapter delta norm 和 logit delta L2 都低于 no-gate，说明输入 gate 确实在压低 xLSTM adapter 对 FFN/MoE 输入的扰动。
- 未观察到 router entropy 归零；但 load balance loss 没有改善，router z_loss 与 base 接近。
- xLSTM 专项报告仍为 `close_or_debug`。当前 checkpoint 未配置 special token 边界 id，special token reset 没有相对 boundary reset 继续增加；因此该报告不能作为“状态治理完整通过”的证据。

## 5. 讨论

Memory Gate 的正向信号主要有三点：最终 trainer loss 最低，训练吞吐明显好于 no-gate，adapter 扰动指标低于 no-gate。这说明 gate 不是单纯增加参数，它确实改变了 xLSTM 对 FFN 输入的作用强度，并且没有在本轮看到明显 router collapse。

负向或不足也很明确：本轮训练没有 eval manifest，所有质量判断只能来自训练 loss 和代理评测；长上下文报告仍未通过机制准入；xLSTM 专项报告也没有完整通过 reset 机制。再加上 avg loss 没有支持 gate 稳定优于 no-gate，因此不能把 final loss 的小幅优势直接归因为 Memory Gate。

从工程角度看，gate 的额外显存成本可控，评测显存几乎不变，训练峰值增加约 314 MiB。吞吐成本比 base 仍高，但相对 no-gate 有改善。这使它适合作为候选继续扩大训练，而不是现在就进入默认配置。

## 6. 结论

结论：放弃当前输入 Memory Gate 方案。

当前不把 `xlstm_memory_gate_enabled=true` 写入默认主干，也不进入组合实验。原因是补充实验已经补齐独立 eval manifest，但质量、资源和长上下文代理指标均不足以支持 gate：

1. `base_continued` 的 latest/best eval loss 最低，Memory Gate 没有超过同预算 base continuation。
2. Memory Gate 的 eval loss 高于 no-gate xLSTM，无法证明 gate 相对 xLSTM 本身有收益。
3. xLSTM 专项评测已从误判的 `close_or_debug` 修正为 `admit_instrumentation_only`，但该机制通过没有转化为 eval 质量收益。
4. 长上下文报告仍为 `close_or_debug`，状态计数仍显示 `RetNet tokens == Paged KV window == 2048`，不能作为 gate 的正向证据。
5. Memory Gate 的资源侧没有稳定优势：prefill/decode 略高于 no-gate，但首 token 延迟和显存峰值更高，不能抵消 eval 质量退化。

## 7. 附录

### 7.1 训练记录

- `help/LPTv2扩展实验/22_xlstm_memory_gate/base_continued训练记录.md`
- `help/LPTv2扩展实验/22_xlstm_memory_gate/exp_22_xlstm_no_gate训练记录.md`
- `help/LPTv2扩展实验/22_xlstm_memory_gate/exp_22_xlstm_memory_gate训练记录.md`

### 7.2 训练产物

- `artifacts/lpt_v2/experiments_exp22/base_continued/checkpoints/latest/model.pt`
- `artifacts/lpt_v2/experiments_exp22/exp_22_xlstm_no_gate/checkpoints/latest/model.pt`
- `artifacts/lpt_v2/experiments_exp22/exp_22_xlstm_memory_gate/checkpoints/latest/model.pt`
- `artifacts/lpt_v2/experiments_exp22/base_continued/metrics.jsonl`
- `artifacts/lpt_v2/experiments_exp22/exp_22_xlstm_no_gate/metrics.jsonl`
- `artifacts/lpt_v2/experiments_exp22/exp_22_xlstm_memory_gate/metrics.jsonl`

### 7.3 评测报告

- `help/LPTv2扩展实验/reports/exp22/base_continued_long_context.json`
- `help/LPTv2扩展实验/reports/exp22/base_continued_resource.json`
- `help/LPTv2扩展实验/reports/exp22/base_continued_memory.json`
- `help/LPTv2扩展实验/reports/exp22/exp_22_xlstm_no_gate_long_context.json`
- `help/LPTv2扩展实验/reports/exp22/exp_22_xlstm_no_gate_resource.json`
- `help/LPTv2扩展实验/reports/exp22/exp_22_xlstm_no_gate_memory.json`
- `help/LPTv2扩展实验/reports/exp22/exp_22_xlstm_memory_gate_long_context.json`
- `help/LPTv2扩展实验/reports/exp22/exp_22_xlstm_memory_gate_resource.json`
- `help/LPTv2扩展实验/reports/exp22/exp_22_xlstm_memory_gate_memory.json`

### 7.4 已知限制

- 首轮没有 eval manifest；补充实验已使用 `data/manifests/chat_sft_eval_exp22.json` 补齐 eval loss / eval PPL。
- 长上下文代理指标只适合做准入辅助，不等同真实 QA/retrieval 质量。
- Memory Gate 只完成输入门控评估，输出门控未参与本轮实验。

## 8. 补充实验执行计划

### 8.1 新增 eval manifest

已新增独立验证 manifest：

```text
data/manifests/chat_sft_eval_exp22.json
```

该 manifest 从当前 `chat_sft.json` 中训练权重为 0 的数据源抽样，避免直接用训练集回看：

| 数据源 | 样本上限 | 用途 |
|---|---:|---|
| `Belle_100.chat.jsonl` | 256 | 通用中文指令验证 |
| `chat-KenithZ-dolly-zh-51k.chat.sft.jsonl` | 256 | Dolly 中文指令验证 |

### 8.2 带 eval 的完整 continuation 训练命令

执行目标：

- 三条分支都从第 22 项已训练完成的 `latest` checkpoint 继续。
- 不覆盖原始 `artifacts/lpt_v2/experiments_exp22/`，补实验输出到 `artifacts/lpt_v2/experiments_exp22_eval/`。
- 使用同一训练数据、验证数据、step 数、eval 间隔、batch size、learning rate 和 seed。
- 去掉 `--max-steps`，按 `epochs=1` 跑完整 1164 条训练 batch；每 100 step 评估一次，最多评估 32 个 eval batch，并保存 `eval_loss` 最优 checkpoint。

#### base-continued eval continuation

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued_eval --init-checkpoint artifacts\lpt_v2\experiments_exp22\base_continued\checkpoints\latest\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp22_eval\base_continued --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp22.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 20 --save-best-checkpoint --best-checkpoint-metric eval_loss --seed 20260503 --run-id exp22_eval_base_continued --no-resume
```

#### no-gate xLSTM eval continuation

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_22_xlstm_no_gate_eval --init-checkpoint artifacts\lpt_v2\experiments_exp22\exp_22_xlstm_no_gate\checkpoints\latest\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp22_eval\exp_22_xlstm_no_gate --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp22.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 20 --save-best-checkpoint --best-checkpoint-metric eval_loss --seed 20260503 --run-id exp22_eval_xlstm_no_gate --no-resume
```

#### Memory Gate eval continuation

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_22_xlstm_memory_gate_eval --init-checkpoint artifacts\lpt_v2\experiments_exp22\exp_22_xlstm_memory_gate\checkpoints\latest\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp22_eval\exp_22_xlstm_memory_gate --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp22.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 20 --save-best-checkpoint --best-checkpoint-metric eval_loss --seed 20260503 --run-id exp22_eval_xlstm_memory_gate --no-resume
```

### 8.3 训练完成后需要回收的证据

训练完成后，补充报告时需要读取以下产物：

- `artifacts/lpt_v2/experiments_exp22_eval/base_continued/metrics.jsonl`
- `artifacts/lpt_v2/experiments_exp22_eval/exp_22_xlstm_no_gate/metrics.jsonl`
- `artifacts/lpt_v2/experiments_exp22_eval/exp_22_xlstm_memory_gate/metrics.jsonl`
- 三条分支的 `checkpoints/latest/trainer_state.json`
- 三条分支的 `checkpoints/latest/model.pt`

重点比较：

1. `latest_eval_loss` 和 `latest_eval_ppl`。
2. `eval_loss` 最优点是否稳定落在 gate 分支。
3. 同等 1164 step continuation 后，训练 loss、eval loss、吞吐、显存峰值是否支持 gate。
4. 再用补实验后的 `latest` checkpoint 重跑 long context / resource / memory report。

### 8.4 xLSTM 专项评测口径修正

已修正 `lpt_eval/memory.py` 的判定口径：当 checkpoint 未配置 `xlstm_memory_boundary_token_ids` 时，special token reset 标为未配置，不再把该项作为 `close_or_debug` 的失败条件。

补实验完成后，仍需确认：

- `boundary_reset_count` 相对 decode 后递增。
- `session_reset_count` 相对 special token probe 后递增。
- `special_token_reset_configured=false` 时，`special_token_reset_ready=true` 表示未配置项已被正确跳过。

## 9. 补充实验结果

### 9.1 eval continuation 训练结果

三条分支均从首轮第 22 项 `latest` checkpoint 继续训练完整 1164 step，并统一使用：

- 训练 manifest：`data/manifests/chat_sft.json`
- eval manifest：`data/manifests/chat_sft_eval_exp22.json`
- `batch_size=1`
- `eval_batch_size=1`
- `eval_interval=100`
- `eval_max_batches=32`
- `seed=20260503`

三条分支的 `latest` checkpoint 均通过严格校验：

```text
checkpoint_ok architecture=lpt_v2 preset=lpt_v2_small_base layers=24
```

| 分支 | latest eval loss | latest eval PPL | best eval loss | best step | trainer global step | tokens_seen | final trainer loss | avg train loss | avg tokens/s | max CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued_eval` | 25.807446 | 161447548855.827 | 25.807446 | 1100 | 1164 | 435335 | 16.507763 | 23.313230 | 277.798603 | 16488.517578 |
| `exp_22_xlstm_no_gate_eval` | 25.830269 | 165174671573.030 | 25.830269 | 1100 | 1164 | 435335 | 16.292540 | 23.387370 | 221.752486 | 16716.676270 |
| `exp_22_xlstm_memory_gate_eval` | 26.108145 | 218083645634.277 | 26.108145 | 1100 | 1164 | 435335 | 16.320471 | 23.303420 | 230.592749 | 16849.024902 |

eval continuation 观察：

- `base_continued_eval` 的 latest/best eval loss 最低。
- no-gate xLSTM 的 final trainer loss 最低，但 eval loss 高于 base。
- Memory Gate 的 avg train loss 略低，但 eval loss 最高，且训练显存峰值最高，不支持进入主干或继续扩大当前 gate 方案。
- 三条分支 best eval checkpoint 均落在 step 1100；训练结束于 step 1164，最后一次 eval 仍是 step 1100。

### 9.2 补实验后长上下文报告

| 分支 | status | long loss | capped PPL | needle rank | code loss | format loss | RetNet tokens | Paged KV window |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued_eval` | `close_or_debug` | 45.867916 | 485165195.410 | 23030 | 45.936504 | 46.110817 | 2048 | 2048 |
| `exp_22_xlstm_no_gate_eval` | `close_or_debug` | 45.835266 | 485165195.410 | 23898 | 45.934086 | 46.104424 | 2048 | 2048 |
| `exp_22_xlstm_memory_gate_eval` | `close_or_debug` | 45.642277 | 485165195.410 | 32384 | 45.687813 | 45.848728 | 2048 | 2048 |

长上下文观察：

- 三条分支仍为 `close_or_debug`。
- Memory Gate 的 long/code/format loss 代理指标最低，但 needle rank 最差。
- long context 代理指标与 eval loss 结论不一致，因此不能覆盖正式 eval loss 的负向结论。
- `RetNet tokens == Paged KV window == 2048` 的状态计数问题仍存在，不能把该报告作为 Memory Gate 正向证据。

### 9.3 补实验后资源报告

| 分支 | prefill tokens/s | decode tokens/s | first token ms | eval peak MiB | router entropy | load balance loss | router z_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_continued_eval` | 4.950206 | 6.202289 | 155.194200 | 2766.887695 | 0.654800 | 3.000000 | 25.736296 |
| `exp_22_xlstm_no_gate_eval` | 21.993789 | 6.097285 | 160.079300 | 2766.922852 | 0.788188 | 3.000000 | 29.911964 |
| `exp_22_xlstm_memory_gate_eval` | 22.989308 | 6.149874 | 166.250000 | 2770.934570 | 0.536397 | 3.000000 | 31.485127 |

资源观察：

- Memory Gate 的 prefill/decode 吞吐略高于 no-gate，但首 token 延迟更高，评测峰值显存也最高。
- base resource prefill 本次偏低，资源报告存在短测波动；不把它作为质量结论依据。
- 未观察到 router entropy 归零；但 load balance loss 无改善，Memory Gate 的 router z_loss 最高。

### 9.4 补实验后 xLSTM 专项报告

| 分支 | memory status | special reset configured | special reset ready | memory norm | adapter delta norm | logit delta L2 | memory router entropy | boundary reset | session reset |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `base_continued_eval` | `close_or_debug` | false | false | 0.000000 | 0.000000 | 0.000000 | 1.707333 | 不适用 | 不适用 |
| `exp_22_xlstm_no_gate_eval` | `admit_instrumentation_only` | false | true | 1.536531 | 1.058695 | 0.161677 | 1.698190 | 1 | 2 |
| `exp_22_xlstm_memory_gate_eval` | `admit_instrumentation_only` | false | true | 1.375472 | 0.873163 | 0.172638 | 1.718724 | 1 | 2 |

xLSTM 专项观察：

- no-gate 和 Memory Gate 分支都已通过 xLSTM 机制观测。
- `special_token_reset_configured=false` 且 `special_token_reset_ready=true` 表示未配置项已被正确跳过，不再误判机制失败。
- Memory Gate 继续降低 memory norm 和 adapter delta norm，但 logit delta L2 高于 no-gate，且没有转化为 eval loss 收益。

### 9.5 补充实验产物

- `artifacts/lpt_v2/experiments_exp22_eval/base_continued/metrics.jsonl`
- `artifacts/lpt_v2/experiments_exp22_eval/exp_22_xlstm_no_gate/metrics.jsonl`
- `artifacts/lpt_v2/experiments_exp22_eval/exp_22_xlstm_memory_gate/metrics.jsonl`
- `help/LPTv2扩展实验/reports/exp22_eval/base_continued_long_context.json`
- `help/LPTv2扩展实验/reports/exp22_eval/base_continued_resource.json`
- `help/LPTv2扩展实验/reports/exp22_eval/base_continued_memory.json`
- `help/LPTv2扩展实验/reports/exp22_eval/exp_22_xlstm_no_gate_long_context.json`
- `help/LPTv2扩展实验/reports/exp22_eval/exp_22_xlstm_no_gate_resource.json`
- `help/LPTv2扩展实验/reports/exp22_eval/exp_22_xlstm_no_gate_memory.json`
- `help/LPTv2扩展实验/reports/exp22_eval/exp_22_xlstm_memory_gate_long_context.json`
- `help/LPTv2扩展实验/reports/exp22_eval/exp_22_xlstm_memory_gate_resource.json`
- `help/LPTv2扩展实验/reports/exp22_eval/exp_22_xlstm_memory_gate_memory.json`

补充训练记录：

- `help/LPTv2扩展实验/22_xlstm_memory_gate/base_continued_eval训练记录.md`
- `help/LPTv2扩展实验/22_xlstm_memory_gate/exp_22_xlstm_no_gate_eval训练记录.md`
- `help/LPTv2扩展实验/22_xlstm_memory_gate/exp_22_xlstm_memory_gate_eval训练记录.md`
