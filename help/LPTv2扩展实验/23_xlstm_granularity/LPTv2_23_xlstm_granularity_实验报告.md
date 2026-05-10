# LPT v2 第 23 项 xLSTMAssist 记忆粒度实验报告

## 摘要

第 23 项聚焦 xLSTMAssist 的层粒度消融：`all_layers`、`every_2_layers`、`every_4_layers`、`selected_layers(后 1/4)`，统一使用训练后的 `text_pretrain` 基座和 `chat_sft` 工作流。

五个分支均完成 1164 step chat SFT 训练，并通过 checkpoint schema 校验、真实 checkpoint forward smoke、长上下文代理评测、资源报告和 xLSTM 专项评测。当前结论：`exp_23_xlstm_every_4_layers` 在 eval loss 最低、状态成本低和机制观测通过之间取得最好平衡，作为后续组合实验中的 xLSTM 记忆粒度候选；由于长上下文报告仍为 `close_or_debug`，本项不单独作为主干定型充分证据。

## 1. 实验目的

1. 找到 xLSTMAssist 的最小可用启用层粒度。
2. 比较不同粒度下的训练收敛、eval loss、长上下文代理指标和显存/吞吐成本。
3. 确认按启用层独立更新状态时，xLSTMAssist 是否仍保持可观测收益。

## 2. 实验材料与环境

### 2.1 基座 checkpoint

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lpt_v2/text_pretrain/checkpoints/latest/model.pt` |
| base source manifest | `data/manifests/text_pretrain.json` |
| branch training manifest | `data/manifests/chat_sft.json` |
| eval manifest | `data/manifests/chat_sft_eval_exp23.json` |
| tokenizer | `lpt_model/ds_tokenizer` |
| tokenizer config sha256 | `E4E7A46CD993C8E7F2422F868C87B41AD16C4EF270EC1417199D4414AFF0549B` |
| tokenizer vocab size | `128000` tokenizer vocab；forward logits padded vocab 为 `129280` |

### 2.2 实验分支

| 分支 | 作用 | 初始化 checkpoint |
|---|---|---|
| `base_continued` | 同等预算继续训练对照 | `artifacts/lpt_v2/experiments_exp23/base_continued/init/model.pt` |
| `exp_23_xlstm_all_layers` | xLSTM 全层启用对照 | `artifacts/lpt_v2/experiments_exp23/exp_23_xlstm_all_layers/init/model.pt` |
| `exp_23_xlstm_every_2_layers` | 每 2 层启用一次 | `artifacts/lpt_v2/experiments_exp23/exp_23_xlstm_every_2_layers/init/model.pt` |
| `exp_23_xlstm_every_4_layers` | 每 4 层启用一次 | `artifacts/lpt_v2/experiments_exp23/exp_23_xlstm_every_4_layers/init/model.pt` |
| `exp_23_xlstm_selected_late_layers` | 仅后 1/4 层启用 | `artifacts/lpt_v2/experiments_exp23/exp_23_xlstm_selected_late_layers/init/model.pt` |

初始化脚本：`tools/init_lpt_v2_exp23_branches.py`。

## 3. 方法

### 3.1 初始化命令

```powershell
.\.venv\Scripts\python.exe tools\init_lpt_v2_exp23_branches.py --base-checkpoint artifacts\lpt_v2\text_pretrain\checkpoints\latest\model.pt --output-root artifacts\lpt_v2\experiments_exp23
```

### 3.2 配置差异

| 分支 | 配置变化 |
|---|---|
| `base_continued` | 无结构变化 |
| `exp_23_xlstm_all_layers` | `xlstm_memory_enabled=true`、`xlstm_memory_layers="all_layers"`、`moe_router_input_mode="memory_augmented_input"`、`xlstm_memory_gate_enabled=false` |
| `exp_23_xlstm_every_2_layers` | 上述基础上 `xlstm_memory_layers="every_2_layers"` |
| `exp_23_xlstm_every_4_layers` | 上述基础上 `xlstm_memory_layers="every_4_layers"` |
| `exp_23_xlstm_selected_late_layers` | 上述基础上 `xlstm_memory_layers="selected_layers"`、`xlstm_memory_selected_layers=后 1/4 层` |

### 3.3 训练命令

统一使用 chat SFT 工作流和 `data/manifests/chat_sft.json`。

省空间策略：第 23 项只保留实验归因必需的 `checkpoints/latest/model.pt`、`trainer_state.json`、`checkpoint_manifest.json`、`metrics.jsonl` 和 config；不保存 optimizer、scheduler、TensorBoard、best checkpoint，也不额外导出 `weights/model_weights.pth` / `weights/model_checkpoint.pt`。`latest_save_interval=0` 表示只在训练结束保存 final latest checkpoint，中途不可依赖 checkpoint 恢复。

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage base_continued --init-checkpoint artifacts\lpt_v2\experiments_exp23\base_continued\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp23\base_continued --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp23.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp23_base_continued --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_23_xlstm_all_layers --init-checkpoint artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_all_layers\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_all_layers --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp23.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp23_xlstm_all_layers --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_23_xlstm_every_2_layers --init-checkpoint artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_every_2_layers\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_every_2_layers --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp23.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp23_xlstm_every_2_layers --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_23_xlstm_every_4_layers --init-checkpoint artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_every_4_layers\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_every_4_layers --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp23.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp23_xlstm_every_4_layers --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

```powershell
.\.venv\Scripts\python.exe tools\train_lpt_v2_experiment_branch.py --stage exp_23_xlstm_selected_late_layers --init-checkpoint artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_selected_late_layers\init\model.pt --artifact-dir artifacts\lpt_v2\experiments_exp23\exp_23_xlstm_selected_late_layers --manifest data\manifests\chat_sft.json --eval-manifest data\manifests\chat_sft_eval_exp23.json --batch-size 1 --eval-batch-size 1 --learning-rate 3e-4 --eval-interval 100 --eval-max-batches 32 --latest-save-interval 0 --seed 20260503 --run-id exp23_xlstm_selected_late_layers --no-resume --no-save-optimizer --no-save-scheduler --no-tensorboard --no-save-inference-weights
```

### 3.4 评测命令

#### 3.4.1 Forward smoke

每个分支训练完成后，先对 `checkpoints/latest/model.pt` 跑一次只读 forward smoke。默认 `use_kv_cache=false`，用于验证训练 forward 路径；如需额外抽查推理前向兼容性，可再补一条 `--use-kv-cache`。

```powershell
.\.venv\Scripts\python.exe tools\run_lpt_v2_forward_smoke.py --checkpoint artifacts\lpt_v2\experiments_exp23\<branch>\checkpoints\latest\model.pt --seq-len 32 --batch-size 1 --device auto --dtype auto --output-json help\LPTv2扩展实验\reports\exp23\<branch>_forward_smoke.json --output-md help\LPTv2扩展实验\reports\exp23\<branch>_forward_smoke.md
```

#### 3.4.2 正式评测

```powershell
.\.venv\Scripts\python.exe tools\validate_lpt_v2_checkpoint.py --checkpoint artifacts\lpt_v2\experiments_exp23\<branch>\checkpoints\latest\model.pt --map-location cpu
.\.venv\Scripts\python.exe tools\run_lpt_v2_long_context_eval.py --checkpoint artifacts\lpt_v2\experiments_exp23\<branch>\checkpoints\latest\model.pt --seq-len 2052 --attention-window-size 2048 --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp23\<branch>_long_context.json --output-md help\LPTv2扩展实验\reports\exp23\<branch>_long_context.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_resource_report.py --checkpoint artifacts\lpt_v2\experiments_exp23\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp23\<branch>_resource.json --output-md help\LPTv2扩展实验\reports\exp23\<branch>_resource.md
.\.venv\Scripts\python.exe tools\run_lpt_v2_memory_eval.py --checkpoint artifacts\lpt_v2\experiments_exp23\<branch>\checkpoints\latest\model.pt --device auto --dtype bf16 --output-json help\LPTv2扩展实验\reports\exp23\<branch>_memory.json --output-md help\LPTv2扩展实验\reports\exp23\<branch>_memory.md
```

### 3.5 数据项覆盖检查

当前省空间训练命令仍满足第 23 项原始数据项要求。区别是：不再保存 `best checkpoint` 文件，best eval 结论从 `metrics.jsonl` 的 eval 行计算；不再保存 optimizer/scheduler/TensorBoard/额外 inference weights，这些不参与实验归因。

| 数据项 | 来源 | 是否满足 | 备注 |
|---|---|---|---|
| 训练 loss / final trainer loss | `checkpoints/latest/trainer_state.json`、`metrics.jsonl` train 行 | 是 | `trainer_state.last_loss` 是 final trainer loss |
| 验证 loss / PPL | `metrics.jsonl` eval 行 | 是 | `--eval-manifest`、`--eval-interval 100`、`--eval-max-batches 32` 已启用 |
| best eval loss / best eval step | `metrics.jsonl` eval 行 | 是 | 计算最小 `eval_loss`，不依赖 best checkpoint 文件 |
| tokens_seen / samples_seen | `trainer_state.json`、`metrics.jsonl` train 行 | 是 | 用于确认训练预算一致 |
| 训练吞吐 | `metrics.jsonl.tokens_per_sec` | 是 | 取平均或分位数均可 |
| 训练序列长度 | `metrics.jsonl.sequence_length` | 是 | 用于确认长样本和 OOM 风险 |
| 训练 CUDA 显存峰值 | `metrics.jsonl.cuda_peak_memory_allocated_mib` | 是 | 仅 CUDA 训练时出现；CPU 训练为空属正常 |
| forward smoke | `help/LPTv2扩展实验/reports/exp23/<branch>_forward_smoke.json/md` | 是 | 检查 logits finite、shape、state count、xLSTM state count |
| 长上下文指标 | `help/LPTv2扩展实验/reports/exp23/<branch>_long_context.json/md` | 是 | 覆盖 loss/PPL、needle rank、机制 ready、RetNet/Paged KV 计数 |
| 资源指标 | `help/LPTv2扩展实验/reports/exp23/<branch>_resource.json/md` | 是 | 覆盖 prefill/decode 吞吐、首 token 延迟、显存、state bytes、MoE 指标 |
| router entropy / load balance / z-loss | `help/LPTv2扩展实验/reports/exp23/<branch>_resource.json` | 是 | 来自 MoE layer state 聚合 |
| RetNet/xLSTM state bytes | `help/LPTv2扩展实验/reports/exp23/<branch>_resource.json` | 是 | 对比状态成本 |
| xLSTM memory / adapter norm | `help/LPTv2扩展实验/reports/exp23/<branch>_memory.json`、`resource.json` | 是 | 覆盖 `memory_norm`、`adapter_delta_norm`、effective beta |
| reset/decay 机制 | `help/LPTv2扩展实验/reports/exp23/<branch>_memory.json` | 是 | 覆盖 boundary/special/session reset 与 decay |
| checkpoint schema | `validate_lpt_v2_checkpoint.py` 控制台输出 | 是 | 需把 `checkpoint_ok ...` 复制进训练记录或报告附录 |

## 4. 结果记录

### 4.0 Forward Smoke

| 分支 | forward_ok | logits_finite | logits_shape | loss | PPL | state_count | xlstm_state_count | expected_xlstm_state_count | paged_kv_page_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `base_continued` | true | true | `1x32x129280` | 11.769736 | 129280.054 | 24 | 0 | 0 | 0 |
| `exp_23_xlstm_all_layers` | true | true | `1x32x129280` | 11.769736 | 129280.054 | 24 | 24 | 24 | 0 |
| `exp_23_xlstm_every_2_layers` | true | true | `1x32x129280` | 11.769736 | 129280.054 | 24 | 12 | 12 | 0 |
| `exp_23_xlstm_every_4_layers` | true | true | `1x32x129280` | 11.769736 | 129280.054 | 24 | 6 | 6 | 0 |
| `exp_23_xlstm_selected_late_layers` | true | true | `1x32x129280` | 11.769736 | 129280.054 | 24 | 6 | 6 | 0 |

### 4.1 训练曲线

`exp_23_xlstm_every_2_layers` 的 `metrics.jsonl` 中包含历史重跑残留，正式汇总只采用最后一段完整 run：从第 152 行开始的 128 条记录，其中 117 条 train、11 条 eval。其它分支从文件起始处汇总。

| 分支 | final trainer loss | latest eval loss | best eval loss | best eval step | latest eval PPL | tokens_seen | avg tokens/s | max seq len | max train CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 17.151045 | 28.172313 | 28.172313 | 1100 | 1.718225e12 | 435335 | 351.764 | 2328 | 16420.070 |
| `exp_23_xlstm_all_layers` | 17.216946 | 28.186723 | 28.186723 | 1100 | 1.743163e12 | 435335 | 305.727 | 2328 | 16708.045 |
| `exp_23_xlstm_every_2_layers` | 17.129784 | 28.085871 | 28.085871 | 1100 | 1.575936e12 | 435335 | 316.036 | 2328 | 16551.296 |
| `exp_23_xlstm_every_4_layers` | 17.137640 | 28.054874 | 28.054874 | 1100 | 1.527837e12 | 435335 | 337.181 | 2328 | 16541.307 |
| `exp_23_xlstm_selected_late_layers` | 17.464487 | 28.340882 | 28.340882 | 1100 | 2.033707e12 | 435335 | 324.454 | 2328 | 16532.057 |

### 4.2 资源与机制指标

| 分支 | prefill tokens/s | decode tokens/s | first token ms | peak MiB | RetNet state bytes | xLSTM state bytes | router entropy | load balance loss | router z_loss | xLSTM memory norm | xLSTM adapter delta norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_continued` | 26.610 | 7.390 | 157.593 | 2766.888 | 3072 | 0 | 0.815915 | 3.000 | 17.953770 | 0.000 | 0.000 |
| `exp_23_xlstm_all_layers` | 29.556 | 8.292 | 94.245 | 2766.923 | 3072 | 3072 | 0.796813 | 3.000 | 18.636315 | 2.591 | 2.478 |
| `exp_23_xlstm_every_2_layers` | 29.403 | 9.294 | 120.898 | 2766.905 | 3072 | 1536 | 0.710727 | 3.000 | 19.299075 | 2.674 | 2.637 |
| `exp_23_xlstm_every_4_layers` | 30.400 | 10.293 | 96.104 | 2766.896 | 3072 | 768 | 0.830289 | 3.000 | 19.833432 | 2.342 | 1.953 |
| `exp_23_xlstm_selected_late_layers` | 28.716 | 10.716 | 98.130 | 2766.896 | 3072 | 768 | 0.779805 | 3.000 | 21.527145 | 4.415 | 6.507 |

### 4.3 长上下文与 xLSTM 专项

| 分支 | long status | long loss | needle rank | RetNet tokens | Paged KV window | memory status | effective beta | memory logit delta L2 | boundary reset | special reset ready | session reset |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| `base_continued` | `close_or_debug` | 46.951973 | 21656 | 2048 | 2048 | `close_or_debug` | 0.00000000 | 0.000000 |  | false |  |
| `exp_23_xlstm_all_layers` | `close_or_debug` | 46.751740 | 23362 | 2048 | 2048 | `admit_instrumentation_only` | 0.00010028 | 0.174189 | 1 | true | 2 |
| `exp_23_xlstm_every_2_layers` | `close_or_debug` | 46.350883 | 30422 | 2048 | 2048 | `admit_instrumentation_only` | 0.00010028 | 0.137757 | 1 | true | 2 |
| `exp_23_xlstm_every_4_layers` | `close_or_debug` | 46.642094 | 30211 | 2048 | 2048 | `admit_instrumentation_only` | 0.00010028 | 0.158132 | 1 | true | 2 |
| `exp_23_xlstm_selected_late_layers` | `close_or_debug` | 46.147652 | 31272 | 2048 | 2048 | `admit_instrumentation_only` | 0.00010028 | 0.018600 | 1 | true | 2 |

### 4.4 结论

- 数据项覆盖：五个分支均完成 checkpoint validate、forward smoke、训练/eval 汇总、长上下文代理评测、资源评测和 xLSTM 专项评测。省空间训练未保存 best checkpoint 文件，但 best eval loss 可由 `metrics.jsonl` 的 eval 行计算，不影响本项归因。
- 收敛与 eval：`exp_23_xlstm_every_4_layers` 的 latest/best eval loss 最低，为 `28.054874`；`every_2_layers` 次优，为 `28.085871`；二者均优于 `base_continued` 的 `28.172313`。`all_layers` 未优于 base，`selected_late_layers` eval loss 最高，不建议继续作为候选。
- 资源指标：`every_4_layers` 的 xLSTM state bytes 为 `768`，只有 `all_layers` 的 25%；训练峰值显存为 `16541.307 MiB`，比 `all_layers` 低约 `166.738 MiB`，比 base 高约 `121.236 MiB`。资源评测的 peak MiB 基本持平，decode 吞吐达到 `10.293 tokens/s`。
- xLSTM 专项：四个 xLSTM 分支均达到 `admit_instrumentation_only`，reset/decay 机制可观测。`every_4_layers` 的 forward smoke `xlstm_state_count=6/6`，状态粒度正确；adapter delta norm 低于 `all_layers` 和 `every_2_layers`，对 MoE 输入扰动更收敛。
- 长上下文：五个分支仍为 `close_or_debug`，因此当前长上下文代理结果只能作为机制观测，不能作为主干定型的充分质量证据。`selected_late_layers` 虽然 long loss 最低，但 eval loss、adapter norm 和 router z_loss 均最差，不采纳。
- 最终结论：第 23 项选择 `exp_23_xlstm_every_4_layers` 作为后续组合实验中的 xLSTM 记忆粒度候选；`every_2_layers` 保留为候补对照；`all_layers` 和 `selected_late_layers` 放弃进入后续组合实验。本项不直接修改主干默认配置，等待组合实验和更强长上下文准入共同确认。

## 5. 附录

### 5.1 训练记录

- `help/LPTv2扩展实验/23_xlstm_granularity/base_continued训练记录.md`
- `help/LPTv2扩展实验/23_xlstm_granularity/exp_23_xlstm_all_layers训练记录.md`
- `help/LPTv2扩展实验/23_xlstm_granularity/exp_23_xlstm_every_2_layers训练记录.md`
- `help/LPTv2扩展实验/23_xlstm_granularity/exp_23_xlstm_every_4_layers训练记录.md`
- `help/LPTv2扩展实验/23_xlstm_granularity/exp_23_xlstm_selected_late_layers训练记录.md`

### 5.2 产物路径

- `artifacts/lpt_v2/experiments_exp23/...`
- `help/LPTv2扩展实验/reports/exp23/...`
