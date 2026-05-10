# LPT v2 模型定型方案变动记录

本文只记录 LPT v2 定型方案的历史变动。当前实现与任务拆解以 `help/LPTv2模型定型方案.md` 为准。

## 2026-05-11

- 补齐 v2 评测工具缺口：新增 `tools/benchmark_lpt_v2_sequence_packing.py`，用于对同一批 chat/text 样本比较 sequence packing 开关下的 token 利用率、吞吐、step 耗时和 CUDA peak memory。
- 完成 LongRoPE2 短期与中期评测补充：新增 `tools/run_lpt_v2_longrope2_factor_sweep.py`、`lpt_eval/longrope2_factor_sweep.py`、`tools/run_lpt_v2_long_context_suite.py` 和 `lpt_eval/long_context_suite.py`；factor sweep 只在评测进程内临时替换 factors，不写回 checkpoint。
- 长上下文准入新增 `needle_depth`，单次评测、suite 与 factor sweep 统一复用真实 checkpoint 准入口径，便于第 22/23/24 项之后补充多位置 needle 和 LongRoPE2 因子对照。

## 2026-05-10

- 完成第 24 项 `RetNetAssist Q/K Adapter` 两分支 1164 step chat SFT 对比，报告路径为 `help/LPTv2扩展实验/24_qk_adapter/LPTv2_24_qk_adapter_实验报告.md`；`exp_24_qk_adapter` eval loss 优于 `base_continued`，且 K adapter norm 可观测，但长上下文代理指标退化、router z-loss 明显升高，因此暂不进入主干或组合实验。
- 完成第 24 项 `RetNetAssist Q/K Adapter` 的代码与工具准备：允许 `retnet_assist_mode="qk_adapter"` 与 `retnet_adapter_target=("q","k")` 的单项实验配置，新增 K adapter 前向、FP32 `alpha_k`、RetNet adapter norm 观测指标、`tools/init_lpt_v2_exp24_branches.py`、`data/manifests/chat_sft_eval_exp24.json` 和实验报告模板；训练分支按省空间口径只保留 `base_continued` 与 `exp_24_qk_adapter`。

## 2026-05-07

- 完成第 23 项 `xLSTMAssist 记忆粒度` 五分支 1164 step chat SFT 对比，报告路径为 `help/LPTv2扩展实验/23_xlstm_granularity/LPTv2_23_xlstm_granularity_实验报告.md`；所有分支均通过 checkpoint validate、真实 checkpoint forward smoke、长上下文代理、资源和 xLSTM 专项评测。
- 第 23 项结论调整为：`every_4_layers` 的 eval loss 最低，xLSTM state bytes 为全层方案的 25%，forward smoke 中 `xlstm_state_count=6/6`，作为后续组合实验中的 xLSTM 记忆粒度候选；`every_2_layers` 保留为候补对照，`all_layers` 和 `selected_late_layers` 不进入后续组合实验。
- 保留长上下文限制：五个分支长上下文报告仍为 `close_or_debug`，因此第 23 项只完成层粒度单项归因，不单独作为主干定型充分证据，需等待组合实验和更强长上下文准入共同确认。

## 2026-05-06

- 完成第 23 项 `xLSTMAssist 记忆粒度` 的代码与工具准备：新增 `xlstm_memory_selected_layers`、显式 `every_2_layers/every_4_layers/selected_layers` 启用逻辑、按实际启用层统计的 xLSTM 参数/状态字节、`tools/init_lpt_v2_exp23_branches.py`、`data/manifests/chat_sft_eval_exp23.json` 和实验报告模板；本轮先做层粒度消融，local/global memory 待 forward 路径接入后单独评估。
- 将只读 forward smoke 纳入第 23 项准入流程：新增 `tools/run_lpt_v2_forward_smoke.py`，每个分支 checkpoint 训练后先检查 logits finite、shape、layer state 数量与 xLSTM state count，再进入长上下文、资源和 xLSTM 专项评测。
- 新增训练省空间开关 `--no-save-inference-weights`，第 23 项训练命令同步关闭 optimizer、scheduler、TensorBoard、best checkpoint 和额外 inference weights，并把 `latest_save_interval` 设为 0，仅保留 final latest checkpoint 与 metrics 供实验分析。

## 2026-05-05

- 完成第 22 项 `xLSTMAssist Memory Gate` 三分支 chat SFT 单项对比和带独立 eval manifest 的完整 1164 step 补充实验，报告路径为 `help/LPTv2扩展实验/22_xlstm_memory_gate/LPTv2_22_xlstm_memory_gate_实验报告.md`；当前结论为放弃当前输入 Memory Gate 方案，暂不默认启用，也不进入组合实验。
- 明确 Paged KV Cache 的训练/推理边界：prefill/decode 默认使用 KV cache，训练 forward 固定关闭 KV cache，不向 page pool 写入训练 K/V，也不保存 dense K/V state。
- 明确训练 LM loss 使用分块 cross entropy，避免一次性构造完整 `batch * sequence_length * vocab_size` 的 FP32 logits 副本，降低长样本触发的显存峰值。
- 明确 `SwiGLUMoE` 运行时只执行 router top-k 命中的 SwiGLU experts，未命中的 experts 不参与当前 batch 的前向与反向计算；物理参数统计仍按全部 experts 计入。
- 补充训练资源观测指标：训练日志记录 `sequence_length`，CUDA 训练时记录 allocated/reserved/peak memory，用于定位 OOM 触发样本。
- 补充 P3 单项分支实验规范：22、23、24、26、27 必须使用训练后的 `artifacts/lpt_v2/text_pretrain` 作为共同基座分别继续训练，保持数据、tokenizer、训练预算、seed、LongRoPE2 和硬件等控制变量一致，并与 base-continued 对照比较。
- 要求 22、23、24、26、27 每一项都产出独立实验报告，报告结构参考 v1 `GLM5.1及DS4的Tokenizer基准对比实验`，必须覆盖摘要、目的、材料环境、方法、结果、讨论、结论和附录。

## 2026-05-03

- 将主方案文件从 `20260503LPT模型定型方案.md` 改名为 `LPTv2模型定型方案.md`。
- 主方案文件职责收敛为只记录当前模型架构、配置字段、运行 Profile 与任务清单。
- 确定 LPT v2 主体为 `Attention-First + RetNetAssist-Q + Paged KV + Memory-Augmented SwiGLU-MoE`。
- Attention 层从初版 `Local FlashAttention-3 Attention` 调整为当前定型的 `Local SDPA Attention`；`flash_attention_2 / flash_attention_3` 降级为 P3 可选性能评估项，不再阻塞 P1 主干闭环。
- Paged KV Cache 只保存局部窗口内真实 token 的 `K/V`，不保存 RetNetAssist 或 xLSTMAssist 状态。
- RetNet 侧定型为 `RetNetAssist`，只维护轻量全局摘要，并默认通过低秩 `Q Adapter` 调制当前 token 的 `query`。
- 默认关闭 `K Adapter`、RetNet KV 替代、Attention logit bias 与直接输出注入；这些方向只保留为 ablation。
- FFN 层从 `MoxE / xLSTM expert` 路线调整为 `Memory-Augmented SwiGLU-MoE`。
- 所有 MoE experts 均定型为无状态 SwiGLU，MoE 只承担静态容量扩展与稀疏 FFN 计算。
- `xLSTM/mLSTM` 从 MoE expert 中移出，作为 FFN 侧外挂记忆模块，在启用层确定性更新状态。
- xLSTMAssist 通过低秩 adapter 生成 `x_ffn`，默认供 Router 与 SwiGLU experts 使用；它不作为 MoE expert 或 router target。
- 补充 `ffn_norm_only_router`、Memory Gate、local/global 记忆粒度等消融任务，但默认不启用。
- 根据 `方案意见.md` 统一 xLSTMAssist 缩放因子命名，将配置中的 `alpha` 口径改为 `beta` 口径。
- 删除主方案中 `xlstm_memory_router_visible` 的重叠语义，由 `moe_router_input_mode` 统一控制 Router 输入来源。
- 将 `retnet_state_sharing` 收敛为 `group | per_layer`，避免全局单状态与 request-bound state pool 语义冲突。
- 补充 xLSTMAssist 的 `chunkwise_recurrent_scan` prefill、`prefill_to_decode` 状态连续性、token interval decay、special token/session event reset 和 `zero_state` 边界重置。
- 补充 xLSTMAssist adapter beta 的 FP32 sigmoid clamp 策略和 effective beta 可观测指标。
- 补充 Paged KV、RetNetAssistState、xLSTMMemoryState 三类状态池隔离约束。
- 补充 MoE router entropy、expert load balance loss、router z_loss 指标，用于评估 xLSTMAssist 对 Router 分布的影响。
- 明确 Memory Gate 的输入门控公式，并将输出门控限定为输出缩放评估，不计入省计算收益。
- 在主方案中补充 Mermaid 图，用于描述 LPT v2 总运行流程、LPTBlockV2 内部结构和状态池隔离关系；原文字规格和 ASCII 全景图保留。
- 在主方案顶部补充 LPT v2 独立开发分支边界，明确不兼容 LPT v1 checkpoint、旧 `ModelConfig`、旧参数名、旧 recipe 和旧推理加载路径。
- 补充 LPT v2 多规格 `ModelConfig` 预设，默认规格设为 `lpt_v2_dev_tiny`，用于开发测试、shape 测试和 checkpoint schema 测试。
- 将模型参数统计计划调整为 MoE-aware 参数统计器，区分物理总参数、每 token 激活参数、共享参数、专家参数、router 参数、adapter 参数和运行态 state bytes。
- 完成主分支 v1/v2 分割：v1 / ds-token 代码只保留在本地只读归档目录`.tmp_lpt_v1_ds_archive`中，不进入版本控制，不作为运行时 import 依赖；v2 若需要复用 v1 基础件，必须复制到 v2 正式模块并按 v2 命名、配置、状态和测试边界改造后再使用，避免重复开发同时禁止保留 v1 兼容分支。
