# LPT v2 模型定型方案变动记录

本文只记录 LPT v2 定型方案的历史变动。当前实现与任务拆解以 `help/LPTv2模型定型方案.md` 为准。

## 2026-05-03

- 将主方案文件从 `20260503LPT模型定型方案.md` 改名为 `LPTv2模型定型方案.md`。
- 主方案文件职责收敛为只记录当前模型架构、配置字段、运行 Profile 与任务清单。
- 确定 LPT v2 主体为 `Attention-First + RetNetAssist-Q + Paged KV + Memory-Augmented SwiGLU-MoE`。
- Attention 层以 `Local FlashAttention-3 Attention` 为唯一 sequence mixer 主干，后端策略为 `FlashAttention-3 -> FlashAttention-2 -> SDPA`。
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
