# LPT v2 模型定型方案变动记录

本文只记录 LPT v2 定型方案的历史变动。`help/任务清单.md` 负责范围、状态和优先级，`help/LPTv2模型定型方案.md` 负责技术契约与验收语义；两者冲突时先停止实现并完成文档裁决，不能由本历史记录覆盖当前事实。

## 2026-08-03

- 冻结停止串首版匹配语义：使用最终可见 answer 文本的 Unicode 标量值序列精确匹配，不隐式执行 NFC/NFKC、大小写折叠或空白规整；停止串长度与 holdback 容量统一按 Unicode 标量值计量，byte pending 独立处理，并补多字节、辅助平面、组合字符和规范等价反例准入。
- 补齐 GPU lease 永久隔离后的恢复边界：只有确定性销毁旧执行上下文或进程重启后才能建立新的空页池代次，全部陈旧引用失效；P3 负责页池重建与启动自检，P2 负责 readiness、请求终态和显式重放。同步将纯离线 DPO 的正式模型前向与 logprob 契约写入任务清单，区分策略/参考模型训练 logprob 与在线采样 raw/behavior logprob，并把 `lm_head_chunk_size` 的恢复校验收敛为默认防漂移、等价准入后允许显式覆盖。

## 2026-08-02

- 重新核查 CLA 当前实现：`cla_share_every_n_layers` 仅为禁用态保留字段，`ModelConfig` 强制其为 1 并拒绝 2；`LPTV2` 各层独立创建 `k_proj/v_proj`，不存在模型消费点或跨层 K/V 复用。当前只有拒绝 CLA 启用值的配置测试，没有启用路径的模型、缓存/checkpoint 测试或实验报告。因此 CLA 保持“未实现、未评估”，不得因字段存在而标记完成。
- 裁决新旧实验原则：22-27 已完成单项分支只作为 2026-08-02 前的历史归因，不追溯改写；后续新增或尚未启动的兼容候选采用组合优先统一测速。同步把任务清单中已完成的 24-27 状态改回 `[x]`，CLA 保持 `[ ]` 并补实现、缓存语义和组合准入边界。
- 为组合优先原则补齐 P2 第 10 项治理载体：新增版本化候选包 manifest、统一基准套件 manifest，以及 run metadata 对两者 id、版本和摘要的强绑定与 fail-closed 校验。
- 根据审计结果，将 P1.5 改为“训练工程与显存优化专项”，把 7.4 定为单机及通用 trainer-state、提交边界和 RNG 语义事实源，P2 第 8 项只扩展 rank/world size/shard/sampler 与全 rank 共识；P2 同时负责 Attention execution plan 控制面，P3-KV2 负责分页描述、block table 和 direct paged decode 数据面。
- 将近期优先级补入 7.4 训练提交正确性和 7.5 纯文本拒绝账本；澄清 P2 单请求采样/logprob 与 P3 逐 row 扩展的继承关系，并给已完成的流式事件通道补充“增量解码与终态正确性仍未完成”的状态说明。
- 冻结“成熟兼容项先组合、统一测速、回归后定向拆分”的技术采用原则：正确性与恢复约束直接实施，不做收益消融；训练稳定性、架构/运行时和多模态交互分别按依赖形成版本化候选包，不做全排列或全因子实验。KV0-KV5 依赖门、从头初始化与 checkpoint 派生边界、互斥结构、目标硬件低精度验证和后训练阶段仍独立守门；已完成的历史单项实验不追溯改写。
- 明确候选技术规范来源为带版本标识的公开论文、官方技术文档/模型卡或许可允许且经审计的公开实现；P2 实验元数据登记 `spec_id / source_type / source_locator / source_revision / license_id`，保证正式方案与任务不依赖参考项目或过程材料。
- 根据技术复核结果，将纯文本路径的 Attention 执行计划、KV0-KV5 分页数据面、generation/执行租约、事务回滚、continuous batching、逐请求采样、终态与 logprob 口径、训练提交边界、完整 RNG/manifest 摘要和 P4 组合优先准入矩阵写入定型方案；这些约束使用 LPT 自有术语，不改变当前默认主干或任务完成状态。
- 同步任务清单 P1.5/P2/P3/P4：增加训练游标与步数提交顺序、批量分词拒绝账本、WSD/稳定性组合准入、预分词与纯张量导出条件专项、KV0-KV5 准入、后端 fail-closed、MTP 草稿事务、thinking 状态重建和离线 DPO 条件任务。
- 明确版本治理：外层 `checkpoint_schema_version=2` 保持不变，纯文本到图像多模态只升级 `model_config_schema_version=3 -> 4`，并使用 `multimodal_schema_version=1`；正式任务状态仍以 `help/任务清单.md` 为唯一来源。
- 完成多模态技术复核：确认现有图像 MVP 架构已覆盖值得保留的感知前端、连接层、分阶段对齐、语义/渲染解耦等通用模式，不引入新的视觉塔、语言主干或外部运行时依赖。
- 补充结构化模态分发、有效媒体行压实执行、禁止零媒体/零值伪梯度、离线坏样本 rejection ledger、request-level 预算预检与失败回滚、单生命周期终态唯一/幂等、未来多模态 prefix sharing 完整身份和多图绑定/顺序反事实准入。
- 收紧音频理解、语音渲染与实时会话的未来边界：原生音频能力声明项不以强制转写为唯一前置，LPT 暴露版本化 answer-channel 语义状态供 renderer 消费，声学流拓扑保持 codec 私有，同时区分文本生成结束、renderer/codec 产出结束和客户端播放确认三个水位；这些内容不进入当前 P5 配置、阶段或完成度。
- 冻结多模态能力声明规则：只登记有限、具备实际用途和依赖关系的能力声明项，并分别记录输入/输出组合、交互模式、原生或显式级联路径及 schema/数据/训练/评测/运行时/发布状态；能力声明项与模型配置档“运行 Profile”严格区分。单组件前向、训练样本或演示结果不能推导未登记能力，也不执行无边界的模态全排列实验。
- 补齐结构化音频输入与能力保护边界：未来 schema 必须记录媒体身份、轮次/part 绑定、有效时间、帧 mask、时间基准和融合 span；输入规范化覆盖容器/编码、声道、采样率、重采样、有限值、削波、资源上限、内容摘要和样本到特征帧映射。新增能力声明项采用隔离对齐、低学习率联合训练和全部既有能力 replay，文本/声学/终止目标绑定同一 answer span 并分别计量。
- 补齐语音 renderer 正确性与复现边界：多声学分量只在完整可播放帧就绪后原子发布；流式解码必须保证时间区间单调、无重复/缺口、尾块至多一次 flush，并与整段解码通过一致性准入。renderer trace 绑定 bridge、renderer/codec revision、私有调度计划、各输出采样配置、seed/RNG 和分块连续性策略，不把固定流数、码本或延迟表写入 LPT 主干。
- 补齐实时轮次事务和跨模态准入：异步图音资产先绑定 request generation/turn 并原子提交，barge-in 输入必须取消旧输出且恰好一次进入下一轮；端点基线登记 pre-roll、起止驻留/迟滞、尾部策略和 reset，不把端点后整段 prefill 宣称为流式音频编码或全双工。图像到语音分别评测媒体答案正确性和语音忠实度，混合图音输入使用逐模态替换/删除反事实；图像 MVP 同步增加合法同轮布局鲁棒性和完整轮次子样本约束。
- 同步任务清单新增“P5 后续图音能力立项前置条件（非当前任务）”，并明确技术约束以定型方案对应章节为唯一事实源；任务清单只保留未来任务拆分、依赖和准入的浓缩摘要，不新增复选框、任务编号、配置、recipe、优先级或进度。当前已立项范围仍只有 P5 第 14 项图像输入、文本输出闭环。
- 修正固定媒体 KV 生命周期表中的页数口径：有效媒体 token 数与物理 page 数分开统计，page 数按 `page_block_size` 向上取整，并禁止 anchor/rolling 混页。本次不改变 P5 任务范围、优先级或完成状态。
- 复核并保留图像 MVP 的视觉 token 预算不变量；将 `kv_cache_scope` 收紧为由 schema、多模态开关和媒体策略共同决定的派生不变量，补齐 schema v4 `local_window_only`/多模态关闭分支，并要求序列化配置与 checkpoint 载入不一致时 fail closed。P4 仍以独立评估矩阵、准入报告和可回滚配置为产物，不改成模型配置字段。
- 为定型方案正文补齐技术栈复核的四类交叉落点：P4 候选类型与最低准入指标映射，服务层 OpenAI/Anthropic/SSE 协议适配边界，first-fit/BFD packing 对照，以及复用正式推理核心的 DPO/GRPO、KL、轨迹账本与回滚契约；不新增任务编号、配置字段或完成状态。
- 补强纯文本工程约束的可实施与可验收口径：rejection ledger 改为版本化可机读 schema，并冻结集合守恒代码断言、拒绝策略及阈值的配置与元数据落点；增量 decoder 明确生命周期状态、停止串 holdback 上界和独立 byte 缓冲；请求级资源预留与 GPU 执行 lease 分离，超时资源必须等待设备完成证明后回收；chunked LM loss 增加 recipe/CLI 参数归属及 vocab-shard 等价准入。同步要求 signal/异常/结束回调不得直接发布 checkpoint、raw/behavior logprob 完成独立 reference 测试、后训练 rollout 复用正式推理核心；不新增任务编号、模型配置字段或完成状态。
- 进一步冻结训练与资源安全边界：`lm_head_chunk_size` 写入 checkpoint training_config/run metadata 并参与 resume 配置核对，TP 有效目标数不得随 TP world size 重复放大，fp16 scaler 与 dtype fallback 必须可恢复、可审计；GPU lease 明确所有权转移、状态迁移、设备完成证明和延迟回收，多模态 reset/error/timeout 不得绕过该约束。后训练推理核心复用要求只约束实际生成 rollout 的路径，纯离线 DPO 复用正式模型前向和统一 token/mask/logprob 契约，但不被强制生成 rollout。

## 2026-08-01

- 完成多模态方案双重审计（Claude Fable 5 + DeepSeek）并采纳关键建议：`inputs_embeds` 路径设计与等价测试、训练数据规划、依赖资产清单、预算 smoke 与早期基线、NaFlex 布局契约、双区 Paged KV 页分类器抽象建议。
- MM0 契约冻结新增 4 项技术交付物：(1) `inputs_embeds` 路径设计与等价测试计划，（2）训练数据规划，（3）依赖资产清单与可复现性要求，（4）可配置视觉预算的 smoke 与早期基线。具体样本量、任务配比、依赖版本、预算档位和文件组织由 MM0 实施方案确定。
- 配置字段补充 `kv_cache_scope` 新值：`model_config_schema_version=3` 纯文本保持 `"local_real_tokens_only"`，`model_config_schema_version=4` 多模态启用 `pinned_media_kv` 时切换为 `"local_real_tokens_plus_bounded_media"`，外层 `checkpoint_schema_version=2` 不变，需放行 `model_config.py` 硬校验。
- NaFlex reducer 补充布局契约：确认对应的 SigLIP2-NaFlex 图像 processor 直接返回 `pixel_values / pixel_attention_mask / spatial_shapes`，方案改为直接消费并校验，MM1 用真实 processor 验证三字段对应关系并明确写入 `EncodedMediaBatch` 契约。
- 多模态方案复核修正（针对第二次审计发现的问题）：xLSTM special-token reset 改为双路径规则（`inputs_embeds` 路径必须显式传 `memory_reset_mask`，`input_ids` 路径未传时继续从边界 token 推导，保持 v2 纯文本行为）；multimodal SFT 默认只训 projector 与 LPT LoRA/白名单低学习率参数，解冻 LPT 全参数降级为 Post-MVP 独立实验；删除未经验证的依赖版本下限和视觉预算近似结论，具体依赖与预算由 MM0 实施方案确定；任务清单明确仅加载视觉子模型（不加载 SigLIP2 文本塔）；完成报告同步修正乐观结论。
- 多模态方案文档层级收口（第三次复核）：`MM0 契约冻结详细交付物` 压缩为五类方向性交付物，具体样本量、任务配比、版本号、预算档位、文件/脚本路径和扫描命令全部下放至 MM0 实施方案；删除 `vision_encoder` 伪代码示例（由实施方案定义接口细节）；xLSTM reset 双路径冲突策略改为 fail closed（不一致直接报错）；视觉加载表述收敛为"仅加载视觉子模型 + 配套图像 processor"，不保留完整双塔备选；两份历史审核记录文档（`ds审核结论`、`DeepSeek审计建议采纳分析`）的旧结论已确认被后续复核覆盖，相关文档已清理，不再保留。
- 任务清单 P5 第 14.1/MM0 同步新增 5 项交付物，确保方案与任务完全对齐。
- 完成多模态方案审计复核：确认视觉 reducer、两区 Paged KV 生命周期、`model_config_schema_version=3 -> 4` 的显式派生、外层 `checkpoint_schema_version=2` 保持不变、InferenceSession 媒体字段、MM0-MM4 完成定义、长图文场景和量化准入属于有效补缺项，并同步到主方案与 P5 任务清单。
- 明确 `google/siglip2-so400m-patch16-naflex` 的 1152 专指 SigLIP2 SO400M Patch16 NaFlex 视觉编码器输出的特征嵌入维度；加载时必须与实际 encoder config 严格校验，projector 再映射到 LPT 语言主干的 `hidden_size`。
- 审计建议中，纯文本 chat 强制转换为 `multimodal_chat` 不纳入正式任务，纯文本 replay 继续使用现有 `chat` schema；质量和成本门槛不写死为跨数据集固定百分比，改为 MM0 在 eval manifest 中登记有限数值阈值、比较方向和统计方法。

## 2026-05-14

- 完成第 27 项 `RetNetContextAdapter` 两分支 1164 step chat SFT 对比，报告路径为 `help/LPTv2扩展实验/27_context_adapter/LPTv2_27_context_adapter_实验报告.md`；`base_continued` 与 `exp_27_context_adapter` 均通过 checkpoint validate、真实 checkpoint forward smoke、4100 长上下文代理和资源评测。ContextAdapter 分支长上下文代理收益明确，但 chat SFT eval loss 退化且吞吐下降，初始自动结论为“暂不进入默认主干或默认组合实验，仅保留为长上下文专项候选”。
- 第 27 项结论经人工介入后修订：用户判断长上下文效果好是必要条件，且本轮小样本 eval loss 小幅上升不足以否定机制收益，因此将 `RetNetContextAdapter` 从“仅保留为长上下文专项候选”调整为“长上下文主候选，进入组合实验”；若后续 `global/per_layer + every_4_layers/rank16 + RetNetContextAdapter` 组合分支相对无 ContextAdapter 对照不劣化，则直接进入主干。

## 2026-05-13

- 完成第 27 项 `RetNetContextAdapter` 的代码与工具准备：新增轻量 `RetNetContextAdapter`，复用 `SharedRetNetAssist` 的 `summary_sequence` 在 Attention 输出投影后做低秩残差注入，不新增状态池、不写入 Paged KV；新增 `context_adapter_delta_norm`、`alpha_context` 观测指标、参数统计、checkpoint schema metadata、`tools/init_lpt_v2_exp27_branches.py`、`data/manifests/chat_sft_eval_exp27.json` 和 `help/LPTv2扩展实验/27_context_adapter/LPTv2_27_context_adapter_实验报告.md`；训练仍由人工按报告命令执行。
- 完成第 26 项 `RetNetAssist 启用层与 rank` 五分支 1164 step chat SFT 对比，报告路径为 `help/LPTv2扩展实验/26_retnet_layers_rank/LPTv2_26_retnet_layers_rank_实验报告.md`；`base_continued(all_layers/rank16)`、`exp_26_retnet_every_2_layers`、`exp_26_retnet_every_4_layers`、`exp_26_retnet_selected_offset_layers`、`exp_26_retnet_rank32` 均通过 checkpoint validate、真实 checkpoint forward smoke、4100 长上下文代理和资源评测。
- 第 26 项结论调整为：`every_4_layers/rank16` 的 4100 long loss 最低、训练吞吐最高且训练显存峰值最低，作为后续组合实验主候选；`every_2_layers/rank16` 的 eval loss 最低，但长上下文代理退化，保留为质量对照/备选；`rank32` 未带来 eval 收益，不进入默认主干。
- 保留组合限制：第 26 项为单项归因，仍保持 `global/group` 共享策略；后续组合实验优先验证第 25 项 `global/per_layer` 与第 26 项 `every_4_layers` 的交互效果，并保留 `every_2_layers` 作为质量对照。

## 2026-05-12

- 完成第 25 项 `RetNetAssist 参数与状态共享策略` 四分支 1164 step chat SFT 对比，报告路径为 `help/LPTv2扩展实验/25_retnet_sharing/LPTv2_25_retnet_sharing_实验报告.md`；`base_continued(global/group)`、`exp_25_global_per_layer`、`exp_25_group_group`、`exp_25_per_layer_per_layer` 均通过 checkpoint validate、真实 checkpoint forward smoke、4100 长上下文代理和资源评测。
- 第 25 项结论调整为：`global/per_layer` 的 eval loss 最低，且不增加 RetNet 物理参数，runtime state bytes 绝对值很小，作为后续组合实验中的 RetNetAssist 共享策略候选；`global/group` 保留为低状态成本 fallback，`group/group` 和 `per_layer/per_layer` 不进入默认主干。
- 保留长上下文限制：四个分支 4100 长上下文评测均跨过 2048 窗口并通过机制准入，但质量代理没有单一完全胜者，因此第 25 项不单独作为长上下文定型充分证据，需等待第 26/27 项及组合实验联合确认。
- 完成第 26 项 `RetNetAssist 启用层与 rank` 的代码与工具准备：新增 `retnet_assist_selected_layers`、RetNet enabled-layer 统计、稀疏启用层不挂载 RetNetAssist 参数、rank32 初始化权重过滤、`tools/init_lpt_v2_exp26_branches.py`、`data/manifests/chat_sft_eval_exp26.json` 和 `help/LPTv2扩展实验/26_retnet_layers_rank/LPTv2_26_retnet_layers_rank_实验报告.md`；训练仍由人工按报告命令执行。

## 2026-05-11

- 完成第 25 项 `RetNetAssist 参数与状态共享策略` 的代码与工具准备：新增真实 `retnet_parameter_sharing="global|group|per_layer"`、`retnet_state_sharing="group|per_layer"` 和 `retnet_sharing_group_size=4` 接入，RetNet state pool 改为按 `state_slot` 存取；新增 `tools/init_lpt_v2_exp25_branches.py`、`data/manifests/chat_sft_eval_exp25.json` 和 `help/LPTv2扩展实验/25_retnet_sharing/LPTv2_25_retnet_sharing_实验报告.md`。
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
