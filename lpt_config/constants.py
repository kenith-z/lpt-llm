"""LPT v2 共享默认值与项目路径。

本文件只保存跨模块复用的稳定默认值，不承载运行时状态。
训练 recipe、评测配置、推理 smoke 与 workflow 入口都从这里读取默认值，
这样可以避免同一个常量在多个文件中重复定义而产生漂移。
"""

from __future__ import annotations

from pathlib import Path


# 仓库根目录；所有相对数据路径和产物路径都以它作为可解释锚点。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 默认 tokenizer 目录；当前保留 DS tokenizer 资产位置，供 workflow 和推理入口统一引用。
TOKENIZER_PATH = PROJECT_ROOT / "lpt_model" / "ds_tokenizer"
# LPT v2 训练、评测、checkpoint 与报告的默认产物根目录。
ARTIFACT_ROOT_DIR = PROJECT_ROOT / "artifacts" / "lpt_v2"

# 模型短名，用于 CLI 输出、报告标题和 checkpoint metadata 的人类可读标识。
MODEL_ABBR = "LPTv2"
# 模型英文全名，用于跨语言报告和 artifact 元信息。
MODEL_NAME_EN = "Ling Pre-trained Transformer v2"
# 模型中文全名，用于中文文档、控制台输出和报告。
MODEL_NAME_ZH = "灵预 v2"
# chat 模板版本号；变更该值意味着训练样本渲染语义发生变化。
CHAT_TEMPLATE_VERSION = "lpt-ds-v1"

# 最小开发规格：用于 CPU smoke、单元测试和快速验证结构连通性。
LPT_V2_DEV_TINY_PRESET = "lpt_v2_dev_tiny"
# 小规格：保留多层、多 expert 形态，用于低成本实验。
LPT_V2_SMALL_PRESET = "lpt_v2_small"
# 小型主训练规格：当前默认主线规格，兼顾可训练性和资源占用。
LPT_V2_SMALL_BASE_PRESET = "lpt_v2_small_base"
# base 规格：用于较完整的正式实验和扩大参数量验证。
LPT_V2_BASE_PRESET = "lpt_v2_base"
# large 规格：用于后续放大验证，不作为默认 smoke 目标。
LPT_V2_LARGE_PRESET = "lpt_v2_large"
# 默认规格必须与 README/help 中的 v2 主线保持一致。
DEFAULT_MODEL_SIZE_PRESET = LPT_V2_SMALL_BASE_PRESET

# bootstrap profile：最小可运行主干，用于排除 MoE/状态辅助模块之外的基础问题。
LPT_V2_BOOTSTRAP_PROFILE = "lpt_v2_bootstrap"
# SDPA local profile：使用 PyTorch SDPA 和 dense KV，验证局部 attention 基线。
LPT_V2_SDPA_LOCAL_PROFILE = "lpt_v2_sdpa_local"
# Paged KV profile：打开 paged KV，但关闭辅助状态，用于隔离缓存后端影响。
LPT_V2_PAGED_KV_PROFILE = "lpt_v2_paged_kv"
# Assist profile：默认 profile，启用 Paged KV 与 RetNetAssist。
LPT_V2_ASSIST_PROFILE = "lpt_v2_assist"
# Base profile：启用更接近正式规格的 MoE/RetNetAssist 组合。
LPT_V2_BASE_PROFILE = "lpt_v2_base"
# Memory profile：在 base profile 基础上启用 xLSTMAssist。
LPT_V2_MEMORY_PROFILE = "lpt_v2_memory"
# workflow 和评测未显式指定 profile 时使用的默认 profile。
DEFAULT_PROFILE = LPT_V2_ASSIST_PROFILE
# 基线评测按该顺序展开 profile，保证报告中横向对照顺序稳定。
LPT_V2_BASELINE_PROFILES = (
    LPT_V2_BOOTSTRAP_PROFILE,
    LPT_V2_SDPA_LOCAL_PROFILE,
    LPT_V2_PAGED_KV_PROFILE,
    LPT_V2_ASSIST_PROFILE,
    LPT_V2_BASE_PROFILE,
    LPT_V2_MEMORY_PROFILE,
)

# 设备默认值；auto 表示由运行入口按 CUDA 可用性和 CLI 覆盖项解析。
DEFAULT_DEVICE = "auto"
# dtype 默认值；auto 表示由运行入口选择适合当前设备的参数/计算 dtype。
DEFAULT_DTYPE = "auto"
# 通用训练阶段名，具体三阶段 recipe 会覆盖为 text_pretrain/chat_sft/chat_lora。
DEFAULT_TRAINING_STAGE = "train"
# 通用 batch size 默认值；阶段 recipe 会按任务覆盖。
DEFAULT_TRAINING_BATCH_SIZE = 1
# 通用 epoch 默认值，保持 smoke 和小样本训练不会意外长跑。
DEFAULT_TRAINING_EPOCHS = 1
# None 表示不额外限制训练步数，由数据集大小和 epoch 决定。
DEFAULT_TRAINING_MAX_STEPS = None
# 通用学习率默认值；阶段 recipe 对不同训练阶段使用更贴合的学习率。
DEFAULT_TRAINING_LEARNING_RATE = 1e-4
# AdamW weight decay 默认值；optimizer 会自动排除 norm/bias/低维参数。
DEFAULT_WEIGHT_DECAY = 0.1
# 梯度累积步数默认值；1 表示每个 batch 都执行一次 optimizer step。
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
# 梯度裁剪阈值，防止小模型/长序列 smoke 中出现异常梯度峰值。
DEFAULT_MAX_GRAD_NORM = 1.0
# warmup 占总训练步数比例，训练步数可估计时用于构造 warmup+cosine schedule。
DEFAULT_WARMUP_RATIO = 0.03
# 固定随机种子，保证 smoke test、基线报告和窗口采样可复现。
DEFAULT_TRAINING_SEED = 20260503
# 默认启用 deterministic algorithms，优先保证回归测试稳定。
DEFAULT_DETERMINISTIC_ALGORITHMS = True
# 默认启用 sequence packing，提高短样本训练的 token 利用率。
DEFAULT_SEQUENCE_PACKING_ENABLED = True
# 控制台/JSONL/TensorBoard 训练指标写入间隔。
DEFAULT_LOG_INTERVAL_STEPS = 10
# 0 表示默认不做周期性 eval，避免无 eval manifest 时入口误跑。
DEFAULT_EVAL_INTERVAL_STEPS = 0
# None 表示验证时不截断验证 batch 数；recipe/CLI 可按成本覆盖。
DEFAULT_EVAL_MAX_BATCHES = None
# 0 表示默认不保存 step_N checkpoint，只保存 latest。
DEFAULT_SAVE_INTERVAL_STEPS = 0
# latest checkpoint 保存间隔，用于中断后恢复训练。
DEFAULT_LATEST_SAVE_INTERVAL_STEPS = 10
# 默认保存 optimizer，保证 latest checkpoint 可真正续训。
DEFAULT_SAVE_OPTIMIZER = True
# 默认保存 scheduler，保证 resume 后学习率曲线连续。
DEFAULT_SAVE_SCHEDULER = True
# 默认不保存 best checkpoint，避免 smoke 场景产生额外目录。
DEFAULT_SAVE_BEST_CHECKPOINT = False
# best checkpoint 默认以训练 loss 做最小化指标。
DEFAULT_BEST_CHECKPOINT_METRIC = "loss"
# best checkpoint 判优的最小改善幅度，0 表示只要严格更小就保存。
DEFAULT_BEST_CHECKPOINT_MIN_DELTA = 0.0
# 默认启用 TensorBoard；缺依赖或写入失败时训练循环会自动降级。
DEFAULT_TENSORBOARD_ENABLED = True

# text_pretrain 阶段默认数据 manifest，相对路径便于跨平台迁移仓库。
TEXT_PRETRAIN_MANIFEST_PATH = Path("data/manifests/text_pretrain.json")
# chat_sft 阶段默认数据 manifest。
CHAT_SFT_MANIFEST_PATH = Path("data/manifests/chat_sft.json")
# chat_lora 阶段默认数据 manifest。
CHAT_LORA_MANIFEST_PATH = Path("data/manifests/chat_lora.json")
# text_pretrain 阶段默认产物目录。
TEXT_PRETRAIN_ARTIFACT_DIR = ARTIFACT_ROOT_DIR / "text_pretrain"
# chat_sft 阶段默认产物目录。
CHAT_SFT_ARTIFACT_DIR = ARTIFACT_ROOT_DIR / "chat_sft"
# chat_lora 阶段默认产物目录。
CHAT_LORA_ARTIFACT_DIR = ARTIFACT_ROOT_DIR / "chat_lora"

# text_pretrain 默认 batch size，作为 recipe 默认值，CLI 可覆盖。
TEXT_PRETRAIN_BATCH_SIZE = 2
# text_pretrain 默认 epoch 数。
TEXT_PRETRAIN_EPOCHS = 1
# text_pretrain 默认学习率。
TEXT_PRETRAIN_LEARNING_RATE = 3e-4
# text_pretrain 默认 warmup 比例。
TEXT_PRETRAIN_WARMUP_RATIO = 0.1

# chat_sft 默认 batch size。
CHAT_SFT_BATCH_SIZE = 1
# chat_sft 默认 epoch 数。
CHAT_SFT_EPOCHS = 1
# chat_sft 默认学习率。
CHAT_SFT_LEARNING_RATE = 3e-4
# chat_sft 默认 warmup 比例。
CHAT_SFT_WARMUP_RATIO = 0.1

# chat_lora 默认 batch size；LoRA 只训练 adapter，通常可用更大的 batch。
CHAT_LORA_BATCH_SIZE = 10
# chat_lora 默认 epoch 数。
CHAT_LORA_EPOCHS = 1
# chat_lora 默认学习率；adapter 参数量小，默认学习率高于全量训练。
CHAT_LORA_LEARNING_RATE = 6e-4
# chat_lora 默认 warmup 比例。
CHAT_LORA_WARMUP_RATIO = 0.1
# LoRA 默认基座来源；text_pretrain 表示从预训练权重挂 adapter。
DEFAULT_LORA_BASE_SOURCE = "text_pretrain"
# 允许的 LoRA 基座来源，防止从不兼容 checkpoint 链路挂载 adapter。
VALID_LORA_BASE_SOURCES = ("text_pretrain", "chat_sft")
# LoRA 低秩矩阵 rank。
DEFAULT_LORA_RANK = 8
# LoRA 缩放系数 alpha。
DEFAULT_LORA_ALPHA = 16.0
# LoRA adapter dropout，降低小样本 SFT 过拟合风险。
DEFAULT_LORA_DROPOUT = 0.05
# 默认注入 LoRA 的 attention 线性层集合。
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")

# 默认启用采样；关闭时推理入口会走贪心解码。
DEFAULT_GENERATION_DO_SAMPLE = True
# softmax temperature，越低越接近确定性输出。
DEFAULT_GENERATION_TEMPERATURE = 0.7
# top-k 采样保留的候选 token 数。
DEFAULT_GENERATION_TOP_K = 50
# top-p nucleus 采样累计概率阈值。
DEFAULT_GENERATION_TOP_P = 0.9
# 兼容旧入口的生成总长度默认值；新入口优先使用 max_new_tokens。
DEFAULT_GENERATION_MAX_LENGTH = 200
# 重复惩罚系数，1.0 表示不惩罚。
DEFAULT_GENERATION_REPETITION_PENALTY = 1.1
# 重复惩罚只检查最近多少个 token，避免长上下文下成本线性扩大过多。
DEFAULT_GENERATION_REPETITION_WINDOW_SIZE = 256

# 评测/smoke 默认词表大小；使用小词表降低 CPU 回归成本。
DEFAULT_EVAL_VOCAB_SIZE = 128
# 评测/smoke 默认 batch size。
DEFAULT_EVAL_BATCH_SIZE = 1
# 评测/smoke 默认序列长度。
DEFAULT_EVAL_SEQUENCE_LENGTH = 16
# 基线 decode smoke 默认生成步数。
DEFAULT_EVAL_DECODE_STEPS = 1
# 资源评测默认 decode 步数，略大于 smoke 以暴露缓存资源变化。
DEFAULT_RESOURCE_DECODE_STEPS = 4
# xLSTMAssist memory 评测默认序列长度，保持状态观测成本很低。
DEFAULT_MEMORY_EVAL_SEQUENCE_LENGTH = 8
# 长上下文评测默认 attention 窗口，用于 tiny/smoke 场景。
DEFAULT_LONG_CONTEXT_ATTENTION_WINDOW_SIZE = 8
# token-id 推理 smoke 的默认 prompt。
DEFAULT_INFERENCE_PROMPT_IDS = "1,2,3"
# token-id 推理 smoke 默认新增 token 数。
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 8
# 聊天推理默认模型来源；chat_sft 是三阶段链路中的对话全量权重。
DEFAULT_CHAT_MODEL_SOURCE = "chat_sft"
