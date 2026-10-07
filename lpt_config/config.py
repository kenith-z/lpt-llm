"""LPT v2 项目级运行配置。"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .constants import (
    ARTIFACT_ROOT_DIR,
    CHAT_LORA_ARTIFACT_DIR,
    CHAT_LORA_TOKEN_CACHE_ENABLED,
    CHAT_LORA_EPOCHS,
    CHAT_LORA_LEARNING_RATE,
    CHAT_LORA_MANIFEST_PATH,
    CHAT_LORA_WARMUP_RATIO,
    CHAT_SFT_ARTIFACT_DIR,
    CHAT_SFT_TOKEN_CACHE_ENABLED,
    CHAT_SFT_EPOCHS,
    CHAT_SFT_LEARNING_RATE,
    CHAT_SFT_MANIFEST_PATH,
    CHAT_SFT_WARMUP_RATIO,
    CHAT_TEMPLATE_VERSION,
    DEFAULT_DATA_NUM_WORKERS,
    DEFAULT_DATA_READER_CHUNK_BYTES,
    DEFAULT_DATA_READER_NUM_WORKERS,
    DEFAULT_DATA_LENGTH_BUCKET_WINDOW,
    DEFAULT_DATA_PACKING_STRATEGY,
    DEFAULT_DATA_PERSISTENT_WORKERS,
    DEFAULT_DATA_PIN_MEMORY,
    DEFAULT_DATA_PREFETCH_FACTOR,
    DEFAULT_TOKEN_CACHE_DIR,
    DEFAULT_TOKEN_CACHE_ENABLED,
    DEFAULT_DETERMINISTIC_ALGORITHMS,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_DECODE_STEPS,
    DEFAULT_EVAL_INTERVAL_STEPS,
    DEFAULT_EVAL_MAX_BATCHES,
    DEFAULT_EVAL_SEQUENCE_LENGTH,
    DEFAULT_EVAL_VOCAB_SIZE,
    DEFAULT_GENERATION_DO_SAMPLE,
    DEFAULT_GENERATION_MAX_LENGTH,
    DEFAULT_GENERATION_REPETITION_PENALTY,
    DEFAULT_GENERATION_REPETITION_WINDOW_SIZE,
    DEFAULT_GENERATION_TEMPERATURE,
    DEFAULT_GENERATION_TOP_K,
    DEFAULT_GENERATION_TOP_P,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_INFERENCE_MAX_NEW_TOKENS,
    DEFAULT_INFERENCE_PROMPT_IDS,
    DEFAULT_LATEST_SAVE_INTERVAL_STEPS,
    DEFAULT_LOG_INTERVAL_STEPS,
    DEFAULT_LONG_CONTEXT_ATTENTION_WINDOW_SIZE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_BASE_SOURCE,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_LM_LOSS_CHUNK_TOKENS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MEMORY_EVAL_SEQUENCE_LENGTH,
    DEFAULT_MODEL_SIZE_PRESET,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_PROFILE,
    DEFAULT_RESOURCE_DECODE_STEPS,
    DEFAULT_BEST_CHECKPOINT_METRIC,
    DEFAULT_BEST_CHECKPOINT_MIN_DELTA,
    DEFAULT_SAVE_INTERVAL_STEPS,
    DEFAULT_SAVE_BEST_CHECKPOINT,
    DEFAULT_SAVE_OPTIMIZER,
    DEFAULT_SAVE_SCHEDULER,
    DEFAULT_SEQUENCE_PACKING_ENABLED,
    DEFAULT_TENSORBOARD_ENABLED,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_MAX_STEPS,
    DEFAULT_TRAINING_SEED,
    DEFAULT_TRAINING_STAGE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    MODEL_ABBR,
    MODEL_NAME_EN,
    MODEL_NAME_ZH,
    TEXT_PRETRAIN_ARTIFACT_DIR,
    TEXT_PRETRAIN_EPOCHS,
    TEXT_PRETRAIN_LEARNING_RATE,
    TEXT_PRETRAIN_MANIFEST_PATH,
    TEXT_PRETRAIN_TOKEN_CACHE_ENABLED,
    TEXT_PRETRAIN_WARMUP_RATIO,
)


class GlobalConfig:
    """全局运行配置。

    这里使用类属性，便于模型、数据批处理和测试在同一进程内共享运行参数。
    """

    # 模型短名，统一用于日志、报告和 checkpoint metadata。
    model_abbr = MODEL_ABBR
    # 模型英文名称，用于英文报告或跨语言产物描述。
    model_name_en = MODEL_NAME_EN
    # 模型中文名称，用于中文 CLI 输出和文档。
    model_name_zh = MODEL_NAME_ZH
    # chat 模板版本，训练与推理必须保持一致才能正确解释角色边界。
    chat_template_version = CHAT_TEMPLATE_VERSION

    # 训练默认最大序列长度；阶段 recipe 或 CLI 可按 LongRoPE2 训练策略覆盖。
    train_max_sequence_length = 4096
    # 推理默认最大序列长度；当前按 32K 设置，避免默认 smoke 意外申请 64K 缓存。
    inference_max_sequence_length = 65536 // 2
    # 训练 RoPE cache 默认覆盖训练最大长度。
    train_rope_cache_max_sequence_length = train_max_sequence_length
    # 推理 RoPE cache 默认覆盖推理最大长度。
    inference_rope_cache_max_sequence_length = inference_max_sequence_length

    # 模型参数默认 dtype；训练入口可以按配置切换到 bf16/fp16。
    parameter_dtype = torch.float32
    # autocast 默认 dtype；CUDA 下训练/评测入口会读取该值。
    autocast_dtype = torch.bfloat16
    # batch padding 对齐粒度，便于 tensor core 和 packing 后张量形状稳定。
    pad_to_multiple_of = 8
    # 流式数据集 shuffle 缓冲区大小，控制随机性与内存占用的折中。
    dataset_shuffle_buffer_size = 2048
    # CUDA 可见设备由启动进程的环境变量决定，导入配置模块时不得覆盖外部选择。
    # 当前进程默认 torch device，由 CUDA 可用性自动决定。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GenerationConfig:
    """文本生成采样配置；v2 推理入口接入前保持为轻量数据结构。"""

    # 是否使用随机采样；False 时推理入口走 argmax 贪心解码。
    do_sample: bool = DEFAULT_GENERATION_DO_SAMPLE
    # 采样温度；值越低输出越保守，值越高随机性越强。
    temperature: float = DEFAULT_GENERATION_TEMPERATURE
    # top-k 候选数量；0 或负数在推理逻辑中通常表示不启用 top-k。
    top_k: int = DEFAULT_GENERATION_TOP_K
    # top-p nucleus 累计概率阈值。
    top_p: float = DEFAULT_GENERATION_TOP_P
    # 兼容旧接口的最大总长度字段。
    max_length: int = DEFAULT_GENERATION_MAX_LENGTH
    # 重复惩罚系数；大于 1 时会降低近期已出现 token 的概率。
    repetition_penalty: float = DEFAULT_GENERATION_REPETITION_PENALTY
    # 重复惩罚窗口大小，限制只扫描最近 token 以控制长上下文成本。
    repetition_window_size: int | None = DEFAULT_GENERATION_REPETITION_WINDOW_SIZE
    # 原生 thinking 模式；off 强制不生成 thinking，on 生成 thinking+answer，auto 交给策略解析。
    thinking_mode: str = "off"
    # thinking 可见性；visible 返回思考链，hidden 只返回最终回答。
    thinking_visibility: str = "hidden"
    # thinking=on/auto 时最多生成的思考 token 数。
    max_thinking_tokens: int = 128
    # 输出格式：text 为普通文本，json 启用 JSON 语法约束，tool_call 启用工具调用结构解析。
    output_format: str = "text"
    # 工具调用策略：none 不解析工具调用，auto/required 解析模型输出中的 tool_calls。
    tool_choice: str = "none"


@dataclass(frozen=True)
class BaseTrainingRecipeConfig:
    """v2 阶段训练默认 recipe。

    训练入口以这个配置为主，CLI/测试注入参数只覆盖明确传入的字段。
    """

    # 当前训练阶段名，会写入 trainer_state 和 checkpoint metadata。
    training_stage: str = DEFAULT_TRAINING_STAGE
    # 构造 ModelConfig 时使用的运行 profile。
    profile: str = DEFAULT_PROFILE
    # 构造 ModelConfig 时使用的模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 目标设备；auto 由 workflow 解析为 CPU/CUDA。
    device: str = DEFAULT_DEVICE
    # 参数/计算 dtype；auto 由 workflow 根据设备选择。
    dtype: str = DEFAULT_DTYPE
    # 训练数据 manifest 路径，None 表示由具体阶段 recipe 或 CLI 填入。
    manifest_path: Path | None = None
    # 训练产物根目录；具体阶段会覆盖到自己的子目录。
    artifact_dir: Path | None = ARTIFACT_ROOT_DIR
    # 目标总 epoch 数；训练循环会结合 max_steps 得到实际 batch 预算。
    target_total_epochs: int = DEFAULT_TRAINING_EPOCHS
    # 最大训练 batch 数；None 表示不按 step 额外截断。
    max_steps: int | None = DEFAULT_TRAINING_MAX_STEPS
    # optimizer 初始学习率。
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    # AdamW weight decay，norm/bias 等参数会在 optimizer 构造时排除。
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    # 优化器类型：adamw 或 adamw8bit（8bit 量化优化器状态，需要 bitsandbytes）。
    optimizer_type: str = DEFAULT_OPTIMIZER_TYPE
    # 是否启用梯度检查点以节省激活值显存；启用后训练速度降低 10-20%。
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING
    # 梯度累积步数，用于小显存下扩大等效 batch。
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    # 梯度裁剪阈值；0/None 通常表示不裁剪。
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    # LM loss 分块 token 数；256 为显存安全默认，显存充足时可调大减少 kernel 数。
    lm_loss_chunk_tokens: int = DEFAULT_LM_LOSS_CHUNK_TOKENS
    # warmup 占总训练步数比例。
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    # 随机种子，覆盖数据 shuffle、LongRoPE2 窗口采样和 torch 随机数。
    random_seed: int = DEFAULT_TRAINING_SEED
    # 是否要求 torch 使用确定性算法，便于回归测试复现。
    deterministic_algorithms: bool = DEFAULT_DETERMINISTIC_ALGORITHMS
    # 是否启用 sequence packing；packing 会生成 segment_ids 阻断跨样本注意力。
    sequence_packing_enabled: bool = DEFAULT_SEQUENCE_PACKING_ENABLED
    # batch 预处理 worker 数；设为 0 时在训练主进程同步执行，适合测试和 CPU smoke。
    data_num_workers: int = DEFAULT_DATA_NUM_WORKERS
    # 每个预处理 worker 的预取 batch 数，仅在 data_num_workers > 0 时生效。
    data_prefetch_factor: int = DEFAULT_DATA_PREFETCH_FACTOR
    # 是否跨 epoch 保持预处理 worker，避免 Windows 重复创建进程。
    data_persistent_workers: bool = DEFAULT_DATA_PERSISTENT_WORKERS
    # 是否使用锁页内存，为 CUDA non_blocking H2D 拷贝提供前提。
    data_pin_memory: bool = DEFAULT_DATA_PIN_MEMORY
    # 原始 JSONL 解析 worker 数；0 表示由训练主进程同步读取。
    data_reader_num_workers: int = DEFAULT_DATA_READER_NUM_WORKERS
    # JSONL reader 单个字节 chunk 大小。
    data_reader_chunk_bytes: int = DEFAULT_DATA_READER_CHUNK_BYTES
    # 有界长度排序窗口；0 表示不做长度重排。
    data_length_bucket_window: int = 0
    # packing 装箱策略：sequential 保持旧 next-fit，first_fit 减少容量碎片。
    data_packing_strategy: str = "sequential"
    # 离线 token cache 目录；None 表示由具体阶段 recipe 或 CLI 指定。
    token_cache_dir: Path | None = None
    # 是否启用离线 token cache；各阶段 recipe 决定默认值。
    token_cache_enabled: bool = DEFAULT_TOKEN_CACHE_ENABLED
    # 训练时使用的原生 thinking 策略；text 阶段默认 off，chat 阶段可覆盖为 auto。
    thinking_mode: str = "off"
    # 训练可见性仅写入审计元数据，不改变 loss 或 batch 构造。
    thinking_visibility: str = "hidden"
    # 训练指标输出间隔。
    log_interval_steps: int = DEFAULT_LOG_INTERVAL_STEPS
    # 验证间隔；0 表示关闭周期性验证。
    eval_interval_steps: int = DEFAULT_EVAL_INTERVAL_STEPS
    # 验证 batch size；None 表示复用训练 batch size。
    eval_batch_size: int | None = None
    # 最多验证多少个 batch；None 表示遍历验证集。
    eval_max_batches: int | None = DEFAULT_EVAL_MAX_BATCHES
    # step checkpoint 保存间隔；0 表示不额外保存 step_N。
    save_interval_steps: int = DEFAULT_SAVE_INTERVAL_STEPS
    # latest checkpoint 保存间隔，用于中断恢复。
    latest_save_interval_steps: int = DEFAULT_LATEST_SAVE_INTERVAL_STEPS
    # 需要强制保存的关键 step 列表。
    key_checkpoints: tuple[int, ...] = field(default_factory=tuple)
    # 是否按指标额外保存 best checkpoint。
    save_best_checkpoint: bool = DEFAULT_SAVE_BEST_CHECKPOINT
    # best checkpoint 使用的指标名。
    best_checkpoint_metric: str = DEFAULT_BEST_CHECKPOINT_METRIC
    # best checkpoint 指标最小改善幅度。
    best_checkpoint_min_delta: float = DEFAULT_BEST_CHECKPOINT_MIN_DELTA
    # 是否保存 optimizer state；关闭后 checkpoint 只能推理/加载，不能完整续训。
    save_optimizer: bool = DEFAULT_SAVE_OPTIMIZER
    # 是否保存 scheduler state；用于恢复学习率曲线。
    save_scheduler: bool = DEFAULT_SAVE_SCHEDULER
    # 是否写 TensorBoard scalar。
    tensorboard_enabled: bool = DEFAULT_TENSORBOARD_ENABLED
    # 本次运行 id；None 时训练循环生成阶段名前缀的随机 id。
    run_id: str | None = None
    # 训练最大序列长度覆盖项；None 表示使用 GlobalConfig 或 ModelConfig 默认。
    train_max_sequence_length: int | None = None
    # 训练 RoPE cache 最大长度覆盖项。
    train_rope_cache_max_sequence_length: int | None = None
    # 推理 RoPE cache 最大长度覆盖项，会写入保存的配置快照。
    inference_rope_cache_max_sequence_length: int | None = None
    # LongRoPE2 原始窗口长度覆盖项。
    longrope2_original_window: int | None = None
    # LongRoPE2 目标窗口长度覆盖项。
    longrope2_target_window: int | None = None
    # 外部 LongRoPE2 long factors 文件路径。
    longrope2_long_factors_path: Path | None = None
    # 训练阶段 LongRoPE2 embedding mode 覆盖项。
    longrope2_train_embedding_mode: str | None = None
    # 推理阶段 LongRoPE2 embedding mode 覆盖项。
    longrope2_inference_embedding_mode: str | None = None
    # mixed mode 原始窗口边界覆盖项。
    longrope2_mixed_original_window: int | None = None
    # 训练时按 batch 随机采样的窗口长度集合。
    longrope2_window_lengths: tuple[int, ...] | None = None
    # 与 longrope2_window_lengths 一一对应的采样权重。
    longrope2_window_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        """校验 recipe 的工程边界，避免 CLI 覆盖项生成不可续训的配置。"""
        if self.target_total_epochs <= 0:
            raise ValueError("target_total_epochs 必须为正整数。")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数。")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正整数。")
        if self.data_num_workers < 0:
            raise ValueError("data_num_workers 必须为非负整数。")
        if self.data_prefetch_factor <= 0:
            raise ValueError("data_prefetch_factor 必须为正整数。")
        if self.data_reader_num_workers < 0:
            raise ValueError("data_reader_num_workers 必须为非负整数。")
        if self.data_reader_chunk_bytes <= 0:
            raise ValueError("data_reader_chunk_bytes 必须为正整数。")
        if self.data_length_bucket_window < 0:
            raise ValueError("data_length_bucket_window 必须为非负整数。")
        if self.data_packing_strategy not in {"sequential", "first_fit"}:
            raise ValueError("data_packing_strategy 必须是 sequential 或 first_fit。")
        if not 0.0 <= float(self.warmup_ratio) <= 1.0:
            raise ValueError("warmup_ratio 必须在 [0, 1] 范围内。")
        if self.eval_max_batches is not None and self.eval_max_batches <= 0:
            raise ValueError("eval_max_batches 必须为正整数。")
        if self.save_interval_steps < 0:
            raise ValueError("save_interval_steps 必须为非负整数。")
        if self.latest_save_interval_steps < 0:
            raise ValueError("latest_save_interval_steps 必须为非负整数。")
        if self.best_checkpoint_metric not in {"loss", "eval_loss"}:
            raise ValueError("best_checkpoint_metric 必须是 loss 或 eval_loss。")
        if self.best_checkpoint_min_delta < 0:
            raise ValueError("best_checkpoint_min_delta 必须为非负数。")
        if self.thinking_mode not in {"off", "on", "auto"}:
            raise ValueError("thinking_mode 必须是 off/on/auto。")
        if self.thinking_visibility not in {"hidden", "visible"}:
            raise ValueError("thinking_visibility 必须是 hidden/visible。")
        if self.optimizer_type not in {"adamw", "adamw8bit"}:
            raise ValueError("optimizer_type 必须是 adamw 或 adamw8bit。")


@dataclass(frozen=True)
class TextPretrainingConfig(BaseTrainingRecipeConfig):
    """text_pretrain 阶段默认 recipe。"""

    # 三阶段链路第 1 阶段：通用文本预训练。
    training_stage: str = "text_pretrain"
    # text_pretrain 默认 manifest。
    manifest_path: Path | None = TEXT_PRETRAIN_MANIFEST_PATH
    # text_pretrain 默认产物目录。
    artifact_dir: Path | None = TEXT_PRETRAIN_ARTIFACT_DIR
    # 文本长短差异较大，按训练窗口 token 预算自动组批。
    # text_pretrain 默认 epoch 数。
    target_total_epochs: int = TEXT_PRETRAIN_EPOCHS
    # text_pretrain 默认学习率。
    learning_rate: float = TEXT_PRETRAIN_LEARNING_RATE
    # text_pretrain 默认 warmup 比例。
    warmup_ratio: float = TEXT_PRETRAIN_WARMUP_RATIO
    # first-fit packing 已处理长度差异；默认不再把长样本集中到同一批。
    data_length_bucket_window: int = DEFAULT_DATA_LENGTH_BUCKET_WINDOW
    # text 预训练优先使用 first-fit，降低 packing 后的容量碎片。
    data_packing_strategy: str = DEFAULT_DATA_PACKING_STRATEGY
    # text 预训练默认启用离线 token cache，避免大 JSONL 启动扫描单核阻塞。
    token_cache_enabled: bool = TEXT_PRETRAIN_TOKEN_CACHE_ENABLED
    # text 预训练默认 cache 目录。
    token_cache_dir: Path | None = DEFAULT_TOKEN_CACHE_DIR / "text_pretrain"


@dataclass(frozen=True)
class ChatSFTTrainingConfig(BaseTrainingRecipeConfig):
    """chat_sft 阶段默认 recipe。"""

    # 三阶段链路第 2 阶段：对话监督微调。
    training_stage: str = "chat_sft"
    # chat_sft 默认 manifest。
    manifest_path: Path | None = CHAT_SFT_MANIFEST_PATH
    # chat_sft 默认产物目录。
    artifact_dir: Path | None = CHAT_SFT_ARTIFACT_DIR
    # 短对话自动增大批量，长对话自动缩小到单样本。
    # chat_sft 默认 epoch 数。
    target_total_epochs: int = CHAT_SFT_EPOCHS
    # chat_sft 默认学习率。
    learning_rate: float = CHAT_SFT_LEARNING_RATE
    # chat_sft 默认 warmup 比例。
    warmup_ratio: float = CHAT_SFT_WARMUP_RATIO
    # chat SFT 默认按样本内非空 thinking 字段自动切换分支。
    thinking_mode: str = "auto"
    # SFT 缓存完整的对话监督序列，训练时仍动态执行截断和 packing。
    token_cache_enabled: bool = CHAT_SFT_TOKEN_CACHE_ENABLED
    token_cache_dir: Path | None = DEFAULT_TOKEN_CACHE_DIR / "chat_sft"


@dataclass(frozen=True)
class ChatLoRATrainingConfig(BaseTrainingRecipeConfig):
    """chat_lora 阶段默认 recipe。"""

    # 三阶段链路第 3 阶段：对话 LoRA adapter 训练。
    training_stage: str = "chat_lora"
    # chat_lora 默认 manifest。
    manifest_path: Path | None = CHAT_LORA_MANIFEST_PATH
    # chat_lora 默认产物目录。
    artifact_dir: Path | None = CHAT_LORA_ARTIFACT_DIR
    # LoRA 与 SFT 使用同一套动态 token 预算，避免长样本造成显存峰值。
    # chat_lora 默认 epoch 数。
    target_total_epochs: int = CHAT_LORA_EPOCHS
    # chat_lora 默认学习率。
    learning_rate: float = CHAT_LORA_LEARNING_RATE
    # chat_lora 默认 warmup 比例。
    warmup_ratio: float = CHAT_LORA_WARMUP_RATIO
    # chat LoRA 默认按样本内非空 thinking 字段自动切换分支。
    thinking_mode: str = "auto"
    # LoRA 与 SFT 使用相同的对话预编码缓存实现，目录按阶段隔离。
    token_cache_enabled: bool = CHAT_LORA_TOKEN_CACHE_ENABLED
    token_cache_dir: Path | None = DEFAULT_TOKEN_CACHE_DIR / "chat_lora"
    # LoRA 训练使用的基座来源。
    lora_base_source: str = DEFAULT_LORA_BASE_SOURCE
    # LoRA 低秩 rank。
    lora_rank: int = DEFAULT_LORA_RANK
    # LoRA alpha 缩放系数。
    lora_alpha: float = DEFAULT_LORA_ALPHA
    # LoRA adapter dropout。
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    # 默认注入 LoRA 的模块名集合。
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES


@dataclass(frozen=True)
class BaselineEvalConfig:
    """profile 基线评测默认配置。"""

    # 要评测的 profile 列表，all 表示 constants 中定义的全部基线 profile。
    profiles: str | None = "all"
    # 模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 随机输入使用的词表大小。
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    # 评测 batch size。
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    # prefill 输入序列长度。
    sequence_length: int = DEFAULT_EVAL_SEQUENCE_LENGTH
    # decode 追加步数。
    decode_steps: int = DEFAULT_EVAL_DECODE_STEPS
    # 评测设备。
    device: str = DEFAULT_DEVICE
    # 评测 dtype。
    dtype: str = DEFAULT_DTYPE
    # 评测随机种子。
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class ResourceEvalConfig:
    """资源评测默认配置。"""

    # 资源评测使用的单个 profile。
    profile: str = DEFAULT_PROFILE
    # 模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 随机输入使用的词表大小。
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    # 资源评测 batch size。
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    # prefill 输入序列长度。
    sequence_length: int = DEFAULT_EVAL_SEQUENCE_LENGTH
    # decode 追加步数；资源评测默认多走几步以暴露 KV/cache 增量。
    decode_steps: int = DEFAULT_RESOURCE_DECODE_STEPS
    # 评测设备。
    device: str = DEFAULT_DEVICE
    # 评测 dtype。
    dtype: str = DEFAULT_DTYPE
    # 评测随机种子。
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class MemoryEvalConfig:
    """xLSTMAssist 评测默认配置。"""

    # 模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 随机输入使用的词表大小。
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    # xLSTMAssist 状态观测用序列长度。
    sequence_length: int = DEFAULT_MEMORY_EVAL_SEQUENCE_LENGTH
    # 评测设备。
    device: str = DEFAULT_DEVICE
    # 评测 dtype。
    dtype: str = DEFAULT_DTYPE
    # 评测随机种子。
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class LongContextEvalConfig:
    """长上下文评测默认配置。"""

    # 模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 随机输入使用的词表大小。
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    # 长上下文输入长度；None 表示由评测入口按配置自动推导。
    sequence_length: int | None = None
    # 局部 attention 窗口大小。
    attention_window_size: int = DEFAULT_LONG_CONTEXT_ATTENTION_WINDOW_SIZE
    # 评测设备。
    device: str = DEFAULT_DEVICE
    # 评测 dtype。
    dtype: str = DEFAULT_DTYPE
    # 评测随机种子。
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class InferenceSmokeConfig:
    """token-id 推理 smoke 默认配置。"""

    # 推理 smoke 使用的 profile。
    profile: str = DEFAULT_PROFILE
    # 模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 随机/测试模型词表大小。
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    # 逗号分隔 token id prompt。
    prompt_ids: str = DEFAULT_INFERENCE_PROMPT_IDS
    # 最大新增 token 数。
    max_new_tokens: int = DEFAULT_INFERENCE_MAX_NEW_TOKENS
    # 推理设备。
    device: str = DEFAULT_DEVICE
    # 推理 dtype。
    dtype: str = DEFAULT_DTYPE
    # 推理随机种子。
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class TrainingSmokeConfig:
    """最小训练 smoke 默认配置。"""

    # 训练 smoke 使用的 profile。
    profile: str = DEFAULT_PROFILE
    # 模型尺寸 preset。
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 随机样本词表大小。
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    # smoke batch size。
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    # smoke 序列长度。
    sequence_length: int = DEFAULT_EVAL_SEQUENCE_LENGTH
    # smoke 训练步数。
    steps: int = 1
    # smoke optimizer 学习率。
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    # 训练设备。
    device: str = DEFAULT_DEVICE
    # 训练 dtype。
    dtype: str = DEFAULT_DTYPE
    # 训练随机种子。
    seed: int = DEFAULT_TRAINING_SEED
