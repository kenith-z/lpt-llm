"""LPT v2 项目级运行配置。"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .constants import (
    ARTIFACT_ROOT_DIR,
    CHAT_LORA_ARTIFACT_DIR,
    CHAT_LORA_BATCH_SIZE,
    CHAT_LORA_EPOCHS,
    CHAT_LORA_LEARNING_RATE,
    CHAT_LORA_MANIFEST_PATH,
    CHAT_LORA_WARMUP_RATIO,
    CHAT_SFT_ARTIFACT_DIR,
    CHAT_SFT_BATCH_SIZE,
    CHAT_SFT_EPOCHS,
    CHAT_SFT_LEARNING_RATE,
    CHAT_SFT_MANIFEST_PATH,
    CHAT_SFT_WARMUP_RATIO,
    CHAT_TEMPLATE_VERSION,
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
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MEMORY_EVAL_SEQUENCE_LENGTH,
    DEFAULT_MODEL_SIZE_PRESET,
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
    DEFAULT_TRAINING_BATCH_SIZE,
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
    TEXT_PRETRAIN_BATCH_SIZE,
    TEXT_PRETRAIN_EPOCHS,
    TEXT_PRETRAIN_LEARNING_RATE,
    TEXT_PRETRAIN_MANIFEST_PATH,
    TEXT_PRETRAIN_WARMUP_RATIO,
)


class GlobalConfig:
    """全局运行配置。

    这里使用类属性，便于模型、数据批处理和测试在同一进程内共享运行参数。
    """

    model_abbr = MODEL_ABBR
    model_name_en = MODEL_NAME_EN
    model_name_zh = MODEL_NAME_ZH
    chat_template_version = CHAT_TEMPLATE_VERSION

    train_max_sequence_length = 4096
    inference_max_sequence_length = 65536 // 2
    train_rope_cache_max_sequence_length = train_max_sequence_length
    inference_rope_cache_max_sequence_length = inference_max_sequence_length

    parameter_dtype = torch.float32
    autocast_dtype = torch.bfloat16
    pad_to_multiple_of = 8
    dataset_shuffle_buffer_size = 2048
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GenerationConfig:
    """文本生成采样配置；v2 推理入口接入前保持为轻量数据结构。"""

    do_sample: bool = DEFAULT_GENERATION_DO_SAMPLE
    temperature: float = DEFAULT_GENERATION_TEMPERATURE
    top_k: int = DEFAULT_GENERATION_TOP_K
    top_p: float = DEFAULT_GENERATION_TOP_P
    max_length: int = DEFAULT_GENERATION_MAX_LENGTH
    repetition_penalty: float = DEFAULT_GENERATION_REPETITION_PENALTY
    repetition_window_size: int | None = DEFAULT_GENERATION_REPETITION_WINDOW_SIZE


@dataclass(frozen=True)
class BaseTrainingRecipeConfig:
    """v2 阶段训练默认 recipe。

    训练入口以这个配置为主，CLI/测试注入参数只覆盖明确传入的字段。
    """

    training_stage: str = DEFAULT_TRAINING_STAGE
    profile: str = DEFAULT_PROFILE
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    manifest_path: Path | None = None
    artifact_dir: Path | None = ARTIFACT_ROOT_DIR
    batch_size: int = DEFAULT_TRAINING_BATCH_SIZE
    target_total_epochs: int = DEFAULT_TRAINING_EPOCHS
    max_steps: int | None = DEFAULT_TRAINING_MAX_STEPS
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    random_seed: int = DEFAULT_TRAINING_SEED
    deterministic_algorithms: bool = DEFAULT_DETERMINISTIC_ALGORITHMS
    sequence_packing_enabled: bool = DEFAULT_SEQUENCE_PACKING_ENABLED
    log_interval_steps: int = DEFAULT_LOG_INTERVAL_STEPS
    eval_interval_steps: int = DEFAULT_EVAL_INTERVAL_STEPS
    eval_batch_size: int | None = None
    eval_max_batches: int | None = DEFAULT_EVAL_MAX_BATCHES
    save_interval_steps: int = DEFAULT_SAVE_INTERVAL_STEPS
    latest_save_interval_steps: int = DEFAULT_LATEST_SAVE_INTERVAL_STEPS
    key_checkpoints: tuple[int, ...] = field(default_factory=tuple)
    save_best_checkpoint: bool = DEFAULT_SAVE_BEST_CHECKPOINT
    best_checkpoint_metric: str = DEFAULT_BEST_CHECKPOINT_METRIC
    best_checkpoint_min_delta: float = DEFAULT_BEST_CHECKPOINT_MIN_DELTA
    save_optimizer: bool = DEFAULT_SAVE_OPTIMIZER
    save_scheduler: bool = DEFAULT_SAVE_SCHEDULER
    tensorboard_enabled: bool = DEFAULT_TENSORBOARD_ENABLED
    run_id: str | None = None
    train_max_sequence_length: int | None = None
    train_rope_cache_max_sequence_length: int | None = None
    inference_rope_cache_max_sequence_length: int | None = None
    longrope2_original_window: int | None = None
    longrope2_target_window: int | None = None
    longrope2_long_factors_path: Path | None = None
    longrope2_train_embedding_mode: str | None = None
    longrope2_inference_embedding_mode: str | None = None
    longrope2_mixed_original_window: int | None = None
    longrope2_window_lengths: tuple[int, ...] | None = None
    longrope2_window_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.target_total_epochs <= 0:
            raise ValueError("target_total_epochs 必须为正整数。")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数。")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正整数。")
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


@dataclass(frozen=True)
class TextPretrainingConfig(BaseTrainingRecipeConfig):
    """text_pretrain 阶段默认 recipe。"""

    training_stage: str = "text_pretrain"
    manifest_path: Path | None = TEXT_PRETRAIN_MANIFEST_PATH
    artifact_dir: Path | None = TEXT_PRETRAIN_ARTIFACT_DIR
    batch_size: int = TEXT_PRETRAIN_BATCH_SIZE
    target_total_epochs: int = TEXT_PRETRAIN_EPOCHS
    learning_rate: float = TEXT_PRETRAIN_LEARNING_RATE
    warmup_ratio: float = TEXT_PRETRAIN_WARMUP_RATIO


@dataclass(frozen=True)
class ChatSFTTrainingConfig(BaseTrainingRecipeConfig):
    """chat_sft 阶段默认 recipe。"""

    training_stage: str = "chat_sft"
    manifest_path: Path | None = CHAT_SFT_MANIFEST_PATH
    artifact_dir: Path | None = CHAT_SFT_ARTIFACT_DIR
    batch_size: int = CHAT_SFT_BATCH_SIZE
    target_total_epochs: int = CHAT_SFT_EPOCHS
    learning_rate: float = CHAT_SFT_LEARNING_RATE
    warmup_ratio: float = CHAT_SFT_WARMUP_RATIO


@dataclass(frozen=True)
class ChatLoRATrainingConfig(BaseTrainingRecipeConfig):
    """chat_lora 阶段默认 recipe。"""

    training_stage: str = "chat_lora"
    manifest_path: Path | None = CHAT_LORA_MANIFEST_PATH
    artifact_dir: Path | None = CHAT_LORA_ARTIFACT_DIR
    batch_size: int = CHAT_LORA_BATCH_SIZE
    target_total_epochs: int = CHAT_LORA_EPOCHS
    learning_rate: float = CHAT_LORA_LEARNING_RATE
    warmup_ratio: float = CHAT_LORA_WARMUP_RATIO
    lora_base_source: str = DEFAULT_LORA_BASE_SOURCE
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: float = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES


@dataclass(frozen=True)
class BaselineEvalConfig:
    """profile 基线评测默认配置。"""

    profiles: str | None = "all"
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    sequence_length: int = DEFAULT_EVAL_SEQUENCE_LENGTH
    decode_steps: int = DEFAULT_EVAL_DECODE_STEPS
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class ResourceEvalConfig:
    """资源评测默认配置。"""

    profile: str = DEFAULT_PROFILE
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    sequence_length: int = DEFAULT_EVAL_SEQUENCE_LENGTH
    decode_steps: int = DEFAULT_RESOURCE_DECODE_STEPS
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class MemoryEvalConfig:
    """xLSTMAssist 评测默认配置。"""

    preset: str = DEFAULT_MODEL_SIZE_PRESET
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    sequence_length: int = DEFAULT_MEMORY_EVAL_SEQUENCE_LENGTH
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class LongContextEvalConfig:
    """长上下文评测默认配置。"""

    preset: str = DEFAULT_MODEL_SIZE_PRESET
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    sequence_length: int | None = None
    attention_window_size: int = DEFAULT_LONG_CONTEXT_ATTENTION_WINDOW_SIZE
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class InferenceSmokeConfig:
    """token-id 推理 smoke 默认配置。"""

    profile: str = DEFAULT_PROFILE
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    prompt_ids: str = DEFAULT_INFERENCE_PROMPT_IDS
    max_new_tokens: int = DEFAULT_INFERENCE_MAX_NEW_TOKENS
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_TRAINING_SEED


@dataclass(frozen=True)
class TrainingSmokeConfig:
    """最小训练 smoke 默认配置。"""

    profile: str = DEFAULT_PROFILE
    preset: str = DEFAULT_MODEL_SIZE_PRESET
    vocabulary_size: int = DEFAULT_EVAL_VOCAB_SIZE
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE
    sequence_length: int = DEFAULT_EVAL_SEQUENCE_LENGTH
    steps: int = 1
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    seed: int = DEFAULT_TRAINING_SEED
