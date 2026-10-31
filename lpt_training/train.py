"""LPT v2 正式训练循环。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
from time import time
from uuid import uuid4

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lpt_config import (
    DEFAULT_EVAL_MAX_BATCHES,
    DEFAULT_DETERMINISTIC_ALGORITHMS,
    DEFAULT_DATA_NUM_WORKERS,
    DEFAULT_DATA_READER_CHUNK_BYTES,
    DEFAULT_DATA_READER_NUM_WORKERS,
    DEFAULT_DATA_PERSISTENT_WORKERS,
    DEFAULT_DATA_PIN_MEMORY,
    DEFAULT_DATA_PREFETCH_FACTOR,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LATEST_SAVE_INTERVAL_STEPS,
    DEFAULT_LM_LOSS_CHUNK_TOKENS,
    DEFAULT_LOG_INTERVAL_STEPS,
    DEFAULT_MAX_GRAD_NORM,
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
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    GlobalConfig,
)
from lpt_config.constants import DEFAULT_OPTIMIZER_TYPE, DEFAULT_GRADIENT_CHECKPOINTING
from lpt_data import DATA_PROGRESS_RECORD_KEY, build_packed_training_batch, build_training_batch
from lpt_data.batching import (
    encode_training_sample,
    PRE_TOKENIZED_CHAT_INPUT_IDS,
    PRE_TOKENIZED_CHAT_LABELS,
    PRE_TOKENIZED_CHAT_THINKING_MODE_IDS,
    PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS,
)
from lpt_data.cache import JSONLReaderPool
from lpt_data.chat_token_cache_dataset import PRE_TOKENIZED_CHAT_THINKING_SUMMARY
from lpt_data.token_cache_dataset import PRE_TOKENIZED_TOKEN_LENGTH
from lpt_lora import save_lora_adapter_state
from lpt_model import save_lpt_v2_checkpoint
from lpt_protocol import THINKING_MODE_OFF, THINKING_MODE_ON, normalize_thinking_mode
from lpt_runtime.files import atomic_torch_save, atomic_write_text, is_torch_save_file_readable


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard 是可选运行依赖
    SummaryWriter = None


tqdm.monitor_interval = 0

MODEL_CHECKPOINT_NAME = "model.pt"
LORA_ADAPTER_CHECKPOINT_NAME = "adapter.pt"
OPTIMIZER_CHECKPOINT_NAME = "optimizer.pt"
SCHEDULER_CHECKPOINT_NAME = "scheduler.pt"
RNG_STATE_NAME = "rng_state.pt"
TRAINER_STATE_NAME = "trainer_state.json"
DATA_PROGRESS_NAME = "data_progress.json"
METRICS_JSONL_NAME = "metrics.jsonl"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
TRAINING_CHECKPOINT_FORMAT = "lpt_v2_training_checkpoint"
TRAINING_CHECKPOINT_SCHEMA_VERSION = 1
DATA_PROGRESS_SCHEMA_VERSION = 1
# 保持旧常量名兼容内部引用；默认值统一由 lpt_config.constants 提供，
# recipe/CLI 可通过 lm_loss_chunk_tokens 覆盖。
LM_LOSS_CHUNK_TOKENS = DEFAULT_LM_LOSS_CHUNK_TOKENS
THINKING_VISIBILITIES = ("hidden", "visible")
TRAINING_EPOCH_RECORD_KEY = "_lpt_training_epoch"
THINKING_SUMMARY_COUNTER_KEYS = (
    "text_samples",
    "chat_samples",
    "assistant_messages",
    "assistant_messages_with_thinking_field",
    "assistant_messages_with_nonempty_thinking",
    "assistant_messages_with_empty_thinking",
    "assistant_messages_thinking_on",
    "assistant_messages_thinking_off",
)


PHASE_DISPLAY_NAMES = {
    "train": "训练(train)",
    "eval": "验证(eval)",
}

METRIC_DISPLAY_NAMES = {
    "stage": "阶段(stage)",
    "epoch": "轮次(epoch)",
    "global_step": "全局步(global_step)",
    "optimizer_step": "优化器步(optimizer_step)",
    "loss": "损失(loss)",
    "learning_rate": "学习率(learning_rate)",
    "samples_seen": "已见样本(samples_seen)",
    "tokens_seen": "已见Token(tokens_seen)",
    "sequence_length": "序列长度(sequence_length)",
    "packed_rows": "填充行数(packed_rows)",
    "padded_tokens": "填充Token(padded_tokens)",
    "attention_token_square": "注意力规模(attention_token_square)",
    "target_tokens": "目标Token(target_tokens)",
    "tokens_per_sec": "Token吞吐(tokens_per_sec)",
    "samples_per_sec": "样本吞吐(samples_per_sec)",
    "grad_norm": "梯度范数(grad_norm)",
    "cuda_memory_allocated_mib": "CUDA已分配MiB(cuda_memory_allocated_mib)",
    "cuda_memory_reserved_mib": "CUDA已保留MiB(cuda_memory_reserved_mib)",
    "cuda_peak_memory_allocated_mib": "CUDA分配峰值MiB(cuda_peak_memory_allocated_mib)",
    "cuda_peak_memory_reserved_mib": "CUDA保留峰值MiB(cuda_peak_memory_reserved_mib)",
    "elapsed_seconds": "耗时秒(elapsed_seconds)",
    "run_id": "运行ID(run_id)",
    "process_id": "进程ID(process_id)",
    "wall_time_unix": "墙钟Unix秒(wall_time_unix)",
    "eval_loss": "验证损失(eval_loss)",
    "eval_ppl": "验证困惑度(eval_ppl)",
    "eval_target_tokens": "验证目标Token(eval_target_tokens)",
    "eval_batches": "验证批次(eval_batches)",
}


@dataclass
class TrainingRunConfig:
    """单次训练运行配置。"""

    training_stage: str
    artifact_dir: Path
    checkpoint_dir: Path
    inference_weight_path: Path
    save_inference_weights: bool = True
    epochs: int = DEFAULT_TRAINING_EPOCHS
    max_steps: int | None = DEFAULT_TRAINING_MAX_STEPS
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer_type: str = DEFAULT_OPTIMIZER_TYPE
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    # LM loss 分块 token 数，控制大 vocab 下 loss 临时张量的峰值显存。
    lm_loss_chunk_tokens: int = LM_LOSS_CHUNK_TOKENS
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    log_interval: int = DEFAULT_LOG_INTERVAL_STEPS
    eval_interval: int = 0
    eval_batch_size: int | None = None
    eval_max_batches: int | None = DEFAULT_EVAL_MAX_BATCHES
    save_interval: int = DEFAULT_SAVE_INTERVAL_STEPS
    latest_save_interval: int = DEFAULT_LATEST_SAVE_INTERVAL_STEPS
    key_checkpoints: tuple[int, ...] = ()
    save_best_checkpoint: bool = DEFAULT_SAVE_BEST_CHECKPOINT
    best_checkpoint_metric: str = DEFAULT_BEST_CHECKPOINT_METRIC
    best_checkpoint_min_delta: float = DEFAULT_BEST_CHECKPOINT_MIN_DELTA
    best_checkpoint_dir: Path | None = None
    save_optimizer: bool = DEFAULT_SAVE_OPTIMIZER
    save_scheduler: bool = DEFAULT_SAVE_SCHEDULER
    max_sequence_length: int | None = None
    sequence_packing: bool = DEFAULT_SEQUENCE_PACKING_ENABLED
    thinking_mode: str = THINKING_MODE_OFF
    thinking_visibility: str = "hidden"
    seed: int = DEFAULT_TRAINING_SEED
    deterministic_algorithms: bool = DEFAULT_DETERMINISTIC_ALGORITHMS
    resume_checkpoint: Path | None = None
    initial_checkpoint: Path | None = None
    source_manifest: Path | None = None
    eval_manifest: Path | None = None
    lora_mode: bool = False
    longrope2_window_lengths: tuple[int, ...] | None = None
    longrope2_window_weights: tuple[float, ...] | None = None
    tokenizer_metadata: dict | None = None
    run_id: str | None = None
    tensorboard_enabled: bool = DEFAULT_TENSORBOARD_ENABLED
    tensorboard_dir: Path | None = None
    data_num_workers: int = DEFAULT_DATA_NUM_WORKERS
    data_prefetch_factor: int = DEFAULT_DATA_PREFETCH_FACTOR
    data_persistent_workers: bool = DEFAULT_DATA_PERSISTENT_WORKERS
    data_pin_memory: bool = DEFAULT_DATA_PIN_MEMORY
    data_reader_num_workers: int = DEFAULT_DATA_READER_NUM_WORKERS
    data_reader_chunk_bytes: int = DEFAULT_DATA_READER_CHUNK_BYTES
    # 直接构造 TrainingRunConfig 时保持旧顺序；阶段 recipe 会显式注入优化值。
    data_length_bucket_window: int = 0
    data_packing_strategy: str = "sequential"
    token_cache_dir: Path | None = None
    token_cache_enabled: bool = False

    def __post_init__(self):
        """规范化路径和数值字段，保证训练状态可以稳定序列化与续训。"""
        if self.run_id is None:
            self.run_id = f"{self.training_stage}-{uuid4().hex[:12]}"
        self.artifact_dir = Path(self.artifact_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.inference_weight_path = Path(self.inference_weight_path)
        self.save_inference_weights = bool(self.save_inference_weights)
        if self.resume_checkpoint is not None:
            self.resume_checkpoint = Path(self.resume_checkpoint)
        if self.initial_checkpoint is not None:
            self.initial_checkpoint = Path(self.initial_checkpoint)
        if self.best_checkpoint_dir is not None:
            self.best_checkpoint_dir = Path(self.best_checkpoint_dir)
        if self.source_manifest is not None:
            self.source_manifest = Path(self.source_manifest)
        if self.eval_manifest is not None:
            self.eval_manifest = Path(self.eval_manifest)
        if self.tensorboard_dir is not None:
            self.tensorboard_dir = Path(self.tensorboard_dir)
        if self.token_cache_dir is not None:
            self.token_cache_dir = Path(self.token_cache_dir)
        self.token_cache_enabled = bool(self.token_cache_enabled)
        if self.longrope2_window_lengths is not None:
            lengths = tuple(int(value) for value in self.longrope2_window_lengths)
            if not lengths or any(value <= 0 for value in lengths):
                raise ValueError("longrope2_window_lengths 必须是正整数序列。")
            self.longrope2_window_lengths = lengths
        if self.longrope2_window_weights is not None:
            weights = tuple(float(value) for value in self.longrope2_window_weights)
            if not weights or any(value < 0 for value in weights):
                raise ValueError("longrope2_window_weights 必须是非负数序列。")
            self.longrope2_window_weights = weights
        if (
            self.longrope2_window_lengths is not None
            and self.longrope2_window_weights is not None
            and len(self.longrope2_window_lengths) != len(self.longrope2_window_weights)
        ):
            raise ValueError("longrope2_window_lengths 与 longrope2_window_weights 长度必须一致。")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数。")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正整数。")
        if not 0.0 <= float(self.warmup_ratio) <= 1.0:
            raise ValueError("warmup_ratio 必须在 [0, 1] 范围内。")
        self.lm_loss_chunk_tokens = int(self.lm_loss_chunk_tokens)
        if self.lm_loss_chunk_tokens <= 0:
            raise ValueError("lm_loss_chunk_tokens 必须为正整数。")
        if self.eval_batch_size is not None and int(self.eval_batch_size) <= 0:
            raise ValueError("eval_batch_size 必须为正整数。")
        if self.eval_max_batches is not None:
            self.eval_max_batches = int(self.eval_max_batches)
        if self.eval_max_batches is not None and self.eval_max_batches <= 0:
            raise ValueError("eval_max_batches 必须为正整数。")
        self.save_interval = int(self.save_interval)
        if self.save_interval < 0:
            raise ValueError("save_interval 必须为非负整数。")
        self.latest_save_interval = int(self.latest_save_interval)
        if self.latest_save_interval < 0:
            raise ValueError("latest_save_interval 必须为非负整数。")
        if self.best_checkpoint_metric not in {"loss", "eval_loss"}:
            raise ValueError("best_checkpoint_metric 必须是 loss 或 eval_loss。")
        self.best_checkpoint_min_delta = float(self.best_checkpoint_min_delta)
        self.key_checkpoints = tuple(sorted({int(value) for value in self.key_checkpoints if int(value) > 0}))
        self.thinking_mode = normalize_thinking_mode(self.thinking_mode)
        self.thinking_visibility = "hidden" if self.thinking_visibility is None else str(self.thinking_visibility).strip().lower()
        self.gradient_checkpointing = bool(self.gradient_checkpointing)
        self.data_num_workers = int(self.data_num_workers)
        self.data_prefetch_factor = int(self.data_prefetch_factor)
        self.data_persistent_workers = bool(self.data_persistent_workers)
        self.data_pin_memory = bool(self.data_pin_memory)
        self.data_reader_num_workers = int(self.data_reader_num_workers)
        self.data_reader_chunk_bytes = int(self.data_reader_chunk_bytes)
        self.data_length_bucket_window = int(self.data_length_bucket_window)
        self.data_packing_strategy = str(self.data_packing_strategy).strip().lower()
        if self.data_packing_strategy not in {"sequential", "first_fit"}:
            raise ValueError("data_packing_strategy 必须是 sequential 或 first_fit。")


def configure_training_runtime(seed=DEFAULT_TRAINING_SEED, *, deterministic=False):
    """设置训练随机种子。"""
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False, warn_only=True)


def _checkpoint_file(checkpoint_root, name=MODEL_CHECKPOINT_NAME):
    """返回 checkpoint 权重文件路径；允许调用方直接传入 .pt 文件。"""
    root = Path(checkpoint_root)
    if root.suffix:
        return root
    return root / name


def _trainer_state_file(checkpoint_root):
    """返回 trainer_state.json 路径。"""
    return Path(checkpoint_root) / TRAINER_STATE_NAME


def _data_progress_file(checkpoint_root):
    """返回 data_progress.json 路径。"""
    return Path(checkpoint_root) / DATA_PROGRESS_NAME


def _checkpoint_manifest_file(checkpoint_root):
    """返回 checkpoint_manifest.json 路径。"""
    return Path(checkpoint_root) / CHECKPOINT_MANIFEST_NAME


def _checkpoint_payload_name(lora_mode):
    """根据是否 LoRA 训练选择主 payload 文件名。"""
    return LORA_ADAPTER_CHECKPOINT_NAME if lora_mode else MODEL_CHECKPOINT_NAME


def _load_json_file(path):
    """以 UTF-8 读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _training_state_flag(state, key, default):
    """从 trainer_state 或内嵌 training_config 中读取布尔开关。"""
    if key in state:
        return bool(state[key])
    training_config = state.get("training_config")
    if isinstance(training_config, dict) and key in training_config:
        return bool(training_config[key])
    return bool(default)


def _has_readable_torch_file(path):
    """检查 torch 保存文件是否可读，避免半写入文件被当作可续训。"""
    return is_torch_save_file_readable(path)


def _checkpoint_file_entries_are_valid(root, file_entries, required_names):
    """校验 manifest 中记录的文件是否存在且大小一致。"""
    entries_by_name = {}
    for entry in file_entries:
        if not isinstance(entry, dict):
            return False
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            return False
        entries_by_name[name] = entry
    if not set(required_names).issubset(entries_by_name):
        return False
    for name, entry in entries_by_name.items():
        path = root / name
        if not path.is_file():
            return False
        size_bytes = int(entry.get("size_bytes", -1))
        if size_bytes <= 0 or path.stat().st_size != size_bytes:
            return False
    return True


def _required_checkpoint_files(lora_mode, state, *, require_optimizer):
    """根据训练模式和保存开关推导必须存在的 checkpoint 文件。"""
    required_names = {
        _checkpoint_payload_name(lora_mode),
        TRAINER_STATE_NAME,
    }
    if require_optimizer:
        required_names.add(OPTIMIZER_CHECKPOINT_NAME)
    if _training_state_flag(state, "save_scheduler", False):
        required_names.add(SCHEDULER_CHECKPOINT_NAME)
    if bool(state.get("rng_state_saved", False)):
        required_names.add(RNG_STATE_NAME)
    return required_names


def _checkpoint_manifest_is_valid(root, *, lora_mode, require_optimizer):
    """按 v2 training checkpoint manifest 校验目录完整性。"""
    manifest_path = _checkpoint_manifest_file(root)
    try:
        manifest = _load_json_file(manifest_path)
        state = _load_json_file(_trainer_state_file(root))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False

    if manifest.get("checkpoint_format") != TRAINING_CHECKPOINT_FORMAT:
        return False
    if manifest.get("checkpoint_schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
        return False
    if bool(manifest.get("lora_mode")) != bool(lora_mode):
        return False
    if bool(state.get("lora_mode", lora_mode)) != bool(lora_mode):
        return False
    if require_optimizer and not _training_state_flag(state, "save_optimizer", True):
        return False
    try:
        if int(manifest.get("global_step")) != int(state.get("global_step")):
            return False
        if int(manifest.get("optimizer_step")) != int(state.get("optimizer_step")):
            return False
    except (TypeError, ValueError):
        return False

    required_names = _required_checkpoint_files(lora_mode, state, require_optimizer=require_optimizer)
    if not _checkpoint_file_entries_are_valid(root, manifest.get("files", ()), required_names):
        return False
    # manifest 文件大小校验只能发现缺失/截断，torch.load 可读性校验可额外发现损坏的二进制。
    torch_files = [root / _checkpoint_payload_name(lora_mode)]
    if require_optimizer:
        torch_files.append(root / OPTIMIZER_CHECKPOINT_NAME)
    if SCHEDULER_CHECKPOINT_NAME in required_names:
        torch_files.append(root / SCHEDULER_CHECKPOINT_NAME)
    if RNG_STATE_NAME in required_names:
        torch_files.append(root / RNG_STATE_NAME)
    return all(_has_readable_torch_file(path) for path in torch_files)


def _legacy_checkpoint_is_valid(root, *, lora_mode, require_optimizer):
    """兼容早期无 manifest 的同分支训练目录，仅用于本项目 v2 训练续跑。"""
    model_or_adapter = root / _checkpoint_payload_name(lora_mode)
    trainer_state_path = _trainer_state_file(root)
    if not root.is_dir() or not trainer_state_path.is_file():
        return False
    try:
        state = _load_json_file(trainer_state_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if bool(state.get("lora_mode", lora_mode)) != bool(lora_mode):
        return False
    if require_optimizer and not _training_state_flag(state, "save_optimizer", True):
        return False
    if not _has_readable_torch_file(model_or_adapter):
        return False
    if require_optimizer and not _has_readable_torch_file(root / OPTIMIZER_CHECKPOINT_NAME):
        return False
    scheduler_path = root / SCHEDULER_CHECKPOINT_NAME
    if scheduler_path.exists() and not _has_readable_torch_file(scheduler_path):
        return False
    return True


def _is_valid_training_checkpoint_root(checkpoint_root, *, lora_mode=False, require_optimizer=True):
    """统一判断 checkpoint 根目录是否满足续训要求。"""
    root = Path(checkpoint_root)
    if _checkpoint_manifest_file(root).is_file():
        return _checkpoint_manifest_is_valid(root, lora_mode=lora_mode, require_optimizer=require_optimizer)
    return _legacy_checkpoint_is_valid(root, lora_mode=lora_mode, require_optimizer=require_optimizer)


def has_complete_training_state(checkpoint_root, *, lora_mode=False):
    """判断目录中是否存在可续训状态。"""
    return _is_valid_training_checkpoint_root(
        checkpoint_root,
        lora_mode=lora_mode,
        require_optimizer=True,
    )


def _parse_step_checkpoint_index(path):
    """从 step_000123 目录名中解析 step 编号。"""
    name = Path(path).name
    if not name.startswith("step_"):
        return -1
    try:
        return int(name.removeprefix("step_"))
    except ValueError:
        return -1


def _iter_resume_checkpoint_candidates(checkpoint_root):
    """按 latest、latest_previous、step_* 的优先级枚举续训候选。"""
    root = Path(checkpoint_root)
    candidates = [root, root.with_name(f"{root.name}_previous")]
    step_roots = sorted(
        (
            path
            for path in root.parent.glob("step_*")
            if path.is_dir() and _parse_step_checkpoint_index(path) >= 0
        ),
        key=_parse_step_checkpoint_index,
        reverse=True,
    )
    candidates.extend(step_roots)
    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate.absolute()
        if normalized in seen:
            continue
        seen.add(normalized)
        yield candidate


def resolve_latest_training_checkpoint(checkpoint_root, *, lora_mode=False):
    """从 latest、latest_previous 和 step_* 中选择最新可续训 checkpoint。"""
    for candidate in _iter_resume_checkpoint_candidates(checkpoint_root):
        if has_complete_training_state(candidate, lora_mode=lora_mode):
            return candidate
    return None


def load_trainer_state(checkpoint_root):
    """读取 trainer_state.json。"""
    state_path = _trainer_state_file(checkpoint_root)
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def load_data_progress(checkpoint_root):
    """读取 checkpoint 内的数据训练进度。"""
    progress_path = _data_progress_file(checkpoint_root)
    if not progress_path.exists():
        return None
    return json.loads(progress_path.read_text(encoding="utf-8"))


def _iter_trainable_parameters(model):
    """迭代当前需要优化的参数，LoRA 模式下只会返回未冻结参数。"""
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield parameter_name, parameter


def _parameter_group_summary(parameter_group):
    """生成 optimizer 参数组摘要，写入 trainer_state 方便审计。"""
    return {
        "parameter_count": len(parameter_group),
        "element_count": int(sum(parameter.numel() for parameter in parameter_group)),
    }


def _build_optimizer(model, *, learning_rate, weight_decay, optimizer_type=DEFAULT_OPTIMIZER_TYPE):
    """构造优化器，并按 norm/bias 与低维参数拆分 weight decay。

    Args:
        model: 待优化的模型。
        learning_rate: 学习率。
        weight_decay: 权重衰减系数。
        optimizer_type: 优化器类型，"adamw" 或 "adamw8bit"。

    Returns:
        (optimizer, summary): 优化器实例和参数分组摘要。

    Raises:
        ValueError: 如果优化器类型不支持或缺少依赖。
    """
    decay_parameters = []
    no_decay_parameters = []
    for parameter_name, parameter in _iter_trainable_parameters(model):
        if parameter.ndim < 2 or parameter_name.endswith(".bias") or "norm" in parameter_name.lower():
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    groups = []
    if decay_parameters:
        groups.append({"params": decay_parameters, "weight_decay": float(weight_decay)})
    if no_decay_parameters:
        groups.append({"params": no_decay_parameters, "weight_decay": 0.0})
    if not groups:
        raise ValueError("模型没有可训练参数。")

    # 根据 optimizer_type 选择优化器
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=float(learning_rate))
        optimizer_name = "AdamW"
    elif optimizer_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ValueError(
                "optimizer_type='adamw8bit' 需要 bitsandbytes 库。\n"
                "请安装: pip install bitsandbytes"
            )
        optimizer = bnb.optim.AdamW8bit(groups, lr=float(learning_rate))
        optimizer_name = "AdamW8bit"
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}，支持 'adamw' 或 'adamw8bit'。")

    summary = {
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "decay": _parameter_group_summary(decay_parameters),
        "no_decay": _parameter_group_summary(no_decay_parameters),
    }
    return optimizer, summary


def _linear_warmup_cosine_lambda(current_step, *, warmup_steps, total_steps):
    """线性 warmup 后接 cosine decay 的 LR multiplier。"""
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def _build_scheduler(optimizer, *, total_steps, warmup_ratio):
    """构造训练 scheduler；未知总步数时保持常数学习率。"""
    if total_steps is None:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    total_steps = max(1, int(total_steps))
    warmup_steps = 0
    if float(warmup_ratio) > 0.0:
        warmup_steps = int(total_steps * float(warmup_ratio))
        warmup_steps = min(max(1, warmup_steps), max(1, int(total_steps) // 2))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _linear_warmup_cosine_lambda(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )


def _resolve_batch_max_sequence_length(config, rng=None):
    """决定当前 batch 的最大长度，支持 LongRoPE2 混合窗口采样。"""
    if not config.longrope2_window_lengths:
        return config.max_sequence_length or GlobalConfig.train_max_sequence_length

    lengths = tuple(int(value) for value in config.longrope2_window_lengths)
    if rng is None:
        selected_length = max(lengths)
    else:
        weights = config.longrope2_window_weights
        # 每个 batch 采样一个窗口长度，用来覆盖 original/target/long window 混合训练策略。
        selected_length = rng.choices(lengths, weights=weights, k=1)[0]

    if config.max_sequence_length is not None:
        selected_length = min(int(config.max_sequence_length), int(selected_length))
    return selected_length


def _resolve_remaining_step_budget(total_step_budget, global_step):
    """根据已完成 global_step 计算续训后的剩余步数上限。"""
    if total_step_budget is None:
        return None
    return max(0, int(total_step_budget) - int(global_step))


def _hash_int_tuple(values):
    """对索引 tuple 生成紧凑 hash，避免把大索引列表写入进度摘要。"""
    if values is None:
        return None
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b",")
    return digest.hexdigest()


def _hash_json_payload(payload):
    """对 JSON 友好对象生成稳定 hash。"""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hash_dataset_content(dataset_path):
    """计算 JSONL 非空行内容指纹，用于续训时发现已有数据被改写。"""
    digest = hashlib.sha256()
    record_count = 0
    with Path(dataset_path).open("r", encoding="utf-8") as dataset_file:
        for raw_line in dataset_file:
            line = raw_line.strip()
            if not line:
                continue
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
            record_count += 1
    return record_count, digest.hexdigest()


@lru_cache(maxsize=8192)
def _progress_path_string(path):
    """把 data_progress 中的路径规范化为项目相对 POSIX 路径。"""
    path = Path(path)
    project_root = Path.cwd().resolve()
    try:
        resolved_path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        return resolved_path.relative_to(project_root).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _entry_progress_key(entry_plan):
    """返回与数据样本元数据一致的 entry key。"""
    return _progress_path_string(entry_plan.path)


def _entry_plan_policy(entry_plan):
    """提取会影响样本选择的 manifest 策略。"""
    return {
        "weight": float(entry_plan.weight),
        "sample_limit": entry_plan.sample_limit,
        "repeat_count": int(entry_plan.repeat_count),
        "base_indices_hash": _hash_int_tuple(entry_plan.base_indices),
        "extra_indices_hash": _hash_int_tuple(entry_plan.extra_indices),
        "selected_count": int(entry_plan.selected_count),
    }


def _build_entry_progress_template(
    entry_plan,
    config,
    *,
    active,
    source_fingerprint=None,
):
    """从当前 manifest entry plan 构造数据进度基础字段。"""
    if active:
        if source_fingerprint is None:
            source_fingerprint = _hash_dataset_content(entry_plan.path)
        source_record_count, content_hash = source_fingerprint
        if source_record_count != int(entry_plan.source_record_count):
            raise ValueError(
                f"数据集扫描结果不一致: {entry_plan.path}，"
                f"plan={entry_plan.source_record_count} fingerprint={source_record_count}"
            )
    else:
        content_hash = None
    policy = _entry_plan_policy(entry_plan)
    return {
        "entry_key": _entry_progress_key(entry_plan),
        "entry_index": int(entry_plan.entry_index),
        "entry_name": entry_plan.name,
        "path": _progress_path_string(entry_plan.path),
        "active": bool(active),
        "weight": float(entry_plan.weight),
        "sample_limit": entry_plan.sample_limit,
        "source_record_count": int(entry_plan.source_record_count),
        "source_content_hash": content_hash,
        "planned_sample_count_per_epoch": int(entry_plan.selected_count),
        "target_total_epochs": int(config.epochs),
        "target_total_samples": int(entry_plan.selected_count) * max(1, int(config.epochs)),
        "plan_policy": policy,
        "plan_fingerprint": _hash_json_payload(policy),
        "completed_epochs": 0,
        "consumed_in_current_epoch": 0,
        "total_consumed_samples": 0,
        "consumed_ordinals_in_current_epoch": set(),
    }


def _previous_progress_entries(progress):
    """把历史 data_progress entries 转成 key 索引。"""
    if not isinstance(progress, dict):
        return {}
    entries = progress.get("entries")
    if not isinstance(entries, list):
        return {}
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_key = entry.get("entry_key")
        if isinstance(entry_key, str) and entry_key:
            result[entry_key] = entry
    return result


def _recover_zero_weight_resume_progress(template, previous_entry):
    """定向修复旧暂停账本归零的采样计划，避免凭空解释部分样本游标。

    旧实现暂停数据集时会把源指纹与计划规模覆盖为零，但保留训练计数。
    只有已完成整轮、恢复全量 weight=1 且样本规模一致时，才有足够信息
    重建每轮样本规模。历史源内容指纹已经丢失，必须明确告知无法回溯核验。
    """
    if not previous_entry:
        return previous_entry
    previous_policy = previous_entry.get("plan_policy")
    if not (
        not previous_entry.get("active", True)
        and isinstance(previous_policy, dict)
        and float(previous_policy.get("weight", -1)) == 0.0
        and int(previous_policy.get("selected_count", -1)) == 0
        and int(previous_entry.get("planned_sample_count_per_epoch", -1)) == 0
        and int(previous_entry.get("total_consumed_samples", 0) or 0) > 0
    ):
        return previous_entry

    policy = template["plan_policy"]
    completed_epochs = int(previous_entry.get("completed_epochs", 0) or 0)
    raw_ordinals = previous_entry.get("consumed_ordinals_in_current_epoch", [])
    consumed_ordinals = {int(value) for value in raw_ordinals}
    consumed_current = int(previous_entry.get("consumed_in_current_epoch", 0) or 0)
    completed_samples = int(previous_entry["total_consumed_samples"]) - consumed_current
    selected_count = int(policy["selected_count"])
    # 已归零的计划无法恢复子集身份；只修复完整全量采样，保留其他保护。
    safe_to_recover = (
        completed_epochs > 0
        and completed_samples == completed_epochs * selected_count
        and consumed_current == len(consumed_ordinals)
        and all(0 <= value < selected_count for value in consumed_ordinals)
        and policy["weight"] == 1.0
        and policy["sample_limit"] is None
        and previous_policy.get("sample_limit") is None
        and policy["repeat_count"] == 1
        and policy["base_indices_hash"] is None
        and policy["extra_indices_hash"] == _hash_int_tuple(())
        and selected_count == int(template["source_record_count"])
    )
    if not safe_to_recover:
        raise ValueError(
            f"旧暂停账本缺少数据集活跃采样计划: {template['entry_name']}。"
            "无法安全恢复当前样本游标；请使用暂停前的 checkpoint，"
            "或在核对数据后重置该数据集进度。"
        )

    recovered = dict(previous_entry)
    recovered["plan_policy"] = dict(policy)
    recovered["planned_sample_count_per_epoch"] = selected_count
    recovered["source_record_count"] = int(template["source_record_count"])
    recovered["source_content_hash"] = None
    print(
        f"续训修复旧暂停账本: {template['entry_name']}。"
        "已根据完成轮次恢复全量采样计划；历史源内容指纹已丢失，"
        "无法核验暂停前的文件内容，本次将重新建立源指纹。"
    )
    return recovered


def _validate_resume_entry_progress(template, previous_entry):
    """校验续训时同一数据集的 manifest 策略和文件内容没有被改写。"""
    if not previous_entry:
        return
    previous_consumed = int(previous_entry.get("total_consumed_samples", 0) or 0)
    if previous_consumed <= 0:
        return
    previous_policy = previous_entry.get("plan_policy")
    if previous_policy is not None and previous_policy != template["plan_policy"]:
        raise ValueError(
            f"续训检测到数据集训练策略变化: {template['entry_name']}。"
            "除 weight=0/移除跳过外，已训练数据集不允许修改 weight 或 sample_limit。"
        )
    if int(previous_entry.get("source_record_count", -1)) != int(template["source_record_count"]):
        raise ValueError(
            f"续训检测到数据集行数变化: {template['path']}。"
            "请把新增内容作为新的 manifest 数据集，或重置该数据集进度后再训练。"
        )
    previous_hash = previous_entry.get("source_content_hash")
    if previous_hash and previous_hash != template["source_content_hash"]:
        raise ValueError(
            f"续训检测到数据集内容被修改: {template['path']}。"
            "为避免进度错位，请恢复原文件或重置该数据集进度。"
        )


class DataProgressTracker:
    """维护 manifest 维度的数据训练账本。"""

    def __init__(self, dataset, config, *, previous_progress=None):
        self.enabled = hasattr(dataset, "entry_plans") and hasattr(dataset, "iter_records_for_epoch")
        self.manifest_path = None if not hasattr(dataset, "manifest_path") else Path(dataset.manifest_path)
        self.target_total_epochs = max(1, int(config.epochs))
        self.entries = {}
        self.inactive_entries = {}
        if not self.enabled:
            return

        previous_entries = _previous_progress_entries(previous_progress)
        entry_plans = tuple(dataset.entry_plans)
        active_entry_plans = tuple(
            entry_plan
            for entry_plan in entry_plans
            if int(entry_plan.selected_count) > 0
        )
        fingerprint_workers = min(
            max(
                1,
                int(config.data_num_workers),
                int(config.data_reader_num_workers),
            ),
            len(active_entry_plans),
        )
        source_fingerprints = {}
        dataset_fingerprints = getattr(dataset, "source_fingerprints", None)
        if isinstance(dataset_fingerprints, dict):
            # token cache 数据集已内置轻量指纹，不再对 JSONL 全量哈希。
            source_fingerprints = dict(dataset_fingerprints)
        elif fingerprint_workers > 1:
            # 文件彼此独立，预检阶段可并行读取；结果仍按 entry 顺序消费，异常可重复。
            with ThreadPoolExecutor(
                max_workers=fingerprint_workers,
                thread_name_prefix="lpt-data-fingerprint",
            ) as executor:
                fingerprint_results = executor.map(
                    _hash_dataset_content,
                    (entry_plan.path for entry_plan in active_entry_plans),
                )
                source_fingerprints = {
                    _entry_progress_key(entry_plan): fingerprint
                    for entry_plan, fingerprint in zip(
                        active_entry_plans,
                        fingerprint_results,
                    )
                }
        current_keys = set()
        for entry_plan in entry_plans:
            entry_key = _entry_progress_key(entry_plan)
            if entry_key in current_keys:
                raise ValueError(
                    f"manifest 包含重复数据集路径，无法按 JSONL 文件维护进度: {entry_plan.path}"
                )
            current_keys.add(entry_key)
            active = int(entry_plan.selected_count) > 0
            template = _build_entry_progress_template(
                entry_plan,
                config,
                active=active,
                source_fingerprint=source_fingerprints.get(entry_key),
            )
            previous_entry = previous_entries.get(entry_key)
            if (
                getattr(dataset, "uses_token_cache", False)
                and previous_entry is not None
                and isinstance(previous_entry.get("source_content_hash"), str)
                and previous_entry["source_content_hash"] != template["source_content_hash"]
            ):
                # 旧进度记录的是 JSONL 内容哈希；cache 后端指纹不同，但源文件
                # 一致性已由 cache manifest 的 stat+头尾采样校验保证。沿用旧哈希
                # 可以避免续训误报“数据集内容被修改”。
                template["source_content_hash"] = previous_entry["source_content_hash"]
            if not active:
                inactive = dict(template)
                if previous_entry:
                    # 暂停只改变当前启用状态，必须保存最后活跃计划和源指纹。
                    # ordinal 指向该计划的样本实例；覆盖为零计划会让恢复时
                    # 无法判断游标属于哪个采样方案，也会丢失文件内容保护。
                    inactive = dict(previous_entry)
                    for key in ("entry_index", "entry_name", "path", "weight", "sample_limit"):
                        inactive[key] = template[key]
                    inactive["target_total_epochs"] = self.target_total_epochs
                    inactive["target_total_samples"] = (
                        int(inactive.get("planned_sample_count_per_epoch", 0))
                        * self.target_total_epochs
                    )
                inactive["active"] = False
                inactive["skip_reason"] = "weight_zero_or_empty"
                self.inactive_entries[entry_key] = inactive
                continue

            previous_entry = _recover_zero_weight_resume_progress(template, previous_entry)
            _validate_resume_entry_progress(template, previous_entry)
            consumed_ordinals = set()
            completed_epochs = 0
            if previous_entry:
                completed_epochs = min(
                    int(previous_entry.get("completed_epochs", 0) or 0),
                    self.target_total_epochs,
                )
                if completed_epochs < self.target_total_epochs:
                    raw_ordinals = previous_entry.get("consumed_ordinals_in_current_epoch", [])
                    if not isinstance(raw_ordinals, list):
                        raw_ordinals = []
                    consumed_ordinals = {
                        int(value)
                        for value in raw_ordinals
                        if 0 <= int(value) < int(template["planned_sample_count_per_epoch"])
                    }
            template["completed_epochs"] = completed_epochs
            template["consumed_ordinals_in_current_epoch"] = consumed_ordinals
            template["consumed_in_current_epoch"] = len(consumed_ordinals)
            template["total_consumed_samples"] = (
                completed_epochs * int(template["planned_sample_count_per_epoch"])
                + len(consumed_ordinals)
            )
            self.entries[entry_key] = template

        for entry_key, previous_entry in previous_entries.items():
            if entry_key in current_keys:
                continue
            inactive = dict(previous_entry)
            inactive["active"] = False
            inactive["skip_reason"] = "not_in_current_manifest"
            self.inactive_entries[entry_key] = inactive

    def attach_epoch(self, sample, epoch_index):
        """把当前训练 epoch 写入样本进度元数据。"""
        meta = sample.get(DATA_PROGRESS_RECORD_KEY) if isinstance(sample, dict) else None
        if not isinstance(meta, dict):
            return sample
        enriched_sample = dict(sample)
        enriched_meta = dict(meta)
        enriched_meta["epoch_index"] = int(epoch_index)
        enriched_sample[DATA_PROGRESS_RECORD_KEY] = enriched_meta
        return enriched_sample

    def should_train(self, sample):
        """判断样本是否还需要训练。"""
        meta = sample.get(DATA_PROGRESS_RECORD_KEY) if isinstance(sample, dict) else None
        if not isinstance(meta, dict):
            return True
        entry = self.entries.get(meta.get("entry_key"))
        if not entry or not entry.get("active", False):
            return False
        if int(entry["planned_sample_count_per_epoch"]) <= 0:
            return False
        completed_epochs = int(entry.get("completed_epochs", 0))
        if completed_epochs >= self.target_total_epochs:
            return False
        epoch_index = int(meta.get("epoch_index", 0))
        if epoch_index >= self.target_total_epochs:
            return False
        if epoch_index < completed_epochs:
            return False
        if epoch_index > completed_epochs:
            return True
        consumed = entry.get("consumed_ordinals_in_current_epoch", set())
        return int(meta.get("sample_ordinal", -1)) not in consumed

    def update_after_batch(self, samples):
        """训练 batch 成功后更新每个 entry 的当前 epoch 进度。"""
        if not self.enabled:
            return
        touched = set()
        for sample in samples:
            meta = sample.get(DATA_PROGRESS_RECORD_KEY) if isinstance(sample, dict) else None
            if not isinstance(meta, dict):
                continue
            entry = self.entries.get(meta.get("entry_key"))
            if not entry or not entry.get("active", False):
                continue
            epoch_index = int(meta.get("epoch_index", 0))
            while int(entry.get("completed_epochs", 0)) < epoch_index:
                entry["completed_epochs"] = int(entry.get("completed_epochs", 0)) + 1
                entry["consumed_ordinals_in_current_epoch"] = set()
            if int(entry.get("completed_epochs", 0)) != epoch_index:
                continue
            ordinal = int(meta.get("sample_ordinal", -1))
            if ordinal < 0:
                continue
            consumed = entry.setdefault("consumed_ordinals_in_current_epoch", set())
            consumed.add(ordinal)
            touched.add(meta.get("entry_key"))

        for entry_key in touched:
            entry = self.entries[entry_key]
            planned_count = int(entry["planned_sample_count_per_epoch"])
            consumed_count = len(entry.get("consumed_ordinals_in_current_epoch", []))
            if planned_count > 0 and consumed_count >= planned_count:
                entry["completed_epochs"] = min(
                    self.target_total_epochs,
                    int(entry.get("completed_epochs", 0)) + 1,
                )
                entry["consumed_ordinals_in_current_epoch"] = set()
                consumed_count = 0
            entry["consumed_in_current_epoch"] = consumed_count
            entry["total_consumed_samples"] = (
                int(entry.get("completed_epochs", 0)) * planned_count + consumed_count
            )

    def remaining_samples(self):
        """返回当前活跃 manifest 中仍需训练的样本实例数。"""
        remaining = 0
        for entry in self.entries.values():
            target_total = int(entry["target_total_samples"])
            consumed = int(entry.get("total_consumed_samples", 0))
            remaining += max(0, target_total - consumed)
        return remaining

    def summary(self):
        """生成写入 trainer_state 的数据进度摘要。"""
        if not self.enabled:
            return None
        return {
            "schema_version": DATA_PROGRESS_SCHEMA_VERSION,
            "path": DATA_PROGRESS_NAME,
            "manifest_path": None if self.manifest_path is None else _progress_path_string(self.manifest_path),
            "active_entry_count": len(self.entries),
            "inactive_entry_count": len(self.inactive_entries),
            "remaining_samples": self.remaining_samples(),
            "entries": [
                {
                    "entry_name": entry["entry_name"],
                    "path": entry["path"],
                    "active": bool(entry.get("active", False)),
                    "completed_epochs": int(entry.get("completed_epochs", 0)),
                    "consumed_in_current_epoch": int(entry.get("consumed_in_current_epoch", 0)),
                    "total_consumed_samples": int(entry.get("total_consumed_samples", 0)),
                    "target_total_samples": int(entry.get("target_total_samples", 0)),
                }
                for entry in sorted(self.entries.values(), key=lambda value: value["entry_key"])
            ],
        }

    def to_payload(self, config, *, global_step, optimizer_step, samples_seen, tokens_seen):
        """生成完整 data_progress.json 内容。"""
        if not self.enabled:
            return None
        entries = []
        for entry in sorted(self.entries.values(), key=lambda value: value["entry_key"]):
            normalized = dict(entry)
            normalized["consumed_ordinals_in_current_epoch"] = sorted(
                normalized.get("consumed_ordinals_in_current_epoch", [])
            )
            entries.append(normalized)
        for entry in sorted(self.inactive_entries.values(), key=lambda value: value.get("entry_key", "")):
            normalized = dict(entry)
            normalized["active"] = False
            normalized["consumed_ordinals_in_current_epoch"] = sorted(
                normalized.get("consumed_ordinals_in_current_epoch", [])
            )
            entries.append(normalized)
        return {
            "schema_version": DATA_PROGRESS_SCHEMA_VERSION,
            "training_stage": config.training_stage,
            "run_id": config.run_id,
            "global_step": int(global_step),
            "optimizer_step": int(optimizer_step),
            "samples_seen": int(samples_seen),
            "tokens_seen": int(tokens_seen),
            "target_total_epochs": self.target_total_epochs,
            "batch_token_budget": _resolve_dynamic_batch_token_budget(config),
            "seed": int(config.seed),
            "manifest_path": None if self.manifest_path is None else _progress_path_string(self.manifest_path),
            "source_manifest": None if config.source_manifest is None else _progress_path_string(config.source_manifest),
            "active_entry_count": len(self.entries),
            "inactive_entry_count": len(self.inactive_entries),
            "remaining_samples": self.remaining_samples(),
            "entries": entries,
        }


def _resolve_dynamic_batch_token_budget(config):
    """返回动态组批的 token 预算，与实际训练截断上限使用同一配置来源。"""
    budget = int(config.max_sequence_length or GlobalConfig.train_max_sequence_length)
    if budget < 2:
        raise ValueError("动态组批要求训练最大序列长度至少为 2。")
    return budget


def _iter_dynamic_progress_batches(
    dataset, *, epochs, progress_tracker, epoch_indices, reader_pool,
    length_bucket_window, max_sequence_length, tokenizer, thinking_mode,
    pad_to_multiple_of, length_records_only,
):
    """按真实长度流式组批，保留 epoch 边界和原样本进度身份。

    多样本批限制为 ``样本数 × 对齐后的最长长度 <= token 预算``。
    即使随后开启 packing，该约束也保守覆盖装箱后的 padding 容量。
    排序窗口只限制预读内存，批可以跨窗口继续累积，因此没有样本数上限。
    """
    budget = int(max_sequence_length)
    if budget < 2:
        raise ValueError("动态组批要求训练最大序列长度至少为 2。")
    alignment = max(1, int(pad_to_multiple_of or 1))
    selected_epochs = range(max(1, int(epochs))) if epoch_indices is None else epoch_indices

    def prepare_sample(sample):
        """缓存路径直接读精确长度；在线 chat 分词一次并交给 collator 复用。"""
        cached_length = sample.get(PRE_TOKENIZED_TOKEN_LENGTH)
        if cached_length is not None:
            return sample, min(budget, int(cached_length))
        if tokenizer is None:
            raise ValueError("动态组批的非缓存样本需要提供 tokenizer。")
        encoded = encode_training_sample(sample, tokenizer, max_length=budget, thinking_mode=thinking_mode)
        if sample.get("type") == "chat":
            # 保留原 messages 和来源信息，thinking 审计和断点进度继续读取原字段。
            sample = dict(sample)
            sample.update({
                PRE_TOKENIZED_CHAT_INPUT_IDS: encoded.input_ids,
                PRE_TOKENIZED_CHAT_LABELS: encoded.labels,
                PRE_TOKENIZED_CHAT_THINKING_MODE_IDS: encoded.thinking_mode_ids,
                PRE_TOKENIZED_CHAT_TARGET_CHANNEL_IDS: encoded.target_channel_ids,
            })
        return sample, encoded.length

    for epoch_index in selected_epochs:
        if length_records_only:
            source = dataset.iter_length_records_for_epoch(int(epoch_index))
        elif hasattr(dataset, "iter_records_for_epoch"):
            source = dataset.iter_records_for_epoch(int(epoch_index), reader_pool=reader_pool)
        else:
            source = iter(dataset)
        bucket = []
        bucket_max_length = 0
        window = []

        def iter_prepared_samples():
            """可选窗口排序与训练、计步共用，保证两次重放得到相同批边界。"""
            for sample in source:
                sample = dict(sample)
                sample[TRAINING_EPOCH_RECORD_KEY] = int(epoch_index)
                if progress_tracker is not None and progress_tracker.enabled:
                    sample = progress_tracker.attach_epoch(sample, epoch_index)
                    if not progress_tracker.should_train(sample):
                        continue
                prepared = prepare_sample(sample)
                if length_bucket_window <= 0:
                    yield prepared
                    continue
                window.append(prepared)
                if len(window) >= int(length_bucket_window):
                    yield from sorted(window, key=lambda item: item[1])
                    window.clear()
            yield from sorted(window, key=lambda item: item[1])

        for sample, length in iter_prepared_samples():
            if length < 2:
                raise ValueError("动态组批样本在截断后不足 2 个 token。")
            aligned_length = math.ceil(length / alignment) * alignment
            projected_max = max(bucket_max_length, aligned_length)
            if bucket and (len(bucket) + 1) * projected_max > budget:
                yield bucket
                bucket = []
                bucket_max_length = 0
            bucket.append(sample)
            bucket_max_length = max(bucket_max_length, aligned_length)
        if bucket:
            # 单样本始终可训练；它已经截断到训练窗口，原始 token 长度不会超预算。
            yield bucket


def _build_progress_batch_iterator(
    dataset,
    *,
    epochs,
    max_steps=None,
    progress_tracker=None,
    epoch_indices=None,
    reader_pool=None,
    length_bucket_window=0,
    max_sequence_length=None,
    tokenizer=None,
    thinking_mode=THINKING_MODE_OFF,
    pad_to_multiple_of=None,
    length_records_only=False,
):
    """按进度过滤样本，并在有界窗口内按长度排序后组织 batch。"""
    max_steps = None if max_steps is None else int(max_steps)
    if max_steps is not None and max_steps <= 0:
        return
    batches = _iter_dynamic_progress_batches(
        dataset, epochs=epochs, progress_tracker=progress_tracker,
        epoch_indices=epoch_indices, reader_pool=reader_pool,
        length_bucket_window=length_bucket_window,
        max_sequence_length=max_sequence_length or GlobalConfig.train_max_sequence_length,
        tokenizer=tokenizer, thinking_mode=thinking_mode,
        pad_to_multiple_of=pad_to_multiple_of, length_records_only=length_records_only,
    )
    for index, batch in enumerate(batches):
        yield batch
        if max_steps is not None and index + 1 >= max_steps:
            return


def _resolve_progress_training_budget(
    dataset,
    config,
    progress_tracker,
    global_step,
    *,
    data_worker_count=0,
    tokenizer=None,
    device=None,
):
    """结合数据进度和 max_steps 计算本次续训剩余 batch 数。"""
    step_limit = (
        None if config.max_steps is None
            else _resolve_remaining_step_budget(int(config.max_steps), global_step)
    )
    if progress_tracker is not None and progress_tracker.enabled and progress_tracker.remaining_samples() == 0:
        remaining_batches = 0
    elif hasattr(dataset, "iter_length_records_for_epoch"):
        # 只重放长度索引和 ordinal，不解析正文，不读取 token 数组内容。
        remaining_batches = sum(1 for _ in _build_progress_batch_iterator(
            dataset, epochs=config.epochs, max_steps=step_limit,
            progress_tracker=progress_tracker,
            length_bucket_window=config.data_length_bucket_window,
            max_sequence_length=_resolve_dynamic_batch_token_budget(config),
            length_records_only=True,
            pad_to_multiple_of=(
                GlobalConfig.pad_to_multiple_of
                if device is not None and device.type == "cuda"
                else None
            ),
        ))
    else:
        # 非缓存流没有廉价的精确长度索引，显式 max_steps 仍作为总步数上限。
        remaining_batches = step_limit
    return (
        None if remaining_batches is None else int(global_step) + remaining_batches,
        remaining_batches,
    )


def _resolve_optimizer_step_budget(optimizer_step, remaining_steps, gradient_accumulation_steps):
    """把剩余 microbatch 数换算成 scheduler 使用的 optimizer step 总预算。"""
    if remaining_steps is None:
        return None
    remaining_optimizer_steps = math.ceil(
        max(0, int(remaining_steps)) / int(gradient_accumulation_steps)
    )
    return int(optimizer_step) + remaining_optimizer_steps


def _stable_training_batch_seed(samples, base_seed):
    """根据稳定样本身份派生 batch RNG seed，保证 resume 后窗口采样一致。"""
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("ascii"))
    digest.update(b"\n")
    for sample in samples:
        progress_metadata = (
            sample.get(DATA_PROGRESS_RECORD_KEY)
            if isinstance(sample, dict)
            else None
        )
        if isinstance(progress_metadata, dict):
            identity = (
                progress_metadata.get("entry_key"),
                int(progress_metadata.get("sample_ordinal", -1)),
                int(progress_metadata.get("epoch_index", 0)),
            )
        else:
            identity = (
                sample.get("id", "<unknown>") if isinstance(sample, dict) else repr(sample),
                int(sample.get(TRAINING_EPOCH_RECORD_KEY, 0)) if isinstance(sample, dict) else 0,
            )
        digest.update(
            json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _build_batch_tensors(samples, tokenizer, config, rng=None):
    """从结构化样本构造模型训练 batch 张量。"""
    max_length = _resolve_batch_max_sequence_length(config, rng=rng)
    if config.sequence_packing:
        # packing 路径额外返回 position_ids/segment_ids，模型侧用 segment_ids 阻断跨样本注意力。
        (
            input_ids,
            labels,
            attention_mask,
            thinking_mode_ids,
            target_channel_ids,
            position_ids,
            segment_ids,
            sample_count,
        ) = build_packed_training_batch(
            samples,
            tokenizer,
            max_length=max_length,
            thinking_mode=config.thinking_mode,
            include_thinking_tensors=True,
            packing_strategy=config.data_packing_strategy,
        )
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "thinking_mode_ids": thinking_mode_ids,
            "target_channel_ids": target_channel_ids,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "sample_count": sample_count,
        }
    else:
        input_ids, labels, attention_mask, thinking_mode_ids, target_channel_ids = build_training_batch(
            samples,
            tokenizer,
            max_length=max_length,
            thinking_mode=config.thinking_mode,
            include_thinking_tensors=True,
        )
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "thinking_mode_ids": thinking_mode_ids,
            "target_channel_ids": target_channel_ids,
            "position_ids": None,
            "segment_ids": None,
            "sample_count": len(samples),
        }
    # 这两个标量在 CPU worker 中计算，避免训练主循环每 batch 强制同步 CUDA。
    batch["token_count"] = int(batch["attention_mask"].sum().item())
    batch["valid_target_count"] = int((batch["labels"][:, 1:] != -100).sum().item())
    if batch["position_ids"] is None:
        batch["rope_sequence_length"] = int(
            batch["attention_mask"].sum(dim=1).max().item()
        )
    else:
        batch["rope_sequence_length"] = int(batch["position_ids"].max().item()) + 1
    return batch


class _TrainingBatchCollator:
    """在 spawn worker 内完成一个全局 batch 的分词、packing 和张量化。"""

    def __init__(self, tokenizer, config, *, target_device_type):
        self.tokenizer = tokenizer
        self.config = config
        self.target_device_type = str(target_device_type)
        self.chat_template_version = GlobalConfig.chat_template_version
        self.pad_to_multiple_of = GlobalConfig.pad_to_multiple_of
        self.train_max_sequence_length = GlobalConfig.train_max_sequence_length

    def __call__(self, samples):
        # spawn worker 会重新导入模块；显式恢复影响 batch 形状和模板的运行配置。
        GlobalConfig.device = torch.device(self.target_device_type)
        GlobalConfig.chat_template_version = self.chat_template_version
        GlobalConfig.pad_to_multiple_of = self.pad_to_multiple_of
        GlobalConfig.train_max_sequence_length = self.train_max_sequence_length
        batch = _build_batch_tensors(
            samples,
            self.tokenizer,
            self.config,
            rng=random.Random(
                _stable_training_batch_seed(samples, self.config.seed)
            ),
        )
        progress_samples = []
        for sample in samples:
            progress_metadata = sample.get(DATA_PROGRESS_RECORD_KEY)
            if isinstance(progress_metadata, dict):
                progress_samples.append(
                    {DATA_PROGRESS_RECORD_KEY: dict(progress_metadata)}
                )
        batch["_progress_samples"] = tuple(progress_samples)
        thinking_delta = _new_thinking_training_summary(self.config)
        _update_thinking_training_summary(
            thinking_delta,
            samples,
            thinking_mode=self.config.thinking_mode,
        )
        batch["_thinking_summary_delta"] = {
                key: int(thinking_delta[key])
                for key in THINKING_SUMMARY_COUNTER_KEYS
        }
        return batch


def _pin_training_batch(batch):
    """在主进程锁定最终 batch 张量，供 CUDA copy stream 异步搬运。"""
    return {
        key: value.pin_memory() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


_TRAINING_BATCH_COLLATOR = None


def _initialize_training_data_worker(tokenizer, config, target_device_type):
    """初始化 spawn 批处理进程，并限制每个进程只占一个 torch 计算线程。"""
    global _TRAINING_BATCH_COLLATOR
    torch.set_num_threads(1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _TRAINING_BATCH_COLLATOR = _TrainingBatchCollator(
        tokenizer,
        config,
        target_device_type=target_device_type,
    )


def _build_training_batch_in_worker(samples):
    """执行已在 initializer 中缓存的 worker collator。"""
    if _TRAINING_BATCH_COLLATOR is None:
        raise RuntimeError("训练批处理 worker 尚未初始化。")
    return _TRAINING_BATCH_COLLATOR(samples)


def _resolve_data_worker_count(dataset, config, progress_tracker):
    """决定批处理 worker 数；单个大 JSONL 也可以把不同 batch 分发到多核。"""
    requested_workers = max(0, int(config.data_num_workers))
    if requested_workers == 0:
        return 0
    if getattr(dataset, "uses_token_cache", False):
        # mmap numpy 样本跨 spawn 进程会产生 IPC 序列化副本，因此 cache
        # 后端改用同进程线程池。numpy/torch 张量化会释放 GIL，既能与主训练
        # 线程重叠，又不会为每批复制整份 token 数组。
        available_cpus = max(1, int(os.cpu_count() or 1))
        return min(requested_workers, available_cpus)
    if progress_tracker is None or not progress_tracker.enabled:
        # 普通可迭代对象未必支持跨进程稳定分片，保留单进程兼容语义。
        return 0
    available_cpus = max(1, int(os.cpu_count() or 1))
    return min(requested_workers, available_cpus)


def _resolve_data_reader_worker_count(dataset, config, progress_tracker, batch_worker_count):
    """决定原始 JSONL 解析 worker 数，并为 batch worker 留出 CPU。"""
    if getattr(dataset, "uses_token_cache", False):
        # token cache 数据集从 mmap 读取，不需要 JSONL reader 进程池。
        return 0
    requested_workers = max(0, int(config.data_reader_num_workers))
    if requested_workers == 0:
        return 0
    if progress_tracker is None or not progress_tracker.enabled:
        # 普通 list/自定义 IterableDataset 没有稳定的 manifest 文件边界，
        # 不把它们强行送入 JSONL reader。
        return 0
    entry_plans = getattr(dataset, "entry_plans", ())
    active_paths = {
        str(Path(entry_plan.path).resolve())
        for entry_plan in entry_plans
        if int(getattr(entry_plan, "selected_count", 0)) > 0
    }
    if not active_paths:
        return 0
    available_cpus = max(1, int(os.cpu_count() or 1))
    spare_cpus = max(1, available_cpus - max(0, int(batch_worker_count)))
    return min(requested_workers, spare_cpus, len(active_paths) if len(active_paths) > 1 else requested_workers)


def _build_jsonl_reader_pool(dataset, config, progress_tracker, batch_worker_count):
    """为流式 manifest 建立一个跨 epoch 复用的 JSONL reader pool。"""
    if getattr(dataset, "uses_token_cache", False):
        return None, 0
    reader_worker_count = _resolve_data_reader_worker_count(
        dataset,
        config,
        progress_tracker,
        batch_worker_count,
    )
    if reader_worker_count <= 0:
        return None, 0
    entry_plans = getattr(dataset, "entry_plans", ())
    paths = tuple(
        dict.fromkeys(
            Path(entry_plan.path)
            for entry_plan in entry_plans
            if int(getattr(entry_plan, "selected_count", 0)) > 0
        )
    )
    if not paths:
        return None, 0
    pool = JSONLReaderPool(
        paths,
        expected_types=getattr(dataset, "expected_types", None),
        num_workers=reader_worker_count,
        chunk_bytes=config.data_reader_chunk_bytes,
        max_in_flight=max(
            reader_worker_count,
            reader_worker_count * max(1, int(config.data_prefetch_factor)),
        ),
    )
    return pool, reader_worker_count


def _iter_parallel_training_batches(
    dataset,
    tokenizer,
    config,
    progress_tracker,
    device,
    *,
    worker_count,
):
    """按稳定全局 batch 边界，有序并发执行分词、packing 与张量化。"""
    worker_count = max(0, int(worker_count))
    uses_token_cache = bool(getattr(dataset, "uses_token_cache", False))
    pin_memory = bool(config.data_pin_memory and device.type == "cuda")
    reader_pool, _reader_worker_count = _build_jsonl_reader_pool(
        dataset,
        config,
        progress_tracker,
        worker_count,
    )

    def build_source_batches(epoch_indices=None):
        return _build_progress_batch_iterator(
            dataset,
            epochs=config.epochs,
            progress_tracker=progress_tracker,
            epoch_indices=epoch_indices,
            reader_pool=reader_pool,
            length_bucket_window=config.data_length_bucket_window,
            max_sequence_length=config.max_sequence_length,
            tokenizer=tokenizer,
            thinking_mode=config.thinking_mode,
            pad_to_multiple_of=GlobalConfig.pad_to_multiple_of if device.type == "cuda" else None,
        )

    collator = _TrainingBatchCollator(
        tokenizer,
        config,
        target_device_type=device.type,
    )

    def process_source_batches(source_batches):
        """把 reader 产出的结构化 batch 送入分词/packing worker。"""
        if uses_token_cache:
            # cache 样本已预编码且来自 mmap；线程池避免 Windows spawn 与
            # numpy IPC 副本，同时保留有界预取和稳定的 batch 返回顺序。
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="lpt-token-cache-batch",
            )
            build_batch = collator
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_training_data_worker,
                initargs=(tokenizer, config, device.type),
            )
            build_batch = _build_training_batch_in_worker
        pending = deque()
        source_exhausted = False

        def submit_next():
            nonlocal source_exhausted
            if source_exhausted:
                return False
            try:
                samples = next(source_batches)
            except StopIteration:
                source_exhausted = True
                return False
            pending.append(
                executor.submit(build_batch, tuple(samples))
            )
            return True

        try:
            for _ in range(prefetch_batches):
                if not submit_next():
                    break
            while pending:
                future = pending.popleft()
                batch = future.result()
                submit_next()
                yield _pin_training_batch(batch) if pin_memory else batch
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    try:
        if worker_count == 0:
            for samples in build_source_batches():
                batch = collator(tuple(samples))
                yield _pin_training_batch(batch) if pin_memory else batch
            return

        prefetch_batches = max(
            worker_count,
            worker_count * int(config.data_prefetch_factor),
        )
        if config.data_persistent_workers:
            yield from process_source_batches(build_source_batches())
            return
        for epoch_index in range(max(1, int(config.epochs))):
            yield from process_source_batches(
                build_source_batches(epoch_indices=(epoch_index,))
            )
    finally:
        if reader_pool is not None:
            reader_pool.close()


def _build_data_pipeline_runtime_metadata(
    config,
    worker_count,
    pin_memory,
    reader_worker_count=0,
    worker_backend=None,
):
    """记录请求配置与实际生效的数据流水线参数，便于 checkpoint 审计。"""
    worker_count = max(0, int(worker_count))
    worker_backend = None if worker_count == 0 else str(worker_backend or "process")
    worker_location = {
        "thread": "batch_thread",
        "process": "batch_worker",
    }.get(worker_backend, "batch_worker" if worker_count > 0 else "training_main")
    return {
        "requested_num_workers": int(config.data_num_workers),
        "effective_num_workers": worker_count,
        "requested_reader_num_workers": int(config.data_reader_num_workers),
        "effective_reader_num_workers": int(reader_worker_count),
        "requested_reader_chunk_bytes": int(config.data_reader_chunk_bytes),
        "effective_reader_chunk_bytes": (
            int(config.data_reader_chunk_bytes) if reader_worker_count > 0 else 0
        ),
        "requested_prefetch_factor": int(config.data_prefetch_factor),
        "effective_prefetch_factor": (
            int(config.data_prefetch_factor) if worker_count > 0 else 0
        ),
        "requested_persistent_workers": bool(config.data_persistent_workers),
        "effective_persistent_workers": bool(
            config.data_persistent_workers and worker_count > 0
        ),
        "requested_pin_memory": bool(config.data_pin_memory),
        "effective_pin_memory": bool(pin_memory),
        "worker_backend": worker_backend,
        "multiprocessing_context": (
            "spawn" if worker_count > 0 and worker_backend == "process" else None
        ),
        "data_read_location": "reader_worker" if reader_worker_count > 0 else "training_main",
        "sample_encoding_location": worker_location,
        "batch_build_location": worker_location,
        "entry_sharding": None,
        "async_h2d_prefetch": bool(pin_memory),
        "length_bucket_window": int(config.data_length_bucket_window),
        "batch_token_budget": _resolve_dynamic_batch_token_budget(config),
        "length_bucket_strategy": (
            "bounded_character_sort" if int(config.data_length_bucket_window) > 0 else "sequential"
        ),
        "packing_strategy": str(config.data_packing_strategy),
        "token_cache_enabled": bool(getattr(config, "token_cache_enabled", False)),
        "token_cache_dir": (
            str(config.token_cache_dir)
            if getattr(config, "token_cache_dir", None) is not None
            else None
        ),
    }


def _move_batch(batch, device, *, non_blocking=False):
    """把 batch 内 tensor 移动到目标设备，非 tensor 元数据保持原样。"""
    moved = {}
    for key, value in batch.items():
        moved[key] = (
            value.to(device, non_blocking=bool(non_blocking))
            if isinstance(value, torch.Tensor)
            else value
        )
    return moved


def _iter_device_batches(data_iterator, device, *, async_h2d=False):
    """把 CPU batch 搬到目标设备；CUDA 下提前一批重叠 H2D 与模型计算。"""
    if device.type != "cuda" or not async_h2d:
        for cpu_batch in data_iterator:
            yield _move_batch(cpu_batch, device, non_blocking=False)
        return

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)
    source_iterator = iter(data_iterator)

    def preload():
        try:
            cpu_batch = next(source_iterator)
        except StopIteration:
            return None
        with torch.cuda.stream(copy_stream):
            return _move_batch(cpu_batch, device, non_blocking=True)

    prefetched_batch = preload()
    while prefetched_batch is not None:
        compute_stream.wait_stream(copy_stream)
        current_batch = prefetched_batch
        for value in current_batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(compute_stream)
        # wait_stream 只等待此前已提交的 copy；此处提交下一批后可与当前计算重叠。
        prefetched_batch = preload()
        yield current_batch


def _compute_lm_loss(
    logits,
    labels,
    *,
    chunk_tokens=LM_LOSS_CHUNK_TOKENS,
    valid_target_count=None,
):
    """计算自回归 LM loss，并分块降低大 vocab 下的峰值显存。"""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    if valid_target_count is None:
        valid_target_count = int((shift_labels != -100).sum().item())
    else:
        valid_target_count = int(valid_target_count)
    if valid_target_count == 0:
        return None, valid_target_count

    chunk_tokens = max(1, int(chunk_tokens))
    # 展平为 (总token数, vocab) 后只保留一层分块循环：既能控制峰值显存
    # （chunk_tokens x vocab 的 fp32 临时张量），又比“按 batch 行再按 chunk”
    # 的双层 Python 循环少一层调度，缓解训练主进程 CPU 单核瓶颈。
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)
    loss_sum = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_logits.size(0), chunk_tokens):
        end = min(start + chunk_tokens, flat_logits.size(0))
        chunk_labels = flat_labels[start:end]
        # reduction="sum" 后统一除以有效 target 数，避免不同 chunk 有效 token 数不等导致加权偏差。
        chunk_logits = flat_logits[start:end]
        # CUDA 下 autocast 的 bf16/fp16 logits 直接交给 cross_entropy（内部按
        # fp32 计算），避免每个 chunk 额外创建 fp32 拷贝：峰值显存减半、kernel
        # 数减半，显存受限场景下的训练吞吐明显改善。CPU/fp32 路径保持 float()
        # 兼容（对 fp32 是 no-op）。
        chunk_input = (
            chunk_logits
            if chunk_logits.is_cuda
            and chunk_logits.dtype in (torch.float16, torch.bfloat16)
            else chunk_logits.float()
        )
        chunk_loss = F.cross_entropy(
            chunk_input,
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        loss_sum = loss_sum + chunk_loss
    return loss_sum / valid_target_count, valid_target_count


def _forward_batch(model, batch):
    """训练前向入口：明确关闭 KV cache，避免训练 K/V 写入 Paged KV 页池。"""
    return model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        segment_ids=batch["segment_ids"],
        thinking_mode_ids=batch.get("thinking_mode_ids"),
        target_channel_ids=batch.get("target_channel_ids"),
        rope_cache_scope="train",
        request_id="train",
        use_kv_cache=False,
        collect_runtime_state=False,
        rope_sequence_length=batch["rope_sequence_length"],
        validate_control_ids=False,
    )


def _autocast_enabled(device):
    """判断当前设备和全局 dtype 是否允许 autocast。"""
    return device.type == "cuda" and GlobalConfig.autocast_dtype in {
        torch.float16,
        torch.bfloat16,
    }


def _compute_grad_norm(parameters):
    """在参数设备上计算 L2 grad norm，避免逐参数回读 CUDA 标量。"""
    parameter_norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norms.append(parameter.grad.detach().float().norm(2))
    if not parameter_norms:
        return None
    if len(parameter_norms) == 1:
        return parameter_norms[0]
    aggregation_device = parameter_norms[0].device
    return torch.stack(
        [norm.to(device=aggregation_device) for norm in parameter_norms]
    ).norm(2)


def _scalar_to_float(value):
    """只在日志或 checkpoint 边界把设备标量转换为 Python 数值。"""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _rescale_partial_accumulated_gradients(parameters, *, completed_steps, target_steps):
    """把尾部 k/G 累积梯度恢复为按实际 k 个 microbatch 求平均的尺度。"""
    completed_steps = int(completed_steps)
    target_steps = int(target_steps)
    if completed_steps <= 0 or completed_steps >= target_steps:
        return
    scale = target_steps / completed_steps
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)


def _reset_cuda_peak_memory_stats(device):
    """重置 CUDA peak memory 计数，CPU 训练时为空操作。"""
    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)


def _collect_cuda_memory_metrics(device):
    """采集训练资源指标，定位长样本或大 vocab loss 导致的显存峰值。"""
    if device.type != "cuda":
        return {}
    to_mib = 1024 * 1024
    return {
        "cuda_memory_allocated_mib": torch.cuda.memory_allocated(device) / to_mib,
        "cuda_memory_reserved_mib": torch.cuda.memory_reserved(device) / to_mib,
        "cuda_peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / to_mib,
        "cuda_peak_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / to_mib,
    }


@torch.no_grad()
def _evaluate_model(model, tokenizer, eval_dataset, config, device):
    """周期性验证，复用训练 batch 构造逻辑但不更新权重。"""
    if eval_dataset is None:
        return None
    was_training = model.training
    model.eval()
    losses = []
    token_count = 0
    max_eval_batches = None if config.eval_max_batches is None else int(config.eval_max_batches)
    eval_batch_size = int(config.eval_batch_size or 1)
    for batch_index, samples in enumerate(
        _batch_iterator(
            eval_dataset,
            batch_size=eval_batch_size,
            epochs=1,
            max_steps=max_eval_batches,
        ),
        start=1,
    ):
        # eval 使用相同 sequence packing 和 LongRoPE2 最大长度配置，确保验证口径接近训练。
        batch = _move_batch(_build_batch_tensors(samples, tokenizer, config), device)
        with torch.autocast(
            device_type=device.type,
            dtype=GlobalConfig.autocast_dtype,
            enabled=_autocast_enabled(device),
        ):
            logits, _states = _forward_batch(model, batch)
            loss, valid_targets = _compute_lm_loss(
                logits,
                batch["labels"],
                valid_target_count=batch["valid_target_count"],
                chunk_tokens=config.lm_loss_chunk_tokens,
            )
        if loss is not None:
            losses.append(loss.detach())
            token_count += valid_targets
        # eval 不需要保留 logits 或辅助状态；及时释放，避免长样本影响后续短样本显存。
        del logits, _states, loss, batch
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
    if was_training:
        model.train()
    if not losses:
        return None
    # 验证阶段只在汇总时同步一次，避免每个 eval batch 都阻塞主线程。
    eval_loss = float(torch.stack(losses).mean().cpu())
    return {
        "eval_loss": eval_loss,
        "eval_ppl": math.exp(eval_loss) if eval_loss < 80 else float("inf"),
        "eval_target_tokens": token_count,
        "eval_batches": len(losses),
    }


def _write_metrics(artifact_dir, metric):
    """把指标追加写入 metrics.jsonl。"""
    metric_path = Path(artifact_dir) / METRICS_JSONL_NAME
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def _display_metric_name(key):
    """把内部指标名转换为控制台展示名。"""
    return METRIC_DISPLAY_NAMES.get(str(key), f"{key}({key})")


def _display_phase_name(namespace):
    """把指标 namespace 转成中文阶段名。"""
    return PHASE_DISPLAY_NAMES.get(str(namespace), f"{namespace}({namespace})")


def _format_metric_value(value):
    """统一控制台指标值格式。"""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_metric_record(namespace, metric):
    """按稳定字段顺序输出训练/验证指标。"""
    ordered_keys = [
        "stage",
        "global_step",
        "optimizer_step",
        "loss",
        "eval_loss",
        "eval_ppl",
        "learning_rate",
        "tokens_per_sec",
        "samples_per_sec",
        "grad_norm",
        "tokens_seen",
        "samples_seen",
        "sequence_length",
        "target_tokens",
        "eval_target_tokens",
        "eval_batches",
        "cuda_memory_allocated_mib",
        "cuda_memory_reserved_mib",
        "cuda_peak_memory_allocated_mib",
        "cuda_peak_memory_reserved_mib",
        "elapsed_seconds",
    ]
    keys = [key for key in ordered_keys if key in metric]
    keys.extend(key for key in metric if key not in set(keys))
    line = " ".join(
        f"{_display_metric_name(key)}={_format_metric_value(metric[key])}"
        for key in keys
    )
    tqdm.write(f"[{_display_phase_name(namespace)}] {line}")


def _build_tensorboard_writer(config):
    """按配置创建 TensorBoard writer；依赖缺失或初始化失败时自动降级。"""
    if not config.tensorboard_enabled:
        return None
    if SummaryWriter is None:
        return None
    tensorboard_dir = config.tensorboard_dir or Path(config.artifact_dir) / "tensorboard"
    try:
        return _SafeTensorBoardWriter(tensorboard_dir)
    except Exception as exc:
        tqdm.write(f"警告: TensorBoard 初始化失败，已关闭 TensorBoard 输出: {exc}")
        return None


class _SafeTensorBoardWriter:
    """隔离 TensorBoard 异步写入失败，避免辅助日志中断训练。"""

    def __init__(self, log_dir):
        """延迟持有真实 SummaryWriter，并记录失败状态。"""
        self.log_dir = Path(log_dir)
        self.disabled = False
        self._warning_emitted = False
        self._writer = None
        self._open_writer()

    def _open_writer(self):
        """创建真实 SummaryWriter。"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir), max_queue=1)

    def _warn_once(self, exc):
        """TensorBoard 失败只提示一次，避免训练日志被刷屏。"""
        if self._warning_emitted:
            return
        tqdm.write(f"警告: TensorBoard 写入失败，已关闭 TensorBoard 输出: {exc}")
        self._warning_emitted = True

    def _close_writer(self):
        """关闭底层 writer，忽略辅助日志清理异常。"""
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception:
            pass
        self._writer = None

    def _disable(self, exc):
        """关闭 TensorBoard 输出，但不中断训练主流程。"""
        self.disabled = True
        self._warn_once(exc)
        self._close_writer()

    def add_scalar(self, tag, value, step):
        """写入单个 scalar；失败时自动降级为禁用状态。"""
        if self.disabled:
            return
        try:
            # TensorBoard 后台线程按文件名追加写入，目录若被外部清理需先补回。
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer.add_scalar(tag, value, step)
            self._writer.flush()
        except Exception as exc:
            self._disable(exc)

    def flush(self):
        """刷新 TensorBoard 缓冲区。"""
        if self.disabled or self._writer is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer.flush()
        except Exception as exc:
            self._disable(exc)

    def close(self):
        """关闭 TensorBoard writer。"""
        self._close_writer()


def _disable_tensorboard_writer(writer, exc):
    """兼容安全 writer 和原始 writer 的禁用逻辑。"""
    if hasattr(writer, "_disable"):
        writer._disable(exc)
        return
    tqdm.write(f"警告: TensorBoard 写入失败，已跳过本次 TensorBoard 输出: {exc}")


def _write_tensorboard_metrics(writer, namespace, metric):
    """把数值型指标写入 TensorBoard。"""
    if writer is None:
        return
    step = int(metric.get("global_step", 0))
    phase_name = _display_phase_name(namespace)
    for key, value in metric.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            writer.add_scalar(f"{phase_name}/{_display_metric_name(key)}", float(value), step)
        except Exception as exc:
            _disable_tensorboard_writer(writer, exc)
            return


def _write_metric_outputs(config, namespace, metric, writer):
    """同时写 JSONL、控制台和 TensorBoard。"""
    metric = dict(metric)
    # 同一个 artifact 目录可能被多次启动追加；写入运行身份和墙钟时间后，
    # 可以区分正常 resume、重复启动和两个训练进程同时争用 GPU 的情况。
    metric.setdefault("run_id", config.run_id)
    metric.setdefault("process_id", os.getpid())
    metric.setdefault("wall_time_unix", round(time(), 4))
    _write_metrics(config.artifact_dir, metric)
    _print_metric_record(namespace, metric)
    _write_tensorboard_metrics(writer, namespace, metric)


def _save_trainer_state(checkpoint_root, state):
    """保存 trainer_state.json，记录可续训元数据。"""
    state_path = _trainer_state_file(checkpoint_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_data_progress(checkpoint_root, data_progress):
    """保存 data_progress.json，记录每个 manifest entry 的续训进度。"""
    if data_progress is None:
        return None
    progress_path = _data_progress_file(checkpoint_root)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(
        progress_path,
        json.dumps(data_progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_artifact_data_progress_snapshot(config, data_progress):
    """在 artifact/config 下保存 latest 数据进度快照，方便人工审计。"""
    if data_progress is None:
        return None
    snapshot_path = Path(config.artifact_dir) / "config" / DATA_PROGRESS_NAME
    return atomic_write_text(
        snapshot_path,
        json.dumps(data_progress, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_checkpoint_manifest(checkpoint_root, config, trainer_state, file_names):
    """构造 checkpoint_manifest，用于发布前后完整性校验。"""
    return {
        "checkpoint_format": TRAINING_CHECKPOINT_FORMAT,
        "checkpoint_schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "training_stage": config.training_stage,
        "lora_mode": bool(config.lora_mode),
        "global_step": int(trainer_state.get("global_step", 0)),
        "optimizer_step": int(trainer_state.get("optimizer_step", 0)),
        "files": [
            {
                "name": name,
                "size_bytes": int((Path(checkpoint_root) / name).stat().st_size),
            }
            for name in file_names
        ],
    }


def _save_checkpoint_manifest(checkpoint_root, config, trainer_state, file_names):
    """保存 checkpoint manifest。"""
    manifest_path = _checkpoint_manifest_file(checkpoint_root)
    manifest = _build_checkpoint_manifest(checkpoint_root, config, trainer_state, file_names)
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _staging_checkpoint_root(checkpoint_root):
    """生成 checkpoint 暂存目录，发布成功前不覆盖正式目录。"""
    checkpoint_root = Path(checkpoint_root)
    return checkpoint_root.with_name(f"{checkpoint_root.name}.staging.{uuid4().hex}")


def _checkpoint_previous_root(checkpoint_root):
    """返回 latest 轮转备份目录。"""
    checkpoint_root = Path(checkpoint_root)
    return checkpoint_root.with_name(f"{checkpoint_root.name}_previous")


def _remove_path(path):
    """删除文件或目录；仅用于 checkpoint 暂存/轮转路径。"""
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_checkpoint_root(staging_root, target_root, *, rotate_existing):
    """把暂存 checkpoint 原子发布到目标目录。"""
    staging_root = Path(staging_root)
    target_root = Path(target_root)
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if rotate_existing and target_root.exists():
        previous_root = _checkpoint_previous_root(target_root)
        _remove_path(previous_root)
        # latest 目录先轮转成 latest_previous，当前 latest 校验通过后再整体 rename 进来。
        target_root.rename(previous_root)
    elif target_root.exists():
        raise FileExistsError(f"目标 checkpoint 已存在: {target_root}")
    staging_root.rename(target_root)


def _cleanup_checkpoint_root(path):
    """保存失败时清理暂存目录；清理异常不遮盖原始错误。"""
    try:
        _remove_path(path)
    except OSError:
        pass


def _capture_training_rng_state():
    """捕获 Python、CPU torch 与当前 CUDA 设备随机状态。"""
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_training_rng_state(payload):
    """恢复 checkpoint 随机状态；CUDA 设备数变化时恢复可对应的前缀设备。"""
    if not isinstance(payload, dict):
        return False
    python_state = payload.get("python")
    torch_cpu_state = payload.get("torch_cpu")
    if python_state is not None:
        random.setstate(python_state)
    if isinstance(torch_cpu_state, torch.Tensor):
        torch.set_rng_state(torch_cpu_state.cpu())
    if torch.cuda.is_available():
        cuda_states = payload.get("torch_cuda") or ()
        for device_index, cuda_state in enumerate(
            tuple(cuda_states)[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)
    return True


def _load_training_rng_state(checkpoint_root):
    """按需加载 RNG 文件，兼容尚未包含该文件的早期 v2 checkpoint。"""
    rng_path = Path(checkpoint_root) / RNG_STATE_NAME
    if not rng_path.is_file():
        return False
    payload = torch.load(rng_path, map_location="cpu", weights_only=False)
    return _restore_training_rng_state(payload)


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    config,
    trainer_state,
    *,
    is_latest=True,
    checkpoint_root=None,
    data_progress=None,
):
    """保存训练 checkpoint，并在暂存目录校验通过后发布。"""
    checkpoint_root = Path(config.checkpoint_dir) if checkpoint_root is None else Path(checkpoint_root)
    if not is_latest:
        checkpoint_root = checkpoint_root.parent / f"step_{trainer_state['global_step']:06d}"
    staging_root = _staging_checkpoint_root(checkpoint_root)
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        file_names = []
        if config.lora_mode:
            # LoRA 训练只保存 adapter 权重；base 模型由训练入口的 initial checkpoint 绑定。
            save_lora_adapter_state(
                model,
                staging_root / LORA_ADAPTER_CHECKPOINT_NAME,
                extra_metadata=trainer_state,
            )
            file_names.append(LORA_ADAPTER_CHECKPOINT_NAME)
        else:
            # 非 LoRA 路径保存完整 v2 checkpoint，包含严格 schema 与完整 ModelConfig 快照。
            save_lpt_v2_checkpoint(
                model,
                staging_root / MODEL_CHECKPOINT_NAME,
                extra_metadata=trainer_state,
            )
            file_names.append(MODEL_CHECKPOINT_NAME)
        if config.save_optimizer:
            atomic_torch_save(optimizer.state_dict(), staging_root / OPTIMIZER_CHECKPOINT_NAME)
            file_names.append(OPTIMIZER_CHECKPOINT_NAME)
        if config.save_scheduler and scheduler is not None:
            atomic_torch_save(scheduler.state_dict(), staging_root / SCHEDULER_CHECKPOINT_NAME)
            file_names.append(SCHEDULER_CHECKPOINT_NAME)
        atomic_torch_save(_capture_training_rng_state(), staging_root / RNG_STATE_NAME)
        file_names.append(RNG_STATE_NAME)
        _save_trainer_state(staging_root, trainer_state)
        file_names.append(TRAINER_STATE_NAME)
        if data_progress is not None:
            _save_data_progress(staging_root, data_progress)
            file_names.append(DATA_PROGRESS_NAME)
        _save_checkpoint_manifest(staging_root, config, trainer_state, file_names)
        if not _is_valid_training_checkpoint_root(
            staging_root,
            lora_mode=bool(config.lora_mode),
            require_optimizer=bool(config.save_optimizer),
        ):
            raise RuntimeError(f"暂存 checkpoint 校验失败: {staging_root}")
        # 只有暂存目录自校验通过，才发布为 latest/step，避免中断进程留下半成品。
        _publish_checkpoint_root(staging_root, checkpoint_root, rotate_existing=bool(is_latest))
        model.config.save_json(Path(config.artifact_dir) / "config" / "model_config.json")
        if data_progress is not None and Path(checkpoint_root) == Path(config.checkpoint_dir):
            _save_artifact_data_progress_snapshot(config, data_progress)
        return checkpoint_root
    except Exception:
        _cleanup_checkpoint_root(staging_root)
        raise


def _best_checkpoint_root(config):
    """返回 best checkpoint 目录。"""
    if config.best_checkpoint_dir is not None:
        return Path(config.best_checkpoint_dir)
    return Path(config.checkpoint_dir).parent / "best_loss"


def _extract_best_metric_value(config, trainer_state):
    """从 trainer_state 中读取用于 best checkpoint 判断的指标。"""
    if str(config.best_checkpoint_metric) == "loss":
        value = trainer_state.get("last_loss")
    elif str(config.best_checkpoint_metric) == "eval_loss":
        value = trainer_state.get("latest_eval_loss")
    else:
        raise ValueError(f"不支持的 best checkpoint 指标: {config.best_checkpoint_metric}")
    if value is None:
        return None
    return float(value)


def _is_better_checkpoint_metric(current_value, best_value, *, min_delta):
    """判断当前指标是否达到更优 checkpoint 标准。"""
    if current_value is None:
        return False
    if best_value is None:
        return True
    return float(current_value) < float(best_value) - float(min_delta)


def _attach_best_checkpoint_state(config, trainer_state, *, best_value, best_global_step):
    """把当前 best checkpoint 摘要写回 trainer_state。"""
    if best_value is None:
        trainer_state["best_checkpoint"] = None
        return trainer_state
    trainer_state["best_checkpoint"] = {
        "metric": str(config.best_checkpoint_metric),
        "mode": "min",
        "value": float(best_value),
        "global_step": int(best_global_step if best_global_step is not None else trainer_state.get("global_step", 0)),
        "path": str(_best_checkpoint_root(config)),
    }
    return trainer_state


def _save_inference_weight(model, config, trainer_state):
    """保存推理加载用权重；完整 checkpoint 作为旁路审计产物一并保存。"""
    if config.lora_mode:
        return save_lora_adapter_state(
            model,
            config.inference_weight_path,
            extra_metadata=trainer_state,
        )
    path = Path(config.inference_weight_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(model.state_dict(), path)
    save_lpt_v2_checkpoint(
        model,
        path.with_name("model_checkpoint.pt"),
        extra_metadata=trainer_state,
    )
    return path


def _serialize_training_config(config):
    """把 TrainingRunConfig 转成 JSON 友好的字典。"""
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }


def _build_longrope2_training_strategy(config):
    """提取本次训练实际使用的 LongRoPE2 窗口策略。"""
    return {
        "max_sequence_length": config.max_sequence_length,
        "window_lengths": None if config.longrope2_window_lengths is None else list(config.longrope2_window_lengths),
        "window_weights": None if config.longrope2_window_weights is None else list(config.longrope2_window_weights),
        "sequence_packing": bool(config.sequence_packing),
    }


def _new_thinking_training_summary(config):
    """创建本次训练累计 thinking 统计。"""
    return {
        "schema_version": 1,
        "legacy_text_tags_supported": False,
        "training_thinking_mode": config.thinking_mode,
        "training_thinking_visibility": config.thinking_visibility,
        "training_visibility_affects_loss": False,
        "text_samples": 0,
        "chat_samples": 0,
        "assistant_messages": 0,
        "assistant_messages_with_thinking_field": 0,
        "assistant_messages_with_nonempty_thinking": 0,
        "assistant_messages_with_empty_thinking": 0,
        "assistant_messages_thinking_on": 0,
        "assistant_messages_thinking_off": 0,
    }


def _resolve_training_assistant_thinking_mode(training_mode, message):
    """按训练策略解析单条 assistant 消息的实际 thinking 分支。"""
    mode = normalize_thinking_mode(training_mode)
    if mode in {THINKING_MODE_ON, THINKING_MODE_OFF}:
        return mode
    return THINKING_MODE_ON if str(message.get("thinking") or "").strip() else THINKING_MODE_OFF


def _update_thinking_training_summary(summary, samples, *, thinking_mode):
    """按 batch 样本更新原生 thinking 统计。"""
    for sample in samples:
        sample_type = sample.get("type")
        if sample_type == "text":
            summary["text_samples"] += 1
            continue
        if sample_type != "chat":
            continue
        summary["chat_samples"] += 1
        cached_summary = sample.get(PRE_TOKENIZED_CHAT_THINKING_SUMMARY)
        if isinstance(cached_summary, dict):
            # 预编码缓存不保留完整 messages；直接累计构建时保存的同义统计。
            for key in THINKING_SUMMARY_COUNTER_KEYS:
                if key not in {"text_samples", "chat_samples"}:
                    summary[key] += int(cached_summary.get(key, 0))
            continue
        for message in sample.get("messages", ()):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            summary["assistant_messages"] += 1
            if "thinking" in message:
                summary["assistant_messages_with_thinking_field"] += 1
                if str(message.get("thinking") or "").strip():
                    summary["assistant_messages_with_nonempty_thinking"] += 1
                else:
                    summary["assistant_messages_with_empty_thinking"] += 1
            resolved_mode = _resolve_training_assistant_thinking_mode(thinking_mode, message)
            if resolved_mode == THINKING_MODE_ON:
                summary["assistant_messages_thinking_on"] += 1
            else:
                summary["assistant_messages_thinking_off"] += 1
    return summary


def _merge_thinking_training_summary(summary, delta):
    """把 worker 返回的 thinking 计数增量合并到主进程累计值。"""
    for key in THINKING_SUMMARY_COUNTER_KEYS:
        summary[key] += int(delta.get(key, 0))
    return summary


def _build_trainer_state(
    config,
    *,
    global_step,
    optimizer_step,
    samples_seen,
    tokens_seen,
    last_loss,
    current_learning_rate=None,
    last_eval_metric=None,
    optimizer_group_summary=None,
    data_progress_summary=None,
    thinking_summary=None,
    data_pipeline_runtime=None,
):
    """生成 trainer_state，覆盖续训、报告和 checkpoint metadata 需要的字段。"""
    latest_eval_loss = None
    latest_eval_ppl = None
    if last_eval_metric is not None:
        latest_eval_loss = last_eval_metric.get("eval_loss")
        latest_eval_ppl = last_eval_metric.get("eval_ppl")
    return {
        "training_stage": config.training_stage,
        "run_id": config.run_id,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "samples_seen": samples_seen,
        "tokens_seen": tokens_seen,
        "last_loss": last_loss,
        "latest_eval_loss": latest_eval_loss,
        "latest_eval_ppl": latest_eval_ppl,
        "current_learning_rate": current_learning_rate,
        "source_manifest": None if config.source_manifest is None else str(config.source_manifest),
        "eval_manifest": None if config.eval_manifest is None else str(config.eval_manifest),
        "initial_checkpoint": None if config.initial_checkpoint is None else str(config.initial_checkpoint),
        "lora_mode": bool(config.lora_mode),
        "warmup_ratio": float(config.warmup_ratio),
        "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
        "sequence_packing_enabled": bool(config.sequence_packing),
        "deterministic_algorithms": bool(config.deterministic_algorithms),
        "save_optimizer": bool(config.save_optimizer),
        "save_scheduler": bool(config.save_scheduler),
        "rng_state_saved": True,
        "save_inference_weights": bool(config.save_inference_weights),
        "eval_batch_size": config.eval_batch_size,
        "eval_max_batches": None if config.eval_max_batches is None else int(config.eval_max_batches),
        "key_checkpoints": list(config.key_checkpoints),
        "longrope2_training_strategy": _build_longrope2_training_strategy(config),
        "optimizer_group_summary": optimizer_group_summary,
        "data_progress": data_progress_summary,
        "data_pipeline_runtime": data_pipeline_runtime,
        "thinking": thinking_summary,
        "tokenizer_metadata": dict(config.tokenizer_metadata or {}),
        "training_config": _serialize_training_config(config),
    }


def _load_optimizer_scheduler_state(checkpoint_root, optimizer, scheduler):
    """从 checkpoint 目录恢复 optimizer/scheduler 状态。"""
    root = Path(checkpoint_root)
    optimizer_path = root / OPTIMIZER_CHECKPOINT_NAME
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=False))
    scheduler_path = root / SCHEDULER_CHECKPOINT_NAME
    if scheduler is not None and scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=False))


def train(
    model,
    tokenizer,
    dataset,
    *,
    config,
    eval_dataset=None,
):
    """执行 LPT v2 训练，并保存 latest checkpoint 与推理权重。"""
    if not isinstance(config, TrainingRunConfig):
        raise TypeError("config 必须是 TrainingRunConfig。")
    configure_training_runtime(
        config.seed,
        deterministic=bool(config.deterministic_algorithms),
    )
    device = next(model.parameters()).device
    GlobalConfig.device = device
    model.train()

    # 根据配置启用梯度检查点
    if config.gradient_checkpointing:
        model.gradient_checkpointing = True
        print(f"已启用梯度检查点（Gradient Checkpointing），将节省激活值显存但训练速度降低 10-20%。")
    else:
        model.gradient_checkpointing = False

    optimizer, optimizer_group_summary = _build_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optimizer_type=config.optimizer_type,
    )
    # 训练期间参数冻结状态不会变化，提前物化一次可训练参数列表，
    # 避免每个 optimizer step 重新遍历 named_parameters（大模型下这部分
    # Python 调度是训练主进程 CPU 单核的固定开销）。
    trainable_parameters = tuple(
        parameter for _, parameter in _iter_trainable_parameters(model)
    )
    global_step = 0
    optimizer_step = 0
    samples_seen = 0
    tokens_seen = 0
    best_checkpoint_value = None
    best_checkpoint_global_step = None
    resume_checkpoint = None
    resume_state = {}
    if config.resume_checkpoint and has_complete_training_state(
        config.resume_checkpoint,
        lora_mode=config.lora_mode,
    ):
        # resume 只在完整训练状态存在时启用；缺 optimizer 的目录不会被当作可续训来源。
        resume_checkpoint = config.resume_checkpoint
        resume_state = load_trainer_state(resume_checkpoint)
        global_step = int(resume_state.get("global_step", 0))
        optimizer_step = int(resume_state.get("optimizer_step", 0))
        samples_seen = int(resume_state.get("samples_seen", 0))
        tokens_seen = int(resume_state.get("tokens_seen", 0))
        best_checkpoint = resume_state.get("best_checkpoint")
        if isinstance(best_checkpoint, dict) and best_checkpoint.get("metric") == config.best_checkpoint_metric:
            best_checkpoint_value = best_checkpoint.get("value")
            best_checkpoint_global_step = best_checkpoint.get("global_step")

    previous_data_progress = load_data_progress(resume_checkpoint) if resume_checkpoint is not None else None
    data_progress_tracker = DataProgressTracker(
        dataset,
        config,
        previous_progress=previous_data_progress,
    )
    data_worker_count = _resolve_data_worker_count(
        dataset,
        config,
        data_progress_tracker,
    )
    data_reader_worker_count = _resolve_data_reader_worker_count(
        dataset,
        config,
        data_progress_tracker,
        data_worker_count,
    )
    _training_batch_budget, remaining_steps = _resolve_progress_training_budget(
        dataset,
        config,
        data_progress_tracker,
        global_step,
        data_worker_count=data_worker_count,
        tokenizer=tokenizer,
        device=device,
    )
    optimizer_step_budget = _resolve_optimizer_step_budget(
        optimizer_step,
        remaining_steps,
        config.gradient_accumulation_steps,
    )
    scheduler = _build_scheduler(
        optimizer,
        total_steps=optimizer_step_budget,
        warmup_ratio=config.warmup_ratio,
    )
    if resume_checkpoint is not None:
        _load_optimizer_scheduler_state(resume_checkpoint, optimizer, scheduler)
        _load_training_rng_state(resume_checkpoint)
    tensorboard_writer = _build_tensorboard_writer(config)
    thinking_summary = _new_thinking_training_summary(config)

    pin_memory = bool(config.data_pin_memory and device.type == "cuda")
    data_worker_backend = (
        "thread"
        if data_worker_count > 0 and getattr(dataset, "uses_token_cache", False)
        else ("process" if data_worker_count > 0 else None)
    )
    data_pipeline_runtime = _build_data_pipeline_runtime_metadata(
        config,
        data_worker_count,
        pin_memory,
        data_reader_worker_count,
        data_worker_backend,
    )
    print(
        "训练数据流水线: "
        f"reader_workers={data_reader_worker_count}, "
        f"batch_workers={data_worker_count}, "
        f"batch_worker_backend={data_worker_backend or 'none'}, "
        f"prefetch_factor={config.data_prefetch_factor if data_worker_count > 0 else 0}, "
        f"persistent_workers={bool(config.data_persistent_workers and data_worker_count > 0)}, "
        f"pin_memory={pin_memory}"
    )
    if getattr(dataset, "uses_token_cache", False):
        print(f"训练数据来源: token_cache（{config.token_cache_dir}）")

    start_time = time()
    run_start_tokens = tokens_seen
    accumulated_steps = 0
    optimizer.zero_grad(set_to_none=True)
    last_loss = resume_state.get("last_loss")
    last_loss_tensor = None
    last_grad_norm = resume_state.get("last_grad_norm")
    last_grad_norm_tensor = None
    last_eval_metric = None
    saved_step_checkpoints = set()
    key_checkpoint_steps = set(config.key_checkpoints)
    pending_save_latest = False
    pending_save_numbered = False
    pending_check_best = False
    training_batch_iterator = None
    device_batch_iterator = None
    progress_bar = None
    try:
        progress_total = remaining_steps
        if remaining_steps is not None and remaining_steps <= 0:
            batch_iterator = ()
        else:
            training_batch_iterator = _iter_parallel_training_batches(
                dataset,
                tokenizer,
                config,
                data_progress_tracker,
                device,
                worker_count=data_worker_count,
            )
            device_batch_iterator = _iter_device_batches(
                training_batch_iterator,
                device,
                async_h2d=pin_memory,
            )
            batch_iterator = device_batch_iterator
        progress_bar = tqdm(
            total=progress_total or None,
            desc="训练批次(dataset_batches)",
            unit="batch",
        )
        for batch in batch_iterator:
            if config.max_steps is not None and global_step >= int(config.max_steps):
                break
            progress_samples = batch.pop("_progress_samples", ())
            thinking_summary_delta = batch.pop("_thinking_summary_delta", {})
            token_count = int(batch["token_count"])
            valid_target_count = int(batch["valid_target_count"])
            packed_rows = int(batch["input_ids"].size(0))
            sequence_length = int(batch["input_ids"].size(1))
            padded_tokens = packed_rows * sequence_length
            attention_token_square = packed_rows * sequence_length * sequence_length
            _reset_cuda_peak_memory_stats(device)
            with torch.autocast(
                device_type=device.type,
                dtype=GlobalConfig.autocast_dtype,
                enabled=_autocast_enabled(device),
            ):
                logits, _states = _forward_batch(model, batch)
                loss, valid_targets = _compute_lm_loss(
                    logits,
                    batch["labels"],
                    valid_target_count=valid_target_count,
                    chunk_tokens=config.lm_loss_chunk_tokens,
                )
            if loss is None:
                # 无监督 token 的 batch 无法产生梯度，但必须标记为已检查，避免续训反复读取。
                data_progress_tracker.update_after_batch(progress_samples)
                del logits, _states, batch
                continue
            scaled_loss = loss / int(config.gradient_accumulation_steps)
            scaled_loss.backward()
            accumulated_steps += 1
            global_step += 1
            progress_bar.update(1)
            samples_seen += int(batch["sample_count"])
            tokens_seen += token_count
            last_loss_tensor = loss.detach()
            _merge_thinking_training_summary(
                thinking_summary,
                thinking_summary_delta,
            )
            data_progress_tracker.update_after_batch(progress_samples)
            # backward 后计算图已释放；这些大张量不再参与后续日志/checkpoint。
            # 主动删除可避免上一轮长序列 logits 在下一轮短序列 forward 期间仍占用显存。
            del logits, _states, loss, scaled_loss, batch

            if accumulated_steps >= int(config.gradient_accumulation_steps):
                if config.max_grad_norm and float(config.max_grad_norm) > 0:
                    last_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        float(config.max_grad_norm),
                    )
                else:
                    last_grad_norm_tensor = _compute_grad_norm(trainable_parameters)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accumulated_steps = 0

            should_log = global_step == 1 or global_step % max(1, int(config.log_interval)) == 0
            if should_log:
                if last_loss_tensor is not None:
                    last_loss = _scalar_to_float(last_loss_tensor)
                    last_loss_tensor = None
                if last_grad_norm_tensor is not None:
                    last_grad_norm = _scalar_to_float(last_grad_norm_tensor)
                    last_grad_norm_tensor = None
                cuda_memory_metrics = _collect_cuda_memory_metrics(device)
                progress_postfix = {
                    "损失(loss)": f"{last_loss:.4g}",
                    "全局步(global_step)": global_step,
                    "优化器步(optimizer_step)": optimizer_step,
                    "序列长度(sequence_length)": sequence_length,
                    "填充行数(packed_rows)": packed_rows,
                    "填充Token(padded_tokens)": padded_tokens,
                    "累积(accum)": accumulated_steps,
                }
                if last_grad_norm is not None:
                    progress_postfix["梯度范数(grad_norm)"] = f"{last_grad_norm:.4g}"
                if cuda_memory_metrics:
                    progress_postfix["CUDA分配峰值MiB(cuda_peak_alloc)"] = (
                        f"{cuda_memory_metrics['cuda_peak_memory_allocated_mib']:.1f}"
                    )
                    progress_postfix["CUDA保留MiB(cuda_reserved)"] = (
                        f"{cuda_memory_metrics['cuda_memory_reserved_mib']:.1f}"
                    )
                progress_bar.set_postfix(progress_postfix)
                elapsed_seconds = max(time() - start_time, 1e-9)
                metric = {
                    "stage": config.training_stage,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "loss": last_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "samples_seen": samples_seen,
                    "tokens_seen": tokens_seen,
                    "sequence_length": sequence_length,
                    "packed_rows": packed_rows,
                    "padded_tokens": padded_tokens,
                    "attention_token_square": attention_token_square,
                    "target_tokens": valid_targets,
                    "tokens_per_sec": (tokens_seen - run_start_tokens) / elapsed_seconds,
                    "grad_norm": last_grad_norm,
                    "elapsed_seconds": round(elapsed_seconds, 4),
                }
                metric.update(cuda_memory_metrics)
                _write_metric_outputs(config, "train", metric, tensorboard_writer)

            if (
                eval_dataset is not None
                and config.eval_interval
                and global_step % int(config.eval_interval) == 0
            ):
                eval_metric_this_step = False
                eval_metric = _evaluate_model(model, tokenizer, eval_dataset, config, device)
                if eval_metric is not None:
                    eval_metric.update({"stage": config.training_stage, "global_step": global_step})
                    last_eval_metric = dict(eval_metric)
                    eval_metric_this_step = True
                    _write_metric_outputs(config, "eval", eval_metric, tensorboard_writer)
            else:
                eval_metric_this_step = False

            save_latest_triggered = bool(
                config.latest_save_interval and global_step % int(config.latest_save_interval) == 0
            )
            save_step_triggered = bool(
                config.save_interval and global_step % int(config.save_interval) == 0
            )
            save_key_triggered = int(global_step) in key_checkpoint_steps
            check_best_triggered = (
                bool(config.save_best_checkpoint)
                and (
                    save_latest_triggered
                    or save_step_triggered
                    or save_key_triggered
                    or (config.best_checkpoint_metric == "eval_loss" and eval_metric_this_step)
                )
            )
            pending_save_latest = pending_save_latest or save_latest_triggered
            pending_save_numbered = pending_save_numbered or save_step_triggered or save_key_triggered
            pending_check_best = pending_check_best or check_best_triggered
            if accumulated_steps != 0:
                # checkpoint 不保存参数梯度；等累计梯度提交后再保存，避免续训跳过未生效样本。
                continue

            should_save_latest = pending_save_latest
            should_save_numbered = bool(
                pending_save_numbered and int(global_step) not in saved_step_checkpoints
            )
            should_check_best = pending_check_best
            pending_save_latest = False
            pending_save_numbered = False
            pending_check_best = False
            needs_checkpoint_state = bool(
                should_save_latest or should_save_numbered or should_check_best
            )
            if not needs_checkpoint_state:
                continue

            # 完整进度包含可能很大的 ordinal 集，只在实际 checkpoint 候选步构造。
            if last_loss_tensor is not None:
                last_loss = _scalar_to_float(last_loss_tensor)
                last_loss_tensor = None
            if last_grad_norm_tensor is not None:
                last_grad_norm = _scalar_to_float(last_grad_norm_tensor)
                last_grad_norm_tensor = None
            trainer_state = _build_trainer_state(
                config,
                global_step=global_step,
                optimizer_step=optimizer_step,
                samples_seen=samples_seen,
                tokens_seen=tokens_seen,
                last_loss=last_loss,
                current_learning_rate=optimizer.param_groups[0]["lr"],
                last_eval_metric=last_eval_metric,
                optimizer_group_summary=optimizer_group_summary,
                data_progress_summary=data_progress_tracker.summary(),
                thinking_summary=dict(thinking_summary),
                data_pipeline_runtime=dict(data_pipeline_runtime),
            )
            if last_grad_norm is not None:
                trainer_state["last_grad_norm"] = last_grad_norm

            should_save_best = False
            if should_check_best:
                # best checkpoint 与 latest/step 保存解耦：只有指标真的变优才额外发布 best 目录。
                best_metric_value = _extract_best_metric_value(config, trainer_state)
                if _is_better_checkpoint_metric(
                    best_metric_value,
                    best_checkpoint_value,
                    min_delta=config.best_checkpoint_min_delta,
                ):
                    best_checkpoint_value = best_metric_value
                    best_checkpoint_global_step = global_step
                    should_save_best = True
            _attach_best_checkpoint_state(
                config,
                trainer_state,
                best_value=best_checkpoint_value,
                best_global_step=best_checkpoint_global_step,
            )
            if not (should_save_best or should_save_numbered or should_save_latest):
                continue
            data_progress_payload = data_progress_tracker.to_payload(
                config,
                global_step=global_step,
                optimizer_step=optimizer_step,
                samples_seen=samples_seen,
                tokens_seen=tokens_seen,
            )
            if should_save_best:
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    trainer_state,
                    checkpoint_root=_best_checkpoint_root(config),
                    data_progress=data_progress_payload,
                )
            if should_save_numbered:
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    trainer_state,
                    is_latest=False,
                    data_progress=data_progress_payload,
                )
                saved_step_checkpoints.add(int(global_step))
            if should_save_latest:
                _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    trainer_state,
                    is_latest=True,
                    data_progress=data_progress_payload,
                )

        if accumulated_steps > 0:
            # 数据集尾部不足一个 gradient_accumulation_steps 时，也要把残余梯度提交一次。
            trainable_parameters = [
                parameter for _, parameter in _iter_trainable_parameters(model)
            ]
            _rescale_partial_accumulated_gradients(
                trainable_parameters,
                completed_steps=accumulated_steps,
                target_steps=config.gradient_accumulation_steps,
            )
            if config.max_grad_norm and float(config.max_grad_norm) > 0:
                last_grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    float(config.max_grad_norm),
                )
            else:
                last_grad_norm_tensor = _compute_grad_norm(trainable_parameters)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if device_batch_iterator is not None:
            device_batch_iterator.close()
        if training_batch_iterator is not None:
            training_batch_iterator.close()
        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()

    if last_loss_tensor is not None:
        last_loss = _scalar_to_float(last_loss_tensor)
    if last_grad_norm_tensor is not None:
        last_grad_norm = _scalar_to_float(last_grad_norm_tensor)
    trainer_state = _build_trainer_state(
        config,
        global_step=global_step,
        optimizer_step=optimizer_step,
        samples_seen=samples_seen,
        tokens_seen=tokens_seen,
        last_loss=last_loss,
        current_learning_rate=optimizer.param_groups[0]["lr"],
        last_eval_metric=last_eval_metric,
        optimizer_group_summary=optimizer_group_summary,
        data_progress_summary=data_progress_tracker.summary(),
        thinking_summary=dict(thinking_summary),
        data_pipeline_runtime=dict(data_pipeline_runtime),
    )
    data_progress_payload = data_progress_tracker.to_payload(
        config,
        global_step=global_step,
        optimizer_step=optimizer_step,
        samples_seen=samples_seen,
        tokens_seen=tokens_seen,
    )
    if last_grad_norm is not None:
        trainer_state["last_grad_norm"] = last_grad_norm
    if config.save_best_checkpoint:
        best_metric_value = _extract_best_metric_value(config, trainer_state)
        if _is_better_checkpoint_metric(
            best_metric_value,
            best_checkpoint_value,
            min_delta=config.best_checkpoint_min_delta,
        ):
            best_checkpoint_value = best_metric_value
            best_checkpoint_global_step = global_step
            _attach_best_checkpoint_state(
                config,
                trainer_state,
                best_value=best_checkpoint_value,
                best_global_step=best_checkpoint_global_step,
            )
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                config,
                trainer_state,
                checkpoint_root=_best_checkpoint_root(config),
                data_progress=data_progress_payload,
            )
    _attach_best_checkpoint_state(
        config,
        trainer_state,
        best_value=best_checkpoint_value,
        best_global_step=best_checkpoint_global_step,
    )
    if pending_save_numbered and int(global_step) not in saved_step_checkpoints:
        # 训练尾部不足一个累积窗口时，残余梯度已在上方提交；此时补发到期的 step/key checkpoint。
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            trainer_state,
            is_latest=False,
            data_progress=data_progress_payload,
        )
        saved_step_checkpoints.add(int(global_step))
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        config,
        trainer_state,
        is_latest=True,
        data_progress=data_progress_payload,
    )
    if config.save_inference_weights:
        _save_inference_weight(model, config, trainer_state)
    return trainer_state
