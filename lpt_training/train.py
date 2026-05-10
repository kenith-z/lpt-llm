"""LPT v2 正式训练循环。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
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
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LATEST_SAVE_INTERVAL_STEPS,
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
    DEFAULT_TRAINING_BATCH_SIZE,
    DEFAULT_TRAINING_EPOCHS,
    DEFAULT_TRAINING_LEARNING_RATE,
    DEFAULT_TRAINING_MAX_STEPS,
    DEFAULT_TRAINING_SEED,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    GlobalConfig,
)
from lpt_data import build_packed_training_batch, build_training_batch
from lpt_lora import save_lora_adapter_state
from lpt_model import save_lpt_v2_checkpoint
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
TRAINER_STATE_NAME = "trainer_state.json"
METRICS_JSONL_NAME = "metrics.jsonl"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
TRAINING_CHECKPOINT_FORMAT = "lpt_v2_training_checkpoint"
TRAINING_CHECKPOINT_SCHEMA_VERSION = 1
LM_LOSS_CHUNK_TOKENS = 256

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
    "target_tokens": "目标Token(target_tokens)",
    "tokens_per_sec": "Token吞吐(tokens_per_sec)",
    "samples_per_sec": "样本吞吐(samples_per_sec)",
    "grad_norm": "梯度范数(grad_norm)",
    "cuda_memory_allocated_mib": "CUDA已分配MiB(cuda_memory_allocated_mib)",
    "cuda_memory_reserved_mib": "CUDA已保留MiB(cuda_memory_reserved_mib)",
    "cuda_peak_memory_allocated_mib": "CUDA峰值MiB(cuda_peak_memory_allocated_mib)",
    "elapsed_seconds": "耗时秒(elapsed_seconds)",
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
    batch_size: int = DEFAULT_TRAINING_BATCH_SIZE
    epochs: int = DEFAULT_TRAINING_EPOCHS
    max_steps: int | None = DEFAULT_TRAINING_MAX_STEPS
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
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

    def __post_init__(self):
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
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数。")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError("gradient_accumulation_steps 必须为正整数。")
        if not 0.0 <= float(self.warmup_ratio) <= 1.0:
            raise ValueError("warmup_ratio 必须在 [0, 1] 范围内。")
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
        if self.best_checkpoint_min_delta < 0:
            raise ValueError("best_checkpoint_min_delta 必须为非负数。")
        self.key_checkpoints = tuple(sorted({int(value) for value in self.key_checkpoints if int(value) > 0}))


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
    root = Path(checkpoint_root)
    if root.suffix:
        return root
    return root / name


def _trainer_state_file(checkpoint_root):
    return Path(checkpoint_root) / TRAINER_STATE_NAME


def _checkpoint_manifest_file(checkpoint_root):
    return Path(checkpoint_root) / CHECKPOINT_MANIFEST_NAME


def _checkpoint_payload_name(lora_mode):
    return LORA_ADAPTER_CHECKPOINT_NAME if lora_mode else MODEL_CHECKPOINT_NAME


def _load_json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _training_state_flag(state, key, default):
    if key in state:
        return bool(state[key])
    training_config = state.get("training_config")
    if isinstance(training_config, dict) and key in training_config:
        return bool(training_config[key])
    return bool(default)


def _has_readable_torch_file(path):
    return is_torch_save_file_readable(path)


def _checkpoint_file_entries_are_valid(root, file_entries, required_names):
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
    required_names = {
        _checkpoint_payload_name(lora_mode),
        TRAINER_STATE_NAME,
    }
    if require_optimizer:
        required_names.add(OPTIMIZER_CHECKPOINT_NAME)
    if _training_state_flag(state, "save_scheduler", False):
        required_names.add(SCHEDULER_CHECKPOINT_NAME)
    return required_names


def _checkpoint_manifest_is_valid(root, *, lora_mode, require_optimizer):
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
    torch_files = [root / _checkpoint_payload_name(lora_mode)]
    if require_optimizer:
        torch_files.append(root / OPTIMIZER_CHECKPOINT_NAME)
    if SCHEDULER_CHECKPOINT_NAME in required_names:
        torch_files.append(root / SCHEDULER_CHECKPOINT_NAME)
    return all(_has_readable_torch_file(path) for path in torch_files)


def _legacy_checkpoint_is_valid(root, *, lora_mode, require_optimizer):
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
    name = Path(path).name
    if not name.startswith("step_"):
        return -1
    try:
        return int(name.removeprefix("step_"))
    except ValueError:
        return -1


def _iter_resume_checkpoint_candidates(checkpoint_root):
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


def _iter_trainable_parameters(model):
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield parameter_name, parameter


def _parameter_group_summary(parameter_group):
    return {
        "parameter_count": len(parameter_group),
        "element_count": int(sum(parameter.numel() for parameter in parameter_group)),
    }


def _build_optimizer(model, *, learning_rate, weight_decay):
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
    summary = {
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "decay": _parameter_group_summary(decay_parameters),
        "no_decay": _parameter_group_summary(no_decay_parameters),
    }
    return torch.optim.AdamW(groups, lr=float(learning_rate)), summary


def _linear_warmup_cosine_lambda(current_step, *, warmup_steps, total_steps):
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def _build_scheduler(optimizer, *, total_steps, warmup_ratio):
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
    if not config.longrope2_window_lengths:
        return config.max_sequence_length or GlobalConfig.train_max_sequence_length

    lengths = tuple(int(value) for value in config.longrope2_window_lengths)
    if rng is None:
        selected_length = max(lengths)
    else:
        weights = config.longrope2_window_weights
        selected_length = rng.choices(lengths, weights=weights, k=1)[0]

    if config.max_sequence_length is not None:
        selected_length = min(int(config.max_sequence_length), int(selected_length))
    return selected_length


def _batch_iterator(dataset, *, batch_size, epochs, max_steps=None):
    max_steps = None if max_steps is None else int(max_steps)
    if max_steps is not None and max_steps <= 0:
        return
    produced_batches = 0
    for _epoch in range(max(1, int(epochs))):
        bucket = []
        for sample in dataset:
            bucket.append(sample)
            if len(bucket) < batch_size:
                continue
            yield bucket
            produced_batches += 1
            bucket = []
            if max_steps is not None and produced_batches >= max_steps:
                return
        if bucket:
            yield bucket
            produced_batches += 1
            if max_steps is not None and produced_batches >= max_steps:
                return


def _estimate_dataset_batches(dataset, config):
    try:
        dataset_size = len(dataset)
    except TypeError:
        return None
    except NotImplementedError:
        return None
    if dataset_size <= 0:
        return None
    batches_per_epoch = math.ceil(int(dataset_size) / int(config.batch_size))
    return batches_per_epoch * max(1, int(config.epochs))


def _resolve_training_batch_budget(dataset, config):
    dataset_batches = _estimate_dataset_batches(dataset, config)
    if config.max_steps is None:
        return dataset_batches
    if dataset_batches is None:
        return int(config.max_steps)
    return min(int(config.max_steps), int(dataset_batches))


def _resolve_remaining_batch_budget(total_batch_budget, global_step):
    if total_batch_budget is None:
        return None
    return max(0, int(total_batch_budget) - int(global_step))


def _build_batch_tensors(samples, tokenizer, config, rng=None):
    max_length = _resolve_batch_max_sequence_length(config, rng=rng)
    if config.sequence_packing:
        input_ids, labels, attention_mask, position_ids, segment_ids, sample_count = build_packed_training_batch(
            samples,
            tokenizer,
            max_length=max_length,
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "sample_count": sample_count,
        }
    input_ids, labels, attention_mask = build_training_batch(
        samples,
        tokenizer,
        max_length=max_length,
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": None,
        "segment_ids": None,
        "sample_count": len(samples),
    }


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _compute_lm_loss(logits, labels, *, chunk_tokens=LM_LOSS_CHUNK_TOKENS):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_target_count = int((shift_labels != -100).sum().item())
    if valid_target_count == 0:
        return None, valid_target_count

    chunk_tokens = max(1, int(chunk_tokens))
    loss_sum = logits.new_zeros((), dtype=torch.float32)
    for batch_index in range(shift_logits.size(0)):
        batch_logits = shift_logits[batch_index]
        batch_labels = shift_labels[batch_index]
        for start in range(0, batch_logits.size(0), chunk_tokens):
            end = min(start + chunk_tokens, batch_logits.size(0))
            chunk_labels = batch_labels[start:end]
            chunk_valid_count = int((chunk_labels != -100).sum().item())
            if chunk_valid_count == 0:
                continue
            chunk_logits = batch_logits[start:end]
            chunk_loss = F.cross_entropy(
                chunk_logits.float(),
                chunk_labels,
                ignore_index=-100,
                reduction="sum",
            )
            loss_sum = loss_sum + chunk_loss
    return loss_sum / valid_target_count, valid_target_count


def _forward_batch(model, batch):
    return model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        segment_ids=batch["segment_ids"],
        rope_cache_scope="train",
        request_id="train",
        use_kv_cache=False,
    )


def _autocast_enabled(device):
    return device.type == "cuda" and GlobalConfig.autocast_dtype in {
        torch.float16,
        torch.bfloat16,
    }


def _compute_grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().norm(2).item()
        total += grad_norm * grad_norm
    return math.sqrt(total)


def _reset_cuda_peak_memory_stats(device):
    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)


def _collect_cuda_memory_metrics(device):
    if device.type != "cuda":
        return {}
    to_mib = 1024 * 1024
    return {
        "cuda_memory_allocated_mib": torch.cuda.memory_allocated(device) / to_mib,
        "cuda_memory_reserved_mib": torch.cuda.memory_reserved(device) / to_mib,
        "cuda_peak_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / to_mib,
    }


@torch.no_grad()
def _evaluate_model(model, tokenizer, eval_dataset, config, device):
    if eval_dataset is None:
        return None
    was_training = model.training
    model.eval()
    losses = []
    token_count = 0
    max_eval_batches = None if config.eval_max_batches is None else int(config.eval_max_batches)
    eval_batch_size = int(config.eval_batch_size or config.batch_size)
    for batch_index, samples in enumerate(
        _batch_iterator(
            eval_dataset,
            batch_size=eval_batch_size,
            epochs=1,
            max_steps=max_eval_batches,
        ),
        start=1,
    ):
        batch = _move_batch(_build_batch_tensors(samples, tokenizer, config), device)
        with torch.autocast(
            device_type=device.type,
            dtype=GlobalConfig.autocast_dtype,
            enabled=_autocast_enabled(device),
        ):
            logits, _states = _forward_batch(model, batch)
            loss, valid_targets = _compute_lm_loss(logits, batch["labels"])
        if loss is not None:
            losses.append(float(loss.detach().cpu()))
            token_count += valid_targets
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
    if was_training:
        model.train()
    if not losses:
        return None
    eval_loss = sum(losses) / len(losses)
    return {
        "eval_loss": eval_loss,
        "eval_ppl": math.exp(eval_loss) if eval_loss < 80 else float("inf"),
        "eval_target_tokens": token_count,
        "eval_batches": len(losses),
    }


def _write_metrics(artifact_dir, metric):
    metric_path = Path(artifact_dir) / METRICS_JSONL_NAME
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def _display_metric_name(key):
    return METRIC_DISPLAY_NAMES.get(str(key), f"{key}({key})")


def _display_phase_name(namespace):
    return PHASE_DISPLAY_NAMES.get(str(namespace), f"{namespace}({namespace})")


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_metric_record(namespace, metric):
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
        self.log_dir = Path(log_dir)
        self.disabled = False
        self._warning_emitted = False
        self._writer = None
        self._open_writer()

    def _open_writer(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.log_dir), max_queue=1)

    def _warn_once(self, exc):
        if self._warning_emitted:
            return
        tqdm.write(f"警告: TensorBoard 写入失败，已关闭 TensorBoard 输出: {exc}")
        self._warning_emitted = True

    def _close_writer(self):
        if self._writer is None:
            return
        try:
            self._writer.close()
        except Exception:
            pass
        self._writer = None

    def _disable(self, exc):
        self.disabled = True
        self._warn_once(exc)
        self._close_writer()

    def add_scalar(self, tag, value, step):
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
        if self.disabled or self._writer is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer.flush()
        except Exception as exc:
            self._disable(exc)

    def close(self):
        self._close_writer()


def _disable_tensorboard_writer(writer, exc):
    if hasattr(writer, "_disable"):
        writer._disable(exc)
        return
    tqdm.write(f"警告: TensorBoard 写入失败，已跳过本次 TensorBoard 输出: {exc}")


def _write_tensorboard_metrics(writer, namespace, metric):
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
    _write_metrics(config.artifact_dir, metric)
    _print_metric_record(namespace, metric)
    _write_tensorboard_metrics(writer, namespace, metric)


def _save_trainer_state(checkpoint_root, state):
    state_path = _trainer_state_file(checkpoint_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_checkpoint_manifest(checkpoint_root, config, trainer_state, file_names):
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
    manifest_path = _checkpoint_manifest_file(checkpoint_root)
    manifest = _build_checkpoint_manifest(checkpoint_root, config, trainer_state, file_names)
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _staging_checkpoint_root(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    return checkpoint_root.with_name(f"{checkpoint_root.name}.staging.{uuid4().hex}")


def _checkpoint_previous_root(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    return checkpoint_root.with_name(f"{checkpoint_root.name}_previous")


def _remove_path(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_checkpoint_root(staging_root, target_root, *, rotate_existing):
    staging_root = Path(staging_root)
    target_root = Path(target_root)
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if rotate_existing and target_root.exists():
        previous_root = _checkpoint_previous_root(target_root)
        _remove_path(previous_root)
        target_root.rename(previous_root)
    elif target_root.exists():
        raise FileExistsError(f"目标 checkpoint 已存在: {target_root}")
    staging_root.rename(target_root)


def _cleanup_checkpoint_root(path):
    try:
        _remove_path(path)
    except OSError:
        pass


def _save_checkpoint(model, optimizer, scheduler, config, trainer_state, *, is_latest=True, checkpoint_root=None):
    checkpoint_root = Path(config.checkpoint_dir) if checkpoint_root is None else Path(checkpoint_root)
    if not is_latest:
        checkpoint_root = checkpoint_root.parent / f"step_{trainer_state['global_step']:06d}"
    staging_root = _staging_checkpoint_root(checkpoint_root)
    staging_root.mkdir(parents=True, exist_ok=False)
    try:
        file_names = []
        if config.lora_mode:
            save_lora_adapter_state(
                model,
                staging_root / LORA_ADAPTER_CHECKPOINT_NAME,
                extra_metadata=trainer_state,
            )
            file_names.append(LORA_ADAPTER_CHECKPOINT_NAME)
        else:
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
        _save_trainer_state(staging_root, trainer_state)
        file_names.append(TRAINER_STATE_NAME)
        _save_checkpoint_manifest(staging_root, config, trainer_state, file_names)
        if not _is_valid_training_checkpoint_root(
            staging_root,
            lora_mode=bool(config.lora_mode),
            require_optimizer=bool(config.save_optimizer),
        ):
            raise RuntimeError(f"暂存 checkpoint 校验失败: {staging_root}")
        _publish_checkpoint_root(staging_root, checkpoint_root, rotate_existing=bool(is_latest))
        model.config.save_json(Path(config.artifact_dir) / "config" / "model_config.json")
        return checkpoint_root
    except Exception:
        _cleanup_checkpoint_root(staging_root)
        raise


def _best_checkpoint_root(config):
    if config.best_checkpoint_dir is not None:
        return Path(config.best_checkpoint_dir)
    return Path(config.checkpoint_dir).parent / "best_loss"


def _extract_best_metric_value(config, trainer_state):
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
    if current_value is None:
        return False
    if best_value is None:
        return True
    return float(current_value) < float(best_value) - float(min_delta)


def _attach_best_checkpoint_state(config, trainer_state, *, best_value, best_global_step):
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
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in asdict(config).items()
    }


def _build_longrope2_training_strategy(config):
    return {
        "max_sequence_length": config.max_sequence_length,
        "window_lengths": None if config.longrope2_window_lengths is None else list(config.longrope2_window_lengths),
        "window_weights": None if config.longrope2_window_weights is None else list(config.longrope2_window_weights),
        "sequence_packing": bool(config.sequence_packing),
    }


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
):
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
        "save_inference_weights": bool(config.save_inference_weights),
        "eval_batch_size": config.eval_batch_size,
        "eval_max_batches": None if config.eval_max_batches is None else int(config.eval_max_batches),
        "key_checkpoints": list(config.key_checkpoints),
        "longrope2_training_strategy": _build_longrope2_training_strategy(config),
        "optimizer_group_summary": optimizer_group_summary,
        "tokenizer_metadata": dict(config.tokenizer_metadata or {}),
        "training_config": _serialize_training_config(config),
    }


def _load_optimizer_scheduler_state(checkpoint_root, optimizer, scheduler):
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
    optimizer, optimizer_group_summary = _build_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_batch_budget = _resolve_training_batch_budget(dataset, config)
    scheduler = _build_scheduler(
        optimizer,
        total_steps=training_batch_budget,
        warmup_ratio=config.warmup_ratio,
    )
    tensorboard_writer = _build_tensorboard_writer(config)
    length_rng = random.Random(int(config.seed) + 7919)

    global_step = 0
    optimizer_step = 0
    samples_seen = 0
    tokens_seen = 0
    best_checkpoint_value = None
    best_checkpoint_global_step = None
    if config.resume_checkpoint and has_complete_training_state(
        config.resume_checkpoint,
        lora_mode=config.lora_mode,
    ):
        state = load_trainer_state(config.resume_checkpoint)
        global_step = int(state.get("global_step", 0))
        optimizer_step = int(state.get("optimizer_step", 0))
        samples_seen = int(state.get("samples_seen", 0))
        tokens_seen = int(state.get("tokens_seen", 0))
        best_checkpoint = state.get("best_checkpoint")
        if isinstance(best_checkpoint, dict) and best_checkpoint.get("metric") == config.best_checkpoint_metric:
            best_checkpoint_value = best_checkpoint.get("value")
            best_checkpoint_global_step = best_checkpoint.get("global_step")
        _load_optimizer_scheduler_state(config.resume_checkpoint, optimizer, scheduler)

    start_time = time()
    accumulated_steps = 0
    optimizer.zero_grad(set_to_none=True)
    last_loss = None
    last_grad_norm = None
    last_eval_metric = None
    saved_step_checkpoints = set()
    try:
        remaining_steps = _resolve_remaining_batch_budget(training_batch_budget, global_step)
        progress_total = remaining_steps
        batch_iterator = (
            _batch_iterator(
                dataset,
                batch_size=config.batch_size,
                epochs=config.epochs,
                max_steps=remaining_steps,
            )
            if remaining_steps is None or remaining_steps > 0
            else ()
        )
        progress_bar = tqdm(
            batch_iterator,
            total=progress_total or None,
            desc="训练批次(dataset_batches)",
            unit="batch",
        )
        for samples in progress_bar:
            batch = _move_batch(_build_batch_tensors(samples, tokenizer, config, rng=length_rng), device)
            sequence_length = int(batch["input_ids"].size(1))
            _reset_cuda_peak_memory_stats(device)
            with torch.autocast(
                device_type=device.type,
                dtype=GlobalConfig.autocast_dtype,
                enabled=_autocast_enabled(device),
            ):
                logits, _states = _forward_batch(model, batch)
                loss, valid_targets = _compute_lm_loss(logits, batch["labels"])
            if loss is None:
                continue
            scaled_loss = loss / int(config.gradient_accumulation_steps)
            scaled_loss.backward()
            accumulated_steps += 1
            global_step += 1
            samples_seen += int(batch["sample_count"])
            tokens_seen += int(batch["attention_mask"].sum().item())
            last_loss = float(loss.detach().cpu())
            cuda_memory_metrics = _collect_cuda_memory_metrics(device)

            if accumulated_steps >= int(config.gradient_accumulation_steps):
                trainable_parameters = [
                    parameter for _, parameter in _iter_trainable_parameters(model)
                ]
                if config.max_grad_norm and float(config.max_grad_norm) > 0:
                    last_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            trainable_parameters,
                            float(config.max_grad_norm),
                        )
                    )
                else:
                    last_grad_norm = _compute_grad_norm(trainable_parameters)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accumulated_steps = 0

            progress_postfix = {
                "损失(loss)": f"{last_loss:.4g}",
                "全局步(global_step)": global_step,
                "优化器步(optimizer_step)": optimizer_step,
                "序列长度(sequence_length)": sequence_length,
                "累积(accum)": accumulated_steps,
            }
            if last_grad_norm is not None:
                progress_postfix["梯度范数(grad_norm)"] = f"{last_grad_norm:.4g}"
            if cuda_memory_metrics:
                progress_postfix["CUDA峰值MiB(cuda_peak)"] = (
                    f"{cuda_memory_metrics['cuda_peak_memory_allocated_mib']:.1f}"
                )
            progress_bar.set_postfix(progress_postfix)

            should_log = global_step == 1 or global_step % max(1, int(config.log_interval)) == 0
            if should_log:
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
                    "target_tokens": valid_targets,
                    "tokens_per_sec": tokens_seen / elapsed_seconds,
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
            )

            should_save_latest = bool(
                config.latest_save_interval and global_step % int(config.latest_save_interval) == 0
            )
            should_save_step = bool(config.save_interval and global_step % int(config.save_interval) == 0)
            should_save_key = int(global_step) in set(config.key_checkpoints)
            should_check_best = (
                bool(config.save_best_checkpoint)
                and (
                    should_save_latest
                    or should_save_step
                    or should_save_key
                    or (config.best_checkpoint_metric == "eval_loss" and eval_metric_this_step)
                )
            )
            if should_check_best:
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
                    )
            _attach_best_checkpoint_state(
                config,
                trainer_state,
                best_value=best_checkpoint_value,
                best_global_step=best_checkpoint_global_step,
            )
            if (should_save_step or should_save_key) and int(global_step) not in saved_step_checkpoints:
                _save_checkpoint(model, optimizer, scheduler, config, trainer_state, is_latest=False)
                saved_step_checkpoints.add(int(global_step))
            if should_save_latest:
                _save_checkpoint(model, optimizer, scheduler, config, trainer_state, is_latest=True)

        if accumulated_steps > 0:
            trainable_parameters = [
                parameter for _, parameter in _iter_trainable_parameters(model)
            ]
            if config.max_grad_norm and float(config.max_grad_norm) > 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        float(config.max_grad_norm),
                    )
                )
            else:
                last_grad_norm = _compute_grad_norm(trainable_parameters)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()

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
            )
    _attach_best_checkpoint_state(
        config,
        trainer_state,
        best_value=best_checkpoint_value,
        best_global_step=best_checkpoint_global_step,
    )
    _save_checkpoint(model, optimizer, scheduler, config, trainer_state, is_latest=True)
    if config.save_inference_weights:
        _save_inference_weight(model, config, trainer_state)
    return trainer_state
