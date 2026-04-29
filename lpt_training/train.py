"""训练流程模块

改动说明:
1. 剥离了模型内部的 Loss 计算，改为在训练循环中通过 F.cross_entropy 显式计算。
2. 根据 attention_mask 动态构建 position_ids，支持 Padding 批次。
3. 对齐 bfloat16 的现代混合精度训练规范，移除了不再需要的 GradScaler。
4. 适配了新模型中 self.config.xxx 的属性访问方式。
"""

import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

tqdm.monitor_interval = 0

from lpt_config import (
    MODEL_CONFIG_SCHEMA_VERSION,
    ChatSFTTrainingConfig,
    GlobalConfig,
    build_model_config_from_checkpoint,
    build_longrope2_uniform_factors,
    model_config_snapshot_path,
)
from lpt_inference.visualization import display_checkpoint_summary
from lpt_model import (
    extract_checkpoint_architecture_metadata,
    get_model_architecture_metadata,
    list_architecture_mismatches,
)

from .data_processing import build_packed_training_batch, build_training_batch, encode_training_sample


CURRENT_CHECKPOINT_SCHEMA_VERSION = 1


def configure_training_runtime(seed=None, deterministic_algorithms=False):
    """配置训练阶段的随机种子与基础确定性选项。"""
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False


def _log_root_from_checkpoint_root(checkpoint_root):
    return _artifact_root_from_checkpoint_root(checkpoint_root) / "logs"


def _format_run_id():
    return time.strftime("%Y%m%d_%H%M%S")


def _split_parameter_name(full_name):
    module_name, _, parameter_name = full_name.rpartition(".")
    return module_name, parameter_name or full_name


def _should_apply_weight_decay(module, parameter_name, parameter):
    if parameter_name.endswith("bias"):
        return False
    if isinstance(module, (nn.Embedding, nn.GroupNorm)):
        return False

    module_class_name = module.__class__.__name__.lower()
    if "norm" in module_class_name:
        return False
    if isinstance(module, nn.Linear):
        return True
    return parameter.ndim >= 2


def _build_optimizer_parameter_groups(model, weight_decay):
    decay_parameters = []
    no_decay_parameters = []
    decay_parameter_names = []
    no_decay_parameter_names = []

    for full_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        module_name, parameter_name = _split_parameter_name(full_name)
        module = model.get_submodule(module_name) if module_name else model
        if weight_decay > 0 and _should_apply_weight_decay(module, parameter_name, parameter):
            decay_parameters.append(parameter)
            decay_parameter_names.append(full_name)
        else:
            no_decay_parameters.append(parameter)
            no_decay_parameter_names.append(full_name)

    optimizer_groups = []
    if decay_parameters:
        optimizer_groups.append({"params": decay_parameters, "weight_decay": float(weight_decay)})
    if no_decay_parameters:
        optimizer_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    group_summary = {
        "weight_decay": float(weight_decay),
        "decay_parameter_count": len(decay_parameter_names),
        "no_decay_parameter_count": len(no_decay_parameter_names),
        "decay_parameter_names": tuple(decay_parameter_names),
        "no_decay_parameter_names": tuple(no_decay_parameter_names),
    }
    return optimizer_groups, group_summary


def _batch_token_count(attention_mask):
    return int(attention_mask.sum().item())


def _batch_supervised_token_count(labels):
    return int(labels[:, 1:].ne(-100).sum().item())


def _compute_language_model_loss(logits, labels, reduction="mean"):
    return F.cross_entropy(
        logits[:, :-1, :].transpose(1, 2),
        labels[:, 1:],
        ignore_index=-100,
        reduction=reduction,
    )


def _forward_batch(
    model,
    input_ids,
    labels,
    attention_mask,
    *,
    position_ids=None,
    segment_ids=None,
):
    if position_ids is None:
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
    with torch.autocast(
        device_type=GlobalConfig.device.type,
        dtype=GlobalConfig.autocast_dtype,
    ):
        logits, _ = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            rope_cache_scope="train",
        )
        loss = _compute_language_model_loss(logits, labels, reduction="mean")
        summed_loss = _compute_language_model_loss(logits, labels, reduction="sum")
    return loss, summed_loss


def _unpack_training_batch(batch):
    if len(batch) == 3:
        input_ids, labels, attention_mask = batch
        return input_ids, labels, attention_mask, None, None, int(input_ids.size(0))
    if len(batch) == 6:
        input_ids, labels, attention_mask, position_ids, segment_ids, sample_count = batch
        return (
            input_ids,
            labels,
            attention_mask,
            position_ids,
            segment_ids,
            int(sample_count),
        )
    raise ValueError(f"未支持的 batch 结构，字段数为 {len(batch)}。")


def _compute_grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.detach().float().norm(2).item()
        total += grad_norm * grad_norm
    return total ** 0.5


def _evaluate_model(model, dataloader, *, max_batches=None):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_supervised_tokens = 0
    total_input_tokens = 0
    total_batches = 0

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader, start=1):
                if max_batches is not None and batch_index > max_batches:
                    break

                input_ids, labels, attention_mask, position_ids, segment_ids, _ = _unpack_training_batch(
                    batch
                )
                input_ids = input_ids.to(GlobalConfig.device, non_blocking=True)
                labels = labels.to(GlobalConfig.device, non_blocking=True)
                attention_mask = attention_mask.to(GlobalConfig.device, non_blocking=True)
                if position_ids is not None:
                    position_ids = position_ids.to(GlobalConfig.device, non_blocking=True)
                if segment_ids is not None:
                    segment_ids = segment_ids.to(GlobalConfig.device, non_blocking=True)

                _, summed_loss = _forward_batch(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                )
                supervised_tokens = _batch_supervised_token_count(labels)
                total_loss += float(summed_loss.item())
                total_supervised_tokens += supervised_tokens
                total_input_tokens += _batch_token_count(attention_mask)
                total_batches += 1
    finally:
        if was_training:
            model.train()

    if total_supervised_tokens == 0:
        raise ValueError("验证集没有可用于监督损失计算的 token。")

    eval_loss = total_loss / total_supervised_tokens
    eval_ppl = math.exp(eval_loss) if eval_loss < 80 else float("inf")
    return {
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "eval_batches": total_batches,
        "eval_supervised_tokens": total_supervised_tokens,
        "eval_input_tokens": total_input_tokens,
    }


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_metric_record(prefix, metrics):
    ordered_items = [
        (key, metrics[key])
        for key in metrics
        if key not in {"phase", "run_id", "timestamp"}
    ]
    line = " ".join(f"{key}={_format_metric_value(value)}" for key, value in ordered_items)
    print(f"[{prefix}] {line}")


class TrainingMetricLogger:
    """统一写 stdout、JSONL 和 TensorBoard。"""

    def __init__(self, checkpoint_root, run_id, *, tensorboard_enabled):
        self.run_id = run_id
        self.log_root = _log_root_from_checkpoint_root(checkpoint_root)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_root / f"{run_id}.training_metrics.jsonl"
        self.tensorboard_dir = self.log_root / "tensorboard" / run_id
        self.writer = None

        if tensorboard_enabled:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

    def log(self, phase, step, metrics):
        record = {
            "phase": phase,
            "run_id": self.run_id,
            "timestamp": time.time(),
            **metrics,
        }
        with self.metrics_path.open("a", encoding="utf-8") as metric_file:
            metric_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        _print_metric_record(phase, record)

        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, bool):
                    self.writer.add_scalar(f"{phase}/{key}", int(value), step)
                elif isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{phase}/{key}", value, step)
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


def _resolve_training_profile(training_profile):
    """返回当前训练调用实际采用的配置档。"""
    if training_profile is None:
        return ChatSFTTrainingConfig
    return training_profile


class LongRoPE2WindowSampler:
    """按训练策略为每个 batch 选择 LongRoPE2 训练窗口。"""

    def __init__(self, lengths, weights=None, *, seed=None, randomize=True):
        self.lengths = tuple(int(length) for length in lengths)
        if not self.lengths:
            raise ValueError("LongRoPE2 训练窗口列表不能为空。")
        if any(length <= 0 for length in self.lengths):
            raise ValueError("LongRoPE2 训练窗口必须全部为正整数。")

        if weights is None:
            weights = tuple(1.0 for _ in self.lengths)
        self.weights = tuple(float(weight) for weight in weights)
        if len(self.weights) != len(self.lengths):
            raise ValueError("LongRoPE2 训练窗口权重数量必须与窗口数量一致。")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("LongRoPE2 训练窗口权重不能为负数。")
        if sum(self.weights) <= 0:
            raise ValueError("LongRoPE2 训练窗口权重总和必须大于 0。")

        self.randomize = bool(randomize and len(self.lengths) > 1)
        self.rng = random.Random(seed)

    def next_length(self):
        if not self.randomize:
            return max(self.lengths)
        return self.rng.choices(self.lengths, weights=self.weights, k=1)[0]

    def to_dict(self):
        weight_total = sum(self.weights)
        return {
            "window_sampling_enabled": self.randomize,
            "window_lengths": self.lengths,
            "window_sampling_weights": tuple(weight / weight_total for weight in self.weights),
            "fixed_window_length": None if self.randomize else max(self.lengths),
        }


def _deduplicate_ints(values):
    deduplicated = []
    for value in values:
        normalized_value = int(value)
        if normalized_value not in deduplicated:
            deduplicated.append(normalized_value)
    return tuple(deduplicated)


def _default_longrope2_window_lengths(model_config):
    train_max_length = int(GlobalConfig.train_max_sequence_length)
    return _deduplicate_ints(
        length
        for length in (
            min(int(model_config.original_max_len), train_max_length),
            min(int(model_config.longrope2_target_length), train_max_length),
        )
        if length > 0
    )


def _normalize_longrope2_window_lengths(model_config, profile):
    configured_lengths = getattr(profile, "longrope2_window_lengths", None)
    if configured_lengths is None:
        lengths = _default_longrope2_window_lengths(model_config)
    else:
        lengths = _deduplicate_ints(configured_lengths)

    if not lengths:
        raise ValueError("LongRoPE2 训练窗口列表不能为空。")

    max_window_length = max(lengths)
    if max_window_length > int(GlobalConfig.train_max_sequence_length):
        raise ValueError(
            f"LongRoPE2 训练窗口上限 ({max_window_length}) 超过 "
            f"GlobalConfig.train_max_sequence_length ({GlobalConfig.train_max_sequence_length})。"
        )
    if max_window_length > int(GlobalConfig.train_rope_cache_max_sequence_length):
        raise ValueError(
            f"LongRoPE2 训练窗口上限 ({max_window_length}) 超过 "
            f"GlobalConfig.train_rope_cache_max_sequence_length "
            f"({GlobalConfig.train_rope_cache_max_sequence_length})。"
        )
    return lengths


def _normalize_longrope2_window_weights(profile, window_lengths):
    configured_weights = getattr(profile, "longrope2_window_sampling_weights", None)
    if configured_weights is None:
        return tuple(1.0 for _ in window_lengths)
    weights = tuple(float(weight) for weight in configured_weights)
    if len(weights) != len(window_lengths):
        raise ValueError("longrope2_window_sampling_weights 数量必须与窗口数量一致。")
    return weights


def _build_longrope2_window_sampler(model_config, profile, *, seed=None, randomize=True):
    if not bool(getattr(profile, "longrope2_window_sampling_enabled", False)):
        return None
    window_lengths = _normalize_longrope2_window_lengths(model_config, profile)
    window_weights = _normalize_longrope2_window_weights(profile, window_lengths)
    return LongRoPE2WindowSampler(
        window_lengths,
        window_weights,
        seed=seed,
        randomize=randomize,
    )


def _build_longrope2_training_strategy(model_config, train_window_sampler):
    """生成随 checkpoint 落盘的 LongRoPE2 训练策略快照。"""
    if train_window_sampler is None:
        window_payload = {
            "window_sampling_enabled": False,
            "window_lengths": (int(GlobalConfig.train_max_sequence_length),),
            "window_sampling_weights": (1.0,),
            "fixed_window_length": int(GlobalConfig.train_max_sequence_length),
        }
    else:
        window_payload = train_window_sampler.to_dict()

    return {
        "original_window": int(model_config.original_max_len),
        "target_window": int(model_config.longrope2_target_length),
        "training_target_window": max(window_payload["window_lengths"]),
        "train_embedding_mode": model_config.longrope2_train_embedding_mode,
        "inference_embedding_mode": model_config.longrope2_inference_embedding_mode,
        "mixed_original_window": (
            model_config.longrope2_mixed_original_window
            if model_config.longrope2_mixed_original_window is not None
            else int(model_config.original_max_len)
        ),
        "factor_max_sequence_length": model_config.longrope2_factor_max_sequence_length,
        "train_max_sequence_length": int(GlobalConfig.train_max_sequence_length),
        "train_rope_cache_max_sequence_length": int(GlobalConfig.train_rope_cache_max_sequence_length),
        "inference_rope_cache_max_sequence_length": int(GlobalConfig.inference_rope_cache_max_sequence_length),
        **window_payload,
    }


def _iter_dataset_for_length_scan(dataset):
    if hasattr(dataset, "iter_records_for_scan"):
        yield from dataset.iter_records_for_scan()
        return
    yield from dataset


def _summarize_dataset_token_lengths(dataset, tokenizer):
    sample_count = 0
    max_sequence_length = 0
    max_sample_id = None

    for sample in _iter_dataset_for_length_scan(dataset):
        encoded_sample = encode_training_sample(sample, tokenizer, max_length=None)
        sample_count += 1
        if encoded_sample.length > max_sequence_length:
            max_sequence_length = encoded_sample.length
            max_sample_id = encoded_sample.sample_id

    if sample_count == 0:
        raise ValueError("训练数据集为空，无法计算 LongRoPE2 搜索因子覆盖长度。")

    return {
        "sample_count": sample_count,
        "max_sequence_length": max_sequence_length,
        "max_sample_id": max_sample_id,
    }


def _ensure_longrope2_dataset_factors(model, dataset, tokenizer, *, enabled=True):
    """按数据集最长 token 长度刷新 LongRoPE2 factors。"""
    if not enabled:
        return None

    length_summary = _summarize_dataset_token_lengths(dataset, tokenizer)
    max_sequence_length = int(length_summary["max_sequence_length"])
    recorded_max_length = model.config.longrope2_factor_max_sequence_length
    has_factors = model.config.longrope2_long_factors is not None

    if has_factors and (recorded_max_length is None or recorded_max_length >= max_sequence_length):
        if recorded_max_length is None:
            model.refresh_longrope2_factors(
                model.config.longrope2_long_factors,
                factor_max_sequence_length=max_sequence_length,
            )
            recorded_max_length = max_sequence_length
        print(
            "LongRoPE2搜索因子复用",
            f"dataset_max={max_sequence_length}",
            f"factor_max={recorded_max_length}",
            f"max_sample={length_summary['max_sample_id']}",
        )
        return length_summary

    refreshed_factors = build_longrope2_uniform_factors(model.config, max_sequence_length)
    model.refresh_longrope2_factors(
        refreshed_factors,
        factor_max_sequence_length=max_sequence_length,
    )
    print(
        "LongRoPE2搜索因子刷新",
        f"dataset_max={max_sequence_length}",
        f"factor={refreshed_factors[0]:.6g}",
        f"max_sample={length_summary['max_sample_id']}",
    )
    return length_summary


def _normalize_checkpoint_path(path):
    if path is None:
        return None
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".pth":
        return checkpoint_path.with_suffix("")
    return checkpoint_path


def _checkpoint_file(checkpoint_root):
    return checkpoint_root.with_suffix(".pth")


def _optimizer_file(checkpoint_root):
    return checkpoint_root.parent / f"{checkpoint_root.name}_optimizer.pth"


def _scheduler_file(checkpoint_root):
    return checkpoint_root.parent / f"{checkpoint_root.name}_scheduler.pth"


def _normalize_manifest_path(manifest_path):
    if manifest_path is None:
        return None
    return str(Path(manifest_path))


def _artifact_root_from_checkpoint_root(checkpoint_root):
    return checkpoint_root.parent.parent


def _save_model_config_snapshot(model, checkpoint_root):
    snapshot_path = model_config_snapshot_path(_artifact_root_from_checkpoint_root(checkpoint_root))
    model.config.save_json(snapshot_path)
    return snapshot_path


def _normalize_loaded_checkpoint_schema(checkpoint):
    """校验 checkpoint schema 并标准化结构。"""
    normalized_checkpoint = dict(checkpoint)
    checkpoint_schema_version = normalized_checkpoint.get("checkpoint_schema_version")

    if checkpoint_schema_version is None:
        raise ValueError("checkpoint 缺少 checkpoint_schema_version。")
    if checkpoint_schema_version != CURRENT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "不支持的 checkpoint_schema_version: "
            f"{checkpoint_schema_version}，当前仅支持 {CURRENT_CHECKPOINT_SCHEMA_VERSION}。"
        )

    model_config_schema_version = normalized_checkpoint.get("model_config_schema_version")
    if model_config_schema_version is None:
        raise ValueError("checkpoint 缺少 model_config_schema_version。")
    if model_config_schema_version != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "不支持的 model_config_schema_version: "
            f"{model_config_schema_version}，当前仅支持 {MODEL_CONFIG_SCHEMA_VERSION}。"
        )

    for required_key in ("training_stage", "source_manifest"):
        if required_key not in normalized_checkpoint:
            raise ValueError(f"checkpoint 缺少 {required_key}。")

    architecture_metadata = extract_checkpoint_architecture_metadata(normalized_checkpoint)
    model_config = build_model_config_from_checkpoint(normalized_checkpoint)
    normalized_checkpoint["model_architecture_metadata"] = architecture_metadata
    normalized_checkpoint["model_config"] = model_config.to_dict()
    return normalized_checkpoint


def load_checkpoint(checkpoint_path, map_location=None):
    """加载 checkpoint，并严格校验 schema。"""
    checkpoint_root = _normalize_checkpoint_path(checkpoint_path)
    if checkpoint_root is None:
        raise ValueError("checkpoint_path 不能为空。")
    if map_location is None:
        map_location = GlobalConfig.device
    checkpoint = torch.load(_checkpoint_file(checkpoint_root), map_location=map_location)
    return _normalize_loaded_checkpoint_schema(checkpoint)


def _list_missing_training_state_files(
    checkpoint_path,
    require_optimizer=True,
    require_scheduler=True,
):
    checkpoint_root = _normalize_checkpoint_path(checkpoint_path)
    if checkpoint_root is None:
        return []

    missing_files = []
    checkpoint_file = _checkpoint_file(checkpoint_root)
    if not checkpoint_file.exists():
        missing_files.append(checkpoint_file.name)

    optimizer_file = _optimizer_file(checkpoint_root)
    if require_optimizer and not optimizer_file.exists():
        missing_files.append(optimizer_file.name)

    scheduler_file = _scheduler_file(checkpoint_root)
    if require_scheduler and not scheduler_file.exists():
        missing_files.append(scheduler_file.name)

    return missing_files


def has_complete_training_state(
    checkpoint_path,
    training_profile=None,
    require_optimizer=None,
    require_scheduler=None,
):
    """检查某个 checkpoint 根路径是否具备完整续训状态。"""
    if checkpoint_path is None:
        return False

    profile = _resolve_training_profile(training_profile)
    if require_optimizer is None:
        require_optimizer = profile.save_optimizer
    if require_scheduler is None:
        require_scheduler = profile.save_scheduler

    return not _list_missing_training_state_files(
        checkpoint_path,
        require_optimizer=require_optimizer,
        require_scheduler=require_scheduler,
    )


def _build_optimizer(model, learning_rate, weight_decay):
    parameter_groups, group_summary = _build_optimizer_parameter_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate)
    return optimizer, group_summary


def _build_scheduler(optimizer, num_batches, total_epochs, warmup_ratio, gradient_accumulation_steps):
    total_steps = max(1, math.ceil(num_batches / gradient_accumulation_steps) * total_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _format_architecture_mismatches(mismatches):
    return "; ".join(
        f"{key}: checkpoint={checkpoint_value}, current={current_value}"
        for key, checkpoint_value, current_value in mismatches
    )


def _current_learning_rate(optimizer):
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _load_resume_state(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    save_optimizer,
    save_scheduler,
    allowed_architecture_mismatch_keys=None,
):
    checkpoint_root = _normalize_checkpoint_path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_root)
    allowed_keys = set(allowed_architecture_mismatch_keys or ())
    mismatches = [
        mismatch
        for mismatch in list_architecture_mismatches(checkpoint, model)
        if mismatch[0] not in allowed_keys
    ]
    if mismatches:
        raise ValueError(_format_architecture_mismatches(mismatches))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if save_optimizer:
        optimizer.load_state_dict(
            torch.load(_optimizer_file(checkpoint_root), map_location=GlobalConfig.device)
        )
        print("已恢复优化器状态。")
    if save_scheduler:
        scheduler.load_state_dict(
            torch.load(_scheduler_file(checkpoint_root), map_location=GlobalConfig.device)
        )
        print("已恢复学习率调度器状态。")

    display_checkpoint_summary(checkpoint)
    print("检测到可恢复训练状态，继续训练。")
    return checkpoint.get("epoch", 0), checkpoint


def _load_initial_model_state(model, checkpoint_path, allowed_architecture_mismatch_keys=None):
    checkpoint_root = _normalize_checkpoint_path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_root)
    allowed_keys = set(allowed_architecture_mismatch_keys or ())
    mismatches = [
        mismatch
        for mismatch in list_architecture_mismatches(checkpoint, model)
        if mismatch[0] not in allowed_keys
    ]
    if mismatches:
        raise ValueError(_format_architecture_mismatches(mismatches))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    display_checkpoint_summary(checkpoint)
    print("未发现可恢复训练状态，已加载初始化模型参数。")
    return checkpoint


def _collect_lora_adapter_state(model):
    adapter_state = {}
    for name, parameter in model.state_dict().items():
        if name.endswith("down_projection.weight") or name.endswith("up_projection.weight"):
            adapter_state[name] = parameter.detach().cpu()
    return adapter_state


def _save_inference_weights(model, weight_path):
    if weight_path is None:
        return

    target_path = Path(weight_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if getattr(GlobalConfig, "lora_mode", False):
        torch.save(_collect_lora_adapter_state(model), target_path)
        return
    torch.save(model.state_dict(), target_path)


def _save_full_checkpoint(
    model,
    tokenizer,
    checkpoint_root,
    manifest_path,
    eval_manifest_path,
    loss_value,
    eval_loss,
    eval_ppl,
    epoch_index,
    total_epochs,
    batch_size,
    learning_rate,
    warmup_ratio,
    weight_decay,
    gradient_accumulation_steps,
    max_grad_norm,
    random_seed,
    deterministic_algorithms,
    log_interval_steps,
    eval_interval_steps,
    optimizer,
    global_step,
    optimizer_step,
    tokens_seen,
    samples_seen,
    run_id,
    optimizer_group_summary,
    longrope2_training_strategy,
):
    """保存可用于恢复训练的完整 checkpoint。"""
    architecture_metadata = get_model_architecture_metadata(model)
    model_config_payload = model.config.to_dict()
    checkpoint = {
        "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "model_abbr": GlobalConfig.model_abbr,
        "model_name_en": GlobalConfig.model_name_en,
        "model_name_zh": GlobalConfig.model_name_zh,
        "model_state_dict": model.state_dict(),
        "model_config": model_config_payload,
        "model_architecture_metadata": architecture_metadata,
        "source_manifest": _normalize_manifest_path(manifest_path),
        "eval_manifest": _normalize_manifest_path(eval_manifest_path),
        "tokenizer_category": tokenizer.name_or_path,
        "chat_template_version": GlobalConfig.chat_template_version,
        "tokenizer_eos_token": tokenizer.eos_token,
        "tokenizer_pad_token": tokenizer.pad_token,
        "training_mode": "lora" if getattr(GlobalConfig, "lora_mode", False) else "full",
        "training_stage": GlobalConfig.training_stage,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "current_learning_rate": _current_learning_rate(optimizer),
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_grad_norm": max_grad_norm,
        "random_seed": random_seed,
        "deterministic_algorithms": deterministic_algorithms,
        "log_interval_steps": log_interval_steps,
        "eval_interval_steps": eval_interval_steps,
        "loss": loss_value,
        "latest_eval_loss": eval_loss,
        "latest_eval_ppl": eval_ppl,
        "epoch": epoch_index,
        "total_epochs": total_epochs,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
        "samples_seen": samples_seen,
        "run_id": run_id,
        "optimizer_group_summary": optimizer_group_summary,
        "longrope2_training_strategy": longrope2_training_strategy,
    }
    torch.save(checkpoint, _checkpoint_file(checkpoint_root))
    return checkpoint


def _persist_training_state(
    checkpoint_root,
    optimizer,
    scheduler,
    save_optimizer,
    save_scheduler,
):
    if save_optimizer:
        torch.save(optimizer.state_dict(), _optimizer_file(checkpoint_root))
    if save_scheduler:
        torch.save(scheduler.state_dict(), _scheduler_file(checkpoint_root))


def _save_training_state_bundle(
    model,
    tokenizer,
    checkpoint_root,
    manifest_path,
    eval_manifest_path,
    optimizer,
    scheduler,
    loss_value,
    eval_loss,
    eval_ppl,
    epoch_index,
    total_epochs,
    batch_size,
    learning_rate,
    warmup_ratio,
    weight_decay,
    gradient_accumulation_steps,
    max_grad_norm,
    random_seed,
    deterministic_algorithms,
    log_interval_steps,
    eval_interval_steps,
    global_step,
    optimizer_step,
    tokens_seen,
    samples_seen,
    run_id,
    optimizer_group_summary,
    longrope2_training_strategy,
    save_optimizer,
    save_scheduler,
):
    _save_model_config_snapshot(model, checkpoint_root)
    checkpoint = _save_full_checkpoint(
        model=model,
        tokenizer=tokenizer,
        checkpoint_root=checkpoint_root,
        manifest_path=manifest_path,
        eval_manifest_path=eval_manifest_path,
        loss_value=loss_value,
        eval_loss=eval_loss,
        eval_ppl=eval_ppl,
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        deterministic_algorithms=deterministic_algorithms,
        log_interval_steps=log_interval_steps,
        eval_interval_steps=eval_interval_steps,
        optimizer=optimizer,
        global_step=global_step,
        optimizer_step=optimizer_step,
        tokens_seen=tokens_seen,
        samples_seen=samples_seen,
        run_id=run_id,
        optimizer_group_summary=optimizer_group_summary,
        longrope2_training_strategy=longrope2_training_strategy,
    )
    _persist_training_state(
        checkpoint_root,
        optimizer,
        scheduler,
        save_optimizer=save_optimizer,
        save_scheduler=save_scheduler,
    )
    return checkpoint


def _build_dataloader(dataset, tokenizer, batch_size, *, shuffle, pack_sequences, window_sampler=None):
    is_iterable_dataset = isinstance(dataset, IterableDataset)
    collate_builder = build_packed_training_batch if pack_sequences else build_training_batch

    def resolve_max_length():
        if window_sampler is None:
            return GlobalConfig.train_max_sequence_length
        return window_sampler.next_length()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not is_iterable_dataset,
        num_workers=0,
        pin_memory=(GlobalConfig.device.type == "cuda"),
        collate_fn=lambda batch: collate_builder(
            batch,
            tokenizer,
            max_length=resolve_max_length(),
        ),
    )


def train(
    model,
    dataset,
    tokenizer,
    save_path=None,
    manifest_path=None,
    eval_dataset=None,
    eval_manifest_path=None,
    training_profile=None,
    batch_size=None,
    target_total_epochs=None,
    lr=None,
    resume_checkpoint_path=None,
    initial_checkpoint_path=None,
    inference_weight_path=None,
    key_checkpoints=None,
):
    profile = _resolve_training_profile(training_profile)
    checkpoint_root = _normalize_checkpoint_path(save_path)
    if checkpoint_root is None:
        raise ValueError("save_path 不能为空，且必须指向用于保存 latest checkpoint 的路径。")

    if batch_size is None:
        batch_size = profile.batch_size
    if target_total_epochs is None:
        target_total_epochs = profile.target_total_epochs
    if lr is None:
        lr = profile.learning_rate
    if key_checkpoints is None:
        key_checkpoints = tuple(getattr(profile, "key_checkpoints", ()))

    warmup_ratio = profile.warmup_ratio
    weight_decay = float(getattr(profile, "weight_decay", 0.0))
    gradient_accumulation_steps = max(1, int(getattr(profile, "gradient_accumulation_steps", 1)))
    max_grad_norm = getattr(profile, "max_grad_norm", None)
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm 必须为正数或 None。")
    random_seed = getattr(profile, "random_seed", None)
    deterministic_algorithms = bool(getattr(profile, "deterministic_algorithms", False))
    tensorboard_enabled = bool(getattr(profile, "tensorboard_enabled", True))
    sequence_packing_enabled = bool(getattr(profile, "sequence_packing_enabled", False))
    log_interval_steps = max(1, int(getattr(profile, "log_interval_steps", 1)))
    eval_interval_steps = getattr(profile, "eval_interval_steps", None)
    if eval_interval_steps is not None:
        eval_interval_steps = int(eval_interval_steps)
        if eval_interval_steps <= 0:
            raise ValueError("eval_interval_steps 必须为正整数或 None。")
    eval_batch_size = getattr(profile, "eval_batch_size", None)
    if eval_batch_size is None:
        eval_batch_size = batch_size
    eval_batch_size = int(eval_batch_size)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size 必须为正整数。")
    eval_max_batches = getattr(profile, "eval_max_batches", None)
    if eval_max_batches is not None:
        eval_max_batches = int(eval_max_batches)
        if eval_max_batches <= 0:
            raise ValueError("eval_max_batches 必须为正整数或 None。")
    save_optimizer = profile.save_optimizer
    save_scheduler = profile.save_scheduler

    if target_total_epochs is None:
        raise ValueError("target_total_epochs 不能为空。")

    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    if inference_weight_path is not None:
        Path(inference_weight_path).parent.mkdir(parents=True, exist_ok=True)

    configure_training_runtime(
        seed=random_seed,
        deterministic_algorithms=deterministic_algorithms,
    )

    longrope2_factor_mismatch_keys = set()
    if bool(getattr(profile, "longrope2_auto_factor_refresh_enabled", True)):
        previous_factor_max_length = model.config.longrope2_factor_max_sequence_length
        _ensure_longrope2_dataset_factors(
            model,
            dataset,
            tokenizer,
            enabled=True,
        )
        if model.config.longrope2_factor_max_sequence_length != previous_factor_max_length:
            longrope2_factor_mismatch_keys.update(
                {
                    "longrope2_long_factors",
                    "longrope2_factor_max_sequence_length",
                }
            )

    train_window_sampler = _build_longrope2_window_sampler(
        model.config,
        profile,
        seed=random_seed,
        randomize=True,
    )
    eval_window_sampler = _build_longrope2_window_sampler(
        model.config,
        profile,
        seed=random_seed,
        randomize=False,
    )
    longrope2_training_strategy = _build_longrope2_training_strategy(
        model.config,
        train_window_sampler,
    )
    print(
        "LongRoPE2训练策略",
        json.dumps(longrope2_training_strategy, ensure_ascii=False),
    )

    dataloader = _build_dataloader(
        dataset,
        tokenizer,
        batch_size,
        shuffle=True,
        pack_sequences=sequence_packing_enabled,
        window_sampler=train_window_sampler,
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = _build_dataloader(
            eval_dataset,
            tokenizer,
            eval_batch_size,
            shuffle=False,
            pack_sequences=sequence_packing_enabled,
            window_sampler=eval_window_sampler,
        )

    optimizer, optimizer_group_summary = _build_optimizer(
        model,
        lr,
        weight_decay=weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        len(dataloader),
        target_total_epochs,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    print(
        "优化器参数分组",
        f"decay={optimizer_group_summary['decay_parameter_count']}",
        f"no_decay={optimizer_group_summary['no_decay_parameter_count']}",
        f"weight_decay={weight_decay}",
    )

    start_epoch = 0
    loaded_checkpoint = None
    global_step = 0
    optimizer_step = 0
    tokens_seen = 0
    samples_seen = 0
    latest_eval_loss = None
    latest_eval_ppl = None
    resume_root = _normalize_checkpoint_path(resume_checkpoint_path)
    if has_complete_training_state(
        resume_root,
        training_profile=profile,
        require_optimizer=save_optimizer,
        require_scheduler=save_scheduler,
    ):
        try:
            start_epoch, loaded_checkpoint = _load_resume_state(
                model,
                optimizer,
                scheduler,
                resume_root,
                save_optimizer=save_optimizer,
                save_scheduler=save_scheduler,
                allowed_architecture_mismatch_keys=longrope2_factor_mismatch_keys,
            )
            loaded_total_epochs = loaded_checkpoint.get("total_epochs")
            if loaded_total_epochs is not None and loaded_total_epochs != target_total_epochs:
                print(
                    f"续训目标轮数已调整: checkpoint 记录为 {loaded_total_epochs}，"
                    f"当前目标为 {target_total_epochs}。"
                )
            global_step = int(loaded_checkpoint.get("global_step", 0))
            optimizer_step = int(loaded_checkpoint.get("optimizer_step", global_step))
            tokens_seen = int(loaded_checkpoint.get("tokens_seen", 0))
            samples_seen = int(loaded_checkpoint.get("samples_seen", 0))
            latest_eval_loss = loaded_checkpoint.get("latest_eval_loss")
            latest_eval_ppl = loaded_checkpoint.get("latest_eval_ppl")
        except ValueError as error:
            print(f"发现不兼容的续训 checkpoint，已忽略自动恢复: {error}")
    elif resume_root is not None and _checkpoint_file(resume_root).exists():
        missing_files = _list_missing_training_state_files(
            resume_root,
            require_optimizer=save_optimizer,
            require_scheduler=save_scheduler,
        )
        print(
            "发现不完整的续训状态，已忽略自动恢复:",
            ", ".join(missing_files),
        )

    initial_root = _normalize_checkpoint_path(initial_checkpoint_path)
    if start_epoch == 0:
        if initial_root is not None and _checkpoint_file(initial_root).exists():
            try:
                loaded_checkpoint = _load_initial_model_state(
                    model,
                    initial_root,
                    allowed_architecture_mismatch_keys=longrope2_factor_mismatch_keys,
                )
            except ValueError as error:
                print(f"发现不兼容的初始化 checkpoint，已忽略加载: {error}")
        else:
            print("未发现可恢复训练状态，将从头开始训练。")

    run_id = _format_run_id() if loaded_checkpoint is None else loaded_checkpoint.get("run_id", _format_run_id())
    metric_logger = TrainingMetricLogger(
        checkpoint_root=checkpoint_root,
        run_id=run_id,
        tensorboard_enabled=tensorboard_enabled,
    )

    if start_epoch >= target_total_epochs:
        print(f"当前 checkpoint 已达到目标轮数 {target_total_epochs}，跳过训练。")
        checkpoint_loss = None if loaded_checkpoint is None else loaded_checkpoint.get("loss")
        try:
            _save_training_state_bundle(
                model=model,
                tokenizer=tokenizer,
                checkpoint_root=checkpoint_root,
                manifest_path=manifest_path,
                eval_manifest_path=eval_manifest_path,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_value=checkpoint_loss,
                eval_loss=latest_eval_loss,
                eval_ppl=latest_eval_ppl,
                epoch_index=start_epoch,
                total_epochs=target_total_epochs,
                batch_size=batch_size,
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                random_seed=random_seed,
                deterministic_algorithms=deterministic_algorithms,
                log_interval_steps=log_interval_steps,
                eval_interval_steps=eval_interval_steps,
                global_step=global_step,
                optimizer_step=optimizer_step,
                tokens_seen=tokens_seen,
                samples_seen=samples_seen,
                run_id=run_id,
                optimizer_group_summary=optimizer_group_summary,
                longrope2_training_strategy=longrope2_training_strategy,
                save_optimizer=save_optimizer,
                save_scheduler=save_scheduler,
            )
            _save_inference_weights(model, inference_weight_path)
            return model
        finally:
            metric_logger.close()

    key_checkpoint_set = set(key_checkpoints)
    optimizer.zero_grad(set_to_none=True)

    try:
        for current_epoch in range(start_epoch + 1, target_total_epochs + 1):
            total_loss = 0.0
            progress_bar = tqdm(dataloader, desc="Training")
            model.train()

            accumulation_loss_sum = 0.0
            accumulation_micro_batches = 0
            accumulation_tokens = 0
            accumulation_samples = 0
            log_window_loss_sum = 0.0
            log_window_grad_norm_sum = 0.0
            log_window_tokens = 0
            log_window_samples = 0
            log_window_steps = 0
            log_window_started_at = time.perf_counter()

            for batch_index, batch in enumerate(progress_bar, start=1):
                input_ids, labels, attention_mask, position_ids, segment_ids, batch_samples = _unpack_training_batch(
                    batch
                )
                input_ids = input_ids.to(GlobalConfig.device, non_blocking=True)
                labels = labels.to(GlobalConfig.device, non_blocking=True)
                attention_mask = attention_mask.to(GlobalConfig.device, non_blocking=True)
                if position_ids is not None:
                    position_ids = position_ids.to(GlobalConfig.device, non_blocking=True)
                if segment_ids is not None:
                    segment_ids = segment_ids.to(GlobalConfig.device, non_blocking=True)

                loss, _ = _forward_batch(
                    model,
                    input_ids,
                    labels,
                    attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                )
                raw_loss = float(loss.item())
                total_loss += raw_loss

                batch_tokens = _batch_token_count(attention_mask)
                accumulation_loss_sum += raw_loss
                accumulation_micro_batches += 1
                accumulation_tokens += batch_tokens
                accumulation_samples += batch_samples
                (loss / gradient_accumulation_steps).backward()

                should_step = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == len(dataloader)
                )
                if not should_step:
                    progress_bar.set_postfix(loss=raw_loss, epoch=current_epoch, accum=accumulation_micro_batches)
                    continue

                if max_grad_norm is not None:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in model.parameters() if parameter.requires_grad],
                            max_grad_norm,
                        ).item()
                    )
                else:
                    grad_norm = _compute_grad_norm(model.parameters())

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                optimizer_step += 1
                tokens_seen += accumulation_tokens
                samples_seen += accumulation_samples

                update_loss = accumulation_loss_sum / max(1, accumulation_micro_batches)
                log_window_loss_sum += update_loss
                log_window_grad_norm_sum += grad_norm
                log_window_tokens += accumulation_tokens
                log_window_samples += accumulation_samples
                log_window_steps += 1

                progress_bar.set_postfix(loss=update_loss, epoch=current_epoch, step=global_step)

                if global_step % log_interval_steps == 0:
                    elapsed = max(1e-6, time.perf_counter() - log_window_started_at)
                    train_metrics = {
                        "epoch": current_epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "loss": log_window_loss_sum / log_window_steps,
                        "lr": _current_learning_rate(optimizer),
                        "tokens_per_sec": log_window_tokens / elapsed,
                        "samples_per_sec": log_window_samples / elapsed,
                        "grad_norm": log_window_grad_norm_sum / log_window_steps,
                        "tokens_seen": tokens_seen,
                        "samples_seen": samples_seen,
                    }
                    metric_logger.log("train", global_step, train_metrics)
                    log_window_loss_sum = 0.0
                    log_window_grad_norm_sum = 0.0
                    log_window_tokens = 0
                    log_window_samples = 0
                    log_window_steps = 0
                    log_window_started_at = time.perf_counter()

                if eval_dataloader is not None and eval_interval_steps is not None and global_step % eval_interval_steps == 0:
                    eval_metrics = _evaluate_model(
                        model,
                        eval_dataloader,
                        max_batches=eval_max_batches,
                    )
                    latest_eval_loss = eval_metrics["eval_loss"]
                    latest_eval_ppl = eval_metrics["eval_ppl"]
                    metric_logger.log(
                        "eval",
                        global_step,
                        {
                            "epoch": current_epoch,
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            **eval_metrics,
                        },
                    )

                accumulation_loss_sum = 0.0
                accumulation_micro_batches = 0
                accumulation_tokens = 0
                accumulation_samples = 0

            progress_bar.close()

            if log_window_steps > 0:
                elapsed = max(1e-6, time.perf_counter() - log_window_started_at)
                metric_logger.log(
                    "train",
                    global_step,
                    {
                        "epoch": current_epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "loss": log_window_loss_sum / log_window_steps,
                        "lr": _current_learning_rate(optimizer),
                        "tokens_per_sec": log_window_tokens / elapsed,
                        "samples_per_sec": log_window_samples / elapsed,
                        "grad_norm": log_window_grad_norm_sum / log_window_steps,
                        "tokens_seen": tokens_seen,
                        "samples_seen": samples_seen,
                    },
                )

            average_loss = total_loss / max(1, len(dataloader))
            print(f"Epoch {current_epoch} average loss: {average_loss:.4f}")

            if eval_dataloader is not None:
                eval_metrics = _evaluate_model(
                    model,
                    eval_dataloader,
                    max_batches=eval_max_batches,
                )
                latest_eval_loss = eval_metrics["eval_loss"]
                latest_eval_ppl = eval_metrics["eval_ppl"]
                metric_logger.log(
                    "eval",
                    global_step,
                    {
                        "epoch": current_epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        **eval_metrics,
                    },
                )

            checkpoint = _save_training_state_bundle(
                model=model,
                tokenizer=tokenizer,
                checkpoint_root=checkpoint_root,
                manifest_path=manifest_path,
                eval_manifest_path=eval_manifest_path,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_value=average_loss,
                eval_loss=latest_eval_loss,
                eval_ppl=latest_eval_ppl,
                epoch_index=current_epoch,
                total_epochs=target_total_epochs,
                batch_size=batch_size,
                learning_rate=lr,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                random_seed=random_seed,
                deterministic_algorithms=deterministic_algorithms,
                log_interval_steps=log_interval_steps,
                eval_interval_steps=eval_interval_steps,
                global_step=global_step,
                optimizer_step=optimizer_step,
                tokens_seen=tokens_seen,
                samples_seen=samples_seen,
                run_id=run_id,
                optimizer_group_summary=optimizer_group_summary,
                longrope2_training_strategy=longrope2_training_strategy,
                save_optimizer=save_optimizer,
                save_scheduler=save_scheduler,
            )

            if current_epoch in key_checkpoint_set:
                epoch_checkpoint_root = checkpoint_root.parent / f"epoch_{current_epoch}"
                torch.save(checkpoint, _checkpoint_file(epoch_checkpoint_root))
                _persist_training_state(
                    epoch_checkpoint_root,
                    optimizer,
                    scheduler,
                    save_optimizer=save_optimizer,
                    save_scheduler=save_scheduler,
                )

            _save_inference_weights(model, inference_weight_path)

        return model
    finally:
        metric_logger.close()
