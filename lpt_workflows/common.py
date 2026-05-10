"""LPT v2 阶段工作流公共工具。"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from lpt_config import (
    BaseTrainingRecipeConfig,
    DEFAULT_MODEL_SIZE_PRESET,
    DEFAULT_TRAINING_SEED,
    GenerationConfig,
    GlobalConfig,
    TOKENIZER_PATH,
    build_longrope2_uniform_factors,
    load_longrope2_factors_file,
)
from lpt_data import (
    build_streaming_manifest_dataset,
    load_dataset_manifest,
    load_dataset_records,
    prepare_tokenizer,
    summarize_dataset_sources,
    summarize_dataset_types,
)
from lpt_eval import build_lpt_v2_profile_config
from lpt_model import LPTV2, load_lpt_v2_checkpoint
from lpt_runtime import (
    apply_inference_execution_plan,
    describe_execution_plan,
    resolve_execution_plan,
)
from lpt_runtime.files import is_torch_save_file_readable


DEFAULT_TRAINING_RECIPE = BaseTrainingRecipeConfig()


@dataclass(frozen=True)
class LongRoPE2WorkflowOptions:
    """工作流级 LongRoPE2 覆盖项。"""

    train_max_sequence_length: int | None = None
    train_rope_cache_max_sequence_length: int | None = None
    inference_rope_cache_max_sequence_length: int | None = None
    original_window: int | None = None
    target_window: int | None = None
    long_factors_path: Path | None = None
    train_embedding_mode: str | None = None
    inference_embedding_mode: str | None = None
    mixed_original_window: int | None = None
    window_lengths: tuple[int, ...] | None = None
    window_weights: tuple[float, ...] | None = None


def build_local_tokenizer(tokenizer_path=TOKENIZER_PATH):
    """加载本地 DS tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    return prepare_tokenizer(tokenizer)


def _hash_file_sha256(path):
    """计算文件 SHA256；文件不存在时返回 None。"""
    if path is None or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tokenizer_metadata(tokenizer, tokenizer_path=TOKENIZER_PATH):
    """生成 tokenizer 元数据。"""
    resolved_path = None if tokenizer_path is None else Path(tokenizer_path)
    tokenizer_config_path = None if resolved_path is None else resolved_path / "tokenizer_config.json"
    special_tokens_map = {
        key: str(value)
        for key, value in (getattr(tokenizer, "special_tokens_map", {}) or {}).items()
    }
    return {
        "tokenizer_path": None if resolved_path is None else str(resolved_path),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_config_sha256": _hash_file_sha256(tokenizer_config_path),
        "vocab_size": len(tokenizer),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "special_tokens_map": special_tokens_map,
        "chat_template_version": GlobalConfig.chat_template_version,
    }


def load_dataset_from_manifest(
    manifest_path,
    *,
    expected_types,
    seed=DEFAULT_TRAINING_SEED,
    shuffle_buffer_size=None,
):
    """按 manifest 构造流式训练数据集。"""
    dataset = build_streaming_manifest_dataset(
        manifest_path,
        expected_types=expected_types,
        shuffle_buffer_size=shuffle_buffer_size or GlobalConfig.dataset_shuffle_buffer_size,
        seed=seed,
    )
    print(f"manifest={manifest_path}")
    print(f"datasets={dataset.loaded_datasets}")
    print(f"types={dataset.summary_types}")
    print(f"sources={dataset.summary_sources}")
    return dataset


def load_structured_dataset(dataset_path, *, expected_types=None):
    """直接读取单个 structured JSONL 数据集。"""
    records = load_dataset_records(dataset_path)
    if expected_types is not None:
        invalid_types = sorted({record["type"] for record in records} - set(expected_types))
        if invalid_types:
            raise ValueError(f"{dataset_path} 包含不被允许的样本类型: {invalid_types}")
    print(f"dataset={dataset_path}")
    print(f"types={summarize_dataset_types(records)}")
    print(f"sources={summarize_dataset_sources(records)}")
    return records


def load_eval_dataset(eval_manifest_path, *, expected_types):
    """按需加载验证 manifest。"""
    if eval_manifest_path is None:
        return None
    records, loaded_datasets = load_dataset_manifest(eval_manifest_path, expected_types=expected_types)
    print(f"eval_manifest={eval_manifest_path}")
    print(f"eval_datasets={loaded_datasets}")
    return records


def resolve_torch_device(raw_device="auto"):
    """解析训练/推理 device。"""
    device_text = str(raw_device)
    if device_text == "auto":
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_text)


def resolve_torch_dtype(raw_dtype="auto", *, device=None):
    """解析 dtype。"""
    dtype_text = str(raw_dtype)
    device = device or resolve_torch_device("auto")
    if dtype_text == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_text in {"fp32", "float32"}:
        return torch.float32
    if dtype_text in {"fp16", "float16"}:
        return torch.float16
    if dtype_text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"不支持的 dtype: {raw_dtype}")


def apply_longrope2_runtime_overrides(options):
    """应用运行时 LongRoPE2 长度覆盖。"""
    if options.train_max_sequence_length is not None:
        GlobalConfig.train_max_sequence_length = int(options.train_max_sequence_length)
        if options.train_rope_cache_max_sequence_length is None:
            GlobalConfig.train_rope_cache_max_sequence_length = int(options.train_max_sequence_length)
    if options.train_rope_cache_max_sequence_length is not None:
        GlobalConfig.train_rope_cache_max_sequence_length = int(options.train_rope_cache_max_sequence_length)
    if options.inference_rope_cache_max_sequence_length is not None:
        GlobalConfig.inference_rope_cache_max_sequence_length = int(options.inference_rope_cache_max_sequence_length)


def apply_longrope2_model_config_overrides(config, options):
    """基于 CLI 覆盖派生 ModelConfig。"""
    overrides = {}
    if options.original_window is not None:
        overrides["original_max_len"] = int(options.original_window)
    if options.target_window is not None:
        overrides["longrope2_target_length"] = int(options.target_window)
    if options.long_factors_path is not None:
        overrides["longrope2_long_factors"] = load_longrope2_factors_file(options.long_factors_path)
    elif options.target_window is not None and options.target_window > config.original_max_len:
        overrides["longrope2_long_factors"] = build_longrope2_uniform_factors(
            config,
            int(options.target_window),
        )
        overrides["longrope2_factor_max_sequence_length"] = int(options.target_window)
    if options.train_embedding_mode is not None:
        overrides["longrope2_train_embedding_mode"] = str(options.train_embedding_mode)
    if options.inference_embedding_mode is not None:
        overrides["longrope2_inference_embedding_mode"] = str(options.inference_embedding_mode)
    if options.mixed_original_window is not None:
        overrides["longrope2_mixed_original_window"] = int(options.mixed_original_window)
    return config.with_overrides(**overrides) if overrides else config


def build_workflow_model_config(args, *, checkpoint_path=None):
    """根据 profile/preset/LongRoPE2 CLI 构建 v2 ModelConfig。"""
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=False)
        return loaded.model.config
    config = build_lpt_v2_profile_config(args.profile, preset=args.preset)
    options = build_longrope2_options_from_args(args)
    apply_longrope2_runtime_overrides(options)
    return apply_longrope2_model_config_overrides(config, options)


def instantiate_model(vocabulary_size, config, *, device="auto", dtype="auto"):
    """实例化并移动 LPTV2。"""
    torch_device = resolve_torch_device(device)
    torch_dtype = resolve_torch_dtype(dtype, device=torch_device)
    GlobalConfig.parameter_dtype = torch_dtype
    GlobalConfig.device = torch_device
    model = LPTV2(vocabulary_size, config)
    model.to(device=torch_device, dtype=torch_dtype)
    return model


def load_checkpoint_model(checkpoint_path, *, device="auto", dtype="auto", strict=True):
    """加载 LPT v2 checkpoint 并放置到目标设备。"""
    loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=strict)
    model = loaded.model
    torch_device = resolve_torch_device(device)
    torch_dtype = resolve_torch_dtype(dtype, device=torch_device)
    model.to(device=torch_device, dtype=torch_dtype)
    GlobalConfig.device = torch_device
    GlobalConfig.parameter_dtype = torch_dtype
    return model


def resolve_checkpoint_file(checkpoint_root):
    """把目录或文件规范化为 model checkpoint 文件路径。"""
    path = Path(checkpoint_root)
    if path.is_file():
        return path
    return path / "model.pt"


def find_existing_model_checkpoint(*candidates):
    """返回第一个存在的模型 checkpoint 文件。"""
    from lpt_training import resolve_latest_training_checkpoint

    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.suffix:
            checkpoint_root = resolve_latest_training_checkpoint(candidate_path, lora_mode=False)
            path = None if checkpoint_root is None else checkpoint_root / "model.pt"
        else:
            path = resolve_checkpoint_file(candidate_path)
        if path is not None and path.exists() and is_torch_save_file_readable(path):
            return path
    return None


def load_state_dict_weights(model, weight_path):
    """加载 plain state_dict 推理权重。"""
    state_dict = torch.load(Path(weight_path), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "推理权重键不匹配: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def apply_execution_plan_for_inference(model, execution_config=None):
    """解析并应用推理执行计划。"""
    execution_plan = resolve_execution_plan(
        execution_config,
        num_layers=model.config.num_layers,
    )
    if execution_config is None or execution_config.print_device_map:
        print(describe_execution_plan(execution_plan))
    return apply_inference_execution_plan(model, execution_plan)


def build_generation_config_from_args(args):
    """从 CLI 生成 GenerationConfig。"""
    return GenerationConfig(
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        repetition_window_size=args.repetition_window_size,
    )


def add_generation_arguments(parser):
    """注册生成参数。"""
    generation = GenerationConfig()
    parser.add_argument("--max-new-tokens", type=int, default=generation.max_length, help="最多生成 token 数。")
    parser.add_argument("--temperature", type=float, default=generation.temperature, help="采样温度。")
    parser.add_argument("--top-k", type=int, default=generation.top_k, help="top-k 采样。")
    parser.add_argument("--top-p", type=float, default=generation.top_p, help="top-p 采样。")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码。")
    parser.add_argument("--repetition-penalty", type=float, default=generation.repetition_penalty, help="重复惩罚。")
    parser.add_argument("--repetition-window-size", type=int, default=generation.repetition_window_size, help="重复惩罚窗口。")


def add_model_arguments(parser, recipe=None):
    """注册模型规格参数。"""
    recipe = recipe or DEFAULT_TRAINING_RECIPE
    parser.add_argument("--profile", default=recipe.profile, help="v2 运行 profile。")
    parser.add_argument("--preset", default=recipe.preset or DEFAULT_MODEL_SIZE_PRESET, help="v2 模型规格 preset。")
    parser.add_argument("--device", default=recipe.device, help="auto/cpu/cuda/cuda:0。")
    parser.add_argument("--dtype", default=recipe.dtype, help="auto/fp32/fp16/bf16。")


def add_training_arguments(parser, recipe=None):
    """注册通用训练参数。"""
    recipe = recipe or DEFAULT_TRAINING_RECIPE
    add_model_arguments(parser, recipe=recipe)
    parser.add_argument("--batch-size", type=int, default=recipe.batch_size, help="训练 batch size。")
    parser.add_argument("--epochs", type=int, default=recipe.target_total_epochs, help="训练 epoch 数。")
    parser.add_argument("--target-total-epochs", dest="epochs", type=int, help="训练总 epoch 数，等价于 --epochs。")
    parser.add_argument("--max-steps", type=int, default=recipe.max_steps, help="最多训练 step 数；不传表示按 epochs 跑完整数据。")
    parser.add_argument("--learning-rate", type=float, default=recipe.learning_rate, help="AdamW 学习率。")
    parser.add_argument("--weight-decay", type=float, default=recipe.weight_decay, help="权重衰减。")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=recipe.gradient_accumulation_steps, help="梯度累积步数。")
    parser.add_argument("--max-grad-norm", type=float, default=recipe.max_grad_norm, help="梯度裁剪阈值。")
    parser.add_argument("--warmup-ratio", type=float, default=recipe.warmup_ratio, help="warmup 占计划训练 step 数的比例。")
    parser.add_argument("--log-interval", type=int, default=recipe.log_interval_steps, help="日志间隔。")
    parser.add_argument("--eval-interval", type=int, default=recipe.eval_interval_steps, help="验证间隔，0 表示关闭。")
    parser.add_argument("--eval-batch-size", type=int, default=recipe.eval_batch_size, help="验证 batch size，默认沿用训练 batch size。")
    parser.add_argument("--eval-max-batches", type=int, default=recipe.eval_max_batches, help="每次验证最多 batch 数；不传表示完整验证集。")
    parser.add_argument("--save-interval", type=int, default=recipe.save_interval_steps, help="额外 step_N checkpoint 间隔，0 表示关闭。")
    parser.add_argument("--latest-save-interval", type=int, default=recipe.latest_save_interval_steps, help="覆盖 latest checkpoint 间隔，0 表示仅训练结束保存。")
    parser.add_argument("--key-checkpoints", type=_parse_int_tuple, default=recipe.key_checkpoints, help="必须额外保存的 step，逗号分隔。")
    parser.add_argument("--save-best-checkpoint", action="store_true", default=recipe.save_best_checkpoint, help="保存 loss 最低的一份完整 checkpoint。")
    parser.add_argument("--best-checkpoint-metric", choices=("loss", "eval_loss"), default=recipe.best_checkpoint_metric, help="best checkpoint 比较指标。")
    parser.add_argument("--best-checkpoint-min-delta", type=float, default=recipe.best_checkpoint_min_delta, help="刷新 best checkpoint 的最小改进幅度。")
    parser.add_argument("--seed", type=int, default=recipe.random_seed, help="随机种子。")
    parser.add_argument("--deterministic", dest="deterministic_algorithms", action="store_true", default=recipe.deterministic_algorithms, help="启用确定性算法。")
    parser.add_argument("--no-deterministic", dest="deterministic_algorithms", action="store_false", help="关闭确定性算法。")
    parser.add_argument("--no-sequence-packing", action="store_true", default=not recipe.sequence_packing_enabled, help="关闭 sequence packing。")
    parser.add_argument("--no-resume", action="store_true", help="忽略 latest checkpoint，重新训练。")
    parser.add_argument("--run-id", default=recipe.run_id, help="实验运行 ID，默认自动生成。")
    parser.add_argument("--no-tensorboard", action="store_true", default=not recipe.tensorboard_enabled, help="关闭 TensorBoard 输出。")
    parser.add_argument("--no-save-optimizer", action="store_true", default=not recipe.save_optimizer, help="checkpoint 不保存 optimizer。")
    parser.add_argument("--no-save-scheduler", action="store_true", default=not recipe.save_scheduler, help="checkpoint 不保存 scheduler。")
    parser.add_argument("--no-save-inference-weights", action="store_true", default=False, help="训练结束不额外导出 weights 目录下的推理权重。")


def add_longrope2_training_arguments(parser: ArgumentParser, recipe=None):
    """注册 LongRoPE2 工作流参数。"""
    recipe = recipe or DEFAULT_TRAINING_RECIPE
    parser.add_argument("--train-max-sequence-length", type=int, default=recipe.train_max_sequence_length, help="训练截断长度。")
    parser.add_argument("--train-rope-cache-max-sequence-length", type=int, default=recipe.train_rope_cache_max_sequence_length, help="训练 RoPE cache 长度。")
    parser.add_argument("--inference-rope-cache-max-sequence-length", type=int, default=recipe.inference_rope_cache_max_sequence_length, help="推理 RoPE cache 长度。")
    parser.add_argument("--longrope2-original-window", type=int, default=recipe.longrope2_original_window, help="LongRoPE2 原始窗口。")
    parser.add_argument("--longrope2-target-window", type=int, default=recipe.longrope2_target_window, help="LongRoPE2 目标窗口。")
    parser.add_argument("--longrope2-long-factors-path", type=Path, default=recipe.longrope2_long_factors_path, help="LongRoPE2 factors 文件。")
    parser.add_argument("--longrope2-train-embedding-mode", default=recipe.longrope2_train_embedding_mode, help="训练 embedding mode。")
    parser.add_argument("--longrope2-inference-embedding-mode", default=recipe.longrope2_inference_embedding_mode, help="推理 embedding mode。")
    parser.add_argument("--longrope2-mixed-original-window", type=int, default=recipe.longrope2_mixed_original_window, help="mixed 原始窗口。")
    parser.add_argument("--longrope2-window-lengths", type=_parse_int_tuple, default=recipe.longrope2_window_lengths, help="训练 batch 混合采样窗口，逗号分隔。")
    parser.add_argument("--longrope2-window-weights", type=_parse_float_tuple, default=recipe.longrope2_window_weights, help="训练 batch 窗口采样权重，逗号分隔。")


def _parse_int_tuple(raw_value):
    values = tuple(int(value.strip()) for value in str(raw_value).split(",") if value.strip())
    if not values:
        raise ValueError("至少需要提供一个整数。")
    return values


def _parse_float_tuple(raw_value):
    values = tuple(float(value.strip()) for value in str(raw_value).split(",") if value.strip())
    if not values:
        raise ValueError("至少需要提供一个数值。")
    return values


def build_longrope2_options_from_args(args):
    """从 argparse 结果构造 LongRoPE2 选项。"""
    return LongRoPE2WorkflowOptions(
        train_max_sequence_length=getattr(args, "train_max_sequence_length", None),
        train_rope_cache_max_sequence_length=getattr(args, "train_rope_cache_max_sequence_length", None),
        inference_rope_cache_max_sequence_length=getattr(args, "inference_rope_cache_max_sequence_length", None),
        original_window=getattr(args, "longrope2_original_window", None),
        target_window=getattr(args, "longrope2_target_window", None),
        long_factors_path=getattr(args, "longrope2_long_factors_path", None),
        train_embedding_mode=getattr(args, "longrope2_train_embedding_mode", None),
        inference_embedding_mode=getattr(args, "longrope2_inference_embedding_mode", None),
        mixed_original_window=getattr(args, "longrope2_mixed_original_window", None),
        window_lengths=getattr(args, "longrope2_window_lengths", None),
        window_weights=getattr(args, "longrope2_window_weights", None),
    )


def merge_training_args_with_recipe(args, recipe):
    """把阶段 recipe 默认值与外部注入参数合并为 argparse 风格对象。"""
    defaults = {
        "profile": recipe.profile,
        "preset": recipe.preset,
        "device": recipe.device,
        "dtype": recipe.dtype,
        "manifest_path": recipe.manifest_path,
        "artifact_dir": recipe.artifact_dir,
        "batch_size": recipe.batch_size,
        "epochs": recipe.target_total_epochs,
        "max_steps": recipe.max_steps,
        "learning_rate": recipe.learning_rate,
        "weight_decay": recipe.weight_decay,
        "gradient_accumulation_steps": recipe.gradient_accumulation_steps,
        "max_grad_norm": recipe.max_grad_norm,
        "warmup_ratio": recipe.warmup_ratio,
        "log_interval": recipe.log_interval_steps,
        "eval_interval": recipe.eval_interval_steps,
        "eval_batch_size": recipe.eval_batch_size,
        "eval_max_batches": recipe.eval_max_batches,
        "save_interval": recipe.save_interval_steps,
        "latest_save_interval": recipe.latest_save_interval_steps,
        "key_checkpoints": recipe.key_checkpoints,
        "save_best_checkpoint": recipe.save_best_checkpoint,
        "best_checkpoint_metric": recipe.best_checkpoint_metric,
        "best_checkpoint_min_delta": recipe.best_checkpoint_min_delta,
        "seed": recipe.random_seed,
        "deterministic_algorithms": recipe.deterministic_algorithms,
        "no_sequence_packing": not recipe.sequence_packing_enabled,
        "no_resume": False,
        "run_id": recipe.run_id,
        "no_tensorboard": not recipe.tensorboard_enabled,
        "no_save_optimizer": not recipe.save_optimizer,
        "no_save_scheduler": not recipe.save_scheduler,
        "no_save_inference_weights": False,
        "train_max_sequence_length": recipe.train_max_sequence_length,
        "train_rope_cache_max_sequence_length": recipe.train_rope_cache_max_sequence_length,
        "inference_rope_cache_max_sequence_length": recipe.inference_rope_cache_max_sequence_length,
        "longrope2_original_window": recipe.longrope2_original_window,
        "longrope2_target_window": recipe.longrope2_target_window,
        "longrope2_long_factors_path": recipe.longrope2_long_factors_path,
        "longrope2_train_embedding_mode": recipe.longrope2_train_embedding_mode,
        "longrope2_inference_embedding_mode": recipe.longrope2_inference_embedding_mode,
        "longrope2_mixed_original_window": recipe.longrope2_mixed_original_window,
        "longrope2_window_lengths": recipe.longrope2_window_lengths,
        "longrope2_window_weights": recipe.longrope2_window_weights,
    }
    if hasattr(recipe, "lora_rank"):
        defaults.update(
            {
                "lora_rank": recipe.lora_rank,
                "lora_alpha": recipe.lora_alpha,
                "lora_dropout": recipe.lora_dropout,
                "lora_base_source": recipe.lora_base_source,
                "lora_target_modules": ",".join(recipe.lora_target_modules),
            }
        )
    if args is not None:
        source = {
            key: getattr(args, key)
            for key in defaults
            if hasattr(args, key)
        }
        if hasattr(args, "__dict__"):
            source.update(vars(args))
        defaults.update(source)
    return SimpleNamespace(**defaults)
