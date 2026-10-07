"""LPT v2 文本预训练工作流。"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lpt_config import GlobalConfig, TextPretrainingConfig
from lpt_config.constants import DEFAULT_OPTIMIZER_TYPE, DEFAULT_GRADIENT_CHECKPOINTING
from lpt_inference import display_model_parameter_summary
from lpt_data.cache import TokenCache
from lpt_data.io import (
    entry_progress_key,
    load_manifest_entries,
    resolve_manifest_dataset_path,
)
from lpt_data.token_cache_dataset import (
    TokenCacheManifestDataset,
    cache_subdir_name,
    ensure_manifest_token_cache,
)
from lpt_model import LPTV2
from lpt_training import (
    TrainingRunConfig,
    load_data_progress,
    resolve_latest_training_checkpoint,
    train,
)

from .common import (
    add_longrope2_training_arguments,
    add_training_arguments,
    build_local_tokenizer,
    build_longrope2_options_from_args,
    build_tokenizer_metadata,
    build_workflow_model_config,
    find_existing_model_checkpoint,
    instantiate_model,
    load_checkpoint_model,
    load_dataset_from_manifest,
    load_eval_dataset,
    load_state_dict_weights,
    merge_training_args_with_recipe,
    apply_longrope2_runtime_overrides,
)


TEXT_PRETRAIN_RECIPE = TextPretrainingConfig()
TEXT_PRETRAIN_MANIFEST_PATH = TEXT_PRETRAIN_RECIPE.manifest_path
TEXT_PRETRAIN_ARTIFACT_DIR = TEXT_PRETRAIN_RECIPE.artifact_dir


def resolve_text_pretrain_checkpoint_root():
    """返回 text_pretrain latest checkpoint 根目录。"""
    return TEXT_PRETRAIN_ARTIFACT_DIR / "checkpoints" / "latest"


def resolve_text_pretrain_inference_weight_path():
    """返回 text_pretrain 推理权重导出路径。"""
    return TEXT_PRETRAIN_ARTIFACT_DIR / "weights" / "model_weights.pth"


def resolve_text_pretrain_resume_checkpoint():
    """查找 text_pretrain 可续训 checkpoint。"""
    checkpoint_root = resolve_text_pretrain_checkpoint_root()
    return resolve_latest_training_checkpoint(checkpoint_root, lora_mode=False)


def load_text_pretrained_model_for_inference(execution_config=None, *, device="auto", dtype="auto"):
    """加载 text_pretrain 模型用于推理。"""
    tokenizer = build_local_tokenizer()
    checkpoint_path = find_existing_model_checkpoint(
        resolve_text_pretrain_inference_weight_path().with_name("model_checkpoint.pt"),
        resolve_text_pretrain_checkpoint_root(),
    )
    if checkpoint_path is not None:
        model = load_checkpoint_model(checkpoint_path, device=device, dtype=dtype, strict=True)
    else:
        config = build_workflow_model_config(
            merge_training_args_with_recipe(None, TEXT_PRETRAIN_RECIPE),
        )
        model = instantiate_model(len(tokenizer), config, device=device, dtype=dtype)
        weight_path = resolve_text_pretrain_inference_weight_path()
        if not weight_path.exists():
            raise FileNotFoundError(f"未找到 text_pretrain 推理权重: {weight_path}")
        load_state_dict_weights(model, weight_path)
    from .common import apply_execution_plan_for_inference

    model = apply_execution_plan_for_inference(model, execution_config)
    return model, tokenizer


def train_text_pretrained_model(
    manifest_path=TEXT_PRETRAIN_MANIFEST_PATH,
    *,
    eval_manifest_path=None,
    args=None,
):
    """执行 text pretrain。"""
    args = merge_training_args_with_recipe(args, TEXT_PRETRAIN_RECIPE)
    apply_longrope2_runtime_overrides(build_longrope2_options_from_args(args))
    tokenizer = build_local_tokenizer()
    resume_checkpoint = (
        None if args.no_resume else resolve_text_pretrain_resume_checkpoint()
    )
    completed_entries = {}
    if (
        resume_checkpoint is not None
        and getattr(args, "token_cache_enabled", False)
        and getattr(args, "token_cache_dir", None) is not None
    ):
        data_progress = load_data_progress(resume_checkpoint)
        completed_entries = _collect_completed_entry_meta(
            manifest_path,
            data_progress,
            args.token_cache_dir,
            target_epochs=args.epochs,
        )
        if _all_manifest_entries_completed(manifest_path, completed_entries):
            if getattr(args, "build_cache_only", False):
                print("所有 manifest 数据集已完成训练且源文件未变动，无需构建 token cache。")
            else:
                print("所有 manifest 数据集已完成训练且源文件未变动，跳过本次训练。")
                print("如需重新调整 weight/sample_limit，请先重置对应数据集进度。")
            return None, tokenizer, None
    dataset, token_cache_dir = load_text_pretrain_dataset(
        manifest_path,
        tokenizer,
        args,
        completed_entries=completed_entries,
    )
    if getattr(args, "build_cache_only", False):
        print("--build-cache-only：token cache 已就绪，跳过模型与训练。")
        return None, tokenizer, None
    eval_dataset = load_eval_dataset(eval_manifest_path, expected_types={"text"})
    checkpoint_file = None if resume_checkpoint is None else resume_checkpoint / "model.pt"
    if checkpoint_file is not None and checkpoint_file.exists():
        # 续训优先恢复 latest/step checkpoint 内的完整 ModelConfig 和权重。
        model = load_checkpoint_model(checkpoint_file, device=args.device, dtype=args.dtype, strict=True)
    else:
        # 首次 text_pretrain 从 profile/preset/LongRoPE2 覆盖项构造 v2-only 模型。
        config = build_workflow_model_config(args)
        model = instantiate_model(len(tokenizer), config, device=args.device, dtype=args.dtype)
    display_model_parameter_summary(model)
    print(f"tokenizer={build_tokenizer_metadata(tokenizer)}")
    trainer_state = train(
        model,
        tokenizer,
        dataset,
        config=TrainingRunConfig(
            training_stage="text_pretrain",
            artifact_dir=TEXT_PRETRAIN_ARTIFACT_DIR,
            checkpoint_dir=resolve_text_pretrain_checkpoint_root(),
            inference_weight_path=resolve_text_pretrain_inference_weight_path(),
            save_inference_weights=not args.no_save_inference_weights,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_type=getattr(args, "optimizer_type", DEFAULT_OPTIMIZER_TYPE),
            gradient_checkpointing=getattr(args, "gradient_checkpointing", DEFAULT_GRADIENT_CHECKPOINTING),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            lm_loss_chunk_tokens=args.lm_loss_chunk_tokens,
            warmup_ratio=args.warmup_ratio,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            eval_batch_size=args.eval_batch_size,
            eval_max_batches=args.eval_max_batches,
            save_interval=args.save_interval,
            latest_save_interval=args.latest_save_interval,
            key_checkpoints=args.key_checkpoints,
            save_best_checkpoint=args.save_best_checkpoint,
            best_checkpoint_metric=args.best_checkpoint_metric,
            best_checkpoint_min_delta=args.best_checkpoint_min_delta,
            save_optimizer=not args.no_save_optimizer,
            save_scheduler=not args.no_save_scheduler,
            max_sequence_length=args.train_max_sequence_length,
            sequence_packing=not args.no_sequence_packing,
            data_num_workers=args.data_num_workers,
            data_prefetch_factor=args.data_prefetch_factor,
            data_persistent_workers=args.data_persistent_workers,
            data_pin_memory=args.data_pin_memory,
            data_reader_num_workers=args.data_reader_num_workers,
            data_reader_chunk_bytes=args.data_reader_chunk_bytes,
            token_cache_dir=token_cache_dir,
            token_cache_enabled=token_cache_dir is not None,
            data_length_bucket_window=args.data_length_bucket_window,
            data_packing_strategy=args.data_packing_strategy,
            thinking_mode=args.thinking_mode,
            thinking_visibility=args.thinking_visibility,
            seed=args.seed,
            deterministic_algorithms=args.deterministic_algorithms,
            resume_checkpoint=resume_checkpoint,
            source_manifest=manifest_path,
            eval_manifest=eval_manifest_path,
            longrope2_window_lengths=args.longrope2_window_lengths,
            longrope2_window_weights=args.longrope2_window_weights,
            tokenizer_metadata=build_tokenizer_metadata(tokenizer),
            run_id=args.run_id,
            tensorboard_enabled=not args.no_tensorboard,
        ),
        eval_dataset=eval_dataset,
    )
    return model, tokenizer, trainer_state


def _count_jsonl_lines(path):
    """轻量统计 JSONL 非空行数，不解析 JSON，用于无 cache 时的条数校验。"""
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return count


def _collect_completed_entry_meta(
    manifest_path,
    data_progress,
    cache_dir,
    *,
    target_epochs,
):
    """收集已完成训练且源文件未变动的 entry_key -> source_record_count。

    ``data_progress`` 来自 latest checkpoint 的 data_progress.json，记录每个
    数据集已完成的 epoch 数和样本数。已完成判定要求：

    * 存在进度记录且 ``completed_epochs >= target_epochs``、样本数达标；
    * 数据未变动：已有 per-file cache 时校验源文件指纹（mtime/size/头尾
      sha256）；没有 cache 时用轻量非空行计数与进度记录的条数对比。

    返回的条目可以安全跳过 cache 构建和训练；若用户改了 weight 或
    sample_limit，训练循环后续的进度校验会给出明确错误，不会静默错位。
    """
    if not isinstance(data_progress, dict):
        return {}
    entries = data_progress.get("entries")
    if not isinstance(entries, list) or not entries:
        return {}
    progress_by_key = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("entry_key"), str):
            progress_by_key[entry["entry_key"]] = entry
    if not progress_by_key:
        return {}

    target_epochs = max(1, int(target_epochs))
    manifest_path, dataset_entries = load_manifest_entries(manifest_path)
    cache_root = Path(cache_dir)
    result = {}
    for entry in dataset_entries:
        if not isinstance(entry, dict) or entry.get("enabled", True) is False:
            continue
        dataset_path = entry.get("path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            continue
        resolved_path = resolve_manifest_dataset_path(manifest_path, dataset_path)
        entry_name = entry.get("name") or resolved_path.stem
        entry_key = entry_progress_key(entry_name, resolved_path)
        progress = progress_by_key.get(entry_key)
        if progress is None:
            continue
        completed_epochs = int(progress.get("completed_epochs", 0) or 0)
        completed_samples = int(progress.get("total_consumed_samples", 0) or 0)
        target_samples = int(progress.get("target_total_samples", 0) or 0)
        if completed_epochs < target_epochs or completed_samples < target_samples:
            continue
        source_record_count = int(progress.get("source_record_count", -1) or -1)
        if source_record_count < 0:
            continue

        # 源文件条数/内容变动时，cache 指纹会失配；失配则不能安全跳过。
        cache_sub = cache_root / cache_subdir_name(resolved_path)
        if cache_sub.is_dir():
            try:
                cache = TokenCache(cache_sub)
            except Exception:
                cache = None
            if cache is None or not cache.fingerprint_matches([resolved_path]):
                # cache 失配说明源文件已变动，交给正常构建流程重建。
                continue
        else:
            # 没有 cache 时只能对比轻量行数；行数不一致视为数据变动。
            if _count_jsonl_lines(resolved_path) != source_record_count:
                continue
        result[entry_key] = source_record_count
    return result


def _all_manifest_entries_completed(manifest_path, completed_entries):
    """判断 manifest 中所有启用数据集是否都已完成。"""
    if not completed_entries:
        return False
    manifest_path, dataset_entries = load_manifest_entries(manifest_path)
    enabled_keys = []
    for entry in dataset_entries:
        if not isinstance(entry, dict) or entry.get("enabled", True) is False:
            continue
        dataset_path = entry.get("path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            continue
        resolved_path = resolve_manifest_dataset_path(manifest_path, dataset_path)
        enabled_keys.append(
            entry_progress_key(
                entry.get("name") or resolved_path.stem,
                resolved_path,
            )
        )
    return bool(enabled_keys) and all(
        key in completed_entries for key in enabled_keys
    )


def _completed_dataset_paths(manifest_path, completed_entries):
    """把已完成 entry_key 集合映射回 manifest 中的源文件路径。"""
    keys = set(completed_entries)
    manifest_path, dataset_entries = load_manifest_entries(manifest_path)
    paths = []
    for entry in dataset_entries:
        if not isinstance(entry, dict) or entry.get("enabled", True) is False:
            continue
        dataset_path = entry.get("path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            continue
        resolved_path = resolve_manifest_dataset_path(manifest_path, dataset_path)
        entry_key = entry_progress_key(
            entry.get("name") or resolved_path.stem,
            resolved_path,
        )
        if entry_key in keys:
            paths.append(resolved_path)
    return paths


def load_text_pretrain_dataset(
    manifest_path,
    tokenizer,
    args,
    *,
    completed_entries=None,
):
    """优先从离线 token cache 构造训练数据集，失败时回退 JSONL 流式。

    cache 未构建时会先并行构建（一次性成本），之后训练启动直接 mmap，
    不再扫描 2GB 级 JSONL，也避免 ``DataProgressTracker`` 对源文件全量哈希。
    """
    completed_entries = dict(completed_entries or {})
    cache_dir = None
    if getattr(args, "token_cache_enabled", True):
        configured_dir = getattr(args, "token_cache_dir", None)
        if configured_dir is not None:
            try:
                cache_dir = ensure_manifest_token_cache(
                    manifest_path,
                    configured_dir,
                    tokenizer,
                    expected_types={"text"},
                    max_sequence_length=args.train_max_sequence_length,
                    num_workers=args.data_reader_num_workers,
                    chunk_bytes=args.data_reader_chunk_bytes,
                    skip_paths=_completed_dataset_paths(
                        manifest_path,
                        completed_entries,
                    ),
                )
                dataset = TokenCacheManifestDataset(
                    manifest_path=manifest_path,
                    cache_dir=cache_dir,
                    expected_types={"text"},
                    shuffle_buffer_size=GlobalConfig.dataset_shuffle_buffer_size,
                    seed=args.seed,
                    completed_entries=completed_entries,
                    # 运行期截断兼容现有未限长 cache，不触发数 GB cache 重建。
                    max_sequence_length=(
                        args.train_max_sequence_length
                        or GlobalConfig.train_max_sequence_length
                    ),
                )
                print(f"manifest={manifest_path}")
                print(f"token_cache={cache_dir}")
                print(f"datasets={dataset.loaded_datasets}")
                print(f"types={dataset.summary_types}")
                print(f"sources={dataset.summary_sources}")
                return dataset, cache_dir
            except Exception as exc:
                # cache 构建失败（磁盘不足、tokenizer 不可 pickle 等）时
                # 回退到原 JSONL 流式路径，训练不被缓存依赖阻塞。
                cache_dir = None
                print(f"token cache 不可用，回退 JSONL 流式读取: {exc}")
    else:
        print(
            "提示: 已禁用 token cache，大语料下 JSONL 流式读取可能出现启动卡顿；"
            "建议保持默认启用。"
        )
    dataset = load_dataset_from_manifest(
        manifest_path,
        expected_types={"text"},
        seed=args.seed,
        reader_num_workers=args.data_reader_num_workers,
        reader_chunk_bytes=args.data_reader_chunk_bytes,
    )
    return dataset, cache_dir


def build_parser():
    """构造 text_pretrain CLI parser。"""
    parser = ArgumentParser(description="运行 LPT v2 text pretrain 阶段。")
    parser.add_argument("--manifest", type=Path, default=TEXT_PRETRAIN_MANIFEST_PATH, help="text pretrain manifest。")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="验证 manifest。")
    parser.add_argument("--build-cache-only", action="store_true", default=False, help="只构建/校验离线 token cache，跳过模型实例化与训练。")
    add_training_arguments(parser, recipe=TEXT_PRETRAIN_RECIPE)
    add_longrope2_training_arguments(parser, recipe=TEXT_PRETRAIN_RECIPE)
    return parser


def main(argv=None):
    """text_pretrain 命令行入口。"""
    args = build_parser().parse_args(argv)
    train_text_pretrained_model(args.manifest, eval_manifest_path=args.eval_manifest, args=args)
    return 0
