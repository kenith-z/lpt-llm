"""LPT v2 chat LoRA 工作流。"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lpt_config import ChatLoRATrainingConfig, VALID_LORA_BASE_SOURCES
from lpt_inference import display_model_parameter_summary
from lpt_lora import (
    LoRAConfig,
    attach_lora_adapters,
    load_lora_adapter_config,
    load_lora_adapter_state,
)
from lpt_training import TrainingRunConfig, resolve_latest_training_checkpoint, train

from .chat_sft import (
    resolve_chat_sft_checkpoint_root,
    resolve_chat_sft_inference_weight_path,
)
from .common import (
    add_longrope2_training_arguments,
    add_training_arguments,
    build_local_tokenizer,
    build_longrope2_options_from_args,
    build_tokenizer_metadata,
    find_existing_model_checkpoint,
    load_checkpoint_model,
    load_dataset_from_manifest,
    load_eval_dataset,
    merge_training_args_with_recipe,
    apply_longrope2_runtime_overrides,
)
from .text_pretrain import (
    resolve_text_pretrain_checkpoint_root,
    resolve_text_pretrain_inference_weight_path,
)


CHAT_LORA_RECIPE = ChatLoRATrainingConfig()
CHAT_LORA_MANIFEST_PATH = CHAT_LORA_RECIPE.manifest_path


def _resolve_chat_lora_artifact_dir(base_source):
    """按基座来源隔离 LoRA 训练产物目录。"""
    if base_source not in VALID_LORA_BASE_SOURCES:
        raise ValueError(f"不支持的 LoRA 基座来源: {base_source}")
    return CHAT_LORA_RECIPE.artifact_dir / f"from_{base_source}"


def _resolve_chat_lora_checkpoint_root(base_source):
    """返回指定基座来源的 LoRA latest checkpoint 根目录。"""
    return _resolve_chat_lora_artifact_dir(base_source) / "checkpoints" / "latest"


def _resolve_chat_lora_adapter_path(base_source):
    """返回指定基座来源的 LoRA adapter 推理权重路径。"""
    return _resolve_chat_lora_artifact_dir(base_source) / "weights" / "adapter_weights.pth"


def resolve_lora_base_initial_checkpoint(base_source):
    """返回 LoRA 基座 checkpoint。"""
    if base_source == "text_pretrain":
        return find_existing_model_checkpoint(
            resolve_text_pretrain_inference_weight_path().with_name("model_checkpoint.pt"),
            resolve_text_pretrain_checkpoint_root(),
        )
    if base_source == "chat_sft":
        return find_existing_model_checkpoint(
            resolve_chat_sft_inference_weight_path().with_name("model_checkpoint.pt"),
            resolve_chat_sft_checkpoint_root(),
        )
    raise ValueError(f"不支持的 LoRA 基座来源: {base_source}")


def resolve_chat_lora_resume_checkpoint(base_source):
    """查找指定基座来源的 LoRA 可续训 checkpoint。"""
    checkpoint_root = _resolve_chat_lora_checkpoint_root(base_source)
    return resolve_latest_training_checkpoint(checkpoint_root, lora_mode=True)


def load_chat_lora_model_for_inference(base_source="text_pretrain", execution_config=None, *, device="auto", dtype="auto"):
    """加载基座 + LoRA adapter 用于推理。"""
    tokenizer = build_local_tokenizer()
    base_checkpoint = resolve_lora_base_initial_checkpoint(base_source)
    if base_checkpoint is None:
        raise FileNotFoundError(f"未找到 {base_source} 基座 checkpoint，无法加载 LoRA。")
    adapter_path = _resolve_chat_lora_adapter_path(base_source)
    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 LoRA adapter 权重: {adapter_path}")
    model = load_checkpoint_model(base_checkpoint, device=device, dtype=dtype, strict=True)
    attach_lora_adapters(model, load_lora_adapter_config(adapter_path))
    load_lora_adapter_state(model, adapter_path, strict=True)
    from .common import apply_execution_plan_for_inference

    model = apply_execution_plan_for_inference(model, execution_config)
    return model, tokenizer


def finetune_chat_with_lora(
    manifest_path=CHAT_LORA_MANIFEST_PATH,
    *,
    base_source=None,
    eval_manifest_path=None,
    args=None,
):
    """执行 chat LoRA 微调。"""
    args = merge_training_args_with_recipe(args, CHAT_LORA_RECIPE)
    base_source = base_source or CHAT_LORA_RECIPE.lora_base_source
    apply_longrope2_runtime_overrides(build_longrope2_options_from_args(args))
    tokenizer = build_local_tokenizer()
    dataset = load_dataset_from_manifest(
        manifest_path,
        expected_types={"chat"},
        seed=args.seed,
    )
    eval_dataset = load_eval_dataset(eval_manifest_path, expected_types={"chat"})
    base_checkpoint = resolve_lora_base_initial_checkpoint(base_source)
    if base_checkpoint is None:
        raise FileNotFoundError(f"未找到 {base_source} 基座 checkpoint，无法训练 LoRA。")
    model = load_checkpoint_model(base_checkpoint, device=args.device, dtype=args.dtype, strict=True)
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout_p=args.lora_dropout,
        target_modules=tuple(
            module_name.strip()
            for module_name in args.lora_target_modules.split(",")
            if module_name.strip()
        ),
    )
    resume_checkpoint = None if args.no_resume else resolve_chat_lora_resume_checkpoint(base_source)
    if resume_checkpoint is not None:
        adapter_path = resume_checkpoint / "adapter.pt"
        if adapter_path.exists():
            # resume 时以 adapter checkpoint 内保存的 LoRAConfig 为准，避免 CLI 覆盖破坏权重形状。
            lora_config = load_lora_adapter_config(adapter_path)
    attach_lora_adapters(model, lora_config)
    if resume_checkpoint is not None:
        adapter_path = resume_checkpoint / "adapter.pt"
        if adapter_path.exists():
            load_lora_adapter_state(model, adapter_path, strict=True)
    display_model_parameter_summary(model)
    print(f"LoRA 基座={base_source}")
    print(f"LoRA 配置={lora_config.to_dict()}")
    print(f"tokenizer={build_tokenizer_metadata(tokenizer)}")
    trainer_state = train(
        model,
        tokenizer,
        dataset,
        config=TrainingRunConfig(
            training_stage="chat_lora",
            artifact_dir=_resolve_chat_lora_artifact_dir(base_source),
            checkpoint_dir=_resolve_chat_lora_checkpoint_root(base_source),
            inference_weight_path=_resolve_chat_lora_adapter_path(base_source),
            save_inference_weights=not args.no_save_inference_weights,
            batch_size=args.batch_size,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
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
            thinking_mode=args.thinking_mode,
            thinking_visibility=args.thinking_visibility,
            seed=args.seed,
            deterministic_algorithms=args.deterministic_algorithms,
            resume_checkpoint=resume_checkpoint,
            initial_checkpoint=base_checkpoint,
            source_manifest=manifest_path,
            eval_manifest=eval_manifest_path,
            lora_mode=True,
            longrope2_window_lengths=args.longrope2_window_lengths,
            longrope2_window_weights=args.longrope2_window_weights,
            tokenizer_metadata=build_tokenizer_metadata(tokenizer),
            run_id=args.run_id,
            tensorboard_enabled=not args.no_tensorboard,
        ),
        eval_dataset=eval_dataset,
    )
    return model, tokenizer, trainer_state


def build_parser():
    """构造 chat_lora CLI parser。"""
    parser = ArgumentParser(description="运行 LPT v2 chat LoRA 阶段。")
    parser.add_argument("--manifest", type=Path, default=CHAT_LORA_MANIFEST_PATH, help="chat LoRA manifest。")
    parser.add_argument("--base-source", choices=sorted(VALID_LORA_BASE_SOURCES), default=CHAT_LORA_RECIPE.lora_base_source, help="LoRA 基座来源。")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="验证 manifest。")
    parser.add_argument("--lora-rank", type=int, default=CHAT_LORA_RECIPE.lora_rank, help="LoRA rank。")
    parser.add_argument("--lora-alpha", type=float, default=CHAT_LORA_RECIPE.lora_alpha, help="LoRA alpha。")
    parser.add_argument("--lora-dropout", type=float, default=CHAT_LORA_RECIPE.lora_dropout, help="LoRA dropout。")
    parser.add_argument("--lora-target-modules", default=",".join(CHAT_LORA_RECIPE.lora_target_modules), help="逗号分隔的投影层名。")
    add_training_arguments(parser, recipe=CHAT_LORA_RECIPE)
    add_longrope2_training_arguments(parser, recipe=CHAT_LORA_RECIPE)
    return parser


def main(argv=None):
    """chat_lora 命令行入口。"""
    args = build_parser().parse_args(argv)
    finetune_chat_with_lora(
        args.manifest,
        base_source=args.base_source,
        eval_manifest_path=args.eval_manifest,
        args=args,
    )
    return 0
