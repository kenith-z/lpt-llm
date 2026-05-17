"""LPT v2 文本预训练工作流。"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lpt_config import TextPretrainingConfig
from lpt_inference import display_model_parameter_summary
from lpt_model import LPTV2
from lpt_training import TrainingRunConfig, resolve_latest_training_checkpoint, train

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
    dataset = load_dataset_from_manifest(
        manifest_path,
        expected_types={"text"},
        seed=args.seed,
    )
    eval_dataset = load_eval_dataset(eval_manifest_path, expected_types={"text"})
    resume_checkpoint = None if args.no_resume else resolve_text_pretrain_resume_checkpoint()
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


def build_parser():
    """构造 text_pretrain CLI parser。"""
    parser = ArgumentParser(description="运行 LPT v2 text pretrain 阶段。")
    parser.add_argument("--manifest", type=Path, default=TEXT_PRETRAIN_MANIFEST_PATH, help="text pretrain manifest。")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="验证 manifest。")
    add_training_arguments(parser, recipe=TEXT_PRETRAIN_RECIPE)
    add_longrope2_training_arguments(parser, recipe=TEXT_PRETRAIN_RECIPE)
    return parser


def main(argv=None):
    """text_pretrain 命令行入口。"""
    args = build_parser().parse_args(argv)
    train_text_pretrained_model(args.manifest, eval_manifest_path=args.eval_manifest, args=args)
    return 0
