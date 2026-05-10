"""LPT v2 chat SFT 工作流。"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from lpt_config import ChatSFTTrainingConfig
from lpt_inference import display_model_parameter_summary
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
from .text_pretrain import (
    resolve_text_pretrain_checkpoint_root,
    resolve_text_pretrain_inference_weight_path,
)


CHAT_SFT_RECIPE = ChatSFTTrainingConfig()
CHAT_SFT_MANIFEST_PATH = CHAT_SFT_RECIPE.manifest_path
CHAT_SFT_ARTIFACT_DIR = CHAT_SFT_RECIPE.artifact_dir


def resolve_chat_sft_checkpoint_root():
    return CHAT_SFT_ARTIFACT_DIR / "checkpoints" / "latest"


def resolve_chat_sft_inference_weight_path():
    return CHAT_SFT_ARTIFACT_DIR / "weights" / "model_weights.pth"


def resolve_chat_sft_resume_checkpoint():
    checkpoint_root = resolve_chat_sft_checkpoint_root()
    return resolve_latest_training_checkpoint(checkpoint_root, lora_mode=False)


def resolve_text_pretrain_initial_checkpoint():
    """返回可作为 SFT 初始化的 text_pretrain checkpoint。"""
    return find_existing_model_checkpoint(
        resolve_text_pretrain_inference_weight_path().with_name("model_checkpoint.pt"),
        resolve_text_pretrain_checkpoint_root(),
    )


def load_chat_sft_model_for_inference(execution_config=None, *, device="auto", dtype="auto"):
    """加载 chat_sft 模型用于推理。"""
    tokenizer = build_local_tokenizer()
    checkpoint_path = find_existing_model_checkpoint(
        resolve_chat_sft_inference_weight_path().with_name("model_checkpoint.pt"),
        resolve_chat_sft_checkpoint_root(),
    )
    if checkpoint_path is not None:
        model = load_checkpoint_model(checkpoint_path, device=device, dtype=dtype, strict=True)
    else:
        raise FileNotFoundError("未找到 chat_sft 推理 checkpoint，请先运行 main-sft.py。")
    from .common import apply_execution_plan_for_inference

    model = apply_execution_plan_for_inference(model, execution_config)
    return model, tokenizer


def _load_or_initialize_model(args, tokenizer):
    resume_checkpoint = None if args.no_resume else resolve_chat_sft_resume_checkpoint()
    if resume_checkpoint is not None:
        checkpoint_file = resume_checkpoint / "model.pt"
        return load_checkpoint_model(checkpoint_file, device=args.device, dtype=args.dtype, strict=True), resume_checkpoint

    initial_checkpoint = resolve_text_pretrain_initial_checkpoint()
    if initial_checkpoint is not None:
        return load_checkpoint_model(initial_checkpoint, device=args.device, dtype=args.dtype, strict=True), None

    print("警告: 未找到 text_pretrain checkpoint，chat_sft 将从随机初始化开始。")
    config = build_workflow_model_config(args)
    return instantiate_model(len(tokenizer), config, device=args.device, dtype=args.dtype), None


def train_chat_sft_model(
    manifest_path=CHAT_SFT_MANIFEST_PATH,
    *,
    eval_manifest_path=None,
    args=None,
):
    """执行 chat SFT。"""
    args = merge_training_args_with_recipe(args, CHAT_SFT_RECIPE)
    apply_longrope2_runtime_overrides(build_longrope2_options_from_args(args))
    tokenizer = build_local_tokenizer()
    dataset = load_dataset_from_manifest(
        manifest_path,
        expected_types={"chat"},
        seed=args.seed,
    )
    eval_dataset = load_eval_dataset(eval_manifest_path, expected_types={"chat"})
    model, resume_checkpoint = _load_or_initialize_model(args, tokenizer)
    display_model_parameter_summary(model)
    print(f"tokenizer={build_tokenizer_metadata(tokenizer)}")
    trainer_state = train(
        model,
        tokenizer,
        dataset,
        config=TrainingRunConfig(
            training_stage="chat_sft",
            artifact_dir=CHAT_SFT_ARTIFACT_DIR,
            checkpoint_dir=resolve_chat_sft_checkpoint_root(),
            inference_weight_path=resolve_chat_sft_inference_weight_path(),
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
            initial_checkpoint=resolve_text_pretrain_initial_checkpoint(),
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
    parser = ArgumentParser(description="运行 LPT v2 chat SFT 阶段。")
    parser.add_argument("--manifest", type=Path, default=CHAT_SFT_MANIFEST_PATH, help="chat SFT manifest。")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="验证 manifest。")
    add_training_arguments(parser, recipe=CHAT_SFT_RECIPE)
    add_longrope2_training_arguments(parser, recipe=CHAT_SFT_RECIPE)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    train_chat_sft_model(args.manifest, eval_manifest_path=args.eval_manifest, args=args)
    return 0
