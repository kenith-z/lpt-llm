"""使用 chat SFT 工作流训练 LPT v2 单项实验分支。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import ChatSFTTrainingConfig
from lpt_inference import display_model_parameter_summary
from lpt_training import TrainingRunConfig, resolve_latest_training_checkpoint, train
from lpt_workflows.common import (
    add_longrope2_training_arguments,
    add_training_arguments,
    apply_longrope2_runtime_overrides,
    build_local_tokenizer,
    build_longrope2_options_from_args,
    build_tokenizer_metadata,
    load_checkpoint_model,
    load_dataset_from_manifest,
    load_eval_dataset,
    merge_training_args_with_recipe,
)


DEFAULT_RECIPE = ChatSFTTrainingConfig()


def build_parser():
    parser = argparse.ArgumentParser(description="训练 LPT v2 单项实验分支。")
    parser.add_argument("--init-checkpoint", type=Path, required=True, help="分支初始化 checkpoint。")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="实验分支 artifact 目录。")
    parser.add_argument("--stage", required=True, help="训练阶段/实验分支名。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RECIPE.manifest_path, help="训练 manifest。")
    parser.add_argument("--eval-manifest", type=Path, default=None, help="验证 manifest。")
    add_training_arguments(parser, recipe=DEFAULT_RECIPE)
    add_longrope2_training_arguments(parser, recipe=DEFAULT_RECIPE)
    return parser


def _resolve_branch_resume_checkpoint(artifact_dir):
    checkpoint_root = Path(artifact_dir) / "checkpoints" / "latest"
    return resolve_latest_training_checkpoint(checkpoint_root, lora_mode=False)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args = merge_training_args_with_recipe(args, DEFAULT_RECIPE)
    apply_longrope2_runtime_overrides(build_longrope2_options_from_args(args))

    artifact_dir = Path(args.artifact_dir)
    init_checkpoint = Path(args.init_checkpoint)
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"分支初始化 checkpoint 不存在: {init_checkpoint}")

    tokenizer = build_local_tokenizer()
    dataset = load_dataset_from_manifest(args.manifest, expected_types={"chat"}, seed=args.seed)
    eval_dataset = load_eval_dataset(args.eval_manifest, expected_types={"chat"})

    resume_checkpoint = None if args.no_resume else _resolve_branch_resume_checkpoint(artifact_dir)
    checkpoint_file = init_checkpoint if resume_checkpoint is None else resume_checkpoint / "model.pt"
    model = load_checkpoint_model(checkpoint_file, device=args.device, dtype=args.dtype, strict=True)

    display_model_parameter_summary(model)
    print(f"tokenizer={build_tokenizer_metadata(tokenizer)}")
    trainer_state = train(
        model,
        tokenizer,
        dataset,
        config=TrainingRunConfig(
            training_stage=args.stage,
            artifact_dir=artifact_dir,
            checkpoint_dir=artifact_dir / "checkpoints" / "latest",
            inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
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
            initial_checkpoint=init_checkpoint,
            source_manifest=args.manifest,
            eval_manifest=args.eval_manifest,
            longrope2_window_lengths=args.longrope2_window_lengths,
            longrope2_window_weights=args.longrope2_window_weights,
            tokenizer_metadata=build_tokenizer_metadata(tokenizer),
            run_id=args.run_id or args.stage,
            tensorboard_enabled=not args.no_tensorboard,
        ),
        eval_dataset=eval_dataset,
    )
    print(f"trainer_state_global_step={trainer_state['global_step']}")
    print(f"artifact_dir={artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
