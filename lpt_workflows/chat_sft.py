"""Chat 全参数监督微调工作流。"""

from argparse import ArgumentParser
from pathlib import Path
import time

import torch

from lpt_config import ChatSFTTrainingConfig, GlobalConfig
from lpt_inference import (
    display_checkpoint_summary,
    display_model_parameter_summary,
    run_chat_session,
)
from lpt_model import LPT
from lpt_runtime import apply_inference_execution_plan
from lpt_training import configure_training_runtime, has_complete_training_state, load_checkpoint, train

from .common import (
    ARTIFACT_ROOT_DIR,
    TOKENIZER_PATH,
    add_longrope2_training_arguments,
    apply_longrope2_model_config_overrides,
    apply_longrope2_runtime_overrides,
    build_default_generation_config,
    build_longrope2_workflow_options,
    build_training_profile_with_longrope2_options,
    load_dataset_from_manifest,
    build_local_tokenizer,
    load_model_config_from_checkpoint_root,
    resolve_artifact_model_config,
    resolve_inference_execution_plan,
    warn_if_longrope2_model_options_ignored,
)
from .text_pretrain import TEXT_PRETRAIN_CHECKPOINT_ROOT


CHAT_SFT_MANIFEST_PATH = Path("data/manifests/chat_sft.json")
CHAT_SFT_ARTIFACT_DIR = ARTIFACT_ROOT_DIR / "chat_sft"
CHAT_SFT_CHECKPOINT_ROOT = CHAT_SFT_ARTIFACT_DIR / "checkpoints" / "latest"
CHAT_SFT_CHECKPOINT_FILE = CHAT_SFT_CHECKPOINT_ROOT.with_suffix(".pth")
CHAT_SFT_WEIGHT_PATH = CHAT_SFT_ARTIFACT_DIR / "weights" / "model_weights.pth"


def resolve_chat_sft_resume_checkpoint():
    """返回 chat SFT 的可恢复 checkpoint。"""
    if has_complete_training_state(
        CHAT_SFT_CHECKPOINT_ROOT,
        training_profile=ChatSFTTrainingConfig,
    ):
        return CHAT_SFT_CHECKPOINT_ROOT
    return None


def resolve_text_pretrain_initial_checkpoint():
    """返回可作为 chat SFT 初始化权重的文本预训练 checkpoint。"""
    if TEXT_PRETRAIN_CHECKPOINT_ROOT.with_suffix(".pth").exists():
        return TEXT_PRETRAIN_CHECKPOINT_ROOT
    return None


def load_chat_sft_model_for_inference(execution_config=None):
    """加载 chat SFT 阶段产出的整模型权重用于推理。"""
    if not CHAT_SFT_WEIGHT_PATH.exists():
        raise FileNotFoundError(
            "未找到 chat_sft 推理权重，请先运行 `python main-sft.py`："
            f" {CHAT_SFT_WEIGHT_PATH}"
        )

    tokenizer = build_local_tokenizer(TOKENIZER_PATH)
    vocabulary_size = len(tokenizer)
    model_config = resolve_artifact_model_config(
        CHAT_SFT_ARTIFACT_DIR,
        CHAT_SFT_CHECKPOINT_ROOT,
    )
    execution_plan = resolve_inference_execution_plan(model_config, execution_config)
    initial_device = "cpu" if execution_plan.is_model_parallel else execution_plan.primary_device
    model = LPT(vocabulary_size=vocabulary_size, config=model_config).to(initial_device)
    model.load_state_dict(
        torch.load(CHAT_SFT_WEIGHT_PATH, map_location=execution_plan.state_dict_map_location)
    )
    apply_inference_execution_plan(model, execution_plan)
    model.eval()
    return model, tokenizer


def train_chat_sft_model(
    manifest_path=CHAT_SFT_MANIFEST_PATH,
    eval_manifest_path=None,
    longrope2_options=None,
):
    """执行 chat 全参数监督微调，并返回可交互模型。"""
    started_at = time.time()
    apply_longrope2_runtime_overrides(longrope2_options)
    training_profile = build_training_profile_with_longrope2_options(
        ChatSFTTrainingConfig,
        longrope2_options,
    )
    configure_training_runtime(
        seed=training_profile.random_seed,
        deterministic_algorithms=training_profile.deterministic_algorithms,
    )
    dataset = load_dataset_from_manifest(
        Path(manifest_path),
        expected_types={"chat"},
        seed=training_profile.random_seed,
    )
    print("训练阶段", "chat_sft")
    print("数据总量", len(dataset))
    eval_dataset = None
    if eval_manifest_path is not None:
        eval_dataset = load_dataset_from_manifest(
            Path(eval_manifest_path),
            expected_types={"chat"},
            seed=training_profile.random_seed,
            shuffle_buffer_size=1,
        )
        print("验证总量", len(eval_dataset))

    tokenizer = build_local_tokenizer(TOKENIZER_PATH)
    vocabulary_size = len(tokenizer)
    print("词表大小：", vocabulary_size)

    resume_checkpoint_path = resolve_chat_sft_resume_checkpoint()
    initial_checkpoint_path = (
        None if resume_checkpoint_path is not None else resolve_text_pretrain_initial_checkpoint()
    )
    model_config = resolve_artifact_model_config(
        CHAT_SFT_ARTIFACT_DIR,
        CHAT_SFT_CHECKPOINT_ROOT,
    )
    if resume_checkpoint_path is not None:
        model_config = load_model_config_from_checkpoint_root(resume_checkpoint_path)
        warn_if_longrope2_model_options_ignored(longrope2_options, "chat_sft 续训状态")
    elif initial_checkpoint_path is not None:
        model_config = load_model_config_from_checkpoint_root(initial_checkpoint_path)
        warn_if_longrope2_model_options_ignored(longrope2_options, "text_pretrain 初始化 checkpoint")
    else:
        model_config = apply_longrope2_model_config_overrides(
            model_config,
            longrope2_options,
        )
    model = LPT(vocabulary_size=vocabulary_size, config=model_config).to(GlobalConfig.device)
    if resume_checkpoint_path is None and initial_checkpoint_path is None:
        print("警告: 未发现 text_pretrain checkpoint 或 chat_sft 续训状态，将从随机初始化模型开始训练。")

    GlobalConfig.training_stage = "chat_sft"
    display_model_parameter_summary(model)
    trained_model = train(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        save_path=str(CHAT_SFT_CHECKPOINT_ROOT),
        manifest_path=manifest_path,
        eval_dataset=eval_dataset,
        eval_manifest_path=eval_manifest_path,
        training_profile=training_profile,
        resume_checkpoint_path=None if resume_checkpoint_path is None else str(resume_checkpoint_path),
        initial_checkpoint_path=None if initial_checkpoint_path is None else str(initial_checkpoint_path),
        inference_weight_path=str(CHAT_SFT_WEIGHT_PATH),
    )

    checkpoint = load_checkpoint(CHAT_SFT_CHECKPOINT_ROOT)
    display_checkpoint_summary(checkpoint)
    trained_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    elapsed_hours = (time.time() - started_at) / 3600
    print(f"耗时: {elapsed_hours:.2f} 小时(hour)")
    return trained_model, tokenizer


def build_argument_parser():
    parser = ArgumentParser(description="运行 chat SFT 阶段。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CHAT_SFT_MANIFEST_PATH,
        help="chat SFT 的多语料 manifest 路径",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="可选的 chat SFT 验证 manifest 路径",
    )
    add_longrope2_training_arguments(parser)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    model, tokenizer = train_chat_sft_model(
        manifest_path=args.manifest,
        eval_manifest_path=args.eval_manifest,
        longrope2_options=build_longrope2_workflow_options(args),
    )
    display_model_parameter_summary(model)
    model.eval()

    model, tokenizer = load_chat_sft_model_for_inference()
    print("词表大小：", len(tokenizer))
    run_chat_session(
        model=model,
        tokenizer=tokenizer,
        conversations=None,
        config=build_default_generation_config(),
    )
