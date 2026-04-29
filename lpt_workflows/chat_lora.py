"""Chat LoRA 微调工作流。"""

from argparse import ArgumentParser
from pathlib import Path
import time

import torch

from lpt_config import (
    ChatLoRATrainingConfig,
    GlobalConfig,
    LoRAConfig,
    ModelConfig,
)
from lpt_inference import (
    display_checkpoint_summary,
    display_model_parameter_summary,
    run_chat_session,
)
from lpt_model import LPT, list_architecture_mismatches
from lpt_runtime import apply_inference_execution_plan
from lpt_training import configure_training_runtime, has_complete_training_state, load_checkpoint, train

from lpt_lora.adapter import attach_lora_adapters

from .common import (
    ARTIFACT_ROOT_DIR,
    TOKENIZER_PATH,
    add_longrope2_training_arguments,
    apply_longrope2_model_config_overrides,
    apply_longrope2_runtime_overrides,
    build_default_generation_config,
    build_longrope2_workflow_options,
    load_dataset_from_manifest,
    build_local_tokenizer,
    build_training_profile_with_longrope2_options,
    load_model_config_from_checkpoint_root,
    resolve_inference_execution_plan,
    warn_if_longrope2_model_options_ignored,
)
from .chat_sft import CHAT_SFT_CHECKPOINT_ROOT
from .text_pretrain import TEXT_PRETRAIN_CHECKPOINT_ROOT


CHAT_LORA_MANIFEST_PATH = Path("data/manifests/chat_lora.json")
VALID_LORA_BASE_SOURCES = frozenset({"text_pretrain", "chat_sft"})


def _format_architecture_mismatches(mismatches):
    return "; ".join(
        f"{key}: checkpoint={checkpoint_value}, current={current_value}"
        for key, checkpoint_value, current_value in mismatches
    )


def _resolve_chat_lora_artifact_dir(base_source):
    if base_source not in VALID_LORA_BASE_SOURCES:
        raise ValueError(f"不支持的 LoRA 基座来源: {base_source}")
    return ARTIFACT_ROOT_DIR / "chat_lora" / f"from_{base_source}"


def _resolve_chat_lora_checkpoint_root(base_source):
    return _resolve_chat_lora_artifact_dir(base_source) / "checkpoints" / "latest"


def _resolve_chat_lora_adapter_path(base_source):
    return _resolve_chat_lora_artifact_dir(base_source) / "weights" / "adapter_weights.pth"


def resolve_lora_base_initial_checkpoint(base_source):
    """返回可作为 chat LoRA 初始化权重的基座 checkpoint。"""
    if base_source == "text_pretrain" and TEXT_PRETRAIN_CHECKPOINT_ROOT.with_suffix(".pth").exists():
        return TEXT_PRETRAIN_CHECKPOINT_ROOT
    if base_source == "chat_sft" and CHAT_SFT_CHECKPOINT_ROOT.with_suffix(".pth").exists():
        return CHAT_SFT_CHECKPOINT_ROOT
    return None


def resolve_chat_lora_resume_checkpoint(base_source):
    """返回可用于 chat LoRA 续训的 latest checkpoint。"""
    checkpoint_root = _resolve_chat_lora_checkpoint_root(base_source)
    if has_complete_training_state(
        checkpoint_root,
        training_profile=ChatLoRATrainingConfig,
    ):
        return checkpoint_root
    return None


def load_chat_lora_model_for_inference(base_source="text_pretrain", execution_config=None):
    """加载指定基座 + LoRA 适配器用于推理。"""
    base_checkpoint_root = resolve_lora_base_initial_checkpoint(base_source)
    adapter_path = _resolve_chat_lora_adapter_path(base_source)
    if base_checkpoint_root is None:
        raise FileNotFoundError(
            f"未找到 {base_source} checkpoint，无法加载 LoRA 推理模型。"
        )
    if not adapter_path.exists():
        raise FileNotFoundError(
            "未找到 chat_lora 适配器权重，请先运行 `python main-lora.py`："
            f" {adapter_path}"
        )

    tokenizer = build_local_tokenizer(TOKENIZER_PATH)
    vocabulary_size = len(tokenizer)
    model_config = load_model_config_from_checkpoint_root(base_checkpoint_root, map_location="cpu")
    execution_plan = resolve_inference_execution_plan(model_config, execution_config)
    initial_device = "cpu" if execution_plan.is_model_parallel else execution_plan.primary_device
    model = LPT(vocabulary_size=vocabulary_size, config=model_config).to(initial_device)
    attach_lora_adapters(model, config=LoRAConfig())

    base_checkpoint = load_checkpoint(
        base_checkpoint_root,
        map_location=execution_plan.state_dict_map_location,
    )
    mismatches = list_architecture_mismatches(base_checkpoint, model)
    if mismatches:
        raise ValueError(
            f"{base_source} checkpoint 与当前 LoRA 模型架构不兼容: "
            f"{_format_architecture_mismatches(mismatches)}"
        )

    model.load_state_dict(base_checkpoint["model_state_dict"], strict=False)
    adapter_checkpoint = torch.load(adapter_path, map_location=execution_plan.state_dict_map_location)
    model.load_state_dict(adapter_checkpoint, strict=False)
    apply_inference_execution_plan(model, execution_plan)
    model.eval()
    return model, tokenizer


def finetune_chat_with_lora(
    manifest_path=CHAT_LORA_MANIFEST_PATH,
    *,
    eval_manifest_path=None,
    base_source="text_pretrain",
    longrope2_options=None,
):
    """加载指定基座模型，挂载 LoRA 层并完成 chat 微调。"""
    started_at = time.time()
    apply_longrope2_runtime_overrides(longrope2_options)
    training_profile = build_training_profile_with_longrope2_options(
        ChatLoRATrainingConfig,
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
    print("训练阶段", "chat_lora")
    print("LoRA 基座", base_source)
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

    GlobalConfig.lora_mode = True
    GlobalConfig.training_stage = "chat_lora"
    try:
        resume_checkpoint_path = resolve_chat_lora_resume_checkpoint(base_source)
        initial_checkpoint_path = (
            None if resume_checkpoint_path is not None else resolve_lora_base_initial_checkpoint(base_source)
        )
        if resume_checkpoint_path is not None:
            model_config = load_model_config_from_checkpoint_root(resume_checkpoint_path)
            warn_if_longrope2_model_options_ignored(longrope2_options, "chat_lora 续训状态")
        elif initial_checkpoint_path is not None:
            model_config = load_model_config_from_checkpoint_root(initial_checkpoint_path)
            warn_if_longrope2_model_options_ignored(longrope2_options, f"{base_source} 初始化 checkpoint")
        else:
            model_config = ModelConfig()
            model_config = apply_longrope2_model_config_overrides(
                model_config,
                longrope2_options,
            )
        model = LPT(vocabulary_size=vocabulary_size, config=model_config).to(GlobalConfig.device)
        attach_lora_adapters(model, config=LoRAConfig())
        if resume_checkpoint_path is None and initial_checkpoint_path is None:
            print(
                f"警告: 未发现 {base_source} checkpoint 或 chat_lora 续训状态，"
                "将从随机初始化模型开始训练。"
            )

        trained_model = train(
            model=model,
            dataset=dataset,
            tokenizer=tokenizer,
            save_path=str(_resolve_chat_lora_checkpoint_root(base_source)),
            manifest_path=manifest_path,
            eval_dataset=eval_dataset,
            eval_manifest_path=eval_manifest_path,
            training_profile=training_profile,
            resume_checkpoint_path=None if resume_checkpoint_path is None else str(resume_checkpoint_path),
            initial_checkpoint_path=None if initial_checkpoint_path is None else str(initial_checkpoint_path),
            inference_weight_path=str(_resolve_chat_lora_adapter_path(base_source)),
        )

        deployment_base_checkpoint = resolve_lora_base_initial_checkpoint(base_source)
        adapter_path = _resolve_chat_lora_adapter_path(base_source)
        if deployment_base_checkpoint is not None and adapter_path.exists():
            base_checkpoint = load_checkpoint(deployment_base_checkpoint)
            mismatches = list_architecture_mismatches(base_checkpoint, trained_model)
            if mismatches:
                print(
                    f"检测到 {base_source} checkpoint 与当前架构不兼容，"
                    f"跳过独立叠加验证: {_format_architecture_mismatches(mismatches)}"
                )
            else:
                display_checkpoint_summary(base_checkpoint)
                trained_model.load_state_dict(base_checkpoint["model_state_dict"], strict=False)

                adapter_checkpoint = torch.load(adapter_path, map_location=GlobalConfig.device)
                trained_model.load_state_dict(adapter_checkpoint, strict=False)
        else:
            print(
                f"未找到完整的 {base_source} checkpoint 或 chat_lora 推理权重，"
                "直接使用当前内存中的模型进行推理。"
            )
    finally:
        GlobalConfig.lora_mode = False
        GlobalConfig.training_stage = "chat_sft"

    elapsed_hours = (time.time() - started_at) / 3600
    print(f"耗时: {elapsed_hours:.2f} 小时(hour)")
    return trained_model, tokenizer


def build_argument_parser():
    parser = ArgumentParser(description="运行 chat LoRA 阶段。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CHAT_LORA_MANIFEST_PATH,
        help="chat LoRA 的多语料 manifest 路径",
    )
    parser.add_argument(
        "--base-source",
        choices=sorted(VALID_LORA_BASE_SOURCES),
        default="text_pretrain",
        help="LoRA 微调使用的基座来源",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="可选的 chat LoRA 验证 manifest 路径",
    )
    add_longrope2_training_arguments(parser)
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    model, tokenizer = finetune_chat_with_lora(
        manifest_path=args.manifest,
        eval_manifest_path=args.eval_manifest,
        base_source=args.base_source,
        longrope2_options=build_longrope2_workflow_options(args),
    )
    display_model_parameter_summary(model)
    model.eval()
    model, tokenizer = load_chat_lora_model_for_inference(base_source=args.base_source)
    conversations = [
        [{"role": "user", "content": "你是谁？"}],
        [{"role": "user", "content": "写一段散文，描述一场雨。"}],
    ]
    run_chat_session(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        config=build_default_generation_config(),
    )
