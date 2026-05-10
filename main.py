"""LPT v2 统一推理入口。"""

from __future__ import annotations

from argparse import ArgumentParser

from lpt_config import (
    DEFAULT_CHAT_MODEL_SOURCE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_LORA_BASE_SOURCE,
    VALID_LORA_BASE_SOURCES,
)
from lpt_inference import (
    display_model_parameter_summary,
    generate_responses_with_token_counts,
    run_chat_session,
)
from lpt_runtime import add_execution_arguments, build_execution_config
from lpt_workflows.common import (
    add_generation_arguments,
    build_generation_config_from_args,
)
from lpt_workflows.chat_lora import load_chat_lora_model_for_inference
from lpt_workflows.chat_sft import load_chat_sft_model_for_inference
from lpt_workflows.text_pretrain import load_text_pretrained_model_for_inference


def build_parser():
    parser = ArgumentParser(description="运行 LPT v2 推理。")
    parser.add_argument(
        "--model",
        choices=("text_pretrain", "chat_sft", "lora", "chat_lora"),
        default=DEFAULT_CHAT_MODEL_SOURCE,
        help="要加载的训练阶段模型。",
    )
    parser.add_argument(
        "--lora-base-source",
        choices=VALID_LORA_BASE_SOURCES,
        default=DEFAULT_LORA_BASE_SOURCE,
        help="LoRA 推理使用的基座来源。",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="auto/cpu/cuda/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--prompt", default=None, help="非交互模式下的一次性用户输入。")
    parser.add_argument("--single-turn", action="store_true", help="交互模式关闭多轮上下文。")
    add_generation_arguments(parser)
    add_execution_arguments(parser)
    return parser


def _load_model_and_tokenizer(args, execution_config):
    if args.model == "text_pretrain":
        return load_text_pretrained_model_for_inference(
            execution_config,
            device=args.device,
            dtype=args.dtype,
        )
    if args.model == "chat_sft":
        return load_chat_sft_model_for_inference(
            execution_config,
            device=args.device,
            dtype=args.dtype,
        )
    if args.model in {"lora", "chat_lora"}:
        return load_chat_lora_model_for_inference(
            base_source=args.lora_base_source,
            execution_config=execution_config,
            device=args.device,
            dtype=args.dtype,
        )
    raise ValueError(f"未支持的模型类型: {args.model}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    execution_config = build_execution_config(args)
    generation_config = build_generation_config_from_args(args)
    model, tokenizer = _load_model_and_tokenizer(args, execution_config)
    display_model_parameter_summary(model)
    if args.prompt is not None:
        result = generate_responses_with_token_counts(
            model,
            tokenizer,
            [args.prompt],
            generation_config=generation_config,
        )[0]
        print(result.response)
        print(
            "tokens "
            f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
        )
        return 0
    run_chat_session(
        model,
        tokenizer,
        generation_config=generation_config,
        multi_turn=not args.single_turn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
