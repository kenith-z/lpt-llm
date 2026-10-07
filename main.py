"""LPT v2 统一推理入口。"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from lpt_config import (
    DEFAULT_CHAT_MODEL_SOURCE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_LORA_BASE_SOURCE,
    VALID_LORA_BASE_SOURCES,
)
from lpt_inference import (
    StreamConsolePrinter,
    display_model_parameter_summary,
    generate_responses_with_token_counts,
    run_chat_session,
    stream_generate_response_events,
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
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=True, help="启用流式逐字输出，默认启用。")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false", help="关闭流式输出，等待完整回复后一次性打印。")
    parser.add_argument("--tools-json", default=None, help="Function Call 工具定义 JSON 字符串或 JSON 文件路径。")
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


def _default_generation_thinking_options(model_name):
    """按模型阶段返回推理 thinking 默认策略。"""
    if model_name in {"chat_sft", "lora", "chat_lora"}:
        return "on", "visible"
    return "off", "hidden"


def _uses_chat_protocol(model_name):
    """判断推理模型是否使用 chat 消息协议。"""
    return model_name in {"chat_sft", "lora", "chat_lora"}


def _load_tools_json(raw_value):
    """加载 Function Call 工具定义。"""
    if raw_value is None:
        return None
    candidate_path = Path(raw_value)
    if candidate_path.is_file():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(raw_value)


def _write_text_by_char(text):
    """逐字符刷新控制台输出，避免 tokenizer delta 过大时交互不顺滑。"""
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()


def _print_single_result(result):
    """打印一次性推理结果。"""
    if result.thinking is not None:
        print("Thinking> ", end="", flush=True)
        _write_text_by_char(result.thinking)
        print()
    print("Assistant> ", end="", flush=True)
    _write_text_by_char(result.response)
    print()
    if result.tool_calls:
        print("tool_calls=" + json.dumps(list(result.tool_calls), ensure_ascii=False))
    if result.structured_output_error:
        print(f"structured_output_error={result.structured_output_error}")
    print(
        "tokens "
        f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    execution_config = build_execution_config(args)
    default_thinking_mode, default_thinking_visibility = _default_generation_thinking_options(
        args.model
    )
    generation_config = build_generation_config_from_args(
        args,
        default_thinking_mode=default_thinking_mode,
        default_thinking_visibility=default_thinking_visibility,
    )
    tools = _load_tools_json(args.tools_json)
    add_default_system = _uses_chat_protocol(args.model)
    model, tokenizer = _load_model_and_tokenizer(args, execution_config)
    display_model_parameter_summary(model)
    if args.prompt is not None:
        if args.stream:
            result = None
            printer = StreamConsolePrinter()
            for event in stream_generate_response_events(
                model,
                tokenizer,
                args.prompt,
                generation_config=generation_config,
                tools=tools,
                add_default_system=add_default_system,
            ):
                if event.event_type == "done":
                    result = event.result
                else:
                    printer.print_event(event)
            printer.finish()
            if result is not None:
                if result.tool_calls:
                    print("tool_calls=" + json.dumps(list(result.tool_calls), ensure_ascii=False))
                if result.structured_output_error:
                    print(f"structured_output_error={result.structured_output_error}")
                print(
                    "tokens "
                    f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
                )
            return 0
        result = generate_responses_with_token_counts(
            model,
            tokenizer,
            [args.prompt],
            generation_config=generation_config,
            tools=tools,
            add_default_system=add_default_system,
        )[0]
        _print_single_result(result)
        return 0
    run_chat_session(
        model,
        tokenizer,
        generation_config=generation_config,
        multi_turn=not args.single_turn,
        stream=args.stream,
        tools=tools,
        add_default_system=add_default_system,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
