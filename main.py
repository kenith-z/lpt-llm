"""统一推理入口。

这个脚本不再兼容转发到某个训练工作流，而是直接按指定模型类型加载：
1. `text_base`：text pretrain 基座模型
2. `chat_sft`：chat 全参数监督微调模型
3. `lora`：text pretrain 基座 + chat LoRA 适配器
"""

from argparse import ArgumentParser

from lpt_inference import display_model_parameter_summary, run_chat_session
from lpt_runtime import add_execution_arguments, build_execution_config
from lpt_workflows.chat_lora import load_chat_lora_model_for_inference
from lpt_workflows.chat_sft import load_chat_sft_model_for_inference
from lpt_workflows.common import build_default_generation_config
from lpt_workflows.text_pretrain import load_text_pretrained_model_for_inference


MODEL_LOADERS = {
    "text_base": load_text_pretrained_model_for_inference,
    "text_pretrain": load_text_pretrained_model_for_inference,
    "chat_sft": load_chat_sft_model_for_inference,
    "lora": load_chat_lora_model_for_inference,
    "chat_lora": load_chat_lora_model_for_inference,
}


def build_argument_parser():
    parser = ArgumentParser(description="统一加载 text base / chat-sft / lora 模型进行推理。")
    parser.add_argument(
        "--model",
        default="chat_sft",
        choices=tuple(MODEL_LOADERS.keys()),
        help="指定要加载的模型类型。",
    )
    parser.add_argument(
        "--lora-base-source",
        choices=("text_pretrain", "chat_sft"),
        default="text_pretrain",
        help="chat_lora 模型使用的基座来源。",
    )
    add_execution_arguments(parser)
    return parser


def main():
    """按模型类型加载推理模型并启动终端对话。"""
    args = build_argument_parser().parse_args()
    model_loader = MODEL_LOADERS[args.model]
    execution_config = build_execution_config(args)
    if args.model in {"lora", "chat_lora"}:
        model, tokenizer = model_loader(
            base_source=args.lora_base_source,
            execution_config=execution_config,
        )
    else:
        model, tokenizer = model_loader(execution_config=execution_config)

    print("当前推理模型：", args.model)
    print("词表大小：", len(tokenizer))
    display_model_parameter_summary(model)
    run_chat_session(
        model=model,
        tokenizer=tokenizer,
        conversations=None,
        config=build_default_generation_config(),
    )


if __name__ == "__main__":
    main()
