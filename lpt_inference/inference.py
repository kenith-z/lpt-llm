"""推理与命令行对话模块"""

from dataclasses import dataclass

import torch

from lpt_config import GlobalConfig
from lpt_protocol import render_prompt_from_messages
from .visualization import render_token_position_table


@dataclass(frozen=True)
class GenerationResult:
    """单条推理结果及其 token 统计。"""

    text: str
    input_token_count: int
    output_token_count: int


def _normalize_conversations(conversations):
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations 必须是非空列表。")

    first_item = conversations[0]
    if isinstance(first_item, dict):
        return [conversations]

    if isinstance(first_item, list):
        return conversations

    raise TypeError("conversations 必须是消息列表，或消息列表组成的批次。")


def _encode_generation_inputs(tokenizer, conversations):
    """把单条或多条结构化对话编码成推理输入。"""
    normalized_conversations = _normalize_conversations(conversations)
    prompts = [
        render_prompt_from_messages(
            messages,
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
        )
        for messages in normalized_conversations
    ]
    return tokenizer(
        prompts,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        return_attention_mask=True,
    )


def count_text_tokens(tokenizer, text):
    """统计一段文本按当前 tokenizer 编码后的 token 数。"""
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def _trim_generated_ids(sequence_ids, eos_token_id, pad_token_id=None):
    trimmed_ids = list(sequence_ids)
    if eos_token_id is not None and eos_token_id in trimmed_ids:
        eos_index = trimmed_ids.index(eos_token_id)
        trimmed_ids = trimmed_ids[:eos_index]

    if pad_token_id is not None and pad_token_id != eos_token_id:
        while trimmed_ids and trimmed_ids[-1] == pad_token_id:
            trimmed_ids.pop()

    return trimmed_ids


def _build_generation_results(tokenizer, token_ids, prompt_width, input_token_counts):
    results = []
    for row in token_ids:
        generated_ids = row[prompt_width:].tolist()
        trimmed_ids = _trim_generated_ids(
            generated_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        results.append(trimmed_ids)

    return [
        GenerationResult(
            text=tokenizer.decode(output_ids, skip_special_tokens=False).strip(),
            input_token_count=int(input_token_count),
            output_token_count=len(output_ids),
        )
        for output_ids, input_token_count in zip(results, input_token_counts)
    ]


def _print_batch_outputs(results):
    """按批次格式打印多条生成结果。"""
    rendered_lines = []
    for index, result in enumerate(results, start=1):
        rendered_lines.append(
            "\n".join(
                [
                    f"{GlobalConfig.model_abbr} #{index}:",
                    f" {result.text}",
                    f" 输入 token 数: {result.input_token_count}",
                    f" 输出 token 数: {result.output_token_count}",
                ]
            )
        )
    print("\n".join(rendered_lines))


def generate_responses_with_token_counts(model, tokenizer, conversations, config=None):
    """基于结构化对话生成回复，并返回输入/输出 token 统计。"""
    encoded_batch = _encode_generation_inputs(tokenizer, conversations)
    input_token_counts = encoded_batch["attention_mask"].sum(dim=1).tolist()
    input_ids = encoded_batch["input_ids"].to(GlobalConfig.device)
    attention_mask = encoded_batch["attention_mask"].to(GlobalConfig.device)

    with torch.no_grad():
        with torch.autocast(device_type=GlobalConfig.device.type, dtype=GlobalConfig.autocast_dtype):
            token_ids = model.generate(
                prompt_tokens=input_ids,
                config=config,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    if getattr(GlobalConfig, "attention_plot_enabled", False):
        render_token_position_table(tokenizer, token_ids[0])

    return _build_generation_results(
        tokenizer,
        token_ids,
        prompt_width=input_ids.size(1),
        input_token_counts=input_token_counts,
    )


def generate_responses(model, tokenizer, conversations, config=None):
    """基于结构化对话生成回复文本。"""
    return [
        result.text
        for result in generate_responses_with_token_counts(
            model,
            tokenizer,
            conversations=conversations,
            config=config,
        )
    ]


def run_chat_session(model, tokenizer, conversations=None, multi_turns=False, config=None):
    """运行批量生成或命令行交互式聊天。"""
    if conversations is not None:
        results = generate_responses_with_token_counts(
            model,
            tokenizer,
            conversations=conversations,
            config=config,
        )
        _print_batch_outputs(results)
        return [result.text for result in results]

    conversation_state = []
    while True:
        user_message = input("User: ").strip()
        if user_message.lower() == "quit":
            break
        if not user_message:
            continue

        current_messages = [{"role": "user", "content": user_message}]
        if multi_turns:
            current_messages = conversation_state + current_messages

        results = generate_responses_with_token_counts(
            model,
            tokenizer,
            conversations=current_messages,
            config=config,
        )
        result = results[0]
        reply = result.text
        print(f"{GlobalConfig.model_abbr}: {reply}")
        print(f"输入 token 数: {result.input_token_count} | 输出 token 数: {result.output_token_count}")

        if multi_turns:
            conversation_state = current_messages + [{"role": "assistant", "content": reply}]
