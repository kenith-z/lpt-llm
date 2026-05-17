"""LPT v2 chat 推理入口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lpt_config import GenerationConfig, GlobalConfig
from lpt_protocol import render_prompt_from_messages

from .session import InferenceSession


@dataclass(frozen=True)
class GenerationResult:
    """单条生成结果及 token 统计。"""

    prompt: str
    response: str
    prompt_token_count: int
    generated_token_count: int
    generated_token_ids: tuple[int, ...]


def build_default_generation_config(**overrides):
    """构造默认生成配置，并允许用显式覆盖项做 smoke test。"""
    payload = GenerationConfig().__dict__
    payload.update(overrides)
    return GenerationConfig(**payload)


def _normalize_conversation(conversation):
    """把单轮字符串或多轮 messages 统一成 messages 列表。"""
    if isinstance(conversation, str):
        return [{"role": "user", "content": conversation}]
    if isinstance(conversation, list):
        return conversation
    raise TypeError("conversation 必须是字符串或 messages 列表。")


def _apply_repetition_penalty(logits, generated_ids, generation_config):
    """对最近生成 token 应用 repetition penalty。"""
    penalty = float(generation_config.repetition_penalty or 1.0)
    window_size = generation_config.repetition_window_size
    if penalty == 1.0 or not generated_ids:
        return logits
    recent_ids = generated_ids[-int(window_size):] if window_size else generated_ids
    for token_id in set(recent_ids):
        value = logits[token_id]
        logits[token_id] = value / penalty if value > 0 else value * penalty
    return logits


def _filter_top_k_top_p(logits, generation_config):
    """按 top-k / top-p 过滤采样分布。"""
    filtered = logits
    top_k = int(generation_config.top_k or 0)
    if top_k > 0 and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, -float("inf"))
    top_p = float(generation_config.top_p or 1.0)
    if 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove_mask = cumulative > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False
        sorted_logits = sorted_logits.masked_fill(remove_mask, -float("inf"))
        filtered = torch.full_like(filtered, -float("inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
    return filtered


def _select_next_token(logits, generated_ids, generation_config):
    """根据采样配置选择下一个 token。"""
    next_logits = logits[-1].float().clone()
    next_logits = _apply_repetition_penalty(next_logits, generated_ids, generation_config)
    if not generation_config.do_sample:
        return int(torch.argmax(next_logits).item())
    temperature = max(float(generation_config.temperature or 1.0), 1e-5)
    next_logits = _filter_top_k_top_p(next_logits / temperature, generation_config)
    probabilities = torch.softmax(next_logits, dim=-1)
    if torch.isnan(probabilities).any() or float(probabilities.sum()) <= 0:
        return int(torch.argmax(logits[-1].float()).item())
    return int(torch.multinomial(probabilities, num_samples=1).item())


def _autocast_enabled(device):
    """判断当前设备是否可启用 autocast。"""
    return device.type == "cuda" and GlobalConfig.autocast_dtype in {
        torch.float16,
        torch.bfloat16,
    }


@torch.no_grad()
def generate_responses_with_token_counts(
    model,
    tokenizer,
    conversations,
    *,
    generation_config=None,
    request_id_prefix="chat",
):
    """对一组 conversation 生成回复并返回 token 统计。"""
    resolved_generation_config = generation_config or build_default_generation_config()
    was_training = model.training
    model.eval()
    results = []
    if isinstance(conversations, (str, dict)):
        conversation_list = [conversations]
    else:
        conversation_list = list(conversations)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for index, conversation in enumerate(conversation_list, start=1):
        messages = _normalize_conversation(conversation)
        prompt = render_prompt_from_messages(
            messages,
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("渲染后的 prompt 没有 token。")
        session = InferenceSession(model, request_id=f"{request_id_prefix}-{index}")
        with torch.autocast(
            device_type=session.device.type,
            dtype=GlobalConfig.autocast_dtype,
            enabled=_autocast_enabled(session.device),
        ):
            logits = session.prefill(prompt_ids)
        generated_ids = []
        max_new_tokens = int(resolved_generation_config.max_length)
        for _step in range(max_new_tokens):
            # decode 逐 token 续接，允许 InferenceSession 自己维护 request-bound 状态。
            next_id = _select_next_token(logits[0], generated_ids, resolved_generation_config)
            if next_id == eos_token_id or next_id == pad_token_id:
                break
            generated_ids.append(next_id)
            with torch.autocast(
                device_type=session.device.type,
                dtype=GlobalConfig.autocast_dtype,
                enabled=_autocast_enabled(session.device),
            ):
                logits = session.append(next_id)
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(
            GenerationResult(
                prompt=prompt,
                response=response,
                prompt_token_count=len(prompt_ids),
                generated_token_count=len(generated_ids),
                generated_token_ids=tuple(generated_ids),
            )
        )
    if was_training:
        model.train()
    return results


def run_chat_session(model, tokenizer, *, generation_config=None, multi_turn=True):
    """运行交互式 chat 会话。"""
    messages = []
    print("进入 LPT v2 chat；输入 exit/quit 结束。")
    while True:
        user_text = input("User> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        if not multi_turn:
            messages = []
        messages.append({"role": "user", "content": user_text})
        result = generate_responses_with_token_counts(
            model,
            tokenizer,
            [messages],
            generation_config=generation_config,
        )[0]
        print(f"Assistant> {result.response}")
        print(
            "tokens "
            f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
        )
        if result.response.strip():
            messages.append({"role": "assistant", "content": result.response})
