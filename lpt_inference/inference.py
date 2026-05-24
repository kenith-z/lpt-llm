"""LPT v2 chat 推理入口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lpt_config import GenerationConfig, GlobalConfig
from lpt_protocol import (
    TARGET_CHANNEL_ANSWER,
    TARGET_CHANNEL_THINKING,
    THINKING_MODE_OFF,
    THINKING_MODE_ON,
    normalize_thinking_mode,
    render_prompt_from_messages,
    render_prompt_segments_from_messages,
    target_channel_to_id,
    thinking_mode_to_id,
)

from .session import InferenceSession


@dataclass(frozen=True)
class GenerationResult:
    """单条生成结果及 token 统计。"""

    prompt: str
    response: str
    prompt_token_count: int
    generated_token_count: int
    generated_token_ids: tuple[int, ...]
    thinking: str | None = None
    thinking_token_count: int = 0
    thinking_token_ids: tuple[int, ...] = ()
    thinking_mode: str = THINKING_MODE_OFF
    thinking_visibility: str = "hidden"


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


def _normalize_visibility(visibility):
    """规范化 thinking 展示策略。"""
    normalized = "hidden" if visibility is None else str(visibility).strip().lower()
    if normalized not in {"hidden", "visible"}:
        raise ValueError("thinking_visibility 必须是 hidden 或 visible。")
    return normalized


def _resolve_generation_thinking_mode(generation_config):
    """把 auto 解析为本次生成实际使用的 on/off。"""
    mode = normalize_thinking_mode(getattr(generation_config, "thinking_mode", THINKING_MODE_OFF))
    if mode == "auto":
        return THINKING_MODE_ON if int(getattr(generation_config, "max_thinking_tokens", 0) or 0) > 0 else THINKING_MODE_OFF
    return mode


def _tokenize_segments_with_control(segments, tokenizer):
    """把 prompt 片段编码为 token 与原生 thinking 控制 id。"""
    input_ids = []
    thinking_mode_ids = []
    token_channel_ids = []
    for segment in segments:
        encoded = tokenizer(segment.text, add_special_tokens=False)
        segment_ids = encoded["input_ids"]
        if not segment_ids:
            continue
        input_ids.extend(segment_ids)
        thinking_mode_ids.extend([thinking_mode_to_id(segment.thinking_mode)] * len(segment_ids))
        token_channel_ids.extend([target_channel_to_id(segment.target_channel)] * len(segment_ids))
    if not input_ids:
        return [], [], []
    target_channel_ids = list(token_channel_ids[1:]) + [token_channel_ids[-1]]
    return input_ids, thinking_mode_ids, target_channel_ids


def _generate_token_ids(
    session,
    logits,
    *,
    eos_token_id,
    pad_token_id,
    generation_config,
    max_new_tokens,
    thinking_mode_id,
    target_channel_id,
):
    """按指定原生通道生成 token，并返回最新 logits。"""
    generated_ids = []
    current_logits = logits
    for _step in range(max(0, int(max_new_tokens))):
        next_id = _select_next_token(current_logits[0], generated_ids, generation_config)
        if next_id == eos_token_id or next_id == pad_token_id:
            break
        generated_ids.append(next_id)
        with torch.autocast(
            device_type=session.device.type,
            dtype=GlobalConfig.autocast_dtype,
            enabled=_autocast_enabled(session.device),
        ):
            current_logits = session.append(
                next_id,
                thinking_mode_id=thinking_mode_id,
                target_channel_id=target_channel_id,
            )
    return generated_ids, current_logits


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
        actual_thinking_mode = _resolve_generation_thinking_mode(resolved_generation_config)
        thinking_visibility = _normalize_visibility(
            getattr(resolved_generation_config, "thinking_visibility", "hidden")
        )
        prompt = render_prompt_from_messages(
            messages,
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
        )
        prompt_segments = render_prompt_segments_from_messages(
            messages,
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
            thinking_mode=actual_thinking_mode,
            include_thinking=True,
        )
        prompt_ids, prompt_thinking_mode_ids, prompt_target_channel_ids = _tokenize_segments_with_control(
            prompt_segments,
            tokenizer,
        )
        if not prompt_ids:
            raise ValueError("渲染后的 prompt 没有 token。")
        session = InferenceSession(model, request_id=f"{request_id_prefix}-{index}")
        with torch.autocast(
            device_type=session.device.type,
            dtype=GlobalConfig.autocast_dtype,
            enabled=_autocast_enabled(session.device),
        ):
            logits = session.prefill(
                prompt_ids,
                thinking_mode_ids=prompt_thinking_mode_ids,
                target_channel_ids=prompt_target_channel_ids,
            )
        thinking_ids = []
        thinking_text = None
        thinking_mode_id = thinking_mode_to_id(actual_thinking_mode)
        answer_channel_id = target_channel_to_id(TARGET_CHANNEL_ANSWER)
        max_new_tokens = int(resolved_generation_config.max_length)
        if actual_thinking_mode == THINKING_MODE_ON:
            thinking_ids, logits = _generate_token_ids(
                session,
                logits,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=resolved_generation_config,
                max_new_tokens=int(getattr(resolved_generation_config, "max_thinking_tokens", 0) or 0),
                thinking_mode_id=thinking_mode_id,
                target_channel_id=target_channel_to_id(TARGET_CHANNEL_THINKING),
            )
            thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=True)
            if session.token_ids:
                # thinking 结束后需要把“下一 token 是 answer”的控制信号作用到最后一个上下文 token。
                with torch.autocast(
                    device_type=session.device.type,
                    dtype=GlobalConfig.autocast_dtype,
                    enabled=_autocast_enabled(session.device),
                ):
                    logits = session.rebuild_on_switch(
                        last_thinking_mode_id=thinking_mode_id,
                        last_target_channel_id=answer_channel_id,
                    )

        answer_budget = max_new_tokens
        generated_ids, _logits = _generate_token_ids(
            session,
            logits,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            generation_config=resolved_generation_config,
            max_new_tokens=answer_budget,
            thinking_mode_id=thinking_mode_id,
            target_channel_id=answer_channel_id,
        )
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(
            GenerationResult(
                prompt=prompt,
                response=response,
                prompt_token_count=len(prompt_ids),
                generated_token_count=len(thinking_ids) + len(generated_ids),
                generated_token_ids=tuple(generated_ids),
                thinking=thinking_text if thinking_visibility == "visible" else None,
                thinking_token_count=len(thinking_ids) if thinking_visibility == "visible" else 0,
                thinking_token_ids=tuple(thinking_ids) if thinking_visibility == "visible" else (),
                thinking_mode=actual_thinking_mode,
                thinking_visibility=thinking_visibility,
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
        if result.thinking is not None:
            print(f"Thinking> {result.thinking}")
        print(f"Assistant> {result.response}")
        print(
            "tokens "
            f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
        )
        if result.response.strip():
            messages.append({"role": "assistant", "content": result.response})
