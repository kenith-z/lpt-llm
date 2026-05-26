"""LPT v2 chat 推理入口。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import sys
import threading

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
    normalize_tool_calls,
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
    output_format: str = "text"
    structured_output: object | None = None
    structured_output_valid: bool = False
    structured_output_error: str | None = None
    tool_calls: tuple[dict, ...] = ()


@dataclass(frozen=True)
class GenerationStreamEvent:
    """流式生成事件。"""

    event_type: str
    text: str = ""
    token_id: int | None = None
    channel: str = ""
    result: GenerationResult | None = None


class StreamConsolePrinter:
    """按流式事件通道打印 Thinking/Assistant 标签。"""

    def __init__(self):
        self._current_channel = None
        self._printed_any = False

    def print_event(self, event):
        """打印单个流式事件，并在通道切换时补充标签。"""
        if event.event_type == "thinking_delta":
            self._switch_channel("thinking")
            _write_text_by_char(event.text)
        elif event.event_type == "answer_delta":
            self._switch_channel("answer")
            _write_text_by_char(event.text)

    def finish(self):
        """结束本轮流式输出，保证后续统计信息从新行开始。"""
        if self._printed_any:
            print()

    def _switch_channel(self, channel):
        if self._current_channel == channel:
            return
        if self._printed_any:
            print()
        label = "Thinking> " if channel == "thinking" else "Assistant> "
        print(label, end="", flush=True)
        self._current_channel = channel
        self._printed_any = True


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


def _normalize_output_format(output_format):
    """规范化结构化输出格式。"""
    normalized = "text" if output_format is None else str(output_format).strip().lower()
    if normalized not in {"text", "json", "tool_call"}:
        raise ValueError("output_format 必须是 text/json/tool_call。")
    return normalized


def _normalize_tool_choice(tool_choice):
    """规范化 Function Call 策略。"""
    normalized = "none" if tool_choice is None else str(tool_choice).strip().lower()
    if normalized not in {"none", "auto", "required"}:
        raise ValueError("tool_choice 必须是 none/auto/required。")
    return normalized


def _normalize_available_tools(tools):
    """规范化推理阶段可用工具定义。"""
    if tools is None:
        return ()
    if not isinstance(tools, list):
        raise TypeError("tools 必须是列表。")
    normalized_tools = []
    for index, tool in enumerate(tools, start=1):
        if not isinstance(tool, dict):
            raise TypeError(f"第 {index} 个工具定义必须是字典。")
        # 同时接受项目原生格式和 OpenAI function tool 形态，服务层后续可直接复用。
        tool_payload = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(tool_payload, dict):
            raise TypeError(f"第 {index} 个 function 工具定义必须是字典。")
        name = str(tool_payload.get("name", "")).strip()
        if not name:
            raise ValueError(f"第 {index} 个工具缺少 name。")
        normalized = {"name": name}
        description = str(tool_payload.get("description", "")).strip()
        if description:
            normalized["description"] = description
        parameters = tool_payload.get("parameters", {})
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(f"第 {index} 个工具 parameters 必须是字典。")
        normalized["parameters"] = parameters or {}
        normalized_tools.append(normalized)
    return tuple(normalized_tools)


def _tool_instruction_message(tools, *, tool_choice):
    """把可用工具定义注入为稳定系统提示，配合结构化 Function Call 解析。"""
    normalized_tools = _normalize_available_tools(tools)
    if not normalized_tools:
        return None
    payload = {
        "tools": list(normalized_tools),
        "tool_choice": tool_choice,
        "response_format": {
            "tool_call": {"tool_calls": [{"name": "工具名", "arguments": {}}]},
            "final_answer": "不需要工具时直接回答用户。",
        },
    }
    return {
        "role": "system",
        "content": "可用工具与输出格式如下。需要调用工具时只输出 JSON: "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    }


def _messages_with_tool_context(messages, *, tools=None, tool_choice="none"):
    """按需在 prompt 前加入工具 schema 说明。"""
    instruction = _tool_instruction_message(tools, tool_choice=tool_choice)
    if instruction is None:
        return messages
    return [instruction, *messages]


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


def _mask_to_allowed_tokens(logits, allowed_token_ids):
    """把 logits 限制到允许 token 集合。"""
    if allowed_token_ids is None:
        return logits
    allowed = tuple(int(token_id) for token_id in allowed_token_ids)
    if not allowed:
        raise ValueError("结构化解码没有可用 token。")
    masked = torch.full_like(logits, -float("inf"))
    index = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    masked[index] = logits[index]
    return masked


def _select_next_token(logits, generated_ids, generation_config, *, allowed_token_ids=None):
    """根据采样配置选择下一个 token。"""
    next_logits = logits[-1].float().clone()
    next_logits = _apply_repetition_penalty(next_logits, generated_ids, generation_config)
    next_logits = _mask_to_allowed_tokens(next_logits, allowed_token_ids)
    if not generation_config.do_sample:
        return int(torch.argmax(next_logits).item())
    temperature = max(float(generation_config.temperature or 1.0), 1e-5)
    next_logits = _filter_top_k_top_p(next_logits / temperature, generation_config)
    probabilities = torch.softmax(next_logits, dim=-1)
    if torch.isnan(probabilities).any() or float(probabilities.sum()) <= 0:
        fallback_logits = _mask_to_allowed_tokens(logits[-1].float().clone(), allowed_token_ids)
        return int(torch.argmax(fallback_logits).item())
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


def _is_json_number_complete(text):
    """判断 JSON number 是否完整。"""
    import re

    return re.fullmatch(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?", text) is not None


def _is_json_number_prefix(text):
    """判断文本是否仍可能补全为合法 JSON number。"""
    import re

    return re.fullmatch(
        r"-?(0|[1-9]\d*)?(\.\d*)?([eE][+-]?\d*)?",
        text,
    ) is not None and text not in {"", "+", "."}


def _json_prefix_status(text):
    """返回 JSON object/array 前缀是否合法以及是否已经完整。"""
    stack = []
    root_done = False
    string_context = None
    escape = False
    unicode_remaining = 0
    literal_target = None
    literal_index = 0
    number_buffer = ""
    index = 0

    def complete_value():
        nonlocal root_done
        if not stack:
            root_done = True
            return True
        frame = stack[-1]
        if frame["type"] == "object" and frame["state"] == "value":
            frame["state"] = "comma_or_end"
            return True
        if frame["type"] == "array" and frame["state"] == "value_or_end":
            frame["state"] = "comma_or_end"
            return True
        return False

    def start_value(char):
        if char == "{":
            stack.append({"type": "object", "state": "key_or_end"})
            return True
        if char == "[":
            stack.append({"type": "array", "state": "value_or_end"})
            return True
        return None

    while index < len(text):
        char = text[index]

        if string_context is not None:
            if unicode_remaining:
                if char.lower() not in "0123456789abcdef":
                    return False, False
                unicode_remaining -= 1
            elif escape:
                if char == "u":
                    unicode_remaining = 4
                elif char not in '"\\/bfnrt':
                    return False, False
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                if string_context == "key":
                    if not stack or stack[-1]["type"] != "object":
                        return False, False
                    stack[-1]["state"] = "colon"
                elif not complete_value():
                    return False, False
                string_context = None
            elif ord(char) < 0x20:
                return False, False
            index += 1
            continue

        if literal_target is not None:
            if char != literal_target[literal_index]:
                return False, False
            literal_index += 1
            index += 1
            if literal_index == len(literal_target):
                literal_target = None
                literal_index = 0
                if not complete_value():
                    return False, False
            continue

        if number_buffer:
            if char in "0123456789.eE+-":
                candidate = number_buffer + char
                if not _is_json_number_prefix(candidate):
                    return False, False
                number_buffer = candidate
                index += 1
                continue
            if char.isspace() or char in ",]}":
                if not _is_json_number_complete(number_buffer):
                    return False, False
                number_buffer = ""
                if not complete_value():
                    return False, False
                continue
            return False, False

        if root_done:
            if not char.isspace():
                return False, False
            index += 1
            continue

        state = "root_value" if not stack else stack[-1]["state"]
        if char.isspace():
            index += 1
            continue

        if state == "root_value":
            started = start_value(char)
            if not started:
                return False, False
            index += 1
            continue

        frame = stack[-1]
        if frame["type"] == "object":
            if state == "key_or_end":
                if char == "}":
                    stack.pop()
                    if not complete_value():
                        return False, False
                elif char == '"':
                    string_context = "key"
                else:
                    return False, False
            elif state == "colon":
                if char != ":":
                    return False, False
                frame["state"] = "value"
            elif state == "value":
                started = start_value(char)
                if started:
                    pass
                elif char == '"':
                    string_context = "value"
                elif char in "-0123456789":
                    number_buffer = char
                    if not _is_json_number_prefix(number_buffer):
                        return False, False
                elif char == "t":
                    literal_target, literal_index = "true", 1
                elif char == "f":
                    literal_target, literal_index = "false", 1
                elif char == "n":
                    literal_target, literal_index = "null", 1
                else:
                    return False, False
            elif state == "comma_or_end":
                if char == ",":
                    frame["state"] = "key_or_end"
                elif char == "}":
                    stack.pop()
                    if not complete_value():
                        return False, False
                else:
                    return False, False
            index += 1
            continue

        if frame["type"] == "array":
            if state == "value_or_end":
                if char == "]":
                    stack.pop()
                    if not complete_value():
                        return False, False
                else:
                    started = start_value(char)
                    if started:
                        pass
                    elif char == '"':
                        string_context = "value"
                    elif char in "-0123456789":
                        number_buffer = char
                        if not _is_json_number_prefix(number_buffer):
                            return False, False
                    elif char == "t":
                        literal_target, literal_index = "true", 1
                    elif char == "f":
                        literal_target, literal_index = "false", 1
                    elif char == "n":
                        literal_target, literal_index = "null", 1
                    else:
                        return False, False
            elif state == "comma_or_end":
                if char == ",":
                    frame["state"] = "value_or_end"
                elif char == "]":
                    stack.pop()
                    if not complete_value():
                        return False, False
                else:
                    return False, False
            index += 1
            continue

    if number_buffer:
        if _is_json_number_complete(number_buffer):
            return bool(complete_value()), bool(root_done and not stack)
        return _is_json_number_prefix(number_buffer), False
    if literal_target is not None:
        return True, False
    if string_context is not None:
        return True, False
    return True, bool(root_done and not stack)


class JsonOutputConstraint:
    """基于 tokenizer 的 JSON object/array 前缀约束。"""

    def __init__(self, tokenizer, *, eos_token_id=None, pad_token_id=None):
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self._decode_cache = {}
        self._vocab_size = self._infer_vocab_size()

    def _infer_vocab_size(self):
        if hasattr(self.tokenizer, "__len__"):
            return int(len(self.tokenizer))
        vocab = getattr(self.tokenizer, "vocab", None)
        if isinstance(vocab, dict):
            return max(int(value) for value in vocab.values()) + 1
        token_to_id = getattr(self.tokenizer, "token_to_id", None)
        if isinstance(token_to_id, dict):
            return max(int(value) for value in token_to_id.values()) + 1
        raise ValueError("JSON 约束解码需要 tokenizer 支持 __len__ 或 vocab/token_to_id。")

    def _decode_token(self, token_id):
        token_id = int(token_id)
        if token_id not in self._decode_cache:
            self._decode_cache[token_id] = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            )
        return self._decode_cache[token_id]

    def is_complete(self, text):
        valid, complete = _json_prefix_status(text)
        return bool(valid and complete)

    def allowed_token_ids(self, text):
        if self.is_complete(text):
            return ()
        allowed = []
        for token_id in range(self._vocab_size):
            if token_id in {self.eos_token_id, self.pad_token_id}:
                continue
            token_text = self._decode_token(token_id)
            if not token_text:
                continue
            valid, _complete = _json_prefix_status(text + token_text)
            if valid:
                allowed.append(token_id)
        return tuple(allowed)


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


def _resolve_effective_output_format(generation_config, *, tools=None):
    """结合 output_format、tool_choice 和 tools 解析实际结构化输出模式。"""
    output_format = _normalize_output_format(getattr(generation_config, "output_format", "text"))
    tool_choice = _normalize_tool_choice(getattr(generation_config, "tool_choice", "none"))
    if tool_choice == "required":
        return "tool_call"
    if output_format == "text" and tool_choice == "auto" and tools:
        return "tool_call"
    return output_format


def _build_output_constraint(output_format, tokenizer, *, eos_token_id, pad_token_id):
    """为结构化输出构造解码约束。"""
    if output_format in {"json", "tool_call"}:
        return JsonOutputConstraint(
            tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    return None


def _parse_structured_output(text, *, output_format, tool_choice):
    """解析结构化输出和 Function Call。"""
    if output_format == "text" and tool_choice == "none":
        return None, False, None, ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        if output_format == "text":
            return None, False, str(error), ()
        return None, False, f"JSON 解析失败: {error}", ()

    if output_format == "tool_call" or tool_choice != "none":
        if not isinstance(parsed, dict) or "tool_calls" not in parsed:
            error = "tool_call 输出必须是包含 tool_calls 的 JSON object。"
            return parsed, False, error, ()
        try:
            tool_calls = normalize_tool_calls(parsed.get("tool_calls"), label="generated tool_calls")
        except (TypeError, ValueError) as error:
            return parsed, False, str(error), ()
        if tool_choice == "required" and not tool_calls:
            return parsed, False, "tool_choice=required 要求至少一个 tool call。", ()
        return parsed, True, None, tuple(dict(tool_call) for tool_call in tool_calls)
    return parsed, True, None, ()


def _generate_token_ids(
    session,
    logits,
    *,
    tokenizer,
    eos_token_id,
    pad_token_id,
    generation_config,
    max_new_tokens,
    thinking_mode_id,
    target_channel_id,
    constraint=None,
    on_token=None,
):
    """按指定原生通道生成 token，并返回最新 logits。"""
    generated_ids = []
    current_logits = logits
    generated_text = ""
    for _step in range(max(0, int(max_new_tokens))):
        if constraint is not None and constraint.is_complete(generated_text):
            break
        allowed_token_ids = None if constraint is None else constraint.allowed_token_ids(generated_text)
        next_id = _select_next_token(
            current_logits[0],
            generated_ids,
            generation_config,
            allowed_token_ids=allowed_token_ids,
        )
        if next_id == eos_token_id or next_id == pad_token_id:
            if constraint is not None and not constraint.is_complete(generated_text):
                continue
            break
        generated_ids.append(next_id)
        token_text = tokenizer.decode([next_id], skip_special_tokens=True)
        generated_text += token_text
        if on_token is not None:
            on_token(next_id, token_text)
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
        if constraint is not None and constraint.is_complete(generated_text):
            break
    return generated_ids, current_logits


@torch.no_grad()
def _generate_single_response(
    model,
    tokenizer,
    *,
    conversation,
    generation_config=None,
    request_id="chat-1",
    tools=None,
    event_sink=None,
):
    """生成单条回复；event_sink 非空时同步推送流式事件。"""
    resolved_generation_config = generation_config or build_default_generation_config()
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    messages = _normalize_conversation(conversation)
    tool_choice = _normalize_tool_choice(getattr(resolved_generation_config, "tool_choice", "none"))
    prompt_messages = _messages_with_tool_context(messages, tools=tools, tool_choice=tool_choice)
    actual_output_format = _resolve_effective_output_format(
        resolved_generation_config,
        tools=tools,
    )
    actual_thinking_mode = _resolve_generation_thinking_mode(resolved_generation_config)
    thinking_visibility = _normalize_visibility(
        getattr(resolved_generation_config, "thinking_visibility", "hidden")
    )
    prompt = render_prompt_from_messages(
        prompt_messages,
        template_version=GlobalConfig.chat_template_version,
        add_generation_prompt=True,
    )
    prompt_segments = render_prompt_segments_from_messages(
        prompt_messages,
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
    session = InferenceSession(model, request_id=request_id)
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
        def emit_thinking(token_id, token_text):
            if event_sink is not None and thinking_visibility == "visible":
                event_sink(
                    GenerationStreamEvent(
                        event_type="thinking_delta",
                        text=token_text,
                        token_id=token_id,
                        channel="thinking",
                    )
                )

        thinking_ids, logits = _generate_token_ids(
            session,
            logits,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            generation_config=resolved_generation_config,
            max_new_tokens=int(getattr(resolved_generation_config, "max_thinking_tokens", 0) or 0),
            thinking_mode_id=thinking_mode_id,
            target_channel_id=target_channel_to_id(TARGET_CHANNEL_THINKING),
            on_token=emit_thinking,
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

    def emit_answer(token_id, token_text):
        if event_sink is not None:
            event_sink(
                GenerationStreamEvent(
                    event_type="answer_delta",
                    text=token_text,
                    token_id=token_id,
                    channel="answer",
                )
            )

    output_constraint = _build_output_constraint(
        actual_output_format,
        tokenizer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    generated_ids, _logits = _generate_token_ids(
        session,
        logits,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        generation_config=resolved_generation_config,
        max_new_tokens=max_new_tokens,
        thinking_mode_id=thinking_mode_id,
        target_channel_id=answer_channel_id,
        constraint=output_constraint,
        on_token=emit_answer,
    )
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    structured_output, structured_valid, structured_error, tool_calls = _parse_structured_output(
        response,
        output_format=actual_output_format,
        tool_choice=tool_choice,
    )
    result = GenerationResult(
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
        output_format=actual_output_format,
        structured_output=structured_output,
        structured_output_valid=structured_valid,
        structured_output_error=structured_error,
        tool_calls=tool_calls,
    )
    if event_sink is not None:
        event_sink(GenerationStreamEvent(event_type="done", result=result))
    return result


@torch.no_grad()
def generate_responses_with_token_counts(
    model,
    tokenizer,
    conversations,
    *,
    generation_config=None,
    request_id_prefix="chat",
    tools=None,
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

    for index, conversation in enumerate(conversation_list, start=1):
        result = _generate_single_response(
            model,
            tokenizer,
            conversation=conversation,
            generation_config=resolved_generation_config,
            request_id=f"{request_id_prefix}-{index}",
            tools=tools,
        )
        results.append(result)
    if was_training:
        model.train()
    return results


@torch.no_grad()
def stream_generate_response_events(
    model,
    tokenizer,
    conversation,
    *,
    generation_config=None,
    request_id="chat-stream",
    tools=None,
):
    """流式生成单条回复，产出 thinking/answer/done 事件。"""
    resolved_generation_config = generation_config or build_default_generation_config()
    was_training = model.training
    model.eval()
    event_queue = queue.Queue()
    sentinel = object()

    def worker():
        try:
            _generate_single_response(
                model,
                tokenizer,
                conversation=conversation,
                generation_config=resolved_generation_config,
                request_id=request_id,
                tools=tools,
                event_sink=event_queue.put,
            )
        except BaseException as error:  # pragma: no cover - 由消费端重新抛出
            event_queue.put(error)
        finally:
            if was_training:
                model.train()
            event_queue.put(sentinel)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        event = event_queue.get()
        if event is sentinel:
            break
        if isinstance(event, BaseException):
            thread.join()
            raise event
        yield event
    thread.join()


def _write_text_by_char(text):
    """逐字符刷新终端输出，保持交互响应更细腻。"""
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()


def _print_generation_result(result):
    """统一打印非流式结果和 token 统计。"""
    if result.thinking is not None:
        print("Thinking> ", end="", flush=True)
        _write_text_by_char(result.thinking)
        print()
    print("Assistant> ", end="", flush=True)
    _write_text_by_char(result.response)
    print()
    if result.tool_calls:
        print("ToolCalls> " + json.dumps(list(result.tool_calls), ensure_ascii=False))
    if result.structured_output_error:
        print(f"StructuredOutputError> {result.structured_output_error}")
    print(
        "tokens "
        f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
    )


def run_chat_session(model, tokenizer, *, generation_config=None, multi_turn=True, stream=False, tools=None):
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
        if stream:
            result = None
            printer = StreamConsolePrinter()
            for event in stream_generate_response_events(
                model,
                tokenizer,
                messages,
                generation_config=generation_config,
                tools=tools,
            ):
                if event.event_type == "done":
                    result = event.result
                else:
                    printer.print_event(event)
            printer.finish()
            if result is not None:
                if result.tool_calls:
                    print("ToolCalls> " + json.dumps(list(result.tool_calls), ensure_ascii=False))
                if result.structured_output_error:
                    print(f"StructuredOutputError> {result.structured_output_error}")
                print(
                    "tokens "
                    f"prompt={result.prompt_token_count} generated={result.generated_token_count}"
                )
        else:
            result = generate_responses_with_token_counts(
                model,
                tokenizer,
                [messages],
                generation_config=generation_config,
                tools=tools,
            )[0]
            _print_generation_result(result)
        if result.tool_calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": list(result.tool_calls)})
        elif result.response.strip():
            messages.append({"role": "assistant", "content": result.response})
