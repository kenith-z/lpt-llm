"""DS tokenizer 上的版本化模板定义。"""

from dataclasses import dataclass
import json


DEFAULT_TEMPLATE_VERSION = "lpt-ds-v1"

DS_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DS_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DS_PAD_TOKEN = "<｜▁pad▁｜>"

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
OBSERVATION_ROLE = "observation"

THINKING_MODE_OFF = "off"
THINKING_MODE_ON = "on"
THINKING_MODE_AUTO = "auto"

THINKING_MODES = (
    THINKING_MODE_OFF,
    THINKING_MODE_ON,
    THINKING_MODE_AUTO,
)
THINKING_MODE_TO_ID = {
    THINKING_MODE_OFF: 0,
    THINKING_MODE_ON: 1,
    THINKING_MODE_AUTO: 2,
}
THINKING_MODE_ID_TO_NAME = {
    value: key
    for key, value in THINKING_MODE_TO_ID.items()
}

TARGET_CHANNEL_PROMPT = "prompt"
TARGET_CHANNEL_THINKING = "thinking"
TARGET_CHANNEL_ANSWER = "answer"

TARGET_CHANNELS = (
    TARGET_CHANNEL_PROMPT,
    TARGET_CHANNEL_THINKING,
    TARGET_CHANNEL_ANSWER,
)
TARGET_CHANNEL_TO_ID = {
    TARGET_CHANNEL_PROMPT: 0,
    TARGET_CHANNEL_THINKING: 1,
    TARGET_CHANNEL_ANSWER: 2,
}
TARGET_CHANNEL_ID_TO_NAME = {
    value: key
    for key, value in TARGET_CHANNEL_TO_ID.items()
}

VALID_ROLES = frozenset(
    {
        SYSTEM_ROLE,
        USER_ROLE,
        ASSISTANT_ROLE,
        OBSERVATION_ROLE,
    }
)


@dataclass(frozen=True)
class TemplateSpec:
    """单个模板版本的不可变定义。"""

    version: str
    prefix: str
    role_tokens: dict[str, str]
    eos_token: str


@dataclass(frozen=True)
class RenderedSegment:
    """渲染后的片段及其监督属性。"""

    text: str
    supervise: bool
    thinking_mode: str = THINKING_MODE_OFF
    target_channel: str = TARGET_CHANNEL_PROMPT


LPT_DS_TEMPLATE = TemplateSpec(
    version=DEFAULT_TEMPLATE_VERSION,
    prefix=DS_BOS_TOKEN,
    role_tokens={
        SYSTEM_ROLE: "System: ",
        USER_ROLE: "\n\nUser: ",
        ASSISTANT_ROLE: "\n\nAssistant: ",
        OBSERVATION_ROLE: "\n\nObservation: ",
    },
    eos_token=DS_EOS_TOKEN,
)

TEMPLATE_REGISTRY = {
    LPT_DS_TEMPLATE.version: LPT_DS_TEMPLATE,
}


def get_template_spec(template_version=None):
    """返回指定版本的模板定义。"""
    resolved_version = DEFAULT_TEMPLATE_VERSION if template_version is None else template_version
    try:
        return TEMPLATE_REGISTRY[resolved_version]
    except KeyError as error:
        raise ValueError(f"未知模板版本: {resolved_version}") from error


def _normalize_content(content, *, label):
    """校验模板内容字段，去除首尾空白并拒绝空字符串。"""
    if not isinstance(content, str):
        raise TypeError(f"{label} 必须是字符串。")
    normalized = content.strip()
    if not normalized:
        raise ValueError(f"{label} 不能为空。")
    return normalized


def _normalize_optional_content(content, *, label):
    """校验可选文本字段，空白值归一为空字符串。"""
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError(f"{label} 必须是字符串。")
    return content.strip()


def _normalize_message_content(message, *, label, allow_empty=False):
    """按消息类型规范化 content；Function Call assistant 可允许空 content。"""
    content = message.get("content", "")
    if allow_empty:
        return _normalize_optional_content(content, label=label)
    return _normalize_content(content, label=label)


def _reject_legacy_think_tags(content, *, label):
    """拒绝旧版自然文本 thinking 边界，要求数据先转换为结构化字段。"""
    if "<think>" in content or "</think>" in content:
        raise ValueError(f"{label} 不能包含 <think> 或 </think>，请先转换为 thinking 字段。")


def _jsonable_tool_arguments(arguments, *, label):
    """校验工具调用参数必须可 JSON 序列化。"""
    if not isinstance(arguments, dict):
        raise TypeError(f"{label} 必须是 JSON object/dict。")
    try:
        json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} 必须可 JSON 序列化。") from error
    return arguments


def _normalize_tool_call(tool_call, *, label):
    """规范化单个 Function Call 结构。"""
    if not isinstance(tool_call, dict):
        raise TypeError(f"{label} 必须是字典。")
    name = _normalize_content(tool_call.get("name"), label=f"{label}.name")
    arguments = _jsonable_tool_arguments(
        tool_call.get("arguments", {}),
        label=f"{label}.arguments",
    )
    normalized = {
        "name": name,
        "arguments": arguments,
    }
    tool_call_id = _normalize_optional_content(tool_call.get("id"), label=f"{label}.id")
    if tool_call_id:
        normalized["id"] = tool_call_id
    return normalized


def normalize_tool_calls(tool_calls, *, label="tool_calls"):
    """规范化 assistant 原生 tool_calls。"""
    if tool_calls is None:
        return ()
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError(f"{label} 必须是非空列表。")
    return tuple(
        _normalize_tool_call(tool_call, label=f"{label}[{index}]")
        for index, tool_call in enumerate(tool_calls)
    )


def serialize_tool_calls(tool_calls):
    """把 tool_calls 渲染成稳定 JSON 文本，作为 LM 监督与推理解析边界。"""
    normalized_tool_calls = normalize_tool_calls(tool_calls)
    payload = {"tool_calls": list(normalized_tool_calls)}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_thinking_mode(mode):
    """把 thinking 模式规范化为 off/on/auto。"""
    normalized = THINKING_MODE_OFF if mode is None else str(mode).strip().lower()
    if normalized not in THINKING_MODES:
        raise ValueError(f"thinking_mode 必须是 {THINKING_MODES} 之一。")
    return normalized


def thinking_mode_to_id(mode):
    """返回 thinking 模式的稳定整数 id。"""
    return THINKING_MODE_TO_ID[normalize_thinking_mode(mode)]


def normalize_target_channel(channel):
    """把目标通道规范化为 prompt/thinking/answer。"""
    normalized = TARGET_CHANNEL_PROMPT if channel is None else str(channel).strip().lower()
    if normalized not in TARGET_CHANNELS:
        raise ValueError(f"target_channel 必须是 {TARGET_CHANNELS} 之一。")
    return normalized


def target_channel_to_id(channel):
    """返回目标通道的稳定整数 id。"""
    return TARGET_CHANNEL_TO_ID[normalize_target_channel(channel)]


def validate_messages(messages):
    """校验并标准化消息列表。"""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空列表。")

    normalized_messages = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise TypeError(f"第 {index} 条消息必须是字典。")

        role = message.get("role")
        if role not in VALID_ROLES:
            raise ValueError(f"第 {index} 条消息的 role 非法: {role}")

        has_tool_calls = "tool_calls" in message
        if has_tool_calls and role != ASSISTANT_ROLE:
            raise ValueError("tool_calls 字段只能出现在 assistant 消息上。")
        tool_calls = normalize_tool_calls(
            message.get("tool_calls"),
            label=f"第 {index} 条 assistant tool_calls",
        ) if has_tool_calls else ()
        content = _normalize_message_content(
            message,
            label=f"第 {index} 条消息内容",
            allow_empty=bool(tool_calls),
        )
        normalized_message = {
            "role": role,
            "content": content,
        }
        if role == ASSISTANT_ROLE:
            _reject_legacy_think_tags(content, label=f"第 {index} 条 assistant 消息内容")
            if tool_calls:
                normalized_message["tool_calls"] = list(tool_calls)
            if "thinking" in message:
                thinking = _normalize_optional_content(
                    message.get("thinking"),
                    label=f"第 {index} 条 assistant thinking",
                )
                _reject_legacy_think_tags(thinking, label=f"第 {index} 条 assistant thinking")
                normalized_message["thinking"] = thinking
        elif "thinking" in message:
            _normalize_optional_content(
                message.get("thinking"),
                label=f"第 {index} 条非 assistant thinking",
            )
            raise ValueError("thinking 字段只能出现在 assistant 消息上。")
        if role == OBSERVATION_ROLE and "tool_call_id" in message:
            tool_call_id = _normalize_optional_content(
                message.get("tool_call_id"),
                label=f"第 {index} 条 observation tool_call_id",
            )
            if tool_call_id:
                normalized_message["tool_call_id"] = tool_call_id

        normalized_messages.append(normalized_message)

    return normalized_messages


def _assistant_response_mode(message, thinking_mode=THINKING_MODE_AUTO):
    """按训练策略和 assistant 消息字段决定该轮原生 thinking 分支。"""
    mode = normalize_thinking_mode(thinking_mode)
    if mode in {THINKING_MODE_ON, THINKING_MODE_OFF}:
        return mode
    # auto 只把非空 thinking 视为 thinking on；字段缺失或空白字符串都按 off。
    return THINKING_MODE_ON if str(message.get("thinking") or "").strip() else THINKING_MODE_OFF


def _assistant_answer_text(message):
    """返回 assistant 最终回答通道文本；tool_calls 优先渲染为结构化 JSON。"""
    if message.get("tool_calls"):
        return serialize_tool_calls(message["tool_calls"])
    return message["content"]


def render_prompt_segments_from_messages(
    messages,
    template_version=None,
    add_generation_prompt=False,
    thinking_mode=THINKING_MODE_OFF,
    include_thinking=False,
):
    """把结构化消息渲染为推理 prompt 片段，并保留原生 thinking 控制信息。"""
    template_spec = get_template_spec(template_version)
    normalized_messages = validate_messages(messages)
    generation_mode = normalize_thinking_mode(thinking_mode)

    rendered_segments = [RenderedSegment(template_spec.prefix, supervise=False)]
    for message in normalized_messages:
        role = message["role"]
        is_assistant = role == ASSISTANT_ROLE
        message_mode = _assistant_response_mode(message) if is_assistant else THINKING_MODE_OFF
        role_channel = TARGET_CHANNEL_THINKING if message_mode == THINKING_MODE_ON else TARGET_CHANNEL_ANSWER
        rendered_segments.append(
            RenderedSegment(
                template_spec.role_tokens[role],
                supervise=False,
                thinking_mode=message_mode if is_assistant else THINKING_MODE_OFF,
                target_channel=role_channel if is_assistant else TARGET_CHANNEL_PROMPT,
            )
        )
        if include_thinking and is_assistant and message.get("thinking"):
            rendered_segments.append(
                RenderedSegment(
                    message["thinking"],
                    supervise=False,
                    thinking_mode=THINKING_MODE_ON,
                    target_channel=TARGET_CHANNEL_THINKING,
                )
            )
        rendered_segments.append(
            RenderedSegment(
                _assistant_answer_text(message) if is_assistant else message["content"],
                supervise=False,
                thinking_mode=message_mode if is_assistant else THINKING_MODE_OFF,
                target_channel=TARGET_CHANNEL_ANSWER if is_assistant else TARGET_CHANNEL_PROMPT,
            )
        )
        if is_assistant:
            rendered_segments.append(
                RenderedSegment(
                    template_spec.eos_token,
                    supervise=False,
                    thinking_mode=message_mode,
                    target_channel=TARGET_CHANNEL_ANSWER,
                )
            )

    if add_generation_prompt:
        rendered_segments.append(
            RenderedSegment(
                template_spec.role_tokens[ASSISTANT_ROLE],
                supervise=False,
                thinking_mode=generation_mode,
                target_channel=(
                    TARGET_CHANNEL_THINKING
                    if generation_mode == THINKING_MODE_ON
                    else TARGET_CHANNEL_ANSWER
                ),
            )
        )

    return rendered_segments


def render_prompt_from_messages(messages, template_version=None, add_generation_prompt=False):
    """把结构化消息渲染为推理 prompt。"""
    rendered_segments = render_prompt_segments_from_messages(
        messages,
        template_version=template_version,
        add_generation_prompt=add_generation_prompt,
        include_thinking=False,
    )
    return "".join(segment.text for segment in rendered_segments)


def _render_chat_segments(messages, template_version=None, thinking_mode=THINKING_MODE_AUTO):
    """把 chat 样本渲染成带 supervise 标记的训练片段。"""
    template_spec = get_template_spec(template_version)
    normalized_messages = validate_messages(messages)
    training_thinking_mode = normalize_thinking_mode(thinking_mode)
    rendered_segments = [RenderedSegment(template_spec.prefix, supervise=False)]
    assistant_message_count = 0

    for message in normalized_messages:
        role = message["role"]
        is_assistant = role == ASSISTANT_ROLE
        message_mode = (
            _assistant_response_mode(message, thinking_mode=training_thinking_mode)
            if is_assistant
            else THINKING_MODE_OFF
        )
        role_channel = TARGET_CHANNEL_THINKING if message_mode == THINKING_MODE_ON else TARGET_CHANNEL_ANSWER
        rendered_segments.append(
            RenderedSegment(
                template_spec.role_tokens[role],
                supervise=False,
                thinking_mode=message_mode if is_assistant else THINKING_MODE_OFF,
                target_channel=role_channel if is_assistant else TARGET_CHANNEL_PROMPT,
            )
        )
        # 只监督 assistant 的原生 thinking/content 和 EOS；system/user/observation 作为条件上下文。
        if is_assistant and message.get("thinking"):
            rendered_segments.append(
                RenderedSegment(
                    message["thinking"],
                    supervise=True,
                    thinking_mode=THINKING_MODE_ON,
                    target_channel=TARGET_CHANNEL_THINKING,
                )
            )
        rendered_segments.append(
            RenderedSegment(
                _assistant_answer_text(message) if is_assistant else message["content"],
                supervise=is_assistant,
                thinking_mode=message_mode if is_assistant else THINKING_MODE_OFF,
                target_channel=TARGET_CHANNEL_ANSWER if is_assistant else TARGET_CHANNEL_PROMPT,
            )
        )
        if is_assistant:
            assistant_message_count += 1
            rendered_segments.append(
                RenderedSegment(
                    template_spec.eos_token,
                    supervise=True,
                    thinking_mode=message_mode,
                    target_channel=TARGET_CHANNEL_ANSWER,
                )
            )

    if assistant_message_count == 0:
        raise ValueError("chat 样本至少需要包含一条 assistant 消息。")

    return rendered_segments


def _render_text_segments(text, template_version=None):
    """把纯文本样本渲染成全监督片段。"""
    template_spec = get_template_spec(template_version)
    normalized_text = _normalize_content(text, label="text 样本文本")
    return [
        RenderedSegment(
            normalized_text,
            supervise=True,
            thinking_mode=THINKING_MODE_OFF,
            target_channel=TARGET_CHANNEL_ANSWER,
        ),
        RenderedSegment(
            template_spec.eos_token,
            supervise=True,
            thinking_mode=THINKING_MODE_OFF,
            target_channel=TARGET_CHANNEL_ANSWER,
        ),
    ]


def render_training_segments(sample, template_version=None, thinking_mode=THINKING_MODE_AUTO):
    """把结构化样本渲染成训练片段。"""
    sample_type = sample.get("type")
    if sample_type == "chat":
        return _render_chat_segments(
            sample["messages"],
            template_version=template_version,
            thinking_mode=thinking_mode,
        )
    if sample_type == "text":
        return _render_text_segments(sample["text"], template_version=template_version)
    raise ValueError(f"不支持的样本类型: {sample_type}")
