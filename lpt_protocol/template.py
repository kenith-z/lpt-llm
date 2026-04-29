"""DS tokenizer 上的版本化模板定义。"""

from dataclasses import dataclass


DEFAULT_TEMPLATE_VERSION = "lpt-ds-v1"

DS_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DS_EOS_TOKEN = "<｜end▁of▁sentence｜>"
DS_PAD_TOKEN = "<｜▁pad▁｜>"

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
OBSERVATION_ROLE = "observation"

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
    if not isinstance(content, str):
        raise TypeError(f"{label} 必须是字符串。")
    normalized = content.strip()
    if not normalized:
        raise ValueError(f"{label} 不能为空。")
    return normalized


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

        normalized_messages.append(
            {
                "role": role,
                "content": _normalize_content(message.get("content"), label=f"第 {index} 条消息内容"),
            }
        )

    return normalized_messages


def render_prompt_from_messages(messages, template_version=None, add_generation_prompt=False):
    """把结构化消息渲染为推理 prompt。"""
    template_spec = get_template_spec(template_version)
    normalized_messages = validate_messages(messages)

    rendered_parts = [template_spec.prefix]
    for message in normalized_messages:
        rendered_parts.append(template_spec.role_tokens[message["role"]])
        rendered_parts.append(message["content"])
        if message["role"] == ASSISTANT_ROLE:
            rendered_parts.append(template_spec.eos_token)

    if add_generation_prompt:
        rendered_parts.append(template_spec.role_tokens[ASSISTANT_ROLE])

    return "".join(rendered_parts)


def _render_chat_segments(messages, template_version=None):
    template_spec = get_template_spec(template_version)
    normalized_messages = validate_messages(messages)
    rendered_segments = [RenderedSegment(template_spec.prefix, supervise=False)]
    assistant_message_count = 0

    for message in normalized_messages:
        role = message["role"]
        rendered_segments.append(RenderedSegment(template_spec.role_tokens[role], supervise=False))
        is_assistant = role == ASSISTANT_ROLE
        rendered_segments.append(RenderedSegment(message["content"], supervise=is_assistant))
        if is_assistant:
            assistant_message_count += 1
            rendered_segments.append(RenderedSegment(template_spec.eos_token, supervise=True))

    if assistant_message_count == 0:
        raise ValueError("chat 样本至少需要包含一条 assistant 消息。")

    return rendered_segments


def _render_text_segments(text, template_version=None):
    template_spec = get_template_spec(template_version)
    normalized_text = _normalize_content(text, label="text 样本文本")
    return [
        RenderedSegment(normalized_text, supervise=True),
        RenderedSegment(template_spec.eos_token, supervise=True),
    ]


def render_training_segments(sample, template_version=None):
    """把结构化样本渲染成训练片段。"""
    sample_type = sample.get("type")
    if sample_type == "chat":
        return _render_chat_segments(sample["messages"], template_version=template_version)
    if sample_type == "text":
        return _render_text_segments(sample["text"], template_version=template_version)
    raise ValueError(f"不支持的样本类型: {sample_type}")
