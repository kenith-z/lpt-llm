"""LPT 协议与模板包。

这个包负责两类长期稳定的约定：
- 结构化消息如何渲染为模型可训练/可推理的字符串模板
- 模板版本如何集中管理，避免训练和推理各写一套协议
"""

from .template import (
    DEFAULT_TEMPLATE_VERSION,
    DS_BOS_TOKEN,
    DS_EOS_TOKEN,
    DS_PAD_TOKEN,
    RenderedSegment,
    TARGET_CHANNEL_ANSWER,
    TARGET_CHANNEL_ID_TO_NAME,
    TARGET_CHANNEL_PROMPT,
    TARGET_CHANNEL_THINKING,
    TARGET_CHANNEL_TO_ID,
    TARGET_CHANNELS,
    THINKING_MODE_AUTO,
    THINKING_MODE_ID_TO_NAME,
    THINKING_MODE_OFF,
    THINKING_MODE_ON,
    THINKING_MODE_TO_ID,
    THINKING_MODES,
    TemplateSpec,
    get_template_spec,
    normalize_target_channel,
    normalize_thinking_mode,
    render_prompt_from_messages,
    render_prompt_segments_from_messages,
    render_training_segments,
    target_channel_to_id,
    thinking_mode_to_id,
    validate_messages,
)
