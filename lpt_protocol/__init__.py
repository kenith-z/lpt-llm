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
    TemplateSpec,
    get_template_spec,
    render_prompt_from_messages,
    render_training_segments,
    validate_messages,
)
