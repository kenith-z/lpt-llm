"""LPT v2 阶段工作流。"""

from .chat_lora import main as run_chat_lora
from .chat_sft import main as run_chat_sft
from .text_pretrain import main as run_text_pretrain

__all__ = [
    "run_chat_lora",
    "run_chat_sft",
    "run_text_pretrain",
]
