"""LPT v2 模型通用基础层。"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


def _supports_sdpa_gqa():
    """检测当前 PyTorch SDPA 是否原生支持 GQA。"""
    try:
        return "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        return False


SDPA_SUPPORTS_GQA = _supports_sdpa_gqa()


class RMSNorm(nn.Module):
    """均方根归一化。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit 前馈网络。"""

    def __init__(self, hidden_size):
        super().__init__()
        intermediate_size = int(8 * hidden_size / 3)
        intermediate_size = ((intermediate_size + 255) // 256) * 256
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def build_position_ids(attention_mask):
    """按 attention_mask 生成从 0 开始的紧凑位置 id。"""
    return attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
