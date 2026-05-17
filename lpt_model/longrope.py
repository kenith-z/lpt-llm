# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""LongRoPE 官方 rotary 组件的本地封装。

本文件保留原始 LongRoPE 实现的核心公式，供 `position_encoding.py` 适配到
LPT v2 的 LongRoPE2 训练/推理策略。项目侧新增注释只解释接入边界：
- LPT 路径使用显式 position_ids，兼容 sequence packing。
- dynamic/mixed 类只影响 RoPE 频率表，不修改 checkpoint 权重。
"""

import math
import torch
import transformers

from lpt_config import GlobalConfig


class LongRoPEScaledRotaryEmbedding(torch.nn.Module):
    """
    LongRoPE Scaled Rotary Positional Encoding class for LPT-like model.

    Args:
        dim (int): Head dimension.
        rescale_factors (list): List of rescale factors for each dimension.
        scale (float, optional): Length scale for code compatibility.
        max_position_embeddings (int, optional): Maximum number of position embeddings (after scaled).
        original_max_position_embeddings (int, optional): Original maximum number of position embeddings (before scaled).
        base (int, optional): Base value for the positional encoding. Defaults to 10000.
        magnitude_scaling_policy (str, optional): Attention temperature scaling function. Defaults to "su".
        device (torch.device, optional): Device on which to create the embedding. Defaults to None.
    """

    def __init__(
        self,
        dim,
        rescale_factors,
        scale=1.0,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        model_type="LPT",
        mscale_factors=None,
        device=None,
    ):
        """初始化 LongRoPE 频率缩放表与 magnitude scaling。"""
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base

        if magnitude_scaling_policy == "su":
            calc_mscale = self._calc_mscale_su
        elif magnitude_scaling_policy == "yarn":
            calc_mscale = self._calc_mscale_yarn
        else:
            calc_mscale = lambda scale: float(magnitude_scaling_policy)
        if mscale_factors is None:
            self.mscale = calc_mscale(self.max_position_embeddings / self.original_max_position_embeddings)
        else:
            self.mscale = torch.tensor([*mscale_factors, *mscale_factors], dtype=GlobalConfig.parameter_dtype, device=device)

        self.rescale_factors = torch.tensor(rescale_factors, dtype=GlobalConfig.parameter_dtype, device=device)
        assert self.rescale_factors.shape == (self.dim // 2, ), \
            f"misaligned shape for LongRoPE rescale factors: {self.rescale_factors.shape}"

        if model_type == "LPT":
            self.forward = self._forward_LPT
        elif model_type == "mistral":
            self.forward = self._forward_mistral
            self.register_buffer("inv_freq", self._calc_inv_freq(max_position_embeddings, device))
        else:
            raise ValueError(f"Unsupported model type for LongRoPE: {model_type}")

    def _calc_mscale_su(self, scale):
        """SU 策略的 attention magnitude scaling。"""
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

    def _calc_mscale_yarn(self, scale):
        """YaRN 策略的 attention magnitude scaling。"""
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _calc_inv_freq(self, seq_len, device):
        """按 rescale_factors 计算 RoPE 反频率。"""
        rescale_factors = self.rescale_factors.to(device)
        exponent = torch.arange(0, self.dim, 2, dtype=GlobalConfig.parameter_dtype, device=device) / self.dim
        return 1.0 / (rescale_factors * (self.base ** exponent))

    @torch.no_grad()
    def _forward_mistral(self, x, seq_len=None):
        """兼容 Mistral 形态的 RoPE cos/sin 生成。"""
        seq_len = x.shape[-2] if seq_len is None else seq_len
        t = torch.arange(seq_len, device=x.device, dtype=GlobalConfig.parameter_dtype)
        inv_freq = self.inv_freq.to(x.device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.mscale).to(x.dtype), (emb.sin() * self.mscale).to(x.dtype)

    @torch.no_grad()
    def _forward_LPT(self, x, position_ids, seq_len=None):
        """LPT 路径：使用显式 position_ids 生成 cos/sin。"""
        seq_len = x.shape[-2] if seq_len is None else seq_len
        inv_freq = self._calc_inv_freq(seq_len, x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.mscale
            sin = emb.sin() * self.mscale
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DynamicLongRoPEScaledRotaryEmbedding(LongRoPEScaledRotaryEmbedding):
    """随当前序列长度动态插值 rescale_factors 的 LongRoPE。"""

    def _calc_inv_freq(self, seq_len, device):
        """基于当前 seq_len 计算动态缩放后的反频率。"""
        rescale_factors = self.rescale_factors.to(device)
        current_scale = seq_len / self.original_max_position_embeddings
        original_scale = self.max_position_embeddings / self.original_max_position_embeddings
        dynamic_scale = (current_scale - 1.0) / (original_scale - 1.0)
        rescale_factors = 1.0 + (self.rescale_factors - 1.0) * dynamic_scale
        exponent = torch.arange(0, self.dim, 2, dtype=GlobalConfig.parameter_dtype, device=device) / self.dim
        return 1.0 / (rescale_factors * (self.base ** exponent))


class MixedLongRoPEScaledRotaryEmbedding(LongRoPEScaledRotaryEmbedding):
    """原始窗口内保留原始 RoPE、窗口外使用 LongRoPE 的混合 embedding。"""

    def __init__(
        self,
        dim,
        rescale_factors,
        start_token_idx,
        original_embeddings,
        scale=1.0,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        base=10000,
        magnitude_scaling_policy="su",
        model_type="LPT",
        device=None,
    ):
        """保存原始窗口 embedding，并初始化长窗口 embedding。"""
        self.start_token_idx = start_token_idx
        self.original_embeddings = tuple(x.to(device) for x in original_embeddings)
        super().__init__(
            dim=dim,
            rescale_factors=rescale_factors,
            scale=scale,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=original_max_position_embeddings,
            base=base,
            magnitude_scaling_policy=magnitude_scaling_policy,
            model_type=model_type,
            device=device,
        )
        self._longrope_forward = self.forward
        self.forward = lambda *inputs: self._add_original_embeddings(*self._longrope_forward(*inputs))

    def _add_original_embeddings(self, emb_cos, emb_sin):
        """把原始窗口的 cos/sin 复制回混合结果。"""
        if self.start_token_idx > 0:
            assert self.original_embeddings is not None, \
                'need input original embeddings for start token index > 0'
            emb_cos_origin, emb_sin_origin = self.original_embeddings
            assert emb_cos_origin.shape == emb_cos.shape and emb_sin_origin.shape == emb_cos.shape, \
                'original embeddings shape should be the same with current embeddings'
            emb_cos[..., :self.start_token_idx, :] = emb_cos_origin[..., :self.start_token_idx, :]
            emb_sin[..., :self.start_token_idx, :] = emb_sin_origin[..., :self.start_token_idx, :]
        return emb_cos, emb_sin
