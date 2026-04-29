"""LongRoPE2 位置编码适配层。

这里不再手写一套“像 LongRoPE2”的实现，而是直接以官方下载的
`lpt_model.longrope` 三个 rotary 组件为底层：
- LongRoPEScaledRotaryEmbedding
- DynamicLongRoPEScaledRotaryEmbedding
- MixedLongRoPEScaledRotaryEmbedding

当前 LPT 推理仍采用 factor-switch inference：
- 短上下文使用原始 RoPE
- 超过 original_max_len 后切到静态 LongRoPE
- 中途跨阈值时需要重建 layer_states

训练侧可以独立选择 static / dynamic / mixed。mixed 模式按 position_ids 判断
原始窗口，因此能兼容 sequence packing 后每个样本各自从 0 开始的位置编号。
"""

import torch
import torch.nn as nn

from lpt_config import (
    LONGROPE2_DYNAMIC_EMBEDDING_MODE,
    LONGROPE2_EMBEDDING_MODES,
    LONGROPE2_MIXED_EMBEDDING_MODE,
    LONGROPE2_STATIC_EMBEDDING_MODE,
)

from .longrope import (
    DynamicLongRoPEScaledRotaryEmbedding,
    LongRoPEScaledRotaryEmbedding,
    MixedLongRoPEScaledRotaryEmbedding,
)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _resolve_target_factor(original_max_len, target_length):
    if target_length is None:
        raise ValueError("LongRoPE2 需要显式提供 longrope2_target_length。")
    return max(float(target_length) / float(original_max_len), 1.0)


def _normalize_long_factors(head_dim, original_max_len, target_length, long_factors):
    factor = _resolve_target_factor(original_max_len=original_max_len, target_length=target_length)
    rotary_dims = head_dim // 2

    if long_factors is None:
        return [factor] * rotary_dims
    if isinstance(long_factors, (int, float)):
        return [float(long_factors)] * rotary_dims

    normalized_factors = list(long_factors)
    if len(normalized_factors) != rotary_dims:
        raise ValueError(
            f"longrope2_long_factors 数量 ({len(normalized_factors)}) 必须等于 head_dim/2 ({rotary_dims})。"
        )
    if any(factor_value <= 0 for factor_value in normalized_factors):
        raise ValueError("longrope2_long_factors 必须全部大于 0。")
    return normalized_factors


class LongRoPE2RotaryPositionEncoding(nn.Module):
    """LPT 对 LongRoPE2 官方 rotary 组件的适配层。"""

    def __init__(
        self,
        head_dim,
        max_seq_len,
        base=10000.0,
        original_max_len=2048,
        target_length=None,
        long_factors=None,
        magnitude_scaling_policy="su",
        mscale_factors=None,
        embedding_mode=LONGROPE2_MIXED_EMBEDDING_MODE,
        mixed_original_window=None,
    ):
        super().__init__()
        self.original_max_len = original_max_len
        self.target_length = target_length
        self.embedding_mode = str(embedding_mode)
        if self.embedding_mode not in LONGROPE2_EMBEDDING_MODES:
            raise ValueError(f"未支持的 LongRoPE2 embedding 模式: {self.embedding_mode}")
        if mixed_original_window is None:
            mixed_original_window = original_max_len
        self.mixed_original_window = int(mixed_original_window)
        if self.mixed_original_window < 0:
            raise ValueError("mixed_original_window 不能为负数。")
        self.rescale_factors = _normalize_long_factors(
            head_dim=head_dim,
            original_max_len=original_max_len,
            target_length=target_length,
            long_factors=long_factors,
        )

        self.original_embedding = LongRoPEScaledRotaryEmbedding(
            dim=head_dim,
            rescale_factors=[1.0] * (head_dim // 2),
            max_position_embeddings=original_max_len,
            original_max_position_embeddings=original_max_len,
            base=base,
            magnitude_scaling_policy="1.0",
            model_type="LPT",
        )
        self.long_embedding = LongRoPEScaledRotaryEmbedding(
            dim=head_dim,
            rescale_factors=self.rescale_factors,
            max_position_embeddings=target_length,
            original_max_position_embeddings=original_max_len,
            base=base,
            magnitude_scaling_policy=magnitude_scaling_policy,
            model_type="LPT",
            mscale_factors=mscale_factors,
        )
        self.dynamic_embedding = DynamicLongRoPEScaledRotaryEmbedding(
            dim=head_dim,
            rescale_factors=self.rescale_factors,
            max_position_embeddings=target_length,
            original_max_position_embeddings=original_max_len,
            base=base,
            magnitude_scaling_policy=magnitude_scaling_policy,
            model_type="LPT",
            mscale_factors=mscale_factors,
        )

        self.max_seq_len = max_seq_len

    def should_use_rescaled_rope(self, position_ids=None, sequence_length=None):
        if sequence_length is None:
            if position_ids is None:
                raise ValueError("position_ids 和 sequence_length 不能同时为空。")
            sequence_length = position_ids.max().item() + 1
        return sequence_length > self.original_max_len

    @staticmethod
    def build_mode_tensor(is_rescaled, device):
        return torch.tensor([1 if is_rescaled else 0], dtype=torch.uint8, device=device)

    @staticmethod
    def validate_attention_state_mode(rope_mode, is_rescaled):
        if rope_mode is None:
            return
        state_is_rescaled = bool(rope_mode.item())
        if state_is_rescaled != is_rescaled:
            raise ValueError("LongRoPE2 跨越原始上下文阈值后需要重建 layer_states。")

    def _forward_cos_sin(self, embedding, q, position_ids):
        seq_len = position_ids.max().item() + 1
        return embedding(q, position_ids, seq_len=seq_len)

    def _lookup_mixed_cos_sin(self, q, position_ids):
        original_cos, original_sin = self._forward_cos_sin(
            self.original_embedding,
            q,
            position_ids,
        )
        long_cos, long_sin = self._forward_cos_sin(
            self.long_embedding,
            q,
            position_ids,
        )
        original_window_mask = position_ids.lt(self.mixed_original_window).unsqueeze(-1)
        cos = torch.where(original_window_mask, original_cos, long_cos)
        sin = torch.where(original_window_mask, original_sin, long_sin)
        return cos, sin

    def _lookup_rescaled_cos_sin(self, q, position_ids):
        if self.embedding_mode == LONGROPE2_DYNAMIC_EMBEDDING_MODE:
            return self._forward_cos_sin(self.dynamic_embedding, q, position_ids)
        if self.embedding_mode == LONGROPE2_MIXED_EMBEDDING_MODE:
            return self._lookup_mixed_cos_sin(q, position_ids)
        return self._forward_cos_sin(self.long_embedding, q, position_ids)

    def _lookup_cos_sin(self, q, position_ids):
        if position_ids.max().item() >= self.max_seq_len:
            raise ValueError(
                f"位置索引超出了 LongRoPE2 缓存上限 ({self.max_seq_len})，"
                "请增大当前运行场景的 RoPE 缓存上限。"
            )

        if self.should_use_rescaled_rope(position_ids=position_ids):
            cos, sin = self._lookup_rescaled_cos_sin(q, position_ids)
        else:
            cos, sin = self._forward_cos_sin(self.original_embedding, q, position_ids)
        return cos.unsqueeze(1), sin.unsqueeze(1)

    def build_mixed_embedding(self, start_token_idx, original_embeddings, device=None):
        """暴露 MixedLongRoPE 组件，便于后续训练/实验直接复用。"""
        return MixedLongRoPEScaledRotaryEmbedding(
            dim=self.original_embedding.dim,
            rescale_factors=self.rescale_factors,
            start_token_idx=start_token_idx,
            original_embeddings=original_embeddings,
            max_position_embeddings=self.long_embedding.max_position_embeddings,
            original_max_position_embeddings=self.long_embedding.original_max_position_embeddings,
            base=self.long_embedding.base,
            magnitude_scaling_policy="su",
            model_type="LPT",
            device=device,
        )

    def apply_to_query(self, q, position_ids):
        cos, sin = self._lookup_cos_sin(q, position_ids)
        return (q * cos) + (_rotate_half(q) * sin)

    def apply_to_query_and_key(self, q, k, position_ids):
        cos, sin = self._lookup_cos_sin(q, position_ids)
        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out

    def forward(self, q, k, position_ids):
        return self.apply_to_query_and_key(q, k, position_ids)


def build_rotary_position_encoding(config, max_seq_len, *, embedding_mode=None):
    """根据 ModelConfig 构造 LongRoPE2 位置编码。"""
    if embedding_mode is None:
        embedding_mode = getattr(
            config,
            "longrope2_inference_embedding_mode",
            LONGROPE2_MIXED_EMBEDDING_MODE,
        )
    return LongRoPE2RotaryPositionEncoding(
        head_dim=config.head_dim,
        max_seq_len=max_seq_len,
        base=config.rope_base,
        original_max_len=config.original_max_len,
        target_length=config.longrope2_target_length,
        long_factors=getattr(config, "longrope2_long_factors", None),
        magnitude_scaling_policy=getattr(config, "longrope2_magnitude_scaling_policy", "su"),
        mscale_factors=getattr(config, "longrope2_mscale_factors", None),
        embedding_mode=embedding_mode,
        mixed_original_window=getattr(config, "longrope2_mixed_original_window", None),
    )
