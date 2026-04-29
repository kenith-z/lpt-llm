"""
LPT (Ling Pre-trained Transformer)

当前版本重点修复并优化了以下问题：
1. 训练参数保持 FP32，前向通过 autocast 使用 BF16。
2. RoPE 缓存改为整模型共享，避免每层重复分配大缓存。
3. 模型主干支持 attention / RetNet 混合层，并统一使用 layer_states 接口。
4. RetNet 支持并行、chunkwise prefill 和递归三种表示。
5. 推理支持 attention_mask，修复批量左填充与状态缓存的位置问题。
6. 优先走原生 GQA 路径，只有在底层算子不支持时才回退到手工扩展 KV。
7. 采样逻辑补齐了 top-k、EOS 提前停止和重复惩罚窗口。
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lpt_config import GenerationConfig, GlobalConfig, ModelConfig, normalize_model_config
from .position_encoding import build_rotary_position_encoding


def _supports_sdpa_gqa():
    try:
        return "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        return False


SDPA_SUPPORTS_GQA = _supports_sdpa_gqa()
ATTENTION_BLOCK_TYPE = "attention"
RETNET_BLOCK_TYPE = "retnet"
ATTENTION_STATE_TYPE = "attention_kv"
RETNET_STATE_TYPE = "retnet_retention"
TRAIN_ROPE_CACHE_SCOPE = "train"
INFERENCE_ROPE_CACHE_SCOPE = "inference"
ROPE_CACHE_MODULE_KEYS = {
    TRAIN_ROPE_CACHE_SCOPE: "train_cache",
    INFERENCE_ROPE_CACHE_SCOPE: "inference_cache",
}

@dataclass(frozen=True)
class LayerState:
    """统一的层状态容器。"""

    state_type: str
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class LayerSpec:
    """描述单层 block 类型与状态路由。"""

    block_type: str
    state_group_id: int | None
    state_slot_index: int | None
    updates_state: bool


def build_layer_state(state_type, *tensors):
    """构造统一的层状态对象。"""
    return LayerState(state_type=state_type, tensors=tuple(tensors))


def build_attention_layer_state(key, value, rope_mode=None):
    """构造注意力层专用的 KV 状态。"""
    if rope_mode is None:
        return build_layer_state(ATTENTION_STATE_TYPE, key, value)
    return build_layer_state(ATTENTION_STATE_TYPE, key, value, rope_mode)


def build_retnet_layer_state(retention_state):
    """构造 RetNet 层专用的保留状态。"""
    return build_layer_state(RETNET_STATE_TYPE, retention_state)


def move_layer_state_tensors(layer_state, *, device=None, dtype=None):
    """按当前层所在设备迁移层状态张量。"""
    if layer_state is None:
        return None
    if not isinstance(layer_state, LayerState):
        raise TypeError("layer_state 必须是 LayerState 或 None。")

    converted_tensors = []
    changed = False
    for tensor in layer_state.tensors:
        target_tensor = tensor
        if device is not None and target_tensor.device != device:
            target_tensor = target_tensor.to(device=device)
            changed = True
        if dtype is not None and target_tensor.dtype != dtype:
            target_tensor = target_tensor.to(dtype=dtype)
            changed = True
        converted_tensors.append(target_tensor)

    if not changed:
        return layer_state
    return build_layer_state(layer_state.state_type, *converted_tensors)


def unpack_attention_layer_state(layer_state):
    """解析注意力层状态。"""
    if layer_state is None:
        return None, None, None

    if not isinstance(layer_state, LayerState):
        raise TypeError("layer_state 必须是 LayerState 或 None。")
    if layer_state.state_type != ATTENTION_STATE_TYPE:
        raise TypeError(
            f"当前注意力层仅支持 {ATTENTION_STATE_TYPE} 状态，"
            f"实际收到 {layer_state.state_type}。"
        )
    if len(layer_state.tensors) == 2:
        key, value = layer_state.tensors
        return key, value, None
    if len(layer_state.tensors) == 3:
        key, value, rope_mode = layer_state.tensors
        return key, value, rope_mode
    raise ValueError("注意力层状态必须包含 key、value，以及可选的 rope_mode 张量。")


def unpack_retnet_layer_state(layer_state):
    """解析 RetNet 层状态。"""
    if layer_state is None:
        return None

    if not isinstance(layer_state, LayerState):
        raise TypeError("layer_state 必须是 LayerState 或 None。")
    if layer_state.state_type != RETNET_STATE_TYPE:
        raise TypeError(
            f"当前 RetNet 层仅支持 {RETNET_STATE_TYPE} 状态，"
            f"实际收到 {layer_state.state_type}。"
        )
    if len(layer_state.tensors) != 1:
        raise ValueError("RetNet 层状态必须只包含 retention_state 一个张量。")
    return layer_state.tensors[0]


def _normalize_cla_group_size(num_layers, cla_share_every_n_layers):
    if not isinstance(cla_share_every_n_layers, int) or cla_share_every_n_layers <= 0:
        raise ValueError("cla_share_every_n_layers 必须是正整数。")
    if num_layers <= 0:
        raise ValueError("num_layers 必须是正整数。")
    return cla_share_every_n_layers


def _normalize_layer_block_types(num_layers, layer_block_types):
    if layer_block_types is None:
        return (ATTENTION_BLOCK_TYPE,) * num_layers
    if len(layer_block_types) != num_layers:
        raise ValueError("layer_block_types 的长度必须与 num_layers 一致。")
    normalized_block_types = tuple(layer_block_types)
    unknown_block_types = sorted(set(normalized_block_types) - {ATTENTION_BLOCK_TYPE, RETNET_BLOCK_TYPE})
    if unknown_block_types:
        raise ValueError(f"发现未注册的 block_type: {', '.join(unknown_block_types)}")
    return normalized_block_types


def _normalize_layer_state_group_ids(
    layer_block_types,
    layer_state_group_ids,
    cla_share_every_n_layers,
):
    num_layers = len(layer_block_types)
    cla_share_every_n_layers = _normalize_cla_group_size(num_layers, cla_share_every_n_layers)
    if layer_state_group_ids is not None:
        if len(layer_state_group_ids) != num_layers:
            raise ValueError("layer_state_group_ids 的长度必须与 num_layers 一致。")
        normalized_group_ids = tuple(layer_state_group_ids)
        for group_id in normalized_group_ids:
            if group_id is not None and (not isinstance(group_id, int) or group_id < 0):
                raise ValueError("layer_state_group_ids 只能包含非负整数或 None。")
        return normalized_group_ids

    num_attention_layers = sum(
        1 for block_type in layer_block_types if block_type == ATTENTION_BLOCK_TYPE
    )
    num_attention_groups = (
        num_attention_layers + cla_share_every_n_layers - 1
    ) // cla_share_every_n_layers
    attention_block_index = 0
    next_unique_group_id = num_attention_groups
    normalized_group_ids = []
    for block_type in layer_block_types:
        if block_type == ATTENTION_BLOCK_TYPE:
            normalized_group_ids.append(attention_block_index // cla_share_every_n_layers)
            attention_block_index += 1
        else:
            normalized_group_ids.append(next_unique_group_id)
            next_unique_group_id += 1
    return tuple(normalized_group_ids)


def _build_layer_specs(layer_block_types, layer_state_group_ids):
    state_group_to_slot_index = {}
    state_group_to_block_type = {}
    layer_specs = []
    for block_type, state_group_id in zip(layer_block_types, layer_state_group_ids):
        if block_type == ATTENTION_BLOCK_TYPE and state_group_id is None:
            raise ValueError("attention block 必须绑定 layer_state_group_id。")
        if block_type == RETNET_BLOCK_TYPE and state_group_id is None:
            raise ValueError("retnet block 必须绑定 layer_state_group_id。")
        state_slot_index = None
        updates_state = False
        if state_group_id is not None:
            existing_block_type = state_group_to_block_type.get(state_group_id)
            if existing_block_type is not None and (
                existing_block_type != ATTENTION_BLOCK_TYPE or block_type != ATTENTION_BLOCK_TYPE
            ):
                raise ValueError("仅 attention 层支持跨层共享状态组，RetNet 层独立占组。")
            if state_group_id not in state_group_to_slot_index:
                state_group_to_slot_index[state_group_id] = len(state_group_to_slot_index)
                state_group_to_block_type[state_group_id] = block_type
                updates_state = True
            state_slot_index = state_group_to_slot_index[state_group_id]
        layer_specs.append(
            LayerSpec(
                block_type=block_type,
                state_group_id=state_group_id,
                state_slot_index=state_slot_index,
                updates_state=updates_state,
            )
        )
    return layer_specs


MODEL_ARCHITECTURE_METADATA_KEYS = (
    "hidden_size",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "num_layers",
    "cla_share_every_n_layers",
    "layer_block_types",
    "layer_state_group_ids",
    "retnet_value_factor",
    "retnet_gate_fn",
    "retnet_chunk_size",
    "rope_base",
    "original_max_len",
    "longrope2_target_length",
    "longrope2_long_factors",
    "longrope2_factor_max_sequence_length",
    "longrope2_magnitude_scaling_policy",
    "longrope2_mscale_factors",
    "longrope2_train_embedding_mode",
    "longrope2_inference_embedding_mode",
    "longrope2_mixed_original_window",
    "num_state_slots",
)


def get_model_architecture_metadata(model):
    """抽取会影响权重语义与状态布局的模型结构元数据。"""
    return {
        "hidden_size": model.config.hidden_size,
        "num_heads": model.config.num_heads,
        "num_kv_heads": model.config.num_kv_heads,
        "head_dim": model.config.head_dim,
        "num_layers": model.config.num_layers,
        "cla_share_every_n_layers": model.config.cla_share_every_n_layers,
        "layer_block_types": model.layer_block_types,
        "layer_state_group_ids": model.layer_state_group_ids,
        "retnet_value_factor": model.config.retnet_value_factor,
        "retnet_gate_fn": model.config.retnet_gate_fn,
        "retnet_chunk_size": model.config.retnet_chunk_size,
        "rope_base": model.config.rope_base,
        "original_max_len": model.config.original_max_len,
        "longrope2_target_length": model.config.longrope2_target_length,
        "longrope2_long_factors": getattr(model, "longrope2_long_factors", model.config.longrope2_long_factors),
        "longrope2_factor_max_sequence_length": model.config.longrope2_factor_max_sequence_length,
        "longrope2_magnitude_scaling_policy": model.config.longrope2_magnitude_scaling_policy,
        "longrope2_mscale_factors": model.config.longrope2_mscale_factors,
        "longrope2_train_embedding_mode": model.config.longrope2_train_embedding_mode,
        "longrope2_inference_embedding_mode": model.config.longrope2_inference_embedding_mode,
        "longrope2_mixed_original_window": model.config.longrope2_mixed_original_window,
        "num_state_slots": model.num_state_slots,
    }


def extract_checkpoint_architecture_metadata(checkpoint):
    """从 checkpoint 中抽取模型结构元数据。"""
    nested_metadata = checkpoint.get("model_architecture_metadata")
    if nested_metadata is None:
        raise ValueError("checkpoint 缺少 model_architecture_metadata。")
    if not isinstance(nested_metadata, dict):
        raise TypeError("model_architecture_metadata 必须是字典。")
    return dict(nested_metadata)


def list_architecture_mismatches(checkpoint, model):
    """返回 checkpoint 与当前模型结构的差异列表。"""
    mismatches = []
    current_metadata = get_model_architecture_metadata(model)
    checkpoint_metadata = extract_checkpoint_architecture_metadata(checkpoint)
    for key, current_value in current_metadata.items():
        checkpoint_value = checkpoint_metadata.get(key, "<missing>")
        if checkpoint_value != current_value:
            mismatches.append((key, checkpoint_value, current_value))
    return mismatches


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


def _build_position_ids(attention_mask):
    return attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)


def _build_sdpa_mask(attention_mask, query_length, key_length, device, segment_ids=None):
    if attention_mask is None and segment_ids is None:
        return None

    if attention_mask is not None:
        if attention_mask.size(1) != key_length:
            raise ValueError(
                f"attention_mask 长度 ({attention_mask.size(1)}) 与 key 长度 ({key_length}) 不一致。"
            )
        key_padding_mask = attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
    else:
        key_padding_mask = None

    segment_mask = None
    if segment_ids is not None:
        if segment_ids.size(1) != key_length:
            raise ValueError(
                f"segment_ids 长度 ({segment_ids.size(1)}) 与 key 长度 ({key_length}) 不一致。"
            )
        key_segment_ids = segment_ids[:, None, None, :].to(device=device, dtype=torch.long)
        query_segment_ids = segment_ids[:, None, -query_length:, None].to(
            device=device,
            dtype=torch.long,
        )
        segment_mask = (query_segment_ids == key_segment_ids) & query_segment_ids.ne(0)

    if (
        attention_mask is not None
        and bool(attention_mask.all())
        and segment_mask is None
    ):
        return None

    if query_length == 1:
        if key_padding_mask is None:
            return segment_mask
        if segment_mask is None:
            return key_padding_mask
        return key_padding_mask & segment_mask

    causal_mask = torch.ones(query_length, key_length, device=device, dtype=torch.bool).tril()
    sdpa_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    if key_padding_mask is not None:
        sdpa_mask = sdpa_mask & key_padding_mask
    if segment_mask is not None:
        sdpa_mask = sdpa_mask & segment_mask
    return sdpa_mask


def _rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(-2)


def _duplicate_interleave(x):
    return torch.repeat_interleave(x, repeats=2, dim=-1)


def _resolve_gate_activation(gate_fn):
    if gate_fn in {"swish", "silu"}:
        return F.silu
    if gate_fn == "gelu":
        return F.gelu
    raise ValueError(f"未支持的 RetNet gate_fn: {gate_fn}")


class XPOSRelativePosition(nn.Module):
    """RetNet 使用的 XPOS 相对位置编码与多尺度衰减参数。"""

    def __init__(self, head_dim, num_heads, scale_base=512, rope_base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RetNet 的 head_dim 必须为偶数，才能进行成对旋转。")

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=GlobalConfig.parameter_dtype) / head_dim))
        xpos_scale = (torch.arange(0, head_dim, 2, dtype=GlobalConfig.parameter_dtype) + 0.4 * head_dim) / (
            1.4 * head_dim
        )
        gamma = 1 - torch.exp(
            torch.linspace(
                math.log(1 / 32),
                math.log(1 / 512),
                num_heads,
                dtype=GlobalConfig.parameter_dtype,
            )
        )

        self.scale_base = scale_base
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("xpos_scale", xpos_scale, persistent=False)
        self.register_buffer("gamma", gamma, persistent=False)
        self.register_buffer("log_gamma", torch.log(gamma), persistent=False)

    def _lookup_sin_cos(self, position_ids, dtype):
        angles = position_ids.to(dtype=GlobalConfig.parameter_dtype).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        sin = _duplicate_interleave(angles.sin().to(dtype=dtype)).unsqueeze(1)
        cos = _duplicate_interleave(angles.cos().to(dtype=dtype)).unsqueeze(1)
        return sin, cos

    def _lookup_scale(self, position_ids, dtype, downscale=False):
        scale = self.xpos_scale.view(1, 1, -1) ** (
            position_ids.to(dtype=GlobalConfig.parameter_dtype).unsqueeze(-1) / self.scale_base
        )
        if downscale:
            scale = scale.reciprocal()
        return _duplicate_interleave(scale.to(dtype=dtype)).unsqueeze(1)

    def apply(self, x, position_ids, downscale=False):
        sin, cos = self._lookup_sin_cos(position_ids, x.dtype)
        scale = self._lookup_scale(position_ids, x.dtype, downscale=downscale)
        rotated = (x * cos) + (_rotate_every_two(x) * sin)
        return rotated * scale

    def apply_to_query(self, q, position_ids):
        return self.apply(q, position_ids, downscale=False)

    def apply_to_query_and_key(self, q, k, position_ids):
        q_out = self.apply(q, position_ids, downscale=False)
        k_out = self.apply(k, position_ids, downscale=True)
        return q_out, k_out

    def build_decay_mask(self, query_positions, key_positions, query_mask, key_mask, dtype):
        relative_position = query_positions.unsqueeze(-1) - key_positions.unsqueeze(-2)
        causal_mask = relative_position >= 0
        decay_mask = torch.exp(
            self.log_gamma.view(1, -1, 1, 1) * relative_position.clamp_min(0).unsqueeze(1).to(GlobalConfig.parameter_dtype)
        ).to(dtype=dtype)
        decay_mask = decay_mask * causal_mask.unsqueeze(1).to(dtype=dtype)

        if query_mask is not None:
            decay_mask = decay_mask * query_mask[:, None, :, None].to(dtype=dtype)
        if key_mask is not None:
            decay_mask = decay_mask * key_mask[:, None, None, :].to(dtype=dtype)
        return decay_mask

    def build_cross_decay(self, query_positions, state_position, query_mask, dtype):
        position_gap = (query_positions - state_position.unsqueeze(-1)).clamp_min(0)
        cross_decay = torch.exp(
            self.log_gamma.view(1, -1, 1) * position_gap.unsqueeze(1).to(GlobalConfig.parameter_dtype)
        ).to(dtype=dtype)
        if query_mask is not None:
            cross_decay = cross_decay * query_mask[:, None, :].to(dtype=dtype)
        return cross_decay

    def build_state_decay(self, token_positions, state_position, token_mask, dtype):
        position_gap = (state_position.unsqueeze(-1) - token_positions).clamp_min(0)
        state_decay = torch.exp(
            self.log_gamma.view(1, -1, 1) * position_gap.unsqueeze(1).to(GlobalConfig.parameter_dtype)
        ).to(dtype=dtype)
        if token_mask is not None:
            state_decay = state_decay * token_mask.unsqueeze(1).to(dtype=dtype)
        return state_decay

    def build_recurrent_decay(self, dtype):
        return self.gamma.view(1, -1, 1, 1).to(dtype=dtype)


class RetNetBlock(nn.Module):
    """适配 LPT Hybrid 主干的多尺度保留模块。

    当前实现支持三种表示：
    - 并行表示：短序列训练与普通 prefill
    - chunkwise 表示：长序列分块预填充
    - 递归表示：自回归增量解码
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        value_factor=1,
        gate_fn="swish",
        chunk_size=256,
        retention_base=10000.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("RetNet 的 hidden_size 必须能被 num_heads 整除。")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = hidden_size // num_heads
        self.value_dim = hidden_size * value_factor
        if self.value_dim % num_heads != 0:
            raise ValueError("RetNet 的 value_dim 必须能被 num_heads 整除。")
        self.value_head_dim = self.value_dim // num_heads
        self.chunk_size = chunk_size
        self.gate_activation = _resolve_gate_activation(gate_fn)

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, self.value_dim)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.relative_position = XPOSRelativePosition(
            head_dim=self.key_dim,
            num_heads=num_heads,
            rope_base=retention_base,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for projection in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.g_proj,
            self.out_proj,
        ):
            nn.init.normal_(projection.weight, mean=0.0, std=1.0 / self.hidden_size)

    def _project_inputs(self, x, position_ids):
        batch_size, query_length, _ = x.shape

        q = self.q_proj(x).view(batch_size, query_length, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, query_length, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size,
            query_length,
            self.num_heads,
            self.value_head_dim,
        ).transpose(1, 2)
        g = self.g_proj(x)

        q, k = self.relative_position.apply_to_query_and_key(q, k, position_ids)
        return q, k, v, g

    def _normalize_output(self, retained, gate, attention_mask, query_length):
        batch_size = retained.size(0)
        retained = retained.transpose(1, 2).reshape(-1, self.value_dim)
        retained = self.group_norm(retained)
        retained = retained.reshape(batch_size, query_length, self.value_dim)
        output = self.out_proj(self.gate_activation(gate) * retained)
        if attention_mask is not None:
            output = output * attention_mask[:, -query_length:].unsqueeze(-1).to(dtype=output.dtype)
        return output

    def _parallel_forward(self, q, k, v, position_ids, attention_mask, segment_ids=None):
        query_length = q.size(2)
        query_positions = position_ids[:, -query_length:]
        query_mask = None if attention_mask is None else attention_mask[:, -query_length:].bool()
        key_mask = None if attention_mask is None else attention_mask.bool()
        decay_mask = self.relative_position.build_decay_mask(
            query_positions=query_positions,
            key_positions=position_ids,
            query_mask=query_mask,
            key_mask=key_mask,
            dtype=q.dtype,
        )
        retention_scores = torch.matmul(q, k.transpose(-1, -2)) * decay_mask
        if segment_ids is not None:
            query_segment_ids = segment_ids[:, -query_length:]
            same_segment_mask = (
                query_segment_ids[:, None, :, None]
                == segment_ids[:, None, None, :]
            ) & query_segment_ids[:, None, :, None].ne(0)
            retention_scores = retention_scores * same_segment_mask.to(dtype=retention_scores.dtype)
        retained = torch.matmul(retention_scores, v)
        return retained

    def _build_parallel_state(self, k, v, position_ids, attention_mask, segment_ids=None):
        if attention_mask is None:
            token_mask = torch.ones_like(position_ids, dtype=torch.bool, device=position_ids.device)
        else:
            token_mask = attention_mask.bool()

        if segment_ids is not None:
            final_segment_ids = segment_ids.masked_fill(~token_mask, 0).amax(dim=-1)
            token_mask = token_mask & segment_ids.eq(final_segment_ids.unsqueeze(-1))

        final_state_position = position_ids.masked_fill(~token_mask, 0).amax(dim=-1)
        if attention_mask is None and segment_ids is None:
            token_mask = None

        state_decay = self.relative_position.build_state_decay(
            token_positions=position_ids,
            state_position=final_state_position,
            token_mask=token_mask,
            dtype=k.dtype,
        )
        retention_state = torch.einsum("bhtd,bhte,bht->bhde", k, v, state_decay)
        return build_retnet_layer_state(retention_state)

    def _chunkwise_forward(self, q, k, v, position_ids, attention_mask):
        batch_size, _, sequence_length, _ = q.shape
        dtype = q.dtype
        device = q.device

        if attention_mask is None:
            token_mask = torch.ones(batch_size, sequence_length, device=device, dtype=torch.bool)
        else:
            token_mask = attention_mask.bool()

        previous_state = torch.zeros(
            batch_size,
            self.num_heads,
            self.key_dim,
            self.value_head_dim,
            device=device,
            dtype=dtype,
        )
        previous_state_position = torch.zeros(batch_size, device=device, dtype=position_ids.dtype)
        has_previous_state = torch.zeros(batch_size, device=device, dtype=torch.bool)
        outputs = []

        for chunk_start in range(0, sequence_length, self.chunk_size):
            chunk_end = min(sequence_length, chunk_start + self.chunk_size)
            q_chunk = q[:, :, chunk_start:chunk_end]
            k_chunk = k[:, :, chunk_start:chunk_end]
            v_chunk = v[:, :, chunk_start:chunk_end]
            chunk_positions = position_ids[:, chunk_start:chunk_end]
            chunk_mask = token_mask[:, chunk_start:chunk_end]

            decay_mask = self.relative_position.build_decay_mask(
                query_positions=chunk_positions,
                key_positions=chunk_positions,
                query_mask=chunk_mask,
                key_mask=chunk_mask,
                dtype=dtype,
            )
            inner_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * decay_mask
            inner_chunk = torch.matmul(inner_scores, v_chunk)

            cross_decay = self.relative_position.build_cross_decay(
                query_positions=chunk_positions,
                state_position=previous_state_position,
                query_mask=chunk_mask,
                dtype=dtype,
            )
            cross_chunk = torch.einsum("bhcd,bhde->bhce", q_chunk, previous_state)
            cross_chunk = cross_chunk * cross_decay.unsqueeze(-1)
            cross_chunk = torch.where(
                has_previous_state[:, None, None, None],
                cross_chunk,
                torch.zeros_like(cross_chunk),
            )

            outputs.append(inner_chunk + cross_chunk)

            chunk_has_active_tokens = chunk_mask.any(dim=-1)
            current_state_position = chunk_positions.masked_fill(~chunk_mask, 0).amax(dim=-1)
            previous_state_decay = self.relative_position.build_recurrent_decay(dtype=dtype)
            position_gap = (current_state_position - previous_state_position).clamp_min(0).to(dtype=GlobalConfig.parameter_dtype)
            if position_gap.numel() > 0:
                previous_state_decay = previous_state_decay ** position_gap.view(batch_size, 1, 1, 1)
            decayed_previous_state = previous_state * previous_state_decay.to(dtype=dtype)
            decayed_previous_state = torch.where(
                has_previous_state[:, None, None, None],
                decayed_previous_state,
                torch.zeros_like(decayed_previous_state),
            )

            state_decay = self.relative_position.build_state_decay(
                token_positions=chunk_positions,
                state_position=current_state_position,
                token_mask=chunk_mask,
                dtype=dtype,
            )
            current_chunk_state = torch.einsum("bhtd,bhte,bht->bhde", k_chunk, v_chunk, state_decay)
            next_state = decayed_previous_state + current_chunk_state

            previous_state = torch.where(
                chunk_has_active_tokens[:, None, None, None],
                next_state,
                previous_state,
            )
            previous_state_position = torch.where(
                chunk_has_active_tokens,
                current_state_position,
                previous_state_position,
            )
            has_previous_state = has_previous_state | chunk_has_active_tokens

        retained = torch.cat(outputs, dim=2)
        return retained, build_retnet_layer_state(previous_state)

    def _recurrent_step(self, q, k, v, update_mask, layer_state, should_update_state):
        previous_state = unpack_retnet_layer_state(layer_state)
        if previous_state is None:
            previous_state = torch.zeros(
                q.size(0),
                self.num_heads,
                self.key_dim,
                self.value_head_dim,
                device=q.device,
                dtype=q.dtype,
            )

        current_outer = torch.einsum("bhd,bhe->bhde", k, v)
        retention_decay = self.relative_position.build_recurrent_decay(dtype=q.dtype)
        active_mask = update_mask[:, None, None, None].to(dtype=torch.bool)

        if should_update_state:
            updated_state = previous_state * retention_decay + current_outer
            next_state = torch.where(active_mask, updated_state, previous_state)
        else:
            next_state = previous_state

        retained = torch.einsum("bhd,bhde->bhe", q, next_state)
        retained = retained * update_mask[:, None, None].to(dtype=retained.dtype)
        return retained.unsqueeze(2), build_retnet_layer_state(next_state)

    def _sequential_forward(self, q, k, v, attention_mask, layer_state, should_update_state):
        query_length = q.size(2)
        if attention_mask is None:
            update_mask = torch.ones(q.size(0), query_length, device=q.device, dtype=q.dtype)
        else:
            update_mask = attention_mask[:, -query_length:].to(dtype=q.dtype)

        outputs = []
        current_state = layer_state
        for step_index in range(query_length):
            step_output, current_state = self._recurrent_step(
                q=q[:, :, step_index],
                k=k[:, :, step_index],
                v=v[:, :, step_index],
                update_mask=update_mask[:, step_index],
                layer_state=current_state,
                should_update_state=should_update_state,
            )
            outputs.append(step_output)
        return torch.cat(outputs, dim=2), current_state

    def forward(
        self,
        x,
        position_ids,
        rope_cache,
        attention_mask=None,
        segment_ids=None,
        layer_state=None,
        should_update_state=True,
    ):
        del rope_cache

        q, k, v, gate = self._project_inputs(x, position_ids)
        query_length = q.size(2)

        if layer_state is None and query_length > 1:
            if segment_ids is None and self.chunk_size is not None and query_length > self.chunk_size:
                retained, new_layer_state = self._chunkwise_forward(
                    q=q,
                    k=k,
                    v=v,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
            else:
                retained = self._parallel_forward(
                    q=q,
                    k=k,
                    v=v,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
                new_layer_state = self._build_parallel_state(
                    k=k,
                    v=v,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                )
        else:
            retained, new_layer_state = self._sequential_forward(
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask,
                layer_state=layer_state,
                should_update_state=should_update_state,
            )

        output = self._normalize_output(
            retained=retained,
            gate=gate,
            attention_mask=attention_mask,
            query_length=query_length,
        )
        return output, new_layer_state


class ModernAttention(nn.Module):
    """融合 GQA、KV Cache 和 SDPA 的注意力模块。"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_rate = dropout_rate

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x,
        position_ids,
        rope_cache,
        attention_mask=None,
        segment_ids=None,
        layer_state=None,
        should_update_state=True,
    ):
        batch_size, query_length, _ = x.shape

        q = self.q_proj(x).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)

        if should_update_state:
            k = self.k_proj(x).view(
                batch_size,
                query_length,
                self.num_kv_heads,
                self.head_dim,
            ).transpose(1, 2)
            v = self.v_proj(x).view(
                batch_size,
                query_length,
                self.num_kv_heads,
                self.head_dim,
            ).transpose(1, 2)

            use_rescaled_rope = rope_cache.should_use_rescaled_rope(position_ids=position_ids)
            rope_mode = rope_cache.build_mode_tensor(use_rescaled_rope, device=x.device)
            q, k = rope_cache(q, k, position_ids)
            past_k, past_v, past_rope_mode = unpack_attention_layer_state(layer_state)
            if past_k is not None:
                rope_cache.validate_attention_state_mode(past_rope_mode, use_rescaled_rope)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_layer_state = build_attention_layer_state(k, v, rope_mode=rope_mode)
        else:
            use_rescaled_rope = rope_cache.should_use_rescaled_rope(position_ids=position_ids)
            rope_mode = rope_cache.build_mode_tensor(use_rescaled_rope, device=x.device)
            q = rope_cache.apply_to_query(q, position_ids)
            k, v, past_rope_mode = unpack_attention_layer_state(layer_state)
            if k is None or v is None:
                raise ValueError("共享状态层在只读模式下必须收到可复用的 layer_state。")
            rope_cache.validate_attention_state_mode(past_rope_mode, use_rescaled_rope)
            new_layer_state = build_attention_layer_state(
                k,
                v,
                rope_mode=rope_mode if past_rope_mode is None else past_rope_mode,
            )

        key_length = k.size(2)
        sdpa_mask = _build_sdpa_mask(
            attention_mask,
            query_length,
            key_length,
            x.device,
            segment_ids=segment_ids,
        )

        sdpa_kwargs = {
            "attn_mask": sdpa_mask,
            "dropout_p": self.dropout_rate if self.training else 0.0,
            "is_causal": sdpa_mask is None and query_length > 1 and key_length == query_length,
        }

        if self.num_heads != self.num_kv_heads and SDPA_SUPPORTS_GQA:
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **sdpa_kwargs)
        else:
            num_kv_groups = self.num_heads // self.num_kv_heads
            if num_kv_groups > 1:
                k = k.repeat_interleave(num_kv_groups, dim=1)
                v = v.repeat_interleave(num_kv_groups, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

        out = out.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        return self.o_proj(out), new_layer_state


def build_sequence_mixer(block_type, config):
    """根据 block 类型构造序列混合器。"""
    if block_type == ATTENTION_BLOCK_TYPE:
        return ModernAttention(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            dropout_rate=config.dropout_rate,
        )
    if block_type == RETNET_BLOCK_TYPE:
        return RetNetBlock(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            value_factor=config.retnet_value_factor,
            gate_fn=config.retnet_gate_fn,
            chunk_size=config.retnet_chunk_size,
            retention_base=config.rope_base,
        )
    raise ValueError(f"未支持的 block_type: {block_type}")


class TransformerBlock(nn.Module):
    """统一的 Decoder block，外层残差/FFN 包装保持 LPT 风格。"""

    def __init__(self, layer_spec, config):
        super().__init__()
        self.layer_spec = layer_spec
        self.block_type = layer_spec.block_type
        self.sequence_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.sequence_mixer = build_sequence_mixer(self.block_type, config)
        self.feed_forward = SwiGLU(config.hidden_size)

    def _forward_impl(
        self,
        x,
        position_ids,
        rope_cache,
        attention_mask=None,
        segment_ids=None,
        layer_state=None,
        should_update_state=True,
    ):
        mixed_out, new_layer_state = self.sequence_mixer(
            self.sequence_norm(x),
            position_ids=position_ids,
            rope_cache=rope_cache,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            layer_state=layer_state,
            should_update_state=should_update_state,
        )
        x = x + mixed_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_layer_state

    def _pack_layer_state_for_checkpoint(self, layer_state, reference_tensor):
        has_state = torch.zeros(1, device=reference_tensor.device, dtype=torch.uint8)
        empty_tensor = reference_tensor.new_empty((0,))
        if layer_state is None:
            if self.block_type == ATTENTION_BLOCK_TYPE:
                return has_state, empty_tensor, empty_tensor, empty_tensor
            return has_state, empty_tensor

        has_state.fill_(1)
        if self.block_type == ATTENTION_BLOCK_TYPE:
            key, value, rope_mode = unpack_attention_layer_state(layer_state)
            return has_state, key, value, rope_mode

        retention_state = unpack_retnet_layer_state(layer_state)
        return has_state, retention_state

    def _unpack_layer_state_from_checkpoint(self, has_state_tensor, *state_tensors):
        if has_state_tensor.numel() == 0 or not bool(has_state_tensor.item()):
            return None
        if self.block_type == ATTENTION_BLOCK_TYPE:
            return build_attention_layer_state(*state_tensors)
        return build_retnet_layer_state(state_tensors[0])

    def forward_with_gradient_checkpointing(
        self,
        x,
        position_ids,
        rope_cache,
        attention_mask=None,
        segment_ids=None,
        layer_state=None,
        should_update_state=True,
    ):
        packed_state_inputs = self._pack_layer_state_for_checkpoint(layer_state, x)

        def checkpointed_forward(hidden_states, *checkpoint_state_tensors):
            restored_layer_state = self._unpack_layer_state_from_checkpoint(
                checkpoint_state_tensors[0],
                *checkpoint_state_tensors[1:],
            )
            hidden_out, updated_layer_state = self._forward_impl(
                hidden_states,
                position_ids=position_ids,
                rope_cache=rope_cache,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                layer_state=restored_layer_state,
                should_update_state=should_update_state,
            )
            return (
                hidden_out,
                *self._pack_layer_state_for_checkpoint(updated_layer_state, hidden_out),
            )

        checkpoint_outputs = checkpoint(
            checkpointed_forward,
            x,
            *packed_state_inputs,
            use_reentrant=False,
        )
        hidden_states = checkpoint_outputs[0]
        updated_layer_state = self._unpack_layer_state_from_checkpoint(
            checkpoint_outputs[1],
            *checkpoint_outputs[2:],
        )
        return hidden_states, updated_layer_state

    def forward(
        self,
        x,
        position_ids,
        rope_cache,
        attention_mask=None,
        segment_ids=None,
        layer_state=None,
        should_update_state=True,
    ):
        return self._forward_impl(
            x,
            position_ids=position_ids,
            rope_cache=rope_cache,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            layer_state=layer_state,
            should_update_state=should_update_state,
        )


class LPT(nn.Module):
    """Ling Pre-trained Transformer 主模型。"""

    def __init__(self, vocabulary_size, config=None):
        super().__init__()
        self.config = normalize_model_config(config)
        self.layer_block_types = _normalize_layer_block_types(
            self.config.num_layers,
            self.config.layer_block_types,
        )
        self.layer_state_group_ids = _normalize_layer_state_group_ids(
            self.layer_block_types,
            self.config.layer_state_group_ids,
            self.config.cla_share_every_n_layers,
        )
        self.layer_specs = _build_layer_specs(
            self.layer_block_types,
            self.layer_state_group_ids,
        )
        self.layer_to_state_slot = [layer_spec.state_slot_index for layer_spec in self.layer_specs]
        self.num_state_slots = sum(
            1 for layer_spec in self.layer_specs if layer_spec.updates_state
        )

        self.token_embedding = nn.Embedding(vocabulary_size, self.config.hidden_size)
        self._rope_cache_limits = {
            TRAIN_ROPE_CACHE_SCOPE: int(GlobalConfig.train_rope_cache_max_sequence_length),
            INFERENCE_ROPE_CACHE_SCOPE: int(GlobalConfig.inference_rope_cache_max_sequence_length),
        }
        for scope, max_seq_len in self._rope_cache_limits.items():
            if max_seq_len <= 0:
                raise ValueError(f"{scope} RoPE 缓存上限必须是正整数。")
        if self._rope_cache_limits[INFERENCE_ROPE_CACHE_SCOPE] < self._rope_cache_limits[TRAIN_ROPE_CACHE_SCOPE]:
            raise ValueError("inference RoPE 缓存上限不能小于 train RoPE 缓存上限。")

        self._reset_rope_caches()
        self.layers = nn.ModuleList([
            TransformerBlock(layer_spec, self.config)
            for layer_spec in self.layer_specs
        ])
        self.final_norm = RMSNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.to(dtype=GlobalConfig.parameter_dtype)
        self._promote_stable_paths_to_fp32()

    def _build_rope_cache(self, max_seq_len, scope):
        if scope == TRAIN_ROPE_CACHE_SCOPE:
            embedding_mode = self.config.longrope2_train_embedding_mode
        elif scope == INFERENCE_ROPE_CACHE_SCOPE:
            embedding_mode = self.config.longrope2_inference_embedding_mode
        else:
            raise ValueError(f"未支持的 rope_cache_scope: {scope}")
        return build_rotary_position_encoding(
            config=self.config,
            max_seq_len=max_seq_len,
            embedding_mode=embedding_mode,
        )

    def _record_resolved_longrope2_factors(self):
        self.longrope2_long_factors = tuple(
            float(value)
            for value in self._rope_caches[ROPE_CACHE_MODULE_KEYS[TRAIN_ROPE_CACHE_SCOPE]].rescale_factors
        )

    def _reset_rope_caches(self):
        self._rope_caches = nn.ModuleDict({
            ROPE_CACHE_MODULE_KEYS[TRAIN_ROPE_CACHE_SCOPE]: self._build_rope_cache(
                self._rope_cache_limits[TRAIN_ROPE_CACHE_SCOPE],
                TRAIN_ROPE_CACHE_SCOPE,
            )
        })
        self._rope_caches.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
        self._record_resolved_longrope2_factors()

    def refresh_longrope2_factors(self, long_factors, factor_max_sequence_length):
        """更新 LongRoPE2 factors，并重建训练/推理 RoPE cache。"""
        self.config = self.config.with_overrides(
            longrope2_long_factors=tuple(float(value) for value in long_factors),
            longrope2_factor_max_sequence_length=int(factor_max_sequence_length),
        )
        self._reset_rope_caches()

    def get_rope_cache(self, scope):
        if scope not in self._rope_cache_limits:
            raise ValueError(f"未支持的 rope_cache_scope: {scope}")
        module_key = ROPE_CACHE_MODULE_KEYS[scope]
        if module_key not in self._rope_caches:
            rope_cache = self._build_rope_cache(self._rope_cache_limits[scope], scope)
            rope_cache.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
            self._rope_caches[module_key] = rope_cache
        return self._rope_caches[module_key]

    def _resolve_rope_cache_scope(self, rope_cache_scope, layer_states=None, segment_ids=None):
        if rope_cache_scope is not None:
            if rope_cache_scope not in self._rope_cache_limits:
                raise ValueError(f"未支持的 rope_cache_scope: {rope_cache_scope}")
            return rope_cache_scope
        if layer_states is not None:
            return INFERENCE_ROPE_CACHE_SCOPE
        if segment_ids is not None or self.training:
            return TRAIN_ROPE_CACHE_SCOPE
        return INFERENCE_ROPE_CACHE_SCOPE

    def _promote_stable_paths_to_fp32(self):
        """对长上下文敏感路径采用 FP32，优先保证缓存一致性。"""
        if GlobalConfig.parameter_dtype == GlobalConfig.parameter_dtype:
            return

        promoted_any_layer = False
        for layer in self.layers:
            if (
                layer.block_type == ATTENTION_BLOCK_TYPE
                and getattr(GlobalConfig, "stable_attention_fp32_enabled", False)
            ):
                layer.to(dtype=GlobalConfig.parameter_dtype)
                promoted_any_layer = True
            elif (
                layer.block_type == RETNET_BLOCK_TYPE
                and getattr(GlobalConfig, "stable_retnet_fp32_enabled", False)
            ):
                layer.to(dtype=GlobalConfig.parameter_dtype)
                promoted_any_layer = True

        if not promoted_any_layer:
            return

        self.token_embedding.to(dtype=GlobalConfig.parameter_dtype)
        self.final_norm.to(dtype=GlobalConfig.parameter_dtype)
        self.lm_head.weight = self.token_embedding.weight

    def _normalize_incoming_layer_states(self, layer_states=None):
        """校验并标准化层状态输入。"""
        if layer_states is None:
            return [None] * self.num_state_slots

        if len(layer_states) != self.num_state_slots:
            raise ValueError(
                f"layer_states 数量 ({len(layer_states)}) 与模型状态槽位数量 "
                f"({self.num_state_slots}) 不一致。"
            )
        return list(layer_states)

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        segment_ids=None,
        layer_states=None,
        rope_cache_scope=None,
    ):
        """统一支持训练、prefill 和带层状态的增量推理。"""

        embedding_device = self.token_embedding.weight.device
        if input_ids.device != embedding_device:
            input_ids = input_ids.to(device=embedding_device)

        if attention_mask is None:
            position_reference_mask = None
            model_attention_mask = None
        else:
            position_reference_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
            model_attention_mask = position_reference_mask
            if bool(model_attention_mask.bool().all()):
                model_attention_mask = None

        if position_ids is None:
            if position_reference_mask is None:
                position_ids = torch.arange(
                    input_ids.size(1),
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(input_ids.size(0), -1)
            else:
                position_ids = _build_position_ids(position_reference_mask)
        else:
            position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)

        if segment_ids is not None:
            segment_ids = segment_ids.to(device=input_ids.device, dtype=torch.long)
            if segment_ids.shape != input_ids.shape:
                raise ValueError(
                    "segment_ids 形状必须与 input_ids 一致，"
                    f"当前分别为 {tuple(segment_ids.shape)} 和 {tuple(input_ids.shape)}。"
                )
            if layer_states is not None:
                raise ValueError("segment_ids 当前仅支持无 layer_states 的整段前向。")

        rope_cache_scope = self._resolve_rope_cache_scope(
            rope_cache_scope,
            layer_states=layer_states,
            segment_ids=segment_ids,
        )
        rope_cache = self.get_rope_cache(rope_cache_scope)
        max_position = position_ids.max().item() + 1
        rope_cache_max_sequence_length = self._rope_cache_limits[rope_cache_scope]
        if max_position > rope_cache_max_sequence_length:
            raise ValueError(
                f"输入上下文长度 ({max_position}) 超出了当前 {rope_cache_scope} RoPE 缓存上限 "
                f"({rope_cache_max_sequence_length})。"
            )

        hidden_states = self.token_embedding(input_ids)
        previous_layer_states = self._normalize_incoming_layer_states(layer_states=layer_states)
        use_gradient_checkpointing = (
            self.training
            and torch.is_grad_enabled()
            and getattr(GlobalConfig, "gradient_checkpointing_enabled", False)
        )

        new_layer_states = [None] * self.num_state_slots
        for layer, layer_spec in zip(self.layers, self.layer_specs):
            layer_device = layer.sequence_norm.weight.device
            if hidden_states.device != layer_device:
                hidden_states = hidden_states.to(device=layer_device)
            layer_position_ids = position_ids.to(device=layer_device)
            layer_attention_mask = (
                None
                if model_attention_mask is None
                else model_attention_mask.to(device=layer_device)
            )
            layer_segment_ids = (
                None
                if segment_ids is None
                else segment_ids.to(device=layer_device)
            )
            layer_rope_cache = rope_cache.to(device=layer_device)
            layer_dtype = layer.sequence_norm.weight.dtype
            if hidden_states.dtype != layer_dtype:
                hidden_states = hidden_states.to(dtype=layer_dtype)

            slot_index = layer_spec.state_slot_index
            layer_state = None if slot_index is None else previous_layer_states[slot_index]
            layer_state = move_layer_state_tensors(
                layer_state,
                device=layer_device,
                dtype=layer_dtype,
            )

            if use_gradient_checkpointing:
                hidden_states, updated_layer_state = layer.forward_with_gradient_checkpointing(
                    hidden_states,
                    position_ids=layer_position_ids,
                    rope_cache=layer_rope_cache,
                    attention_mask=layer_attention_mask,
                    segment_ids=layer_segment_ids,
                    layer_state=layer_state,
                    should_update_state=layer_spec.updates_state,
                )
            else:
                hidden_states, updated_layer_state = layer(
                    hidden_states,
                    position_ids=layer_position_ids,
                    rope_cache=layer_rope_cache,
                    attention_mask=layer_attention_mask,
                    segment_ids=layer_segment_ids,
                    layer_state=layer_state,
                    should_update_state=layer_spec.updates_state,
                )
            if slot_index is not None:
                previous_layer_states[slot_index] = updated_layer_state
                new_layer_states[slot_index] = updated_layer_state

        final_norm_device = self.final_norm.weight.device
        if hidden_states.device != final_norm_device:
            hidden_states = hidden_states.to(device=final_norm_device)
        final_hidden_dtype = self.final_norm.weight.dtype
        if hidden_states.dtype != final_hidden_dtype:
            hidden_states = hidden_states.to(dtype=final_hidden_dtype)
        logits = self.lm_head(self.final_norm(hidden_states))
        return logits, new_layer_states

    def _apply_repetition_penalty_vectorized(
        self,
        logits,
        token_history,
        penalty=1.0,
        history_mask=None,
        repetition_window_size=None,
    ):
        """对最近窗口内的历史 token 施加重复惩罚。"""

        if penalty == 1.0 or token_history.numel() == 0:
            return logits

        if repetition_window_size is not None and token_history.size(1) > repetition_window_size:
            token_history = token_history[:, -repetition_window_size:]
            if history_mask is not None:
                history_mask = history_mask[:, -repetition_window_size:]

        if history_mask is None:
            history_mask = torch.ones_like(token_history, dtype=torch.bool)
        else:
            history_mask = history_mask.to(device=token_history.device, dtype=torch.bool)

        safe_history = token_history.masked_fill(~history_mask, 0)
        repeated_token_counts = torch.zeros_like(logits)
        repeated_token_counts.scatter_add_(
            dim=1,
            index=safe_history,
            src=history_mask.to(dtype=logits.dtype),
        )
        repeated_token_mask = repeated_token_counts > 0

        penalized_logits = torch.where(
            logits > 0,
            logits / penalty,
            logits * penalty,
        )
        return torch.where(repeated_token_mask, penalized_logits, logits)

    def _temperature_and_top_p(self, logits, config):
        """温度缩放、top-k 与 top-p 采样融合。"""

        if config.temperature > 0:
            logits = logits / max(config.temperature, 1e-5)

        if 0 < config.top_k < logits.size(-1):
            topk_values = torch.topk(logits, k=config.top_k, dim=-1).values
            cutoff = topk_values[..., -1:].clone()
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        if config.top_p >= 1.0:
            return probs

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative <= config.top_p
        keep_mask[..., 0] = True

        sorted_probs = sorted_probs.masked_fill(~keep_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, sorted_probs)
        return filtered_probs

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens,
        config: GenerationConfig,
        attention_mask=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """采用统一 layer_states 的自回归生成。"""
        from lpt_inference.session import InferenceSession

        self.eval()
        batch_size = prompt_tokens.size(0)
        device = prompt_tokens.device

        if attention_mask is None:
            full_attention_mask = torch.ones_like(prompt_tokens, dtype=torch.long, device=device)
        else:
            full_attention_mask = attention_mask.to(device=device, dtype=torch.long)

        output_sequence = prompt_tokens.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        session = InferenceSession(self)
        logits = session.prefill(prompt_tokens, attention_mask=full_attention_mask)

        for _ in range(config.max_length):
            if not (~finished).any():
                break
            if session.attention_mask.size(1) >= GlobalConfig.inference_max_sequence_length:
                break

            next_token_logits = logits[:, -1, :].float()

            history_mask = session.attention_mask.bool()
            next_token_logits = self._apply_repetition_penalty_vectorized(
                next_token_logits,
                output_sequence,
                penalty=config.repetition_penalty,
                history_mask=history_mask,
                repetition_window_size=config.repetition_window_size,
            )

            if pad_token_id is not None and pad_token_id != eos_token_id:
                next_token_logits[:, pad_token_id] = float("-inf")

            if config.do_sample:
                probs = self._temperature_and_top_p(next_token_logits, config)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            active_mask = ~finished
            if not active_mask.all():
                fill_value = 0 if pad_token_id is None else pad_token_id
                next_token = torch.where(
                    active_mask.unsqueeze(-1),
                    next_token,
                    torch.full_like(next_token, fill_value),
                )

            output_sequence = torch.cat([output_sequence, next_token], dim=1)
            step_attention_mask = active_mask.to(dtype=session.attention_mask.dtype).unsqueeze(-1)

            if eos_token_id is not None:
                finished = finished | (active_mask & next_token.squeeze(-1).eq(eos_token_id))

            if not (~finished).any():
                break
            if session.attention_mask.size(1) + next_token.size(1) > GlobalConfig.inference_max_sequence_length:
                break

            logits = session.append(next_token, attention_mask=step_attention_mask)

        return output_sequence
