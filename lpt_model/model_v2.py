"""LPT v2 主干模型。

本文件实现 v2 P1 的最小可运行主干：
- Local SDPA Attention 主干，FA3/FA2 只作为后续评估后端保留在选择器中。
- Shared RetNetAssist 生成低维摘要，默认通过 Q-only adapter 调制 query。
- Paged KV cache 以页池保存局部窗口 K/V，Assist 状态与 KV 生命周期隔离。
- 同质 SwiGLU-MoE FFN，router 使用 FP32，experts 无状态。

代码注释重点对齐 `help/LPTv2模型定型方案.md`：
- 训练 forward 默认关闭 KV cache，避免把训练 K/V 写入运行态页池。
- RetNetAssist 只通过低秩 adapter 影响当前 Q/K 或上下文残差，不写入 Paged KV。
- xLSTMMemory 只作为 FFN 输入 adapter，不作为 expert、router target 或 Attention 状态。
- Paged KV、RetNetAssistState、xLSTMMemoryState 三类状态由 request_id 隔离。
"""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpt_config import (
    GlobalConfig,
    LPT_V2_ARCHITECTURE_VERSION,
    PAGED_KV_CACHE_BACKEND,
    count_retnet_assist_enabled_layers,
    is_retnet_assist_enabled_for_layer,
    is_xlstm_memory_enabled_for_layer,
    normalize_model_config,
)
from lpt_runtime import resolve_attention_backend

from .common import RMSNorm, SDPA_SUPPORTS_GQA, SwiGLU, build_position_ids
from .position_encoding import build_rotary_position_encoding
from .state_v2 import (
    AttentionLayerState,
    LayerStateV2,
    MoELayerState,
    PagedKVReference,
    RetNetAssistState,
    xLSTMMemoryState,
)
from .state_pool_v2 import RetNetAssistStatePool, xLSTMMemoryStatePool


DEFAULT_REQUEST_ID = "default"


def _as_layer_state_v2(layer_state):
    """把调用方传入的 layer_state 规范化为 v2 状态对象。"""
    if layer_state is None:
        return LayerStateV2()
    if not isinstance(layer_state, LayerStateV2):
        raise TypeError("LPT v2 layer_state 必须是 LayerStateV2 或 None。")
    return layer_state


def _slice_tail_mask(attention_mask, key_length):
    """取与当前 K/V 长度对齐的 attention mask 尾段。"""
    if attention_mask is None:
        return None
    if attention_mask.size(1) < key_length:
        raise ValueError("attention_mask 长度不能短于当前 K/V 长度。")
    return attention_mask[:, -key_length:]


def _move_optional_tensor(tensor, device, *, dtype=None):
    """移动可选张量，避免调用方到处重复 None 判断。"""
    if tensor is None:
        return None
    if dtype is None:
        return tensor.to(device=device)
    return tensor.to(device=device, dtype=dtype)


def _retnet_enabled_for_layer(config, layer_index):
    """按配置判断当前层是否实际挂载 RetNetAssist。"""
    return is_retnet_assist_enabled_for_layer(config, layer_index)


def _retnet_group_id(config, layer_index):
    """把层号映射到 RetNet 参数/状态共享组。"""
    group_size = int(config.retnet_sharing_group_size)
    if group_size <= 0:
        raise ValueError("retnet_sharing_group_size 必须为正整数。")
    return int(layer_index) // group_size


def _retnet_parameter_slot_for_layer(config, layer_index):
    """计算当前层应该复用的 RetNet 参数槽位。"""
    if not _retnet_enabled_for_layer(config, layer_index):
        return None
    sharing = str(config.retnet_parameter_sharing)
    if sharing == "global":
        return 0
    if sharing == "group":
        return _retnet_group_id(config, layer_index)
    if sharing == "per_layer":
        return int(layer_index)
    raise ValueError(f"未知 retnet_parameter_sharing: {sharing}")


def _retnet_state_slot_for_layer(config, layer_index):
    """计算当前层应该绑定的 RetNet request-bound 状态槽位。"""
    if not _retnet_enabled_for_layer(config, layer_index):
        return int(layer_index)
    sharing = str(config.retnet_state_sharing)
    if sharing == "group":
        return _retnet_group_id(config, layer_index)
    if sharing == "per_layer":
        return int(layer_index)
    raise ValueError(f"未知 retnet_state_sharing: {sharing}")


def _retnet_layer_to_state_slots(config):
    """生成 layer -> state_slot 映射，供状态池和 block 初始化复用。"""
    return tuple(_retnet_state_slot_for_layer(config, layer_index) for layer_index in range(int(config.num_layers)))


def _retnet_state_slot_count(config):
    """返回状态池需要预留的 RetNet slot 数量。"""
    slots = {
        _retnet_state_slot_for_layer(config, layer_index)
        for layer_index in range(int(config.num_layers))
        if _retnet_enabled_for_layer(config, layer_index)
    }
    if not slots:
        return int(config.num_layers)
    return max(slots) + 1


def _build_local_attention_mask(
    attention_mask,
    query_length,
    key_length,
    device,
    *,
    window_size,
    segment_ids=None,
):
    """构造 sliding-window causal mask，兼容 prefill、decode 与 sequence packing。"""
    key_positions = torch.arange(key_length, device=device)
    query_positions = torch.arange(
        key_length - query_length,
        key_length,
        device=device,
    )
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    window_mask = key_positions.unsqueeze(0) >= (
        query_positions.unsqueeze(1) - int(window_size) + 1
    )
    mask = (causal_mask & window_mask).unsqueeze(0).unsqueeze(0)

    tail_attention_mask = _slice_tail_mask(attention_mask, key_length)
    if tail_attention_mask is not None:
        # 只屏蔽当前 K/V 可见范围内的 padding；prefill/decode 拼接历史 K/V 后，
        # 前缀 token 已通过 key_length 对齐，因此这里不能直接使用完整 attention_mask。
        key_padding_mask = tail_attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
        mask = mask & key_padding_mask

    if segment_ids is not None:
        if segment_ids.size(1) < key_length:
            raise ValueError("segment_ids 长度不能短于当前 K/V 长度。")
        key_segment_ids = segment_ids[:, -key_length:][:, None, None, :].to(device=device, dtype=torch.long)
        query_segment_ids = segment_ids[:, -query_length:][:, None, :, None].to(device=device, dtype=torch.long)
        # sequence packing 通过 segment_id 阻断同一 packed row 内不同样本之间的监督泄漏。
        segment_mask = (query_segment_ids == key_segment_ids) & query_segment_ids.ne(0)
        mask = mask & segment_mask

    if bool(mask.all()):
        return None
    return mask


class PagedKVCache:
    """轻量页池，用 page 引用保存每层局部窗口 K/V。"""

    def __init__(self, page_block_size, attention_window_size):
        self.page_block_size = int(page_block_size)
        self.attention_window_size = int(attention_window_size)
        if self.page_block_size <= 0:
            raise ValueError("page_block_size 必须为正整数。")
        if self.attention_window_size <= 0:
            raise ValueError("attention_window_size 必须为正整数。")
        self._next_page_id = 1
        self._pages: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._request_layer_pages: dict[tuple[str, int], tuple[int, ...]] = {}

    @property
    def allocated_page_count(self):
        """当前页池中实际持有的页数量，用于资源报告和泄漏排查。"""
        return len(self._pages)

    @property
    def allocated_bytes(self):
        """估算页池内 K/V 张量占用字节数，不包含 Python 容器开销。"""
        total_bytes = 0
        for page_key, page_value in self._pages.values():
            total_bytes += page_key.numel() * page_key.element_size()
            total_bytes += page_value.numel() * page_value.element_size()
        return int(total_bytes)

    def reset(self, request_id=None):
        """释放页池；指定 request_id 时只释放该请求相关页。"""
        if request_id is None:
            self._pages.clear()
            self._request_layer_pages.clear()
            return

        request_id = str(request_id)
        for key, page_ids in list(self._request_layer_pages.items()):
            if key[0] != request_id:
                continue
            for page_id in page_ids:
                self._pages.pop(page_id, None)
            self._request_layer_pages.pop(key, None)

    def read(self, paged_kv_ref):
        """按页表引用还原连续 K/V 张量。"""
        if paged_kv_ref is None or not paged_kv_ref.page_ids:
            return None, None
        keys = []
        values = []
        for page_id in paged_kv_ref.page_ids:
            page_key, page_value = self._pages[page_id]
            keys.append(page_key)
            values.append(page_value)
        return torch.cat(keys, dim=2), torch.cat(values, dim=2)

    def update(self, request_id, layer_index, new_key, new_value, previous_ref=None):
        """追加 K/V 后按局部窗口裁剪，并返回新的页表引用。"""
        request_id = str(request_id)
        layer_index = int(layer_index)
        previous_key, previous_value = self.read(previous_ref)
        if previous_key is not None:
            full_key = torch.cat([previous_key.to(new_key.device), new_key], dim=2)
            full_value = torch.cat([previous_value.to(new_value.device), new_value], dim=2)
        else:
            full_key = new_key
            full_value = new_value

        if full_key.size(2) > self.attention_window_size:
            # Paged KV 只保留局部窗口内真实 token 的 K/V；全局摘要由 RetNet/xLSTM
            # 独立状态池维护，不能依赖这里的窗口裁剪生命周期。
            full_key = full_key[:, :, -self.attention_window_size:]
            full_value = full_value[:, :, -self.attention_window_size:]

        key = (request_id, layer_index)
        for page_id in self._request_layer_pages.get(key, ()):
            # 当前实现以“重写该 request/layer 页表”的方式保持简单可靠。
            # 旧页先释放，新的窗口内容再按 page_block_size 重新切页。
            self._pages.pop(page_id, None)

        page_ids = []
        for start in range(0, full_key.size(2), self.page_block_size):
            end = min(start + self.page_block_size, full_key.size(2))
            page_id = self._next_page_id
            self._next_page_id += 1
            page_ids.append(page_id)
            self._pages[page_id] = (
                full_key[:, :, start:end],
                full_value[:, :, start:end],
            )
        self._request_layer_pages[key] = tuple(page_ids)
        token_count = (
            int(previous_ref.token_count)
            if previous_ref is not None
            else 0
        ) + int(new_key.size(2))
        return PagedKVReference(
            request_id=request_id,
            layer_index=layer_index,
            page_ids=tuple(page_ids),
            token_count=token_count,
            window_token_count=int(full_key.size(2)),
        )

    def runtime_metadata(self):
        """返回 Paged KV 当前页池元数据，不包含实际 K/V 张量内容。"""
        request_layers = []
        for (request_id, layer_index), page_ids in sorted(self._request_layer_pages.items()):
            token_count = 0
            element_count = 0
            dtype = None
            device = None
            for page_id in page_ids:
                page_key, page_value = self._pages[page_id]
                token_count += int(page_key.size(2))
                element_count += int(page_key.numel() + page_value.numel())
                dtype = str(page_key.dtype).removeprefix("torch.")
                device = str(page_key.device)
            request_layers.append(
                {
                    "request_id": request_id,
                    "layer_index": int(layer_index),
                    "page_ids": list(page_ids),
                    "page_count": len(page_ids),
                    "window_token_count": token_count,
                    "element_count": element_count,
                    "dtype": dtype,
                    "device": device,
                }
            )
        return {
            "cache_backend": "paged_kv",
            "page_block_size": self.page_block_size,
            "attention_window_size": self.attention_window_size,
            "allocated_page_count": self.allocated_page_count,
            "allocated_bytes": self.allocated_bytes,
            "request_layers": request_layers,
        }


class SharedRetNetAssist(nn.Module):
    """跨层共享的 RetNetAssist 轻量摘要模块。"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.state_dim = int(config.retnet_state_dim)
        self.input_proj = nn.Linear(self.hidden_size, self.state_dim, bias=False)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.activation = nn.SiLU()

    def _initial_summary(self, x):
        """创建与 batch/device/dtype 对齐的 RetNet 初始摘要。"""
        return torch.zeros(
            x.size(0),
            self.state_dim,
            device=x.device,
            dtype=x.dtype,
        )

    def forward(
        self,
        x_norm,
        attention_mask=None,
        previous_state=None,
        request_id=DEFAULT_REQUEST_ID,
        layer_index=0,
        state_slot=None,
    ):
        """返回每个 token 的摘要序列和更新后的 request-bound 状态。"""
        projected = self.activation(self.input_proj(x_norm))
        if attention_mask is None:
            token_mask = torch.ones(
                projected.size(0),
                projected.size(1),
                device=projected.device,
                dtype=projected.dtype,
            )
        else:
            token_mask = attention_mask[:, -projected.size(1):].to(device=projected.device, dtype=projected.dtype)
        masked_projected = projected * token_mask.unsqueeze(-1)

        if previous_state is None or previous_state.summary is None:
            previous_summary = self._initial_summary(x_norm)
            previous_count = 0
        else:
            previous_summary = previous_state.summary.to(device=x_norm.device, dtype=x_norm.dtype)
            previous_count = int(previous_state.token_count)

        prefix_sum = torch.cumsum(masked_projected, dim=1)
        active_counts = torch.cumsum(token_mask, dim=1)
        total_counts = active_counts + float(previous_count)
        # 状态保存未投影的 running summary，避免 decode 时对历史摘要重复套用 state_proj。
        # 这里用 prefix-scan 近似 recurrent 摘要，使 prefill 不退化成 Python 逐 token 循环。
        raw_summary_sequence = (
            prefix_sum + previous_summary.unsqueeze(1) * float(previous_count)
        ) / total_counts.clamp_min(1.0).unsqueeze(-1)
        summary_sequence = self.state_proj(raw_summary_sequence)

        final_count = int(previous_count + token_mask.sum(dim=1).max().item())
        has_new_tokens = token_mask.sum(dim=1).gt(0)
        final_summary = torch.where(
            has_new_tokens[:, None],
            raw_summary_sequence[:, -1],
            previous_summary,
        )
        summary_norm = None
        if not torch.is_grad_enabled():
            summary_norm = float(final_summary.float().norm(dim=-1).mean().detach().cpu())
        new_state = RetNetAssistState(
            request_id=request_id,
            layer_index=layer_index,
            state_slot=layer_index if state_slot is None else state_slot,
            summary=final_summary.detach(),
            token_count=final_count,
            summary_norm=summary_norm,
        )
        return summary_sequence, new_state


class QOnlyRetNetAdapter(nn.Module):
    """RetNetAssist 低秩 adapter；默认只调制 Q，第 24 项可同时调制 K。"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = int(config.num_heads)
        self.num_kv_heads = int(config.num_kv_heads)
        self.head_dim = int(config.head_dim)
        self.down_projection = nn.Linear(config.retnet_state_dim, config.retnet_adapter_rank, bias=False)
        self.up_projection = nn.Linear(
            config.retnet_adapter_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.alpha_q = nn.Parameter(torch.tensor(float(config.retnet_adapter_alpha_q_init), dtype=torch.float32))
        self.alpha_q.requires_grad_(bool(config.retnet_adapter_alpha_q_trainable))
        self.k_adapter_enabled = bool(config.retnet_k_adapter_enabled)
        if self.k_adapter_enabled:
            self.k_down_projection = nn.Linear(config.retnet_state_dim, config.retnet_adapter_rank, bias=False)
            self.k_up_projection = nn.Linear(
                config.retnet_adapter_rank,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            self.alpha_k = nn.Parameter(torch.tensor(float(config.retnet_adapter_alpha_k_init), dtype=torch.float32))
            self.alpha_k.requires_grad_(bool(config.retnet_adapter_alpha_k_trainable))
        else:
            self.k_down_projection = None
            self.k_up_projection = None
            self.alpha_k = None

    def _apply(self, fn):
        """保持 RetNet scale 参数始终为 FP32，避免半精度下小 scale 被吞掉。"""
        super()._apply(fn)
        self.alpha_q.data = self.alpha_q.data.float()
        if self.alpha_q.grad is not None:
            self.alpha_q.grad.data = self.alpha_q.grad.data.float()
        if self.alpha_k is not None:
            self.alpha_k.data = self.alpha_k.data.float()
            if self.alpha_k.grad is not None:
                self.alpha_k.grad.data = self.alpha_k.grad.data.float()
        return self

    def _query_delta(self, summary_sequence, query):
        """把低维摘要投影到 Q 的 head 形状。"""
        delta = self.up_projection(self.down_projection(summary_sequence))
        return delta.view(query.size(0), query.size(2), self.num_heads, self.head_dim).transpose(1, 2)

    def _key_delta(self, summary_sequence, key):
        """实验性 K adapter；默认关闭，避免污染当前 Q-only 主线。"""
        if not self.k_adapter_enabled or self.k_down_projection is None or self.k_up_projection is None:
            return None
        delta = self.k_up_projection(self.k_down_projection(summary_sequence))
        return delta.view(key.size(0), key.size(2), self.num_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, summary_sequence, query):
        """只调制 Q 的默认路径，保持 K/V cache 语义稳定。"""
        delta = self._query_delta(summary_sequence, query)
        alpha = self.alpha_q.to(dtype=query.dtype)
        return query + alpha * delta

    def apply_to_qk(self, summary_sequence, query, key):
        """同时返回调制后的 Q/K 以及 adapter 观测指标。"""
        q_delta = self._query_delta(summary_sequence, query)
        alpha_q = self.alpha_q.to(dtype=query.dtype)
        query = query + alpha_q * q_delta

        k_delta = self._key_delta(summary_sequence, key)
        alpha_k_value = None
        if k_delta is not None and self.alpha_k is not None:
            alpha_k = self.alpha_k.to(dtype=key.dtype)
            key = key + alpha_k * k_delta
            if not torch.is_grad_enabled():
                alpha_k_value = float(self.alpha_k.detach().cpu())

        metrics = {
            "q_adapter_delta_norm": None,
            "k_adapter_delta_norm": None,
            "alpha_q": None,
            "alpha_k": None,
        }
        if not torch.is_grad_enabled():
            metrics = {
                "q_adapter_delta_norm": float(q_delta.float().norm(dim=-1).mean().detach().cpu()),
                "k_adapter_delta_norm": 0.0
                if k_delta is None
                else float(k_delta.float().norm(dim=-1).mean().detach().cpu()),
                "alpha_q": float(self.alpha_q.detach().cpu()),
                "alpha_k": alpha_k_value,
            }
        return query, key, metrics


class RetNetContextAdapter(nn.Module):
    """RetNetAssist 低秩上下文注入 adapter。

    该模块只复用 SharedRetNetAssist 产生的 summary_sequence，不新增检索状态；
    注入位置放在 Attention 输出投影之后，由 block 外层残差统一完成
    ``x = x + attention_output``。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.down_projection = nn.Linear(config.retnet_state_dim, config.retnet_adapter_rank, bias=False)
        self.up_projection = nn.Linear(config.retnet_adapter_rank, self.hidden_size, bias=False)
        self.alpha_context = nn.Parameter(
            torch.tensor(float(config.retnet_context_adapter_alpha), dtype=torch.float32)
        )

    def _apply(self, fn):
        """保持 context 注入 scale 为 FP32，便于小初值稳定训练。"""
        super()._apply(fn)
        self.alpha_context.data = self.alpha_context.data.float()
        if self.alpha_context.grad is not None:
            self.alpha_context.grad.data = self.alpha_context.grad.data.float()
        return self

    def forward(self, summary_sequence, hidden):
        """在 Attention 输出投影后注入轻量上下文残差。"""
        delta = self.up_projection(self.down_projection(summary_sequence))
        alpha = self.alpha_context.to(dtype=hidden.dtype)
        output = hidden + alpha * delta
        metrics = {
            "context_adapter_delta_norm": None,
            "alpha_context": None,
        }
        if not torch.is_grad_enabled():
            metrics = {
                "context_adapter_delta_norm": float(delta.float().norm(dim=-1).mean().detach().cpu()),
                "alpha_context": float(self.alpha_context.detach().cpu()),
            }
        return output, metrics


class LocalAttentionMixerV2(nn.Module):
    """Local Attention + RetNetAssist Q/QK adapter。"""

    def __init__(
        self,
        config,
        layer_index,
        retnet_assist,
        q_adapter,
        context_adapter,
        paged_kv_cache,
        *,
        retnet_state_slot=None,
    ):
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.retnet_state_slot = self.layer_index if retnet_state_slot is None else int(retnet_state_slot)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.num_kv_heads = int(config.num_kv_heads)
        self.head_dim = int(config.head_dim)
        self.dropout_rate = float(config.dropout_rate)
        self.retnet_assist = retnet_assist
        self.paged_kv_cache = paged_kv_cache
        self.q_adapter = q_adapter
        self.context_adapter = context_adapter
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.backend_decision = resolve_attention_backend(
            config.attention_backend_policy,
            priority=config.attention_backend_priority,
            required_capabilities=("training", "prefill", "sliding_window", "longrope2"),
            dtype=GlobalConfig.parameter_dtype,
        )

    def _retnet_enabled_for_layer(self):
        """本层是否参与 RetNetAssist 参数和状态更新。"""
        return _retnet_enabled_for_layer(self.config, self.layer_index)

    def _read_past_kv(self, attention_state):
        """读取上一轮 decode/prefill 保存的 K/V，兼容 paged 与 dense fallback。"""
        if attention_state is None:
            return None, None
        if self.config.cache_backend == PAGED_KV_CACHE_BACKEND:
            return self.paged_kv_cache.read(attention_state.paged_kv_ref)
        if attention_state.dense_kv_state:
            return attention_state.dense_kv_state
        return None, None

    def forward(
        self,
        x_norm,
        position_ids,
        rope_cache,
        attention_mask=None,
        segment_ids=None,
        layer_state=None,
        request_id=DEFAULT_REQUEST_ID,
        use_kv_cache=True,
    ):
        """执行单层局部注意力，并返回新的 Attention/RetNet 状态。"""
        layer_state = _as_layer_state_v2(layer_state)
        batch_size, query_length, _ = x_norm.shape

        # Q/K/V 先由当前 token 的归一化特征生成，RetNetAssist 只在后续调制 Q/K；
        # 这样已写入 cache 的历史 K/V 不会被后验修改。
        q = self.q_proj(x_norm).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        summary_sequence = None
        if self._retnet_enabled_for_layer():
            if self.retnet_assist is None or self.q_adapter is None:
                raise RuntimeError("启用 RetNetAssist 的层缺少 retnet_assist 或 q_adapter 模块。")
            summary_sequence, retnet_state = self.retnet_assist(
                x_norm,
                attention_mask=attention_mask,
                previous_state=layer_state.retnet_assist,
                request_id=request_id,
                layer_index=self.layer_index,
                state_slot=self.retnet_state_slot,
            )
            # adapter metrics 会写入 RetNetAssistState，资源/实验报告可直接观察
            # delta_norm 与 alpha，便于区分机制是否真正参与前向。
            q, k, adapter_metrics = self.q_adapter.apply_to_qk(summary_sequence, q, k)
            retnet_state = replace(retnet_state, **adapter_metrics)
        else:
            # 禁用 RetNetAssist 的层必须清空传入状态，避免 state pool 绑定时把共享 slot 的状态串到未启用层。
            retnet_state = None

        q, k = rope_cache(q, k, position_ids)
        use_kv_cache = bool(use_kv_cache)
        past_k, past_v = self._read_past_kv(layer_state.attention) if use_kv_cache else (None, None)

        if use_kv_cache and self.config.cache_backend == PAGED_KV_CACHE_BACKEND:
            previous_ref = None if layer_state.attention is None else layer_state.attention.paged_kv_ref
            paged_ref = self.paged_kv_cache.update(
                request_id=request_id,
                layer_index=self.layer_index,
                new_key=k,
                new_value=v,
                previous_ref=previous_ref,
            )
            k, v = self.paged_kv_cache.read(paged_ref)
            attention_state = AttentionLayerState(
                request_id=request_id,
                layer_index=self.layer_index,
                paged_kv_ref=paged_ref,
            )
        else:
            # 训练 forward 会传入 use_kv_cache=False，因此不会向页池写入训练 K/V；
            # dense fallback 仅用于非 paged cache 的调试路径。
            if use_kv_cache and past_k is not None:
                k = torch.cat([past_k.to(k.device), k], dim=2)
                v = torch.cat([past_v.to(v.device), v], dim=2)
            if k.size(2) > int(self.config.attention_window_size):
                k = k[:, :, -int(self.config.attention_window_size):]
                v = v[:, :, -int(self.config.attention_window_size):]
            attention_state = None
            if use_kv_cache:
                attention_state = AttentionLayerState(
                    request_id=request_id,
                    layer_index=self.layer_index,
                    dense_kv_state=(k.detach(), v.detach()),
                )

        sdpa_mask = _build_local_attention_mask(
            attention_mask,
            query_length,
            k.size(2),
            x_norm.device,
            window_size=self.config.attention_window_size,
            segment_ids=segment_ids,
        )
        sdpa_kwargs = {
            "attn_mask": sdpa_mask,
            "dropout_p": self.dropout_rate if self.training else 0.0,
            "is_causal": False,
        }
        if self.num_heads != self.num_kv_heads and SDPA_SUPPORTS_GQA:
            # PyTorch 原生 GQA 可用时直接走 enable_gqa；否则手动 repeat KV heads。
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **sdpa_kwargs)
        else:
            num_kv_groups = self.num_heads // self.num_kv_heads
            if num_kv_groups > 1:
                k = k.repeat_interleave(num_kv_groups, dim=1)
                v = v.repeat_interleave(num_kv_groups, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        out = out.transpose(1, 2).contiguous().view(batch_size, query_length, self.hidden_size)
        attention_output = self.o_proj(out)
        if self._retnet_enabled_for_layer() and bool(self.config.retnet_context_adapter_enabled):
            if self.context_adapter is None or summary_sequence is None:
                raise RuntimeError("启用 RetNetContextAdapter 的层缺少 context_adapter 或 summary_sequence。")
            attention_output, context_metrics = self.context_adapter(summary_sequence, attention_output)
            retnet_state = replace(retnet_state, **context_metrics)

        new_state = replace(
            layer_state,
            attention=attention_state,
            retnet_assist=retnet_state,
        )
        return attention_output, new_state


class SwiGLUMoE(nn.Module):
    """同质 SwiGLU-MoE FFN。"""

    def __init__(self, config, layer_index):
        super().__init__()
        self.layer_index = int(layer_index)
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.moe_num_experts)
        self.top_k = int(config.moe_top_k)
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(self.hidden_size) for _ in range(self.num_experts)])

    def forward(self, x_ffn, request_id=DEFAULT_REQUEST_ID):
        """按 router top-k 稀疏执行 SwiGLU experts。"""
        router_logits = self.router(x_ffn).float()
        topk_logits, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1).to(dtype=x_ffn.dtype)

        flat_x = x_ffn.reshape(-1, self.hidden_size)
        flat_indices = topk_indices.reshape(-1, self.top_k)
        flat_weights = topk_weights.reshape(-1, self.top_k)
        flat_output = torch.zeros_like(flat_x)
        for route_rank in range(self.top_k):
            route_indices = flat_indices[:, route_rank]
            route_weights = flat_weights[:, route_rank].unsqueeze(-1)
            for expert_index, expert in enumerate(self.experts):
                selected_positions = torch.nonzero(
                    route_indices.eq(expert_index),
                    as_tuple=False,
                ).flatten()
                if selected_positions.numel() == 0:
                    continue
                # 只对当前 token 命中的 expert 建图，未命中的 expert 不参与本 batch 的前向/反向。
                selected_x = flat_x.index_select(0, selected_positions)
                routed_output = expert(selected_x)
                routed_output = routed_output * route_weights.index_select(0, selected_positions)
                flat_output = flat_output.index_add(0, selected_positions, routed_output)
        moe_output = flat_output.view_as(x_ffn)

        expert_counts = torch.bincount(
            topk_indices.reshape(-1),
            minlength=self.num_experts,
        )
        router_probs = F.softmax(router_logits, dim=-1)
        # MoE 状态只保存观测指标，不保存专家中间激活，避免 checkpoint 膨胀。
        router_entropy = -(router_probs * router_probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        load_fraction = expert_counts.to(dtype=router_logits.dtype) / max(1, topk_indices.numel())
        load_balance_loss = (load_fraction * load_fraction).sum() * self.num_experts
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        moe_state = MoELayerState(
            request_id=request_id,
            layer_index=self.layer_index,
            expert_token_counts=tuple(int(value) for value in expert_counts.detach().cpu().tolist()),
            router_entropy=float(router_entropy.detach().cpu()),
            load_balance_loss=float(load_balance_loss.detach().cpu()),
            router_z_loss=float(router_z_loss.detach().cpu()),
        )
        return moe_output, moe_state


class xLSTMMemoryAssist(nn.Module):
    """FFN 侧外挂记忆 adapter。"""

    def __init__(self, config, layer_index):
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.hidden_size = int(config.hidden_size)
        self.state_dim = int(config.xlstm_memory_state_dim)
        self.input_proj = nn.Linear(self.hidden_size, self.state_dim, bias=False)
        self.state_proj = nn.Linear(self.state_dim, self.state_dim, bias=False)
        self.down_projection = nn.Linear(self.state_dim, int(config.xlstm_memory_adapter_rank), bias=False)
        self.up_projection = nn.Linear(int(config.xlstm_memory_adapter_rank), self.hidden_size, bias=False)
        self.beta = nn.Parameter(self._init_beta_parameter(config))
        self.memory_gate = None
        if bool(config.xlstm_memory_gate_enabled):
            self.memory_gate = nn.Linear(self.hidden_size, self.state_dim, bias=True)

    @staticmethod
    def _init_beta_parameter(config):
        """把期望 beta 初值反解到 sigmoid 参数空间。"""
        beta_min, beta_max = config.xlstm_memory_adapter_beta_range
        target_beta = min(max(float(config.xlstm_memory_adapter_beta_init), float(beta_min)), float(beta_max))
        ratio = (target_beta - float(beta_min)) / max(float(beta_max) - float(beta_min), 1e-12)
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        return torch.logit(torch.tensor(ratio, dtype=torch.float32))

    def _apply(self, fn):
        """保持 xLSTM beta 为 FP32，避免低秩记忆注入 scale 在混精中失真。"""
        super()._apply(fn)
        self.beta.data = self.beta.data.float()
        if self.beta.grad is not None:
            self.beta.grad.data = self.beta.grad.data.float()
        return self

    def effective_beta(self):
        """返回 clamp 到配置范围内的实际 memory adapter scale。"""
        beta_min, beta_max = self.config.xlstm_memory_adapter_beta_range
        return torch.sigmoid(self.beta.float()) * (float(beta_max) - float(beta_min)) + float(beta_min)

    def _enabled_for_layer(self):
        """当前层是否启用 xLSTMMemory。"""
        return is_xlstm_memory_enabled_for_layer(self.config, self.layer_index)

    def _boundary_metadata_triggers_reset(self, boundary_metadata):
        """根据外部边界元数据判断是否清零记忆。"""
        if boundary_metadata is None or "boundary_metadata" not in self.config.xlstm_memory_reset_trigger_mode:
            return False, None
        if isinstance(boundary_metadata, dict):
            if boundary_metadata.get("reset"):
                return True, str(boundary_metadata.get("reason") or "boundary_metadata")
            boundary_type = boundary_metadata.get("boundary_type") or boundary_metadata.get("type")
            if boundary_type in self.config.xlstm_memory_reset_boundary_policy:
                return True, f"boundary:{boundary_type}"
        elif isinstance(boundary_metadata, str):
            if boundary_metadata in self.config.xlstm_memory_reset_boundary_policy:
                return True, f"boundary:{boundary_metadata}"
        return False, None

    def _session_event_triggers_reset(self, session_event):
        """根据会话事件判断是否清零记忆。"""
        if session_event is None or "session_event" not in self.config.xlstm_memory_reset_trigger_mode:
            return False, None
        event_text = str(session_event)
        if event_text in {"reset", "session_reset"}:
            return True, f"session_event:{event_text}"
        return False, None

    def _special_token_triggers_reset(self, input_ids):
        """根据特殊 token 判断是否触发记忆边界。"""
        boundary_token_ids = set(self.config.xlstm_memory_boundary_token_ids)
        if (
            input_ids is None
            or not boundary_token_ids
            or "special_token" not in self.config.xlstm_memory_reset_trigger_mode
        ):
            return False, None
        token_ids = input_ids.detach().reshape(-1).cpu().tolist()
        if any(int(token_id) in boundary_token_ids for token_id in token_ids):
            return True, "special_token"
        return False, None

    def _resolve_reset_reason(self, *, boundary_metadata=None, input_ids=None, session_event=None):
        """按会话事件、显式边界、特殊 token 的顺序解析 reset 原因。"""
        for triggered, reason in (
            self._session_event_triggers_reset(session_event),
            self._boundary_metadata_triggers_reset(boundary_metadata),
            self._special_token_triggers_reset(input_ids),
        ):
            if triggered:
                return reason
        return None

    def _apply_decay(self, memory, *, previous_last_decay_token_count, final_token_count):
        """按 token interval 对记忆做指数衰减。"""
        interval = int(self.config.xlstm_memory_state_decay_interval)
        factor = float(self.config.xlstm_memory_state_decay_factor)
        elapsed = max(0, int(final_token_count) - int(previous_last_decay_token_count))
        decay_steps = elapsed // interval
        if decay_steps <= 0:
            return memory, int(previous_last_decay_token_count), 0
        decayed_memory = memory * (factor ** decay_steps)
        return decayed_memory, int(previous_last_decay_token_count) + decay_steps * interval, int(decay_steps)

    def forward(
        self,
        h_ffn,
        layer_state=None,
        attention_mask=None,
        input_ids=None,
        boundary_metadata=None,
        session_event=None,
        request_id=DEFAULT_REQUEST_ID,
    ):
        """更新 FFN 侧外挂记忆，并生成 memory-augmented FFN 输入。"""
        if not self._enabled_for_layer():
            return h_ffn, None
        previous_state = None if layer_state is None else layer_state.xlstm_memory
        memory_input = torch.tanh(self.input_proj(h_ffn))
        if self.memory_gate is not None:
            gate_m = torch.sigmoid(self.memory_gate(h_ffn))
            memory_input = memory_input * gate_m
        if attention_mask is None:
            token_mask = torch.ones(h_ffn.size(0), h_ffn.size(1), device=h_ffn.device, dtype=h_ffn.dtype)
        else:
            token_mask = attention_mask[:, -h_ffn.size(1):].to(device=h_ffn.device, dtype=h_ffn.dtype)

        reset_reason = self._resolve_reset_reason(
            boundary_metadata=boundary_metadata,
            input_ids=input_ids,
            session_event=session_event,
        )
        previous_count = 0 if previous_state is None else int(previous_state.token_count)
        previous_decay_count = 0 if previous_state is None else int(previous_state.decay_count)
        previous_reset_count = 0 if previous_state is None else int(previous_state.reset_count)
        previous_last_decay = 0 if previous_state is None else int(previous_state.last_decay_token_count)
        if previous_state is None or previous_state.memory is None or reset_reason is not None:
            # reset 使用 zero_state，符合方案中“边界污染宁可清空，不做隐式迁移”的约束。
            previous_memory = torch.zeros(
                h_ffn.size(0),
                self.state_dim,
                device=h_ffn.device,
                dtype=h_ffn.dtype,
            )
            history_count = 0
            last_reset_reason = reset_reason
            reset_count = previous_reset_count + (1 if reset_reason is not None else 0)
            previous_last_decay = previous_count
        else:
            previous_memory = previous_state.memory.to(device=h_ffn.device, dtype=h_ffn.dtype)
            history_count = previous_count
            if self.config.xlstm_memory_state_window_size is not None:
                history_count = min(history_count, int(self.config.xlstm_memory_state_window_size))
            last_reset_reason = previous_state.last_reset_reason
            reset_count = previous_reset_count

        prefix_sum = torch.cumsum(memory_input * token_mask.unsqueeze(-1), dim=1)
        active_counts = torch.cumsum(token_mask, dim=1)
        total_counts = active_counts + float(history_count)
        raw_memory_sequence = (
            prefix_sum + previous_memory.unsqueeze(1) * float(history_count)
        ) / total_counts.clamp_min(1.0).unsqueeze(-1)
        # xLSTM 当前实现采用向量化 chunkwise recurrent scan 的近似形式；
        # prefill 期间避免逐 token Python 循环，decode 时由 previous_state 保持连续性。
        memory_sequence = self.state_proj(raw_memory_sequence)

        effective_beta_fp32 = self.effective_beta()
        adapter_delta = self.up_projection(self.down_projection(memory_sequence))
        effective_beta = effective_beta_fp32.to(dtype=h_ffn.dtype)
        if self.config.moe_router_input_mode == "memory_augmented_input":
            x_ffn = h_ffn + effective_beta * adapter_delta
        else:
            # ffn_norm_only_eval 用于消融：状态继续更新，但 Router/experts 不读取 adapter 输出。
            x_ffn = h_ffn

        active_token_count = int(token_mask.sum(dim=1).max().item())
        final_token_count = previous_count + active_token_count
        final_memory = raw_memory_sequence[:, -1]
        final_memory, last_decay_token_count, new_decay_count = self._apply_decay(
            final_memory,
            previous_last_decay_token_count=previous_last_decay,
            final_token_count=final_token_count,
        )
        has_new_tokens = token_mask.sum(dim=1).gt(0)
        if previous_state is not None and previous_state.memory is not None and reset_reason is None:
            fallback_memory = previous_state.memory.to(device=h_ffn.device, dtype=h_ffn.dtype)
        else:
            fallback_memory = torch.zeros_like(final_memory)
        final_memory = torch.where(has_new_tokens[:, None], final_memory, fallback_memory)

        memory_norm = float(final_memory.float().norm(dim=-1).mean().detach().cpu())
        adapter_delta_norm = float(adapter_delta.float().norm(dim=-1).mean().detach().cpu())
        previous_count = 0 if previous_state is None else int(previous_state.token_count)
        new_state = xLSTMMemoryState(
            request_id=request_id,
            layer_index=self.layer_index,
            memory=final_memory.detach(),
            token_count=final_token_count,
            last_decay_token_count=last_decay_token_count,
            decay_count=previous_decay_count + new_decay_count,
            reset_count=reset_count,
            last_reset_reason=last_reset_reason,
            effective_beta=float(effective_beta_fp32.detach().cpu()),
            memory_norm=memory_norm,
            adapter_delta_norm=adapter_delta_norm,
        )
        return x_ffn, new_state


class LPTBlockV2(nn.Module):
    """LPT v2 Decoder block。"""

    def __init__(
        self,
        config,
        layer_index,
        retnet_assist,
        q_adapter,
        context_adapter,
        paged_kv_cache,
        *,
        retnet_state_slot=None,
    ):
        super().__init__()
        self.layer_index = int(layer_index)
        self.sequence_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attention_mixer = LocalAttentionMixerV2(
            config,
            layer_index,
            retnet_assist,
            q_adapter,
            context_adapter,
            paged_kv_cache,
            retnet_state_slot=retnet_state_slot,
        )
        self.xlstm_memory = xLSTMMemoryAssist(config, layer_index)
        self.feed_forward = SwiGLUMoE(config, layer_index)

    def forward(
        self,
        x,
        position_ids,
        rope_cache,
        input_ids=None,
        attention_mask=None,
        segment_ids=None,
        memory_boundary_metadata=None,
        session_event=None,
        layer_state=None,
        request_id=DEFAULT_REQUEST_ID,
        use_kv_cache=True,
    ):
        """执行一个 LPT v2 block：Attention-First，再进入记忆增强 MoE FFN。"""
        layer_state = _as_layer_state_v2(layer_state)
        attn_out, layer_state = self.attention_mixer(
            self.sequence_norm(x),
            position_ids=position_ids,
            rope_cache=rope_cache,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            layer_state=layer_state,
            request_id=request_id,
            use_kv_cache=use_kv_cache,
        )
        x = x + attn_out
        h_ffn = self.ffn_norm(x)
        # xLSTMMemory 读取 FFN 前归一化特征，只影响 FFN 输入，不进入 Attention/Paged KV。
        x_ffn, xlstm_state = self.xlstm_memory(
            h_ffn,
            layer_state=layer_state,
            attention_mask=attention_mask,
            input_ids=input_ids,
            boundary_metadata=memory_boundary_metadata,
            session_event=session_event,
            request_id=request_id,
        )
        ffn_out, moe_state = self.feed_forward(x_ffn, request_id=request_id)
        x = x + ffn_out
        layer_state = replace(layer_state, moe=moe_state, xlstm_memory=xlstm_state)
        return x, layer_state


class LPTV2(nn.Module):
    """LPT v2 完整模型。"""

    def __init__(self, vocabulary_size, config=None):
        super().__init__()
        self.config = normalize_model_config(config)
        if self.config.architecture_version != LPT_V2_ARCHITECTURE_VERSION:
            raise ValueError("LPTV2 只接受 architecture_version='lpt_v2' 的 ModelConfig。")
        self.vocabulary_size = int(vocabulary_size)
        self.token_embedding = nn.Embedding(self.vocabulary_size, self.config.hidden_size)
        self.paged_kv_cache = PagedKVCache(
            page_block_size=self.config.page_block_size,
            attention_window_size=self.config.attention_window_size,
        )
        self.retnet_layer_to_state_slot = _retnet_layer_to_state_slots(self.config)
        self.retnet_state_pool = RetNetAssistStatePool(
            self.config.num_layers,
            layer_to_state_slot=self.retnet_layer_to_state_slot,
            state_slot_count=_retnet_state_slot_count(self.config),
        )
        self.xlstm_memory_state_pool = xLSTMMemoryStatePool(self.config.num_layers)
        enabled_parameter_slots = {
            int(slot)
            for layer_index in range(int(self.config.num_layers))
            for slot in (_retnet_parameter_slot_for_layer(self.config, layer_index),)
            if slot is not None
        }
        # RetNet 参数共享与状态共享是两个独立维度：parameter_slot 决定模块复用，
        # state_slot 决定 request-bound 摘要复用。这里先按参数槽位构造共享模块。
        self.shared_retnet_assist = (
            SharedRetNetAssist(self.config)
            if 0 in enabled_parameter_slots
            else None
        )
        retnet_assist_by_slot = (
            {0: self.shared_retnet_assist}
            if self.shared_retnet_assist is not None
            else {}
        )
        q_adapter_by_slot = {}
        context_adapter_by_slot = {}

        def retnet_assist_for(parameter_slot):
            if parameter_slot not in retnet_assist_by_slot:
                retnet_assist_by_slot[parameter_slot] = SharedRetNetAssist(self.config)
            return retnet_assist_by_slot[parameter_slot]

        def q_adapter_for(parameter_slot):
            if parameter_slot not in q_adapter_by_slot:
                q_adapter_by_slot[parameter_slot] = QOnlyRetNetAdapter(self.config)
            return q_adapter_by_slot[parameter_slot]

        def context_adapter_for(parameter_slot):
            if parameter_slot not in context_adapter_by_slot:
                context_adapter_by_slot[parameter_slot] = RetNetContextAdapter(self.config)
            return context_adapter_by_slot[parameter_slot]

        def retnet_modules_for(layer_index):
            slot = _retnet_parameter_slot_for_layer(self.config, layer_index)
            if slot is None:
                return None, None, None
            context_adapter = (
                context_adapter_for(int(slot))
                if bool(self.config.retnet_context_adapter_enabled)
                else None
            )
            return retnet_assist_for(int(slot)), q_adapter_for(int(slot)), context_adapter

        self.layers = nn.ModuleList([
            LPTBlockV2(
                self.config,
                layer_index,
                *retnet_modules_for(layer_index),
                self.paged_kv_cache,
                retnet_state_slot=self.retnet_layer_to_state_slot[layer_index],
            )
            for layer_index in range(self.config.num_layers)
        ])
        self.final_norm = RMSNorm(self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, self.vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self._rope_caches = nn.ModuleDict()
        self.to(dtype=GlobalConfig.parameter_dtype)

    @property
    def num_state_slots(self):
        """对外暴露每层 LayerStateV2 槽位数量。"""
        return self.config.num_layers

    @property
    def retnet_enabled_layer_count(self):
        """返回实际启用 RetNetAssist 的层数，供参数统计和报告使用。"""
        return count_retnet_assist_enabled_layers(self.config)

    def _build_rope_cache(self, max_seq_len, embedding_mode):
        """按训练/推理 scope 创建 LongRoPE2 rotary cache。"""
        return build_rotary_position_encoding(
            config=self.config,
            max_seq_len=max_seq_len,
            embedding_mode=embedding_mode,
        )

    def get_rope_cache(self, scope="inference"):
        """按 scope 缓存 RoPE 表，避免每个 forward 重建位置编码。"""
        if scope == "train":
            max_seq_len = int(GlobalConfig.train_rope_cache_max_sequence_length)
            embedding_mode = self.config.longrope2_train_embedding_mode
        elif scope == "inference":
            max_seq_len = int(GlobalConfig.inference_rope_cache_max_sequence_length)
            embedding_mode = self.config.longrope2_inference_embedding_mode
        else:
            raise ValueError(f"未支持的 rope_cache_scope: {scope}")
        cache_key = f"{scope}_{embedding_mode}_{max_seq_len}"
        if cache_key not in self._rope_caches:
            rope_cache = self._build_rope_cache(max_seq_len, embedding_mode)
            rope_cache.to(device=self.token_embedding.weight.device, dtype=self.token_embedding.weight.dtype)
            self._rope_caches[cache_key] = rope_cache
        return self._rope_caches[cache_key]

    def reset_request_state(self, request_id=DEFAULT_REQUEST_ID):
        """释放指定 request 的 Paged KV；Assist 状态由专用 release 方法管理。"""
        self.paged_kv_cache.reset(request_id=request_id)

    def release_retnet_assist_state(self, request_id=DEFAULT_REQUEST_ID, reason="request_finished"):
        """显式释放 request-bound RetNetAssist 状态池条目。"""
        return self.retnet_state_pool.release(request_id, reason=reason)

    def release_xlstm_memory_state(self, request_id=DEFAULT_REQUEST_ID, reason="request_finished"):
        """显式释放 request-bound xLSTMMemory 状态池条目。"""
        return self.xlstm_memory_state_pool.release(request_id, reason=reason)

    def _normalize_incoming_layer_states(self, layer_states=None):
        """把外部 layer_states 补齐为每层一个状态槽。"""
        if layer_states is None:
            return [None] * self.config.num_layers
        if len(layer_states) != self.config.num_layers:
            raise ValueError("LPTV2 layer_states 数量必须等于 num_layers。")
        return list(layer_states)

    def _collect_retnet_previous_states(self, layer_states):
        """按 state_slot 收集上一轮 RetNet 状态，支持 group/per-layer 状态共享。"""
        states_by_slot = {}
        for layer_index, layer_state in enumerate(layer_states):
            if not _retnet_enabled_for_layer(self.config, layer_index):
                continue
            if layer_state is None or layer_state.retnet_assist is None:
                continue
            state_slot = self.retnet_layer_to_state_slot[layer_index]
            states_by_slot[state_slot] = layer_state.retnet_assist
        return states_by_slot

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        segment_ids=None,
        memory_boundary_metadata=None,
        session_event=None,
        layer_states=None,
        rope_cache_scope="inference",
        request_id=DEFAULT_REQUEST_ID,
        use_kv_cache=True,
    ):
        """执行模型前向，返回 logits 与每层新的 LayerStateV2。"""
        embedding_device = self.token_embedding.weight.device
        input_ids = input_ids.to(device=embedding_device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=embedding_device)
        else:
            attention_mask = attention_mask.to(device=embedding_device, dtype=torch.long)
        if position_ids is None:
            position_ids = build_position_ids(attention_mask)[:, -input_ids.size(1):]
        else:
            position_ids = position_ids.to(device=embedding_device, dtype=torch.long)
        if segment_ids is not None:
            segment_ids = segment_ids.to(device=embedding_device, dtype=torch.long)

        rope_cache = self.get_rope_cache(rope_cache_scope)
        hidden_states = self.token_embedding(input_ids)
        previous_states = self._normalize_incoming_layer_states(layer_states)
        previous_retnet_states = self._collect_retnet_previous_states(previous_states)
        new_states = []
        for layer_index, (layer, layer_state) in enumerate(zip(self.layers, previous_states)):
            layer_state = _as_layer_state_v2(layer_state)
            if _retnet_enabled_for_layer(self.config, layer_index):
                state_slot = self.retnet_layer_to_state_slot[layer_index]
                if state_slot in previous_retnet_states:
                    # 同一个 RetNet state_slot 可能被多个层共享；传入 block 前只改 layer_index，
                    # 状态张量本身不复制，避免 group/per-layer 策略分叉时语义漂移。
                    layer_state = replace(
                        layer_state,
                        retnet_assist=replace(previous_retnet_states[state_slot], layer_index=layer_index),
                    )
            layer_device = next(layer.parameters()).device
            if hidden_states.device != layer_device:
                # execution plan 可能把不同 block 放到不同设备；forward 在层边界显式迁移。
                hidden_states = hidden_states.to(device=layer_device)
            layer_input_ids = input_ids.to(device=layer_device)
            layer_position_ids = position_ids.to(device=layer_device)
            layer_attention_mask = attention_mask.to(device=layer_device, dtype=torch.long)
            layer_segment_ids = _move_optional_tensor(segment_ids, layer_device, dtype=torch.long)
            if rope_cache.original_embedding.rescale_factors.device != layer_device:
                rope_cache.to(device=layer_device, dtype=hidden_states.dtype)
            hidden_states, new_layer_state = layer(
                hidden_states,
                input_ids=layer_input_ids,
                position_ids=layer_position_ids,
                rope_cache=rope_cache,
                attention_mask=layer_attention_mask,
                segment_ids=layer_segment_ids,
                memory_boundary_metadata=memory_boundary_metadata,
                session_event=session_event,
                layer_state=layer_state,
                request_id=request_id,
                use_kv_cache=use_kv_cache,
            )
            new_states.append(new_layer_state)

        final_norm_device = next(self.final_norm.parameters()).device
        if hidden_states.device != final_norm_device:
            hidden_states = hidden_states.to(device=final_norm_device)
        normalized_states = self.final_norm(hidden_states)
        lm_head_device = self.lm_head.weight.device
        if normalized_states.device != lm_head_device:
            normalized_states = normalized_states.to(device=lm_head_device)
        logits = self.lm_head(normalized_states)
        return logits, tuple(new_states)

    def prefill(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        segment_ids=None,
        memory_boundary_metadata=None,
        session_event=None,
        rope_cache_scope="inference",
        request_id=DEFAULT_REQUEST_ID,
    ):
        """执行 request prefill，并把 RetNetAssist 状态写入状态池。"""
        logits, layer_states = self.forward(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            memory_boundary_metadata=memory_boundary_metadata,
            session_event=session_event,
            layer_states=None,
            rope_cache_scope=rope_cache_scope,
            request_id=request_id,
        )
        self.retnet_state_pool.update_from_layer_states(
            request_id,
            layer_states,
            phase="prefill",
        )
        self.xlstm_memory_state_pool.update_from_layer_states(
            request_id,
            layer_states,
            phase="prefill",
        )
        return logits, layer_states

    def decode(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        segment_ids=None,
        memory_boundary_metadata=None,
        session_event=None,
        layer_states=None,
        rope_cache_scope="inference",
        request_id=DEFAULT_REQUEST_ID,
    ):
        """执行 request decode，并从 RetNetAssist 状态池续接辅助状态。"""
        pooled_layer_states = self.retnet_state_pool.bind_to_layer_states(
            request_id,
            layer_states=layer_states,
        )
        pooled_layer_states = self.xlstm_memory_state_pool.bind_to_layer_states(
            request_id,
            layer_states=pooled_layer_states,
        )
        logits, new_layer_states = self.forward(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            memory_boundary_metadata=memory_boundary_metadata,
            session_event=session_event,
            layer_states=pooled_layer_states,
            rope_cache_scope=rope_cache_scope,
            request_id=request_id,
        )
        self.retnet_state_pool.update_from_layer_states(
            request_id,
            new_layer_states,
            phase="decode",
        )
        self.xlstm_memory_state_pool.update_from_layer_states(
            request_id,
            new_layer_states,
            phase="decode",
        )
        return logits, new_layer_states
