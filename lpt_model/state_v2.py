"""LPT v2 请求级状态骨架。

本模块只定义状态边界与生命周期元数据，不承载具体算子实现。
Paged KV、RetNetAssist 与 xLSTMAssist 必须物理隔离，避免缓存裁剪误伤辅助状态。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


ATTENTION_LAYER_STATE_V2_TYPE = "attention_layer_state_v2"
RETNET_ASSIST_STATE_TYPE = "retnet_assist_state"
MOE_LAYER_STATE_TYPE = "moe_layer_state"
XLSTM_MEMORY_STATE_TYPE = "xlstm_memory_state"


def _normalize_request_id(request_id):
    request_id_text = str(request_id)
    if not request_id_text:
        raise ValueError("request_id 不能为空。")
    return request_id_text


@dataclass(frozen=True)
class StateReleaseMetadata:
    """request-bound 状态释放元数据。"""

    request_id: str
    released: bool = False
    release_reason: str | None = None
    released_at_token_count: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        if self.released_at_token_count is not None:
            object.__setattr__(
                self,
                "released_at_token_count",
                int(self.released_at_token_count),
            )

    def mark_released(self, reason, token_count=None):
        """标记为显式释放；Paged KV 裁剪不应调用这个入口。"""
        return replace(
            self,
            released=True,
            release_reason=str(reason),
            released_at_token_count=None if token_count is None else int(token_count),
        )

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "released": self.released,
            "release_reason": self.release_reason,
            "released_at_token_count": self.released_at_token_count,
        }


@dataclass(frozen=True)
class PagedKVReference:
    """Paged KV 页表引用，只保存 Attention 局部窗口真实 token 的 K/V。"""

    request_id: str
    layer_index: int
    page_ids: tuple[int, ...] = ()
    token_count: int = 0
    window_token_count: int = 0

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        object.__setattr__(self, "layer_index", int(self.layer_index))
        object.__setattr__(self, "page_ids", tuple(int(page_id) for page_id in self.page_ids))
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(self, "window_token_count", int(self.window_token_count))
        if self.token_count < 0 or self.window_token_count < 0:
            raise ValueError("token_count 与 window_token_count 不能为负数。")

    def trim(self, kept_page_ids, *, token_count=None, window_token_count=None):
        """返回裁剪后的 Paged KV 引用，不触碰任何 Assist state。"""
        return replace(
            self,
            page_ids=tuple(int(page_id) for page_id in kept_page_ids),
            token_count=self.token_count if token_count is None else int(token_count),
            window_token_count=(
                self.window_token_count
                if window_token_count is None
                else int(window_token_count)
            ),
        )

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "page_ids": list(self.page_ids),
            "token_count": self.token_count,
            "window_token_count": self.window_token_count,
        }


@dataclass(frozen=True)
class AttentionLayerState:
    """Attention v2 状态，只持有 Paged KV 引用或 dense KV 兼容槽。"""

    request_id: str
    layer_index: int
    paged_kv_ref: PagedKVReference | None = None
    dense_kv_state: tuple[Any, ...] = ()
    state_type: str = ATTENTION_LAYER_STATE_V2_TYPE

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        object.__setattr__(self, "layer_index", int(self.layer_index))
        object.__setattr__(self, "dense_kv_state", tuple(self.dense_kv_state))
        if self.paged_kv_ref is not None:
            if self.paged_kv_ref.request_id != self.request_id:
                raise ValueError("Paged KV 引用的 request_id 必须与 AttentionLayerState 一致。")
            if self.paged_kv_ref.layer_index != self.layer_index:
                raise ValueError("Paged KV 引用的 layer_index 必须与 AttentionLayerState 一致。")

    def trim_paged_kv(self, kept_page_ids, *, token_count=None, window_token_count=None):
        if self.paged_kv_ref is None:
            return self
        return replace(
            self,
            paged_kv_ref=self.paged_kv_ref.trim(
                kept_page_ids,
                token_count=token_count,
                window_token_count=window_token_count,
            ),
        )

    def to_dict(self):
        return {
            "state_type": self.state_type,
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "paged_kv_ref": None if self.paged_kv_ref is None else self.paged_kv_ref.to_dict(),
            "dense_kv_tensor_count": len(self.dense_kv_state),
        }


@dataclass(frozen=True)
class RetNetAssistState:
    """RetNetAssist 全局摘要状态，独立于 Attention KV 生命周期。"""

    request_id: str
    layer_index: int
    state_slot: int | None = None
    summary: Any = None
    token_count: int = 0
    summary_norm: float | None = None
    q_adapter_delta_norm: float | None = None
    k_adapter_delta_norm: float | None = None
    alpha_q: float | None = None
    alpha_k: float | None = None
    release_metadata: StateReleaseMetadata | None = None
    state_type: str = RETNET_ASSIST_STATE_TYPE

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        object.__setattr__(self, "layer_index", int(self.layer_index))
        state_slot = self.layer_index if self.state_slot is None else int(self.state_slot)
        if state_slot < 0:
            raise ValueError("state_slot 不能为负数。")
        object.__setattr__(self, "state_slot", state_slot)
        object.__setattr__(self, "token_count", int(self.token_count))
        if self.token_count < 0:
            raise ValueError("token_count 不能为负数。")
        for field_name in (
            "summary_norm",
            "q_adapter_delta_norm",
            "k_adapter_delta_norm",
            "alpha_q",
            "alpha_k",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, float(value))
        release_metadata = self.release_metadata or StateReleaseMetadata(self.request_id)
        if release_metadata.request_id != self.request_id:
            raise ValueError("release_metadata 的 request_id 必须与 RetNetAssistState 一致。")
        object.__setattr__(self, "release_metadata", release_metadata)

    def mark_released(self, reason, token_count=None):
        return replace(
            self,
            release_metadata=self.release_metadata.mark_released(reason, token_count),
        )

    def to_dict(self):
        return {
            "state_type": self.state_type,
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "state_slot": self.state_slot,
            "token_count": self.token_count,
            "summary_norm": self.summary_norm,
            "q_adapter_delta_norm": self.q_adapter_delta_norm,
            "k_adapter_delta_norm": self.k_adapter_delta_norm,
            "alpha_q": self.alpha_q,
            "alpha_k": self.alpha_k,
            "release_metadata": self.release_metadata.to_dict(),
        }


@dataclass(frozen=True)
class MoELayerState:
    """MoE 路由观测状态；专家本身保持无状态。"""

    request_id: str
    layer_index: int
    expert_token_counts: tuple[int, ...] = ()
    router_entropy: float | None = None
    load_balance_loss: float | None = None
    router_z_loss: float | None = None
    state_type: str = MOE_LAYER_STATE_TYPE

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        object.__setattr__(self, "layer_index", int(self.layer_index))
        object.__setattr__(
            self,
            "expert_token_counts",
            tuple(int(value) for value in self.expert_token_counts),
        )

    def to_dict(self):
        return {
            "state_type": self.state_type,
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "expert_token_counts": list(self.expert_token_counts),
            "router_entropy": self.router_entropy,
            "load_balance_loss": self.load_balance_loss,
            "router_z_loss": self.router_z_loss,
        }


@dataclass(frozen=True)
class xLSTMMemoryState:
    """xLSTMAssist FFN 侧记忆状态，独立于 Attention 与 RetNetAssist。"""

    request_id: str
    layer_index: int
    memory: Any = None
    token_count: int = 0
    last_decay_token_count: int = 0
    decay_count: int = 0
    reset_count: int = 0
    last_reset_reason: str | None = None
    effective_beta: float | None = None
    memory_norm: float | None = None
    adapter_delta_norm: float | None = None
    release_metadata: StateReleaseMetadata | None = None
    state_type: str = XLSTM_MEMORY_STATE_TYPE

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        object.__setattr__(self, "layer_index", int(self.layer_index))
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(self, "last_decay_token_count", int(self.last_decay_token_count))
        object.__setattr__(self, "decay_count", int(self.decay_count))
        object.__setattr__(self, "reset_count", int(self.reset_count))
        if self.token_count < 0 or self.last_decay_token_count < 0 or self.decay_count < 0 or self.reset_count < 0:
            raise ValueError("token_count、last_decay_token_count、decay_count 与 reset_count 不能为负数。")
        for field_name in ("effective_beta", "memory_norm", "adapter_delta_norm"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, float(value))
        release_metadata = self.release_metadata or StateReleaseMetadata(self.request_id)
        if release_metadata.request_id != self.request_id:
            raise ValueError("release_metadata 的 request_id 必须与 xLSTMMemoryState 一致。")
        object.__setattr__(self, "release_metadata", release_metadata)

    def mark_released(self, reason, token_count=None):
        return replace(
            self,
            release_metadata=self.release_metadata.mark_released(reason, token_count),
        )

    def to_dict(self):
        return {
            "state_type": self.state_type,
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "token_count": self.token_count,
            "last_decay_token_count": self.last_decay_token_count,
            "decay_count": self.decay_count,
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "effective_beta": self.effective_beta,
            "memory_norm": self.memory_norm,
            "adapter_delta_norm": self.adapter_delta_norm,
            "release_metadata": self.release_metadata.to_dict(),
        }


@dataclass(frozen=True)
class LayerStateV2:
    """单层 v2 状态组合，四类状态互相隔离。"""

    attention: AttentionLayerState | None = None
    retnet_assist: RetNetAssistState | None = None
    moe: MoELayerState | None = None
    xlstm_memory: xLSTMMemoryState | None = None

    def __post_init__(self):
        states = tuple(
            state
            for state in (
                self.attention,
                self.retnet_assist,
                self.moe,
                self.xlstm_memory,
            )
            if state is not None
        )
        if not states:
            return

        request_ids = {state.request_id for state in states}
        if len(request_ids) != 1:
            raise ValueError("LayerStateV2 内所有状态必须绑定同一个 request_id。")
        layer_indices = {state.layer_index for state in states}
        if len(layer_indices) != 1:
            raise ValueError("LayerStateV2 内所有状态必须绑定同一个 layer_index。")

    @property
    def request_id(self):
        for state in (self.attention, self.retnet_assist, self.moe, self.xlstm_memory):
            if state is not None:
                return state.request_id
        return None

    @property
    def layer_index(self):
        for state in (self.attention, self.retnet_assist, self.moe, self.xlstm_memory):
            if state is not None:
                return state.layer_index
        return None

    def trim_paged_kv(self, kept_page_ids, *, token_count=None, window_token_count=None):
        """只裁剪 Attention Paged KV；Assist 状态对象保持原样。"""
        if self.attention is None:
            return self
        return replace(
            self,
            attention=self.attention.trim_paged_kv(
                kept_page_ids,
                token_count=token_count,
                window_token_count=window_token_count,
            ),
        )

    def release_assist_states(self, reason, token_count=None):
        """显式释放 Assist 状态；Attention Paged KV 生命周期由页池单独负责。"""
        return replace(
            self,
            retnet_assist=(
                None
                if self.retnet_assist is None
                else self.retnet_assist.mark_released(reason, token_count)
            ),
            xlstm_memory=(
                None
                if self.xlstm_memory is None
                else self.xlstm_memory.mark_released(reason, token_count)
            ),
        )

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "attention": None if self.attention is None else self.attention.to_dict(),
            "retnet_assist": None if self.retnet_assist is None else self.retnet_assist.to_dict(),
            "moe": None if self.moe is None else self.moe.to_dict(),
            "xlstm_memory": None if self.xlstm_memory is None else self.xlstm_memory.to_dict(),
        }
