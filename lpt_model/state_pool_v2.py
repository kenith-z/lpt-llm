"""LPT v2 request-bound RetNetAssist 状态池。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import time

from .state_v2 import LayerStateV2, RetNetAssistState, xLSTMMemoryState


RETNET_POOL_PHASE_PREFILL = "prefill"
RETNET_POOL_PHASE_DECODE = "decode"
RETNET_POOL_PHASE_PREEMPTED = "preempted"
RETNET_POOL_PHASE_RELEASED = "released"
RETNET_POOL_PHASE_RESET = "reset"
SUPPORTED_RETNET_POOL_PHASES = (
    RETNET_POOL_PHASE_PREFILL,
    RETNET_POOL_PHASE_DECODE,
    RETNET_POOL_PHASE_PREEMPTED,
    RETNET_POOL_PHASE_RELEASED,
    RETNET_POOL_PHASE_RESET,
)

XLSTM_POOL_PHASE_PREFILL = "prefill"
XLSTM_POOL_PHASE_DECODE = "decode"
XLSTM_POOL_PHASE_PREEMPTED = "preempted"
XLSTM_POOL_PHASE_RELEASED = "released"
XLSTM_POOL_PHASE_RESET = "reset"
SUPPORTED_XLSTM_POOL_PHASES = (
    XLSTM_POOL_PHASE_PREFILL,
    XLSTM_POOL_PHASE_DECODE,
    XLSTM_POOL_PHASE_PREEMPTED,
    XLSTM_POOL_PHASE_RELEASED,
    XLSTM_POOL_PHASE_RESET,
)


def _normalize_request_id(request_id):
    request_id_text = str(request_id)
    if not request_id_text:
        raise ValueError("request_id 不能为空。")
    return request_id_text


@dataclass(frozen=True)
class RetNetAssistPoolMetadata:
    """单个 request 在 RetNetAssist 状态池中的生命周期元数据。"""

    request_id: str
    phase: str
    token_count: int = 0
    state_count: int = 0
    update_count: int = 0
    preempt_count: int = 0
    released: bool = False
    release_reason: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        if self.phase not in SUPPORTED_RETNET_POOL_PHASES:
            raise ValueError(f"未知 RetNetAssist 状态池阶段: {self.phase}")
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(self, "state_count", int(self.state_count))
        object.__setattr__(self, "update_count", int(self.update_count))
        object.__setattr__(self, "preempt_count", int(self.preempt_count))
        object.__setattr__(self, "released", bool(self.released))
        object.__setattr__(self, "created_at", float(self.created_at))
        object.__setattr__(self, "updated_at", float(self.updated_at))

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "token_count": self.token_count,
            "state_count": self.state_count,
            "update_count": self.update_count,
            "preempt_count": self.preempt_count,
            "released": self.released,
            "release_reason": self.release_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class RetNetAssistStatePool:
    """按 request_id 隔离的 RetNetAssist 状态池。

    Paged KV 裁剪和释放不经过本池；request 结束时应显式调用 release/reset。
    """

    def __init__(self, num_layers):
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须为正整数。")
        self._states: dict[tuple[str, int], RetNetAssistState] = {}
        self._metadata: dict[str, RetNetAssistPoolMetadata] = {}

    @property
    def request_count(self):
        return len(self._metadata)

    @property
    def state_count(self):
        return len(self._states)

    def _metadata_for(self, request_id):
        request_id = _normalize_request_id(request_id)
        now = time()
        metadata = self._metadata.get(request_id)
        if metadata is not None:
            return metadata
        metadata = RetNetAssistPoolMetadata(
            request_id=request_id,
            phase=RETNET_POOL_PHASE_RESET,
            created_at=now,
            updated_at=now,
        )
        self._metadata[request_id] = metadata
        return metadata

    def _set_metadata(self, request_id, *, phase, token_count=None, state_count=None, released=False, release_reason=None):
        previous = self._metadata_for(request_id)
        now = time()
        metadata = replace(
            previous,
            phase=phase,
            token_count=previous.token_count if token_count is None else int(token_count),
            state_count=previous.state_count if state_count is None else int(state_count),
            update_count=previous.update_count + 1,
            released=bool(released),
            release_reason=release_reason,
            updated_at=now,
        )
        self._metadata[previous.request_id] = metadata
        return metadata

    def update_from_layer_states(self, request_id, layer_states, *, phase):
        """从 LayerStateV2 序列提取 RetNetAssistState 并写入状态池。"""
        request_id = _normalize_request_id(request_id)
        if phase not in {RETNET_POOL_PHASE_PREFILL, RETNET_POOL_PHASE_DECODE}:
            raise ValueError("update_from_layer_states 的 phase 必须是 prefill 或 decode。")
        if len(layer_states) != self.num_layers:
            raise ValueError("layer_states 数量必须等于状态池 num_layers。")

        max_token_count = 0
        state_count = 0
        for layer_index, layer_state in enumerate(layer_states):
            if layer_state is None or layer_state.retnet_assist is None:
                continue
            retnet_state = layer_state.retnet_assist
            if retnet_state.request_id != request_id:
                raise ValueError("RetNetAssistState request_id 与状态池更新 request_id 不一致。")
            if int(retnet_state.layer_index) != layer_index:
                raise ValueError("RetNetAssistState layer_index 与所在层不一致。")
            self._states[(request_id, layer_index)] = retnet_state
            max_token_count = max(max_token_count, int(retnet_state.token_count))
            state_count += 1

        return self._set_metadata(
            request_id,
            phase=phase,
            token_count=max_token_count,
            state_count=state_count,
        )

    def get(self, request_id, layer_index):
        request_id = _normalize_request_id(request_id)
        layer_index = int(layer_index)
        return self._states.get((request_id, layer_index))

    def get_request_states(self, request_id):
        request_id = _normalize_request_id(request_id)
        return tuple(
            self._states[(request_id, layer_index)]
            for layer_index in range(self.num_layers)
            if (request_id, layer_index) in self._states
        )

    def bind_to_layer_states(self, request_id, layer_states=None):
        """把池中 RetNetAssist 状态合并进 LayerStateV2，保留其它状态槽。"""
        request_id = _normalize_request_id(request_id)
        if layer_states is None:
            normalized_states = [LayerStateV2() for _ in range(self.num_layers)]
        else:
            if len(layer_states) != self.num_layers:
                raise ValueError("layer_states 数量必须等于状态池 num_layers。")
            normalized_states = [
                LayerStateV2() if layer_state is None else layer_state
                for layer_state in layer_states
            ]

        merged_states = []
        for layer_index, layer_state in enumerate(normalized_states):
            pooled_state = self._states.get((request_id, layer_index))
            if pooled_state is None:
                merged_states.append(layer_state)
            else:
                merged_states.append(replace(layer_state, retnet_assist=pooled_state))
        return tuple(merged_states)

    def mark_preempted(self, request_id):
        """标记 request 被抢占；状态保留在池内，后续 decode 可继续使用。"""
        request_id = _normalize_request_id(request_id)
        previous = self._metadata_for(request_id)
        metadata = replace(
            previous,
            phase=RETNET_POOL_PHASE_PREEMPTED,
            preempt_count=previous.preempt_count + 1,
            updated_at=time(),
        )
        self._metadata[request_id] = metadata
        return metadata

    def release(self, request_id, reason="request_finished"):
        """释放指定 request 的 RetNetAssist 状态，并返回释放前的状态。"""
        request_id = _normalize_request_id(request_id)
        released_states = []
        for key, state in list(self._states.items()):
            if key[0] != request_id:
                continue
            released_states.append(state.mark_released(reason, state.token_count))
            self._states.pop(key, None)
        self._set_metadata(
            request_id,
            phase=RETNET_POOL_PHASE_RELEASED,
            state_count=0,
            released=True,
            release_reason=str(reason),
        )
        return tuple(released_states)

    def reset(self, request_id=None):
        """重置状态池；指定 request_id 时只清理该请求。"""
        if request_id is None:
            self._states.clear()
            self._metadata.clear()
            return

        request_id = _normalize_request_id(request_id)
        for key in list(self._states):
            if key[0] == request_id:
                self._states.pop(key, None)
        self._set_metadata(
            request_id,
            phase=RETNET_POOL_PHASE_RESET,
            state_count=0,
            released=False,
            release_reason=None,
        )

    def to_runtime_metadata(self):
        return {
            "num_layers": self.num_layers,
            "request_count": self.request_count,
            "state_count": self.state_count,
            "requests": {
                request_id: metadata.to_dict()
                for request_id, metadata in sorted(self._metadata.items())
            },
        }


@dataclass(frozen=True)
class xLSTMMemoryPoolMetadata:
    """单个 request 在 xLSTMMemory 状态池中的生命周期元数据。"""

    request_id: str
    phase: str
    token_count: int = 0
    state_count: int = 0
    update_count: int = 0
    preempt_count: int = 0
    decay_count: int = 0
    reset_count: int = 0
    released: bool = False
    release_reason: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "request_id", _normalize_request_id(self.request_id))
        if self.phase not in SUPPORTED_XLSTM_POOL_PHASES:
            raise ValueError(f"未知 xLSTMMemory 状态池阶段: {self.phase}")
        object.__setattr__(self, "token_count", int(self.token_count))
        object.__setattr__(self, "state_count", int(self.state_count))
        object.__setattr__(self, "update_count", int(self.update_count))
        object.__setattr__(self, "preempt_count", int(self.preempt_count))
        object.__setattr__(self, "decay_count", int(self.decay_count))
        object.__setattr__(self, "reset_count", int(self.reset_count))
        object.__setattr__(self, "released", bool(self.released))
        object.__setattr__(self, "created_at", float(self.created_at))
        object.__setattr__(self, "updated_at", float(self.updated_at))

    def to_dict(self):
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "token_count": self.token_count,
            "state_count": self.state_count,
            "update_count": self.update_count,
            "preempt_count": self.preempt_count,
            "decay_count": self.decay_count,
            "reset_count": self.reset_count,
            "released": self.released,
            "release_reason": self.release_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class xLSTMMemoryStatePool:
    """按 request_id 隔离的 xLSTMMemory 状态池。"""

    def __init__(self, num_layers):
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers 必须为正整数。")
        self._states: dict[tuple[str, int], xLSTMMemoryState] = {}
        self._metadata: dict[str, xLSTMMemoryPoolMetadata] = {}

    @property
    def request_count(self):
        return len(self._metadata)

    @property
    def state_count(self):
        return len(self._states)

    def _metadata_for(self, request_id):
        request_id = _normalize_request_id(request_id)
        now = time()
        metadata = self._metadata.get(request_id)
        if metadata is not None:
            return metadata
        metadata = xLSTMMemoryPoolMetadata(
            request_id=request_id,
            phase=XLSTM_POOL_PHASE_RESET,
            created_at=now,
            updated_at=now,
        )
        self._metadata[request_id] = metadata
        return metadata

    def _set_metadata(
        self,
        request_id,
        *,
        phase,
        token_count=None,
        state_count=None,
        decay_count=None,
        reset_count=None,
        released=False,
        release_reason=None,
    ):
        previous = self._metadata_for(request_id)
        metadata = replace(
            previous,
            phase=phase,
            token_count=previous.token_count if token_count is None else int(token_count),
            state_count=previous.state_count if state_count is None else int(state_count),
            update_count=previous.update_count + 1,
            decay_count=previous.decay_count if decay_count is None else int(decay_count),
            reset_count=previous.reset_count if reset_count is None else int(reset_count),
            released=bool(released),
            release_reason=release_reason,
            updated_at=time(),
        )
        self._metadata[previous.request_id] = metadata
        return metadata

    def update_from_layer_states(self, request_id, layer_states, *, phase):
        """从 LayerStateV2 序列提取 xLSTMMemoryState 并写入状态池。"""
        request_id = _normalize_request_id(request_id)
        if phase not in {XLSTM_POOL_PHASE_PREFILL, XLSTM_POOL_PHASE_DECODE}:
            raise ValueError("update_from_layer_states 的 phase 必须是 prefill 或 decode。")
        if len(layer_states) != self.num_layers:
            raise ValueError("layer_states 数量必须等于状态池 num_layers。")

        max_token_count = 0
        state_count = 0
        decay_count = 0
        reset_count = 0
        for layer_index, layer_state in enumerate(layer_states):
            if layer_state is None or layer_state.xlstm_memory is None:
                continue
            memory_state = layer_state.xlstm_memory
            if memory_state.request_id != request_id:
                raise ValueError("xLSTMMemoryState request_id 与状态池更新 request_id 不一致。")
            if int(memory_state.layer_index) != layer_index:
                raise ValueError("xLSTMMemoryState layer_index 与所在层不一致。")
            self._states[(request_id, layer_index)] = memory_state
            max_token_count = max(max_token_count, int(memory_state.token_count))
            decay_count = max(decay_count, int(memory_state.decay_count))
            reset_count = max(reset_count, int(memory_state.reset_count))
            state_count += 1

        return self._set_metadata(
            request_id,
            phase=phase,
            token_count=max_token_count,
            state_count=state_count,
            decay_count=decay_count,
            reset_count=reset_count,
        )

    def get(self, request_id, layer_index):
        request_id = _normalize_request_id(request_id)
        return self._states.get((request_id, int(layer_index)))

    def get_request_states(self, request_id):
        request_id = _normalize_request_id(request_id)
        return tuple(
            self._states[(request_id, layer_index)]
            for layer_index in range(self.num_layers)
            if (request_id, layer_index) in self._states
        )

    def bind_to_layer_states(self, request_id, layer_states=None):
        """把池中 xLSTMMemory 状态合并进 LayerStateV2，保留其它状态槽。"""
        request_id = _normalize_request_id(request_id)
        if layer_states is None:
            normalized_states = [LayerStateV2() for _ in range(self.num_layers)]
        else:
            if len(layer_states) != self.num_layers:
                raise ValueError("layer_states 数量必须等于状态池 num_layers。")
            normalized_states = [
                LayerStateV2() if layer_state is None else layer_state
                for layer_state in layer_states
            ]

        merged_states = []
        for layer_index, layer_state in enumerate(normalized_states):
            pooled_state = self._states.get((request_id, layer_index))
            if pooled_state is None:
                merged_states.append(layer_state)
            else:
                merged_states.append(replace(layer_state, xlstm_memory=pooled_state))
        return tuple(merged_states)

    def mark_preempted(self, request_id):
        """标记 request 被抢占；状态保留在池内，后续 decode 可继续使用。"""
        previous = self._metadata_for(request_id)
        metadata = replace(
            previous,
            phase=XLSTM_POOL_PHASE_PREEMPTED,
            preempt_count=previous.preempt_count + 1,
            updated_at=time(),
        )
        self._metadata[previous.request_id] = metadata
        return metadata

    def release(self, request_id, reason="request_finished"):
        """释放指定 request 的 xLSTMMemory 状态，并返回释放前的状态。"""
        request_id = _normalize_request_id(request_id)
        released_states = []
        for key, state in list(self._states.items()):
            if key[0] != request_id:
                continue
            released_states.append(state.mark_released(reason, state.token_count))
            self._states.pop(key, None)
        self._set_metadata(
            request_id,
            phase=XLSTM_POOL_PHASE_RELEASED,
            state_count=0,
            released=True,
            release_reason=str(reason),
        )
        return tuple(released_states)

    def reset(self, request_id=None):
        """重置状态池；指定 request_id 时只清理该请求。"""
        if request_id is None:
            self._states.clear()
            self._metadata.clear()
            return

        request_id = _normalize_request_id(request_id)
        for key in list(self._states):
            if key[0] == request_id:
                self._states.pop(key, None)
        self._set_metadata(
            request_id,
            phase=XLSTM_POOL_PHASE_RESET,
            state_count=0,
            released=False,
            release_reason=None,
        )

    def to_runtime_metadata(self):
        return {
            "num_layers": self.num_layers,
            "request_count": self.request_count,
            "state_count": self.state_count,
            "requests": {
                request_id: metadata.to_dict()
                for request_id, metadata in sorted(self._metadata.items())
            },
        }
