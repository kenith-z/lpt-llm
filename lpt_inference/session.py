"""推理 session 与缓存管理。"""

from dataclasses import dataclass

import torch

from lpt_model import LayerState


def _build_position_ids(attention_mask):
    return attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)


def _clone_layer_state(layer_state):
    if layer_state is None:
        return None
    return LayerState(
        state_type=layer_state.state_type,
        tensors=tuple(tensor.clone() for tensor in layer_state.tensors),
    )


@dataclass(frozen=True)
class InferenceStateSnapshot:
    """导出的推理状态快照。"""

    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    using_rescaled_rope: bool
    layer_states: tuple[LayerState | None, ...] | None


@dataclass(frozen=True)
class CacheUpdatePlan:
    """一次缓存更新对应的模型执行计划。"""

    model_input_ids: torch.Tensor
    model_attention_mask: torch.Tensor
    model_position_ids: torch.Tensor
    incoming_layer_states: list[LayerState | None] | None
    next_token_ids: torch.Tensor
    next_attention_mask: torch.Tensor
    next_position_ids: torch.Tensor
    using_rescaled_rope: bool
    returned_query_length: int


class CacheManager:
    """统一维护推理阶段的缓存状态。"""

    def __init__(self, rope_cache):
        self.rope_cache = rope_cache
        self.reset()

    def reset(self):
        """清空当前缓存。"""
        self.token_ids = None
        self.attention_mask = None
        self.position_ids = None
        self.layer_states = None
        self.using_rescaled_rope = False

    def export_state(self):
        """导出当前状态快照，避免把内部可变对象直接暴露给调用方。"""
        if self.token_ids is None:
            return None

        layer_states = None
        if self.layer_states is not None:
            layer_states = tuple(_clone_layer_state(state) for state in self.layer_states)

        return InferenceStateSnapshot(
            token_ids=self.token_ids.clone(),
            attention_mask=self.attention_mask.clone(),
            position_ids=self.position_ids.clone(),
            using_rescaled_rope=self.using_rescaled_rope,
            layer_states=layer_states,
        )

    def prefill(self, input_ids, attention_mask=None):
        """为整段上下文构造 prefill 执行计划。"""
        normalized_mask = self._normalize_full_attention_mask(input_ids, attention_mask)
        position_ids = _build_position_ids(normalized_mask)
        using_rescaled_rope = self.rope_cache.should_use_rescaled_rope(position_ids=position_ids)
        return CacheUpdatePlan(
            model_input_ids=input_ids,
            model_attention_mask=normalized_mask,
            model_position_ids=position_ids,
            incoming_layer_states=None,
            next_token_ids=input_ids,
            next_attention_mask=normalized_mask,
            next_position_ids=position_ids,
            using_rescaled_rope=using_rescaled_rope,
            returned_query_length=input_ids.size(1),
        )

    def append(self, input_ids, attention_mask=None):
        """为增量 token 构造 append 执行计划。"""
        if self.token_ids is None:
            return self.prefill(input_ids, attention_mask=attention_mask)

        self._validate_append_inputs(input_ids)
        step_attention_mask = self._normalize_step_attention_mask(input_ids, attention_mask)

        next_token_ids = torch.cat([self.token_ids, input_ids], dim=1)
        next_attention_mask = torch.cat([self.attention_mask, step_attention_mask], dim=1)
        next_position_ids = _build_position_ids(next_attention_mask)
        target_uses_rescaled_rope = self.rope_cache.should_use_rescaled_rope(position_ids=next_position_ids)

        if self.layer_states is not None and target_uses_rescaled_rope and not self.using_rescaled_rope:
            return CacheUpdatePlan(
                model_input_ids=next_token_ids,
                model_attention_mask=next_attention_mask,
                model_position_ids=next_position_ids,
                incoming_layer_states=None,
                next_token_ids=next_token_ids,
                next_attention_mask=next_attention_mask,
                next_position_ids=next_position_ids,
                using_rescaled_rope=True,
                returned_query_length=input_ids.size(1),
            )

        return CacheUpdatePlan(
            model_input_ids=input_ids,
            model_attention_mask=next_attention_mask,
            model_position_ids=next_position_ids[:, -input_ids.size(1):],
            incoming_layer_states=None if self.layer_states is None else list(self.layer_states),
            next_token_ids=next_token_ids,
            next_attention_mask=next_attention_mask,
            next_position_ids=next_position_ids,
            using_rescaled_rope=target_uses_rescaled_rope,
            returned_query_length=input_ids.size(1),
        )

    def rebuild_on_switch(self):
        """在 LongRoPE2 需要切换编码模式时重建整段缓存。"""
        if self.token_ids is None:
            raise ValueError("当前 session 还没有可重建的上下文。")

        target_uses_rescaled_rope = self.rope_cache.should_use_rescaled_rope(
            position_ids=self.position_ids
        )
        if not target_uses_rescaled_rope or self.using_rescaled_rope:
            return None

        return CacheUpdatePlan(
            model_input_ids=self.token_ids,
            model_attention_mask=self.attention_mask,
            model_position_ids=self.position_ids,
            incoming_layer_states=None,
            next_token_ids=self.token_ids,
            next_attention_mask=self.attention_mask,
            next_position_ids=self.position_ids,
            using_rescaled_rope=True,
            returned_query_length=self.token_ids.size(1),
        )

    def commit(self, plan, layer_states):
        """提交一次模型执行后的缓存结果。"""
        self.token_ids = plan.next_token_ids
        self.attention_mask = plan.next_attention_mask
        self.position_ids = plan.next_position_ids
        self.layer_states = None if layer_states is None else list(layer_states)
        self.using_rescaled_rope = plan.using_rescaled_rope

    @staticmethod
    def _normalize_full_attention_mask(input_ids, attention_mask):
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask 形状必须与 input_ids 一致，"
                f"当前分别为 {tuple(attention_mask.shape)} 和 {tuple(input_ids.shape)}。"
            )
        return attention_mask.to(device=input_ids.device, dtype=torch.long)

    @staticmethod
    def _normalize_step_attention_mask(input_ids, attention_mask):
        if attention_mask is None:
            return torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "append 阶段的 attention_mask 形状必须与新增 input_ids 一致，"
                f"当前分别为 {tuple(attention_mask.shape)} 和 {tuple(input_ids.shape)}。"
            )
        return attention_mask.to(device=input_ids.device, dtype=torch.long)

    def _validate_append_inputs(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError("input_ids 必须是形如 [batch, seq] 的二维张量。")
        if input_ids.size(0) != self.token_ids.size(0):
            raise ValueError(
                f"append 的 batch 大小 ({input_ids.size(0)}) 必须与当前 session "
                f"的 batch 大小 ({self.token_ids.size(0)}) 一致。"
            )
        if input_ids.device != self.token_ids.device:
            raise ValueError("append 的 input_ids 设备必须与当前 session 保持一致。")


class InferenceSession:
    """对外暴露的推理 session API。"""

    def __init__(self, model, cache_manager=None):
        self.model = model
        self.cache_manager = cache_manager or CacheManager(model.get_rope_cache("inference"))

    @property
    def token_ids(self):
        return self.cache_manager.token_ids

    @property
    def attention_mask(self):
        return self.cache_manager.attention_mask

    @property
    def position_ids(self):
        return self.cache_manager.position_ids

    @property
    def using_rescaled_rope(self):
        return self.cache_manager.using_rescaled_rope

    def prefill(self, input_ids, attention_mask=None):
        """对完整上下文做一次 prefill，并写入缓存。"""
        plan = self.cache_manager.prefill(input_ids, attention_mask=attention_mask)
        return self._execute_plan(plan)

    def append(self, input_ids, attention_mask=None):
        """向当前 session 追加 token，并自动处理 LongRoPE2 切换重建。"""
        plan = self.cache_manager.append(input_ids, attention_mask=attention_mask)
        return self._execute_plan(plan)

    def rebuild_on_switch(self):
        """显式触发一次 LongRoPE2 切换重建。"""
        plan = self.cache_manager.rebuild_on_switch()
        if plan is None:
            return None
        return self._execute_plan(plan)

    def reset(self):
        """重置当前 session。"""
        self.cache_manager.reset()

    def export_state(self):
        """导出当前缓存状态。"""
        return self.cache_manager.export_state()

    def _execute_plan(self, plan):
        logits, layer_states = self.model(
            plan.model_input_ids,
            position_ids=plan.model_position_ids,
            attention_mask=plan.model_attention_mask,
            layer_states=plan.incoming_layer_states,
            rope_cache_scope="inference",
        )
        self.cache_manager.commit(plan, layer_states)
        return logits[:, -plan.returned_query_length:, :]
