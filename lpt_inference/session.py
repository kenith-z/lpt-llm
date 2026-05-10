"""LPT v2 推理会话状态管理。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class InferenceSession:
    """封装 prefill/decode 所需的 request-bound 状态。"""

    model: object
    request_id: str = "chat"
    token_ids: list[int] = field(default_factory=list)
    layer_states: tuple | None = None

    def _release_model_state(self):
        if hasattr(self.model, "reset_request_state"):
            self.model.reset_request_state(self.request_id)
        if hasattr(self.model, "release_retnet_assist_state"):
            self.model.release_retnet_assist_state(self.request_id, reason="session_reset")
        if hasattr(self.model, "release_xlstm_memory_state"):
            self.model.release_xlstm_memory_state(self.request_id, reason="session_reset")

    def reset(self):
        """清空本会话缓存。"""
        self._release_model_state()
        self.token_ids.clear()
        self.layer_states = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _full_attention_mask(self):
        return torch.ones(1, len(self.token_ids), dtype=torch.long, device=self.device)

    def prefill(self, input_ids):
        """用完整 prompt 初始化缓存。"""
        self.reset()
        self.token_ids.extend(int(value) for value in input_ids)
        tensor = torch.tensor([self.token_ids], dtype=torch.long, device=self.device)
        logits, states = self.model.prefill(tensor, request_id=self.request_id)
        self.layer_states = tuple(states)
        return logits

    def rebuild_on_switch(self):
        """按当前 token 序列重建缓存，用于 LongRoPE2 策略切换后的显式恢复。"""
        if not self.token_ids:
            raise ValueError("当前 session 没有可重建的上下文。")
        self._release_model_state()
        tensor = torch.tensor([self.token_ids], dtype=torch.long, device=self.device)
        logits, states = self.model.prefill(tensor, request_id=self.request_id)
        self.layer_states = tuple(states)
        return logits

    def append(self, token_id):
        """追加一个 token 并执行 decode。"""
        self.token_ids.append(int(token_id))
        next_input = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        logits, states = self.model.decode(
            next_input,
            attention_mask=self._full_attention_mask(),
            layer_states=self.layer_states,
            request_id=self.request_id,
        )
        self.layer_states = tuple(states)
        return logits

    def export_state(self):
        """导出轻量调试状态。"""
        return {
            "request_id": self.request_id,
            "token_count": len(self.token_ids),
            "has_layer_states": self.layer_states is not None,
        }
