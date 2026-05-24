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
    thinking_mode_ids: list[int] = field(default_factory=list)
    target_channel_ids: list[int] = field(default_factory=list)
    layer_states: tuple | None = None

    def _release_model_state(self):
        """释放模型内与本 request_id 绑定的 Paged KV、RetNet 和 xLSTM 状态。"""
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
        self.thinking_mode_ids.clear()
        self.target_channel_ids.clear()
        self.layer_states = None

    @property
    def device(self):
        """返回模型当前主设备。"""
        return next(self.model.parameters()).device

    def _full_attention_mask(self):
        """生成覆盖当前会话全部 token 的 attention mask。"""
        return torch.ones(1, len(self.token_ids), dtype=torch.long, device=self.device)

    def _control_tensor(self, values):
        """把会话控制 id 列表转为模型输入张量。"""
        return torch.tensor([values], dtype=torch.long, device=self.device)

    def prefill(self, input_ids, *, thinking_mode_ids=None, target_channel_ids=None):
        """用完整 prompt 初始化缓存。"""
        self.reset()
        self.token_ids.extend(int(value) for value in input_ids)
        thinking_mode_values = [0] * len(input_ids) if thinking_mode_ids is None else list(thinking_mode_ids)
        target_channel_values = [0] * len(input_ids) if target_channel_ids is None else list(target_channel_ids)
        if len(thinking_mode_values) != len(input_ids) or len(target_channel_values) != len(input_ids):
            raise ValueError("thinking_mode_ids / target_channel_ids 长度必须与 input_ids 一致。")
        self.thinking_mode_ids.extend(int(value) for value in thinking_mode_values)
        self.target_channel_ids.extend(int(value) for value in target_channel_values)
        tensor = torch.tensor([self.token_ids], dtype=torch.long, device=self.device)
        logits, states = self.model.prefill(
            tensor,
            thinking_mode_ids=self._control_tensor(self.thinking_mode_ids),
            target_channel_ids=self._control_tensor(self.target_channel_ids),
            request_id=self.request_id,
        )
        self.layer_states = tuple(states)
        return logits

    def rebuild_on_switch(self, *, last_thinking_mode_id=None, last_target_channel_id=None):
        """按当前 token 序列重建缓存，用于 LongRoPE2 策略切换后的显式恢复。"""
        if not self.token_ids:
            raise ValueError("当前 session 没有可重建的上下文。")
        self._release_model_state()
        if last_thinking_mode_id is not None:
            self.thinking_mode_ids[-1] = int(last_thinking_mode_id)
        if last_target_channel_id is not None:
            self.target_channel_ids[-1] = int(last_target_channel_id)
        tensor = torch.tensor([self.token_ids], dtype=torch.long, device=self.device)
        logits, states = self.model.prefill(
            tensor,
            thinking_mode_ids=self._control_tensor(self.thinking_mode_ids),
            target_channel_ids=self._control_tensor(self.target_channel_ids),
            request_id=self.request_id,
        )
        self.layer_states = tuple(states)
        return logits

    def append(self, token_id, *, thinking_mode_id=0, target_channel_id=0):
        """追加一个 token 并执行 decode。"""
        self.token_ids.append(int(token_id))
        self.thinking_mode_ids.append(int(thinking_mode_id))
        self.target_channel_ids.append(int(target_channel_id))
        next_input = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        # decode 只输入新 token，但 attention_mask 必须覆盖完整上下文长度，
        # 这样模型能把新 token 与已缓存的历史 K/V 对齐。
        logits, states = self.model.decode(
            next_input,
            attention_mask=self._full_attention_mask(),
            thinking_mode_ids=torch.tensor([[int(thinking_mode_id)]], dtype=torch.long, device=self.device),
            target_channel_ids=torch.tensor([[int(target_channel_id)]], dtype=torch.long, device=self.device),
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
