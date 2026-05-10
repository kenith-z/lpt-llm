"""LPT v2 xLSTMAssist 专项评测。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from lpt_config import MemoryEvalConfig
from lpt_config.profiles import LPT_V2_MEMORY_PROFILE, build_lpt_v2_profile_config
from lpt_model import LPTV2, load_lpt_v2_checkpoint

from .utils import build_deterministic_input, dtype_name, resolve_eval_device, resolve_eval_dtype, set_eval_seed


DEFAULT_MEMORY_EVAL_CONFIG = MemoryEvalConfig()


@dataclass(frozen=True)
class MemoryAssistReport:
    """xLSTMAssist 专项评测报告。"""

    preset: str
    device: str
    dtype: str
    vocabulary_size: int
    sequence_length: int
    metrics: dict

    def to_dict(self):
        return {
            "report_type": "lpt_v2_xlstm_memory",
            "preset": self.preset,
            "device": self.device,
            "dtype": self.dtype,
            "vocabulary_size": self.vocabulary_size,
            "sequence_length": self.sequence_length,
            "metrics": dict(self.metrics),
        }

    def to_markdown(self):
        metrics = self.metrics
        lines = [
            "# LPT v2 xLSTMAssist Report",
            "",
            f"- preset: `{self.preset}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| prefill_token_count | {metrics['prefill_token_count']} |",
            f"| decode_token_count | {metrics['decode_token_count']} |",
            f"| decay_count | {metrics['decay_count']} |",
            f"| boundary_reset_count | {metrics['boundary_reset_count']} |",
            f"| special_token_reset_count | {metrics['special_token_reset_count']} |",
            f"| special_token_reset_configured | {metrics['special_token_reset_configured']} |",
            f"| session_reset_count | {metrics['session_reset_count']} |",
            f"| effective_beta | {metrics['effective_beta']:.8f} |",
            f"| memory_norm | {metrics['memory_norm']:.6f} |",
            f"| adapter_delta_norm | {metrics['adapter_delta_norm']:.6f} |",
            f"| memory_vs_eval_switch_logit_delta_l2 | {metrics['memory_vs_eval_switch_logit_delta_l2']:.6f} |",
            f"| router_entropy | {metrics['router_entropy']:.6f} |",
            "",
            f"decision: `{metrics['decision']['status']}`",
            "",
            metrics["decision"]["reason"],
        ]
        return "\n".join(lines) + "\n"


def _first_xlstm_state(states):
    for state in states:
        if state.xlstm_memory is not None:
            return state.xlstm_memory
    return None


def run_lpt_v2_memory_assist_report(
    *,
    preset=DEFAULT_MEMORY_EVAL_CONFIG.preset,
    vocabulary_size=DEFAULT_MEMORY_EVAL_CONFIG.vocabulary_size,
    sequence_length=DEFAULT_MEMORY_EVAL_CONFIG.sequence_length,
    device=DEFAULT_MEMORY_EVAL_CONFIG.device,
    dtype=DEFAULT_MEMORY_EVAL_CONFIG.dtype,
    seed=DEFAULT_MEMORY_EVAL_CONFIG.seed,
    checkpoint_path=None,
    **config_overrides,
):
    """运行 xLSTMAssist 专项评测。"""
    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    set_eval_seed(seed)
    if checkpoint_path is not None:
        loaded = load_lpt_v2_checkpoint(Path(checkpoint_path), map_location="cpu", strict=True)
        memory_model = loaded.model.to(device=target_device, dtype=target_dtype).eval()
        memory_config = memory_model.config
        vocabulary_size = int(memory_model.vocabulary_size)
        preset = memory_config.model_size_preset
    else:
        common_overrides = {
            "xlstm_memory_state_dim": 8,
            "xlstm_memory_adapter_rank": 4,
            "xlstm_memory_state_decay_interval": 2,
            "xlstm_memory_state_decay_factor": 0.5,
            "xlstm_memory_boundary_token_ids": (int(vocabulary_size) - 1,),
        }
        common_overrides.update(config_overrides)
        memory_config = build_lpt_v2_profile_config(
            LPT_V2_MEMORY_PROFILE,
            preset=preset,
            **common_overrides,
        )
        memory_model = LPTV2(vocabulary_size, memory_config).to(device=target_device, dtype=target_dtype).eval()
    eval_switch_config = memory_config.with_overrides(moe_router_input_mode="ffn_norm_only_eval")
    eval_switch_model = LPTV2(vocabulary_size, eval_switch_config).to(device=target_device, dtype=target_dtype).eval()
    eval_switch_model.load_state_dict(memory_model.state_dict(), strict=True)

    input_ids = build_deterministic_input(
        vocabulary_size,
        1,
        sequence_length,
        offset=3,
        device=target_device,
    )
    with torch.no_grad():
        memory_logits, memory_states = memory_model.prefill(input_ids, request_id="memory-report")
        eval_logits, eval_states = eval_switch_model.prefill(input_ids, request_id="memory-report-eval")
        decode_logits, decode_states = memory_model.decode(
            torch.tensor([[7]], dtype=torch.long, device=target_device),
            attention_mask=torch.ones(1, sequence_length + 1, dtype=torch.long, device=target_device),
            layer_states=memory_states,
            request_id="memory-report",
        )
        _boundary_logits, boundary_states = memory_model.decode(
            torch.tensor([[8]], dtype=torch.long, device=target_device),
            attention_mask=torch.ones(1, sequence_length + 2, dtype=torch.long, device=target_device),
            layer_states=decode_states,
            memory_boundary_metadata={"boundary_type": "document"},
            request_id="memory-report",
        )
        _special_logits, special_states = memory_model.decode(
            torch.tensor([[vocabulary_size - 1]], dtype=torch.long, device=target_device),
            attention_mask=torch.ones(1, sequence_length + 3, dtype=torch.long, device=target_device),
            layer_states=boundary_states,
            request_id="memory-report",
        )
        _session_logits, session_states = memory_model.decode(
            torch.tensor([[9]], dtype=torch.long, device=target_device),
            attention_mask=torch.ones(1, sequence_length + 4, dtype=torch.long, device=target_device),
            layer_states=special_states,
            session_event="session_reset",
            request_id="memory-report",
        )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    prefill_state = _first_xlstm_state(memory_states)
    decode_state = _first_xlstm_state(decode_states)
    boundary_state = _first_xlstm_state(boundary_states)
    special_state = _first_xlstm_state(special_states)
    session_state = _first_xlstm_state(session_states)
    logit_delta_l2 = float((memory_logits.float() - eval_logits.float()).pow(2).mean().sqrt().detach().cpu())
    router_entropy = 0.0 if decode_states[0].moe is None else float(decode_states[0].moe.router_entropy)
    boundary_token_ids = tuple(getattr(memory_config, "xlstm_memory_boundary_token_ids", ()) or ())
    special_token_reset_configured = bool(boundary_token_ids)
    special_token_reset_ready = (
        bool(special_state is not None and boundary_state is not None)
        and (
            special_state.reset_count > boundary_state.reset_count
            if special_token_reset_configured
            else special_state.reset_count == boundary_state.reset_count
        )
    )
    mechanism_ready = bool(
        prefill_state is not None
        and decode_state is not None
        and boundary_state is not None
        and special_state is not None
        and session_state is not None
        and decode_state.token_count == sequence_length + 1
        and decode_state.adapter_delta_norm is not None
        and decode_state.adapter_delta_norm > 0.0
        and boundary_state.reset_count > decode_state.reset_count
        and special_token_reset_ready
        and session_state.reset_count > special_state.reset_count
    )
    decision_reason = (
        "xLSTMAssist 状态连续性、boundary/session reset 和输入 adapter 均形成可观测机制；"
        "当前配置未启用 special token reset，不将该项作为失败条件。"
        if mechanism_ready and not special_token_reset_configured
        else (
            "xLSTMAssist 状态连续性、decay/reset 和输入 adapter 均形成可观测机制；质量收益需训练 checkpoint 进一步评估。"
            if mechanism_ready
            else "xLSTMAssist 机制观测未全部通过，应检查状态池、reset 触发或 adapter。"
        )
    )
    metrics = {
        "prefill_token_count": None if prefill_state is None else prefill_state.token_count,
        "decode_token_count": None if decode_state is None else decode_state.token_count,
        "decay_count": None if session_state is None else session_state.decay_count,
        "boundary_reset_count": None if boundary_state is None else boundary_state.reset_count,
        "special_token_reset_count": None if special_state is None else special_state.reset_count,
        "special_token_reset_configured": special_token_reset_configured,
        "special_token_reset_ready": special_token_reset_ready,
        "session_reset_count": None if session_state is None else session_state.reset_count,
        "effective_beta": 0.0 if decode_state is None else float(decode_state.effective_beta or 0.0),
        "memory_norm": 0.0 if decode_state is None else float(decode_state.memory_norm or 0.0),
        "adapter_delta_norm": 0.0 if decode_state is None else float(decode_state.adapter_delta_norm or 0.0),
        "memory_vs_eval_switch_logit_delta_l2": logit_delta_l2,
        "router_entropy": router_entropy,
        "pool_metadata": memory_model.xlstm_memory_state_pool.to_runtime_metadata(),
        "eval_switch_state_token_count": None if _first_xlstm_state(eval_states) is None else _first_xlstm_state(eval_states).token_count,
        "decision": {
            "status": "admit_instrumentation_only" if mechanism_ready else "close_or_debug",
            "reason": decision_reason,
        },
    }
    return MemoryAssistReport(
        preset=str(preset),
        device=str(target_device),
        dtype=dtype_name(target_dtype),
        vocabulary_size=int(vocabulary_size),
        sequence_length=int(sequence_length),
        metrics=metrics,
    )
