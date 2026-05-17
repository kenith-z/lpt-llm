"""LPT v2 checkpoint forward smoke。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from lpt_config import GlobalConfig, count_retnet_assist_enabled_layers, count_xlstm_memory_enabled_layers
from lpt_model import load_lpt_v2_checkpoint

from .utils import (
    build_deterministic_input,
    dtype_name,
    next_token_loss,
    resolve_eval_device,
    resolve_eval_dtype,
    set_eval_seed,
)


@dataclass(frozen=True)
class ForwardSmokeReport:
    """真实 checkpoint 的只读 forward smoke 报告。"""

    checkpoint_path: str
    device: str
    dtype: str
    batch_size: int
    sequence_length: int
    use_kv_cache: bool
    metrics: dict

    @property
    def success(self):
        """forward_ok 是只读 smoke 的唯一成功判定。"""
        return bool(self.metrics.get("forward_ok"))

    def to_dict(self):
        """生成 forward smoke JSON 载荷。"""
        return {
            "report_type": "lpt_v2_forward_smoke",
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "use_kv_cache": self.use_kv_cache,
            "metrics": dict(self.metrics),
        }

    def to_markdown(self):
        """生成 forward smoke Markdown 报告。"""
        metrics = self.metrics
        lines = [
            "# LPT v2 Forward Smoke",
            "",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            f"- use_kv_cache: `{self.use_kv_cache}`",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| forward_ok | {metrics['forward_ok']} |",
            f"| logits_finite | {metrics['logits_finite']} |",
            f"| logits_shape | `{metrics['logits_shape']}` |",
            f"| loss | {metrics['loss']:.6f} |",
            f"| ppl | {metrics['ppl']:.6f} |",
            f"| state_count | {metrics['state_count']} |",
            f"| attention_state_count | {metrics['attention_state_count']} |",
            f"| retnet_state_count | {metrics['retnet_state_count']} |",
            f"| retnet_layer_state_count | {metrics['retnet_layer_state_count']} |",
            f"| expected_retnet_layer_state_count | {metrics['expected_retnet_layer_state_count']} |",
            f"| retnet_assist_layers | `{metrics['retnet_assist_layers']}` |",
            f"| retnet_assist_selected_layers | `{metrics['retnet_assist_selected_layers']}` |",
            f"| retnet_adapter_rank | {metrics['retnet_adapter_rank']} |",
            f"| retnet_parameter_sharing | `{metrics['retnet_parameter_sharing']}` |",
            f"| retnet_state_sharing | `{metrics['retnet_state_sharing']}` |",
            f"| retnet_sharing_group_size | {metrics['retnet_sharing_group_size']} |",
            f"| retnet_assist_mode | `{metrics['retnet_assist_mode']}` |",
            f"| retnet_adapter_target | `{metrics['retnet_adapter_target']}` |",
            f"| retnet_k_adapter_enabled | {metrics['retnet_k_adapter_enabled']} |",
            f"| retnet_context_adapter_enabled | {metrics['retnet_context_adapter_enabled']} |",
            f"| retnet_q_adapter_delta_norm_mean | {metrics['retnet_q_adapter_delta_norm_mean']:.6f} |",
            f"| retnet_k_adapter_delta_norm_mean | {metrics['retnet_k_adapter_delta_norm_mean']:.6f} |",
            f"| retnet_context_adapter_delta_norm_mean | {metrics['retnet_context_adapter_delta_norm_mean']:.6f} |",
            f"| retnet_alpha_context_mean | {metrics['retnet_alpha_context_mean']:.8f} |",
            f"| xlstm_state_count | {metrics['xlstm_state_count']} |",
            f"| expected_xlstm_state_count | {metrics['expected_xlstm_state_count']} |",
            f"| paged_kv_page_count | {metrics['paged_kv_page_count']} |",
        ]
        return "\n".join(lines) + "\n"


def _count_layer_states(layer_states):
    """统计 LayerStateV2 中 Attention、RetNet、xLSTM 的实际数量。"""
    attention_state_count = 0
    retnet_state_slots = set()
    retnet_layer_state_count = 0
    xlstm_state_count = 0
    for layer_state in layer_states:
        if layer_state.attention is not None:
            attention_state_count += 1
        if layer_state.retnet_assist is not None:
            retnet_layer_state_count += 1
            retnet_state_slots.add(int(layer_state.retnet_assist.state_slot))
        if layer_state.xlstm_memory is not None:
            xlstm_state_count += 1
    return attention_state_count, len(retnet_state_slots), retnet_layer_state_count, xlstm_state_count


def _mean_optional(values):
    """跳过 None 后求均值，没有有效值时返回 0。"""
    values = [float(value) for value in values if value is not None]
    return 0.0 if not values else sum(values) / len(values)


def _retnet_observability(layer_states):
    """汇总 RetNet adapter 观测指标。"""
    summary_norms = []
    q_delta_norms = []
    k_delta_norms = []
    context_delta_norms = []
    alpha_q_values = []
    alpha_k_values = []
    alpha_context_values = []
    for layer_state in layer_states:
        state = layer_state.retnet_assist
        if state is None:
            continue
        summary_norms.append(state.summary_norm)
        q_delta_norms.append(state.q_adapter_delta_norm)
        k_delta_norms.append(state.k_adapter_delta_norm)
        context_delta_norms.append(state.context_adapter_delta_norm)
        alpha_q_values.append(state.alpha_q)
        alpha_k_values.append(state.alpha_k)
        alpha_context_values.append(state.alpha_context)
    return {
        "retnet_summary_norm_mean": _mean_optional(summary_norms),
        "retnet_q_adapter_delta_norm_mean": _mean_optional(q_delta_norms),
        "retnet_k_adapter_delta_norm_mean": _mean_optional(k_delta_norms),
        "retnet_context_adapter_delta_norm_mean": _mean_optional(context_delta_norms),
        "retnet_alpha_q_mean": _mean_optional(alpha_q_values),
        "retnet_alpha_k_mean": _mean_optional(alpha_k_values),
        "retnet_alpha_context_mean": _mean_optional(alpha_context_values),
    }


def run_lpt_v2_forward_smoke_report(
    *,
    checkpoint_path,
    batch_size=1,
    sequence_length=32,
    device="auto",
    dtype="auto",
    seed=20260503,
    use_kv_cache=False,
):
    """加载 checkpoint 后执行一次只读 forward，确认前向链路可用。"""
    set_eval_seed(seed)
    torch_device = resolve_eval_device(device)
    torch_dtype = resolve_eval_dtype(dtype, device=torch_device)
    GlobalConfig.device = torch_device
    GlobalConfig.parameter_dtype = torch_dtype

    loaded = load_lpt_v2_checkpoint(Path(checkpoint_path), map_location="cpu", strict=True)
    model = loaded.model.to(device=torch_device, dtype=torch_dtype)
    model.eval()
    vocabulary_size = int(model.vocabulary_size)
    input_ids = build_deterministic_input(
        vocabulary_size,
        batch_size,
        sequence_length,
        device=torch_device,
    )

    with torch.inference_mode():
        # 只读 smoke 不更新 optimizer；use_kv_cache 可覆盖训练/推理两种前向边界。
        logits, layer_states = model(
            input_ids,
            rope_cache_scope="inference",
            request_id="forward-smoke",
            use_kv_cache=bool(use_kv_cache),
        )
        loss, ppl = next_token_loss(logits, input_ids)

    attention_count, retnet_count, retnet_layer_count, xlstm_count = _count_layer_states(layer_states)
    expected_shape = [int(batch_size), int(sequence_length), vocabulary_size]
    logits_shape = list(logits.shape)
    logits_finite = bool(torch.isfinite(logits).all().detach().cpu())
    expected_retnet_count = count_retnet_assist_enabled_layers(model.config)
    expected_xlstm_count = count_xlstm_memory_enabled_layers(model.config)
    paged_kv_metadata = model.paged_kv_cache.runtime_metadata()
    retnet_metrics = _retnet_observability(layer_states)
    metrics = {
        "forward_ok": bool(
            logits_shape == expected_shape
            and logits_finite
            and len(layer_states) == int(model.config.num_layers)
            and retnet_layer_count == expected_retnet_count
            and xlstm_count == expected_xlstm_count
        ),
        "logits_finite": logits_finite,
        "logits_shape": logits_shape,
        "expected_logits_shape": expected_shape,
        "loss": 0.0 if loss is None else float(loss),
        "ppl": 0.0 if ppl is None else float(ppl),
        "state_count": int(len(layer_states)),
        "attention_state_count": int(attention_count),
        "retnet_state_count": int(retnet_count),
        "retnet_layer_state_count": int(retnet_layer_count),
        "expected_retnet_layer_state_count": int(expected_retnet_count),
        "xlstm_state_count": int(xlstm_count),
        "expected_xlstm_state_count": int(expected_xlstm_count),
        "paged_kv_page_count": int(paged_kv_metadata.get("allocated_page_count", 0)),
        "model_size_preset": model.config.model_size_preset,
        "retnet_assist_layers": model.config.retnet_assist_layers,
        "retnet_assist_selected_layers": list(model.config.retnet_assist_selected_layers),
        "retnet_adapter_rank": int(model.config.retnet_adapter_rank),
        "retnet_parameter_sharing": model.config.retnet_parameter_sharing,
        "retnet_state_sharing": model.config.retnet_state_sharing,
        "retnet_sharing_group_size": int(model.config.retnet_sharing_group_size),
        "retnet_assist_mode": model.config.retnet_assist_mode,
        "retnet_adapter_target": list(model.config.retnet_adapter_target),
        "retnet_k_adapter_enabled": bool(model.config.retnet_k_adapter_enabled),
        "retnet_context_adapter_enabled": bool(model.config.retnet_context_adapter_enabled),
        "retnet_context_adapter_alpha": float(model.config.retnet_context_adapter_alpha),
        **retnet_metrics,
        "xlstm_memory_layers": model.config.xlstm_memory_layers,
        "xlstm_memory_selected_layers": list(model.config.xlstm_memory_selected_layers),
    }
    return ForwardSmokeReport(
        checkpoint_path=str(checkpoint_path),
        device=str(torch_device),
        dtype=dtype_name(torch_dtype),
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        use_kv_cache=bool(use_kv_cache),
        metrics=metrics,
    )
