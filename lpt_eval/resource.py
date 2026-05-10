"""LPT v2 资源指标报告。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from lpt_config import ResourceEvalConfig
from lpt_config.profiles import build_lpt_v2_profile_config
from lpt_model import LPTV2, load_lpt_v2_checkpoint

from .utils import (
    build_deterministic_input,
    dtype_name,
    resolve_eval_device,
    resolve_eval_dtype,
    set_eval_seed,
)


DEFAULT_RESOURCE_EVAL_CONFIG = ResourceEvalConfig()


@dataclass(frozen=True)
class ResourceReport:
    """LPT v2 资源指标报告。"""

    profile: str
    preset: str
    device: str
    dtype: str
    batch_size: int
    sequence_length: int
    decode_steps: int
    metrics: dict

    def to_dict(self):
        return {
            "report_type": "lpt_v2_resource",
            "profile": self.profile,
            "preset": self.preset,
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "decode_steps": self.decode_steps,
            "metrics": dict(self.metrics),
        }

    def to_markdown(self):
        metrics = self.metrics
        lines = [
            "# LPT v2 Resource Report",
            "",
            f"- profile: `{self.profile}`",
            f"- preset: `{self.preset}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| prefill_tokens_per_sec | {metrics['prefill_tokens_per_sec']:.2f} |",
            f"| decode_tokens_per_sec | {metrics['decode_tokens_per_sec']:.2f} |",
            f"| first_token_latency_ms | {metrics['first_token_latency_ms']:.4f} |",
            f"| peak_memory_bytes | {metrics['peak_memory_bytes']} |",
            f"| paged_kv_page_bytes | {metrics['paged_kv_page_bytes']} |",
            f"| retnet_state_bytes | {metrics['retnet_state_bytes']} |",
            f"| retnet_summary_norm_mean | {metrics['retnet_summary_norm_mean']:.6f} |",
            f"| retnet_q_adapter_delta_norm_mean | {metrics['retnet_q_adapter_delta_norm_mean']:.6f} |",
            f"| retnet_k_adapter_delta_norm_mean | {metrics['retnet_k_adapter_delta_norm_mean']:.6f} |",
            f"| xlstm_memory_state_bytes | {metrics['xlstm_memory_state_bytes']} |",
            f"| xlstm_effective_beta_mean | {metrics['xlstm_effective_beta_mean']:.8f} |",
            f"| xlstm_memory_norm_mean | {metrics['xlstm_memory_norm_mean']:.6f} |",
            f"| xlstm_adapter_delta_norm_mean | {metrics['xlstm_adapter_delta_norm_mean']:.6f} |",
            f"| router_entropy_mean | {metrics['router_entropy_mean']:.6f} |",
            f"| load_balance_loss_mean | {metrics['load_balance_loss_mean']:.6f} |",
            f"| router_z_loss_mean | {metrics['router_z_loss_mean']:.6f} |",
            "",
            "## Layer Time",
            "",
            "| layer | mean_ms | calls |",
            "|---:|---:|---:|",
        ]
        for item in metrics["per_layer_ms"]:
            lines.append(f"| {item['layer_index']} | {item['mean_ms']:.4f} | {item['calls']} |")
        return "\n".join(lines) + "\n"


def _synchronize_if_needed(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _state_bytes(states):
    retnet_bytes = 0
    xlstm_bytes = 0
    for state in states:
        if state.retnet_assist is not None and state.retnet_assist.summary is not None:
            summary = state.retnet_assist.summary
            retnet_bytes += int(summary.numel() * summary.element_size())
        if state.xlstm_memory is not None and state.xlstm_memory.memory is not None:
            memory = state.xlstm_memory.memory
            xlstm_bytes += int(memory.numel() * memory.element_size())
    return retnet_bytes, xlstm_bytes


def _router_metrics(states):
    entropies = []
    load_losses = []
    z_losses = []
    for state in states:
        if state.moe is None:
            continue
        if state.moe.router_entropy is not None:
            entropies.append(float(state.moe.router_entropy))
        if state.moe.load_balance_loss is not None:
            load_losses.append(float(state.moe.load_balance_loss))
        if state.moe.router_z_loss is not None:
            z_losses.append(float(state.moe.router_z_loss))
    mean = lambda values: 0.0 if not values else sum(values) / len(values)
    return mean(entropies), mean(load_losses), mean(z_losses)


def _xlstm_observability(states):
    effective_betas = []
    memory_norms = []
    adapter_delta_norms = []
    for state in states:
        if state.xlstm_memory is None:
            continue
        if state.xlstm_memory.effective_beta is not None:
            effective_betas.append(float(state.xlstm_memory.effective_beta))
        if state.xlstm_memory.memory_norm is not None:
            memory_norms.append(float(state.xlstm_memory.memory_norm))
        if state.xlstm_memory.adapter_delta_norm is not None:
            adapter_delta_norms.append(float(state.xlstm_memory.adapter_delta_norm))
    mean = lambda values: 0.0 if not values else sum(values) / len(values)
    return mean(effective_betas), mean(memory_norms), mean(adapter_delta_norms)


def _retnet_observability(states):
    summary_norms = []
    q_delta_norms = []
    k_delta_norms = []
    alpha_q_values = []
    alpha_k_values = []
    for state in states:
        if state.retnet_assist is None:
            continue
        for source, target in (
            (state.retnet_assist.summary_norm, summary_norms),
            (state.retnet_assist.q_adapter_delta_norm, q_delta_norms),
            (state.retnet_assist.k_adapter_delta_norm, k_delta_norms),
            (state.retnet_assist.alpha_q, alpha_q_values),
            (state.retnet_assist.alpha_k, alpha_k_values),
        ):
            if source is not None:
                target.append(float(source))
    mean = lambda values: 0.0 if not values else sum(values) / len(values)
    return mean(summary_norms), mean(q_delta_norms), mean(k_delta_norms), mean(alpha_q_values), mean(alpha_k_values)


def run_lpt_v2_resource_report(
    *,
    profile=DEFAULT_RESOURCE_EVAL_CONFIG.profile,
    preset=DEFAULT_RESOURCE_EVAL_CONFIG.preset,
    vocabulary_size=DEFAULT_RESOURCE_EVAL_CONFIG.vocabulary_size,
    batch_size=DEFAULT_RESOURCE_EVAL_CONFIG.batch_size,
    sequence_length=DEFAULT_RESOURCE_EVAL_CONFIG.sequence_length,
    decode_steps=DEFAULT_RESOURCE_EVAL_CONFIG.decode_steps,
    device=DEFAULT_RESOURCE_EVAL_CONFIG.device,
    dtype=DEFAULT_RESOURCE_EVAL_CONFIG.dtype,
    seed=DEFAULT_RESOURCE_EVAL_CONFIG.seed,
    checkpoint_path=None,
    **config_overrides,
):
    """运行 LPT v2 资源指标 smoke。"""
    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    set_eval_seed(seed)
    if checkpoint_path is not None:
        loaded = load_lpt_v2_checkpoint(Path(checkpoint_path), map_location="cpu", strict=True)
        model = loaded.model.to(device=target_device, dtype=target_dtype).eval()
        config = model.config
        vocabulary_size = int(model.vocabulary_size)
        profile = "checkpoint"
        preset = config.model_size_preset
    else:
        config = build_lpt_v2_profile_config(profile, preset=preset, **config_overrides)
        model = LPTV2(vocabulary_size, config).to(device=target_device, dtype=target_dtype).eval()

    layer_times = defaultdict(list)
    handles = []

    def make_pre_hook(layer_index):
        def pre_hook(_module, _args):
            _synchronize_if_needed(target_device)
            layer_times[(layer_index, "start")].append(perf_counter())
        return pre_hook

    def make_post_hook(layer_index):
        def post_hook(_module, _args, _output):
            _synchronize_if_needed(target_device)
            start_values = layer_times[(layer_index, "start")]
            if start_values:
                start = start_values.pop()
                layer_times[layer_index].append((perf_counter() - start) * 1000.0)
        return post_hook

    for layer_index, layer in enumerate(model.layers):
        handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_index)))
        handles.append(layer.register_forward_hook(make_post_hook(layer_index)))

    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    input_ids = build_deterministic_input(
        vocabulary_size,
        batch_size,
        sequence_length,
        offset=11,
        device=target_device,
    )
    with torch.no_grad():
        _synchronize_if_needed(target_device)
        prefill_start = perf_counter()
        logits, states = model.prefill(input_ids, request_id="resource-report")
        _synchronize_if_needed(target_device)
        prefill_elapsed = perf_counter() - prefill_start

        decode_states = states
        first_token_latency_ms = 0.0
        decode_elapsed_total = 0.0
        for step in range(int(decode_steps)):
            next_ids = build_deterministic_input(
                vocabulary_size,
                batch_size,
                1,
                offset=sequence_length + step + 13,
                device=target_device,
            )
            full_mask = torch.ones(
                batch_size,
                sequence_length + step + 1,
                dtype=torch.long,
                device=target_device,
            )
            _synchronize_if_needed(target_device)
            decode_start = perf_counter()
            decode_logits, decode_states = model.decode(
                next_ids,
                attention_mask=full_mask,
                layer_states=decode_states,
                request_id="resource-report",
            )
            _synchronize_if_needed(target_device)
            elapsed = perf_counter() - decode_start
            decode_elapsed_total += elapsed
            if step == 0:
                first_token_latency_ms = elapsed * 1000.0

    for handle in handles:
        handle.remove()

    retnet_bytes, xlstm_bytes = _state_bytes(decode_states)
    (
        retnet_summary_norm,
        retnet_q_adapter_delta_norm,
        retnet_k_adapter_delta_norm,
        retnet_alpha_q,
        retnet_alpha_k,
    ) = _retnet_observability(decode_states)
    router_entropy, load_balance_loss, router_z_loss = _router_metrics(decode_states)
    xlstm_effective_beta, xlstm_memory_norm, xlstm_adapter_delta_norm = _xlstm_observability(decode_states)
    peak_memory_bytes = 0
    if target_device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    per_layer_ms = []
    for layer_index in range(config.num_layers):
        values = layer_times[layer_index]
        per_layer_ms.append(
            {
                "layer_index": layer_index,
                "mean_ms": 0.0 if not values else sum(values) / len(values),
                "calls": len(values),
            }
        )

    metrics = {
        "prefill_tokens_per_sec": float(batch_size * sequence_length / max(prefill_elapsed, 1e-9)),
        "decode_tokens_per_sec": float(batch_size * max(1, decode_steps) / max(decode_elapsed_total, 1e-9)),
        "first_token_latency_ms": float(first_token_latency_ms),
        "per_layer_ms": per_layer_ms,
        "peak_memory_bytes": peak_memory_bytes,
        "paged_kv_page_bytes": model.paged_kv_cache.allocated_bytes,
        "paged_kv_runtime_metadata": model.paged_kv_cache.runtime_metadata(),
        "retnet_state_bytes": retnet_bytes,
        "retnet_summary_norm_mean": float(retnet_summary_norm),
        "retnet_q_adapter_delta_norm_mean": float(retnet_q_adapter_delta_norm),
        "retnet_k_adapter_delta_norm_mean": float(retnet_k_adapter_delta_norm),
        "retnet_alpha_q_mean": float(retnet_alpha_q),
        "retnet_alpha_k_mean": float(retnet_alpha_k),
        "xlstm_memory_state_bytes": xlstm_bytes,
        "xlstm_memory_pool_metadata": model.xlstm_memory_state_pool.to_runtime_metadata(),
        "xlstm_effective_beta_mean": float(xlstm_effective_beta),
        "xlstm_memory_norm_mean": float(xlstm_memory_norm),
        "xlstm_adapter_delta_norm_mean": float(xlstm_adapter_delta_norm),
        "router_entropy_mean": float(router_entropy),
        "load_balance_loss_mean": float(load_balance_loss),
        "router_z_loss_mean": float(router_z_loss),
        "logits_shape": list(logits.shape),
        "decode_logits_shape": list(decode_logits.shape),
    }
    return ResourceReport(
        profile=str(profile),
        preset=str(preset),
        device=str(target_device),
        dtype=dtype_name(target_dtype),
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        decode_steps=int(decode_steps),
        metrics=metrics,
    )
