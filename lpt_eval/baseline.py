"""LPT v2 对比基线报告。"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from lpt_config import BaselineEvalConfig
from lpt_config.profiles import build_lpt_v2_profile_config, parse_profile_list
from lpt_model import LPTV2

from .utils import (
    build_deterministic_input,
    dtype_name,
    next_token_loss,
    resolve_eval_device,
    resolve_eval_dtype,
    set_eval_seed,
)


DEFAULT_BASELINE_EVAL_CONFIG = BaselineEvalConfig()


@dataclass(frozen=True)
class BaselineProfileResult:
    """单个 profile 的 smoke/shape/状态观测结果。"""

    profile: str
    success: bool
    elapsed_ms: float
    selected_backend: str | None
    cache_backend: str | None
    retnet_enabled: bool | None
    xlstm_enabled: bool | None
    logits_shape: tuple[int, ...] = ()
    decode_shape: tuple[int, ...] = ()
    loss: float | None = None
    perplexity: float | None = None
    paged_kv_pages: int | None = None
    paged_kv_bytes: int | None = None
    retnet_token_count: int | None = None
    xlstm_token_count: int | None = None
    router_entropy: float | None = None
    load_balance_loss: float | None = None
    router_z_loss: float | None = None
    expert_count_sum: int | None = None
    error: str | None = None

    def to_dict(self):
        """序列化单个 profile 的结果，便于 JSON 报告落盘。"""
        return {
            "profile": self.profile,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "selected_backend": self.selected_backend,
            "cache_backend": self.cache_backend,
            "retnet_enabled": self.retnet_enabled,
            "xlstm_enabled": self.xlstm_enabled,
            "logits_shape": list(self.logits_shape),
            "decode_shape": list(self.decode_shape),
            "loss": self.loss,
            "perplexity": self.perplexity,
            "paged_kv_pages": self.paged_kv_pages,
            "paged_kv_bytes": self.paged_kv_bytes,
            "retnet_token_count": self.retnet_token_count,
            "xlstm_token_count": self.xlstm_token_count,
            "router_entropy": self.router_entropy,
            "load_balance_loss": self.load_balance_loss,
            "router_z_loss": self.router_z_loss,
            "expert_count_sum": self.expert_count_sum,
            "error": self.error,
        }


@dataclass(frozen=True)
class BaselineReport:
    """统一 JSON / Markdown 的 LPT v2 基线报告。"""

    preset: str
    vocabulary_size: int
    batch_size: int
    sequence_length: int
    decode_steps: int
    device: str
    dtype: str
    results: tuple[BaselineProfileResult, ...]

    @property
    def success(self):
        """所有 profile 都通过时，整体 baseline 才算成功。"""
        return all(result.success for result in self.results)

    def to_dict(self):
        """生成统一 JSON 报告。"""
        return {
            "report_type": "lpt_v2_baseline",
            "success": self.success,
            "preset": self.preset,
            "vocabulary_size": self.vocabulary_size,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "decode_steps": self.decode_steps,
            "device": self.device,
            "dtype": self.dtype,
            "results": [result.to_dict() for result in self.results],
        }

    def to_markdown(self):
        """生成便于人工查看的 Markdown 摘要表。"""
        lines = [
            "# LPT v2 Baseline Report",
            "",
            f"- preset: `{self.preset}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            f"- sequence_length: `{self.sequence_length}`",
            "",
            "| profile | status | backend | cache | loss | ppl | pages | retnet_tokens | xlstm_tokens | router_entropy |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for result in self.results:
            status = "ok" if result.success else "failed"
            lines.append(
                "| {profile} | {status} | {backend} | {cache} | {loss} | {ppl} | {pages} | {ret} | {mem} | {entropy} |".format(
                    profile=result.profile,
                    status=status,
                    backend=result.selected_backend or "",
                    cache=result.cache_backend or "",
                    loss="" if result.loss is None else f"{result.loss:.4f}",
                    ppl="" if result.perplexity is None else f"{result.perplexity:.2f}",
                    pages="" if result.paged_kv_pages is None else result.paged_kv_pages,
                    ret="" if result.retnet_token_count is None else result.retnet_token_count,
                    mem="" if result.xlstm_token_count is None else result.xlstm_token_count,
                    entropy="" if result.router_entropy is None else f"{result.router_entropy:.4f}",
                )
            )
        return "\n".join(lines) + "\n"


def _summarize_states(states):
    """从 layer states 中抽取 RetNet/xLSTM/MoE 观测指标。"""
    first_state = states[0]
    retnet_token_count = None
    if first_state.retnet_assist is not None:
        retnet_token_count = int(first_state.retnet_assist.token_count)
    xlstm_token_count = None
    if first_state.xlstm_memory is not None:
        xlstm_token_count = int(first_state.xlstm_memory.token_count)
    moe_state = first_state.moe
    return {
        "retnet_token_count": retnet_token_count,
        "xlstm_token_count": xlstm_token_count,
        "router_entropy": None if moe_state is None else moe_state.router_entropy,
        "load_balance_loss": None if moe_state is None else moe_state.load_balance_loss,
        "router_z_loss": None if moe_state is None else moe_state.router_z_loss,
        "expert_count_sum": None if moe_state is None else int(sum(moe_state.expert_token_counts)),
    }


def run_lpt_v2_baselines(
    *,
    profiles=DEFAULT_BASELINE_EVAL_CONFIG.profiles,
    preset=DEFAULT_BASELINE_EVAL_CONFIG.preset,
    vocabulary_size=DEFAULT_BASELINE_EVAL_CONFIG.vocabulary_size,
    batch_size=DEFAULT_BASELINE_EVAL_CONFIG.batch_size,
    sequence_length=DEFAULT_BASELINE_EVAL_CONFIG.sequence_length,
    decode_steps=DEFAULT_BASELINE_EVAL_CONFIG.decode_steps,
    device=DEFAULT_BASELINE_EVAL_CONFIG.device,
    dtype=DEFAULT_BASELINE_EVAL_CONFIG.dtype,
    seed=DEFAULT_BASELINE_EVAL_CONFIG.seed,
):
    """运行 6 个 LPT v2 profile 基线并返回报告对象。"""
    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    profile_names = parse_profile_list(profiles)
    results = []

    for index, profile in enumerate(profile_names):
        start_time = perf_counter()
        config = None
        try:
            set_eval_seed(int(seed) + index)
            config = build_lpt_v2_profile_config(profile, preset=preset)
            model = LPTV2(vocabulary_size, config).to(device=target_device, dtype=target_dtype)
            model.eval()
            # baseline 使用确定性 token 输入，只验证 shape、状态与资源路径，不声明质量收益。
            input_ids = build_deterministic_input(
                vocabulary_size,
                batch_size,
                sequence_length,
                offset=1 + index,
                device=target_device,
            )
            with torch.no_grad():
                logits, states = model.prefill(input_ids, request_id=f"baseline-{profile}")
                loss, perplexity = next_token_loss(logits, input_ids)
                decode_states = states
                decode_logits = logits[:, -1:, :]
                for step in range(int(decode_steps)):
                    # decode smoke 逐 token 续接，验证 Paged KV 与 Assist state pool 是否可连续更新。
                    next_ids = build_deterministic_input(
                        vocabulary_size,
                        batch_size,
                        1,
                        offset=sequence_length + step + index + 1,
                        device=target_device,
                    )
                    full_mask = torch.ones(
                        batch_size,
                        sequence_length + step + 1,
                        dtype=torch.long,
                        device=target_device,
                    )
                    decode_logits, decode_states = model.decode(
                        next_ids,
                        attention_mask=full_mask,
                        layer_states=decode_states,
                        request_id=f"baseline-{profile}",
                    )
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)
            state_summary = _summarize_states(decode_states)
            results.append(
                BaselineProfileResult(
                    profile=profile,
                    success=True,
                    elapsed_ms=(perf_counter() - start_time) * 1000.0,
                    selected_backend=model.layers[0].attention_mixer.backend_decision.selected_backend,
                    cache_backend=config.cache_backend,
                    retnet_enabled=config.retnet_assist_enabled,
                    xlstm_enabled=config.xlstm_memory_enabled,
                    logits_shape=tuple(int(value) for value in logits.shape),
                    decode_shape=tuple(int(value) for value in decode_logits.shape),
                    loss=loss,
                    perplexity=perplexity,
                    paged_kv_pages=model.paged_kv_cache.allocated_page_count,
                    paged_kv_bytes=model.paged_kv_cache.allocated_bytes,
                    **state_summary,
                )
            )
        except Exception as exc:  # pragma: no cover - 报告模式需要保留失败项
            results.append(
                BaselineProfileResult(
                    profile=profile,
                    success=False,
                    elapsed_ms=(perf_counter() - start_time) * 1000.0,
                    selected_backend=None,
                    cache_backend=None if config is None else config.cache_backend,
                    retnet_enabled=None if config is None else config.retnet_assist_enabled,
                    xlstm_enabled=None if config is None else config.xlstm_memory_enabled,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return BaselineReport(
        preset=str(preset),
        vocabulary_size=int(vocabulary_size),
        batch_size=int(batch_size),
        sequence_length=int(sequence_length),
        decode_steps=int(decode_steps),
        device=str(target_device),
        dtype=dtype_name(target_dtype),
        results=tuple(results),
    )
