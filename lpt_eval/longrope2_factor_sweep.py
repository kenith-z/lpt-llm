"""LPT v2 LongRoPE2 候选因子 sweep 评测。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
from pathlib import Path

import torch

from lpt_config import (
    GlobalConfig,
    build_longrope2_uniform_factors,
    load_longrope2_factors_file,
)
from lpt_model import LPTV2, load_lpt_v2_checkpoint

from .long_context import (
    _checkpoint_training_metadata,
    run_lpt_v2_long_context_admission_for_model,
)
from .long_context_suite import DEFAULT_NEEDLE_DEPTHS, _normalize_float_tuple, _normalize_int_tuple
from .utils import resolve_eval_device, resolve_eval_dtype


DEFAULT_SEQUENCE_LENGTHS = (2052,)
DEFAULT_ATTENTION_WINDOW_SIZES = (2048,)


def summarize_longrope2_factors(factors):
    """生成 LongRoPE2 factors 的轻量摘要，避免 Markdown 展开大数组。"""
    if factors is None:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    values = tuple(float(value) for value in factors)
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _normalize_candidate_name(name):
    """规范化候选名称，名称会用于报告和去重。"""
    normalized = str(name).strip()
    if not normalized:
        raise ValueError("LongRoPE2 候选名称不能为空。")
    return normalized


def _normalize_candidate_factors(model_config, factors):
    """校验候选 factors 数量必须等于 head_dim/2。"""
    values = tuple(float(value) for value in factors)
    rotary_dims = int(model_config.head_dim) // 2
    if len(values) != rotary_dims:
        raise ValueError(
            "LongRoPE2 候选因子数量不匹配: "
            f"{len(values)} != head_dim/2({rotary_dims})。"
        )
    if any(value <= 0 for value in values):
        raise ValueError("LongRoPE2 候选因子必须全部大于 0。")
    return values


def _mean(values):
    """跳过 None 后求均值。"""
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value, digits=4):
    """Markdown 中的可选浮点格式化。"""
    return "n/a" if value is None else f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class LongRoPE2FactorCandidate:
    """一组待评估的 LongRoPE2 long factors。"""

    name: str
    long_factors: tuple[float, ...]
    source: str
    factor_max_sequence_length: int | None = None

    def to_dict(self):
        """生成候选因子的 JSON 摘要。"""
        return {
            "name": self.name,
            "source": self.source,
            "factor_max_sequence_length": self.factor_max_sequence_length,
            "long_factors": list(self.long_factors),
            "longrope2_factor_summary": summarize_longrope2_factors(self.long_factors),
        }


def build_uniform_factor_candidate(name, factor, model_config, *, source="uniform"):
    """按单个缩放值生成与 head_dim/2 等长的候选因子。"""
    rotary_dims = int(model_config.head_dim) // 2
    return LongRoPE2FactorCandidate(
        name=_normalize_candidate_name(name),
        long_factors=tuple(float(factor) for _ in range(rotary_dims)),
        source=f"{source}:{float(factor):.6g}",
    )


def build_bootstrap_factor_candidate(name, model_config, sequence_length):
    """复用 v2 当前 bootstrap 规则生成候选因子。"""
    return LongRoPE2FactorCandidate(
        name=_normalize_candidate_name(name),
        long_factors=tuple(build_longrope2_uniform_factors(model_config, sequence_length)),
        source=f"bootstrap:sequence_length={int(sequence_length)}",
        factor_max_sequence_length=int(sequence_length),
    )


def _append_candidate(candidates, seen_names, candidate, model_config):
    """校验、去重并追加一个候选因子。"""
    normalized = LongRoPE2FactorCandidate(
        name=_normalize_candidate_name(candidate.name),
        long_factors=_normalize_candidate_factors(model_config, candidate.long_factors),
        source=str(candidate.source),
        factor_max_sequence_length=(
            None
            if candidate.factor_max_sequence_length is None
            else int(candidate.factor_max_sequence_length)
        ),
    )
    if normalized.name in seen_names:
        raise ValueError(f"LongRoPE2 候选名称重复: {normalized.name}")
    seen_names.add(normalized.name)
    candidates.append(normalized)


def build_longrope2_factor_candidates(
    model_config,
    *,
    include_current=True,
    include_bootstrap=True,
    bootstrap_sequence_length=None,
    uniform_factor_candidates=(),
    factor_file_candidates=(),
    explicit_candidates=(),
):
    """从 checkpoint 配置和 CLI 输入生成候选因子列表。"""
    candidates = []
    seen_names = set()

    for candidate in explicit_candidates:
        _append_candidate(candidates, seen_names, candidate, model_config)

    if include_current and model_config.longrope2_long_factors is not None:
        _append_candidate(
            candidates,
            seen_names,
            LongRoPE2FactorCandidate(
                name="current",
                long_factors=tuple(model_config.longrope2_long_factors),
                source="checkpoint:model_config",
                factor_max_sequence_length=model_config.longrope2_factor_max_sequence_length,
            ),
            model_config,
        )

    if include_bootstrap:
        bootstrap_length = bootstrap_sequence_length
        if bootstrap_length is None:
            bootstrap_length = (
                model_config.longrope2_factor_max_sequence_length
                or model_config.longrope2_target_length
            )
        _append_candidate(
            candidates,
            seen_names,
            build_bootstrap_factor_candidate("bootstrap", model_config, bootstrap_length),
            model_config,
        )

    for name, factor in uniform_factor_candidates:
        _append_candidate(
            candidates,
            seen_names,
            build_uniform_factor_candidate(name, factor, model_config),
            model_config,
        )

    for name, factor_path in factor_file_candidates:
        _append_candidate(
            candidates,
            seen_names,
            LongRoPE2FactorCandidate(
                name=name,
                long_factors=tuple(load_longrope2_factors_file(factor_path)),
                source=f"factors_file:{Path(factor_path)}",
            ),
            model_config,
        )

    if not candidates:
        raise ValueError("至少需要提供一组 LongRoPE2 候选因子。")
    return tuple(candidates)


def _build_candidate_model_config(base_config, candidate, *, max_sequence_length):
    """在评测进程内生成临时 ModelConfig，不写回 checkpoint。"""
    factor_max_sequence_length = max(
        int(max_sequence_length),
        int(candidate.factor_max_sequence_length or 0),
        int(base_config.longrope2_factor_max_sequence_length or 0),
    )
    target_length = max(
        int(base_config.longrope2_target_length),
        int(max_sequence_length),
        factor_max_sequence_length,
    )
    return base_config.with_overrides(
        longrope2_long_factors=candidate.long_factors,
        longrope2_factor_max_sequence_length=factor_max_sequence_length,
        longrope2_target_length=target_length,
    )


def _load_candidate_model(loaded_checkpoint, candidate_config, *, device, dtype):
    """用候选配置重建模型，再加载同一 checkpoint 权重。"""
    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    GlobalConfig.parameter_dtype = target_dtype
    model = LPTV2(
        int(loaded_checkpoint.model.token_embedding.num_embeddings),
        candidate_config,
    )
    model.load_state_dict(loaded_checkpoint.checkpoint["model_state_dict"], strict=True)
    model.to(device=target_device, dtype=target_dtype).eval()
    return model


def _release_candidate_model(model):
    """释放候选模型，降低 sweep 多候选循环的显存残留。"""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summarize_candidate_cases(case_results):
    """汇总单个候选在所有长上下文 case 上的表现。"""
    if not case_results:
        return {
            "case_count": 0,
            "status_counts": {},
            "mechanism_ready_count": 0,
            "avg_assist_loss": None,
            "avg_assist_ppl": None,
            "avg_needle_rank": None,
            "avg_needle_logprob": None,
        }
    statuses = Counter(
        result["metrics"]["quality_decision"]["status"]
        for result in case_results
    )
    mechanism_ready_count = sum(
        1
        for result in case_results
        if result["metrics"]["mechanism"].get("mechanism_ready")
    )
    return {
        "case_count": len(case_results),
        "status_counts": dict(sorted(statuses.items())),
        "mechanism_ready_count": mechanism_ready_count,
        "avg_assist_loss": _mean(
            result["metrics"]["long_text_ppl"].get("assist_loss")
            for result in case_results
        ),
        "avg_assist_ppl": _mean(
            result["metrics"]["long_text_ppl"].get("assist_ppl")
            for result in case_results
        ),
        "avg_needle_rank": _mean(
            result["metrics"]["needle"].get("assist_rank")
            for result in case_results
        ),
        "avg_needle_logprob": _mean(
            result["metrics"]["needle"].get("assist_logprob")
            for result in case_results
        ),
    }


@dataclass(frozen=True)
class LongRoPE2FactorSweepReport:
    """LongRoPE2 候选因子 sweep 报告。"""

    checkpoint_path: str
    checkpoint_metadata: dict
    sequence_lengths: tuple[int, ...]
    attention_window_sizes: tuple[int, ...]
    needle_depths: tuple[float, ...]
    device: str
    dtype: str
    candidates: tuple[dict, ...]
    summary: dict

    def to_dict(self):
        """生成 factor sweep JSON 载荷。"""
        return {
            "report_type": "lpt_v2_longrope2_factor_sweep",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_metadata": dict(self.checkpoint_metadata),
            "sequence_lengths": list(self.sequence_lengths),
            "attention_window_sizes": list(self.attention_window_sizes),
            "needle_depths": list(self.needle_depths),
            "device": self.device,
            "dtype": self.dtype,
            "summary": dict(self.summary),
            "candidates": list(self.candidates),
        }

    def to_markdown(self):
        """生成 factor sweep Markdown 报告。"""
        lines = [
            "# LPT v2 LongRoPE2 Factor Sweep",
            "",
            f"- checkpoint: `{self.checkpoint_path}`",
            f"- training_stage: `{self.checkpoint_metadata.get('training_stage')}`",
            f"- global_step: `{self.checkpoint_metadata.get('global_step')}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            f"- sequence_lengths: `{list(self.sequence_lengths)}`",
            f"- attention_window_sizes: `{list(self.attention_window_sizes)}`",
            f"- needle_depths: `{list(self.needle_depths)}`",
            "",
            "## Summary",
            "",
            f"- candidate_count: `{self.summary['candidate_count']}`",
            f"- best_by_avg_loss: `{self.summary.get('best_by_avg_loss')}`",
            "",
            "## Candidates",
            "",
            "| name | source | factor_min | factor_max | factor_mean | cases | ready | avg_loss | avg_ppl | avg_rank | status_counts |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for candidate in self.candidates:
            factor_summary = candidate["candidate"]["longrope2_factor_summary"]
            summary = candidate["summary"]
            lines.append(
                "| {name} | {source} | {factor_min} | {factor_max} | {factor_mean} | {cases} | {ready} | {loss} | {ppl} | {rank} | {statuses} |".format(
                    name=candidate["candidate"]["name"],
                    source=candidate["candidate"]["source"],
                    factor_min=_format_float(factor_summary["min"]),
                    factor_max=_format_float(factor_summary["max"]),
                    factor_mean=_format_float(factor_summary["mean"]),
                    cases=summary.get("case_count", 0),
                    ready=summary.get("mechanism_ready_count", 0),
                    loss=_format_float(summary.get("avg_assist_loss")),
                    ppl=_format_float(summary.get("avg_assist_ppl")),
                    rank=_format_float(summary.get("avg_needle_rank")),
                    statuses=summary.get("status_counts", {}),
                )
            )
            if candidate.get("error"):
                lines.append(f"- `{candidate['candidate']['name']}` error: {candidate['error']}")
        return "\n".join(lines) + "\n"


def _build_sweep_summary(candidates):
    """从候选摘要中挑选平均 loss 最低的候选。"""
    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["summary"].get("avg_assist_loss") is not None
    ]
    best_by_loss = None
    if valid_candidates:
        best_by_loss = min(
            valid_candidates,
            key=lambda item: item["summary"]["avg_assist_loss"],
        )["candidate"]["name"]
    return {
        "candidate_count": len(candidates),
        "best_by_avg_loss": best_by_loss,
    }


def run_lpt_v2_longrope2_factor_sweep(
    *,
    checkpoint_path,
    sequence_lengths=DEFAULT_SEQUENCE_LENGTHS,
    attention_window_sizes=DEFAULT_ATTENTION_WINDOW_SIZES,
    needle_depths=DEFAULT_NEEDLE_DEPTHS,
    include_current=True,
    include_bootstrap=True,
    bootstrap_sequence_length=None,
    uniform_factor_candidates=(),
    factor_file_candidates=(),
    explicit_candidates=(),
    device="auto",
    dtype="auto",
):
    """对同一 checkpoint 临时替换 LongRoPE2 factors 并运行长上下文代理评测。"""
    checkpoint_path = Path(checkpoint_path)
    sequence_lengths = _normalize_int_tuple(sequence_lengths, default=DEFAULT_SEQUENCE_LENGTHS)
    attention_window_sizes = _normalize_int_tuple(
        attention_window_sizes,
        default=DEFAULT_ATTENTION_WINDOW_SIZES,
    )
    needle_depths = _normalize_float_tuple(needle_depths, default=DEFAULT_NEEDLE_DEPTHS)
    loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=True)
    base_config = loaded.model.config
    candidates = build_longrope2_factor_candidates(
        base_config,
        include_current=include_current,
        include_bootstrap=include_bootstrap,
        bootstrap_sequence_length=bootstrap_sequence_length,
        uniform_factor_candidates=uniform_factor_candidates,
        factor_file_candidates=factor_file_candidates,
        explicit_candidates=explicit_candidates,
    )
    max_sequence_length = max(sequence_lengths)
    GlobalConfig.inference_rope_cache_max_sequence_length = max(
        int(GlobalConfig.inference_rope_cache_max_sequence_length),
        int(max_sequence_length),
    )
    checkpoint_metadata = _checkpoint_training_metadata(loaded.checkpoint)
    candidate_reports = []
    for candidate in candidates:
        case_results = []
        error = None
        candidate_config = _build_candidate_model_config(
            base_config,
            candidate,
            max_sequence_length=max_sequence_length,
        )
        model = None
        try:
            # 每个候选使用独立模型实例，确保 RoPE factors 只影响当前候选 case。
            model = _load_candidate_model(
                loaded,
                candidate_config,
                device=device,
                dtype=dtype,
            )
            for window in attention_window_sizes:
                for seq_len in sequence_lengths:
                    if int(seq_len) <= int(window):
                        continue
                    for depth in needle_depths:
                        report = run_lpt_v2_long_context_admission_for_model(
                            model=model,
                            preset=base_config.model_size_preset,
                            checkpoint_path=checkpoint_path,
                            checkpoint_metadata=checkpoint_metadata,
                            sequence_length=seq_len,
                            attention_window_size=window,
                            device=device,
                            dtype=dtype,
                            needle_depth=depth,
                        )
                        case_results.append(report.to_dict())
        except Exception as exc:  # pragma: no cover - sweep 需要保留其它候选结果
            error = str(exc)
        finally:
            if model is not None:
                _release_candidate_model(model)
        candidate_payload = {
            "candidate": candidate.to_dict(),
            "model_config_overrides": {
                "longrope2_factor_max_sequence_length": candidate_config.longrope2_factor_max_sequence_length,
                "longrope2_target_length": candidate_config.longrope2_target_length,
            },
            "summary": _summarize_candidate_cases(case_results),
            "case_results": case_results,
        }
        if error is not None:
            candidate_payload["error"] = error
            candidate_payload["summary"]["status_counts"] = {"error": 1}
        candidate_reports.append(candidate_payload)

    if not any(candidate["case_results"] for candidate in candidate_reports):
        raise ValueError("没有可运行的 LongRoPE2 sweep case；请确保 seq_len 大于 attention_window_size。")
    first_case = next(candidate["case_results"][0] for candidate in candidate_reports if candidate["case_results"])
    return LongRoPE2FactorSweepReport(
        checkpoint_path=str(checkpoint_path),
        checkpoint_metadata=checkpoint_metadata,
        sequence_lengths=sequence_lengths,
        attention_window_sizes=attention_window_sizes,
        needle_depths=needle_depths,
        device=first_case["device"],
        dtype=first_case["dtype"],
        candidates=tuple(candidate_reports),
        summary=_build_sweep_summary(candidate_reports),
    )
