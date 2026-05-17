"""LPT v2 长上下文评测套件。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from lpt_config import LongContextEvalConfig
from lpt_model import load_lpt_v2_checkpoint

from .long_context import (
    _checkpoint_training_metadata,
    run_lpt_v2_long_context_admission,
    run_lpt_v2_long_context_admission_for_model,
)


DEFAULT_CONFIG = LongContextEvalConfig()
DEFAULT_NEEDLE_DEPTHS = (0.2, 0.5, 0.8)


def _normalize_int_tuple(values, *, default):
    """规范化 seq_len/window 这类正整数列表。"""
    if values is None:
        return tuple(int(value) for value in default)
    normalized = tuple(int(value) for value in values)
    if not normalized:
        raise ValueError("至少需要一个整数取值。")
    if any(value <= 0 for value in normalized):
        raise ValueError("整数取值必须全部为正数。")
    return normalized


def _normalize_float_tuple(values, *, default):
    """规范化 needle depth 列表，范围固定在 [0, 1]。"""
    if values is None:
        return tuple(float(value) for value in default)
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("至少需要一个浮点取值。")
    if any(value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError("needle_depth 必须在 [0, 1] 范围内。")
    return normalized


def _mean(values):
    """跳过 None 后求均值。"""
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _summarize_cases(case_results):
    """聚合多 case 的状态分布和平均代理指标。"""
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
        "all_mechanism_ready": mechanism_ready_count == len(case_results),
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


def _format_float(value, digits=4):
    """Markdown 中的可选浮点格式化。"""
    return "n/a" if value is None else f"{float(value):.{digits}f}"


@dataclass(frozen=True)
class LongContextSuiteReport:
    """长上下文多长度、多深度评测报告。"""

    preset: str
    device: str
    dtype: str
    vocabulary_size: int
    sequence_lengths: tuple[int, ...]
    attention_window_sizes: tuple[int, ...]
    needle_depths: tuple[float, ...]
    case_results: tuple[dict, ...]
    summary: dict
    checkpoint_path: str | None = None
    checkpoint_metadata: dict | None = None

    def to_dict(self):
        """生成 suite JSON 载荷。"""
        payload = {
            "report_type": "lpt_v2_long_context_suite",
            "preset": self.preset,
            "device": self.device,
            "dtype": self.dtype,
            "vocabulary_size": self.vocabulary_size,
            "sequence_lengths": list(self.sequence_lengths),
            "attention_window_sizes": list(self.attention_window_sizes),
            "needle_depths": list(self.needle_depths),
            "summary": dict(self.summary),
            "case_results": list(self.case_results),
        }
        if self.checkpoint_path is not None:
            payload["checkpoint_path"] = self.checkpoint_path
            payload["checkpoint_metadata"] = dict(self.checkpoint_metadata or {})
        return payload

    def to_markdown(self):
        """生成 suite Markdown 报告。"""
        checkpoint_lines = []
        if self.checkpoint_path is not None:
            checkpoint_lines = [
                f"- checkpoint: `{self.checkpoint_path}`",
                f"- training_stage: `{(self.checkpoint_metadata or {}).get('training_stage')}`",
                f"- global_step: `{(self.checkpoint_metadata or {}).get('global_step')}`",
            ]
        lines = [
            "# LPT v2 Long Context Suite",
            "",
            f"- preset: `{self.preset}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            f"- sequence_lengths: `{list(self.sequence_lengths)}`",
            f"- attention_window_sizes: `{list(self.attention_window_sizes)}`",
            f"- needle_depths: `{list(self.needle_depths)}`",
            *checkpoint_lines,
            "",
            "## Summary",
            "",
            f"- case_count: `{self.summary['case_count']}`",
            f"- mechanism_ready_count: `{self.summary['mechanism_ready_count']}`",
            f"- status_counts: `{self.summary['status_counts']}`",
            f"- avg_assist_loss: `{_format_float(self.summary['avg_assist_loss'])}`",
            f"- avg_needle_rank: `{_format_float(self.summary['avg_needle_rank'])}`",
            "",
            "## Cases",
            "",
            "| seq_len | window | depth | status | ready | loss | ppl | needle_rank | needle_logprob |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for result in self.case_results:
            needle = result["metrics"]["needle"]
            long_text = result["metrics"]["long_text_ppl"]
            mechanism = result["metrics"]["mechanism"]
            decision = result["metrics"]["quality_decision"]
            lines.append(
                "| {seq} | {window} | {depth:.2f} | {status} | {ready} | {loss} | {ppl} | {rank} | {logprob} |".format(
                    seq=result["sequence_length"],
                    window=result["attention_window_size"],
                    depth=float(result["needle_depth"]),
                    status=decision["status"],
                    ready=int(bool(mechanism.get("mechanism_ready"))),
                    loss=_format_float(long_text.get("assist_loss")),
                    ppl=_format_float(long_text.get("assist_ppl")),
                    rank=needle.get("assist_rank"),
                    logprob=_format_float(needle.get("assist_logprob")),
                )
            )
        return "\n".join(lines) + "\n"


def run_lpt_v2_long_context_suite(
    *,
    preset=DEFAULT_CONFIG.preset,
    vocabulary_size=DEFAULT_CONFIG.vocabulary_size,
    sequence_lengths=None,
    attention_window_sizes=None,
    needle_depths=DEFAULT_NEEDLE_DEPTHS,
    checkpoint_path=None,
    device=DEFAULT_CONFIG.device,
    dtype=DEFAULT_CONFIG.dtype,
    seed=DEFAULT_CONFIG.seed,
):
    """运行 v2 长上下文多长度、多 depth 套件。"""
    attention_window_sizes = _normalize_int_tuple(
        attention_window_sizes,
        default=(DEFAULT_CONFIG.attention_window_size,),
    )
    if sequence_lengths is None:
        sequence_lengths = tuple(window * 2 + 4 for window in attention_window_sizes)
    sequence_lengths = _normalize_int_tuple(sequence_lengths, default=sequence_lengths)
    needle_depths = _normalize_float_tuple(needle_depths, default=DEFAULT_NEEDLE_DEPTHS)

    case_results = []
    checkpoint_metadata = None
    checkpoint_path_text = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=True)
        checkpoint_metadata = _checkpoint_training_metadata(loaded.checkpoint)
        checkpoint_path_text = str(checkpoint_path)
        preset = loaded.checkpoint["model_config"].get("model_size_preset", "checkpoint")
        vocabulary_size = int(loaded.model.token_embedding.num_embeddings)
        # checkpoint 模式复用同一个模型对象，单个 case 内部会释放 request-bound 状态。
        for window in attention_window_sizes:
            for seq_len in sequence_lengths:
                if int(seq_len) <= int(window):
                    continue
                for depth in needle_depths:
                    report = run_lpt_v2_long_context_admission_for_model(
                        model=loaded.model,
                        preset=preset,
                        checkpoint_path=checkpoint_path,
                        checkpoint_metadata=checkpoint_metadata,
                        sequence_length=seq_len,
                        attention_window_size=window,
                        device=device,
                        dtype=dtype,
                        needle_depth=depth,
                    )
                    case_results.append(report.to_dict())
    else:
        for case_index, window in enumerate(attention_window_sizes):
            for seq_len in sequence_lengths:
                if int(seq_len) <= int(window):
                    continue
                for depth in needle_depths:
                    # 随机初始化模式每个 case 独立构造模型，避免不同窗口配置互相污染。
                    report = run_lpt_v2_long_context_admission(
                        preset=preset,
                        vocabulary_size=vocabulary_size,
                        sequence_length=seq_len,
                        attention_window_size=window,
                        device=device,
                        dtype=dtype,
                        seed=int(seed) + case_index,
                        needle_depth=depth,
                    )
                    case_results.append(report.to_dict())

    if not case_results:
        raise ValueError("没有可运行的长上下文 case；请确保 seq_len 大于 attention_window_size。")
    summary = _summarize_cases(case_results)
    first_case = case_results[0]
    return LongContextSuiteReport(
        preset=str(preset),
        device=first_case["device"],
        dtype=first_case["dtype"],
        vocabulary_size=int(vocabulary_size),
        sequence_lengths=sequence_lengths,
        attention_window_sizes=attention_window_sizes,
        needle_depths=needle_depths,
        case_results=tuple(case_results),
        summary=summary,
        checkpoint_path=checkpoint_path_text,
        checkpoint_metadata=checkpoint_metadata,
    )
