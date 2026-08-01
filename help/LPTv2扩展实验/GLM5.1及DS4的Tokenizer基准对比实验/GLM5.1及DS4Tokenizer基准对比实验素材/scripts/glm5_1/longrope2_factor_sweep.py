"""LongRoPE2 候选因子 sweep 评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
from pathlib import Path
import time

import torch

from lpt_config import (
    GenerationConfig,
    GlobalConfig,
    LoRAConfig,
    build_longrope2_uniform_factors,
    build_model_config_from_checkpoint,
    load_longrope2_factors_file,
)
from lpt_lora.adapter import attach_lora_adapters
from lpt_model import LPT
from lpt_training import load_checkpoint
from lpt_workflows.common import TOKENIZER_PATH, build_local_tokenizer

from .long_context import (
    DEFAULT_GENERATION_CONFIG,
    evaluate_needle_in_a_haystack,
    evaluate_long_text_perplexity,
    evaluate_retrieval_qa,
    summarize_longrope_factors,
    _resolve_checkpoint_root,
)


@dataclass(frozen=True)
class LongRoPE2FactorCandidate:
    """一组待评估的 LongRoPE2 long factors。"""

    name: str
    long_factors: tuple[float, ...]
    source: str
    factor_max_sequence_length: int | None = None

    def to_report_dict(self):
        return {
            "name": self.name,
            "source": self.source,
            "factor_max_sequence_length": self.factor_max_sequence_length,
            "long_factors": list(self.long_factors),
            "longrope_factor_summary": summarize_longrope_factors(self.long_factors),
        }


@dataclass(frozen=True)
class LongRoPE2FactorSweepConfig:
    """LongRoPE2 因子 sweep 评估配置。"""

    model_type: str = "chat_sft"
    checkpoint_root: str | None = None
    lora_base_source: str = "text_pretrain"
    cache_strategy: str = "session_rebuild"
    text_manifest_path: str = "data/manifests/text_pretrain.json"
    needle_lengths: tuple[int, ...] = (2048, 4096)
    needle_depths: tuple[float, ...] = (0.2, 0.5, 0.8)
    retrieval_lengths: tuple[int, ...] = (2048, 4096)
    ppl_lengths: tuple[int, ...] = (1024, 2048)
    ppl_max_windows: int = 4
    seed: int = 42
    max_generation_tokens: int = 48
    output_format: str = "both"
    output_dir: str | None = None
    include_current_factors: bool = True
    include_bootstrap_factors: bool = True
    bootstrap_sequence_length: int | None = None
    uniform_factor_candidates: tuple[tuple[str, float], ...] = ()
    factor_file_candidates: tuple[tuple[str, str], ...] = ()
    explicit_candidates: tuple[LongRoPE2FactorCandidate, ...] = ()


def _normalize_candidate_name(name):
    normalized_name = str(name).strip()
    if not normalized_name:
        raise ValueError("候选因子名称不能为空。")
    return normalized_name


def _normalize_candidate_factors(values):
    factors = tuple(float(value) for value in values)
    if not factors:
        raise ValueError("LongRoPE2 候选因子不能为空。")
    if any(factor <= 0 for factor in factors):
        raise ValueError("LongRoPE2 候选因子必须全部大于 0。")
    return factors


def build_uniform_factor_candidate(name, factor, model_config, *, source="uniform"):
    """按单个缩放值生成 head_dim/2 长度的候选因子。"""
    factor = float(factor)
    rotary_dims = int(model_config.head_dim) // 2
    if rotary_dims <= 0:
        raise ValueError("head_dim 必须至少包含一组 rotary 维度。")
    return LongRoPE2FactorCandidate(
        name=_normalize_candidate_name(name),
        long_factors=tuple(factor for _ in range(rotary_dims)),
        source=source,
    )


def build_bootstrap_factor_candidate(name, model_config, sequence_length):
    """复用当前工程的 bootstrap 规则生成候选因子。"""
    factors = build_longrope2_uniform_factors(model_config, sequence_length)
    return LongRoPE2FactorCandidate(
        name=_normalize_candidate_name(name),
        long_factors=_normalize_candidate_factors(factors),
        source=f"bootstrap:sequence_length={int(sequence_length)}",
        factor_max_sequence_length=int(sequence_length),
    )


def _append_candidate(candidate_list, seen_names, candidate):
    if candidate.name in seen_names:
        raise ValueError(f"LongRoPE2 候选因子名称重复: {candidate.name}")
    seen_names.add(candidate.name)
    candidate_list.append(candidate)


def build_longrope2_factor_candidates(model_config, sweep_config):
    """根据 sweep 配置生成候选因子列表。"""
    candidates = []
    seen_names = set()

    for candidate in sweep_config.explicit_candidates:
        normalized_candidate = LongRoPE2FactorCandidate(
            name=_normalize_candidate_name(candidate.name),
            long_factors=_normalize_candidate_factors(candidate.long_factors),
            source=candidate.source,
            factor_max_sequence_length=candidate.factor_max_sequence_length,
        )
        _append_candidate(candidates, seen_names, normalized_candidate)

    if sweep_config.include_current_factors and model_config.longrope2_long_factors is not None:
        current_candidate = LongRoPE2FactorCandidate(
            name="current",
            long_factors=_normalize_candidate_factors(model_config.longrope2_long_factors),
            source="checkpoint:model_config",
            factor_max_sequence_length=model_config.longrope2_factor_max_sequence_length,
        )
        _append_candidate(candidates, seen_names, current_candidate)

    if sweep_config.include_bootstrap_factors:
        bootstrap_sequence_length = sweep_config.bootstrap_sequence_length
        if bootstrap_sequence_length is None:
            bootstrap_sequence_length = (
                model_config.longrope2_factor_max_sequence_length
                or model_config.longrope2_target_length
            )
        bootstrap_candidate = build_bootstrap_factor_candidate(
            "bootstrap",
            model_config,
            bootstrap_sequence_length,
        )
        _append_candidate(candidates, seen_names, bootstrap_candidate)

    for name, factor in sweep_config.uniform_factor_candidates:
        candidate = build_uniform_factor_candidate(
            name,
            factor,
            model_config,
            source=f"uniform_factor:{float(factor):.6g}",
        )
        _append_candidate(candidates, seen_names, candidate)

    for name, factor_path in sweep_config.factor_file_candidates:
        factors = load_longrope2_factors_file(factor_path)
        candidate = LongRoPE2FactorCandidate(
            name=_normalize_candidate_name(name),
            long_factors=_normalize_candidate_factors(factors),
            source=f"factors_file:{Path(factor_path)}",
        )
        _append_candidate(candidates, seen_names, candidate)

    if not candidates:
        raise ValueError("至少需要提供一组 LongRoPE2 候选因子。")
    return tuple(candidates)


def _build_generation_config(max_generation_tokens):
    return GenerationConfig(
        do_sample=DEFAULT_GENERATION_CONFIG.do_sample,
        temperature=DEFAULT_GENERATION_CONFIG.temperature,
        top_k=DEFAULT_GENERATION_CONFIG.top_k,
        top_p=DEFAULT_GENERATION_CONFIG.top_p,
        max_length=int(max_generation_tokens),
        repetition_penalty=DEFAULT_GENERATION_CONFIG.repetition_penalty,
        repetition_window_size=DEFAULT_GENERATION_CONFIG.repetition_window_size,
    )


def _build_base_checkpoint_summary(checkpoint, checkpoint_root, model_config):
    architecture_metadata = dict(checkpoint.get("model_architecture_metadata", {}))
    return {
        "checkpoint_path": str(checkpoint_root.with_suffix(".pth")),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "model_config_schema_version": checkpoint.get("model_config_schema_version"),
        "training_stage": checkpoint.get("training_stage"),
        "training_mode": checkpoint.get("training_mode"),
        "source_manifest": checkpoint.get("source_manifest"),
        "model_config": model_config.to_dict(),
        "model_architecture_metadata": architecture_metadata,
        "longrope_factor_summary": summarize_longrope_factors(
            model_config.longrope2_long_factors
        ),
    }


def _build_candidate_model_config(base_model_config, candidate):
    factor_max_sequence_length = (
        candidate.factor_max_sequence_length
        if candidate.factor_max_sequence_length is not None
        else base_model_config.longrope2_factor_max_sequence_length
    )
    return base_model_config.with_overrides(
        longrope2_long_factors=candidate.long_factors,
        longrope2_factor_max_sequence_length=factor_max_sequence_length,
    )


def _load_candidate_model(checkpoint, tokenizer, model_config, *, model_type):
    model = LPT(vocabulary_size=len(tokenizer), config=model_config).to(GlobalConfig.device)
    if checkpoint.get("training_mode") == "lora" or model_type == "chat_lora":
        attach_lora_adapters(model, config=LoRAConfig())
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def _release_candidate_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_candidate(
    *,
    candidate,
    checkpoint,
    tokenizer,
    base_model_config,
    sweep_config,
    generation_config,
):
    candidate_started_at = time.perf_counter()
    model_config = _build_candidate_model_config(base_model_config, candidate)
    candidate_report = {
        **candidate.to_report_dict(),
        "model_config_overrides": {
            "longrope2_long_factors": list(candidate.long_factors),
            "longrope2_factor_max_sequence_length": model_config.longrope2_factor_max_sequence_length,
        },
        "status": "ok",
        "tasks": {},
    }

    model = None
    try:
        model = _load_candidate_model(
            checkpoint,
            tokenizer,
            model_config,
            model_type=sweep_config.model_type,
        )
        candidate_report["tasks"]["needle_in_a_haystack"] = evaluate_needle_in_a_haystack(
            model,
            tokenizer,
            cache_strategy=sweep_config.cache_strategy,
            generation_config=generation_config,
            lengths=sweep_config.needle_lengths,
            depths=sweep_config.needle_depths,
            seed=sweep_config.seed,
        )
        candidate_report["tasks"]["long_text_ppl"] = evaluate_long_text_perplexity(
            model,
            tokenizer,
            manifest_path=sweep_config.text_manifest_path,
            lengths=sweep_config.ppl_lengths,
            max_windows=sweep_config.ppl_max_windows,
        )
        candidate_report["tasks"]["qa_retrieval"] = evaluate_retrieval_qa(
            model,
            tokenizer,
            cache_strategy=sweep_config.cache_strategy,
            generation_config=generation_config,
            lengths=sweep_config.retrieval_lengths,
            seed=sweep_config.seed + 1000,
        )
    except Exception as error:
        candidate_report["status"] = "error"
        candidate_report["error_type"] = type(error).__name__
        candidate_report["error_message"] = str(error)
    finally:
        if model is not None:
            _release_candidate_model(model)

    candidate_report["runtime"] = {
        "latency_sec": round(time.perf_counter() - candidate_started_at, 6),
    }
    return candidate_report


def evaluate_longrope2_factor_sweep(config: LongRoPE2FactorSweepConfig):
    """对多组 LongRoPE2 候选因子运行长上下文评测。"""
    checkpoint_root = _resolve_checkpoint_root(
        config.model_type,
        config.checkpoint_root,
        lora_base_source=config.lora_base_source,
    )
    checkpoint = load_checkpoint(checkpoint_root)
    tokenizer = build_local_tokenizer(TOKENIZER_PATH)
    base_model_config = build_model_config_from_checkpoint(checkpoint)
    candidates = build_longrope2_factor_candidates(base_model_config, config)
    generation_config = _build_generation_config(config.max_generation_tokens)

    started_at = time.perf_counter()
    report = {
        "report_type": "longrope2_factor_sweep",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": _build_base_checkpoint_summary(
            checkpoint,
            checkpoint_root,
            base_model_config,
        ),
        "runtime": {
            "device": str(GlobalConfig.device),
            "cache_strategy": config.cache_strategy,
            "generation_config": asdict(generation_config),
        },
        "config": {
            "model_type": config.model_type,
            "checkpoint_root": str(checkpoint_root),
            "lora_base_source": config.lora_base_source,
            "text_manifest_path": str(Path(config.text_manifest_path)),
            "needle_lengths": list(config.needle_lengths),
            "needle_depths": list(config.needle_depths),
            "retrieval_lengths": list(config.retrieval_lengths),
            "ppl_lengths": list(config.ppl_lengths),
            "ppl_max_windows": config.ppl_max_windows,
            "seed": config.seed,
            "candidate_count": len(candidates),
        },
        "candidates": [],
    }

    for candidate in candidates:
        report["candidates"].append(
            _evaluate_candidate(
                candidate=candidate,
                checkpoint=checkpoint,
                tokenizer=tokenizer,
                base_model_config=base_model_config,
                sweep_config=config,
                generation_config=generation_config,
            )
        )

    report["runtime"]["total_latency_sec"] = round(time.perf_counter() - started_at, 6)
    return report, checkpoint_root


def _format_rate(value):
    if value is None:
        return "-"
    return f"{float(value):.4g}"


def _extract_generation_rate(candidate_report, task_name):
    if candidate_report.get("status") != "ok":
        return None
    task_report = candidate_report.get("tasks", {}).get(task_name)
    if task_report is None:
        return None
    return task_report.get("aggregate", {}).get("exact_match_rate")


def _extract_ppl_summary(candidate_report):
    if candidate_report.get("status") != "ok":
        return "-"
    task_report = candidate_report.get("tasks", {}).get("long_text_ppl")
    if task_report is None:
        return "-"
    items = []
    for item in task_report.get("aggregate", ()):
        ppl = item.get("perplexity")
        ppl_text = "overflow" if ppl is None else f"{float(ppl):.4g}"
        items.append(f"{item.get('window_size')}:{ppl_text}")
    return ", ".join(items) if items else "-"


def format_longrope2_factor_sweep_report_markdown(report):
    """把 LongRoPE2 因子 sweep 结果格式化为 Markdown。"""
    checkpoint = report["checkpoint"]
    lines = [
        "# LongRoPE2 候选因子 Sweep 报告",
        "",
        "## Checkpoint",
        f"- checkpoint_path: {checkpoint['checkpoint_path']}",
        f"- training_stage: {checkpoint['training_stage']}",
        f"- source_manifest: {checkpoint['source_manifest']}",
        "",
        "## Runtime",
        f"- device: {report['runtime']['device']}",
        f"- cache_strategy: {report['runtime']['cache_strategy']}",
        f"- total_latency_sec: {report['runtime']['total_latency_sec']}",
        "",
        "## Candidates",
        "| name | status | source | factor_mode | min_factor | max_factor | needle_exact | retrieval_exact | ppl | latency_sec |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in report["candidates"]:
        factor_summary = candidate["longrope_factor_summary"]
        lines.append(
            "| {name} | {status} | {source} | {factor_mode} | {min_factor} | {max_factor} "
            "| {needle_rate} | {retrieval_rate} | {ppl_summary} | {latency} |".format(
                name=candidate["name"],
                status=candidate["status"],
                source=candidate["source"],
                factor_mode=factor_summary["factor_mode"],
                min_factor=factor_summary["min_factor"],
                max_factor=factor_summary["max_factor"],
                needle_rate=_format_rate(_extract_generation_rate(candidate, "needle_in_a_haystack")),
                retrieval_rate=_format_rate(_extract_generation_rate(candidate, "qa_retrieval")),
                ppl_summary=_extract_ppl_summary(candidate),
                latency=candidate.get("runtime", {}).get("latency_sec", "-"),
            )
        )
    return "\n".join(lines).strip() + "\n"


def _default_sweep_output_dir(checkpoint_root):
    artifact_root = checkpoint_root.parent.parent
    return artifact_root / "evaluations" / "longrope2_factor_sweep"


def save_longrope2_factor_sweep_report(report, checkpoint_root, *, output_dir=None, output_format="both"):
    """保存 LongRoPE2 因子 sweep JSON / Markdown 报告。"""
    target_dir = _default_sweep_output_dir(checkpoint_root) if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stage = report["checkpoint"]["training_stage"] or "unknown"
    file_stem = f"{timestamp}_{model_stage}_longrope2_factor_sweep"
    saved_paths = {}

    if output_format in {"json", "both"}:
        json_path = target_dir / f"{file_stem}.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths["json"] = str(json_path)

    if output_format in {"markdown", "md", "both"}:
        markdown_path = target_dir / f"{file_stem}.md"
        markdown_path.write_text(
            format_longrope2_factor_sweep_report_markdown(report),
            encoding="utf-8",
        )
        saved_paths["markdown"] = str(markdown_path)

    return saved_paths
