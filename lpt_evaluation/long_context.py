"""长上下文评测套件。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from lpt_config import (
    GenerationConfig,
    GlobalConfig,
    LoRAConfig,
    build_model_config_from_checkpoint,
)
from lpt_data import load_dataset_manifest
from lpt_inference import InferenceSession
from lpt_lora.adapter import attach_lora_adapters
from lpt_model import LPT
from lpt_protocol import render_prompt_from_messages
from lpt_training import load_checkpoint
from lpt_workflows.chat_lora import _resolve_chat_lora_checkpoint_root
from lpt_workflows.chat_sft import CHAT_SFT_CHECKPOINT_ROOT
from lpt_workflows.common import TOKENIZER_PATH, build_local_tokenizer
from lpt_workflows.text_pretrain import TEXT_PRETRAIN_CHECKPOINT_ROOT


DEFAULT_NEEDLE_LENGTHS = (2048, 4096)
DEFAULT_NEEDLE_DEPTHS = (0.2, 0.5, 0.8)
DEFAULT_PPL_LENGTHS = (1024, 2048)
DEFAULT_RETRIEVAL_LENGTHS = (2048, 4096)
DEFAULT_GENERATION_CONFIG = GenerationConfig(
    do_sample=False,
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    max_length=48,
    repetition_penalty=1.0,
    repetition_window_size=None,
)


@dataclass(frozen=True)
class LongContextEvaluationConfig:
    """长上下文评测配置。"""

    model_type: str = "chat_sft"
    checkpoint_root: str | None = None
    lora_base_source: str = "text_pretrain"
    cache_strategy: str = "session_rebuild"
    output_format: str = "both"
    output_dir: str | None = None
    text_manifest_path: str = "data/manifests/text_pretrain.json"
    needle_lengths: tuple[int, ...] = DEFAULT_NEEDLE_LENGTHS
    needle_depths: tuple[float, ...] = DEFAULT_NEEDLE_DEPTHS
    retrieval_lengths: tuple[int, ...] = DEFAULT_RETRIEVAL_LENGTHS
    ppl_lengths: tuple[int, ...] = DEFAULT_PPL_LENGTHS
    ppl_max_windows: int = 4
    seed: int = 42
    max_generation_tokens: int = 48


def _resolve_checkpoint_root(model_type, checkpoint_root=None, *, lora_base_source="text_pretrain"):
    if checkpoint_root is not None:
        return Path(checkpoint_root)
    if model_type == "text_pretrain":
        return TEXT_PRETRAIN_CHECKPOINT_ROOT
    if model_type == "chat_sft":
        return CHAT_SFT_CHECKPOINT_ROOT
    if model_type == "chat_lora":
        return _resolve_chat_lora_checkpoint_root(lora_base_source)
    raise ValueError(f"不支持的模型类型: {model_type}")


def _autocast_context():
    if GlobalConfig.device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(
        device_type=GlobalConfig.device.type,
        dtype=GlobalConfig.autocast_dtype,
    )


def _count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _normalize_short_answer(text):
    normalized = text.strip().strip("。；;，,：:")
    return "".join(normalized.split())


def _trim_generated_ids(sequence_ids, eos_token_id, pad_token_id=None):
    trimmed_ids = list(sequence_ids)
    if eos_token_id is not None and eos_token_id in trimmed_ids:
        eos_index = trimmed_ids.index(eos_token_id)
        trimmed_ids = trimmed_ids[:eos_index]
    if pad_token_id is not None and pad_token_id != eos_token_id:
        while trimmed_ids and trimmed_ids[-1] == pad_token_id:
            trimmed_ids.pop()
    return trimmed_ids


def summarize_longrope_factors(longrope_factors):
    """把 checkpoint 中的 longrope factors 摘要化，便于跨实验比较。"""
    if longrope_factors is None:
        return {
            "present": False,
            "factor_mode": "missing",
            "factor_count": 0,
            "unique_factor_count": 0,
            "min_factor": None,
            "max_factor": None,
            "uniform_factor": None,
        }

    if isinstance(longrope_factors, (int, float)):
        normalized_factors = [float(longrope_factors)]
    else:
        normalized_factors = [float(value) for value in longrope_factors]

    unique_factors = sorted(set(normalized_factors))
    factor_mode = "uniform" if len(unique_factors) == 1 else "per_dimension"
    return {
        "present": True,
        "factor_mode": factor_mode,
        "factor_count": len(normalized_factors),
        "unique_factor_count": len(unique_factors),
        "min_factor": min(normalized_factors),
        "max_factor": max(normalized_factors),
        "uniform_factor": unique_factors[0] if len(unique_factors) == 1 else None,
    }


def _build_checkpoint_summary(checkpoint, checkpoint_root):
    architecture_metadata = dict(checkpoint.get("model_architecture_metadata", {}))
    return {
        "checkpoint_path": str(checkpoint_root.with_suffix(".pth")),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "model_config_schema_version": checkpoint.get("model_config_schema_version"),
        "training_stage": checkpoint.get("training_stage"),
        "training_mode": checkpoint.get("training_mode"),
        "source_manifest": checkpoint.get("source_manifest"),
        "model_config": checkpoint.get("model_config"),
        "model_architecture_metadata": architecture_metadata,
        "longrope_factor_summary": summarize_longrope_factors(
            architecture_metadata.get("longrope2_long_factors")
        ),
    }


def load_model_for_long_context_evaluation(
    model_type,
    checkpoint_root=None,
    *,
    lora_base_source="text_pretrain",
):
    """按 checkpoint 加载模型和 tokenizer，用于长上下文评测。"""
    resolved_checkpoint_root = _resolve_checkpoint_root(
        model_type,
        checkpoint_root,
        lora_base_source=lora_base_source,
    )
    checkpoint_file = resolved_checkpoint_root.with_suffix(".pth")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_file}")

    checkpoint = load_checkpoint(resolved_checkpoint_root)
    tokenizer = build_local_tokenizer(TOKENIZER_PATH)
    model_config = build_model_config_from_checkpoint(checkpoint)
    model = LPT(vocabulary_size=len(tokenizer), config=model_config).to(GlobalConfig.device)
    if checkpoint.get("training_mode") == "lora" or model_type == "chat_lora":
        attach_lora_adapters(model, config=LoRAConfig())
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    checkpoint_summary = _build_checkpoint_summary(checkpoint, resolved_checkpoint_root)
    return model, tokenizer, checkpoint_summary, resolved_checkpoint_root


def _generate_filler_paragraph(rng, index):
    topics = (
        "古籍整理",
        "植物观察",
        "课堂讨论",
        "档案编目",
        "城市地理",
        "数学笔记",
        "物理实验",
        "历史摘录",
    )
    verbs = ("记录", "整理", "分析", "讨论", "归纳", "标注", "复核", "补充")
    topic = topics[index % len(topics)]
    verb = verbs[(index * 3) % len(verbs)]
    marker = rng.randint(100, 999)
    return (
        f"段落{index:04d}：研究人员围绕{topic}{verb}样本，"
        f"补充了编号为{marker}的说明，并强调需要保持上下文顺序稳定。"
    )


def _build_paragraphs_to_target(tokenizer, seed, target_tokens, fixed_overhead_text):
    rng = random.Random(seed)
    paragraphs = []
    current_text = fixed_overhead_text
    paragraph_index = 1
    while _count_tokens(tokenizer, current_text) < target_tokens:
        paragraph = _generate_filler_paragraph(rng, paragraph_index)
        paragraphs.append(paragraph)
        current_text = fixed_overhead_text + "\n".join(paragraphs)
        paragraph_index += 1
    return paragraphs


def build_needle_case(tokenizer, sequence_length, depth, *, seed):
    """构造 needle-in-a-haystack 样本。"""
    answer = f"NIAH-{seed:04d}-{int(depth * 100):02d}"
    needle_sentence = f"隐藏针：本段资料唯一需要记住的代号是 {answer}。"
    question = "问题：隐藏针中的代号是什么？请只输出代号。"
    fixed_overhead = f"{needle_sentence}\n{question}\n"
    paragraphs = _build_paragraphs_to_target(
        tokenizer,
        seed=seed,
        target_tokens=sequence_length,
        fixed_overhead_text=fixed_overhead,
    )
    insert_index = min(len(paragraphs), max(0, int(len(paragraphs) * depth)))
    context_parts = paragraphs[:insert_index] + [needle_sentence] + paragraphs[insert_index:]
    prompt_messages = [
        {
            "role": "user",
            "content": (
                "下面是一段很长的资料，请先通读，再回答最后的问题。\n\n"
                + "\n".join(context_parts)
                + f"\n\n{question}"
            ),
        }
    ]
    return {
        "expected_answer": answer,
        "prompt_text": render_prompt_from_messages(
            prompt_messages,
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
        ),
    }


def build_retrieval_case(tokenizer, sequence_length, *, seed):
    """构造长上下文 QA/检索样本。"""
    rng = random.Random(seed)
    target_id = rng.randint(8, 32)
    target_answer = f"KEY-{seed:04d}-{target_id:03d}"
    question = f"问题：档案 A{target_id:03d} 的密钥是什么？请只输出密钥。"

    records = []
    record_index = 1
    while True:
        archive_id = f"A{record_index:03d}"
        city = ("上海", "北京", "广州", "深圳", "成都", "武汉")[record_index % 6]
        color = ("红", "蓝", "绿", "黄", "紫", "灰")[record_index % 6]
        key = f"KEY-{seed:04d}-{record_index:03d}"
        if record_index == target_id:
            key = target_answer
        records.append(f"档案 {archive_id}：城市={city}；颜色={color}；密钥={key}。")
        prompt_text = render_prompt_from_messages(
            [
                {
                    "role": "user",
                    "content": (
                        "下面是一个很长的档案目录，请根据目录回答最后的问题。\n\n"
                        + "\n".join(records)
                        + f"\n\n{question}"
                    ),
                }
            ],
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
        )
        if _count_tokens(tokenizer, prompt_text) >= sequence_length:
            break
        record_index += 1

    return {
        "expected_answer": target_answer,
        "prompt_text": prompt_text,
    }


def build_text_ppl_windows(records, tokenizer, window_size, *, max_windows):
    """把 text manifest 样本压成定长 token 窗口。"""
    token_buffer = []
    windows = []
    eos_token_id = tokenizer.eos_token_id

    for record in records:
        if record.get("type") != "text":
            continue
        token_ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_buffer.extend(token_ids)
        if eos_token_id is not None:
            token_buffer.append(eos_token_id)

        while len(token_buffer) >= window_size and len(windows) < max_windows:
            windows.append(token_buffer[:window_size])
            token_buffer = token_buffer[window_size:]
        if len(windows) >= max_windows:
            break

    return windows


def _select_next_token(model, logits, generated_ids, attention_mask, config, pad_token_id, eos_token_id=None):
    next_token_logits = logits[:, -1, :].float()
    next_token_logits = model._apply_repetition_penalty_vectorized(
        next_token_logits,
        generated_ids,
        penalty=config.repetition_penalty,
        history_mask=attention_mask.bool(),
        repetition_window_size=config.repetition_window_size,
    )

    if pad_token_id is not None and pad_token_id != eos_token_id:
        next_token_logits[:, pad_token_id] = float("-inf")

    if config.do_sample:
        probs = model._temperature_and_top_p(next_token_logits, config)
        return torch.multinomial(probs, num_samples=1)
    return next_token_logits.argmax(dim=-1, keepdim=True)


def generate_with_cache_strategy(
    model,
    tokenizer,
    prompt_text,
    *,
    cache_strategy,
    generation_config,
):
    """用指定缓存策略执行单条生成。"""
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_tokens = encoded["input_ids"].to(GlobalConfig.device)
    attention_mask = encoded["attention_mask"].to(GlobalConfig.device)
    output_sequence = prompt_tokens.clone()
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        with _autocast_context():
            if cache_strategy == "session_rebuild":
                session = InferenceSession(model)
                logits = session.prefill(prompt_tokens, attention_mask=attention_mask)
                for _ in range(generation_config.max_length):
                    if session.attention_mask.size(1) >= GlobalConfig.inference_max_sequence_length:
                        break
                    next_token = _select_next_token(
                        model,
                        logits,
                        output_sequence,
                        session.attention_mask,
                        generation_config,
                        pad_token_id,
                        eos_token_id,
                    )
                    output_sequence = torch.cat([output_sequence, next_token], dim=1)
                    if eos_token_id is not None and next_token.item() == eos_token_id:
                        break
                    logits = session.append(next_token)
            elif cache_strategy == "full_recompute":
                full_attention_mask = attention_mask.clone()
                for _ in range(generation_config.max_length):
                    if full_attention_mask.size(1) >= GlobalConfig.inference_max_sequence_length:
                        break
                    logits, _ = model(
                        output_sequence,
                        attention_mask=full_attention_mask,
                        layer_states=None,
                    )
                    next_token = _select_next_token(
                        model,
                        logits,
                        output_sequence,
                        full_attention_mask,
                        generation_config,
                        pad_token_id,
                        eos_token_id,
                    )
                    output_sequence = torch.cat([output_sequence, next_token], dim=1)
                    full_attention_mask = torch.cat(
                        [
                            full_attention_mask,
                            torch.ones(
                                (1, 1),
                                device=full_attention_mask.device,
                                dtype=full_attention_mask.dtype,
                            ),
                        ],
                        dim=1,
                    )
                    if eos_token_id is not None and next_token.item() == eos_token_id:
                        break
            else:
                raise ValueError(f"不支持的 cache_strategy: {cache_strategy}")

    generated_ids = output_sequence[0, prompt_tokens.size(1):].tolist()
    trimmed_ids = _trim_generated_ids(
        generated_ids,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return tokenizer.decode(trimmed_ids, skip_special_tokens=False).strip(), prompt_tokens.size(1)


def evaluate_single_generation_case(
    model,
    tokenizer,
    *,
    task_name,
    prompt_text,
    expected_answer,
    cache_strategy,
    generation_config,
):
    """执行单条生成型评测样本。"""
    started_at = time.perf_counter()
    try:
        response_text, input_token_count = generate_with_cache_strategy(
            model,
            tokenizer,
            prompt_text,
            cache_strategy=cache_strategy,
            generation_config=generation_config,
        )
        normalized_prediction = _normalize_short_answer(response_text)
        normalized_expected = _normalize_short_answer(expected_answer)
        matched = normalized_prediction == normalized_expected or normalized_expected in normalized_prediction
        return {
            "task_name": task_name,
            "status": "ok",
            "input_token_count": input_token_count,
            "latency_sec": round(time.perf_counter() - started_at, 6),
            "expected_answer": expected_answer,
            "prediction": response_text,
            "normalized_expected": normalized_expected,
            "normalized_prediction": normalized_prediction,
            "exact_match": matched,
        }
    except Exception as error:
        return {
            "task_name": task_name,
            "status": "error",
            "input_token_count": _count_tokens(tokenizer, prompt_text),
            "latency_sec": round(time.perf_counter() - started_at, 6),
            "expected_answer": expected_answer,
            "prediction": None,
            "exact_match": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def evaluate_needle_in_a_haystack(
    model,
    tokenizer,
    *,
    cache_strategy,
    generation_config,
    lengths,
    depths,
    seed,
):
    """执行 needle-in-a-haystack 评测。"""
    samples = []
    case_seed = seed
    for sequence_length in lengths:
        for depth in depths:
            case = build_needle_case(
                tokenizer,
                sequence_length,
                depth,
                seed=case_seed,
            )
            result = evaluate_single_generation_case(
                model,
                tokenizer,
                task_name="needle_in_a_haystack",
                prompt_text=case["prompt_text"],
                expected_answer=case["expected_answer"],
                cache_strategy=cache_strategy,
                generation_config=generation_config,
            )
            result["target_sequence_length"] = sequence_length
            result["needle_depth"] = depth
            samples.append(result)
            case_seed += 1
    return {
        "task_name": "needle_in_a_haystack",
        "aggregate": _aggregate_generation_results(samples),
        "samples": samples,
    }


def evaluate_retrieval_qa(
    model,
    tokenizer,
    *,
    cache_strategy,
    generation_config,
    lengths,
    seed,
):
    """执行长上下文 QA/检索评测。"""
    samples = []
    case_seed = seed
    for sequence_length in lengths:
        case = build_retrieval_case(
            tokenizer,
            sequence_length,
            seed=case_seed,
        )
        result = evaluate_single_generation_case(
            model,
            tokenizer,
            task_name="qa_retrieval",
            prompt_text=case["prompt_text"],
            expected_answer=case["expected_answer"],
            cache_strategy=cache_strategy,
            generation_config=generation_config,
        )
        result["target_sequence_length"] = sequence_length
        samples.append(result)
        case_seed += 1
    return {
        "task_name": "qa_retrieval",
        "aggregate": _aggregate_generation_results(samples),
        "samples": samples,
    }


def evaluate_long_text_perplexity(
    model,
    tokenizer,
    *,
    manifest_path,
    lengths,
    max_windows,
):
    """执行长文本 PPL 评测。"""
    records, loaded_datasets = load_dataset_manifest(Path(manifest_path), expected_types={"text"})
    length_results = []
    for window_size in lengths:
        windows = build_text_ppl_windows(
            records,
            tokenizer,
            window_size,
            max_windows=max_windows,
        )
        total_nll = 0.0
        total_tokens = 0
        sample_results = []
        for window_index, window_ids in enumerate(windows, start=1):
            started_at = time.perf_counter()
            try:
                input_ids = torch.tensor([window_ids], device=GlobalConfig.device, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=GlobalConfig.device)
                with torch.no_grad():
                    with _autocast_context():
                        logits, _ = model(
                            input_ids,
                            attention_mask=attention_mask,
                            layer_states=None,
                        )
                        nll = F.cross_entropy(
                            logits[:, :-1, :].transpose(1, 2),
                            input_ids[:, 1:],
                            reduction="sum",
                        )
                token_count = max(0, input_ids.size(1) - 1)
                nll_value = float(nll.item())
                total_nll += nll_value
                total_tokens += token_count
                sample_results.append(
                    {
                        "status": "ok",
                        "window_index": window_index,
                        "window_size": window_size,
                        "token_count": token_count,
                        "negative_log_likelihood": nll_value,
                        "latency_sec": round(time.perf_counter() - started_at, 6),
                    }
                )
            except Exception as error:
                sample_results.append(
                    {
                        "status": "error",
                        "window_index": window_index,
                        "window_size": window_size,
                        "token_count": len(window_ids),
                        "latency_sec": round(time.perf_counter() - started_at, 6),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )

        average_nll = None if total_tokens == 0 else total_nll / total_tokens
        perplexity = None
        perplexity_overflow = False
        if average_nll is not None:
            perplexity_overflow = average_nll > 80
            perplexity = None if perplexity_overflow else math.exp(average_nll)
        length_results.append(
            {
                "window_size": window_size,
                "window_count": len(windows),
                "effective_token_count": total_tokens,
                "average_negative_log_likelihood": average_nll,
                "perplexity": perplexity,
                "perplexity_overflow": perplexity_overflow,
                "samples": sample_results,
            }
        )

    return {
        "task_name": "long_text_ppl",
        "manifest_path": str(Path(manifest_path)),
        "loaded_datasets": loaded_datasets,
        "aggregate": _aggregate_ppl_results(length_results),
        "length_results": length_results,
    }


def _aggregate_generation_results(samples):
    successful_samples = [sample for sample in samples if sample["status"] == "ok"]
    matched_samples = [sample for sample in successful_samples if sample["exact_match"]]
    average_latency = None
    if successful_samples:
        average_latency = sum(sample["latency_sec"] for sample in successful_samples) / len(successful_samples)
    return {
        "sample_count": len(samples),
        "success_count": len(successful_samples),
        "error_count": len(samples) - len(successful_samples),
        "exact_match_count": len(matched_samples),
        "exact_match_rate": (len(matched_samples) / len(samples)) if samples else None,
        "average_latency_sec": average_latency,
    }


def _aggregate_ppl_results(length_results):
    summary = []
    for item in length_results:
        summary.append(
            {
                "window_size": item["window_size"],
                "window_count": item["window_count"],
                "effective_token_count": item["effective_token_count"],
                "perplexity": item["perplexity"],
            }
        )
    return summary


def evaluate_long_context_suite(config: LongContextEvaluationConfig):
    """执行完整长上下文评测套件。"""
    model, tokenizer, checkpoint_summary, checkpoint_root = load_model_for_long_context_evaluation(
        config.model_type,
        config.checkpoint_root,
        lora_base_source=config.lora_base_source,
    )
    generation_config = GenerationConfig(
        do_sample=DEFAULT_GENERATION_CONFIG.do_sample,
        temperature=DEFAULT_GENERATION_CONFIG.temperature,
        top_k=DEFAULT_GENERATION_CONFIG.top_k,
        top_p=DEFAULT_GENERATION_CONFIG.top_p,
        max_length=config.max_generation_tokens,
        repetition_penalty=DEFAULT_GENERATION_CONFIG.repetition_penalty,
        repetition_window_size=DEFAULT_GENERATION_CONFIG.repetition_window_size,
    )
    started_at = time.perf_counter()
    report = {
        "report_type": "long_context_evaluation",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": checkpoint_summary,
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
        },
        "tasks": {},
    }

    report["tasks"]["needle_in_a_haystack"] = evaluate_needle_in_a_haystack(
        model,
        tokenizer,
        cache_strategy=config.cache_strategy,
        generation_config=generation_config,
        lengths=config.needle_lengths,
        depths=config.needle_depths,
        seed=config.seed,
    )
    report["tasks"]["long_text_ppl"] = evaluate_long_text_perplexity(
        model,
        tokenizer,
        manifest_path=config.text_manifest_path,
        lengths=config.ppl_lengths,
        max_windows=config.ppl_max_windows,
    )
    report["tasks"]["qa_retrieval"] = evaluate_retrieval_qa(
        model,
        tokenizer,
        cache_strategy=config.cache_strategy,
        generation_config=generation_config,
        lengths=config.retrieval_lengths,
        seed=config.seed + 1000,
    )
    report["runtime"]["total_latency_sec"] = round(time.perf_counter() - started_at, 6)
    return report, checkpoint_root


def _format_generation_task_markdown(title, task_report):
    lines = [f"## {title}"]
    aggregate = task_report["aggregate"]
    lines.append(f"- 样本数: {aggregate['sample_count']}")
    lines.append(f"- 成功数: {aggregate['success_count']}")
    lines.append(f"- 错误数: {aggregate['error_count']}")
    lines.append(f"- 精确匹配率: {aggregate['exact_match_rate']}")
    lines.append(f"- 平均延迟(秒): {aggregate['average_latency_sec']}")
    lines.append("")
    lines.append("| target_length | extra | status | exact_match | latency_sec | prediction |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for sample in task_report["samples"]:
        extra = sample.get("needle_depth", "-")
        prediction = sample.get("prediction") or sample.get("error_type", "-")
        lines.append(
            f"| {sample.get('target_sequence_length', '-')} | {extra} | {sample['status']} "
            f"| {sample.get('exact_match')} | {sample['latency_sec']} | {prediction} |"
        )
    lines.append("")
    return lines


def format_long_context_report_markdown(report):
    """把评测结果格式化为 Markdown。"""
    checkpoint = report["checkpoint"]
    factor_summary = checkpoint["longrope_factor_summary"]
    lines = [
        "# 长上下文评测报告",
        "",
        "## Checkpoint",
        f"- checkpoint_path: {checkpoint['checkpoint_path']}",
        f"- checkpoint_schema_version: {checkpoint['checkpoint_schema_version']}",
        f"- model_config_schema_version: {checkpoint['model_config_schema_version']}",
        f"- training_stage: {checkpoint['training_stage']}",
        f"- training_mode: {checkpoint['training_mode']}",
        f"- source_manifest: {checkpoint['source_manifest']}",
        "",
        "## LongRoPE Factors",
        f"- factor_mode: {factor_summary['factor_mode']}",
        f"- factor_count: {factor_summary['factor_count']}",
        f"- unique_factor_count: {factor_summary['unique_factor_count']}",
        f"- min_factor: {factor_summary['min_factor']}",
        f"- max_factor: {factor_summary['max_factor']}",
        "",
        "## Runtime",
        f"- device: {report['runtime']['device']}",
        f"- cache_strategy: {report['runtime']['cache_strategy']}",
        f"- total_latency_sec: {report['runtime']['total_latency_sec']}",
        "",
    ]
    lines.extend(
        _format_generation_task_markdown(
            "Needle In A Haystack",
            report["tasks"]["needle_in_a_haystack"],
        )
    )
    lines.append("## Long Text PPL")
    lines.append("| window_size | window_count | effective_token_count | perplexity |")
    lines.append("| --- | --- | --- | --- |")
    for item in report["tasks"]["long_text_ppl"]["aggregate"]:
        lines.append(
            f"| {item['window_size']} | {item['window_count']} | "
            f"{item['effective_token_count']} | {item['perplexity']} |"
        )
    lines.append("")
    lines.extend(
        _format_generation_task_markdown(
            "QA Retrieval",
            report["tasks"]["qa_retrieval"],
        )
    )
    return "\n".join(lines).strip() + "\n"


def _default_report_output_dir(checkpoint_root):
    artifact_root = checkpoint_root.parent.parent
    return artifact_root / "evaluations" / "long_context"


def save_long_context_report(report, checkpoint_root, *, output_dir=None, output_format="both"):
    """把评测结果保存为 JSON / Markdown。"""
    target_dir = _default_report_output_dir(checkpoint_root) if output_dir is None else Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_strategy = report["runtime"]["cache_strategy"]
    model_stage = report["checkpoint"]["training_stage"] or "unknown"
    file_stem = f"{timestamp}_{model_stage}_{cache_strategy}"
    saved_paths = {}

    if output_format in {"json", "both"}:
        json_path = target_dir / f"{file_stem}.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths["json"] = str(json_path)

    if output_format in {"markdown", "md", "both"}:
        markdown_path = target_dir / f"{file_stem}.md"
        markdown_path.write_text(format_long_context_report_markdown(report), encoding="utf-8")
        saved_paths["markdown"] = str(markdown_path)

    return saved_paths
