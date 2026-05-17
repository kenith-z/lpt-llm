"""LPT v2 长上下文准入评测。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from lpt_config import LongContextEvalConfig, count_retnet_assist_enabled_layers
from lpt_config.profiles import LPT_V2_ASSIST_PROFILE, LPT_V2_PAGED_KV_PROFILE, build_lpt_v2_profile_config
from lpt_model import LPTV2, load_lpt_v2_checkpoint

from .utils import (
    build_deterministic_input,
    dtype_name,
    next_token_loss,
    resolve_eval_device,
    resolve_eval_dtype,
    set_eval_seed,
)


DEFAULT_LONG_CONTEXT_EVAL_CONFIG = LongContextEvalConfig()


def _target_rank(logits, target_token_id):
    """计算目标 token 在最后一步 logits 中的排序名次。"""
    last_logits = logits[0, -1].float()
    sorted_indices = torch.argsort(last_logits, descending=True)
    rank = (sorted_indices == int(target_token_id)).nonzero(as_tuple=False)
    if rank.numel() == 0:
        return None
    return int(rank[0, 0].item()) + 1


def _target_logprob(logits, target_token_id):
    """计算目标 token 在最后一步的 log probability。"""
    log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)
    return float(log_probs[int(target_token_id)].detach().cpu())


def _mean_optional(values):
    """跳过 None 后求均值，没有有效值时返回 0。"""
    values = [float(value) for value in values if value is not None]
    return 0.0 if not values else sum(values) / len(values)


def _collect_retnet_mechanism(states):
    """从所有启用层汇总 RetNetAssist 机制指标，避免稀疏启用层被第 0 层误判。"""
    retnet_states = [
        (layer_index, layer_state.retnet_assist)
        for layer_index, layer_state in enumerate(states)
        if layer_state.retnet_assist is not None
    ]
    if not retnet_states:
        return {
            "first_layer_index": None,
            "token_count": 0,
            "q_adapter_delta_norm": 0.0,
            "k_adapter_delta_norm": 0.0,
            "context_adapter_delta_norm": 0.0,
            "alpha_context": 0.0,
        }
    return {
        "first_layer_index": int(retnet_states[0][0]),
        "token_count": max(int(state.token_count) for _, state in retnet_states),
        "q_adapter_delta_norm": _mean_optional(state.q_adapter_delta_norm for _, state in retnet_states),
        "k_adapter_delta_norm": _mean_optional(state.k_adapter_delta_norm for _, state in retnet_states),
        "context_adapter_delta_norm": _mean_optional(
            state.context_adapter_delta_norm for _, state in retnet_states
        ),
        "alpha_context": _mean_optional(state.alpha_context for _, state in retnet_states),
    }


def _resolve_needle_index(sequence_length, needle_depth):
    """根据 depth 把 needle 放到序列内部，首尾各保留一个控制位置。"""
    depth = min(max(float(needle_depth), 0.0), 1.0)
    last_allowed_index = max(1, int(sequence_length) - 2)
    if last_allowed_index <= 1:
        return 1
    return 1 + int(round((last_allowed_index - 1) * depth))


def _release_model_request_state(model, request_id):
    """释放一次评测写入的 request-bound 状态，避免 suite 循环时累积显存。"""
    if hasattr(model, "reset_request_state"):
        model.reset_request_state(request_id=request_id)
    if hasattr(model, "release_retnet_assist_state"):
        model.release_retnet_assist_state(request_id=request_id, reason="eval_finished")
    if hasattr(model, "release_xlstm_memory_state"):
        model.release_xlstm_memory_state(request_id=request_id, reason="eval_finished")


def _build_probe_inputs(*, vocabulary_size, sequence_length, attention_window_size, needle_depth, device):
    """构造长上下文、代码数学和格式代理输入。"""
    needle_token_id = max(1, int(vocabulary_size) - 3)
    needle_index = _resolve_needle_index(sequence_length, needle_depth)
    input_ids = build_deterministic_input(
        vocabulary_size,
        1,
        sequence_length,
        offset=7,
        device=device,
    )
    input_ids[0, needle_index] = needle_token_id
    input_ids[0, -1] = 2
    # code/math 和 format 代理输入较短，主要用于确认局部窗口内普通 next-token 路径仍可运行。
    code_math_ids = build_deterministic_input(
        vocabulary_size,
        1,
        min(sequence_length, attention_window_size + 4),
        offset=17,
        device=device,
    )
    format_ids = build_deterministic_input(
        vocabulary_size,
        1,
        min(sequence_length, attention_window_size + 2),
        offset=31,
        device=device,
    )
    return input_ids, code_math_ids, format_ids, needle_token_id, needle_index


def _run_model_probe(model, *, input_ids, code_math_ids, format_ids, needle_token_id, request_id):
    """对单个模型执行一次长上下文代理评测。"""
    _release_model_request_state(model, request_id)
    try:
        with torch.no_grad():
            logits, states = model.prefill(input_ids, request_id=request_id)
            loss, ppl = next_token_loss(logits, input_ids)
            # 代理项不写入 KV cache，避免它们干扰主长上下文 request 的状态观察。
            code_logits, _ = model(
                code_math_ids,
                request_id=f"{request_id}-code",
                use_kv_cache=False,
            )
            format_logits, _ = model(
                format_ids,
                request_id=f"{request_id}-format",
                use_kv_cache=False,
            )
            code_loss, _ = next_token_loss(code_logits, code_math_ids)
            format_loss, _ = next_token_loss(format_logits, format_ids)
    finally:
        _release_model_request_state(model, request_id)
    return {
        "logits": logits,
        "states": states,
        "loss": loss,
        "ppl": ppl,
        "rank": _target_rank(logits, needle_token_id),
        "logprob": _target_logprob(logits, needle_token_id),
        "code_loss": code_loss,
        "format_loss": format_loss,
    }


@dataclass(frozen=True)
class LongContextAdmissionReport:
    """长上下文准入报告。"""

    preset: str
    device: str
    dtype: str
    vocabulary_size: int
    sequence_length: int
    attention_window_size: int
    needle_depth: float
    metrics: dict
    checkpoint_path: str | None = None
    checkpoint_metadata: dict | None = None

    def to_dict(self):
        """生成长上下文准入 JSON 载荷。"""
        payload = {
            "report_type": "lpt_v2_long_context_admission",
            "preset": self.preset,
            "device": self.device,
            "dtype": self.dtype,
            "vocabulary_size": self.vocabulary_size,
            "sequence_length": self.sequence_length,
            "attention_window_size": self.attention_window_size,
            "needle_depth": self.needle_depth,
            "metrics": dict(self.metrics),
        }
        if self.checkpoint_path is not None:
            payload["checkpoint_path"] = self.checkpoint_path
            payload["checkpoint_metadata"] = dict(self.checkpoint_metadata or {})
        return payload

    def to_markdown(self):
        """生成长上下文准入 Markdown 报告。"""
        needle = self.metrics["needle"]
        long_text = self.metrics["long_text_ppl"]
        decision = self.metrics["quality_decision"]
        checkpoint_lines = []
        if self.checkpoint_path is not None:
            checkpoint_lines = [
                f"- checkpoint: `{self.checkpoint_path}`",
                f"- training_stage: `{(self.checkpoint_metadata or {}).get('training_stage')}`",
                f"- global_step: `{(self.checkpoint_metadata or {}).get('global_step')}`",
            ]
        no_assist_rank = _markdown_value(needle["no_assist_rank"])
        no_assist_logprob = _markdown_float(needle["no_assist_logprob"])
        rank_delta = _markdown_value(needle["rank_delta"])
        logprob_delta = _markdown_float(needle["logprob_delta"])
        no_assist_ppl = _markdown_float(long_text["no_assist_ppl"])
        relative_delta = _markdown_float(long_text["relative_delta"])
        lines = [
            "# LPT v2 Long Context Admission",
            "",
            f"- preset: `{self.preset}`",
            f"- device: `{self.device}`",
            f"- dtype: `{self.dtype}`",
            f"- sequence_length: `{self.sequence_length}`",
            f"- attention_window_size: `{self.attention_window_size}`",
            f"- needle_depth: `{self.needle_depth}`",
            *checkpoint_lines,
            "",
            "| metric | assist | no_assist | delta |",
            "|---|---:|---:|---:|",
            f"| needle_rank | {needle['assist_rank']} | {no_assist_rank} | {rank_delta} |",
            f"| needle_logprob | {needle['assist_logprob']:.4f} | {no_assist_logprob} | {logprob_delta} |",
            f"| long_text_ppl | {long_text['assist_ppl']:.2f} | {no_assist_ppl} | {relative_delta} |",
            "",
            f"decision: `{decision['status']}`",
            "",
            decision["reason"],
        ]
        return "\n".join(lines) + "\n"


def _markdown_value(value):
    """把可选值渲染为 Markdown 单元格。"""
    return "n/a" if value is None else str(value)


def _markdown_float(value, *, digits=4):
    """把可选浮点值渲染为 Markdown 单元格。"""
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _checkpoint_training_metadata(checkpoint):
    """提取 checkpoint 中与实验可追溯性相关的训练元数据。"""
    extra = checkpoint.get("runtime_metadata", {}).get("extra", {})
    tokenizer_metadata = extra.get("tokenizer_metadata") or {}
    return {
        "checkpoint_format": checkpoint.get("checkpoint_format"),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "architecture_version": checkpoint.get("architecture_version"),
        "model_config_schema_version": checkpoint.get("model_config_schema_version"),
        "training_stage": extra.get("training_stage"),
        "run_id": extra.get("run_id"),
        "global_step": extra.get("global_step"),
        "optimizer_step": extra.get("optimizer_step"),
        "tokens_seen": extra.get("tokens_seen"),
        "source_manifest": extra.get("source_manifest"),
        "eval_manifest": extra.get("eval_manifest"),
        "tokenizer_metadata": {
            "tokenizer_path": tokenizer_metadata.get("tokenizer_path"),
            "tokenizer_config_sha256": tokenizer_metadata.get("tokenizer_config_sha256"),
            "vocab_size": tokenizer_metadata.get("vocab_size"),
            "chat_template_version": tokenizer_metadata.get("chat_template_version"),
        },
    }


def _run_checkpoint_admission(
    *,
    checkpoint_path,
    sequence_length,
    attention_window_size,
    device,
    dtype,
    needle_depth,
):
    """加载真实 checkpoint 后运行长上下文准入。"""
    loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=True)
    return run_lpt_v2_long_context_admission_for_model(
        model=loaded.model,
        preset=loaded.checkpoint["model_config"].get("model_size_preset", "checkpoint"),
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=_checkpoint_training_metadata(loaded.checkpoint),
        sequence_length=sequence_length,
        attention_window_size=attention_window_size,
        device=device,
        dtype=dtype,
        needle_depth=needle_depth,
    )


def run_lpt_v2_long_context_admission_for_model(
    *,
    model,
    preset="checkpoint",
    sequence_length=None,
    attention_window_size=None,
    device=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.device,
    dtype=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.dtype,
    needle_depth=0.0,
    checkpoint_path=None,
    checkpoint_metadata=None,
):
    """对已构造好的 LPTV2 模型运行 checkpoint 口径长上下文准入。"""
    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    model.to(device=target_device, dtype=target_dtype).eval()
    config = model.config
    vocabulary_size = int(model.token_embedding.num_embeddings)
    attention_window_size = int(attention_window_size or config.attention_window_size)
    sequence_length = int(sequence_length or attention_window_size * 2 + 4)
    if sequence_length <= attention_window_size:
        raise ValueError("sequence_length 必须大于 attention_window_size，才能验证窗口外信息。")

    from lpt_config import GlobalConfig

    GlobalConfig.inference_rope_cache_max_sequence_length = max(
        int(GlobalConfig.inference_rope_cache_max_sequence_length),
        int(sequence_length),
    )
    # 长上下文评测可能临时超过默认推理 RoPE cache，上调全局上限但不写回 checkpoint。
    input_ids, code_math_ids, format_ids, needle_token_id, needle_index = _build_probe_inputs(
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        attention_window_size=attention_window_size,
        needle_depth=needle_depth,
        device=target_device,
    )
    result = _run_model_probe(
        model,
        input_ids=input_ids,
        code_math_ids=code_math_ids,
        format_ids=format_ids,
        needle_token_id=needle_token_id,
        request_id="long-context-checkpoint",
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    states = result["states"]
    retnet_mechanism = _collect_retnet_mechanism(states)
    retnet_tokens = int(retnet_mechanism["token_count"])
    q_adapter_delta_norm = float(retnet_mechanism["q_adapter_delta_norm"])
    k_adapter_delta_norm = float(retnet_mechanism["k_adapter_delta_norm"])
    context_adapter_delta_norm = float(retnet_mechanism["context_adapter_delta_norm"])
    paged_window = int(states[0].attention.paged_kv_ref.window_token_count)
    # 机制准入只要求 RetNet 摘要 token_count 跨过局部窗口；质量收益要看真实评测集。
    mechanism_ready = bool(retnet_tokens > paged_window)
    status = "admit_checkpoint_path" if mechanism_ready else "close_or_debug"
    reason = (
        "已加载真实 v2 checkpoint 完成长上下文前向、PPL 与状态池准入；质量结论需结合独立验证集。"
        if mechanism_ready
        else "checkpoint 可加载但长上下文状态未跨越局部窗口，应检查配置或输入长度。"
    )
    metrics = {
        "needle": {
            "target_token_id": int(needle_token_id),
            "needle_index": int(needle_index),
            "needle_depth": float(needle_depth),
            "assist_rank": result["rank"],
            "no_assist_rank": None,
            "rank_delta": None,
            "assist_logprob": result["logprob"],
            "no_assist_logprob": None,
            "logprob_delta": None,
        },
        "long_text_ppl": {
            "assist_loss": result["loss"],
            "no_assist_loss": None,
            "assist_ppl": result["ppl"],
            "no_assist_ppl": None,
            "relative_delta": None,
        },
        "qa_retrieval": {
            "proxy": "needle_rank",
            "assist_reciprocal_rank": 1.0 / max(1, result["rank"] or 1),
            "no_assist_reciprocal_rank": None,
        },
        "code_math": {
            "proxy": "deterministic_pattern_next_token_loss",
            "assist_loss": result["code_loss"],
            "no_assist_loss": None,
        },
        "format_following": {
            "proxy": "structured_pattern_next_token_loss",
            "assist_loss": result["format_loss"],
            "no_assist_loss": None,
        },
        "mechanism": {
            "assist_retnet_token_count": retnet_tokens,
            "paged_kv_window_token_count": paged_window,
            "q_adapter_delta_norm": q_adapter_delta_norm,
            "k_adapter_delta_norm": k_adapter_delta_norm,
            "context_adapter_delta_norm": context_adapter_delta_norm,
            "alpha_context": float(retnet_mechanism["alpha_context"]),
            "mechanism_ready": mechanism_ready,
            "retnet_assist_layers": config.retnet_assist_layers,
            "retnet_assist_selected_layers": list(config.retnet_assist_selected_layers),
            "retnet_enabled_layer_count": count_retnet_assist_enabled_layers(config),
            "retnet_first_enabled_layer": retnet_mechanism["first_layer_index"],
            "retnet_assist_mode": config.retnet_assist_mode,
            "retnet_adapter_rank": int(config.retnet_adapter_rank),
            "retnet_parameter_sharing": config.retnet_parameter_sharing,
            "retnet_state_sharing": config.retnet_state_sharing,
            "retnet_sharing_group_size": int(config.retnet_sharing_group_size),
            "retnet_adapter_target": list(config.retnet_adapter_target),
            "retnet_k_adapter_enabled": bool(config.retnet_k_adapter_enabled),
            "retnet_context_adapter_enabled": bool(config.retnet_context_adapter_enabled),
        },
        "quality_decision": {
            "status": status,
            "reason": reason,
        },
    }
    return LongContextAdmissionReport(
        preset=str(preset),
        device=str(target_device),
        dtype=dtype_name(target_dtype),
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        attention_window_size=attention_window_size,
        needle_depth=float(needle_depth),
        metrics=metrics,
        checkpoint_path=None if checkpoint_path is None else str(Path(checkpoint_path)),
        checkpoint_metadata=None if checkpoint_metadata is None else dict(checkpoint_metadata),
    )


def run_lpt_v2_long_context_admission(
    *,
    preset=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.preset,
    vocabulary_size=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.vocabulary_size,
    sequence_length=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.sequence_length,
    attention_window_size=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.attention_window_size,
    device=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.device,
    dtype=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.dtype,
    seed=DEFAULT_LONG_CONTEXT_EVAL_CONFIG.seed,
    checkpoint_path=None,
    needle_depth=0.0,
):
    """运行长上下文准入 smoke 评测。"""
    if checkpoint_path is not None:
        return _run_checkpoint_admission(
            checkpoint_path=checkpoint_path,
            sequence_length=sequence_length,
            attention_window_size=attention_window_size,
            device=device,
            dtype=dtype,
            needle_depth=needle_depth,
        )

    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    sequence_length = int(sequence_length or attention_window_size * 2 + 4)
    if sequence_length <= attention_window_size:
        raise ValueError("sequence_length 必须大于 attention_window_size，才能验证窗口外信息。")

    from lpt_config import GlobalConfig

    GlobalConfig.inference_rope_cache_max_sequence_length = max(
        int(GlobalConfig.inference_rope_cache_max_sequence_length),
        int(sequence_length),
    )

    common_overrides = {
        "attention_window_size": int(attention_window_size),
        "page_block_size": max(1, int(attention_window_size) // 2),
        "original_max_len": max(4, int(attention_window_size)),
        "longrope2_target_length": max(sequence_length, int(attention_window_size) * 2),
        "retnet_assist_layers": "all_layers",
    }
    assist_config = build_lpt_v2_profile_config(
        LPT_V2_ASSIST_PROFILE,
        preset=preset,
        **common_overrides,
    )
    # 无 checkpoint 路径只做 assist vs no_assist 机制对照，随机初始化不能证明质量收益。
    no_assist_config = build_lpt_v2_profile_config(
        LPT_V2_PAGED_KV_PROFILE,
        preset=preset,
        **common_overrides,
    )

    set_eval_seed(seed)
    assist_model = LPTV2(vocabulary_size, assist_config).to(device=target_device, dtype=target_dtype).eval()
    set_eval_seed(seed)
    no_assist_model = LPTV2(vocabulary_size, no_assist_config).to(device=target_device, dtype=target_dtype).eval()

    input_ids, code_math_ids, format_ids, needle_token_id, needle_index = _build_probe_inputs(
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        attention_window_size=attention_window_size,
        needle_depth=needle_depth,
        device=target_device,
    )
    assist_result = _run_model_probe(
        assist_model,
        input_ids=input_ids,
        code_math_ids=code_math_ids,
        format_ids=format_ids,
        needle_token_id=needle_token_id,
        request_id="long-context-assist",
    )
    no_assist_result = _run_model_probe(
        no_assist_model,
        input_ids=input_ids,
        code_math_ids=code_math_ids,
        format_ids=format_ids,
        needle_token_id=needle_token_id,
        request_id="long-context-no-assist",
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    assist_logits = assist_result["logits"]
    no_assist_logits = no_assist_result["logits"]
    assist_states = assist_result["states"]
    logit_delta_l2 = float((assist_logits.float() - no_assist_logits.float()).pow(2).mean().sqrt().detach().cpu())
    retnet_mechanism = _collect_retnet_mechanism(assist_states)
    retnet_tokens = int(retnet_mechanism["token_count"])
    q_adapter_delta_norm = float(retnet_mechanism["q_adapter_delta_norm"])
    k_adapter_delta_norm = float(retnet_mechanism["k_adapter_delta_norm"])
    context_adapter_delta_norm = float(retnet_mechanism["context_adapter_delta_norm"])
    adapter_delta_l2 = 0.0
    alpha_q = 0.0
    first_retnet_layer = retnet_mechanism["first_layer_index"]
    if first_retnet_layer is not None:
        first_state = assist_states[int(first_retnet_layer)].retnet_assist
        q_adapter = assist_model.layers[int(first_retnet_layer)].attention_mixer.q_adapter
        if first_state is not None and first_state.summary is not None and q_adapter is not None:
            summary = first_state.summary[:, None].to(device=target_device, dtype=target_dtype)
            alpha_q = float(q_adapter.alpha_q.detach().cpu())
            dummy_query = torch.zeros(
                1,
                assist_config.num_heads,
                1,
                assist_config.head_dim,
                device=target_device,
                dtype=target_dtype,
            )
            adapted_query = q_adapter(summary, dummy_query)
            adapter_delta_l2 = float(
                (adapted_query.float() - dummy_query.float()).pow(2).mean().sqrt().detach().cpu()
            )
    paged_window = int(assist_states[0].attention.paged_kv_ref.window_token_count)
    mechanism_ready = bool(
        retnet_tokens > paged_window
        and (
            logit_delta_l2 > 0.0
            or adapter_delta_l2 > 0.0
            or q_adapter_delta_norm > 0.0
            or k_adapter_delta_norm > 0.0
            or context_adapter_delta_norm > 0.0
        )
    )

    relative_ppl_delta = float((assist_result["ppl"] - no_assist_result["ppl"]) / max(no_assist_result["ppl"], 1e-9))
    if relative_ppl_delta < -0.01:
        status = "admit_quality_benefit"
        reason = "当前输入上 assist PPL 明显低于 no_assist，可进入更大评测。"
    elif mechanism_ready:
        status = "admit_instrumentation_only"
        reason = "RetNetAssist 已跨越局部窗口保留状态且 Q adapter 形成可观测调制；当前随机权重不能证明质量收益，需要训练 checkpoint 后再做准入。"
    else:
        status = "close_or_debug"
        reason = "RetNetAssist 未形成可观测机制差异，应先检查配置、状态池或 adapter。"

    metrics = {
        "needle": {
            "target_token_id": int(needle_token_id),
            "needle_index": int(needle_index),
            "needle_depth": float(needle_depth),
            "assist_rank": assist_result["rank"],
            "no_assist_rank": no_assist_result["rank"],
            "rank_delta": None,
            "assist_logprob": assist_result["logprob"],
            "no_assist_logprob": no_assist_result["logprob"],
            "logprob_delta": None,
        },
        "long_text_ppl": {
            "assist_loss": assist_result["loss"],
            "no_assist_loss": no_assist_result["loss"],
            "assist_ppl": assist_result["ppl"],
            "no_assist_ppl": no_assist_result["ppl"],
            "relative_delta": relative_ppl_delta,
        },
        "qa_retrieval": {
            "proxy": "needle_rank",
            "assist_reciprocal_rank": 1.0 / max(1, assist_result["rank"]),
            "no_assist_reciprocal_rank": 1.0 / max(1, no_assist_result["rank"]),
        },
        "code_math": {
            "proxy": "deterministic_pattern_next_token_loss",
            "assist_loss": assist_result["code_loss"],
            "no_assist_loss": no_assist_result["code_loss"],
        },
        "format_following": {
            "proxy": "structured_pattern_next_token_loss",
            "assist_loss": assist_result["format_loss"],
            "no_assist_loss": no_assist_result["format_loss"],
        },
        "mechanism": {
            "assist_retnet_token_count": retnet_tokens,
            "paged_kv_window_token_count": paged_window,
            "logit_delta_l2": logit_delta_l2,
            "adapter_delta_l2": adapter_delta_l2,
            "q_adapter_delta_norm": q_adapter_delta_norm,
            "k_adapter_delta_norm": k_adapter_delta_norm,
            "context_adapter_delta_norm": context_adapter_delta_norm,
            "alpha_q": alpha_q,
            "alpha_context": float(retnet_mechanism["alpha_context"]),
            "retnet_assist_layers": assist_config.retnet_assist_layers,
            "retnet_assist_selected_layers": list(assist_config.retnet_assist_selected_layers),
            "retnet_enabled_layer_count": count_retnet_assist_enabled_layers(assist_config),
            "retnet_first_enabled_layer": retnet_mechanism["first_layer_index"],
            "retnet_assist_mode": assist_config.retnet_assist_mode,
            "retnet_adapter_rank": int(assist_config.retnet_adapter_rank),
            "retnet_parameter_sharing": assist_config.retnet_parameter_sharing,
            "retnet_state_sharing": assist_config.retnet_state_sharing,
            "retnet_sharing_group_size": int(assist_config.retnet_sharing_group_size),
            "retnet_adapter_target": list(assist_config.retnet_adapter_target),
            "retnet_k_adapter_enabled": bool(assist_config.retnet_k_adapter_enabled),
            "retnet_context_adapter_enabled": bool(assist_config.retnet_context_adapter_enabled),
            "mechanism_ready": mechanism_ready,
        },
        "quality_decision": {
            "status": status,
            "reason": reason,
        },
    }
    metrics["needle"]["rank_delta"] = (
        None
        if metrics["needle"]["assist_rank"] is None or metrics["needle"]["no_assist_rank"] is None
        else metrics["needle"]["assist_rank"] - metrics["needle"]["no_assist_rank"]
    )
    metrics["needle"]["logprob_delta"] = (
        metrics["needle"]["assist_logprob"] - metrics["needle"]["no_assist_logprob"]
    )

    return LongContextAdmissionReport(
        preset=str(preset),
        device=str(target_device),
        dtype=dtype_name(target_dtype),
        vocabulary_size=int(vocabulary_size),
        sequence_length=sequence_length,
        attention_window_size=int(attention_window_size),
        needle_depth=float(needle_depth),
        metrics=metrics,
    )
