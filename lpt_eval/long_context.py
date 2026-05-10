"""LPT v2 长上下文准入评测。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from lpt_config import LongContextEvalConfig
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
    last_logits = logits[0, -1].float()
    sorted_indices = torch.argsort(last_logits, descending=True)
    rank = (sorted_indices == int(target_token_id)).nonzero(as_tuple=False)
    if rank.numel() == 0:
        return None
    return int(rank[0, 0].item()) + 1


def _target_logprob(logits, target_token_id):
    log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)
    return float(log_probs[int(target_token_id)].detach().cpu())


@dataclass(frozen=True)
class LongContextAdmissionReport:
    """长上下文准入报告。"""

    preset: str
    device: str
    dtype: str
    vocabulary_size: int
    sequence_length: int
    attention_window_size: int
    metrics: dict
    checkpoint_path: str | None = None
    checkpoint_metadata: dict | None = None

    def to_dict(self):
        payload = {
            "report_type": "lpt_v2_long_context_admission",
            "preset": self.preset,
            "device": self.device,
            "dtype": self.dtype,
            "vocabulary_size": self.vocabulary_size,
            "sequence_length": self.sequence_length,
            "attention_window_size": self.attention_window_size,
            "metrics": dict(self.metrics),
        }
        if self.checkpoint_path is not None:
            payload["checkpoint_path"] = self.checkpoint_path
            payload["checkpoint_metadata"] = dict(self.checkpoint_metadata or {})
        return payload

    def to_markdown(self):
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
    return "n/a" if value is None else str(value)


def _markdown_float(value, *, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _checkpoint_training_metadata(checkpoint):
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
):
    loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=True)
    model = loaded.model
    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    model.to(device=target_device, dtype=target_dtype).eval()
    config = model.config
    vocabulary_size = int(model.token_embedding.num_embeddings)
    attention_window_size = int(attention_window_size or config.attention_window_size)
    sequence_length = int(sequence_length or attention_window_size * 2 + 4)
    if sequence_length <= attention_window_size:
        raise ValueError("sequence_length 必须大于 attention_window_size，才能验证窗口外信息。")

    needle_token_id = max(1, vocabulary_size - 3)
    input_ids = build_deterministic_input(
        vocabulary_size,
        1,
        sequence_length,
        offset=7,
        device=target_device,
    )
    input_ids[0, 1] = needle_token_id
    input_ids[0, -1] = 2
    code_math_ids = build_deterministic_input(
        vocabulary_size,
        1,
        min(sequence_length, attention_window_size + 4),
        offset=17,
        device=target_device,
    )
    format_ids = build_deterministic_input(
        vocabulary_size,
        1,
        min(sequence_length, attention_window_size + 2),
        offset=31,
        device=target_device,
    )

    with torch.no_grad():
        logits, states = model.prefill(input_ids, request_id="long-context-checkpoint")
        loss, ppl = next_token_loss(logits, input_ids)
        code_loss, _ = next_token_loss(model(code_math_ids)[0], code_math_ids)
        format_loss, _ = next_token_loss(model(format_ids)[0], format_ids)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    retnet_tokens = 0
    q_adapter_delta_norm = 0.0
    k_adapter_delta_norm = 0.0
    if states[0].retnet_assist is not None:
        retnet_tokens = int(states[0].retnet_assist.token_count)
        q_adapter_delta_norm = float(states[0].retnet_assist.q_adapter_delta_norm or 0.0)
        k_adapter_delta_norm = float(states[0].retnet_assist.k_adapter_delta_norm or 0.0)
    paged_window = int(states[0].attention.paged_kv_ref.window_token_count)
    mechanism_ready = bool(retnet_tokens > paged_window)
    status = "admit_checkpoint_path" if mechanism_ready else "close_or_debug"
    reason = (
        "已加载真实 v2 checkpoint 完成长上下文前向、PPL 与状态池准入；质量结论需结合独立验证集。"
        if mechanism_ready
        else "checkpoint 可加载但长上下文状态未跨越局部窗口，应检查配置或输入长度。"
    )
    assist_rank = _target_rank(logits, needle_token_id)
    metrics = {
        "needle": {
            "target_token_id": int(needle_token_id),
            "assist_rank": assist_rank,
            "no_assist_rank": None,
            "rank_delta": None,
            "assist_logprob": _target_logprob(logits, needle_token_id),
            "no_assist_logprob": None,
            "logprob_delta": None,
        },
        "long_text_ppl": {
            "assist_loss": loss,
            "no_assist_loss": None,
            "assist_ppl": ppl,
            "no_assist_ppl": None,
            "relative_delta": None,
        },
        "qa_retrieval": {
            "proxy": "needle_rank",
            "assist_reciprocal_rank": 1.0 / max(1, assist_rank or 1),
            "no_assist_reciprocal_rank": None,
        },
        "code_math": {
            "proxy": "deterministic_pattern_next_token_loss",
            "assist_loss": code_loss,
            "no_assist_loss": None,
        },
        "format_following": {
            "proxy": "structured_pattern_next_token_loss",
            "assist_loss": format_loss,
            "no_assist_loss": None,
        },
        "mechanism": {
            "assist_retnet_token_count": retnet_tokens,
            "paged_kv_window_token_count": paged_window,
            "q_adapter_delta_norm": q_adapter_delta_norm,
            "k_adapter_delta_norm": k_adapter_delta_norm,
            "mechanism_ready": mechanism_ready,
            "retnet_assist_mode": config.retnet_assist_mode,
            "retnet_adapter_target": list(config.retnet_adapter_target),
            "retnet_k_adapter_enabled": bool(config.retnet_k_adapter_enabled),
        },
        "quality_decision": {
            "status": status,
            "reason": reason,
        },
    }
    return LongContextAdmissionReport(
        preset=loaded.checkpoint["model_config"].get("preset_name", "checkpoint"),
        device=str(target_device),
        dtype=dtype_name(target_dtype),
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
        attention_window_size=attention_window_size,
        metrics=metrics,
        checkpoint_path=str(Path(checkpoint_path)),
        checkpoint_metadata=_checkpoint_training_metadata(loaded.checkpoint),
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
):
    """运行长上下文准入 smoke 评测。"""
    if checkpoint_path is not None:
        return _run_checkpoint_admission(
            checkpoint_path=checkpoint_path,
            sequence_length=sequence_length,
            attention_window_size=attention_window_size,
            device=device,
            dtype=dtype,
        )

    target_device = resolve_eval_device(device)
    target_dtype = resolve_eval_dtype(dtype, device=target_device)
    sequence_length = int(sequence_length or attention_window_size * 2 + 4)
    if sequence_length <= attention_window_size:
        raise ValueError("sequence_length 必须大于 attention_window_size，才能验证窗口外信息。")

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
    no_assist_config = build_lpt_v2_profile_config(
        LPT_V2_PAGED_KV_PROFILE,
        preset=preset,
        **common_overrides,
    )

    needle_token_id = max(1, int(vocabulary_size) - 3)
    set_eval_seed(seed)
    assist_model = LPTV2(vocabulary_size, assist_config).to(device=target_device, dtype=target_dtype).eval()
    set_eval_seed(seed)
    no_assist_model = LPTV2(vocabulary_size, no_assist_config).to(device=target_device, dtype=target_dtype).eval()

    input_ids = build_deterministic_input(
        vocabulary_size,
        1,
        sequence_length,
        offset=7,
        device=target_device,
    )
    input_ids[0, 1] = needle_token_id
    input_ids[0, -1] = 2
    code_math_ids = build_deterministic_input(
        vocabulary_size,
        1,
        min(sequence_length, attention_window_size + 4),
        offset=17,
        device=target_device,
    )
    format_ids = build_deterministic_input(
        vocabulary_size,
        1,
        min(sequence_length, attention_window_size + 2),
        offset=31,
        device=target_device,
    )

    with torch.no_grad():
        assist_logits, assist_states = assist_model.prefill(input_ids, request_id="long-context-assist")
        no_assist_logits, no_assist_states = no_assist_model.prefill(input_ids, request_id="long-context-no-assist")
        assist_loss, assist_ppl = next_token_loss(assist_logits, input_ids)
        no_assist_loss, no_assist_ppl = next_token_loss(no_assist_logits, input_ids)
        assist_code_loss, _ = next_token_loss(assist_model(code_math_ids)[0], code_math_ids)
        no_assist_code_loss, _ = next_token_loss(no_assist_model(code_math_ids)[0], code_math_ids)
        assist_format_loss, _ = next_token_loss(assist_model(format_ids)[0], format_ids)
        no_assist_format_loss, _ = next_token_loss(no_assist_model(format_ids)[0], format_ids)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    logit_delta_l2 = float((assist_logits.float() - no_assist_logits.float()).pow(2).mean().sqrt().detach().cpu())
    retnet_tokens = 0
    q_adapter_delta_norm = 0.0
    k_adapter_delta_norm = 0.0
    if assist_states[0].retnet_assist is not None:
        retnet_tokens = int(assist_states[0].retnet_assist.token_count)
        q_adapter_delta_norm = float(assist_states[0].retnet_assist.q_adapter_delta_norm or 0.0)
        k_adapter_delta_norm = float(assist_states[0].retnet_assist.k_adapter_delta_norm or 0.0)
    adapter_delta_l2 = 0.0
    if assist_states[0].retnet_assist is not None and assist_states[0].retnet_assist.summary is not None:
        q_adapter = assist_model.layers[0].attention_mixer.q_adapter
        summary = assist_states[0].retnet_assist.summary[:, None].to(device=target_device, dtype=target_dtype)
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
        and (logit_delta_l2 > 0.0 or adapter_delta_l2 > 0.0 or q_adapter_delta_norm > 0.0 or k_adapter_delta_norm > 0.0)
    )

    relative_ppl_delta = float((assist_ppl - no_assist_ppl) / max(no_assist_ppl, 1e-9))
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
            "assist_rank": _target_rank(assist_logits, needle_token_id),
            "no_assist_rank": _target_rank(no_assist_logits, needle_token_id),
            "rank_delta": None,
            "assist_logprob": _target_logprob(assist_logits, needle_token_id),
            "no_assist_logprob": _target_logprob(no_assist_logits, needle_token_id),
            "logprob_delta": None,
        },
        "long_text_ppl": {
            "assist_loss": assist_loss,
            "no_assist_loss": no_assist_loss,
            "assist_ppl": assist_ppl,
            "no_assist_ppl": no_assist_ppl,
            "relative_delta": relative_ppl_delta,
        },
        "qa_retrieval": {
            "proxy": "needle_rank",
            "assist_reciprocal_rank": 1.0 / max(1, _target_rank(assist_logits, needle_token_id)),
            "no_assist_reciprocal_rank": 1.0 / max(1, _target_rank(no_assist_logits, needle_token_id)),
        },
        "code_math": {
            "proxy": "deterministic_pattern_next_token_loss",
            "assist_loss": assist_code_loss,
            "no_assist_loss": no_assist_code_loss,
        },
        "format_following": {
            "proxy": "structured_pattern_next_token_loss",
            "assist_loss": assist_format_loss,
            "no_assist_loss": no_assist_format_loss,
        },
        "mechanism": {
            "assist_retnet_token_count": retnet_tokens,
            "paged_kv_window_token_count": paged_window,
            "logit_delta_l2": logit_delta_l2,
            "adapter_delta_l2": adapter_delta_l2,
            "q_adapter_delta_norm": q_adapter_delta_norm,
            "k_adapter_delta_norm": k_adapter_delta_norm,
            "alpha_q": float(assist_model.layers[0].attention_mixer.q_adapter.alpha_q.detach().cpu()),
            "retnet_assist_mode": assist_config.retnet_assist_mode,
            "retnet_adapter_target": list(assist_config.retnet_adapter_target),
            "retnet_k_adapter_enabled": bool(assist_config.retnet_k_adapter_enabled),
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
        metrics=metrics,
    )
