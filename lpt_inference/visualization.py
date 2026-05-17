"""LPT v2 推理与 checkpoint 展示工具。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lpt_model import estimate_moe_aware_parameter_counts, load_lpt_v2_checkpoint


def _format_int(value):
    """把整数按千分位格式展示。"""
    return f"{int(value):,}"


PARAMETER_SUMMARY_LABELS = {
    "model_size_preset": "模型规格(model_size_preset)",
    "total_physical_params": "物理总参数(total_physical_params)",
    "active_params_per_token": "每Token激活参数(active_params_per_token)",
    "shared_params": "共享参数(shared_params)",
    "expert_params": "专家参数(expert_params)",
    "router_params": "路由参数(router_params)",
    "adapter_params": "适配器参数(adapter_params)",
    "state_runtime_bytes": "运行态状态字节(state_runtime_bytes)",
}


def _format_summary_line(key, value, *, integer=False):
    """按固定中文标签渲染参数摘要行。"""
    rendered_value = _format_int(value) if integer else value
    return f"{PARAMETER_SUMMARY_LABELS[key]}={rendered_value}"


def display_model_parameter_summary(model, *, vocabulary_size=None):
    """打印 MoE-aware 参数摘要。"""
    vocab_size = model.vocabulary_size if vocabulary_size is None else int(vocabulary_size)
    report = estimate_moe_aware_parameter_counts(model.config, vocabulary_size=vocab_size)
    payload = report.to_dict()
    lines = [
        "LPT v2 参数摘要",
        _format_summary_line("model_size_preset", payload["model_size_preset"]),
        _format_summary_line("total_physical_params", payload["total_physical_params"], integer=True),
        _format_summary_line("active_params_per_token", payload["active_params_per_token"], integer=True),
        _format_summary_line("shared_params", payload["shared_params"], integer=True),
        _format_summary_line("expert_params", payload["expert_params"], integer=True),
        _format_summary_line("router_params", payload["router_params"], integer=True),
        _format_summary_line("adapter_params", payload["adapter_params"], integer=True),
        _format_summary_line("state_runtime_bytes", payload["state_runtime_bytes"], integer=True),
    ]
    print("\n".join(lines))
    return payload


def display_checkpoint_summary(checkpoint_path):
    """加载并打印 checkpoint 关键元数据。"""
    loaded = load_lpt_v2_checkpoint(checkpoint_path, map_location="cpu", strict=False)
    checkpoint = loaded.checkpoint
    summary = {
        "path": str(Path(checkpoint_path)),
        "checkpoint_format": checkpoint.get("checkpoint_format"),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "architecture_version": checkpoint.get("architecture_version"),
        "model_size_preset": loaded.model.config.model_size_preset,
        "vocabulary_size": loaded.model.vocabulary_size,
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
