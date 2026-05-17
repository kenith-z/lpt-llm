"""LPT v2 checkpoint schema 与严格加载器。

本模块只接受 `checkpoint_format="lpt_v2_checkpoint"` 且 schema version 为 2
的产物，不对 v1 checkpoint、旧字段名或旧训练 recipe 做隐式兼容。checkpoint
保存的是完整 `ModelConfig` 快照、runtime metadata 和模型权重；Paged KV /
RetNetAssist / xLSTM 的运行态只保存元数据摘要，不保存 request 中间张量。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lpt_config import (
    LPT_V2_ARCHITECTURE_VERSION,
    MODEL_CONFIG_SCHEMA_VERSION,
    build_model_config_from_checkpoint,
    count_retnet_assist_enabled_layers,
    count_xlstm_memory_enabled_layers,
)
from lpt_runtime.files import atomic_torch_save

from .model_v2 import LPTV2


LPT_V2_CHECKPOINT_FORMAT = "lpt_v2_checkpoint"
LPT_V2_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class LoadedLPTV2Checkpoint:
    """严格加载后的 LPT v2 checkpoint。"""

    model: LPTV2
    checkpoint: dict[str, Any]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _layer_backend_decisions(model):
    """收集每层 Attention 后端选择结果，便于报告和 checkpoint 审计。"""
    decisions = []
    for layer_index, layer in enumerate(model.layers):
        decision = layer.attention_mixer.backend_decision.to_log_dict()
        decision["layer_index"] = layer_index
        decisions.append(decision)
    return decisions


def _state_schema_metadata(config):
    """生成 LayerStateV2 相关 schema 元数据。"""
    return {
        "layer_state_schema": "LayerStateV2",
        "attention_state": {
            "state_type": "attention_layer_state_v2",
            "cache_backend": config.cache_backend,
            "kv_cache_scope": config.kv_cache_scope,
            "page_block_size": config.page_block_size,
        },
        "retnet_assist_state": {
            "state_type": "retnet_assist_state",
            "state_dim": config.retnet_state_dim,
            "enabled_layer_count": count_retnet_assist_enabled_layers(config),
            "layers": config.retnet_assist_layers,
            "selected_layers": list(config.retnet_assist_selected_layers),
            "parameter_sharing": config.retnet_parameter_sharing,
            "state_sharing": config.retnet_state_sharing,
            "sharing_group_size": int(config.retnet_sharing_group_size),
            "lifecycle": config.retnet_state_lifecycle,
            "assist_mode": config.retnet_assist_mode,
            "adapter_rank": int(config.retnet_adapter_rank),
            "adapter_target": list(config.retnet_adapter_target),
            "k_adapter_enabled": bool(config.retnet_k_adapter_enabled),
            "context_adapter_enabled": bool(config.retnet_context_adapter_enabled),
            "context_adapter_alpha": float(config.retnet_context_adapter_alpha),
        },
        "moe_state": {
            "state_type": "moe_layer_state",
            "num_experts": config.moe_num_experts,
            "top_k": config.moe_top_k,
            "router_dtype": config.moe_router_dtype,
        },
        "xlstm_memory_state": {
            "state_type": "xlstm_memory_state",
            "enabled": config.xlstm_memory_enabled,
            "layers": config.xlstm_memory_layers,
            "selected_layers": list(config.xlstm_memory_selected_layers),
            "enabled_layer_count": count_xlstm_memory_enabled_layers(config),
            "granularity": config.xlstm_memory_granularity,
            "state_dim": config.xlstm_memory_state_dim,
            "lifecycle": config.xlstm_memory_state_lifecycle,
            "state_policy": config.xlstm_memory_state_policy,
            "decay_interval": config.xlstm_memory_state_decay_interval,
            "decay_factor": config.xlstm_memory_state_decay_factor,
            "reset_trigger_mode": list(config.xlstm_memory_reset_trigger_mode),
            "reset_boundary_policy": list(config.xlstm_memory_reset_boundary_policy),
        },
    }


def build_lpt_v2_checkpoint_payload(model, *, extra_metadata=None):
    """构造 LPT v2 checkpoint 载荷。"""
    if not isinstance(model, LPTV2):
        raise TypeError("只能保存 LPTV2 模型。")
    config = model.config
    if config.architecture_version != LPT_V2_ARCHITECTURE_VERSION:
        raise ValueError("LPT v2 checkpoint 只能保存 architecture_version='lpt_v2' 的模型。")

    runtime_metadata = {
        "attention_backend": {
            "policy": config.attention_backend_policy,
            "priority": list(config.attention_backend_priority),
            "decisions": _layer_backend_decisions(model),
        },
        "cache_backend": {
            "name": config.cache_backend,
            "kv_cache_scope": config.kv_cache_scope,
            "paged_kv": model.paged_kv_cache.runtime_metadata(),
        },
        "state_schema": _state_schema_metadata(config),
        "retnet_state_pool": model.retnet_state_pool.to_runtime_metadata(),
        "xlstm_memory_state_pool": model.xlstm_memory_state_pool.to_runtime_metadata(),
    }
    if extra_metadata:
        runtime_metadata["extra"] = dict(extra_metadata)

    return {
        "checkpoint_format": LPT_V2_CHECKPOINT_FORMAT,
        "checkpoint_schema_version": LPT_V2_CHECKPOINT_SCHEMA_VERSION,
        "architecture_version": LPT_V2_ARCHITECTURE_VERSION,
        "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "model_config": config.to_dict(),
        "runtime_metadata": runtime_metadata,
        "model_state_dict": model.state_dict(),
    }


def validate_lpt_v2_checkpoint_payload(checkpoint):
    """校验 checkpoint 外层 schema，并返回严格恢复出的 ModelConfig。"""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是字典。")
    if checkpoint.get("checkpoint_format") != LPT_V2_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint_format 不是 lpt_v2_checkpoint。")
    if checkpoint.get("checkpoint_schema_version") != LPT_V2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "不支持的 checkpoint_schema_version: "
            f"{checkpoint.get('checkpoint_schema_version')}，当前仅支持 {LPT_V2_CHECKPOINT_SCHEMA_VERSION}。"
        )
    if checkpoint.get("architecture_version") != LPT_V2_ARCHITECTURE_VERSION:
        raise ValueError("checkpoint architecture_version 不是 lpt_v2。")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_state_dict。")
    if "runtime_metadata" not in checkpoint:
        raise ValueError("checkpoint 缺少 runtime_metadata。")

    config = build_model_config_from_checkpoint(checkpoint)
    runtime_metadata = checkpoint["runtime_metadata"]
    cache_backend = runtime_metadata.get("cache_backend", {})
    if cache_backend.get("name") != config.cache_backend:
        raise ValueError("checkpoint runtime_metadata.cache_backend 与 model_config 不一致。")
    state_schema = runtime_metadata.get("state_schema", {})
    if state_schema.get("layer_state_schema") != "LayerStateV2":
        raise ValueError("checkpoint state_schema 不是 LayerStateV2。")
    return config


def _infer_vocabulary_size(state_dict):
    """从 tied embedding/lm_head 权重推断词表大小。"""
    embedding_weight = state_dict.get("token_embedding.weight")
    if embedding_weight is None:
        embedding_weight = state_dict.get("lm_head.weight")
    if embedding_weight is None:
        raise ValueError("model_state_dict 缺少 token_embedding.weight，无法推断 vocabulary_size。")
    return int(embedding_weight.shape[0])


def save_lpt_v2_checkpoint(model, path, *, extra_metadata=None):
    """保存 LPT v2 checkpoint。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_lpt_v2_checkpoint_payload(model, extra_metadata=extra_metadata)
    atomic_torch_save(payload, target_path)
    return target_path


def load_lpt_v2_checkpoint(path, *, vocabulary_size=None, map_location="cpu", strict=True):
    """严格加载 LPT v2 checkpoint。"""
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    config = validate_lpt_v2_checkpoint_payload(checkpoint)
    state_dict = checkpoint["model_state_dict"]
    model_vocab_size = _infer_vocabulary_size(state_dict) if vocabulary_size is None else int(vocabulary_size)
    # 先由 checkpoint 内的完整 ModelConfig 重建结构，再加载权重；strict=True 时任何
    # missing/unexpected key 都直接失败，避免旧结构权重被“部分兼容”地加载进 v2。
    model = LPTV2(model_vocab_size, config)
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    if strict and (missing_keys or unexpected_keys):
        raise ValueError(
            "checkpoint 权重键不匹配: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    return LoadedLPTV2Checkpoint(
        model=model,
        checkpoint=checkpoint,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
