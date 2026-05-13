"""LPT v2 MoE-aware 参数统计。"""

from __future__ import annotations

from dataclasses import dataclass

from lpt_config import (
    ModelConfig,
    PARAMETER_COUNT_POLICY_MOE_AWARE,
    is_retnet_assist_enabled_for_layer,
    count_xlstm_memory_enabled_layers,
)


FP32_BYTES = 4


@dataclass(frozen=True)
class MoEAwareParameterReport:
    """区分 MoE 物理参数与每 token 激活参数的统计报告。"""

    total_physical_params: int
    active_params_per_token: int
    shared_params: int
    expert_params: int
    router_params: int
    adapter_params: int
    state_runtime_bytes: int
    module_breakdown: dict[str, int]
    active_module_breakdown: dict[str, int]
    model_size_preset: str
    parameter_count_policy: str = PARAMETER_COUNT_POLICY_MOE_AWARE

    def to_dict(self):
        return {
            "parameter_count_policy": self.parameter_count_policy,
            "model_size_preset": self.model_size_preset,
            "total_physical_params": self.total_physical_params,
            "active_params_per_token": self.active_params_per_token,
            "shared_params": self.shared_params,
            "expert_params": self.expert_params,
            "router_params": self.router_params,
            "adapter_params": self.adapter_params,
            "state_runtime_bytes": self.state_runtime_bytes,
            "module_breakdown": dict(self.module_breakdown),
            "active_module_breakdown": dict(self.active_module_breakdown),
        }


def _swiglu_intermediate_size(hidden_size):
    intermediate_size = int(8 * int(hidden_size) / 3)
    return ((intermediate_size + 255) // 256) * 256


def _resolve_layer_interval(layer_policy, default_interval=None):
    policy = str(layer_policy)
    if policy == "all_layers":
        return 1
    if policy == "every_n_layers":
        return default_interval
    if policy.startswith("every_") and policy.endswith("_layers"):
        raw_interval = policy.removeprefix("every_").removesuffix("_layers")
        if raw_interval.isdigit():
            return int(raw_interval)
    return None


def _count_enabled_layers(num_layers, layer_policy, *, default_interval=None):
    policy = str(layer_policy)
    if policy in {"disabled", "selected_layers"}:
        return 0
    interval = _resolve_layer_interval(policy, default_interval=default_interval)
    if interval is None:
        return 0
    if interval <= 0:
        raise ValueError("启用层 interval 必须为正整数。")
    return (int(num_layers) + interval - 1) // interval


def _enabled_retnet_layer_indices(config):
    return tuple(
        layer_index
        for layer_index in range(int(config.num_layers))
        if is_retnet_assist_enabled_for_layer(config, layer_index)
    )


def _retnet_group_ids(config, layer_indices):
    group_size = int(config.retnet_sharing_group_size)
    if group_size <= 0:
        raise ValueError("retnet_sharing_group_size 必须为正整数。")
    return tuple(sorted({int(layer_index) // group_size for layer_index in layer_indices}))


def _count_retnet_parameter_groups(config):
    enabled_layers = _enabled_retnet_layer_indices(config)
    enabled_layer_count = len(enabled_layers)
    if enabled_layer_count == 0:
        return 0, 0
    if config.retnet_parameter_sharing == "global":
        parameter_groups = 1
    elif config.retnet_parameter_sharing == "group":
        parameter_groups = len(_retnet_group_ids(config, enabled_layers))
    else:
        parameter_groups = enabled_layer_count
    if config.retnet_state_sharing == "group":
        state_slots = len(_retnet_group_ids(config, enabled_layers))
    else:
        state_slots = enabled_layer_count
    return parameter_groups, state_slots


def _count_xlstm_enabled_layers(config):
    return count_xlstm_memory_enabled_layers(config)


def estimate_moe_aware_parameter_counts(
    config,
    *,
    vocabulary_size=0,
    state_dtype_bytes=FP32_BYTES,
):
    """按 LPT v2 配置估算 MoE-aware 参数量。

    统计口径：
    - experts 物理参数按全部 experts 计入。
    - active_params_per_token 只按 top_k 计入激活 expert。
    - embedding 与 lm_head 当前按 tied weight 计一次。
    """
    model_config = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
    hidden_size = int(model_config.hidden_size)
    num_layers = int(model_config.num_layers)
    kv_hidden_size = int(model_config.num_kv_heads) * int(model_config.head_dim)
    vocabulary_size = int(vocabulary_size or 0)
    state_dtype_bytes = int(state_dtype_bytes)

    embedding_params = vocabulary_size * hidden_size
    attention_params_per_layer = (
        hidden_size * hidden_size
        + hidden_size * kv_hidden_size
        + hidden_size * kv_hidden_size
        + hidden_size * hidden_size
    )
    attention_params = num_layers * attention_params_per_layer
    dense_norm_params = (2 * num_layers + 1) * hidden_size

    retnet_parameter_groups, retnet_state_slots = _count_retnet_parameter_groups(model_config)
    retnet_core_params_per_group = (
        hidden_size * int(model_config.retnet_state_dim)
        + int(model_config.retnet_state_dim) * int(model_config.retnet_state_dim)
    )
    retnet_adapter_params_per_group = (
        int(model_config.retnet_state_dim) * int(model_config.retnet_adapter_rank)
        + int(model_config.retnet_adapter_rank) * hidden_size
        + (1 if model_config.retnet_adapter_alpha_q_trainable else 0)
    )
    retnet_k_adapter_params_per_group = 0
    if model_config.retnet_k_adapter_enabled:
        retnet_k_adapter_params_per_group = (
            int(model_config.retnet_state_dim) * int(model_config.retnet_adapter_rank)
            + int(model_config.retnet_adapter_rank) * kv_hidden_size
            + (1 if model_config.retnet_adapter_alpha_k_trainable else 0)
        )
    retnet_context_adapter_params_per_group = 0
    if model_config.retnet_context_adapter_enabled:
        retnet_context_adapter_params_per_group = (
            int(model_config.retnet_state_dim) * int(model_config.retnet_adapter_rank)
            + int(model_config.retnet_adapter_rank) * hidden_size
            + 1
        )
    retnet_core_params = retnet_parameter_groups * retnet_core_params_per_group
    retnet_adapter_params = retnet_parameter_groups * retnet_adapter_params_per_group
    retnet_k_adapter_params = retnet_parameter_groups * retnet_k_adapter_params_per_group
    retnet_context_adapter_params = retnet_parameter_groups * retnet_context_adapter_params_per_group

    swiglu_intermediate_size = _swiglu_intermediate_size(hidden_size)
    swiglu_expert_params = 3 * hidden_size * swiglu_intermediate_size
    expert_params = num_layers * int(model_config.moe_num_experts) * swiglu_expert_params
    active_expert_params = num_layers * int(model_config.moe_top_k) * swiglu_expert_params
    router_params = num_layers * hidden_size * int(model_config.moe_num_experts)

    xlstm_enabled_layers = _count_xlstm_enabled_layers(model_config)
    xlstm_core_params_per_layer = (
        hidden_size * int(model_config.xlstm_memory_state_dim)
        + int(model_config.xlstm_memory_state_dim) * int(model_config.xlstm_memory_state_dim)
    )
    xlstm_adapter_params_per_layer = (
        int(model_config.xlstm_memory_state_dim) * int(model_config.xlstm_memory_adapter_rank)
        + int(model_config.xlstm_memory_adapter_rank) * hidden_size
        + 1
    )
    xlstm_core_params = xlstm_enabled_layers * xlstm_core_params_per_layer
    xlstm_adapter_params = xlstm_enabled_layers * xlstm_adapter_params_per_layer

    shared_params = (
        embedding_params
        + attention_params
        + dense_norm_params
        + retnet_core_params
        + xlstm_core_params
    )
    adapter_params = (
        retnet_adapter_params
        + retnet_k_adapter_params
        + retnet_context_adapter_params
        + xlstm_adapter_params
    )
    total_physical_params = shared_params + expert_params + router_params + adapter_params
    active_params_per_token = shared_params + active_expert_params + router_params + adapter_params
    state_runtime_bytes = (
        retnet_state_slots * int(model_config.retnet_state_dim) * state_dtype_bytes
        + xlstm_enabled_layers * int(model_config.xlstm_memory_state_dim) * state_dtype_bytes
    )

    module_breakdown = {
        "token_embedding_tied_lm_head": embedding_params,
        "attention": attention_params,
        "dense_norms": dense_norm_params,
        "retnet_assist_core": retnet_core_params,
        "retnet_q_adapter": retnet_adapter_params,
        "retnet_k_adapter": retnet_k_adapter_params,
        "retnet_context_adapter": retnet_context_adapter_params,
        "swiglu_experts": expert_params,
        "moe_router": router_params,
        "xlstm_memory_core": xlstm_core_params,
        "xlstm_memory_adapter": xlstm_adapter_params,
    }
    active_module_breakdown = {
        **module_breakdown,
        "swiglu_experts": active_expert_params,
    }

    return MoEAwareParameterReport(
        total_physical_params=int(total_physical_params),
        active_params_per_token=int(active_params_per_token),
        shared_params=int(shared_params),
        expert_params=int(expert_params),
        router_params=int(router_params),
        adapter_params=int(adapter_params),
        state_runtime_bytes=int(state_runtime_bytes),
        module_breakdown={key: int(value) for key, value in module_breakdown.items()},
        active_module_breakdown={
            key: int(value)
            for key, value in active_module_breakdown.items()
        },
        model_size_preset=model_config.model_size_preset,
    )
