"""LPT v2 运行 profile 配置。"""

from __future__ import annotations

from .constants import (
    DEFAULT_MODEL_SIZE_PRESET,
    LPT_V2_ASSIST_PROFILE,
    LPT_V2_BASE_PROFILE,
    LPT_V2_BASELINE_PROFILES,
    LPT_V2_BOOTSTRAP_PROFILE,
    LPT_V2_MEMORY_PROFILE,
    LPT_V2_PAGED_KV_PROFILE,
    LPT_V2_SDPA_LOCAL_PROFILE,
)
from .model_config import DENSE_KV_CACHE_BACKEND, PAGED_KV_CACHE_BACKEND, ModelConfig


def _with_memory_disabled(payload):
    """关闭 xLSTM 相关字段，并把 MoE router 输入切回纯 FFN norm。"""
    payload.update(
        {
            "xlstm_memory_enabled": False,
            "xlstm_memory_layers": "disabled",
            "moe_router_input_mode": "ffn_norm_only_eval",
        }
    )
    return payload


def build_lpt_v2_profile_config(profile_name, *, preset=DEFAULT_MODEL_SIZE_PRESET, **overrides):
    """按运行 profile 展开 ModelConfig。"""
    # profile 是 CLI/测试传入的运行剖面名，先转字符串再校验，避免枚举外值静默落入默认配置。
    profile = str(profile_name)
    if profile not in LPT_V2_BASELINE_PROFILES:
        raise ValueError(f"未知 LPT v2 profile: {profile}")

    # payload 只保存 profile 相对 preset 的差异字段，最终仍交给 ModelConfig.from_preset 做完整校验。
    payload = {}
    if profile == LPT_V2_BOOTSTRAP_PROFILE:
        # bootstrap 关闭 Paged KV、RetNetAssist 和 xLSTMAssist，只保留最小 dense attention + 单 expert。
        payload.update(
            {
                "cache_backend": DENSE_KV_CACHE_BACKEND,
                "retnet_assist_enabled": False,
                "retnet_assist_layers": "disabled",
                "retnet_context_adapter_enabled": False,
                "moe_num_experts": 1,
                "moe_top_k": 1,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_SDPA_LOCAL_PROFILE:
        # SDPA local profile 用 dense KV 跑局部 attention，作为 Paged KV 接入前的稳定基线。
        payload.update(
            {
                "cache_backend": DENSE_KV_CACHE_BACKEND,
                "retnet_assist_enabled": False,
                "retnet_assist_layers": "disabled",
                "retnet_context_adapter_enabled": False,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_PAGED_KV_PROFILE:
        # Paged KV profile 只切换缓存后端，用于隔离 Paged KV 对资源和行为的影响。
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": False,
                "retnet_assist_layers": "disabled",
                "retnet_context_adapter_enabled": False,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_ASSIST_PROFILE:
        # Assist profile 是默认主线：启用 Paged KV 与 every_4_layers RetNetAssist，但仍关闭 xLSTMAssist。
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": True,
                "retnet_assist_layers": "every_4_layers",
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_BASE_PROFILE:
        # Base profile 在 Assist 的基础上固定更完整的 MoE 配置，用于正式基线对照。
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": True,
                "retnet_assist_layers": "every_4_layers",
                "moe_num_experts": 8,
                "moe_top_k": 2,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_MEMORY_PROFILE:
        # Memory profile 打开 xLSTMAssist，并允许 router 读取 memory-augmented FFN 输入。
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": True,
                "retnet_assist_layers": "every_4_layers",
                "moe_num_experts": 8,
                "moe_top_k": 2,
                "xlstm_memory_enabled": True,
                "xlstm_memory_layers": "every_4_layers",
                "moe_router_input_mode": "memory_augmented_input",
            }
        )

    # 显式 overrides 优先级最高，方便 smoke test 和 CLI 做最小覆盖。
    payload.update(overrides)
    return ModelConfig.from_preset(preset, **payload)


def parse_profile_list(raw_profiles=None):
    """解析逗号分隔 profile 列表。"""
    # 空值和 all 都表示跑完整基线矩阵。
    if raw_profiles is None or str(raw_profiles).strip() in {"", "all"}:
        return LPT_V2_BASELINE_PROFILES
    # profiles 保留用户输入顺序，报告输出也会按该顺序排列。
    profiles = tuple(profile.strip() for profile in str(raw_profiles).split(",") if profile.strip())
    unknown_profiles = sorted(set(profiles) - set(LPT_V2_BASELINE_PROFILES))
    if unknown_profiles:
        raise ValueError(f"未知 LPT v2 profile: {unknown_profiles}")
    return profiles
