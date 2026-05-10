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
    profile = str(profile_name)
    if profile not in LPT_V2_BASELINE_PROFILES:
        raise ValueError(f"未知 LPT v2 profile: {profile}")

    payload = {}
    if profile == LPT_V2_BOOTSTRAP_PROFILE:
        payload.update(
            {
                "cache_backend": DENSE_KV_CACHE_BACKEND,
                "retnet_assist_enabled": False,
                "moe_num_experts": 1,
                "moe_top_k": 1,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_SDPA_LOCAL_PROFILE:
        payload.update(
            {
                "cache_backend": DENSE_KV_CACHE_BACKEND,
                "retnet_assist_enabled": False,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_PAGED_KV_PROFILE:
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": False,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_ASSIST_PROFILE:
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": True,
                "retnet_assist_layers": "all_layers",
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_BASE_PROFILE:
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": True,
                "retnet_assist_layers": "all_layers",
                "moe_num_experts": 8,
                "moe_top_k": 2,
            }
        )
        _with_memory_disabled(payload)
    elif profile == LPT_V2_MEMORY_PROFILE:
        payload.update(
            {
                "cache_backend": PAGED_KV_CACHE_BACKEND,
                "retnet_assist_enabled": True,
                "retnet_assist_layers": "all_layers",
                "moe_num_experts": 8,
                "moe_top_k": 2,
                "xlstm_memory_enabled": True,
                "xlstm_memory_layers": "every_n_layers",
                "moe_router_input_mode": "memory_augmented_input",
            }
        )

    payload.update(overrides)
    return ModelConfig.from_preset(preset, **payload)


def parse_profile_list(raw_profiles=None):
    """解析逗号分隔 profile 列表。"""
    if raw_profiles is None or str(raw_profiles).strip() in {"", "all"}:
        return LPT_V2_BASELINE_PROFILES
    profiles = tuple(profile.strip() for profile in str(raw_profiles).split(",") if profile.strip())
    unknown_profiles = sorted(set(profiles) - set(LPT_V2_BASELINE_PROFILES))
    if unknown_profiles:
        raise ValueError(f"未知 LPT v2 profile: {unknown_profiles}")
    return profiles
