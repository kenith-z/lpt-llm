"""Attention 后端能力描述与选择策略。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys

import torch

from lpt_config import (
    AUTO_ATTENTION_BACKEND_POLICY,
    DEFAULT_ATTENTION_BACKEND_PRIORITY,
    FLASH_ATTENTION_2_BACKEND,
    FLASH_ATTENTION_3_BACKEND,
    SDPA_ATTENTION_BACKEND,
    SUPPORTED_ATTENTION_BACKEND_POLICIES,
    SUPPORTED_ATTENTION_BACKENDS,
)


ATTENTION_BACKEND_CAPABILITY_NAMES = (
    "training",
    "prefill",
    "decode_kvcache",
    "paged_kv",
    "sliding_window",
    "gqa",
    "longrope2",
)


def _as_tuple(value):
    """把配置字段统一成 tuple，便于能力匹配和日志记录。"""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return tuple(value)


@dataclass(frozen=True)
class AttentionBackendCapability:
    """单个 Attention 后端的能力声明。"""

    name: str
    training: bool
    prefill: bool
    decode_kvcache: bool
    paged_kv: bool
    sliding_window: bool
    gqa: bool
    longrope2: bool
    dtypes: tuple[str, ...]
    platforms: tuple[str, ...]

    def supports(self, required_capabilities=(), *, dtype=None, platform=None):
        """判断当前后端是否满足能力、dtype 和平台约束。"""
        for capability_name in _as_tuple(required_capabilities):
            if capability_name not in ATTENTION_BACKEND_CAPABILITY_NAMES:
                raise ValueError(f"未知 Attention capability: {capability_name}")
            if not bool(getattr(self, capability_name)):
                return False
        if dtype is not None and normalize_dtype_name(dtype) not in self.dtypes:
            return False
        if platform is not None and normalize_platform_name(platform) not in self.platforms:
            return False
        return True

    def to_dict(self):
        """序列化后端能力，供 checkpoint/runtime metadata 记录。"""
        return {
            "name": self.name,
            "training": self.training,
            "prefill": self.prefill,
            "decode_kvcache": self.decode_kvcache,
            "paged_kv": self.paged_kv,
            "sliding_window": self.sliding_window,
            "gqa": self.gqa,
            "longrope2": self.longrope2,
            "dtypes": list(self.dtypes),
            "platforms": list(self.platforms),
        }


@dataclass(frozen=True)
class AttentionBackendAttempt:
    """一次后端候选尝试记录。"""

    backend: str
    status: str
    reason: str | None = None

    def to_dict(self):
        """序列化一次候选后端尝试结果。"""
        return {
            "backend": self.backend,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttentionBackendDecision:
    """可落盘的后端选择结果。"""

    selected_backend: str
    capability: AttentionBackendCapability
    policy: str
    priority: tuple[str, ...]
    attempted_backends: tuple[AttentionBackendAttempt, ...]
    fallback_reason: str | None = None
    warnings: tuple[str, ...] = ()

    def to_log_dict(self):
        """序列化最终选择结果，便于每层 Attention 后端审计。"""
        return {
            "selected_backend": self.selected_backend,
            "policy": self.policy,
            "priority": list(self.priority),
            "fallback_reason": self.fallback_reason,
            "attempted_backends": [
                attempt.to_dict()
                for attempt in self.attempted_backends
            ],
            "capability": self.capability.to_dict(),
            "warnings": list(self.warnings),
        }


def normalize_dtype_name(dtype):
    """统一 dtype 名称，便于配置、日志和测试比较。"""
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    return str(dtype).removeprefix("torch.")


def normalize_platform_name(platform):
    """把 sys.platform 这类平台字符串规整到能力表使用的名称。"""
    platform_text = str(platform).lower()
    if platform_text.startswith("linux"):
        return "linux"
    if platform_text.startswith("win"):
        return "windows"
    if platform_text.startswith("darwin") or platform_text.startswith("mac"):
        return "darwin"
    return platform_text


def _sdpa_supports_gqa():
    """探测当前 PyTorch SDPA 是否支持 enable_gqa 参数。"""
    try:
        import inspect

        return "enable_gqa" in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters
    except (TypeError, ValueError):
        return False


ATTENTION_BACKEND_CAPABILITIES = {
    FLASH_ATTENTION_3_BACKEND: AttentionBackendCapability(
        name=FLASH_ATTENTION_3_BACKEND,
        training=True,
        prefill=True,
        decode_kvcache=True,
        paged_kv=True,
        sliding_window=True,
        gqa=True,
        longrope2=True,
        dtypes=("float16", "bfloat16"),
        platforms=("linux",),
    ),
    FLASH_ATTENTION_2_BACKEND: AttentionBackendCapability(
        name=FLASH_ATTENTION_2_BACKEND,
        training=True,
        prefill=True,
        decode_kvcache=True,
        paged_kv=False,
        sliding_window=True,
        gqa=True,
        longrope2=True,
        dtypes=("float16", "bfloat16"),
        platforms=("linux", "windows"),
    ),
    SDPA_ATTENTION_BACKEND: AttentionBackendCapability(
        name=SDPA_ATTENTION_BACKEND,
        training=True,
        prefill=True,
        decode_kvcache=True,
        paged_kv=False,
        sliding_window=True,
        gqa=_sdpa_supports_gqa(),
        longrope2=True,
        dtypes=("float32", "float16", "bfloat16"),
        platforms=("linux", "windows", "darwin"),
    ),
}


def detect_available_attention_backends():
    """轻量探测当前 Python 环境中可导入的 Attention 后端。"""
    available = {SDPA_ATTENTION_BACKEND}
    if importlib.util.find_spec("flash_attn_3") is not None:
        available.add(FLASH_ATTENTION_3_BACKEND)
    if importlib.util.find_spec("flash_attn") is not None:
        available.add(FLASH_ATTENTION_2_BACKEND)
    return tuple(backend for backend in SUPPORTED_ATTENTION_BACKENDS if backend in available)


def _normalize_priority(priority):
    """校验并规范化 attention backend 优先级。"""
    normalized_priority = tuple(
        str(backend)
        for backend in (_as_tuple(priority) or DEFAULT_ATTENTION_BACKEND_PRIORITY)
    )
    unknown_backends = sorted(set(normalized_priority) - set(SUPPORTED_ATTENTION_BACKENDS))
    if unknown_backends:
        raise ValueError(f"attention backend priority 包含未知后端: {unknown_backends}")
    if not normalized_priority:
        raise ValueError("attention backend priority 不能为空。")
    return normalized_priority


def resolve_attention_backend(
    policy=AUTO_ATTENTION_BACKEND_POLICY,
    *,
    priority=None,
    required_capabilities=(),
    dtype=None,
    platform=None,
    available_backends=None,
):
    """按策略选择 Attention 后端，并返回可记录的决策对象。"""
    normalized_policy = str(policy)
    if normalized_policy not in SUPPORTED_ATTENTION_BACKEND_POLICIES:
        raise ValueError(f"attention backend policy 必须是 {SUPPORTED_ATTENTION_BACKEND_POLICIES} 之一。")

    normalized_priority = _normalize_priority(priority)
    candidate_backends = (
        normalized_priority
        if normalized_policy == AUTO_ATTENTION_BACKEND_POLICY
        else (normalized_policy,)
    )
    available = set(
        detect_available_attention_backends()
        if available_backends is None
        else tuple(str(backend) for backend in _as_tuple(available_backends))
    )
    current_platform = normalize_platform_name(platform or sys.platform)

    attempts = []
    for backend in candidate_backends:
        capability = ATTENTION_BACKEND_CAPABILITIES[backend]
        if backend not in available:
            attempts.append(AttentionBackendAttempt(backend, "skipped", "backend_unavailable"))
            continue
        if not capability.supports(
            required_capabilities,
            dtype=dtype,
            platform=current_platform,
        ):
            # 不满足 sliding_window / longrope2 / dtype / platform 等硬约束时直接跳过，
            # 避免自动选择一个语义不完整但可 import 的后端。
            attempts.append(AttentionBackendAttempt(backend, "skipped", "capability_mismatch"))
            continue

        attempts.append(AttentionBackendAttempt(backend, "selected"))
        fallback_reason = None
        warnings = ()
        if normalized_policy == AUTO_ATTENTION_BACKEND_POLICY and backend != normalized_priority[0]:
            skipped = ", ".join(
                attempt.backend
                for attempt in attempts
                if attempt.status == "skipped"
            )
            fallback_reason = f"fallback_from={skipped}"
            warnings = (fallback_reason,)
        return AttentionBackendDecision(
            selected_backend=backend,
            capability=capability,
            policy=normalized_policy,
            priority=normalized_priority,
            attempted_backends=tuple(attempts),
            fallback_reason=fallback_reason,
            warnings=warnings,
        )

    attempt_summary = ", ".join(
        f"{attempt.backend}:{attempt.reason or attempt.status}"
        for attempt in attempts
    )
    raise ValueError(f"没有可用 Attention 后端满足要求: {attempt_summary}")
