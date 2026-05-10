"""运行时执行配置与设备映射工具。"""

from .attention_backend import (
    ATTENTION_BACKEND_CAPABILITIES,
    ATTENTION_BACKEND_CAPABILITY_NAMES,
    AttentionBackendAttempt,
    AttentionBackendCapability,
    AttentionBackendDecision,
    detect_available_attention_backends,
    resolve_attention_backend,
)
from .execution import (
    CUDA_VISIBLE_DEVICES_ENV,
    DeviceInfo,
    DeviceMapPlan,
    ExecutionConfig,
    add_execution_arguments,
    apply_inference_execution_plan,
    build_execution_config,
    describe_execution_plan,
    discover_visible_cuda_devices,
    parse_cuda_visible_devices,
    resolve_execution_plan,
)

__all__ = [
    "ATTENTION_BACKEND_CAPABILITIES",
    "ATTENTION_BACKEND_CAPABILITY_NAMES",
    "AttentionBackendAttempt",
    "AttentionBackendCapability",
    "AttentionBackendDecision",
    "CUDA_VISIBLE_DEVICES_ENV",
    "DeviceInfo",
    "DeviceMapPlan",
    "ExecutionConfig",
    "add_execution_arguments",
    "apply_inference_execution_plan",
    "build_execution_config",
    "describe_execution_plan",
    "detect_available_attention_backends",
    "discover_visible_cuda_devices",
    "parse_cuda_visible_devices",
    "resolve_attention_backend",
    "resolve_execution_plan",
]
