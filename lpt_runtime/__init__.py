"""运行时执行配置与设备映射工具。"""

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
    "CUDA_VISIBLE_DEVICES_ENV",
    "DeviceInfo",
    "DeviceMapPlan",
    "ExecutionConfig",
    "add_execution_arguments",
    "apply_inference_execution_plan",
    "build_execution_config",
    "describe_execution_plan",
    "discover_visible_cuda_devices",
    "parse_cuda_visible_devices",
    "resolve_execution_plan",
]
