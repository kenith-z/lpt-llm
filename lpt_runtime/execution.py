"""单机执行配置、可见 CUDA 设备发现与推理 device map。"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch

from lpt_config import GlobalConfig


CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"
SINGLE_EXECUTION_MODE = "single"
AUTO_EXECUTION_MODE = "auto"
MODEL_PARALLEL_EXECUTION_MODE = "model_parallel"
TENSOR_PARALLEL_EXECUTION_MODE = "tensor_parallel"
FSDP_EXECUTION_MODE = "fsdp"
SUPPORTED_EXECUTION_MODES = (
    SINGLE_EXECUTION_MODE,
    AUTO_EXECUTION_MODE,
    MODEL_PARALLEL_EXECUTION_MODE,
    TENSOR_PARALLEL_EXECUTION_MODE,
    FSDP_EXECUTION_MODE,
)
AUTO_DEVICE_MAP = "auto"


@dataclass(frozen=True)
class DeviceInfo:
    """一张对当前进程可见的逻辑 CUDA 卡。"""

    logical_index: int
    logical_name: str
    visible_device: str | None
    name: str | None = None
    total_memory_bytes: int | None = None

    @property
    def memory_weight(self):
        if self.total_memory_bytes is None or self.total_memory_bytes <= 0:
            return 1.0
        return float(self.total_memory_bytes)

    def to_dict(self):
        return {
            "logical_index": self.logical_index,
            "logical_name": self.logical_name,
            "visible_device": self.visible_device,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
        }


@dataclass(frozen=True)
class ExecutionConfig:
    """用户侧执行配置。

    `CUDA_VISIBLE_DEVICES` 只决定本进程可见资源池；这里的 device map
    始终使用 PyTorch 看到的逻辑设备名，例如 cuda:0 / cuda:1。
    """

    mode: str = SINGLE_EXECUTION_MODE
    device_map: str | Path | dict[str, Any] | list[Any] | tuple[Any, ...] = AUTO_DEVICE_MAP
    print_device_map: bool = True

    def __post_init__(self):
        normalized_mode = str(self.mode)
        if normalized_mode not in SUPPORTED_EXECUTION_MODES:
            raise ValueError(f"execution mode 必须是 {SUPPORTED_EXECUTION_MODES} 之一。")
        object.__setattr__(self, "mode", normalized_mode)


@dataclass(frozen=True)
class DeviceMapPlan:
    """解析后的执行计划。"""

    mode: str
    primary_device: str
    visible_cuda_devices: tuple[DeviceInfo, ...]
    layer_devices: tuple[str, ...]
    module_devices: dict[str, str]
    source: str
    warnings: tuple[str, ...] = ()

    @property
    def is_model_parallel(self):
        return self.mode == MODEL_PARALLEL_EXECUTION_MODE

    @property
    def torch_primary_device(self):
        return torch.device(self.primary_device)

    @property
    def state_dict_map_location(self):
        if self.is_model_parallel:
            return "cpu"
        return self.torch_primary_device

    def to_dict(self):
        return {
            "mode": self.mode,
            "primary_device": self.primary_device,
            "visible_cuda_devices": [device.to_dict() for device in self.visible_cuda_devices],
            "layer_devices": list(self.layer_devices),
            "module_devices": dict(self.module_devices),
            "source": self.source,
            "warnings": list(self.warnings),
        }


def parse_cuda_visible_devices(raw_value=None):
    """解析 CUDA_VISIBLE_DEVICES。

    返回 None 表示环境变量未设置；返回空元组表示显式隐藏 CUDA。
    """
    if raw_value is None:
        raw_value = os.environ.get(CUDA_VISIBLE_DEVICES_ENV)
    if raw_value is None:
        return None

    normalized_value = str(raw_value).strip()
    if normalized_value in {"", "-1", "none", "None"}:
        return ()
    return tuple(value.strip() for value in normalized_value.split(",") if value.strip())


def discover_visible_cuda_devices():
    """按 PyTorch 当前可见视角返回逻辑 CUDA 卡列表。"""
    if not torch.cuda.is_available():
        return ()

    visible_tokens = parse_cuda_visible_devices()
    device_infos = []
    for logical_index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(logical_index)
        visible_device = None
        if visible_tokens is not None and logical_index < len(visible_tokens):
            visible_device = visible_tokens[logical_index]
        device_infos.append(
            DeviceInfo(
                logical_index=logical_index,
                logical_name=f"cuda:{logical_index}",
                visible_device=visible_device,
                name=getattr(properties, "name", None),
                total_memory_bytes=int(getattr(properties, "total_memory", 0) or 0),
            )
        )
    return tuple(device_infos)


def _normalize_device_name(device):
    if isinstance(device, int):
        return f"cuda:{device}"
    device_text = str(device)
    if device_text.isdigit():
        return f"cuda:{device_text}"
    return device_text


def _validate_device_names(device_names, *, visible_cuda_devices, allow_cpu=True):
    visible_names = {device.logical_name for device in visible_cuda_devices}
    invalid_devices = []
    for device_name in device_names:
        normalized_name = _normalize_device_name(device_name)
        if normalized_name == "cpu" and allow_cpu:
            continue
        if normalized_name not in visible_names:
            invalid_devices.append(normalized_name)
    if invalid_devices:
        raise ValueError(
            "device_map 包含当前进程不可见的设备: "
            f"{sorted(set(invalid_devices))}；可见 CUDA 设备为 {sorted(visible_names)}。"
        )


def _allocate_layers_by_memory(num_layers, visible_cuda_devices):
    if num_layers <= 0:
        raise ValueError("num_layers 必须为正整数。")
    if not visible_cuda_devices:
        return ("cpu",) * num_layers
    if len(visible_cuda_devices) == 1:
        return (visible_cuda_devices[0].logical_name,) * num_layers

    weights = [device.memory_weight for device in visible_cuda_devices]
    total_weight = sum(weights)
    cumulative_weights = []
    running_weight = 0.0
    for weight in weights:
        running_weight += weight
        cumulative_weights.append(running_weight)

    layer_devices = []
    for layer_index in range(num_layers):
        midpoint = (layer_index + 0.5) / num_layers * total_weight
        target_device_index = 0
        while (
            target_device_index < len(cumulative_weights) - 1
            and midpoint > cumulative_weights[target_device_index]
        ):
            target_device_index += 1
        layer_devices.append(visible_cuda_devices[target_device_index].logical_name)
    return tuple(layer_devices)


def _load_device_map_spec(device_map):
    if isinstance(device_map, (dict, list, tuple)):
        return device_map, "inline"
    if device_map is None:
        return AUTO_DEVICE_MAP, AUTO_DEVICE_MAP

    device_map_text = str(device_map)
    if device_map_text == AUTO_DEVICE_MAP:
        return AUTO_DEVICE_MAP, AUTO_DEVICE_MAP

    device_map_path = Path(device_map_text)
    payload = json.loads(device_map_path.read_text(encoding="utf-8"))
    return payload, str(device_map_path)


def _build_layer_devices_from_manual_spec(spec, num_layers):
    if isinstance(spec, (list, tuple)):
        layer_devices = tuple(_normalize_device_name(device) for device in spec)
        if len(layer_devices) != num_layers:
            raise ValueError(
                f"device_map 列表长度 ({len(layer_devices)}) 必须等于模型层数 ({num_layers})。"
            )
        return layer_devices, {}

    if not isinstance(spec, dict):
        raise TypeError("device_map 必须是 'auto'、JSON 文件、列表或字典。")

    raw_layers = spec.get("layers")
    if raw_layers is None:
        raise ValueError("手工 device_map 必须包含 layers 字段。")

    if isinstance(raw_layers, (list, tuple)):
        layer_devices = tuple(_normalize_device_name(device) for device in raw_layers)
    elif isinstance(raw_layers, dict):
        missing_layers = [str(index) for index in range(num_layers) if str(index) not in raw_layers]
        if missing_layers:
            raise ValueError(f"device_map.layers 缺少层: {missing_layers}")
        layer_devices = tuple(
            _normalize_device_name(raw_layers[str(index)])
            for index in range(num_layers)
        )
    else:
        raise TypeError("device_map.layers 必须是列表或字典。")

    if len(layer_devices) != num_layers:
        raise ValueError(
            f"device_map.layers 长度 ({len(layer_devices)}) 必须等于模型层数 ({num_layers})。"
        )

    module_devices = {
        str(key): _normalize_device_name(value)
        for key, value in spec.get("modules", {}).items()
    }
    for key in ("embedding", "lm_head", "final_norm"):
        if key in spec:
            module_devices[key] = _normalize_device_name(spec[key])
    return layer_devices, module_devices


def _resolve_single_plan(*, visible_cuda_devices, source):
    if visible_cuda_devices:
        primary_device = visible_cuda_devices[0].logical_name
    else:
        primary_device = "cpu"
    return DeviceMapPlan(
        mode=SINGLE_EXECUTION_MODE,
        primary_device=primary_device,
        visible_cuda_devices=tuple(visible_cuda_devices),
        layer_devices=(),
        module_devices={
            "embedding": primary_device,
            "lm_head": primary_device,
            "final_norm": primary_device,
        },
        source=source,
    )


def resolve_execution_plan(
    execution_config=None,
    *,
    num_layers=None,
    visible_cuda_devices=None,
):
    """根据执行配置生成单卡或单机多卡推理计划。"""
    config = execution_config or ExecutionConfig()
    visible_devices = (
        tuple(visible_cuda_devices)
        if visible_cuda_devices is not None
        else discover_visible_cuda_devices()
    )

    requested_mode = config.mode
    if requested_mode == AUTO_EXECUTION_MODE:
        requested_mode = MODEL_PARALLEL_EXECUTION_MODE if len(visible_devices) > 1 else SINGLE_EXECUTION_MODE

    if requested_mode == SINGLE_EXECUTION_MODE:
        return _resolve_single_plan(visible_cuda_devices=visible_devices, source="single")

    if requested_mode in {TENSOR_PARALLEL_EXECUTION_MODE, FSDP_EXECUTION_MODE}:
        raise NotImplementedError(
            f"{requested_mode} 已预留执行配置入口，但当前实现只支持 single 与 model_parallel 推理。"
        )

    if requested_mode != MODEL_PARALLEL_EXECUTION_MODE:
        raise ValueError(f"未支持的 execution mode: {requested_mode}")
    if num_layers is None:
        raise ValueError("model_parallel 执行计划需要 num_layers。")
    if len(visible_devices) < 2:
        raise ValueError(
            "model_parallel 至少需要 2 张当前进程可见的 CUDA 卡；"
            f"请检查 {CUDA_VISIBLE_DEVICES_ENV} 或改用 --execution-mode single。"
        )

    spec, source = _load_device_map_spec(config.device_map)
    if spec == AUTO_DEVICE_MAP:
        layer_devices = _allocate_layers_by_memory(num_layers, visible_devices)
        module_devices = {}
        source = AUTO_DEVICE_MAP
    else:
        layer_devices, module_devices = _build_layer_devices_from_manual_spec(spec, num_layers)

    _validate_device_names(layer_devices, visible_cuda_devices=visible_devices, allow_cpu=False)
    _validate_device_names(module_devices.values(), visible_cuda_devices=visible_devices, allow_cpu=False)

    primary_device = module_devices.get("embedding", visible_devices[0].logical_name)
    module_devices = {
        "embedding": primary_device,
        # 当前模型 tie embedding/lm_head 权重，第一版将二者固定在同一设备。
        "lm_head": primary_device,
        "final_norm": module_devices.get("final_norm", primary_device),
        **{
            key: value
            for key, value in module_devices.items()
            if key not in {"embedding", "lm_head", "final_norm"}
        },
    }
    return DeviceMapPlan(
        mode=MODEL_PARALLEL_EXECUTION_MODE,
        primary_device=primary_device,
        visible_cuda_devices=visible_devices,
        layer_devices=layer_devices,
        module_devices=module_devices,
        source=source,
    )


def apply_inference_execution_plan(model, execution_plan):
    """把模型按推理执行计划放置到设备上。"""
    plan = execution_plan or resolve_execution_plan(num_layers=len(model.layers))
    if plan.mode == SINGLE_EXECUTION_MODE:
        target_device = torch.device(plan.primary_device)
        model.to(target_device)
        GlobalConfig.device = target_device
        setattr(model, "execution_plan", plan)
        return model

    if plan.mode != MODEL_PARALLEL_EXECUTION_MODE:
        raise NotImplementedError(f"当前只支持 single/model_parallel 推理计划: {plan.mode}")
    if len(plan.layer_devices) != len(model.layers):
        raise ValueError(
            f"device_map 层数 ({len(plan.layer_devices)}) 与模型层数 ({len(model.layers)}) 不一致。"
        )

    primary_device = torch.device(plan.primary_device)
    model.token_embedding.to(primary_device)
    model._rope_caches.to(primary_device)
    for layer, device_name in zip(model.layers, plan.layer_devices):
        layer.to(torch.device(device_name))
    model.final_norm.to(torch.device(plan.module_devices.get("final_norm", plan.primary_device)))
    # lm_head 与 token_embedding 共享权重，必须固定在同一设备。
    model.lm_head.to(primary_device)
    model.lm_head.weight = model.token_embedding.weight
    GlobalConfig.device = primary_device
    setattr(model, "execution_plan", plan)
    return model


def describe_execution_plan(execution_plan):
    """生成面向日志的执行计划摘要。"""
    plan = execution_plan
    visible_devices = [
        {
            "logical": device.logical_name,
            "visible": device.visible_device,
            "memory_gib": (
                None
                if device.total_memory_bytes is None
                else round(device.total_memory_bytes / (1024 ** 3), 2)
            ),
            "name": device.name,
        }
        for device in plan.visible_cuda_devices
    ]
    lines = [
        f"execution_mode={plan.mode}",
        f"primary_device={plan.primary_device}",
        f"device_map_source={plan.source}",
        f"visible_cuda_devices={json.dumps(visible_devices, ensure_ascii=False)}",
    ]
    if plan.layer_devices:
        ranges = []
        start_index = 0
        current_device = plan.layer_devices[0]
        for index, device_name in enumerate(plan.layer_devices[1:], start=1):
            if device_name == current_device:
                continue
            ranges.append((start_index, index - 1, current_device))
            start_index = index
            current_device = device_name
        ranges.append((start_index, len(plan.layer_devices) - 1, current_device))
        for start, end, device_name in ranges:
            layer_range = f"{start}" if start == end else f"{start}-{end}"
            lines.append(f"layers {layer_range} -> {device_name}")
    for warning in plan.warnings:
        lines.append(f"warning={warning}")
    return "\n".join(lines)


def add_execution_arguments(parser: ArgumentParser):
    """给入口脚本注册执行层参数。"""
    parser.add_argument(
        "--execution-mode",
        choices=SUPPORTED_EXECUTION_MODES,
        default=SINGLE_EXECUTION_MODE,
        help="执行模式。auto 会在可见 CUDA 卡超过 1 张时选择 model_parallel。",
    )
    parser.add_argument(
        "--device-map",
        default=AUTO_DEVICE_MAP,
        help="model_parallel 的设备映射；可为 auto 或 JSON 文件路径。",
    )
    parser.add_argument(
        "--hide-device-map",
        action="store_true",
        help="不打印解析后的执行计划。",
    )


def build_execution_config(args):
    """从 argparse 结果构造执行配置。"""
    return ExecutionConfig(
        mode=args.execution_mode,
        device_map=args.device_map,
        print_device_map=not args.hide_device_map,
    )
