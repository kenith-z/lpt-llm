"""LPT v2 LoRA 适配器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from lpt_config import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGET_MODULES,
)
from lpt_model import LocalAttentionMixerV2
from lpt_runtime.files import atomic_torch_save


LORA_ADAPTER_FORMAT = "lpt_v2_lora_adapter"
LORA_ADAPTER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA 注入配置。"""

    rank: int = DEFAULT_LORA_RANK
    alpha: float = DEFAULT_LORA_ALPHA
    dropout_p: float = DEFAULT_LORA_DROPOUT
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES

    def __post_init__(self):
        """校验 LoRA 超参，并把 target_modules 固定为 tuple。"""
        if int(self.rank) <= 0:
            raise ValueError("LoRA rank 必须为正整数。")
        if float(self.alpha) <= 0:
            raise ValueError("LoRA alpha 必须为正数。")
        if not 0 <= float(self.dropout_p) < 1:
            raise ValueError("LoRA dropout_p 必须位于 [0, 1) 区间。")
        normalized_targets = tuple(str(value) for value in self.target_modules)
        if not normalized_targets:
            raise ValueError("LoRA target_modules 不能为空。")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout_p", float(self.dropout_p))
        object.__setattr__(self, "target_modules", normalized_targets)

    def to_dict(self):
        """导出可序列化配置。"""
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """从 checkpoint 载荷恢复配置。"""
        if payload is None:
            return cls()
        normalized = dict(payload)
        if "target_modules" in normalized and isinstance(normalized["target_modules"], list):
            normalized["target_modules"] = tuple(normalized["target_modules"])
        return cls(**normalized)


class LowRankLinearAdapter(nn.Module):
    """冻结原线性层并追加低秩可训练分支。"""

    def __init__(
        self,
        source_linear,
        *,
        rank=DEFAULT_LORA_RANK,
        alpha=DEFAULT_LORA_ALPHA,
        dropout_p=DEFAULT_LORA_DROPOUT,
    ):
        """包装原线性层；base_layer 冻结，只训练低秩分支。"""
        super().__init__()
        if not isinstance(source_linear, nn.Linear):
            raise TypeError("source_linear 必须是 nn.Linear。")
        self.base_layer = source_linear
        self.scaling = float(alpha) / int(rank)
        self.dropout = nn.Dropout(float(dropout_p))
        self.down_projection = nn.Linear(source_linear.in_features, int(rank), bias=False)
        self.up_projection = nn.Linear(int(rank), source_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down_projection.weight, a=0)
        nn.init.zeros_(self.up_projection.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, inputs):
        """输出主干分支与 LoRA 增量分支之和。"""
        base_output = self.base_layer(inputs)
        adapter_output = self.up_projection(self.dropout(self.down_projection(inputs))) * self.scaling
        return base_output + adapter_output


def _replace_linear_layer(parent_module, attribute_name, config):
    """把父模块中的指定 Linear 替换为 LowRankLinearAdapter。"""
    source_layer = getattr(parent_module, attribute_name)
    if isinstance(source_layer, LowRankLinearAdapter):
        return
    if not isinstance(source_layer, nn.Linear):
        raise TypeError(f"{attribute_name} 不是 nn.Linear，不能注入 LoRA。")
    replacement = LowRankLinearAdapter(
        source_layer,
        rank=config.rank,
        alpha=config.alpha,
        dropout_p=config.dropout_p,
    )
    replacement.to(device=source_layer.weight.device, dtype=source_layer.weight.dtype)
    setattr(parent_module, attribute_name, replacement)


def _iter_attention_mixers(model):
    """遍历 LPTV2 中可注入 LoRA 的 attention mixer。"""
    for module in model.modules():
        if isinstance(module, LocalAttentionMixerV2):
            yield module


def attach_lora_adapters(model, config=None):
    """在 LPT v2 attention 投影层上注入 LoRA，并冻结非 LoRA 参数。"""
    lora_config = config if isinstance(config, LoRAConfig) else LoRAConfig.from_dict(config)
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced_count = 0
    for attention_mixer in _iter_attention_mixers(model):
        for attribute_name in lora_config.target_modules:
            _replace_linear_layer(attention_mixer, attribute_name, lora_config)
            replaced_count += 1

    for parameter_name, parameter in model.named_parameters():
        # 只解冻 LoRA 新增的上下投影权重，冻结基座模型，保证 adapter-only checkpoint 语义清晰。
        parameter.requires_grad = (
            parameter_name.endswith("down_projection.weight")
            or parameter_name.endswith("up_projection.weight")
        )
    setattr(model, "lora_config", lora_config)
    setattr(model, "lora_adapter_count", replaced_count)
    return model


def collect_lora_adapter_state(model):
    """收集 LoRA 新增权重，避免把冻结基座写入 adapter checkpoint。"""
    adapter_state = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("down_projection.weight") or name.endswith("up_projection.weight"):
            adapter_state[name] = tensor.detach().cpu()
    if not adapter_state:
        raise ValueError("模型中没有可保存的 LoRA adapter 权重。")
    return adapter_state


def save_lora_adapter_state(model, path, *, config=None, extra_metadata=None):
    """保存 adapter-only checkpoint。"""
    adapter_config = config or getattr(model, "lora_config", LoRAConfig())
    if not isinstance(adapter_config, LoRAConfig):
        adapter_config = LoRAConfig.from_dict(adapter_config)
    payload = {
        "adapter_format": LORA_ADAPTER_FORMAT,
        "adapter_schema_version": LORA_ADAPTER_SCHEMA_VERSION,
        "adapter_config": adapter_config.to_dict(),
        "adapter_state_dict": collect_lora_adapter_state(model),
        "metadata": dict(extra_metadata or {}),
    }
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(payload, target_path)
    return target_path


def _load_lora_adapter_payload(path):
    """读取并校验 adapter-only checkpoint 外层 schema。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("adapter_format") != LORA_ADAPTER_FORMAT:
        raise ValueError("adapter_format 不是 lpt_v2_lora_adapter。")
    if payload.get("adapter_schema_version") != LORA_ADAPTER_SCHEMA_VERSION:
        raise ValueError(
            "不支持的 adapter_schema_version: "
            f"{payload.get('adapter_schema_version')}"
        )
    return payload


def load_lora_adapter_config(path):
    """读取 adapter checkpoint 中保存的 LoRA 配置。"""
    payload = _load_lora_adapter_payload(path)
    return LoRAConfig.from_dict(payload.get("adapter_config"))


def load_lora_adapter_state(model, path, *, strict=True):
    """加载 adapter-only checkpoint 到已注入 LoRA 的模型。"""
    payload = _load_lora_adapter_payload(path)
    adapter_state = payload.get("adapter_state_dict")
    if not isinstance(adapter_state, dict) or not adapter_state:
        raise ValueError("adapter checkpoint 缺少 adapter_state_dict。")
    incompatible = model.load_state_dict(adapter_state, strict=False)
    unexpected = tuple(incompatible.unexpected_keys)
    missing_lora_keys = tuple(
        key
        for key in incompatible.missing_keys
        if key.endswith("down_projection.weight") or key.endswith("up_projection.weight")
    )
    # 非 LoRA 基座权重缺失是预期现象；strict 只约束 adapter 权重和 unexpected key。
    if strict and (missing_lora_keys or unexpected):
        raise ValueError(
            "LoRA adapter 权重键不匹配: "
            f"missing={missing_lora_keys}, unexpected={unexpected}"
        )
    return payload
