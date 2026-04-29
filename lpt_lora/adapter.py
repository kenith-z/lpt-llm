"""LoRA 适配器"""

import torch.nn as nn
import torch.nn.functional as F

from lpt_config import LoRAConfig
from lpt_model import ModernAttention


class LowRankLinearAdapter(nn.Module):
    """LoRA 线性层包装器。"""

    def __init__(self, source_linear, rank=2, alpha=4, dropout_p=0.05):
        super().__init__()
        # 复用原始线性层参数，作为“冻结主干”。
        self.weight = source_linear.weight
        # 现代 LLM 的投影层通常 bias=False，这里会是 None
        self.bias = source_linear.bias
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout_p)

        self.down_projection = nn.Linear(source_linear.in_features, rank, bias=False)
        self.up_projection = nn.Linear(rank, source_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.down_projection.weight, a=0)
        nn.init.zeros_(self.up_projection.weight)

    def forward(self, inputs):
        """输出 = 原始线性层输出 + LoRA 增量输出。"""
        # F.linear 可以安全地处理 self.bias 为 None 的情况
        frozen_branch = F.linear(inputs, self.weight, self.bias)
        adapter_branch = self.up_projection(self.dropout(self.down_projection(inputs))) * self.scaling
        return frozen_branch + adapter_branch


def _replace_linear_layer(parent_module, attribute_name, rank, alpha, dropout_p):
    """把模块中的指定线性层替换成 LoRA 版本。"""
    source_layer = getattr(parent_module, attribute_name)
    if not isinstance(source_layer, nn.Linear):
        raise TypeError(f"{attribute_name} 不是 nn.Linear")

    replacement = LowRankLinearAdapter(
        source_layer,
        rank=rank,
        alpha=alpha,
        dropout_p=dropout_p,
    )
    # 新适配器必须跟随源层所在设备和精度，避免混合 dtype/device 报错。
    replacement.to(device=source_layer.weight.device, dtype=source_layer.weight.dtype)
    setattr(parent_module, attribute_name, replacement)


def _iter_attention_layers(model):
    """遍历模型中的所有注意力层。"""
    for module in model.modules():
        if isinstance(module, ModernAttention):
            yield module


def attach_lora_adapters(model, config=None):
    """在模型注意力层上挂载 LoRA 适配器，并冻结非 LoRA 参数。"""
    if config is None:
        config = LoRAConfig()

    for attention_layer in _iter_attention_layers(model):
        for attribute_name in config.target_modules:
            _replace_linear_layer(
                attention_layer,
                attribute_name,
                config.rank,
                config.alpha,
                config.dropout_p,
            )

    for parameter_name, parameter in model.named_parameters():
        # 训练阶段仅更新 LoRA 新增的低秩矩阵
        parameter.requires_grad = (
            parameter_name.endswith("down_projection.weight")
            or parameter_name.endswith("up_projection.weight")
        )
