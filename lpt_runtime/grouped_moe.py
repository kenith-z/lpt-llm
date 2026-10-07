"""SwiGLU-MoE 的分组执行后端。

模型参数使用 ``Linear`` 的权重布局保存：

* ``expert_w1`` / ``expert_w3``: ``[experts, intermediate, hidden]``；
* ``expert_w2``: ``[experts, hidden, intermediate]``。

CUDA 的 ``torch.nn.functional.grouped_mm`` 要求右操作数采用
``[experts, in_features, out_features]`` 布局，因此 CUDA 路径只创建
三块参数的 transpose view，不复制专家权重。CPU 路径使用独立的逐专家
参考实现，供 CPU smoke test 和没有 CUDA 的开发环境使用。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_swiglu_cpu(
    sorted_x: torch.Tensor,
    expert_w1: torch.Tensor,
    expert_w2: torch.Tensor,
    expert_w3: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """在 CPU 上按 ``offsets`` 执行分组 SwiGLU。

    ``sorted_x`` 已经按 expert id 排序；``offsets[i]`` 是第 ``i`` 个
    expert 的结束位置。这里每个 expert 只调用一次矩阵乘法，不使用
    ``nonzero`` 扫描，也不依赖专家模块对象，因而与打包参数布局一致。
    切片和 ``F.linear`` 都保持 autograd 图，未被选中的专家仍会得到
    形状完整的零梯度。
    """
    if sorted_x.ndim != 2:
        raise ValueError("sorted_x 必须是二维 [tokens, hidden] 张量。")
    if expert_w1.ndim != 3 or expert_w2.ndim != 3 or expert_w3.ndim != 3:
        raise ValueError("expert_w1/w2/w3 必须是三维打包权重。")
    if offsets.ndim != 1 or offsets.numel() != expert_w1.size(0):
        raise ValueError("offsets 长度必须等于专家数。")

    # offsets 只用于 Python 切片边界；该路径限定为 CPU，因此不会同步
    # 正式 CUDA 训练的执行流。
    ends = offsets.to(device="cpu", dtype=torch.int64).tolist()
    start = 0
    chunks = []
    for expert_index, end_value in enumerate(ends):
        end = int(end_value)
        if end < start or end > sorted_x.size(0):
            raise ValueError("offsets 必须单调递增且不超过路由 token 数。")
        if end == start:
            continue
        expert_x = sorted_x.narrow(0, start, end - start)
        hidden_a = F.linear(expert_x, expert_w1[expert_index])
        hidden_b = F.linear(expert_x, expert_w3[expert_index])
        gated = F.silu(hidden_a) * hidden_b
        chunks.append(F.linear(gated, expert_w2[expert_index]))
        start = end

    if start != sorted_x.size(0):
        raise ValueError("offsets 最后一个值必须等于路由 token 数。")
    if chunks:
        return torch.cat(chunks, dim=0)
    return sorted_x.new_empty((0, expert_w2.size(1)))


def grouped_swiglu_cuda(
    sorted_x: torch.Tensor,
    expert_w1: torch.Tensor,
    expert_w2: torch.Tensor,
    expert_w3: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """使用当前 PyTorch 公开 grouped GEMM 执行三次 SwiGLU 投影。

    当前运行时约束是 CUDA + bfloat16（由 PyTorch grouped_mm kernel 定义）。
    调用方可以传入 float32/float16 输入；此处统一转为 bfloat16，返回值
    再由模型恢复到输入 dtype。专家权重的 transpose 是 view，不产生
    ``[experts, in, out]`` 的额外参数副本。
    """
    if not sorted_x.is_cuda:
        raise ValueError("grouped_swiglu_cuda 只接受 CUDA 张量。")
    grouped_x = sorted_x if sorted_x.dtype == torch.bfloat16 else sorted_x.to(torch.bfloat16)
    w1 = expert_w1.to(device=grouped_x.device, dtype=torch.bfloat16).transpose(1, 2)
    w2 = expert_w2.to(device=grouped_x.device, dtype=torch.bfloat16).transpose(1, 2)
    w3 = expert_w3.to(device=grouped_x.device, dtype=torch.bfloat16).transpose(1, 2)

    hidden_a = F.grouped_mm(grouped_x, w1, offs=offsets)
    hidden_b = F.grouped_mm(grouped_x, w3, offs=offsets)
    gated = F.silu(hidden_a) * hidden_b
    return F.grouped_mm(gated, w2, offs=offsets)


__all__ = ["grouped_swiglu_cpu", "grouped_swiglu_cuda"]
