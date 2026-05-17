"""LPT v2 评测工具函数。"""

from __future__ import annotations

from pathlib import Path
import json
import math
import random

import torch
import torch.nn.functional as F


def resolve_eval_device(device="auto"):
    """解析评测设备；auto 优先使用当前可见 cuda:0。"""
    device_text = str(device)
    if device_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def resolve_eval_dtype(dtype="auto", *, device=None):
    """解析评测 dtype；CUDA auto 默认 fp16，CPU auto 默认 fp32。"""
    dtype_text = str(dtype)
    if dtype_text == "auto":
        if device is not None and torch.device(device).type == "cuda":
            return torch.float16
        return torch.float32
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_text not in dtype_map:
        raise ValueError(f"未知 dtype: {dtype}")
    return dtype_map[dtype_text]


def dtype_name(dtype):
    """把 torch.dtype 转成短字符串。"""
    return str(dtype).removeprefix("torch.")


def set_eval_seed(seed):
    """设置评测随机种子，保证 smoke 结果可复现。"""
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_deterministic_input(vocabulary_size, batch_size, sequence_length, *, offset=1, device="cpu"):
    """生成不含 0 的确定性 token-id 输入，避免与 padding id 混淆。"""
    values = torch.arange(offset, offset + int(batch_size) * int(sequence_length), device=device)
    values = values.remainder(max(1, int(vocabulary_size) - 1)).add(1)
    return values.view(int(batch_size), int(sequence_length)).long()


def next_token_loss(logits, input_ids):
    """计算简单 next-token 交叉熵和 PPL。"""
    if logits.size(1) < 2:
        return None, None
    shifted_logits = logits[:, :-1].float().reshape(-1, logits.size(-1))
    shifted_labels = input_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(shifted_logits, shifted_labels)
    loss_value = float(loss.detach().cpu())
    ppl = float(math.exp(min(loss_value, 20.0)))
    return loss_value, ppl


def write_json_report(path, payload):
    """以 UTF-8 写入 JSON 报告。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path


def write_text_report(path, text):
    """以 UTF-8 写入文本/Markdown 报告。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(text, encoding="utf-8")
    return target_path
