"""初始化 LPT v2 第 22 项 Memory Gate 单项实验分支。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_model import LPTV2, load_lpt_v2_checkpoint, save_lpt_v2_checkpoint
from lpt_config import GlobalConfig


DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "text_pretrain" / "checkpoints" / "latest" / "model.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "experiments_exp22"

BRANCH_DEFINITIONS = {
    "base_continued": {
        "description": "同等预算继续训练对照分支，不改变模型结构。",
        "overrides": {},
        "init_strategy": "完全沿用 base checkpoint 配置和权重。",
    },
    "exp_22_xlstm_no_gate": {
        "description": "xLSTMAssist 无 gate 对照分支，用于区分 xLSTM 本身收益。",
        "overrides": {
            "xlstm_memory_enabled": True,
            "xlstm_memory_layers": "all_layers",
            "moe_router_input_mode": "memory_augmented_input",
            "xlstm_memory_gate_enabled": False,
        },
        "init_strategy": "加载 base 可匹配权重；xLSTM 参数沿用 base checkpoint 中已存在的初始化权重。",
    },
    "exp_22_xlstm_memory_gate": {
        "description": "第 22 项 Memory Gate 单项实验分支。",
        "overrides": {
            "xlstm_memory_enabled": True,
            "xlstm_memory_layers": "all_layers",
            "moe_router_input_mode": "memory_augmented_input",
            "xlstm_memory_gate_enabled": True,
            "xlstm_memory_gate_mode": "input_conditioned_eval",
        },
        "init_strategy": "加载 base 可匹配权重；新增 memory_gate 权重置零、bias 置为 gate_bias，让 gate 初始接近常开。",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="初始化 LPT v2 第 22 项实验分支 checkpoint。")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT, help="已训练 text_pretrain/base checkpoint。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录。")
    parser.add_argument(
        "--branches",
        default="base_continued,exp_22_xlstm_no_gate,exp_22_xlstm_memory_gate",
        help="逗号分隔分支名。",
    )
    parser.add_argument("--gate-bias", type=float, default=2.0, help="Memory Gate 初始 bias。")
    return parser


def _parse_branches(raw_value):
    branches = tuple(value.strip() for value in str(raw_value).split(",") if value.strip())
    unknown = sorted(set(branches) - set(BRANCH_DEFINITIONS))
    if unknown:
        raise ValueError(f"未知实验分支: {unknown}")
    return branches


def _initialize_memory_gate(model, *, gate_bias):
    initialized = []
    with torch.no_grad():
        for module_name, module in model.named_modules():
            memory_gate = getattr(module, "memory_gate", None)
            if memory_gate is None:
                continue
            memory_gate.weight.zero_()
            memory_gate.bias.fill_(float(gate_bias))
            initialized.append(f"{module_name}.memory_gate")
    return initialized


def _write_json(path, payload):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_state_dict_dtype(state_dict):
    for tensor in state_dict.values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return GlobalConfig.parameter_dtype


def _build_branch_model(base_loaded, branch_name, *, gate_bias):
    definition = BRANCH_DEFINITIONS[branch_name]
    base_config = base_loaded.model.config
    branch_config = base_config.with_overrides(**definition["overrides"])
    model = LPTV2(base_loaded.model.vocabulary_size, branch_config)
    incompatible = model.load_state_dict(base_loaded.checkpoint["model_state_dict"], strict=False)
    initialized_gate_modules = []
    if bool(branch_config.xlstm_memory_gate_enabled):
        initialized_gate_modules = _initialize_memory_gate(model, gate_bias=gate_bias)
    return model, branch_config, tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys), initialized_gate_modules


def main(argv=None):
    args = build_parser().parse_args(argv)
    branches = _parse_branches(args.branches)
    base_checkpoint = args.base_checkpoint.resolve()
    output_root = args.output_root.resolve()
    if not base_checkpoint.exists():
        raise FileNotFoundError(f"base checkpoint 不存在: {base_checkpoint}")

    base_loaded = load_lpt_v2_checkpoint(base_checkpoint, map_location="cpu", strict=True)
    checkpoint_dtype = _infer_state_dict_dtype(base_loaded.checkpoint["model_state_dict"])
    GlobalConfig.parameter_dtype = checkpoint_dtype
    reports = []
    for branch_name in branches:
        model, branch_config, missing_keys, unexpected_keys, initialized_gate_modules = _build_branch_model(
            base_loaded,
            branch_name,
            gate_bias=args.gate_bias,
        )
        branch_root = output_root / branch_name
        init_root = branch_root / "init"
        checkpoint_path = init_root / "model.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"分支初始化 checkpoint 已存在，避免覆盖: {checkpoint_path}")

        report = {
            "experiment": "lpt_v2_exp22_memory_gate",
            "branch": branch_name,
            "description": BRANCH_DEFINITIONS[branch_name]["description"],
            "base_checkpoint": str(base_checkpoint),
            "checkpoint_path": str(checkpoint_path),
            "created_at_unix": time(),
            "config_overrides": BRANCH_DEFINITIONS[branch_name]["overrides"],
            "init_strategy": BRANCH_DEFINITIONS[branch_name]["init_strategy"],
            "gate_bias": float(args.gate_bias) if initialized_gate_modules else None,
            "initialized_gate_modules": initialized_gate_modules,
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "vocabulary_size": int(model.vocabulary_size),
            "checkpoint_dtype": str(checkpoint_dtype).removeprefix("torch."),
            "base_global_step": base_loaded.checkpoint.get("runtime_metadata", {})
            .get("extra", {})
            .get("global_step"),
            "base_tokens_seen": base_loaded.checkpoint.get("runtime_metadata", {})
            .get("extra", {})
            .get("tokens_seen"),
        }
        save_lpt_v2_checkpoint(
            model,
            checkpoint_path,
            extra_metadata={
                "experiment": report["experiment"],
                "branch": branch_name,
                "base_checkpoint": str(base_checkpoint),
                "config_overrides": dict(BRANCH_DEFINITIONS[branch_name]["overrides"]),
                "missing_keys": list(missing_keys),
                "unexpected_keys": list(unexpected_keys),
                "initialized_gate_modules": initialized_gate_modules,
            },
        )
        branch_config.save_json(branch_root / "config" / "model_config.json")
        _write_json(init_root / "init_report.json", report)
        reports.append(report)
        print(f"branch_initialized={branch_name} checkpoint={checkpoint_path}")

    _write_json(output_root / "exp22_init_summary.json", {"branches": reports})
    print(f"summary={output_root / 'exp22_init_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
