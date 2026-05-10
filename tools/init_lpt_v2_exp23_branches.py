"""初始化 LPT v2 第 23 项 xLSTMAssist 层粒度实验分支。"""

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

from lpt_config import GlobalConfig, count_xlstm_memory_enabled_layers
from lpt_model import LPTV2, load_lpt_v2_checkpoint, save_lpt_v2_checkpoint


DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "text_pretrain" / "checkpoints" / "latest" / "model.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "experiments_exp23"

BRANCH_DEFINITIONS = {
    "base_continued": {
        "description": "同等预算继续训练对照分支，不改变模型结构。",
        "overrides": {},
        "init_strategy": "完全沿用 base checkpoint 配置和权重。",
    },
    "exp_23_xlstm_all_layers": {
        "description": "xLSTMAssist 全层启用对照分支。",
        "overrides": {
            "xlstm_memory_enabled": True,
            "xlstm_memory_layers": "all_layers",
            "xlstm_memory_selected_layers": (),
            "moe_router_input_mode": "memory_augmented_input",
            "xlstm_memory_gate_enabled": False,
        },
        "init_strategy": "加载 base 可匹配权重；xLSTM 参数沿用 base checkpoint 中已存在的初始化权重。",
    },
    "exp_23_xlstm_every_2_layers": {
        "description": "xLSTMAssist 每 2 层启用一次，用于评估半频启用成本和质量。",
        "overrides": {
            "xlstm_memory_enabled": True,
            "xlstm_memory_layers": "every_2_layers",
            "xlstm_memory_selected_layers": (),
            "moe_router_input_mode": "memory_augmented_input",
            "xlstm_memory_gate_enabled": False,
        },
        "init_strategy": "加载 base 可匹配权重；仅偶数层实际更新 xLSTM 状态并注入 FFN 输入。",
    },
    "exp_23_xlstm_every_4_layers": {
        "description": "xLSTMAssist 每 4 层启用一次，用于评估低频启用下限。",
        "overrides": {
            "xlstm_memory_enabled": True,
            "xlstm_memory_layers": "every_4_layers",
            "xlstm_memory_selected_layers": (),
            "moe_router_input_mode": "memory_augmented_input",
            "xlstm_memory_gate_enabled": False,
        },
        "init_strategy": "加载 base 可匹配权重；仅 4 的倍数层实际更新 xLSTM 状态并注入 FFN 输入。",
    },
    "exp_23_xlstm_selected_late_layers": {
        "description": "xLSTMAssist 仅后 1/4 层启用，用于评估后段记忆注入。",
        "overrides": {
            "xlstm_memory_enabled": True,
            "xlstm_memory_layers": "selected_layers",
            "moe_router_input_mode": "memory_augmented_input",
            "xlstm_memory_gate_enabled": False,
        },
        "selected_layer_strategy": "last_quarter",
        "init_strategy": "加载 base 可匹配权重；按 base 层数自动选择后 1/4 层启用。",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="初始化 LPT v2 第 23 项实验分支 checkpoint。")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT, help="已训练 text_pretrain/base checkpoint。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录。")
    parser.add_argument(
        "--branches",
        default=(
            "base_continued,"
            "exp_23_xlstm_all_layers,"
            "exp_23_xlstm_every_2_layers,"
            "exp_23_xlstm_every_4_layers,"
            "exp_23_xlstm_selected_late_layers"
        ),
        help="逗号分隔分支名。",
    )
    return parser


def _parse_branches(raw_value):
    branches = tuple(value.strip() for value in str(raw_value).split(",") if value.strip())
    unknown = sorted(set(branches) - set(BRANCH_DEFINITIONS))
    if unknown:
        raise ValueError(f"未知实验分支: {unknown}")
    return branches


def _write_json(path, payload):
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_state_dict_dtype(state_dict):
    for tensor in state_dict.values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return GlobalConfig.parameter_dtype


def _last_quarter_layers(config):
    num_layers = int(config.num_layers)
    selected_count = max(1, num_layers // 4)
    start_layer = num_layers - selected_count
    return tuple(range(start_layer, num_layers))


def _resolve_branch_overrides(base_config, branch_name):
    definition = BRANCH_DEFINITIONS[branch_name]
    overrides = dict(definition["overrides"])
    if definition.get("selected_layer_strategy") == "last_quarter":
        overrides["xlstm_memory_selected_layers"] = _last_quarter_layers(base_config)
    return overrides


def _build_branch_model(base_loaded, branch_name):
    base_config = base_loaded.model.config
    overrides = _resolve_branch_overrides(base_config, branch_name)
    branch_config = base_config.with_overrides(**overrides)
    model = LPTV2(base_loaded.model.vocabulary_size, branch_config)
    incompatible = model.load_state_dict(base_loaded.checkpoint["model_state_dict"], strict=False)
    return model, branch_config, overrides, tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys)


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
        model, branch_config, overrides, missing_keys, unexpected_keys = _build_branch_model(base_loaded, branch_name)
        branch_root = output_root / branch_name
        init_root = branch_root / "init"
        checkpoint_path = init_root / "model.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"分支初始化 checkpoint 已存在，避免覆盖: {checkpoint_path}")

        report = {
            "experiment": "lpt_v2_exp23_xlstm_granularity",
            "branch": branch_name,
            "description": BRANCH_DEFINITIONS[branch_name]["description"],
            "base_checkpoint": str(base_checkpoint),
            "checkpoint_path": str(checkpoint_path),
            "created_at_unix": time(),
            "config_overrides": overrides,
            "init_strategy": BRANCH_DEFINITIONS[branch_name]["init_strategy"],
            "xlstm_enabled_layer_count": count_xlstm_memory_enabled_layers(branch_config),
            "xlstm_memory_layers": branch_config.xlstm_memory_layers,
            "xlstm_memory_selected_layers": list(branch_config.xlstm_memory_selected_layers),
            "xlstm_memory_gate_enabled": bool(branch_config.xlstm_memory_gate_enabled),
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
                "config_overrides": dict(overrides),
                "missing_keys": list(missing_keys),
                "unexpected_keys": list(unexpected_keys),
            },
        )
        branch_config.save_json(branch_root / "config" / "model_config.json")
        _write_json(init_root / "init_report.json", report)
        reports.append(report)
        print(
            "branch_initialized="
            f"{branch_name} enabled_xlstm_layers={report['xlstm_enabled_layer_count']} checkpoint={checkpoint_path}"
        )

    _write_json(output_root / "exp23_init_summary.json", {"branches": reports})
    print(f"summary={output_root / 'exp23_init_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
