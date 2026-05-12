"""初始化 LPT v2 第 25 项 RetNetAssist 共享策略实验分支。"""

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

from lpt_config import GlobalConfig
from lpt_model import LPTV2, load_lpt_v2_checkpoint, save_lpt_v2_checkpoint


DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "text_pretrain" / "checkpoints" / "latest" / "model.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "experiments_exp25"

BRANCH_DEFINITIONS = {
    "base_continued": {
        "description": "第 25 项 global/group 共享策略基线；使用真实共享语义继续训练。",
        "overrides": {
            "retnet_parameter_sharing": "global",
            "retnet_state_sharing": "group",
            "retnet_sharing_group_size": 4,
        },
        "init_strategy": (
            "加载 text_pretrain 可匹配权重；历史 checkpoint 中重复保存的 layer adapter/core "
            "会折叠到共享参数槽。"
        ),
    },
    "exp_25_global_per_layer": {
        "description": "参数 global 共享，状态 per-layer 独立，用于隔离状态共享影响。",
        "overrides": {
            "retnet_parameter_sharing": "global",
            "retnet_state_sharing": "per_layer",
            "retnet_sharing_group_size": 4,
        },
        "init_strategy": "共享参数从 text_pretrain 映射，状态运行时按启用层独立维护。",
    },
    "exp_25_group_group": {
        "description": "参数和状态均按连续 4 层 group 共享。",
        "overrides": {
            "retnet_parameter_sharing": "group",
            "retnet_state_sharing": "group",
            "retnet_sharing_group_size": 4,
        },
        "init_strategy": "每组 RetNetAssist 参数从组内历史层权重折叠初始化，状态按同组共享。",
    },
    "exp_25_per_layer_per_layer": {
        "description": "参数和状态均 per-layer 独立，作为容量上限和成本上限对照。",
        "overrides": {
            "retnet_parameter_sharing": "per_layer",
            "retnet_state_sharing": "per_layer",
            "retnet_sharing_group_size": 4,
        },
        "init_strategy": "保留 text_pretrain 中逐层 RetNetAssist adapter/core 权重，状态按层独立。",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="初始化 LPT v2 第 25 项实验分支 checkpoint。")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT, help="已训练 text_pretrain/base checkpoint。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录。")
    parser.add_argument(
        "--branches",
        default=(
            "base_continued,"
            "exp_25_global_per_layer,"
            "exp_25_group_group,"
            "exp_25_per_layer_per_layer"
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


def _enabled_retnet_layers(config):
    if not bool(config.retnet_assist_enabled):
        return ()
    policy = str(config.retnet_assist_layers)
    if policy == "all_layers":
        return tuple(range(int(config.num_layers)))
    if policy.startswith("every_") and policy.endswith("_layers"):
        interval_text = policy.removeprefix("every_").removesuffix("_layers")
        if interval_text.isdigit():
            interval = int(interval_text)
            return tuple(range(0, int(config.num_layers), interval))
    return ()


def _sharing_group_count(config):
    enabled_layers = _enabled_retnet_layers(config)
    if not enabled_layers:
        return 0
    group_size = int(config.retnet_sharing_group_size)
    return len({int(layer_index) // group_size for layer_index in enabled_layers})


def _parameter_group_count(config):
    enabled_count = len(_enabled_retnet_layers(config))
    if enabled_count == 0:
        return 0
    if config.retnet_parameter_sharing == "global":
        return 1
    if config.retnet_parameter_sharing == "group":
        return _sharing_group_count(config)
    return enabled_count


def _state_slot_count(config):
    enabled_count = len(_enabled_retnet_layers(config))
    if enabled_count == 0:
        return 0
    if config.retnet_state_sharing == "group":
        return _sharing_group_count(config)
    return enabled_count


def _assert_base_contract(base_config):
    """第 25 项必须从 Q-only RetNetAssist 且未启用 xLSTM 的共同基座开始。"""
    if not bool(base_config.retnet_assist_enabled):
        raise ValueError("base checkpoint 未启用 RetNetAssist，不能作为第 25 项共同基座。")
    if base_config.retnet_assist_mode != "q_adapter":
        raise ValueError(f"base checkpoint 不是 q_adapter: {base_config.retnet_assist_mode}")
    if base_config.retnet_adapter_target != ("q",):
        raise ValueError(f"base checkpoint 不是 Q-only adapter: {base_config.retnet_adapter_target}")
    if base_config.retnet_k_adapter_enabled:
        raise ValueError("base checkpoint 已启用 retnet_k_adapter_enabled，会污染共享策略归因。")
    if base_config.xlstm_memory_enabled:
        raise ValueError("base checkpoint 已启用 xLSTMAssist，会污染第 25 项单项归因。")


def _build_branch_model(base_loaded, branch_name):
    base_config = base_loaded.model.config
    overrides = dict(BRANCH_DEFINITIONS[branch_name]["overrides"])
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
    _assert_base_contract(base_loaded.model.config)
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
            "experiment": "lpt_v2_exp25_retnet_sharing",
            "branch": branch_name,
            "description": BRANCH_DEFINITIONS[branch_name]["description"],
            "base_checkpoint": str(base_checkpoint),
            "checkpoint_path": str(checkpoint_path),
            "created_at_unix": time(),
            "config_overrides": overrides,
            "init_strategy": BRANCH_DEFINITIONS[branch_name]["init_strategy"],
            "retnet_parameter_sharing": branch_config.retnet_parameter_sharing,
            "retnet_state_sharing": branch_config.retnet_state_sharing,
            "retnet_sharing_group_size": int(branch_config.retnet_sharing_group_size),
            "retnet_enabled_layer_count": len(_enabled_retnet_layers(branch_config)),
            "retnet_parameter_group_count": _parameter_group_count(branch_config),
            "retnet_state_slot_count": _state_slot_count(branch_config),
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
            f"{branch_name} param_groups={report['retnet_parameter_group_count']} "
            f"state_slots={report['retnet_state_slot_count']} checkpoint={checkpoint_path}"
        )

    _write_json(output_root / "exp25_init_summary.json", {"branches": reports})
    print(f"summary={output_root / 'exp25_init_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
