"""初始化 LPT v2 第 26 项 RetNetAssist 启用层与 rank 实验分支。"""

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

from lpt_config import GlobalConfig, is_retnet_assist_enabled_for_layer
from lpt_model import LPTV2, load_lpt_v2_checkpoint, save_lpt_v2_checkpoint


DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "text_pretrain" / "checkpoints" / "latest" / "model.pt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "experiments_exp26"

BRANCH_DEFINITIONS = {
    "base_continued": {
        "description": "第 26 项 all_layers/rank16 对照；保持 text_pretrain 基座 RetNetAssist 配置继续训练。",
        "overrides": {},
        "changed_dimension": "none",
        "init_strategy": "完全沿用 text_pretrain 可匹配权重，作为 all_layers/rank16 baseline。",
    },
    "exp_26_retnet_every_2_layers": {
        "description": "RetNetAssist 每 2 层启用一次，只改变启用层密度。",
        "overrides": {
            "retnet_assist_layers": "every_2_layers",
            "retnet_assist_selected_layers": (),
        },
        "changed_dimension": "retnet_assist_layers",
        "init_strategy": "加载共同基座中可匹配的 RetNetAssist 权重，未启用层不挂载 RetNetAssist 参数。",
    },
    "exp_26_retnet_every_4_layers": {
        "description": "RetNetAssist 每 4 层启用一次，只改变启用层密度。",
        "overrides": {
            "retnet_assist_layers": "every_4_layers",
            "retnet_assist_selected_layers": (),
        },
        "changed_dimension": "retnet_assist_layers",
        "init_strategy": "加载共同基座中可匹配的 RetNetAssist 权重，未启用层不挂载 RetNetAssist 参数。",
    },
    "exp_26_retnet_selected_offset_layers": {
        "description": "RetNetAssist 使用 selected_layers 启用 2/6/10/14/18/22 层，与 every_4_layers 同层数但相位偏移。",
        "overrides": {
            "retnet_assist_layers": "selected_layers",
            "retnet_assist_selected_layers": (2, 6, 10, 14, 18, 22),
        },
        "changed_dimension": "retnet_assist_layers",
        "init_strategy": "加载共同基座中可匹配的 RetNetAssist 权重，用于隔离同等启用层数量下的位置相位影响。",
    },
    "exp_26_retnet_rank32": {
        "description": "RetNetAssist adapter rank 从 16 提升到 32，只改变 adapter rank。",
        "overrides": {
            "retnet_adapter_rank": 32,
        },
        "changed_dimension": "retnet_adapter_rank",
        "init_strategy": "RetNet core、Attention、MoE 和可匹配权重从基座加载；rank 形状变化的 adapter projection 使用新初始化。",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="初始化 LPT v2 第 26 项实验分支 checkpoint。")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT, help="已训练 text_pretrain/base checkpoint。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录。")
    parser.add_argument(
        "--branches",
        default=(
            "base_continued,"
            "exp_26_retnet_every_2_layers,"
            "exp_26_retnet_every_4_layers,"
            "exp_26_retnet_selected_offset_layers,"
            "exp_26_retnet_rank32"
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
    return tuple(
        layer_index
        for layer_index in range(int(config.num_layers))
        if is_retnet_assist_enabled_for_layer(config, layer_index)
    )


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
    """第 26 项必须从 Q-only RetNetAssist 且未启用 xLSTM 的共同基座开始。"""
    if not bool(base_config.retnet_assist_enabled):
        raise ValueError("base checkpoint 未启用 RetNetAssist，不能作为第 26 项共同基座。")
    if base_config.retnet_assist_mode != "q_adapter":
        raise ValueError(f"base checkpoint 不是 q_adapter: {base_config.retnet_assist_mode}")
    if base_config.retnet_adapter_target != ("q",):
        raise ValueError(f"base checkpoint 不是 Q-only adapter: {base_config.retnet_adapter_target}")
    if base_config.retnet_k_adapter_enabled:
        raise ValueError("base checkpoint 已启用 retnet_k_adapter_enabled，会污染第 26 项单项归因。")
    if base_config.xlstm_memory_enabled:
        raise ValueError("base checkpoint 已启用 xLSTMAssist，会污染第 26 项单项归因。")
    if base_config.retnet_assist_layers != "all_layers":
        raise ValueError(f"第 26 项共同基线应为 all_layers，实际为: {base_config.retnet_assist_layers}")
    if int(base_config.retnet_adapter_rank) != 16:
        raise ValueError(f"第 26 项共同基线应为 rank16，实际为: {base_config.retnet_adapter_rank}")


def _filter_compatible_state_dict(source_state_dict, target_state_dict):
    """仅加载目标模型中存在且 shape 一致的权重，rank 变化的 adapter 权重保留新初始化。"""
    compatible = {}
    skipped_mismatch = []
    for key, target_tensor in target_state_dict.items():
        source_tensor = source_state_dict.get(key)
        if source_tensor is None:
            continue
        if torch.is_tensor(source_tensor) and torch.is_tensor(target_tensor):
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                skipped_mismatch.append(
                    {
                        "key": key,
                        "source_shape": list(source_tensor.shape),
                        "target_shape": list(target_tensor.shape),
                    }
                )
                continue
        compatible[key] = source_tensor
    unused_source_keys = sorted(set(source_state_dict) - set(target_state_dict))
    return compatible, skipped_mismatch, unused_source_keys


def _build_branch_model(base_loaded, branch_name):
    base_config = base_loaded.model.config
    overrides = dict(BRANCH_DEFINITIONS[branch_name]["overrides"])
    branch_config = base_config.with_overrides(**overrides)
    model = LPTV2(base_loaded.model.vocabulary_size, branch_config)
    compatible_state, skipped_mismatch, unused_source_keys = _filter_compatible_state_dict(
        base_loaded.checkpoint["model_state_dict"],
        model.state_dict(),
    )
    incompatible = model.load_state_dict(compatible_state, strict=False)
    return (
        model,
        branch_config,
        overrides,
        tuple(incompatible.missing_keys),
        tuple(incompatible.unexpected_keys),
        skipped_mismatch,
        tuple(unused_source_keys),
    )


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
        (
            model,
            branch_config,
            overrides,
            missing_keys,
            unexpected_keys,
            skipped_mismatch,
            unused_source_keys,
        ) = _build_branch_model(base_loaded, branch_name)
        branch_root = output_root / branch_name
        init_root = branch_root / "init"
        checkpoint_path = init_root / "model.pt"
        if checkpoint_path.exists():
            raise FileExistsError(f"分支初始化 checkpoint 已存在，避免覆盖: {checkpoint_path}")

        enabled_layers = _enabled_retnet_layers(branch_config)
        report = {
            "experiment": "lpt_v2_exp26_retnet_layers_rank",
            "branch": branch_name,
            "description": BRANCH_DEFINITIONS[branch_name]["description"],
            "changed_dimension": BRANCH_DEFINITIONS[branch_name]["changed_dimension"],
            "base_checkpoint": str(base_checkpoint),
            "checkpoint_path": str(checkpoint_path),
            "created_at_unix": time(),
            "config_overrides": overrides,
            "init_strategy": BRANCH_DEFINITIONS[branch_name]["init_strategy"],
            "retnet_assist_layers": branch_config.retnet_assist_layers,
            "retnet_assist_selected_layers": list(branch_config.retnet_assist_selected_layers),
            "retnet_adapter_rank": int(branch_config.retnet_adapter_rank),
            "retnet_parameter_sharing": branch_config.retnet_parameter_sharing,
            "retnet_state_sharing": branch_config.retnet_state_sharing,
            "retnet_sharing_group_size": int(branch_config.retnet_sharing_group_size),
            "retnet_enabled_layers": list(enabled_layers),
            "retnet_enabled_layer_count": len(enabled_layers),
            "retnet_parameter_group_count": _parameter_group_count(branch_config),
            "retnet_state_slot_count": _state_slot_count(branch_config),
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "skipped_shape_mismatch_keys": skipped_mismatch,
            "unused_source_key_count": len(unused_source_keys),
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
                "skipped_shape_mismatch_keys": skipped_mismatch,
                "unused_source_key_count": len(unused_source_keys),
            },
        )
        branch_config.save_json(branch_root / "config" / "model_config.json")
        _write_json(init_root / "init_report.json", report)
        reports.append(report)
        print(
            "branch_initialized="
            f"{branch_name} layers={report['retnet_assist_layers']} "
            f"rank={report['retnet_adapter_rank']} "
            f"enabled_layers={report['retnet_enabled_layer_count']} "
            f"checkpoint={checkpoint_path}"
        )

    _write_json(output_root / "exp26_init_summary.json", {"branches": reports})
    print(f"summary={output_root / 'exp26_init_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
