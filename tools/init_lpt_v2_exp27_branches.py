"""初始化 LPT v2 第 27 项 RetNetContextAdapter 实验分支。"""

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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lpt_v2" / "experiments_exp27"

BRANCH_DEFINITIONS = {
    "base_continued": {
        "description": "第 27 项 Q-only RetNetAssist 对照；完全沿用 text_pretrain 基座配置继续训练。",
        "overrides": {},
        "changed_dimension": "none",
        "init_strategy": "完全沿用 base checkpoint 配置和权重，作为无 ContextAdapter baseline。",
    },
    "exp_27_context_adapter": {
        "description": "启用 RetNetContextAdapter，把 RetNet summary 低秩注入 Attention 输出残差。",
        "overrides": {
            "retnet_context_adapter_enabled": True,
            "retnet_context_adapter_alpha": 1e-4,
        },
        "changed_dimension": "retnet_context_adapter",
        "init_strategy": (
            "加载 base 可匹配权重；新增 context adapter down/up projection 与 "
            "FP32 alpha_context，按小 scale 初始化。"
        ),
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="初始化 LPT v2 第 27 项实验分支 checkpoint。")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT, help="已训练 text_pretrain/base checkpoint。")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="实验输出根目录。")
    parser.add_argument(
        "--branches",
        default="base_continued,exp_27_context_adapter",
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
    """第 27 项必须从 Q-only RetNetAssist 且未启用 xLSTM/ContextAdapter 的共同基座开始。"""
    if not bool(base_config.retnet_assist_enabled):
        raise ValueError("base checkpoint 未启用 RetNetAssist，不能作为第 27 项共同基座。")
    if base_config.retnet_assist_mode != "q_adapter":
        raise ValueError(f"base checkpoint 不是 q_adapter: {base_config.retnet_assist_mode}")
    if base_config.retnet_adapter_target != ("q",):
        raise ValueError(f"base checkpoint 不是 Q-only adapter: {base_config.retnet_adapter_target}")
    if base_config.retnet_k_adapter_enabled:
        raise ValueError("base checkpoint 已启用 retnet_k_adapter_enabled，会污染第 27 项单项归因。")
    if base_config.retnet_context_adapter_enabled:
        raise ValueError("base checkpoint 已启用 RetNetContextAdapter，不能作为第 27 项对照基座。")
    if base_config.xlstm_memory_enabled:
        raise ValueError("base checkpoint 已启用 xLSTMAssist，会污染第 27 项单项归因。")


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

        enabled_layers = _enabled_retnet_layers(branch_config)
        report = {
            "experiment": "lpt_v2_exp27_context_adapter",
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
            "retnet_context_adapter_enabled": bool(branch_config.retnet_context_adapter_enabled),
            "retnet_context_adapter_alpha": float(branch_config.retnet_context_adapter_alpha),
            "retnet_enabled_layers": list(enabled_layers),
            "retnet_enabled_layer_count": len(enabled_layers),
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
            f"{branch_name} context_adapter={report['retnet_context_adapter_enabled']} "
            f"alpha={report['retnet_context_adapter_alpha']} checkpoint={checkpoint_path}"
        )

    _write_json(output_root / "exp27_init_summary.json", {"branches": reports})
    print(f"summary={output_root / 'exp27_init_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
