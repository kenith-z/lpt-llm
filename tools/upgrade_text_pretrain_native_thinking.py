"""把旧 text_pretrain 产物升级为原生 thinking 控制版本。

旧 text_pretrain 基座是在原生 thinking embedding 引入前训练完成的，权重中缺少
`thinking_mode_embedding.weight` 和 `thinking_channel_embedding.weight`。这两个
embedding 在当前模型中按零初始化接入，因此可以在不重复预训练的情况下补齐：

- 原有主干权重保持不变。
- 新增 thinking mode/channel embedding 使用与 token embedding 相同 dtype 的零张量。
- checkpoint 和 `config/model_config.json` 的 `model_config_schema_version` 升级到当前版本。

默认不原地覆盖旧产物，而是复制到旁路目录后升级。确实要原地升级时需要显式传入
`--in-place --yes`。
"""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import (  # noqa: E402
    LPT_V2_ARCHITECTURE_VERSION,
    MODEL_CONFIG_SCHEMA_VERSION,
    ModelConfig,
    THINKING_CHANNEL_COUNT,
    THINKING_CONTROL_SCHEMA_VERSION,
    THINKING_MODE_COUNT,
)
from lpt_model import (  # noqa: E402
    LPT_V2_CHECKPOINT_FORMAT,
    LPT_V2_CHECKPOINT_SCHEMA_VERSION,
    validate_lpt_v2_checkpoint_payload,
)
from lpt_runtime.files import atomic_torch_save, atomic_write_text  # noqa: E402


DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "lpt_v2" / "text_pretrain"
MODEL_CHECKPOINT_NAME = "model.pt"
INFERENCE_CHECKPOINT_NAME = "model_checkpoint.pt"
INFERENCE_WEIGHTS_NAME = "model_weights.pth"
MODEL_CONFIG_NAME = "model_config.json"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
TRAINER_STATE_NAME = "trainer_state.json"
THINKING_MODE_KEY = "thinking_mode_embedding.weight"
THINKING_CHANNEL_KEY = "thinking_channel_embedding.weight"


@dataclass(frozen=True)
class UpgradeReport:
    """单个文件或目录的升级结果。"""

    path: Path
    kind: str
    changed: bool
    message: str


def _utc_timestamp():
    """返回稳定可读的 UTC 时间戳，写入升级审计元数据。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _copy_mapping(mapping):
    """浅拷贝权重映射，避免直接修改 torch.load 返回对象。"""
    if isinstance(mapping, OrderedDict):
        return OrderedDict(mapping)
    return dict(mapping)


def _find_reference_embedding(state_dict):
    """从已有权重中找到 hidden size、dtype 和 device 的参考张量。"""
    reference = state_dict.get("token_embedding.weight")
    if reference is None:
        reference = state_dict.get("lm_head.weight")
    if reference is None:
        raise ValueError("权重缺少 token_embedding.weight/lm_head.weight，无法推断 hidden size。")
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError("token_embedding.weight/lm_head.weight 必须是二维 tensor。")
    return reference


def upgrade_state_dict_for_native_thinking(
    state_dict,
    *,
    mode_count=THINKING_MODE_COUNT,
    channel_count=THINKING_CHANNEL_COUNT,
    force=False,
):
    """给 state_dict 补齐原生 thinking 控制 embedding。

    新增 embedding 必须是零初始化；这样对于 text_pretrain 基座，默认
    `thinking=off` 的前向结果不会因为升级脚本本身产生额外偏移。
    """
    if not isinstance(state_dict, MutableMapping):
        raise TypeError("state_dict 必须是可变映射。")

    upgraded_state = _copy_mapping(state_dict)
    reference = _find_reference_embedding(upgraded_state)
    hidden_size = int(reference.shape[1])
    changed = False

    def ensure_zero_embedding(key, row_count):
        nonlocal changed
        expected_shape = (int(row_count), hidden_size)
        existing = upgraded_state.get(key)
        if existing is not None:
            if not isinstance(existing, torch.Tensor):
                raise TypeError(f"{key} 必须是 tensor。")
            if tuple(existing.shape) == expected_shape:
                return
            if not force:
                raise ValueError(f"{key} 形状为 {tuple(existing.shape)}，期望 {expected_shape}。")
        upgraded_state[key] = torch.zeros(
            expected_shape,
            dtype=reference.dtype,
            device=reference.device,
        )
        changed = True

    ensure_zero_embedding(THINKING_MODE_KEY, mode_count)
    ensure_zero_embedding(THINKING_CHANNEL_KEY, channel_count)
    return upgraded_state, changed


def _build_upgraded_model_config(config_payload):
    """把旧 ModelConfig payload 升级为当前 schema 的完整配置。"""
    if not isinstance(config_payload, dict):
        raise TypeError("checkpoint.model_config 必须是字典。")
    upgraded_payload = dict(config_payload)
    upgraded_payload["thinking_control_enabled"] = True
    upgraded_payload["thinking_control_schema_version"] = THINKING_CONTROL_SCHEMA_VERSION
    upgraded_payload["thinking_mode_count"] = THINKING_MODE_COUNT
    upgraded_payload["thinking_channel_count"] = THINKING_CHANNEL_COUNT
    return ModelConfig.from_dict(upgraded_payload)


def _upgrade_runtime_metadata(runtime_metadata, *, source_label, old_model_config_schema_version):
    """补齐 runtime metadata 中的 thinking 控制审计信息。"""
    if not isinstance(runtime_metadata, dict):
        raise TypeError("checkpoint.runtime_metadata 必须是字典。")
    upgraded_metadata = dict(runtime_metadata)
    state_schema = dict(upgraded_metadata.get("state_schema") or {})
    state_schema.setdefault("layer_state_schema", "LayerStateV2")
    state_schema["thinking_control"] = {
        "schema_version": THINKING_CONTROL_SCHEMA_VERSION,
        "enabled": True,
        "mode_count": THINKING_MODE_COUNT,
        "channel_count": THINKING_CHANNEL_COUNT,
        "control_source": "structured_tensor",
        "legacy_text_tags_supported": False,
        "upgrade_source": "tools/upgrade_text_pretrain_native_thinking.py",
    }
    upgraded_metadata["state_schema"] = state_schema

    extra = dict(upgraded_metadata.get("extra") or {})
    upgrade_events = list(extra.get("native_thinking_upgrades", ()))
    upgrade_events.append(
        {
            "tool": "tools/upgrade_text_pretrain_native_thinking.py",
            "source": source_label,
            "upgraded_at": _utc_timestamp(),
            "old_model_config_schema_version": old_model_config_schema_version,
            "new_model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
            "new_state_keys": [THINKING_MODE_KEY, THINKING_CHANNEL_KEY],
            "new_state_initialization": "zeros",
        }
    )
    extra["native_thinking_upgrades"] = upgrade_events
    upgraded_metadata["extra"] = extra
    return upgraded_metadata


def upgrade_checkpoint_payload(checkpoint, *, source_label="<memory>", force=False):
    """升级完整 LPT v2 checkpoint payload。"""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是字典。")
    if checkpoint.get("checkpoint_format") != LPT_V2_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint_format 不是 lpt_v2_checkpoint。")
    if checkpoint.get("checkpoint_schema_version") != LPT_V2_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint_schema_version 不匹配: "
            f"{checkpoint.get('checkpoint_schema_version')}"
        )
    if checkpoint.get("architecture_version") != LPT_V2_ARCHITECTURE_VERSION:
        raise ValueError("checkpoint architecture_version 不是 lpt_v2。")

    old_config_schema = checkpoint.get("model_config_schema_version")
    if old_config_schema is None:
        raise ValueError("checkpoint 缺少 model_config_schema_version。")
    if int(old_config_schema) > MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint model_config_schema_version 高于当前代码支持版本: "
            f"{old_config_schema} > {MODEL_CONFIG_SCHEMA_VERSION}"
        )

    upgraded_config = _build_upgraded_model_config(checkpoint.get("model_config"))
    upgraded_state, state_changed = upgrade_state_dict_for_native_thinking(
        checkpoint.get("model_state_dict"),
        mode_count=upgraded_config.thinking_mode_count,
        channel_count=upgraded_config.thinking_channel_count,
        force=force,
    )
    upgraded_checkpoint = dict(checkpoint)
    upgraded_checkpoint["model_config_schema_version"] = MODEL_CONFIG_SCHEMA_VERSION
    upgraded_checkpoint["model_config"] = upgraded_config.to_dict()
    upgraded_checkpoint["model_state_dict"] = upgraded_state
    upgraded_checkpoint["runtime_metadata"] = _upgrade_runtime_metadata(
        checkpoint.get("runtime_metadata"),
        source_label=source_label,
        old_model_config_schema_version=old_config_schema,
    )

    validate_lpt_v2_checkpoint_payload(upgraded_checkpoint)
    changed = (
        bool(state_changed)
        or int(old_config_schema) != MODEL_CONFIG_SCHEMA_VERSION
        or checkpoint.get("model_config") != upgraded_checkpoint["model_config"]
    )
    return upgraded_checkpoint, changed


def upgrade_model_config_json_payload(payload):
    """升级独立的 config/model_config.json payload。"""
    if not isinstance(payload, dict):
        raise TypeError("model_config.json payload 必须是字典。")
    old_schema = payload.get("model_config_schema_version")
    if old_schema is None:
        raise ValueError("model_config.json 缺少 model_config_schema_version。")
    if int(old_schema) > MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "model_config.json schema 高于当前代码支持版本: "
            f"{old_schema} > {MODEL_CONFIG_SCHEMA_VERSION}"
        )
    upgraded_config = _build_upgraded_model_config(payload.get("model_config"))
    upgraded_payload = {
        "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
        "model_config": upgraded_config.to_dict(),
    }
    changed = int(old_schema) != MODEL_CONFIG_SCHEMA_VERSION or payload != upgraded_payload
    return upgraded_payload, changed


def _is_checkpoint_payload(payload):
    """判断 torch 文件是否是完整 LPT v2 checkpoint，而不是 plain state_dict。"""
    return isinstance(payload, dict) and payload.get("checkpoint_format") == LPT_V2_CHECKPOINT_FORMAT


def upgrade_torch_file(path, *, dry_run=False, force=False):
    """升级单个 torch 文件，自动识别 checkpoint 和 plain state_dict。"""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if _is_checkpoint_payload(payload):
        upgraded_payload, changed = upgrade_checkpoint_payload(
            payload,
            source_label=str(path),
            force=force,
        )
        kind = "checkpoint"
    else:
        upgraded_payload, changed = upgrade_state_dict_for_native_thinking(
            payload,
            force=force,
        )
        kind = "state_dict"
    if changed and not dry_run:
        atomic_torch_save(upgraded_payload, path)
    message = "已升级" if changed else "已是当前 thinking 权重格式"
    if dry_run and changed:
        message = "dry-run: 将升级"
    return UpgradeReport(path=path, kind=kind, changed=changed, message=message)


def upgrade_model_config_json_file(path, *, dry_run=False):
    """升级单个 model_config.json 文件。"""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    upgraded_payload, changed = upgrade_model_config_json_payload(payload)
    if changed and not dry_run:
        atomic_write_text(
            path,
            json.dumps(upgraded_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    message = "已升级" if changed else "已是当前 model config schema"
    if dry_run and changed:
        message = "dry-run: 将升级"
    return UpgradeReport(path=path, kind="model_config_json", changed=changed, message=message)


def _upgrade_trainer_state(checkpoint_root, *, dry_run=False):
    """给 trainer_state 添加升级审计信息；不存在时跳过。"""
    state_path = Path(checkpoint_root) / TRAINER_STATE_NAME
    if not state_path.is_file():
        return UpgradeReport(state_path, "trainer_state", False, "不存在，跳过")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    upgraded_state = dict(state)
    existing_upgrade = upgraded_state.get("native_thinking_upgrade")
    if not (
        isinstance(existing_upgrade, dict)
        and existing_upgrade.get("tool") == "tools/upgrade_text_pretrain_native_thinking.py"
        and existing_upgrade.get("new_state_keys") == [THINKING_MODE_KEY, THINKING_CHANNEL_KEY]
    ):
        upgraded_state["native_thinking_upgrade"] = {
            "tool": "tools/upgrade_text_pretrain_native_thinking.py",
            "upgraded_at": _utc_timestamp(),
            "new_state_keys": [THINKING_MODE_KEY, THINKING_CHANNEL_KEY],
            "new_state_initialization": "zeros",
        }
    training_config = dict(upgraded_state.get("training_config", {}))
    training_config.setdefault("thinking_mode", "off")
    training_config.setdefault("thinking_visibility", "hidden")
    upgraded_state["training_config"] = training_config
    thinking = dict(upgraded_state.get("thinking", {}))
    thinking.setdefault("schema_version", 1)
    thinking.setdefault("legacy_text_tags_supported", False)
    thinking.setdefault("training_thinking_mode", "off")
    thinking.setdefault("training_thinking_visibility", "hidden")
    thinking.setdefault("training_visibility_affects_loss", False)
    upgraded_state["thinking"] = thinking

    changed = upgraded_state != state
    if changed and not dry_run:
        atomic_write_text(
            state_path,
            json.dumps(upgraded_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    message = "已写入升级审计" if changed else "升级审计已存在"
    if dry_run and changed:
        message = "dry-run: 将写入升级审计"
    return UpgradeReport(state_path, "trainer_state", changed, message)


def _refresh_checkpoint_manifest(checkpoint_root, *, dry_run=False):
    """刷新 checkpoint_manifest.json 中的文件大小。"""
    manifest_path = Path(checkpoint_root) / CHECKPOINT_MANIFEST_NAME
    if not manifest_path.is_file():
        return UpgradeReport(manifest_path, "checkpoint_manifest", False, "不存在，跳过")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = False
    for entry in manifest.get("files", ()):
        name = entry.get("name")
        file_path = Path(checkpoint_root) / str(name)
        if not file_path.is_file():
            continue
        size_bytes = int(file_path.stat().st_size)
        if entry.get("size_bytes") != size_bytes:
            entry["size_bytes"] = size_bytes
            changed = True
    if changed and not dry_run:
        atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    message = "已刷新文件大小" if changed else "文件大小无需刷新"
    if dry_run and changed:
        message = "dry-run: 将刷新文件大小"
    return UpgradeReport(manifest_path, "checkpoint_manifest", changed, message)


def _iter_checkpoint_roots(artifact_dir, *, include_previous=True, include_step_checkpoints=False):
    """枚举需要升级的训练 checkpoint 目录。"""
    checkpoint_dir = Path(artifact_dir) / "checkpoints"
    roots = [checkpoint_dir / "latest"]
    if include_previous:
        roots.append(checkpoint_dir / "latest_previous")
    if include_step_checkpoints and checkpoint_dir.is_dir():
        roots.extend(sorted(path for path in checkpoint_dir.glob("step_*") if path.is_dir()))
    for root in roots:
        if root.is_dir():
            yield root


def _prepare_target_artifact(source_dir, output_dir, *, in_place=False, overwrite_output=False, dry_run=False):
    """准备升级目标目录；默认复制到旁路目录，避免覆盖旧基座。"""
    source_dir = Path(source_dir).resolve()
    if in_place:
        return source_dir
    output_dir = Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("output_artifact_dir 与 artifact_dir 相同；如需原地升级请使用 --in-place --yes。")
    if dry_run:
        return output_dir
    if output_dir.exists():
        if not overwrite_output:
            raise FileExistsError(f"输出目录已存在: {output_dir}。如需覆盖请传 --overwrite-output。")
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    return output_dir


def upgrade_text_pretrain_artifact(
    artifact_dir=DEFAULT_ARTIFACT_DIR,
    *,
    output_artifact_dir=None,
    in_place=False,
    overwrite_output=False,
    include_previous=True,
    include_step_checkpoints=False,
    dry_run=False,
    force=False,
):
    """升级 text_pretrain artifact，返回逐文件报告。"""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact_dir 不存在: {artifact_dir}")
    if output_artifact_dir is None:
        output_artifact_dir = artifact_dir.with_name(f"{artifact_dir.name}_native_thinking")

    target_dir = _prepare_target_artifact(
        artifact_dir,
        output_artifact_dir,
        in_place=in_place,
        overwrite_output=overwrite_output,
        dry_run=dry_run,
    )
    reports = []
    if not in_place:
        reports.append(
            UpgradeReport(
                path=target_dir,
                kind="artifact_copy",
                changed=True,
                message="已复制待升级 artifact" if not dry_run else "dry-run: 将复制 artifact",
            )
        )

    # dry-run 不复制输出目录，检查源目录即可得到同等的升级计划。
    working_dir = artifact_dir.resolve() if dry_run and not in_place else target_dir

    config_path = working_dir / "config" / MODEL_CONFIG_NAME
    if config_path.is_file():
        reports.append(upgrade_model_config_json_file(config_path, dry_run=dry_run))

    weights_dir = working_dir / "weights"
    for weight_path in (
        weights_dir / INFERENCE_CHECKPOINT_NAME,
        weights_dir / INFERENCE_WEIGHTS_NAME,
    ):
        if weight_path.is_file():
            reports.append(upgrade_torch_file(weight_path, dry_run=dry_run, force=force))

    for checkpoint_root in _iter_checkpoint_roots(
        working_dir,
        include_previous=include_previous,
        include_step_checkpoints=include_step_checkpoints,
    ):
        model_path = checkpoint_root / MODEL_CHECKPOINT_NAME
        if model_path.is_file():
            reports.append(upgrade_torch_file(model_path, dry_run=dry_run, force=force))
        reports.append(_upgrade_trainer_state(checkpoint_root, dry_run=dry_run))
        reports.append(_refresh_checkpoint_manifest(checkpoint_root, dry_run=dry_run))

    return reports


def build_parser():
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(
        description="把旧 text_pretrain artifact 升级为带原生 thinking 控制 embedding 的版本。"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="旧 text_pretrain artifact 目录。",
    )
    parser.add_argument(
        "--output-artifact-dir",
        type=Path,
        default=None,
        help="升级后的输出目录；默认写到同级 text_pretrain_native_thinking。",
    )
    parser.add_argument("--in-place", action="store_true", help="原地升级 artifact。")
    parser.add_argument("--yes", action="store_true", help="确认允许 --in-place 原地覆盖。")
    parser.add_argument("--overwrite-output", action="store_true", help="允许覆盖已存在的输出目录。")
    parser.add_argument("--dry-run", action="store_true", help="只检查并打印将执行的升级，不写文件。")
    parser.add_argument("--force", action="store_true", help="已有 thinking embedding 但形状不符时强制重建。")
    parser.add_argument(
        "--no-previous",
        dest="include_previous",
        action="store_false",
        default=True,
        help="不升级 latest_previous。",
    )
    parser.add_argument(
        "--include-step-checkpoints",
        action="store_true",
        help="同时升级 checkpoints/step_* 目录。",
    )
    return parser


def main(argv=None):
    """命令行入口。"""
    args = build_parser().parse_args(argv)
    if args.in_place and not args.yes:
        raise SystemExit("原地升级会覆盖现有 artifact；请显式传入 --in-place --yes。")

    reports = upgrade_text_pretrain_artifact(
        args.artifact_dir,
        output_artifact_dir=args.output_artifact_dir,
        in_place=args.in_place,
        overwrite_output=args.overwrite_output,
        include_previous=args.include_previous,
        include_step_checkpoints=args.include_step_checkpoints,
        dry_run=args.dry_run,
        force=args.force,
    )
    for report in reports:
        changed_text = "changed" if report.changed else "unchanged"
        print(f"{changed_text}\t{report.kind}\t{report.path}\t{report.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
