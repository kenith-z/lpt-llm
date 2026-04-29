"""模型结构配置与快照读写。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

from .config import GlobalConfig


MODEL_CONFIG_SCHEMA_VERSION = 1
ATTENTION_BLOCK_TYPE = "attention"
RETNET_BLOCK_TYPE = "retnet"
LONGROPE2_STATIC_EMBEDDING_MODE = "static"
LONGROPE2_DYNAMIC_EMBEDDING_MODE = "dynamic"
LONGROPE2_MIXED_EMBEDDING_MODE = "mixed"
LONGROPE2_EMBEDDING_MODES = (
    LONGROPE2_STATIC_EMBEDDING_MODE,
    LONGROPE2_DYNAMIC_EMBEDDING_MODE,
    LONGROPE2_MIXED_EMBEDDING_MODE,
)
DEFAULT_LAYER_BLOCK_TYPES = (
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    RETNET_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
)


@dataclass(frozen=True)
class ModelConfig:
    """可序列化的模型结构配置。"""

    num_layers: int = 16
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 64
    hidden_size: int | None = None
    cla_share_every_n_layers: int = 2
    layer_block_types: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LAYER_BLOCK_TYPES)
    layer_state_group_ids: tuple[int | None, ...] | None = None
    retnet_value_factor: int = 1
    retnet_gate_fn: str = "swish"
    retnet_chunk_size: int = 256
    dropout_rate: float = 0.0
    rope_base: float = 10000.0
    original_max_len: int = 2048
    longrope2_target_length: int = field(default_factory=lambda: int(GlobalConfig.inference_max_sequence_length))
    longrope2_long_factors: tuple[float, ...] | float | None = None
    longrope2_factor_max_sequence_length: int | None = None
    longrope2_magnitude_scaling_policy: str = "su"
    longrope2_mscale_factors: tuple[float, ...] | None = None
    longrope2_train_embedding_mode: str = LONGROPE2_MIXED_EMBEDDING_MODE
    longrope2_inference_embedding_mode: str = LONGROPE2_MIXED_EMBEDDING_MODE
    longrope2_mixed_original_window: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "num_layers", int(self.num_layers))
        object.__setattr__(self, "num_heads", int(self.num_heads))
        object.__setattr__(self, "num_kv_heads", int(self.num_kv_heads))
        object.__setattr__(self, "head_dim", int(self.head_dim))
        object.__setattr__(self, "cla_share_every_n_layers", int(self.cla_share_every_n_layers))
        object.__setattr__(self, "retnet_value_factor", int(self.retnet_value_factor))
        object.__setattr__(self, "retnet_chunk_size", int(self.retnet_chunk_size))
        object.__setattr__(self, "dropout_rate", float(self.dropout_rate))
        object.__setattr__(self, "rope_base", float(self.rope_base))
        object.__setattr__(self, "original_max_len", int(self.original_max_len))
        object.__setattr__(self, "longrope2_target_length", int(self.longrope2_target_length))
        if self.original_max_len <= 0:
            raise ValueError("original_max_len 必须为正整数。")
        if self.longrope2_target_length <= 0:
            raise ValueError("longrope2_target_length 必须为正整数。")
        if self.longrope2_target_length < self.original_max_len:
            raise ValueError("longrope2_target_length 不能小于 original_max_len。")

        train_embedding_mode = str(self.longrope2_train_embedding_mode)
        inference_embedding_mode = str(self.longrope2_inference_embedding_mode)
        if train_embedding_mode not in LONGROPE2_EMBEDDING_MODES:
            raise ValueError(
                f"longrope2_train_embedding_mode 必须是 {LONGROPE2_EMBEDDING_MODES} 之一。"
            )
        if inference_embedding_mode not in LONGROPE2_EMBEDDING_MODES:
            raise ValueError(
                f"longrope2_inference_embedding_mode 必须是 {LONGROPE2_EMBEDDING_MODES} 之一。"
            )
        object.__setattr__(self, "longrope2_train_embedding_mode", train_embedding_mode)
        object.__setattr__(self, "longrope2_inference_embedding_mode", inference_embedding_mode)

        if self.longrope2_mixed_original_window is not None:
            mixed_original_window = int(self.longrope2_mixed_original_window)
            if mixed_original_window < 0:
                raise ValueError("longrope2_mixed_original_window 不能为负数。")
            object.__setattr__(self, "longrope2_mixed_original_window", mixed_original_window)

        inferred_hidden_size = self.num_heads * self.head_dim
        if self.hidden_size is None:
            object.__setattr__(self, "hidden_size", inferred_hidden_size)
        else:
            normalized_hidden_size = int(self.hidden_size)
            if normalized_hidden_size != inferred_hidden_size:
                raise ValueError(
                    f"hidden_size ({normalized_hidden_size}) 必须等于 "
                    f"num_heads * head_dim ({inferred_hidden_size})。"
                )
            object.__setattr__(self, "hidden_size", normalized_hidden_size)

        normalized_layer_block_types = tuple(str(value) for value in self.layer_block_types)
        if len(normalized_layer_block_types) != self.num_layers:
            raise ValueError(
                f"layer_block_types 长度 ({len(normalized_layer_block_types)}) "
                f"必须等于 num_layers ({self.num_layers})。"
            )
        object.__setattr__(self, "layer_block_types", normalized_layer_block_types)

        if self.layer_state_group_ids is not None:
            normalized_group_ids = tuple(
                None if value is None else int(value)
                for value in self.layer_state_group_ids
            )
            if len(normalized_group_ids) != self.num_layers:
                raise ValueError(
                    f"layer_state_group_ids 长度 ({len(normalized_group_ids)}) "
                    f"必须等于 num_layers ({self.num_layers})。"
                )
            object.__setattr__(self, "layer_state_group_ids", normalized_group_ids)

        if isinstance(self.longrope2_long_factors, list):
            object.__setattr__(
                self,
                "longrope2_long_factors",
                tuple(float(value) for value in self.longrope2_long_factors),
            )
        elif isinstance(self.longrope2_long_factors, tuple):
            object.__setattr__(
                self,
                "longrope2_long_factors",
                tuple(float(value) for value in self.longrope2_long_factors),
            )
        elif self.longrope2_long_factors is not None:
            factor = float(self.longrope2_long_factors)
            object.__setattr__(
                self,
                "longrope2_long_factors",
                tuple(factor for _ in range(max(1, self.head_dim // 2))),
            )

        if self.longrope2_factor_max_sequence_length is not None:
            factor_max_sequence_length = int(self.longrope2_factor_max_sequence_length)
            if factor_max_sequence_length <= 0:
                raise ValueError("longrope2_factor_max_sequence_length 必须为正整数或 None。")
            object.__setattr__(
                self,
                "longrope2_factor_max_sequence_length",
                factor_max_sequence_length,
            )

        if self.longrope2_mscale_factors is not None:
            object.__setattr__(
                self,
                "longrope2_mscale_factors",
                tuple(float(value) for value in self.longrope2_mscale_factors),
            )

    def to_dict(self):
        """导出为可 JSON 序列化的字典。"""
        return asdict(self)

    def to_json_payload(self):
        """导出包含 schema version 的 JSON 载荷。"""
        return {
            "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
            "model_config": self.to_dict(),
        }

    def with_overrides(self, **overrides):
        """基于当前配置派生一个新配置。"""
        return replace(self, **overrides)

    def save_json(self, path):
        """把配置快照保存为 JSON。"""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(self.to_json_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, payload):
        """从字典恢复模型结构配置。"""
        if payload is None:
            return cls()
        normalized_payload = dict(payload)
        if "layer_block_types" in normalized_payload and normalized_payload["layer_block_types"] is not None:
            normalized_payload["layer_block_types"] = tuple(normalized_payload["layer_block_types"])
        if "layer_state_group_ids" in normalized_payload and normalized_payload["layer_state_group_ids"] is not None:
            normalized_payload["layer_state_group_ids"] = tuple(normalized_payload["layer_state_group_ids"])
        if "longrope2_long_factors" in normalized_payload and isinstance(
            normalized_payload["longrope2_long_factors"], list
        ):
            normalized_payload["longrope2_long_factors"] = tuple(normalized_payload["longrope2_long_factors"])
        if "longrope2_mscale_factors" in normalized_payload and normalized_payload["longrope2_mscale_factors"] is not None:
            normalized_payload["longrope2_mscale_factors"] = tuple(normalized_payload["longrope2_mscale_factors"])
        return cls(**normalized_payload)

    @classmethod
    def from_json_payload(cls, payload):
        """从包含 schema version 的 JSON 载荷恢复配置。"""
        if not isinstance(payload, dict):
            raise TypeError("model config JSON 载荷必须是字典。")
        if "model_config" not in payload:
            raise ValueError("model config JSON 载荷缺少 model_config。")

        config_schema_version = payload.get("model_config_schema_version")
        if config_schema_version is None:
            raise ValueError("model config JSON 载荷缺少 model_config_schema_version。")
        if config_schema_version != MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "不支持的 model_config_schema_version: "
                f"{config_schema_version}，当前仅支持 {MODEL_CONFIG_SCHEMA_VERSION}。"
            )
        return cls.from_dict(payload["model_config"])

    @classmethod
    def load_json(cls, path):
        """从 JSON 文件加载模型结构配置。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_payload(payload)


def normalize_model_config(config=None):
    """标准化模型结构配置输入。"""
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config
    raise TypeError("config 必须是 ModelConfig 实例或 None。")


def build_model_config_from_checkpoint(checkpoint):
    """从 checkpoint 中恢复模型配置。"""
    if checkpoint is None:
        raise ValueError("checkpoint 不能为空。")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是字典。")

    model_config_schema_version = checkpoint.get("model_config_schema_version")
    if model_config_schema_version is None:
        raise ValueError("checkpoint 缺少 model_config_schema_version。")
    if model_config_schema_version != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "不支持的 model_config_schema_version: "
            f"{model_config_schema_version}，当前仅支持 {MODEL_CONFIG_SCHEMA_VERSION}。"
        )

    model_config_payload = checkpoint.get("model_config")
    if model_config_payload is None:
        raise ValueError("checkpoint 缺少 model_config 快照。")
    return ModelConfig.from_dict(model_config_payload)


def model_config_snapshot_path(artifact_dir):
    """返回约定的模型配置快照路径。"""
    return Path(artifact_dir) / "config" / "model_config.json"


def load_model_config_json(path):
    """从独立 JSON 快照中加载模型配置。"""
    return ModelConfig.load_json(path)


def load_longrope2_factors_file(path):
    """从搜索因子文件导入 LongRoPE2 factors，仅作为配置数组的导入入口。"""
    factor_path = Path(path)
    if not factor_path.exists():
        raise FileNotFoundError(f"未找到 LongRoPE2 long factors 文件: {factor_path}")

    raw_text = factor_path.read_text(encoding="utf-8").replace(",", "\n")
    factors = tuple(float(value) for value in raw_text.split() if value.strip())
    if not factors:
        raise ValueError(f"LongRoPE2 long factors 文件为空: {factor_path}")
    if any(factor <= 0 for factor in factors):
        raise ValueError("LongRoPE2 long factors 必须全部大于 0。")
    return factors


def build_longrope2_uniform_factors(config, sequence_length):
    """基于数据最长 token 长度生成一组确定性 bootstrap factors。"""
    rotary_dims = int(config.head_dim) // 2
    if rotary_dims <= 0:
        raise ValueError("head_dim 必须至少包含一组 rotary 维度。")
    coverage_length = max(int(sequence_length), int(config.longrope2_target_length))
    factor = max(float(coverage_length) / float(config.original_max_len), 1.0)
    return tuple(float(factor) for _ in range(rotary_dims))
