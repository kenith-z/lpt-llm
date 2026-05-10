"""模型结构配置与快照读写。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

from .config import GlobalConfig
from .constants import (
    DEFAULT_MODEL_SIZE_PRESET,
    LPT_V2_BASE_PRESET,
    LPT_V2_DEV_TINY_PRESET,
    LPT_V2_LARGE_PRESET,
    LPT_V2_SMALL_PRESET, LPT_V2_SMALL_BASE_PRESET,
)


# 当前支持的 ModelConfig JSON schema 版本；checkpoint 加载时必须严格匹配。
MODEL_CONFIG_SCHEMA_VERSION = 2
# LPT v2 独立分支架构标识，是 v2 checkpoint 与配置的主入口。
LPT_V2_ARCHITECTURE_VERSION = "lpt_v2"
# LPT v2 block 类型，约束每层使用 Attention 主干 + RetNetAssist Q adapter。
LPT_V2_BLOCK_TYPE = "lpt_attention_retnet_q_adapter"
# LPT v2 sequence mixer 模式，描述局部 Attention 与 RetNet Q-only 辅助的组合语义。
LPT_V2_SEQUENCE_MIXER_MODE = "local_attention_with_retnet_q_adapter"
# 参数量统计策略，要求按 MoE 物理参数和激活参数分别计数。
PARAMETER_COUNT_POLICY_MOE_AWARE = "moe_aware"
# 参数量报告固定输出口径，保证实验报告可横向比较。
PARAMETER_COUNT_MODES = (
    "total_physical_params",
    "active_params_per_token",
    "shared_params",
    "expert_params",
    "router_params",
    "adapter_params",
    "state_runtime_bytes",
)
# LPT v2 block 类型布局值：Attention 层。
ATTENTION_BLOCK_TYPE = "attention"
# Attention 后端：FlashAttention-3，后续性能评估用扩展后端。
FLASH_ATTENTION_3_BACKEND = "flash_attention_3"
# Attention 后端：FlashAttention-2，后续性能评估用扩展后端。
FLASH_ATTENTION_2_BACKEND = "flash_attention_2"
# Attention 后端：PyTorch SDPA，当前 v2 定型默认后端。
SDPA_ATTENTION_BACKEND = "sdpa"
# Attention 后端策略：自动按优先级和能力选择。
AUTO_ATTENTION_BACKEND_POLICY = "auto"
# v2 后端选择器允许识别的具体 Attention 后端。
SUPPORTED_ATTENTION_BACKENDS = (
    FLASH_ATTENTION_3_BACKEND,
    FLASH_ATTENTION_2_BACKEND,
    SDPA_ATTENTION_BACKEND,
)
# v2 后端选择器允许的策略值，包含 auto 与固定后端。
SUPPORTED_ATTENTION_BACKEND_POLICIES = (
    AUTO_ATTENTION_BACKEND_POLICY,
    *SUPPORTED_ATTENTION_BACKENDS,
)
# auto 策略下的默认后端尝试顺序；v2 当前定型为 SDPA-only，避免环境差异静默切换 FA。
DEFAULT_ATTENTION_BACKEND_PRIORITY = (
    SDPA_ATTENTION_BACKEND,
)
# v2 正式缓存后端：Paged KV。
PAGED_KV_CACHE_BACKEND = "paged_kv"
# 启动/对照 profile 可使用的 dense KV 缓存后端。
DENSE_KV_CACHE_BACKEND = "dense_kv"
# 配置校验接受的 KV cache 后端集合。
SUPPORTED_CACHE_BACKENDS = (PAGED_KV_CACHE_BACKEND, DENSE_KV_CACHE_BACKEND)
# LongRoPE2 模式：固定使用长因子。
LONGROPE2_STATIC_EMBEDDING_MODE = "static"
# LongRoPE2 模式：按输入位置动态切换缩放。
LONGROPE2_DYNAMIC_EMBEDDING_MODE = "dynamic"
# LongRoPE2 模式：原始窗口内使用原始 RoPE，窗口外使用长因子。
LONGROPE2_MIXED_EMBEDDING_MODE = "mixed"
# LongRoPE2 配置校验接受的 embedding mode 集合。
LONGROPE2_EMBEDDING_MODES = (
    LONGROPE2_STATIC_EMBEDDING_MODE,
    LONGROPE2_DYNAMIC_EMBEDDING_MODE,
    LONGROPE2_MIXED_EMBEDDING_MODE,
)
# 默认 v2 tiny 层类型布局；v2 主干全部是 Attention block。
DEFAULT_LAYER_BLOCK_TYPES = (
    ATTENTION_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
    ATTENTION_BLOCK_TYPE,
)
# 从 JSON/list 恢复时需要规范化为 tuple 的 ModelConfig 字段。
_LIST_LIKE_MODEL_CONFIG_FIELDS = (
    "layer_block_types",
    "layer_state_group_ids",
    "parameter_count_modes",
    "longrope2_long_factors",
    "longrope2_mscale_factors",
    "attention_backend_priority",
    "retnet_adapter_target",
    "xlstm_memory_adapter_beta_range",
    "xlstm_memory_selected_layers",
    "xlstm_memory_reset_trigger_mode",
    "xlstm_memory_reset_boundary_policy",
    "xlstm_memory_boundary_token_ids",
)

# LPT v2 规格预设表；每个 preset 只负责生成初始完整 ModelConfig。
LPT_V2_MODEL_SIZE_PRESETS = {
    # 默认开发规格：层数和隐藏维度最小，适合快速单元测试。
    LPT_V2_DEV_TINY_PRESET: {
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
        "moe_num_experts": 2,
        "moe_top_k": 1,
        "attention_window_size": 512,
    },
    # 小规格验证：保留多层和多 expert 形态，成本仍较低。
    LPT_V2_SMALL_PRESET: {
        "num_layers": 12,
        "num_heads": 12,
        "num_kv_heads": 4,
        "head_dim": 64,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "attention_window_size": 2048,
    },
    # 小规格主训练规格：用于 base 级别正式实验。
    LPT_V2_SMALL_BASE_PRESET: {
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 64,
        "moe_num_experts": 6,
        "moe_top_k": 2,
        "attention_window_size": 2048,
    },
    # 主训练规格：用于 base 级别正式实验。
    LPT_V2_BASE_PRESET: {
        "num_layers": 24,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 96,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "attention_window_size": 4096,
    },
    # 扩展规格：更深更宽，用于后续放大验证。
    LPT_V2_LARGE_PRESET: {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "attention_window_size": 4096,
    },
}
DEFAULT_MODEL_SIZE_PRESET_VALUES = LPT_V2_MODEL_SIZE_PRESETS[DEFAULT_MODEL_SIZE_PRESET]


def _as_tuple(value):
    """把配置输入统一成 tuple，便于 dataclass frozen 字段稳定比较。"""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _resolve_every_n_layers_interval(layer_policy, *, default_interval=1):
    """解析 all/every_N 层策略，返回启用间隔。"""
    policy = str(layer_policy)
    if policy == "all_layers":
        return 1
    if policy == "every_n_layers":
        return int(default_interval)
    if policy.startswith("every_") and policy.endswith("_layers"):
        raw_interval = policy.removeprefix("every_").removesuffix("_layers")
        if raw_interval.isdigit():
            return int(raw_interval)
    return None


def is_valid_xlstm_memory_layer_policy(layer_policy):
    """判断 xLSTMAssist 层启用策略是否为当前可执行口径。"""
    policy = str(layer_policy)
    if policy in {"disabled", "selected_layers"}:
        return True
    interval = _resolve_every_n_layers_interval(policy)
    return interval is not None and interval > 0


def is_xlstm_memory_enabled_for_layer(config, layer_index):
    """判断指定层是否启用 xLSTMAssist。"""
    if not bool(config.xlstm_memory_enabled):
        return False
    policy = str(config.xlstm_memory_layers)
    if policy == "disabled":
        return False
    if policy == "selected_layers":
        return int(layer_index) in config.xlstm_memory_selected_layers
    interval = _resolve_every_n_layers_interval(policy)
    if interval is None or interval <= 0:
        return False
    return int(layer_index) % interval == 0


def count_xlstm_memory_enabled_layers(config):
    """按 ModelConfig 统计实际启用 xLSTMAssist 的层数。"""
    if not bool(config.xlstm_memory_enabled):
        return 0
    return sum(
        1
        for layer_index in range(int(config.num_layers))
        if is_xlstm_memory_enabled_for_layer(config, layer_index)
    )


@dataclass(frozen=True)
class ModelConfig:
    """可序列化的模型结构配置。"""

    # 记录当前配置来自哪个 LPT v2 规格预设；checkpoint 会保存展开后的完整字段和该标识。
    model_size_preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 项目默认规格标识，用于明确无参构造时应走 dev tiny。
    default_model_size_preset: str = DEFAULT_MODEL_SIZE_PRESET
    # 参数量统计策略；当前固定为 MoE-aware，区分物理参数与每 token 激活参数。
    parameter_count_policy: str = PARAMETER_COUNT_POLICY_MOE_AWARE
    # 参数量报告需要输出的统计维度集合。
    parameter_count_modes: tuple[str, ...] = field(default_factory=lambda: PARAMETER_COUNT_MODES)
    # 模型架构大版本；v2 loader 用它拒绝 v1 或其它未知结构。
    architecture_version: str = LPT_V2_ARCHITECTURE_VERSION
    # v2 block 结构标识，描述每层使用 Attention 主干加 RetNetAssist Q adapter。
    block_type: str = LPT_V2_BLOCK_TYPE
    # sequence mixer 语义标识，约束主干为局部 Attention + RetNet Q-only 辅助。
    sequence_mixer_mode: str = LPT_V2_SEQUENCE_MIXER_MODE
    # Transformer block 层数。
    num_layers: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["num_layers"]
    # Attention query 头数。
    num_heads: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["num_heads"]
    # GQA 的 key/value 头数。
    num_kv_heads: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["num_kv_heads"]
    # 单个 attention head 的维度。
    head_dim: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["head_dim"]
    # 隐藏层宽度；为空时由 num_heads * head_dim 推导。
    hidden_size: int | None = None
    # CLA/KV 共享粒度；v2 固定为 1，表示 Attention 层不共享 KV。
    cla_share_every_n_layers: int = 1
    # 每层 block 类型；v2 必须全部为 attention。
    layer_block_types: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LAYER_BLOCK_TYPES)
    # 层状态分组 id；为空时由层布局和 CLA 规则自动生成。
    layer_state_group_ids: tuple[int | None, ...] | None = None
    # Attention 后端选择策略，auto 表示按优先级和能力自动选择。
    attention_backend_policy: str = AUTO_ATTENTION_BACKEND_POLICY
    # auto 策略下的后端优先级。
    attention_backend_priority: tuple[str, ...] = field(default_factory=lambda: DEFAULT_ATTENTION_BACKEND_PRIORITY)
    # 局部 Attention 滑动窗口大小。
    attention_window_size: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["attention_window_size"]
    # Attention 是否使用 causal mask；语言模型场景固定为 true。
    attention_is_causal: bool = True
    # Attention 位置编码类型；v2 固定为 LongRoPE2。
    attention_position_encoding: str = "longrope2"
    # KV cache 后端；v2 正式路径使用 paged_kv。
    cache_backend: str = PAGED_KV_CACHE_BACKEND
    # KV cache 保存范围；只缓存局部窗口内真实 token 的 K/V。
    kv_cache_scope: str = "local_real_tokens_only"
    # Paged KV 单个 page/block 包含的 token 数。
    page_block_size: int = 256
    # 是否启用 v2 RetNetAssist 全局摘要。
    retnet_assist_enabled: bool = True
    # RetNetAssist 作用模式；主线默认 Q adapter，第 24 项允许 Q/K adapter 做单项实验。
    retnet_assist_mode: str = "q_adapter"
    # RetNetAssist 启用层策略，如 every_4_layers / selected_layers / all_layers。
    retnet_assist_layers: str = "every_4_layers"
    # RetNetAssist 参数共享策略，global 表示跨层共享一组参数。
    retnet_parameter_sharing: str = "global"
    # RetNetAssist 状态共享策略，group 表示按共享组维护状态。
    retnet_state_sharing: str = "group"
    # RetNetAssist prefill 扫描策略，要求支持 sequence parallel 友好的 chunkwise scan。
    retnet_prefill_scan_policy: str = "sp_compatible_chunkwise_scan"
    # RetNetAssist 在 sequence parallel 下的状态交接策略。
    retnet_sequence_parallel_policy: str = "disabled"
    # 是否记录 RetNetAssist sequence-parallel handoff 指标。
    retnet_sp_handoff_metrics_enabled: bool = True
    # RetNetAssist 状态生命周期；必须绑定 request state pool。
    retnet_state_lifecycle: str = "request_bound_state_pool"
    # RetNetAssist 轻量全局摘要状态维度。
    retnet_state_dim: int = 64
    # RetNetAssist Q adapter 的低秩投影 rank。
    retnet_adapter_rank: int = 16
    # RetNetAssist adapter 目标；默认只调制 Q，第 24 项允许同时调制 Q/K。
    retnet_adapter_target: tuple[str, ...] = field(default_factory=lambda: ("q",))
    # Q adapter FP32 scale 初始值。
    retnet_adapter_alpha_q_init: float = 1e-4
    # Q adapter alpha 的参数 dtype；当前固定 FP32。
    retnet_adapter_alpha_q_dtype: str = "fp32"
    # Q adapter alpha 是否可训练。
    retnet_adapter_alpha_q_trainable: bool = True
    # K adapter FP32 scale 初始值；只有 retnet_k_adapter_enabled=true 时实例化参数。
    retnet_adapter_alpha_k_init: float = 1e-4
    # K adapter alpha 的参数 dtype；当前固定 FP32。
    retnet_adapter_alpha_k_dtype: str = "fp32"
    # K adapter alpha 是否可训练。
    retnet_adapter_alpha_k_trainable: bool = True
    # 是否启用 K adapter；默认关闭，仅第 24 项实验打开。
    retnet_k_adapter_enabled: bool = False
    # 是否启用上下文 adapter；当前仅预留评估入口。
    retnet_context_adapter_enabled: bool = False
    # 上下文 adapter 的初始 scale；未启用时保持 0。
    retnet_context_adapter_alpha: float = 0.0
    # RetNetAssist 是否写入 Paged KV；必须为 false。
    retnet_enters_paged_kv: bool = False
    # RetNetAssist 是否替换 KV cache；必须为 false。
    retnet_kv_replacement: bool = False
    # 是否从 RetNetAssist 生成 attention logit bias；当前禁止。
    attention_logit_bias_from_retnet: bool = False
    # FFN 类型；v2 使用 memory-augmented SwiGLU MoE。
    ffn_type: str = "memory_augmented_swiglu_moe"
    # 每层 MoE 的物理 expert 数。
    moe_num_experts: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["moe_num_experts"]
    # 每 token 路由激活的 expert 数。
    moe_top_k: int = DEFAULT_MODEL_SIZE_PRESET_VALUES["moe_top_k"]
    # MoE router logits dtype；固定 FP32 以保证路由稳定。
    moe_router_dtype: str = "fp32"
    # 是否启用 MoE load balance loss。
    moe_load_balance_loss_enabled: bool = True
    # 是否启用 router z-loss。
    moe_router_z_loss_enabled: bool = True
    # Router 输入模式，控制是否读取 xLSTM memory 调制后的 x_ffn。
    moe_router_input_mode: str = "ffn_norm_only_eval"
    # 是否启用 FFN 侧 xLSTMAssist 记忆。
    xlstm_memory_enabled: bool = False
    # xLSTMAssist 作用模式；当前作为 FFN 输入 adapter。
    xlstm_memory_mode: str = "ffn_input_adapter"
    # xLSTMAssist 启用层策略；关闭时必须为 disabled。
    xlstm_memory_layers: str = "disabled"
    # selected_layers 策略下显式启用的层号，0 基索引。
    xlstm_memory_selected_layers: tuple[int, ...] = field(default_factory=tuple)
    # xLSTMAssist 记忆状态维度。
    xlstm_memory_state_dim: int = 64
    # xLSTMAssist memory adapter 的低秩投影 rank。
    xlstm_memory_adapter_rank: int = 16
    # xLSTM memory adapter beta 初始值。
    xlstm_memory_adapter_beta_init: float = 1e-4
    # xLSTM memory adapter beta 参数 dtype；固定 FP32。
    xlstm_memory_adapter_beta_dtype: str = "fp32"
    # xLSTM memory adapter beta 的有效值约束策略。
    xlstm_memory_adapter_beta_policy: str = "fp32_sigmoid_clamped"
    # xLSTM memory adapter beta 的有效取值范围。
    xlstm_memory_adapter_beta_range: tuple[float, float] = (1e-5, 1.0)
    # xLSTMAssist 是否作为 router 目标；必须为 false。
    xlstm_memory_as_router_target: bool = False
    # 是否启用额外 memory gate；当前为实验预留。
    xlstm_memory_gate_enabled: bool = False
    # memory gate 模式；仅在启用 gate 时生效。
    xlstm_memory_gate_mode: str = "input_conditioned_eval"
    # xLSTMAssist 状态粒度策略。
    xlstm_memory_granularity: str = "selected_layers"
    # xLSTMAssist prefill 策略。
    xlstm_memory_prefill_policy: str = "chunkwise_recurrent_scan"
    # xLSTMAssist 状态从 prefill 延续到 decode 的策略。
    xlstm_memory_state_continuity: str = "prefill_to_decode"
    # xLSTMAssist 状态生命周期；必须绑定 request state pool。
    xlstm_memory_state_lifecycle: str = "request_bound_state_pool"
    # xLSTMAssist 状态更新策略；启用层确定性更新。
    xlstm_memory_update_policy: str = "deterministic_on_enabled_layers"
    # xLSTMAssist 状态遗忘/重置策略。
    xlstm_memory_state_policy: str = "window_decay_and_boundary_reset"
    # xLSTMAssist 状态窗口大小；None 表示不按固定窗口截断。
    xlstm_memory_state_window_size: int | None = None
    # xLSTMAssist decay 计数单位。
    xlstm_memory_decay_counter_unit: str = "tokens"
    # xLSTMAssist 每隔多少 token 触发一次 decay。
    xlstm_memory_state_decay_interval: int = 1024
    # xLSTMAssist decay 乘法因子。
    xlstm_memory_state_decay_factor: float = 0.95
    # 触发 xLSTMAssist reset 的事件来源。
    xlstm_memory_reset_trigger_mode: tuple[str, ...] = field(
        default_factory=lambda: ("boundary_metadata", "special_token", "session_event")
    )
    # reset 支持的语义边界类型。
    xlstm_memory_reset_boundary_policy: tuple[str, ...] = field(
        default_factory=lambda: ("document", "file", "chapter", "session_reset")
    )
    # 触发 reset 的特殊 token id 列表。
    xlstm_memory_boundary_token_ids: tuple[int, ...] = field(default_factory=tuple)
    # reset 动作；当前固定清零状态。
    xlstm_memory_reset_action: str = "zero_state"
    # xLSTMAssist 是否作为 MoE expert；必须为 false。
    xlstm_memory_as_expert: bool = False
    # xLSTM expert 数量；v2 中必须为 0。
    xlstm_expert_count: int = 0
    # xLSTM 是否作为独立主干 block；必须为 false。
    xlstm_as_standalone_block: bool = False
    # MoE router warmup 策略；当前只使用标准 balance 约束。
    moe_router_warmup_policy: str = "standard_balance_only"
    # 模型内部 dropout 概率。
    dropout_rate: float = 0.0
    # RoPE/LongRoPE2 的 base 参数。
    rope_base: float = 10000.0
    # 原始上下文窗口长度，LongRoPE2 mixed/static 以此为基准。
    original_max_len: int = 2048
    # LongRoPE2 目标上下文长度。
    longrope2_target_length: int = field(default_factory=lambda: int(GlobalConfig.inference_max_sequence_length))
    # LongRoPE2 long factors；可为标量、数组或 None。
    longrope2_long_factors: tuple[float, ...] | float | None = None
    # 当前 long factors 覆盖的最大序列长度。
    longrope2_factor_max_sequence_length: int | None = None
    # LongRoPE2 magnitude scaling 策略。
    longrope2_magnitude_scaling_policy: str = "su"
    # LongRoPE2 mscale 因子数组；为空时由位置编码实现推导。
    longrope2_mscale_factors: tuple[float, ...] | None = None
    # 训练侧 LongRoPE2 embedding mode。
    longrope2_train_embedding_mode: str = LONGROPE2_MIXED_EMBEDDING_MODE
    # 推理侧 LongRoPE2 embedding mode。
    longrope2_inference_embedding_mode: str = LONGROPE2_MIXED_EMBEDDING_MODE
    # mixed 模式下使用原始 RoPE 的窗口大小；为空时使用 original_max_len。
    longrope2_mixed_original_window: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "model_size_preset", str(self.model_size_preset))
        object.__setattr__(self, "default_model_size_preset", str(self.default_model_size_preset))
        object.__setattr__(self, "parameter_count_policy", str(self.parameter_count_policy))
        object.__setattr__(
            self,
            "parameter_count_modes",
            tuple(str(value) for value in _as_tuple(self.parameter_count_modes)),
        )
        object.__setattr__(self, "architecture_version", str(self.architecture_version))
        object.__setattr__(self, "block_type", str(self.block_type))
        object.__setattr__(self, "sequence_mixer_mode", str(self.sequence_mixer_mode))
        object.__setattr__(self, "num_layers", int(self.num_layers))
        object.__setattr__(self, "num_heads", int(self.num_heads))
        object.__setattr__(self, "num_kv_heads", int(self.num_kv_heads))
        object.__setattr__(self, "head_dim", int(self.head_dim))
        object.__setattr__(self, "cla_share_every_n_layers", int(self.cla_share_every_n_layers))
        object.__setattr__(self, "attention_backend_policy", str(self.attention_backend_policy))
        object.__setattr__(
            self,
            "attention_backend_priority",
            tuple(str(value) for value in _as_tuple(self.attention_backend_priority)),
        )
        object.__setattr__(self, "attention_window_size", int(self.attention_window_size))
        object.__setattr__(self, "attention_is_causal", bool(self.attention_is_causal))
        object.__setattr__(self, "attention_position_encoding", str(self.attention_position_encoding))
        object.__setattr__(self, "cache_backend", str(self.cache_backend))
        object.__setattr__(self, "kv_cache_scope", str(self.kv_cache_scope))
        object.__setattr__(self, "page_block_size", int(self.page_block_size))
        object.__setattr__(self, "retnet_assist_enabled", bool(self.retnet_assist_enabled))
        object.__setattr__(self, "retnet_assist_mode", str(self.retnet_assist_mode))
        object.__setattr__(self, "retnet_assist_layers", str(self.retnet_assist_layers))
        object.__setattr__(self, "retnet_parameter_sharing", str(self.retnet_parameter_sharing))
        object.__setattr__(self, "retnet_state_sharing", str(self.retnet_state_sharing))
        object.__setattr__(self, "retnet_prefill_scan_policy", str(self.retnet_prefill_scan_policy))
        object.__setattr__(self, "retnet_sequence_parallel_policy", str(self.retnet_sequence_parallel_policy))
        object.__setattr__(
            self,
            "retnet_sp_handoff_metrics_enabled",
            bool(self.retnet_sp_handoff_metrics_enabled),
        )
        object.__setattr__(self, "retnet_state_lifecycle", str(self.retnet_state_lifecycle))
        object.__setattr__(self, "retnet_state_dim", int(self.retnet_state_dim))
        object.__setattr__(self, "retnet_adapter_rank", int(self.retnet_adapter_rank))
        object.__setattr__(
            self,
            "retnet_adapter_target",
            tuple(str(value) for value in _as_tuple(self.retnet_adapter_target)),
        )
        object.__setattr__(self, "retnet_adapter_alpha_q_init", float(self.retnet_adapter_alpha_q_init))
        object.__setattr__(self, "retnet_adapter_alpha_q_dtype", str(self.retnet_adapter_alpha_q_dtype))
        object.__setattr__(
            self,
            "retnet_adapter_alpha_q_trainable",
            bool(self.retnet_adapter_alpha_q_trainable),
        )
        object.__setattr__(self, "retnet_adapter_alpha_k_init", float(self.retnet_adapter_alpha_k_init))
        object.__setattr__(self, "retnet_adapter_alpha_k_dtype", str(self.retnet_adapter_alpha_k_dtype))
        object.__setattr__(
            self,
            "retnet_adapter_alpha_k_trainable",
            bool(self.retnet_adapter_alpha_k_trainable),
        )
        object.__setattr__(self, "retnet_k_adapter_enabled", bool(self.retnet_k_adapter_enabled))
        object.__setattr__(
            self,
            "retnet_context_adapter_enabled",
            bool(self.retnet_context_adapter_enabled),
        )
        object.__setattr__(self, "retnet_context_adapter_alpha", float(self.retnet_context_adapter_alpha))
        object.__setattr__(self, "retnet_enters_paged_kv", bool(self.retnet_enters_paged_kv))
        object.__setattr__(self, "retnet_kv_replacement", bool(self.retnet_kv_replacement))
        object.__setattr__(
            self,
            "attention_logit_bias_from_retnet",
            bool(self.attention_logit_bias_from_retnet),
        )
        object.__setattr__(self, "ffn_type", str(self.ffn_type))
        object.__setattr__(self, "moe_num_experts", int(self.moe_num_experts))
        object.__setattr__(self, "moe_top_k", int(self.moe_top_k))
        object.__setattr__(self, "moe_router_dtype", str(self.moe_router_dtype))
        object.__setattr__(
            self,
            "moe_load_balance_loss_enabled",
            bool(self.moe_load_balance_loss_enabled),
        )
        object.__setattr__(self, "moe_router_z_loss_enabled", bool(self.moe_router_z_loss_enabled))
        object.__setattr__(self, "moe_router_input_mode", str(self.moe_router_input_mode))
        object.__setattr__(self, "xlstm_memory_enabled", bool(self.xlstm_memory_enabled))
        object.__setattr__(self, "xlstm_memory_mode", str(self.xlstm_memory_mode))
        object.__setattr__(self, "xlstm_memory_layers", str(self.xlstm_memory_layers))
        selected_layers = tuple(int(value) for value in (_as_tuple(self.xlstm_memory_selected_layers) or ()))
        if len(set(selected_layers)) != len(selected_layers):
            raise ValueError("xlstm_memory_selected_layers 不能包含重复层号。")
        if any(layer_index < 0 or layer_index >= self.num_layers for layer_index in selected_layers):
            raise ValueError("xlstm_memory_selected_layers 中的层号必须在 [0, num_layers) 范围内。")
        object.__setattr__(self, "xlstm_memory_selected_layers", tuple(sorted(selected_layers)))
        object.__setattr__(self, "xlstm_memory_state_dim", int(self.xlstm_memory_state_dim))
        object.__setattr__(self, "xlstm_memory_adapter_rank", int(self.xlstm_memory_adapter_rank))
        object.__setattr__(self, "xlstm_memory_adapter_beta_init", float(self.xlstm_memory_adapter_beta_init))
        object.__setattr__(self, "xlstm_memory_adapter_beta_dtype", str(self.xlstm_memory_adapter_beta_dtype))
        object.__setattr__(self, "xlstm_memory_adapter_beta_policy", str(self.xlstm_memory_adapter_beta_policy))
        object.__setattr__(
            self,
            "xlstm_memory_adapter_beta_range",
            tuple(float(value) for value in _as_tuple(self.xlstm_memory_adapter_beta_range)),
        )
        object.__setattr__(
            self,
            "xlstm_memory_as_router_target",
            bool(self.xlstm_memory_as_router_target),
        )
        object.__setattr__(self, "xlstm_memory_gate_enabled", bool(self.xlstm_memory_gate_enabled))
        object.__setattr__(self, "xlstm_memory_gate_mode", str(self.xlstm_memory_gate_mode))
        object.__setattr__(self, "xlstm_memory_granularity", str(self.xlstm_memory_granularity))
        object.__setattr__(self, "xlstm_memory_prefill_policy", str(self.xlstm_memory_prefill_policy))
        object.__setattr__(self, "xlstm_memory_state_continuity", str(self.xlstm_memory_state_continuity))
        object.__setattr__(self, "xlstm_memory_state_lifecycle", str(self.xlstm_memory_state_lifecycle))
        object.__setattr__(self, "xlstm_memory_update_policy", str(self.xlstm_memory_update_policy))
        object.__setattr__(self, "xlstm_memory_state_policy", str(self.xlstm_memory_state_policy))
        if self.xlstm_memory_state_window_size is not None:
            object.__setattr__(
                self,
                "xlstm_memory_state_window_size",
                int(self.xlstm_memory_state_window_size),
            )
        object.__setattr__(self, "xlstm_memory_decay_counter_unit", str(self.xlstm_memory_decay_counter_unit))
        object.__setattr__(
            self,
            "xlstm_memory_state_decay_interval",
            int(self.xlstm_memory_state_decay_interval),
        )
        object.__setattr__(
            self,
            "xlstm_memory_state_decay_factor",
            float(self.xlstm_memory_state_decay_factor),
        )
        object.__setattr__(
            self,
            "xlstm_memory_reset_trigger_mode",
            tuple(str(value) for value in _as_tuple(self.xlstm_memory_reset_trigger_mode)),
        )
        object.__setattr__(
            self,
            "xlstm_memory_reset_boundary_policy",
            tuple(str(value) for value in _as_tuple(self.xlstm_memory_reset_boundary_policy)),
        )
        object.__setattr__(
            self,
            "xlstm_memory_boundary_token_ids",
            tuple(int(value) for value in _as_tuple(self.xlstm_memory_boundary_token_ids)),
        )
        object.__setattr__(self, "xlstm_memory_reset_action", str(self.xlstm_memory_reset_action))
        object.__setattr__(self, "xlstm_memory_as_expert", bool(self.xlstm_memory_as_expert))
        object.__setattr__(self, "xlstm_expert_count", int(self.xlstm_expert_count))
        object.__setattr__(self, "xlstm_as_standalone_block", bool(self.xlstm_as_standalone_block))
        object.__setattr__(self, "moe_router_warmup_policy", str(self.moe_router_warmup_policy))
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
        self._validate_lpt_v2_fields()

    def _validate_lpt_v2_fields(self):
        """校验 LPT v2 配置骨架，避免 checkpoint 写入互相冲突的语义。"""
        if self.model_size_preset not in LPT_V2_MODEL_SIZE_PRESETS:
            raise ValueError(f"model_size_preset 必须是 {tuple(LPT_V2_MODEL_SIZE_PRESETS)} 之一。")
        if self.default_model_size_preset != DEFAULT_MODEL_SIZE_PRESET:
            raise ValueError(f"default_model_size_preset 必须是 {DEFAULT_MODEL_SIZE_PRESET}。")
        if self.parameter_count_policy != PARAMETER_COUNT_POLICY_MOE_AWARE:
            raise ValueError("parameter_count_policy 当前必须是 moe_aware。")
        if self.parameter_count_modes != PARAMETER_COUNT_MODES:
            raise ValueError(f"parameter_count_modes 必须严格为 {PARAMETER_COUNT_MODES}。")
        if self.architecture_version != LPT_V2_ARCHITECTURE_VERSION:
            raise ValueError(f"architecture_version 当前只支持 {LPT_V2_ARCHITECTURE_VERSION}。")
        if self.attention_backend_policy not in SUPPORTED_ATTENTION_BACKEND_POLICIES:
            raise ValueError(
                f"attention_backend_policy 必须是 {SUPPORTED_ATTENTION_BACKEND_POLICIES} 之一。"
            )
        unknown_backend_priority = sorted(
            set(self.attention_backend_priority) - set(SUPPORTED_ATTENTION_BACKENDS)
        )
        if unknown_backend_priority:
            raise ValueError(f"attention_backend_priority 包含未支持后端: {unknown_backend_priority}")
        if not self.attention_backend_priority:
            raise ValueError("attention_backend_priority 不能为空。")
        if self.attention_window_size <= 0:
            raise ValueError("attention_window_size 必须为正整数。")
        if self.attention_position_encoding != "longrope2":
            raise ValueError("attention_position_encoding 当前必须是 longrope2。")
        if self.cache_backend not in SUPPORTED_CACHE_BACKENDS:
            raise ValueError(f"cache_backend 必须是 {SUPPORTED_CACHE_BACKENDS} 之一。")
        if self.page_block_size <= 0:
            raise ValueError("page_block_size 必须为正整数。")
        if self.kv_cache_scope != "local_real_tokens_only":
            raise ValueError("kv_cache_scope 当前必须是 local_real_tokens_only。")

        if self.retnet_assist_mode not in {"q_adapter", "qk_adapter"}:
            raise ValueError("retnet_assist_mode 必须是 q_adapter 或 qk_adapter。")
        if self.retnet_parameter_sharing not in {"global", "group"}:
            raise ValueError("retnet_parameter_sharing 必须是 global 或 group。")
        if self.retnet_state_sharing not in {"group", "per_layer"}:
            raise ValueError("retnet_state_sharing 必须是 group 或 per_layer。")
        if self.retnet_state_lifecycle != "request_bound_state_pool":
            raise ValueError("retnet_state_lifecycle 必须是 request_bound_state_pool。")
        if self.retnet_state_dim <= 0:
            raise ValueError("retnet_state_dim 必须为正整数。")
        if self.retnet_adapter_rank <= 0:
            raise ValueError("retnet_adapter_rank 必须为正整数。")
        adapter_target = tuple(dict.fromkeys(str(value) for value in self.retnet_adapter_target))
        if set(adapter_target) == {"q"}:
            adapter_target = ("q",)
        elif set(adapter_target) == {"q", "k"}:
            adapter_target = ("q", "k")
        else:
            raise ValueError("retnet_adapter_target 必须是 ('q',) 或 ('q', 'k')。")
        object.__setattr__(self, "retnet_adapter_target", adapter_target)
        if self.retnet_adapter_alpha_q_init < 0:
            raise ValueError("retnet_adapter_alpha_q_init 不能为负数。")
        if self.retnet_adapter_alpha_q_dtype != "fp32":
            raise ValueError("retnet_adapter_alpha_q_dtype 当前必须是 fp32。")
        if self.retnet_adapter_alpha_k_init < 0:
            raise ValueError("retnet_adapter_alpha_k_init 不能为负数。")
        if self.retnet_adapter_alpha_k_dtype != "fp32":
            raise ValueError("retnet_adapter_alpha_k_dtype 当前必须是 fp32。")
        if self.retnet_k_adapter_enabled != ("k" in self.retnet_adapter_target):
            raise ValueError("retnet_k_adapter_enabled 必须与 retnet_adapter_target 是否包含 k 一致。")
        if self.retnet_assist_mode == "q_adapter" and self.retnet_k_adapter_enabled:
            raise ValueError("retnet_assist_mode=q_adapter 时不能启用 K adapter。")
        if self.retnet_assist_mode == "qk_adapter" and not self.retnet_k_adapter_enabled:
            raise ValueError("retnet_assist_mode=qk_adapter 时必须启用 K adapter。")
        if self.retnet_enters_paged_kv or self.retnet_kv_replacement:
            raise ValueError("RetNetAssist 不能进入或替换 Paged KV。")
        if self.attention_logit_bias_from_retnet:
            raise ValueError("P0 定型配置禁止 attention_logit_bias_from_retnet。")

        if self.ffn_type != "memory_augmented_swiglu_moe":
            raise ValueError("ffn_type 当前必须是 memory_augmented_swiglu_moe。")
        if self.moe_num_experts <= 0:
            raise ValueError("moe_num_experts 必须为正整数。")
        if self.moe_top_k <= 0 or self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k 必须在 1 到 moe_num_experts 之间。")
        if self.moe_router_dtype != "fp32":
            raise ValueError("moe_router_dtype 当前必须是 fp32。")
        if self.moe_router_input_mode not in {"memory_augmented_input", "ffn_norm_only_eval"}:
            raise ValueError("moe_router_input_mode 必须是 memory_augmented_input 或 ffn_norm_only_eval。")

        if self.xlstm_memory_state_dim <= 0:
            raise ValueError("xlstm_memory_state_dim 必须为正整数。")
        if self.xlstm_memory_adapter_rank <= 0:
            raise ValueError("xlstm_memory_adapter_rank 必须为正整数。")
        if self.xlstm_memory_adapter_beta_init < 0:
            raise ValueError("xlstm_memory_adapter_beta_init 不能为负数。")
        if self.xlstm_memory_adapter_beta_dtype != "fp32":
            raise ValueError("xlstm_memory_adapter_beta_dtype 当前必须是 fp32。")
        if self.xlstm_memory_adapter_beta_policy != "fp32_sigmoid_clamped":
            raise ValueError("xlstm_memory_adapter_beta_policy 当前必须是 fp32_sigmoid_clamped。")
        if len(self.xlstm_memory_adapter_beta_range) != 2:
            raise ValueError("xlstm_memory_adapter_beta_range 必须包含 min/max 两个值。")
        beta_min, beta_max = self.xlstm_memory_adapter_beta_range
        if beta_min <= 0 or beta_max < beta_min:
            raise ValueError("xlstm_memory_adapter_beta_range 必须满足 0 < min <= max。")
        if self.xlstm_memory_as_router_target:
            raise ValueError("xLSTMAssist 不能作为 MoE router target。")
        if self.xlstm_memory_as_expert or self.xlstm_expert_count != 0:
            raise ValueError("xLSTMAssist 不能作为 MoE expert。")
        if self.xlstm_as_standalone_block:
            raise ValueError("xLSTMAssist 不能作为独立主干 block。")
        if self.xlstm_memory_state_lifecycle != "request_bound_state_pool":
            raise ValueError("xlstm_memory_state_lifecycle 必须是 request_bound_state_pool。")
        if self.xlstm_memory_update_policy != "deterministic_on_enabled_layers":
            raise ValueError("xlstm_memory_update_policy 必须是 deterministic_on_enabled_layers。")
        if self.xlstm_memory_reset_action != "zero_state":
            raise ValueError("xlstm_memory_reset_action 当前必须是 zero_state。")
        if self.xlstm_memory_state_decay_interval <= 0:
            raise ValueError("xlstm_memory_state_decay_interval 必须为正整数。")
        if not 0 < self.xlstm_memory_state_decay_factor <= 1:
            raise ValueError("xlstm_memory_state_decay_factor 必须在 (0, 1] 范围内。")
        if self.xlstm_memory_state_window_size is not None and self.xlstm_memory_state_window_size <= 0:
            raise ValueError("xlstm_memory_state_window_size 必须为正整数或 None。")
        if not is_valid_xlstm_memory_layer_policy(self.xlstm_memory_layers):
            raise ValueError(
                "xlstm_memory_layers 必须是 disabled、all_layers、every_n_layers、"
                "every_<正整数>_layers 或 selected_layers。"
            )
        if self.xlstm_memory_layers == "selected_layers":
            if not self.xlstm_memory_selected_layers:
                raise ValueError("xlstm_memory_layers=selected_layers 时必须配置 xlstm_memory_selected_layers。")
        elif self.xlstm_memory_selected_layers:
            raise ValueError("xlstm_memory_selected_layers 只能在 selected_layers 策略下配置。")
        if self.xlstm_memory_enabled:
            if self.xlstm_memory_layers == "disabled":
                raise ValueError("xlstm_memory_enabled=true 时 xlstm_memory_layers 不能是 disabled。")
            if self.moe_router_input_mode not in {"memory_augmented_input", "ffn_norm_only_eval"}:
                raise ValueError("启用 xLSTMAssist 时 moe_router_input_mode 必须是 memory_augmented_input 或 ffn_norm_only_eval。")
        else:
            if self.xlstm_memory_layers != "disabled":
                raise ValueError("xlstm_memory_enabled=false 时 xlstm_memory_layers 必须是 disabled。")
            if self.moe_router_input_mode != "ffn_norm_only_eval":
                raise ValueError("关闭 xLSTMAssist 时 moe_router_input_mode 必须是 ffn_norm_only_eval。")
            if self.xlstm_memory_gate_enabled:
                raise ValueError("xlstm_memory_gate_enabled=true 时必须启用 xLSTMAssist。")
        if self.xlstm_memory_gate_enabled and self.xlstm_memory_gate_mode != "input_conditioned_eval":
            raise ValueError("xlstm_memory_gate_mode 当前仅支持 input_conditioned_eval。")

        if self.block_type != LPT_V2_BLOCK_TYPE:
            raise ValueError(f"lpt_v2 的 block_type 必须是 {LPT_V2_BLOCK_TYPE}。")
        if self.sequence_mixer_mode != LPT_V2_SEQUENCE_MIXER_MODE:
            raise ValueError(f"lpt_v2 的 sequence_mixer_mode 必须是 {LPT_V2_SEQUENCE_MIXER_MODE}。")
        if self.cla_share_every_n_layers != 1:
            raise ValueError("lpt_v2 固化 cla_share_every_n_layers=1，不共享 Attention KV。")
        if any(block_type != ATTENTION_BLOCK_TYPE for block_type in self.layer_block_types):
            raise ValueError("lpt_v2 的主干 layer_block_types 当前必须全部是 attention。")

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

    @classmethod
    def from_preset(cls, preset, **overrides):
        """按 LPT v2 规格预设展开为完整 ModelConfig。"""
        # payload 是 preset 展开的完整配置字典，随后再应用显式覆盖项。
        payload = expand_lpt_v2_model_config_preset(preset)
        payload.update(overrides)
        return cls(**payload)

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
        # normalized_payload 保存可被 dataclass 构造器直接接收的规范化字段。
        normalized_payload = dict(payload)
        for field_name in _LIST_LIKE_MODEL_CONFIG_FIELDS:
            if field_name in normalized_payload and isinstance(normalized_payload[field_name], list):
                normalized_payload[field_name] = tuple(normalized_payload[field_name])
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


def expand_lpt_v2_model_config_preset(preset=DEFAULT_MODEL_SIZE_PRESET):
    """把 LPT v2 规格预设展开为完整配置字段字典。"""
    # preset_name 是外部传入的规格名称，先转成字符串再校验。
    preset_name = str(preset)
    if preset_name not in LPT_V2_MODEL_SIZE_PRESETS:
        raise ValueError(f"未知 LPT v2 model_size_preset: {preset_name}")
    # preset_payload 是该规格的核心尺寸字段，返回时补齐 v2 必需结构字段。
    preset_payload = dict(LPT_V2_MODEL_SIZE_PRESETS[preset_name])
    # num_layers 用于生成与层数一致的 v2 attention-only layer_block_types。
    num_layers = int(preset_payload["num_layers"])
    return {
        "model_size_preset": preset_name,
        "default_model_size_preset": DEFAULT_MODEL_SIZE_PRESET,
        "architecture_version": LPT_V2_ARCHITECTURE_VERSION,
        "block_type": LPT_V2_BLOCK_TYPE,
        "sequence_mixer_mode": LPT_V2_SEQUENCE_MIXER_MODE,
        "layer_block_types": tuple(ATTENTION_BLOCK_TYPE for _ in range(num_layers)),
        "cla_share_every_n_layers": 1,
        **preset_payload,
    }


def build_lpt_v2_model_config_preset(preset=DEFAULT_MODEL_SIZE_PRESET, **overrides):
    """构造某个 LPT v2 规格预设对应的完整 ModelConfig。"""
    return ModelConfig.from_preset(preset, **overrides)


def build_model_config_from_checkpoint(checkpoint):
    """从 checkpoint 中恢复模型配置。"""
    if checkpoint is None:
        raise ValueError("checkpoint 不能为空。")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 必须是字典。")

    # model_config_schema_version 决定是否允许从 checkpoint 恢复结构配置。
    model_config_schema_version = checkpoint.get("model_config_schema_version")
    if model_config_schema_version is None:
        raise ValueError("checkpoint 缺少 model_config_schema_version。")
    if model_config_schema_version != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "不支持的 model_config_schema_version: "
            f"{model_config_schema_version}，当前仅支持 {MODEL_CONFIG_SCHEMA_VERSION}。"
        )

    # model_config_payload 是 checkpoint 内保存的完整 ModelConfig 快照。
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
    # factor_path 指向外部搜索或手工导出的 factors 文件。
    factor_path = Path(path)
    if not factor_path.exists():
        raise FileNotFoundError(f"未找到 LongRoPE2 long factors 文件: {factor_path}")

    # raw_text 同时兼容逗号分隔和换行分隔的 factors 文件。
    raw_text = factor_path.read_text(encoding="utf-8").replace(",", "\n")
    # factors 是最终写入 ModelConfig 的 LongRoPE2 因子数组。
    factors = tuple(float(value) for value in raw_text.split() if value.strip())
    if not factors:
        raise ValueError(f"LongRoPE2 long factors 文件为空: {factor_path}")
    if any(factor <= 0 for factor in factors):
        raise ValueError("LongRoPE2 long factors 必须全部大于 0。")
    return factors


def build_longrope2_uniform_factors(config, sequence_length):
    """基于数据最长 token 长度生成一组确定性 bootstrap factors。"""
    # rotary_dims 是每个 head 参与 RoPE 的成对旋转维度数量。
    rotary_dims = int(config.head_dim) // 2
    if rotary_dims <= 0:
        raise ValueError("head_dim 必须至少包含一组 rotary 维度。")
    # coverage_length 取数据最长长度和目标长度的较大值，避免因子覆盖不足。
    coverage_length = max(int(sequence_length), int(config.longrope2_target_length))
    # factor 是 uniform bootstrap 缩放倍数。
    factor = max(float(coverage_length) / float(config.original_max_len), 1.0)
    return tuple(float(factor) for _ in range(rotary_dims))
