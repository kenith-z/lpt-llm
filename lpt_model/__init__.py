"""LPT v2 模型本体包。"""

from .common import RMSNorm, SDPA_SUPPORTS_GQA, SwiGLU, build_position_ids
from .checkpoint_v2 import (
    LPT_V2_CHECKPOINT_FORMAT,
    LPT_V2_CHECKPOINT_SCHEMA_VERSION,
    LoadedLPTV2Checkpoint,
    build_lpt_v2_checkpoint_payload,
    load_lpt_v2_checkpoint,
    save_lpt_v2_checkpoint,
    validate_lpt_v2_checkpoint_payload,
)
from .model_v2 import (
    LPTBlockV2,
    LPTV2,
    LocalAttentionMixerV2,
    PagedKVCache,
    QOnlyRetNetAdapter,
    RetNetContextAdapter,
    SharedRetNetAssist,
    SwiGLUMoE,
)
from .parameter_count import MoEAwareParameterReport, estimate_moe_aware_parameter_counts
from .position_encoding import LongRoPE2RotaryPositionEncoding, build_rotary_position_encoding
from .state_v2 import (
    ATTENTION_LAYER_STATE_V2_TYPE,
    MOE_LAYER_STATE_TYPE,
    RETNET_ASSIST_STATE_TYPE,
    XLSTM_MEMORY_STATE_TYPE,
    AttentionLayerState,
    LayerStateV2,
    MoELayerState,
    PagedKVReference,
    RetNetAssistState,
    StateReleaseMetadata,
    xLSTMMemoryState,
)
from .state_pool_v2 import (
    RETNET_POOL_PHASE_DECODE,
    RETNET_POOL_PHASE_PREFILL,
    RETNET_POOL_PHASE_PREEMPTED,
    RETNET_POOL_PHASE_RELEASED,
    RETNET_POOL_PHASE_RESET,
    XLSTM_POOL_PHASE_DECODE,
    XLSTM_POOL_PHASE_PREFILL,
    XLSTM_POOL_PHASE_PREEMPTED,
    XLSTM_POOL_PHASE_RELEASED,
    XLSTM_POOL_PHASE_RESET,
    RetNetAssistPoolMetadata,
    RetNetAssistStatePool,
    xLSTMMemoryPoolMetadata,
    xLSTMMemoryStatePool,
)
