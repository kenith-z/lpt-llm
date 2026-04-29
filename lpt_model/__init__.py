"""LPT 模型本体包。

这个包只放与模型结构本身直接相关的代码：
- 位置编码
- Transformer 主体
"""

from .model import (
    ATTENTION_BLOCK_TYPE,
    LPT,
    LayerState,
    LayerSpec,
    MODEL_ARCHITECTURE_METADATA_KEYS,
    ModelConfig,
    ModernAttention,
    RETNET_BLOCK_TYPE,
    RetNetBlock,
    extract_checkpoint_architecture_metadata,
    get_model_architecture_metadata,
    list_architecture_mismatches,
)
from .position_encoding import LongRoPE2RotaryPositionEncoding, build_rotary_position_encoding


