"""项目配置包。

这个包只放配置相关的代码：
- 全局配置与生成配置
"""

from .config import (
    ChatLoRATrainingConfig,
    ChatSFTTrainingConfig,
    GenerationConfig,
    GlobalConfig,
    LoRAConfig,
    TextPretrainingConfig,
)
from .model_config import (
    LONGROPE2_DYNAMIC_EMBEDDING_MODE,
    LONGROPE2_EMBEDDING_MODES,
    LONGROPE2_MIXED_EMBEDDING_MODE,
    LONGROPE2_STATIC_EMBEDDING_MODE,
    MODEL_CONFIG_SCHEMA_VERSION,
    ModelConfig,
    build_model_config_from_checkpoint,
    build_longrope2_uniform_factors,
    load_longrope2_factors_file,
    load_model_config_json,
    model_config_snapshot_path,
    normalize_model_config,
)
