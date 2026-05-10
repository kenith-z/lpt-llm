"""兼容导出：运行 profile 配置已迁移到 lpt_config.profiles。"""

from lpt_config.profiles import (
    LPT_V2_ASSIST_PROFILE,
    LPT_V2_BASE_PROFILE,
    LPT_V2_BASELINE_PROFILES,
    LPT_V2_BOOTSTRAP_PROFILE,
    LPT_V2_MEMORY_PROFILE,
    LPT_V2_PAGED_KV_PROFILE,
    LPT_V2_SDPA_LOCAL_PROFILE,
    build_lpt_v2_profile_config,
    parse_profile_list,
)
