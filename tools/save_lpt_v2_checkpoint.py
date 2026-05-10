"""保存一个 LPT v2 checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import InferenceSmokeConfig
from lpt_eval import build_lpt_v2_profile_config
from lpt_model import LPTV2, save_lpt_v2_checkpoint


DEFAULT_CONFIG = InferenceSmokeConfig()


def build_parser():
    parser = argparse.ArgumentParser(description="保存 LPT v2 checkpoint。")
    parser.add_argument("--output", required=True, help="checkpoint 输出路径。")
    parser.add_argument("--profile", default=DEFAULT_CONFIG.profile, help="运行 profile。")
    parser.add_argument("--preset", default=DEFAULT_CONFIG.preset, help="模型规格 preset。")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocabulary_size, help="词表大小。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = build_lpt_v2_profile_config(args.profile, preset=args.preset)
    model = LPTV2(args.vocab_size, config)
    target_path = save_lpt_v2_checkpoint(
        model,
        args.output,
        extra_metadata={"source": "tools/save_lpt_v2_checkpoint.py"},
    )
    print(f"checkpoint_saved={target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
