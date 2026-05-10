"""严格校验 LPT v2 checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_model import load_lpt_v2_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description="严格校验 LPT v2 checkpoint。")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径。")
    parser.add_argument("--map-location", default="cpu", help="torch.load map_location。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    loaded = load_lpt_v2_checkpoint(args.checkpoint, map_location=args.map_location, strict=True)
    print(
        "checkpoint_ok "
        f"architecture={loaded.model.config.architecture_version} "
        f"preset={loaded.model.config.model_size_preset} "
        f"layers={loaded.model.config.num_layers}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
