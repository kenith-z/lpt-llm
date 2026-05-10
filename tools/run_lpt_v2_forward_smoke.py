"""运行 LPT v2 checkpoint forward smoke。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_eval import run_lpt_v2_forward_smoke_report
from lpt_eval.utils import write_json_report, write_text_report


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 checkpoint forward smoke。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="真实 LPT v2 checkpoint。")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size。")
    parser.add_argument("--seq-len", type=int, default=32, help="输入序列长度。")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default="auto", help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=20260503, help="随机种子。")
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="前向时启用 KV cache；默认关闭以验证训练 forward 路径。",
    )
    parser.add_argument("--output-json", help="JSON 报告输出路径。")
    parser.add_argument("--output-md", help="Markdown 报告输出路径。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_lpt_v2_forward_smoke_report(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        use_kv_cache=args.use_kv_cache,
    )
    if args.output_json:
        write_json_report(args.output_json, report.to_dict())
    if args.output_md:
        write_text_report(args.output_md, report.to_markdown())
    if not args.output_json and not args.output_md:
        print(report.to_markdown())
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
