"""运行 LPT v2 长上下文准入报告。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_eval import run_lpt_v2_long_context_admission
from lpt_eval.utils import write_json_report, write_text_report
from lpt_config import LongContextEvalConfig


DEFAULT_CONFIG = LongContextEvalConfig()


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 长上下文准入报告。")
    parser.add_argument("--preset", default=DEFAULT_CONFIG.preset, help="模型规格 preset。")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocabulary_size, help="测试词表大小。")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.sequence_length, help="测试序列长度，默认 window*2+4。")
    parser.add_argument("--attention-window-size", type=int, default=DEFAULT_CONFIG.attention_window_size, help="局部 attention 窗口；加载 checkpoint 时可省略以使用 checkpoint 配置。")
    parser.add_argument("--needle-depth", type=float, default=0.0, help="needle 插入深度，0 表示靠前，1 表示靠近末尾。")
    parser.add_argument("--checkpoint", type=Path, default=None, help="真实 LPT v2 checkpoint，指定后运行 checkpoint 准入路径。")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG.dtype, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="随机种子。")
    parser.add_argument("--output-json", help="JSON 报告输出路径。")
    parser.add_argument("--output-md", help="Markdown 报告输出路径。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_lpt_v2_long_context_admission(
        preset=args.preset,
        vocabulary_size=args.vocab_size,
        sequence_length=args.seq_len,
        attention_window_size=args.attention_window_size,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        needle_depth=args.needle_depth,
    )
    if args.output_json:
        write_json_report(args.output_json, report.to_dict())
    if args.output_md:
        write_text_report(args.output_md, report.to_markdown())
    if not args.output_json and not args.output_md:
        print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
