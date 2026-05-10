"""运行 LPT v2 长上下文评测套件。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import LongContextEvalConfig
from lpt_eval import run_lpt_v2_long_context_suite
from lpt_eval.utils import write_json_report, write_text_report


DEFAULT_CONFIG = LongContextEvalConfig()


def _parse_int_values(raw_values):
    values = []
    for raw_value in raw_values:
        values.extend(int(value.strip()) for value in str(raw_value).split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个整数。")
    return tuple(values)


def _parse_float_values(raw_values):
    values = []
    for raw_value in raw_values:
        values.extend(float(value.strip()) for value in str(raw_value).split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("至少需要一个浮点数。")
    return tuple(values)


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 长上下文多长度、多 depth 评测套件。")
    parser.add_argument("--preset", default=DEFAULT_CONFIG.preset, help="模型规格 preset。")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocabulary_size, help="测试词表大小。")
    parser.add_argument("--checkpoint", type=Path, default=None, help="真实 LPT v2 checkpoint。")
    parser.add_argument("--seq-lens", nargs="+", default=None, help="测试序列长度列表，支持空格或逗号分隔。")
    parser.add_argument("--attention-window-sizes", nargs="+", default=(str(DEFAULT_CONFIG.attention_window_size),), help="局部 attention 窗口列表。")
    parser.add_argument("--needle-depths", nargs="+", default=("0.2", "0.5", "0.8"), help="needle 插入深度列表。")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG.dtype, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="随机种子。")
    parser.add_argument("--output-json", help="JSON 报告输出路径。")
    parser.add_argument("--output-md", help="Markdown 报告输出路径。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sequence_lengths = None if args.seq_lens is None else _parse_int_values(args.seq_lens)
    report = run_lpt_v2_long_context_suite(
        preset=args.preset,
        vocabulary_size=args.vocab_size,
        sequence_lengths=sequence_lengths,
        attention_window_sizes=_parse_int_values(args.attention_window_sizes),
        needle_depths=_parse_float_values(args.needle_depths),
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
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
