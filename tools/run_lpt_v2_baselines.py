"""运行 LPT v2 profile 基线报告。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_eval import run_lpt_v2_baselines
from lpt_eval.utils import write_json_report, write_text_report
from lpt_config import BaselineEvalConfig


DEFAULT_CONFIG = BaselineEvalConfig()


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 profile 基线报告。")
    parser.add_argument("--profiles", default=DEFAULT_CONFIG.profiles, help="逗号分隔 profile 列表，默认 all。")
    parser.add_argument("--preset", default=DEFAULT_CONFIG.preset, help="模型规格 preset。")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocabulary_size, help="测试词表大小。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size, help="batch size。")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.sequence_length, help="prefill 序列长度。")
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_CONFIG.decode_steps, help="decode 步数。")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG.dtype, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="随机种子。")
    parser.add_argument("--output-json", help="JSON 报告输出路径。")
    parser.add_argument("--output-md", help="Markdown 报告输出路径。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_lpt_v2_baselines(
        profiles=args.profiles,
        preset=args.preset,
        vocabulary_size=args.vocab_size,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        decode_steps=args.decode_steps,
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
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
