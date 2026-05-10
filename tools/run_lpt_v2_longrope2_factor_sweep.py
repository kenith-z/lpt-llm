"""运行 LPT v2 LongRoPE2 候选因子 sweep。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_eval import run_lpt_v2_longrope2_factor_sweep
from lpt_eval.utils import write_json_report, write_text_report


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


def _parse_named_value(raw_value, *, value_parser, option_name):
    if "=" not in str(raw_value):
        raise argparse.ArgumentTypeError(f"{option_name} 必须使用 name=value 格式。")
    name, value = str(raw_value).split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise argparse.ArgumentTypeError(f"{option_name} 的 name 和 value 都不能为空。")
    return name, value_parser(value)


def _parse_uniform_factor(raw_value):
    return _parse_named_value(raw_value, value_parser=float, option_name="--uniform-factor")


def _parse_factors_file(raw_value):
    return _parse_named_value(raw_value, value_parser=Path, option_name="--factors-file")


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 LongRoPE2 候选因子 sweep。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="真实 LPT v2 checkpoint。")
    parser.add_argument("--seq-lens", nargs="+", default=("2052",), help="测试序列长度列表，支持空格或逗号分隔。")
    parser.add_argument("--attention-window-sizes", nargs="+", default=("2048",), help="局部 attention 窗口列表。")
    parser.add_argument("--needle-depths", nargs="+", default=("0.2", "0.5", "0.8"), help="needle 插入深度列表。")
    parser.add_argument("--no-current", action="store_true", help="不评估 checkpoint 当前保存的 factors。")
    parser.add_argument("--no-bootstrap", action="store_true", help="不评估 v2 bootstrap 规则生成的 factors。")
    parser.add_argument("--bootstrap-sequence-length", type=int, default=None, help="bootstrap factors 覆盖长度。")
    parser.add_argument("--uniform-factor", action="append", default=None, help="追加统一缩放候选，格式 name=value，可重复。")
    parser.add_argument("--factors-file", action="append", default=None, help="追加 factors 文件候选，格式 name=path，可重复。")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default="auto", help="auto/fp32/fp16/bf16。")
    parser.add_argument("--output-json", help="JSON 报告输出路径。")
    parser.add_argument("--output-md", help="Markdown 报告输出路径。")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        uniform_factor_candidates = tuple(
            _parse_uniform_factor(raw_value)
            for raw_value in (args.uniform_factor or ())
        )
        factor_file_candidates = tuple(
            _parse_factors_file(raw_value)
            for raw_value in (args.factors_file or ())
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    report = run_lpt_v2_longrope2_factor_sweep(
        checkpoint_path=args.checkpoint,
        sequence_lengths=_parse_int_values(args.seq_lens),
        attention_window_sizes=_parse_int_values(args.attention_window_sizes),
        needle_depths=_parse_float_values(args.needle_depths),
        include_current=not args.no_current,
        include_bootstrap=not args.no_bootstrap,
        bootstrap_sequence_length=args.bootstrap_sequence_length,
        uniform_factor_candidates=uniform_factor_candidates,
        factor_file_candidates=factor_file_candidates,
        device=args.device,
        dtype=args.dtype,
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
