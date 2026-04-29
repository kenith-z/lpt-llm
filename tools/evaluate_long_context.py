"""运行长上下文评测套件。"""

from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_evaluation import (
    LongContextEvaluationConfig,
    evaluate_long_context_suite,
    save_long_context_report,
)


def _parse_int_list(raw_values):
    return tuple(int(value) for value in raw_values)


def _parse_float_list(raw_values):
    return tuple(float(value) for value in raw_values)


def build_argument_parser():
    parser = ArgumentParser(description="运行长上下文评测：needle / PPL / QA-检索。")
    parser.add_argument(
        "--model",
        default="chat_sft",
        choices=("text_pretrain", "chat_sft", "chat_lora"),
        help="指定评测模型类型。",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="指定待评测 checkpoint 根路径（无需 .pth 后缀）。",
    )
    parser.add_argument(
        "--lora-base-source",
        default="text_pretrain",
        choices=("text_pretrain", "chat_sft"),
        help="chat_lora 模式下使用的基座来源。",
    )
    parser.add_argument(
        "--cache-strategy",
        default="session_rebuild",
        choices=("session_rebuild", "full_recompute"),
        help="生成型评测采用的缓存策略。",
    )
    parser.add_argument(
        "--text-manifest",
        type=Path,
        default=Path("data/manifests/text_pretrain.json"),
        help="长文本 PPL 使用的 text manifest。",
    )
    parser.add_argument(
        "--needle-lengths",
        nargs="+",
        default=("2048", "4096"),
        help="needle-in-a-haystack 的目标 token 长度列表。",
    )
    parser.add_argument(
        "--needle-depths",
        nargs="+",
        default=("0.2", "0.5", "0.8"),
        help="needle 的插入深度列表，范围 0~1。",
    )
    parser.add_argument(
        "--retrieval-lengths",
        nargs="+",
        default=("2048", "4096"),
        help="QA/检索评测的目标 token 长度列表。",
    )
    parser.add_argument(
        "--ppl-lengths",
        nargs="+",
        default=("1024", "2048"),
        help="长文本 PPL 的窗口长度列表。",
    )
    parser.add_argument(
        "--ppl-max-windows",
        type=int,
        default=4,
        help="每个 PPL 窗口长度最多评测多少个窗口。",
    )
    parser.add_argument(
        "--max-generation-tokens",
        type=int,
        default=48,
        help="生成型任务的最大输出 token 数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="评测随机种子。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果输出目录；为空时自动落到 checkpoint 对应 artifact 下。",
    )
    parser.add_argument(
        "--output-format",
        default="both",
        choices=("json", "markdown", "md", "both"),
        help="结果输出格式。",
    )
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = LongContextEvaluationConfig(
        model_type=args.model,
        checkpoint_root=None if args.checkpoint_root is None else str(args.checkpoint_root),
        lora_base_source=args.lora_base_source,
        cache_strategy=args.cache_strategy,
        output_format=args.output_format,
        output_dir=None if args.output_dir is None else str(args.output_dir),
        text_manifest_path=str(args.text_manifest),
        needle_lengths=_parse_int_list(args.needle_lengths),
        needle_depths=_parse_float_list(args.needle_depths),
        retrieval_lengths=_parse_int_list(args.retrieval_lengths),
        ppl_lengths=_parse_int_list(args.ppl_lengths),
        ppl_max_windows=args.ppl_max_windows,
        seed=args.seed,
        max_generation_tokens=args.max_generation_tokens,
    )
    report, checkpoint_root = evaluate_long_context_suite(config)
    saved_paths = save_long_context_report(
        report,
        checkpoint_root,
        output_dir=config.output_dir,
        output_format=config.output_format,
    )
    print("长上下文评测完成。")
    for label, path in saved_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
