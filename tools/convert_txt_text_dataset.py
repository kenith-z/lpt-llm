"""把纯文本 TXT 数据集转换为结构化 text JSONL。"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record
from lpt_config import TOKENIZER_PATH
from tools.convert_raw_text_jsonl import _chunk_text_by_lines, _load_tokenizer


def _resolve_source_name(source_name, input_path, *, root_path=None):
    """生成结构化记录 source 字段。"""
    if root_path is None:
        return source_name or input_path.stem
    relative_stem = input_path.relative_to(root_path).with_suffix("").as_posix()
    if source_name:
        return f"{source_name}/{relative_stem}"
    return relative_stem


def convert_txt_text_dataset(
    input_path,
    output_path,
    *,
    source_name,
    encoding="utf-8",
    max_tokens=None,
    tokenizer_path=TOKENIZER_PATH,
):
    """把 TXT 文本数据集转换为结构化 text JSONL。"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if max_tokens is not None and tokenizer_path is None:
        raise ValueError("启用按 token 分块时必须提供 tokenizer_path。")

    text = input_path.read_text(encoding=encoding)
    tokenizer = None if max_tokens is None else _load_tokenizer(tokenizer_path)
    text_chunks = [text]
    if tokenizer is not None and text:
        text_chunks = _chunk_text_by_lines(text, tokenizer, max_tokens=max_tokens)

    converted_count = 0
    skipped_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk_index, text_chunk in enumerate(text_chunks, start=1):
            if not text_chunk:
                skipped_count += 1
                continue

            record_id = source_name if len(text_chunks) == 1 else f"{source_name}-{chunk_index:04d}"
            try:
                structured_record = normalize_dataset_record(
                    {
                        "id": record_id,
                        "type": "text",
                        "text": text_chunk,
                        "source": source_name,
                        "split": "train",
                    }
                )
            except Exception:
                skipped_count += 1
                continue

            output_file.write(json.dumps(structured_record, ensure_ascii=False) + "\n")
            converted_count += 1

    print(
        f"迁移完成: source={source_name}, converted={converted_count}, skipped={skipped_count}, "
        f"output={output_path}"
    )


def _iter_txt_files(input_dir):
    """递归枚举目录下的 TXT 文件。"""
    return sorted(
        path
        for path in Path(input_dir).rglob("*.txt")
        if path.is_file()
    )


def convert_txt_text_path(
    input_path,
    output_path,
    *,
    source_name=None,
    encoding="utf-8",
    max_tokens=None,
    tokenizer_path=TOKENIZER_PATH,
):
    """兼容单文件和目录批量的 TXT 转换入口。"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.is_file():
        convert_txt_text_dataset(
            input_path,
            output_path,
            source_name=_resolve_source_name(source_name, input_path),
            encoding=encoding,
            max_tokens=max_tokens,
            tokenizer_path=tokenizer_path,
        )
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在或不是文件/目录: {input_path}")
    if output_path.suffix:
        raise ValueError("批量转换目录时，output 必须是输出目录，不能是文件路径。")

    txt_files = _iter_txt_files(input_path)
    if not txt_files:
        print(f"未找到 TXT 文件: {input_path}")
        return

    converted_files = 0
    for txt_file in txt_files:
        relative_output = txt_file.relative_to(input_path).with_suffix(".text.jsonl")
        target_path = output_path / relative_output
        convert_txt_text_dataset(
            txt_file,
            target_path,
            source_name=_resolve_source_name(source_name, txt_file, root_path=input_path),
            encoding=encoding,
            max_tokens=max_tokens,
            tokenizer_path=tokenizer_path,
        )
        converted_files += 1

    print(f"批量转换完成: input={input_path}, output={output_path}, files={converted_files}")


def build_argument_parser():
    parser = ArgumentParser(description="把 TXT 文本数据集转换为结构化 text JSONL，支持单文件或目录批量。")
    parser.add_argument("input", type=Path, help="输入 TXT 文件路径，或包含 TXT 的目录")
    parser.add_argument("output", type=Path, help="输出 JSONL 文件路径，或批量模式下的输出目录")
    parser.add_argument("--source", default=None, help="写入 source 字段；批量模式下作为 source 前缀")
    parser.add_argument("--encoding", default="utf-8", help="TXT 文件编码")
    parser.add_argument("--max-tokens", type=int, default=None, help="按 token 分块的上限")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=TOKENIZER_PATH,
        help="按 token 分块时使用的本地 tokenizer 路径",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    convert_txt_text_path(
        args.input,
        args.output,
        source_name=args.source,
        encoding=args.encoding,
        max_tokens=args.max_tokens,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()

