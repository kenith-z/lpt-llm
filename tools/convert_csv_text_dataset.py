"""把 CSV 文本数据集转换为结构化 text JSONL。"""

from argparse import ArgumentParser
import csv
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record
from tools.convert_raw_text_jsonl import _chunk_text_by_lines, _load_tokenizer


def _build_text_content(row, *, body_field, title_field):
    body = (row.get(body_field) or "").strip()
    title = (row.get(title_field) or "").strip()
    if not body and not title:
        return ""
    if title and body:
        return f"标题：{title}\n\n正文：\n{body}"
    return title or body


def convert_csv_text_dataset(
    input_path,
    output_path,
    *,
    source_name,
    body_field="text1",
    title_field="text2",
    encoding="utf-8-sig",
    max_tokens=None,
    tokenizer_path=None,
):
    """把 CSV 文本数据集转换为结构化 text JSONL。"""
    if max_tokens is not None and tokenizer_path is None:
        raise ValueError("启用按 token 分块时必须提供 tokenizer_path。")

    tokenizer = None if max_tokens is None else _load_tokenizer(tokenizer_path)
    converted_count = 0
    skipped_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding=encoding, newline="") as input_file, output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} 不包含表头。")
        missing_fields = [field for field in (body_field, title_field) if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(f"{input_path} 缺少字段: {missing_fields}")

        for row_index, row in enumerate(reader, start=1):
            text = _build_text_content(row, body_field=body_field, title_field=title_field)
            text_chunks = [text]
            if tokenizer is not None and text:
                text_chunks = _chunk_text_by_lines(text, tokenizer, max_tokens=max_tokens)

            if not text_chunks:
                skipped_count += 1
                continue

            base_id = f"{source_name}-{row_index:06d}"
            for chunk_index, text_chunk in enumerate(text_chunks, start=1):
                record_id = base_id if len(text_chunks) == 1 else f"{base_id}-{chunk_index:04d}"
                try:
                    structured_record = normalize_dataset_record(
                        {
                            "id": record_id,
                            "type": "text",
                            "text": text_chunk,
                            "source": source_name,
                            "split": "train",
                            "paper_title": (row.get(title_field) or "").strip() or None,
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


def build_argument_parser():
    parser = ArgumentParser(description="把 CSV 文本数据集转换为结构化 text JSONL。")
    parser.add_argument("input", type=Path, help="输入 CSV 路径")
    parser.add_argument("output", type=Path, help="输出 JSONL 路径")
    parser.add_argument("--source", required=True, help="写入结构化记录的 source 字段")
    parser.add_argument("--body-field", default="text1", help="正文文本字段名")
    parser.add_argument("--title-field", default="text2", help="标题字段名")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV 文件编码")
    parser.add_argument("--max-tokens", type=int, default=None, help="按 token 分块的上限")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="按 token 分块时使用的本地 tokenizer 路径",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    convert_csv_text_dataset(
        args.input,
        args.output,
        source_name=args.source,
        body_field=args.body_field,
        title_field=args.title_field,
        encoding=args.encoding,
        max_tokens=args.max_tokens,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()

