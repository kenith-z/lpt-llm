"""把 parquet 文本数据集转换为结构化 text JSONL。"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record
from tools.convert_raw_text_jsonl import _chunk_text_by_lines, _load_tokenizer


def convert_parquet_text_dataset(
    input_path,
    output_path,
    *,
    source_name,
    text_field="text",
    max_tokens=None,
    tokenizer_path=None,
):
    """把 parquet 文本数据集转换为结构化 text JSONL。"""
    if max_tokens is not None and tokenizer_path is None:
        raise ValueError("启用按 token 分块时必须提供 tokenizer_path。")

    tokenizer = None if max_tokens is None else _load_tokenizer(tokenizer_path)
    table = pq.read_table(input_path)
    available_fields = set(table.schema.names)
    if text_field not in available_fields:
        raise ValueError(
            f"{input_path} 缺少文本字段 {text_field}，现有字段为: {sorted(available_fields)}"
        )

    # metadata_fields = [field for field in ("score", "source") if field in available_fields]
    metadata_fields=[]
    batch_fields = [text_field] + metadata_fields

    converted_count = 0
    skipped_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        row_index = 0
        for record_batch in pq.ParquetFile(input_path).iter_batches(columns=batch_fields):
            batch_table = pa.Table.from_batches([record_batch])
            for row in batch_table.to_pylist():
                row_index += 1
                text = row.get(text_field)
                text_chunks = [text]
                if tokenizer is not None and text is not None:
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
                                **{field: row.get(field) for field in metadata_fields},
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
    parser = ArgumentParser(description="把 parquet 文本数据集转换为结构化 text JSONL。")
    parser.add_argument("input", type=Path, help="输入 parquet 路径")
    parser.add_argument("output", type=Path, help="输出 JSONL 路径")
    parser.add_argument("--source", required=True, help="写入结构化记录的 source 字段")
    parser.add_argument("--text-field", default="text", help="正文文本字段名")
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
    convert_parquet_text_dataset(
        args.input,
        args.output,
        source_name=args.source,
        text_field=args.text_field,
        max_tokens=args.max_tokens,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
