"""把原始 text JSONL 标准化为结构化 text 数据。"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record


def _load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )


def _chunk_token_ids(token_ids, max_tokens):
    for begin in range(0, len(token_ids), max_tokens):
        yield token_ids[begin : begin + max_tokens]


def _chunk_text_by_lines(text, tokenizer, max_tokens):
    normalized_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not normalized_lines:
        return []

    chunks = []
    current_lines = []

    def current_text():
        return "\n".join(current_lines)

    for line in normalized_lines:
        if not current_lines:
            current_lines.append(line)
            continue

        candidate_lines = current_lines + [line]
        candidate_text = "\n".join(candidate_lines)
        candidate_length = len(tokenizer(candidate_text, add_special_tokens=False)["input_ids"])
        if candidate_length <= max_tokens:
            current_lines = candidate_lines
            continue

        chunks.append(current_text())
        current_lines = [line]

    if current_lines:
        chunks.append(current_text())

    normalized_chunks = []
    for chunk in chunks:
        token_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_tokens:
            normalized_chunks.append(chunk)
            continue
        for token_id_chunk in _chunk_token_ids(token_ids, max_tokens):
            normalized_chunks.append(tokenizer.decode(token_id_chunk, skip_special_tokens=False).strip())

    return [chunk for chunk in normalized_chunks if chunk]


def convert_raw_text_jsonl(
    input_path,
    output_path,
    source_name,
    content_field="content",
    max_tokens=None,
    tokenizer_path=None,
):
    """把包含原始文本字段的 JSONL 转成结构化 text 样本。"""
    converted_count = 0
    skipped_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    if max_tokens is not None:
        if tokenizer_path is None:
            raise ValueError("启用按 token 分块时必须提供 tokenizer_path。")
        tokenizer = _load_tokenizer(tokenizer_path)

    with input_path.open("r", encoding="utf-8") as input_file, output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
        ) as output_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            text = payload.pop(content_field, None)
            text_chunks = [text]
            if tokenizer is not None and text is not None:
                text_chunks = _chunk_text_by_lines(text, tokenizer, max_tokens=max_tokens)

            if not text_chunks:
                skipped_count += 1
                continue

            original_id = payload.get("id") or f"{source_name}-{line_number:06d}"
            for chunk_index, text_chunk in enumerate(text_chunks, start=1):
                try:
                    structured_record = normalize_dataset_record(
                        {
                            **payload,
                            "id": original_id if len(text_chunks) == 1 else f"{original_id}-{chunk_index:04d}",
                            "type": "text",
                            "text": text_chunk,
                            "source": payload.get("source") or source_name,
                            "split": payload.get("split") or "train",
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
    parser = ArgumentParser(description="把原始 text JSONL 标准化为结构化 text 数据。")
    parser.add_argument("input", type=Path, help="输入 JSONL 路径")
    parser.add_argument("output", type=Path, help="输出 JSONL 路径")
    parser.add_argument("--source", required=True, help="写入结构化记录的 source 字段")
    parser.add_argument("--content-field", default="content", help="原始文本字段名")
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
    convert_raw_text_jsonl(
        args.input,
        args.output,
        args.source,
        content_field=args.content_field,
        max_tokens=args.max_tokens,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
