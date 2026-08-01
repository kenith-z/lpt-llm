"""对比 JSONL 样本在 GLM 与 DS tokenizer 下的 token 数。"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import unittest

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GLM_TOKENIZER_PATH = PROJECT_ROOT / "lpt_model" / "glm_tokenizer"
DS_TOKENIZER_PATH = PROJECT_ROOT / "lpt_model" / "ds_tokenizer"
JSONL_PATH_ENV = "TOKENIZER_COMPARE_JSONL"


def _load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )


def _extract_record_text(record):
    """从常见结构化样本中提取用于比较的正文。"""
    if isinstance(record, str):
        return record

    if not isinstance(record, dict):
        return json.dumps(record, ensure_ascii=False, sort_keys=True)

    if isinstance(record.get("text"), str):
        return record["text"]

    messages = record.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if content:
                parts.append(f"{role}: {content}" if role else content)
        if parts:
            return "\n".join(parts)

    common_fields = ("instruction", "context", "input", "output", "response", "content")
    parts = [
        str(record[field]).strip()
        for field in common_fields
        if isinstance(record.get(field), str) and record[field].strip()
    ]
    if parts:
        return "\n".join(parts)

    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _iter_jsonl_records(jsonl_path):
    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def _count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _collect_token_counts(jsonl_path, glm_tokenizer, ds_tokenizer):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
    if not jsonl_path.is_file():
        raise ValueError(f"JSONL 路径不是文件: {jsonl_path}")

    rows = []
    for line_number, record in _iter_jsonl_records(jsonl_path):
        text = _extract_record_text(record)
        rows.append(
            {
                "source": jsonl_path.name,
                "line": line_number,
                "id": record.get("id", "") if isinstance(record, dict) else "",
                "type": record.get("type", "") if isinstance(record, dict) else type(record).__name__,
                "chars": len(text),
                "glm_tokens": _count_tokens(glm_tokenizer, text),
                "ds_tokens": _count_tokens(ds_tokenizer, text),
            }
        )
    return rows


def collect_token_counts(jsonl_paths):
    """返回 JSONL 记录在两个 tokenizer 下的 token 统计。"""
    if isinstance(jsonl_paths, (str, Path)):
        jsonl_paths = [jsonl_paths]

    glm_tokenizer = _load_tokenizer(GLM_TOKENIZER_PATH)
    ds_tokenizer = _load_tokenizer(DS_TOKENIZER_PATH)
    rows = []
    for jsonl_path in jsonl_paths:
        rows.extend(_collect_token_counts(jsonl_path, glm_tokenizer, ds_tokenizer))
    return rows


def print_token_counts(jsonl_paths):
    rows = collect_token_counts(jsonl_paths)
    if isinstance(jsonl_paths, (str, Path)):
        jsonl_paths = [jsonl_paths]
    print("JSONL files:")
    for jsonl_path in jsonl_paths:
        print(f"- {Path(jsonl_path)}")
    print(f"GLM tokenizer: {GLM_TOKENIZER_PATH}")
    print(f"DS tokenizer: {DS_TOKENIZER_PATH}")
    print("source\tline\tid\ttype\tchars\tglm_tokens\tds_tokens\tds-glm")
    for row in rows:
        delta = row["ds_tokens"] - row["glm_tokens"]
        print(
            f"{row['source']}\t{row['line']}\t{row['id']}\t{row['type']}\t{row['chars']}\t"
            f"{row['glm_tokens']}\t{row['ds_tokens']}\t{delta}"
        )
    print(f"total_records\t{len(rows)}")
    print(f"total_glm_tokens\t{sum(row['glm_tokens'] for row in rows)}")
    print(f"total_ds_tokens\t{sum(row['ds_tokens'] for row in rows)}")
    return rows


class TestTokenizerJsonlTokenCounts(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get(JSONL_PATH_ENV),
        f"设置 {JSONL_PATH_ENV}=path/to/file.jsonl 后运行该诊断测试。",
    )
    def test_print_token_counts_for_jsonl(self):
        rows = print_token_counts(
            [
                Path(raw_path)
                for raw_path in os.environ[JSONL_PATH_ENV].split(os.pathsep)
                if raw_path
            ]
        )
        self.assertTrue(rows)


def build_argument_parser():
    parser = ArgumentParser(description="输出 JSONL 每条记录在 GLM 与 DS tokenizer 下的 token 数。")
    parser.add_argument("jsonl_path", type=Path, nargs="+", help="待统计的 JSONL 文件路径，可传多个")
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    rows = print_token_counts(args.jsonl_path)
    if not rows:
        raise SystemExit("JSONL 中没有可统计的记录。")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        unittest.main()
