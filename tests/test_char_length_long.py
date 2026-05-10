# @File test_char_length_long.py
# @author Kenith-Z
# @version 1.0.0
# @since 2026/5/4
#!/usr/bin/env python3
"""
扫描指定目录下所有 .jsonl 文件，输出每个文件中：
  - 最长行的字符数（含换行符）
  - messages 字段序列化后的最长字符数
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "lpt_model" / "ds_tokenizer"


def load_tokenizer(tokenizer_path: Path | None):
    """按需加载本地 tokenizer；传入 None 时只做字符统计。"""
    if tokenizer_path is None:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("计算 token 长度需要安装 transformers。") from exc
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )


def token_length(tokenizer, text: str):
    """返回文本 token 数；未启用 tokenizer 时返回 0。"""
    if tokenizer is None or not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def process_jsonl_file(filepath: Path, *, tokenizer=None):
    max_line_len = 0
    max_json_token_len = 0
    max_msg_len = 0
    max_msg_token_len = 0
    max_text_token_len = 0
    max_file_token_len = 0
    line_count = 0

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_len = len(line)  # 包含换行符
                if line_len > max_line_len:
                    max_line_len = line_len

                # 尝试解析 JSON
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[警告] {filepath}: 第 {line_num} 行 JSON 解析失败: {e}",
                          file=sys.stderr)
                    continue

                json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                json_token_len = token_length(tokenizer, json_str)
                if json_token_len > max_json_token_len:
                    max_json_token_len = json_token_len

                messages = data.get("messages")
                msg_token_len = 0
                if messages is not None:
                    # 将 messages 序列化为紧凑 JSON 字符串（无 ASCII 转义）
                    msg_str = json.dumps(messages, ensure_ascii=False)
                    msg_len = len(msg_str)
                    if msg_len > max_msg_len:
                        max_msg_len = msg_len
                    msg_token_len = token_length(tokenizer, msg_str)
                    if msg_token_len > max_msg_token_len:
                        max_msg_token_len = msg_token_len

                text_token_len = 0
                text = data.get("text")
                if isinstance(text, str):
                    text_token_len = token_length(tokenizer, text)
                    if text_token_len > max_text_token_len:
                        max_text_token_len = text_token_len

                record_token_len = max(json_token_len, msg_token_len, text_token_len)
                if record_token_len > max_file_token_len:
                    max_file_token_len = record_token_len

                line_count += 1
    except Exception as e:
        print(f"[错误] 无法读取文件 {filepath}: {e}", file=sys.stderr)
        return

    # 输出该文件的结果
    print(f"文件: {filepath}")
    print(f"  总行数: {line_count}")
    print(f"  最长行字符数: {max_line_len}")
    print(f"  messages 最长字符数: {max_msg_len}")
    if tokenizer is not None:
        print(f"  JSON 元素最长 token 数: {max_json_token_len}")
        print(f"  messages 最长 token 数: {max_msg_token_len}")
        print(f"  text 最长 token 数: {max_text_token_len}")
        print(f"  文件最长 token 数: {max_file_token_len}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="统计 JSONL 文件中最长行长度和 messages 字段的最长序列化长度"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="要扫描的目录，默认为当前目录"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="用于统计 token 长度的本地 tokenizer 路径；传入空字符串可关闭 token 统计"
    )
    args = parser.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        print(f"错误: '{root}' 不是一个有效的目录", file=sys.stderr)
        sys.exit(1)

    # 收集所有 .jsonl 文件
    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"在 '{root}' 中没有找到 .jsonl 文件")
        return

    tokenizer_path = None if str(args.tokenizer_path).strip() == "" else args.tokenizer_path
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"在 '{root}' 中找到 {len(jsonl_files)} 个 .jsonl 文件\n")
    for fpath in jsonl_files:
        process_jsonl_file(fpath, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
