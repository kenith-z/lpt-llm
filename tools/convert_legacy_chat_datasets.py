"""把 DS 模板字符串对话数据迁移为结构化 JSONL。"""

from argparse import ArgumentParser
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record
from lpt_protocol import DS_EOS_TOKEN


ROLE_LABEL_TO_ROLE = {
    "System": "system",
    "User": "user",
    "Assistant": "assistant",
    "Observation": "observation",
}
ROLE_LABEL_PATTERN = re.compile(r"(System|User|Assistant|Observation):\s*")
ALL_PROTOCOL_TOKENS_PATTERN = re.compile(
    rf"(System|User|Assistant|Observation):\s*|{re.escape(DS_EOS_TOKEN)}"
)


def _flush_message(messages, role, content_buffer):
    if role is None:
        return
    content = content_buffer.strip()
    if not content:
        raise ValueError(f"检测到空的 {role} 消息。")
    messages.append({"role": role, "content": content})


def parse_legacy_chat_record(text):
    """把 DS 模板标签串解析为 messages。"""
    if not isinstance(text, str):
        raise TypeError("旧版记录必须是字符串。")

    messages = []
    current_role = None
    cursor = 0

    for match in ALL_PROTOCOL_TOKENS_PATTERN.finditer(text):
        token = match.group(0)
        chunk = text[cursor:match.start()]
        if current_role is not None:
            _flush_message(messages, current_role, chunk)

        if token == DS_EOS_TOKEN:
            current_role = None
        else:
            current_role = ROLE_LABEL_TO_ROLE[ROLE_LABEL_PATTERN.fullmatch(token).group(1)]

        cursor = match.end()

    if current_role is not None:
        trailing_content = text[cursor:]
        _flush_message(messages, current_role, trailing_content)

    if not messages:
        raise ValueError("未解析出任何消息。")
    if messages[0]["role"] != "user":
        raise ValueError("首条消息必须是 user。")

    return messages


def convert_dataset(input_path, output_path, source_name):
    """把 JSON 数组格式的 DS 模板数据集转换为结构化 JSONL。"""
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, list):
        raise ValueError(f"{input_path} 不是 JSON 数组。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_count = 0
    skipped_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for index, record in enumerate(payload, start=1):
            try:
                structured_record = normalize_dataset_record(
                    {
                        "id": f"{source_name}-{index:06d}",
                        "type": "chat",
                        "messages": parse_legacy_chat_record(record),
                        "source": source_name,
                        "split": "train",
                        "language": "zh",
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
    parser = ArgumentParser(description="把 DS 模板字符串对话数据迁移为结构化 JSONL。")
    parser.add_argument("input", type=Path, help="DS 模板 JSON 数组文件路径")
    parser.add_argument("output", type=Path, help="输出 JSONL 文件路径")
    parser.add_argument("--source", required=True, help="写入结构化记录的 source 字段")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    convert_dataset(args.input, args.output, args.source)


if __name__ == "__main__":
    main()
