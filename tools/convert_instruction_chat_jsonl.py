"""把 instruction/context/response JSONL 转为结构化 chat 数据，并可拆出 LoRA 子集。"""

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record


def _build_user_content(instruction, context):
    normalized_instruction = (instruction or "").strip()
    normalized_context = (context or "").strip()
    if not normalized_instruction:
        raise ValueError("instruction 不能为空。")

    if not normalized_context:
        return normalized_instruction

    return (
        f"{normalized_instruction}\n\n"
        f"补充上下文：\n{normalized_context}"
    )


def _load_chat_records(input_path, source_name):
    structured_records = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            user_content = _build_user_content(
                payload.get("instruction"),
                payload.get("context"),
            )
            assistant_content = (payload.get("response") or "").strip()
            if not assistant_content:
                continue

            structured_records.append(
                normalize_dataset_record(
                    {
                        "id": f"{source_name}-{line_number:06d}",
                        "type": "chat",
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ],
                        "source": source_name,
                        "split": "train",
                        "language": "zh",
                        "category": payload.get("category"),
                    }
                )
            )

    if not structured_records:
        raise ValueError(f"未从 {input_path} 解析出任何有效 chat 样本。")
    return structured_records


def _write_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_instruction_chat_jsonl(
    input_path,
    sft_output_path,
    *,
    source_name,
    lora_output_path=None,
    lora_count=0,
    seed=42,
):
    """转换 instruction/context/response JSONL，并可拆出 LoRA 子集。"""
    records = _load_chat_records(input_path, source_name)
    lora_records = []
    sft_records = records

    if lora_output_path is not None and lora_count > 0:
        if lora_count >= len(records):
            raise ValueError("lora_count 必须小于 chat 数据总量。")
        rng = random.Random(seed)
        lora_indices = set(rng.sample(range(len(records)), lora_count))
        lora_records = [records[index] for index in sorted(lora_indices)]
        sft_records = [record for index, record in enumerate(records) if index not in lora_indices]

    _write_jsonl(sft_records, sft_output_path)
    if lora_output_path is not None and lora_records:
        _write_jsonl(lora_records, lora_output_path)

    print(
        f"迁移完成: source={source_name}, sft={len(sft_records)}, "
        f"lora={len(lora_records)}, sft_output={sft_output_path}, "
        f"lora_output={lora_output_path}"
    )


def build_argument_parser():
    parser = ArgumentParser(description="把 instruction/context/response JSONL 转为结构化 chat 数据。")
    parser.add_argument("input", type=Path, help="输入 JSONL 路径")
    parser.add_argument("sft_output", type=Path, help="SFT 输出 JSONL 路径")
    parser.add_argument("--source", required=True, help="写入结构化记录的 source 字段")
    parser.add_argument("--lora-output", type=Path, default=None, help="LoRA 子集输出路径")
    parser.add_argument("--lora-count", type=int, default=0, help="拆给 LoRA 的样本数量")
    parser.add_argument("--seed", type=int, default=42, help="LoRA 抽样随机种子")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    convert_instruction_chat_jsonl(
        args.input,
        args.sft_output,
        source_name=args.source,
        lora_output_path=args.lora_output,
        lora_count=args.lora_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

