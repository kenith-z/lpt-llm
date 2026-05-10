"""把带思考链的推理 JSONL 数据转换为结构化 chat SFT JSONL。"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import normalize_dataset_record


LOGIC_DATASETS = (
    {
        "source": "dev_all_1500_zh",
        "language": "zh",
        "input": "dev-all-1500-zh.jsonl",
        "output": "dev-all-1500-zh.chat.sft.jsonl",
    },
    {
        "source": "provergen_5000_zh",
        "language": "zh",
        "input": "provergen-5000-zh.jsonl",
        "output": "provergen-5000-zh.chat.sft.jsonl",
    },
    {
        "source": "dev_all_1500_en",
        "language": "en",
        "input": "dev-all-1500-en.jsonl",
        "output": "dev-all-1500-en.chat.sft.jsonl",
    },
    {
        "source": "provergen_5000_en",
        "language": "en",
        "input": ("provergen-5000-en.jsonl", "provergen-5000_en.jsonl"),
        "output": "provergen-5000-en.chat.sft.jsonl",
    },
)

DOLLY_GLD_DATASET = {
    "source": "dolly_gld_zh",
    "language": "zh",
    "input": "dolly-gld-zh.shuffled.jsonl",
    "output": "dolly-gld-zh.shuffled.chat.sft.jsonl",
}


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON。") from exc


def _write_jsonl(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _first_existing_path(data_root, relative_or_candidates):
    candidates = (
        relative_or_candidates
        if isinstance(relative_or_candidates, tuple)
        else (relative_or_candidates,)
    )
    for candidate in candidates:
        path = Path(data_root) / candidate
        if path.exists():
            return path
    joined = ", ".join(str(Path(data_root) / candidate) for candidate in candidates)
    raise FileNotFoundError(f"未找到输入文件: {joined}")


def _required_text(payload, field_name, *, path, line_number):
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number} 缺少非空 {field_name} 字段。")
    return value.strip()


def _wrap_think(thinking):
    thinking = str(thinking).strip()
    if thinking.startswith("<think>") and thinking.endswith("</think>"):
        thinking = thinking[len("<think>") : -len("</think>")].strip()
    return f"<think>\n{thinking}\n</think>"


def _compact_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def _format_nl2fol_mapping(nl2fol):
    if isinstance(nl2fol, dict):
        lines = []
        for index, (natural_language, fol) in enumerate(nl2fol.items(), start=1):
            natural_language_text = _compact_text(natural_language)
            fol_text = _compact_text(fol)
            if natural_language_text and fol_text:
                lines.append(f"{index}. {natural_language_text} => {fol_text}")
        return "\n".join(lines)
    if isinstance(nl2fol, list):
        lines = [_compact_text(item) for item in nl2fol]
        return "\n".join(line for line in lines if line)
    return _compact_text(nl2fol)


def _build_logic_thinking(payload, reasoning, language):
    mapping = _format_nl2fol_mapping(payload.get("nl2fol"))
    conclusion_fol = _compact_text(payload.get("conclusion_fol"))
    if language == "zh":
        missing = "未提供"
        return "\n\n".join(
            [
                f"形式化映射 nl2fol：\n{mapping or missing}",
                f"待判断公式 conclusion_fol：\n{conclusion_fol or missing}",
                f"原 reasoning：\n{reasoning}",
            ]
        )

    missing = "Not provided"
    return "\n\n".join(
        [
            f"Formal mapping nl2fol:\n{mapping or missing}",
            f"Formula to judge conclusion_fol:\n{conclusion_fol or missing}",
            f"Original reasoning:\n{reasoning}",
        ]
    )


def _build_category_sentence(category):
    if not isinstance(category, dict):
        category_text = _compact_text(category)
        return f"我先识别自然语言任务分类 category：{category_text or '未提供明确分类'}。"

    skill = _compact_text(category.get("任务类型/认知技能"))
    domain = _compact_text(category.get("领域/主题"))
    output_format = _compact_text(category.get("输出格式"))
    difficulty = _compact_text(category.get("复杂性/难度级别"))
    intent = _compact_text(category.get("意图/应用场景"))

    parts = []
    if domain:
        parts.append(f"这是一个{domain}相关任务")
    if skill:
        parts.append(f"主要考察{skill}")
    if output_format:
        parts.append(f"期望输出格式是{output_format}")
    if difficulty:
        parts.append(f"难度为{difficulty}")
    if intent:
        parts.append(f"应用场景是{intent}")

    known_keys = {
        "任务类型/认知技能",
        "领域/主题",
        "输出格式",
        "复杂性/难度级别",
        "意图/应用场景",
    }
    for key, value in category.items():
        if key not in known_keys:
            value_text = _compact_text(value)
            if value_text:
                parts.append(f"{key}为{value_text}")

    if not parts:
        return "我先识别自然语言任务分类 category：未提供明确分类。"
    return f"我先识别自然语言任务分类 category：{'，'.join(parts)}。"


def _format_options(options):
    if not isinstance(options, list) or not options:
        return ""
    return "\n".join(str(option).strip() for option in options if str(option).strip())


def _build_logic_user_content(payload, language):
    context = str(payload.get("context") or "").strip()
    question = str(payload.get("question") or "").strip()
    options = _format_options(payload.get("options"))
    word_mapping = str(payload.get("word_mapping") or "").strip()

    if language == "zh":
        parts = [
            "请根据上下文进行一阶逻辑推理，判断问题中的陈述，并只从给定选项中选择答案。",
            f"上下文：\n{context}",
        ]
        if word_mapping:
            parts.append(f"中英词汇对照：\n{word_mapping}")
        parts.extend(
            [
                f"问题：\n{question}",
                f"选项：\n{options}",
            ]
        )
    else:
        parts = [
            "Use first-order logical reasoning to judge the statement in the question, and choose only from the given options.",
            f"Context:\n{context}",
            f"Question:\n{question}",
            f"Options:\n{options}",
        ]
    return "\n\n".join(part for part in parts if part.strip())


def _build_logic_assistant_content(payload, reasoning, answer, language):
    final_answer = f"答案：{answer}" if language == "zh" else f"Answer: {answer}"
    return f"{_wrap_think(_build_logic_thinking(payload, reasoning, language))}\n{final_answer}"


def _build_dolly_user_content(payload):
    instruction = str(payload.get("instruction") or "").strip()
    context = str(payload.get("context") or "").strip()
    if not instruction:
        raise ValueError("instruction 不能为空。")
    if not context:
        return instruction
    return f"{instruction}\n\n补充上下文：\n{context}"


def _build_dolly_assistant_content(thinking, response, category):
    dolly_thinking = f"{_build_category_sentence(category)}\n\n{thinking}"
    return f"{_wrap_think(dolly_thinking)}\n{response.strip()}"


def _record_id(source_name, raw_id, line_number):
    if raw_id is None or str(raw_id).strip() == "":
        return f"{source_name}-{line_number:06d}"
    return f"{source_name}-{str(raw_id).strip()}"


def convert_logic_dataset(input_path, output_path, *, source_name, language):
    records = []
    for line_number, payload in _read_jsonl(input_path):
        reasoning = _required_text(payload, "reasoning", path=input_path, line_number=line_number)
        answer = _required_text(payload, "answer", path=input_path, line_number=line_number)
        user_content = _build_logic_user_content(payload, language)
        assistant_content = _build_logic_assistant_content(payload, reasoning, answer, language)

        record = {
            "id": _record_id(source_name, payload.get("id"), line_number),
            "type": "chat",
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": source_name,
            "split": "train",
            "language": language,
        }
        records.append(normalize_dataset_record(record))

    if not records:
        raise ValueError(f"未从 {input_path} 解析出任何有效样本。")
    _write_jsonl(records, output_path)
    return len(records)


def convert_dolly_gld_dataset(input_path, output_path, *, source_name, language):
    records = []
    for line_number, payload in _read_jsonl(input_path):
        thinking = _required_text(payload, "think", path=input_path, line_number=line_number)
        response = _required_text(payload, "response", path=input_path, line_number=line_number)
        user_content = _build_dolly_user_content(payload)
        assistant_content = _build_dolly_assistant_content(thinking, response, payload.get("category"))

        records.append(
            normalize_dataset_record(
                {
                    "id": _record_id(source_name, payload.get("id"), line_number),
                    "type": "chat",
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "source": source_name,
                    "split": "train",
                    "language": language,
                }
            )
        )

    if not records:
        raise ValueError(f"未从 {input_path} 解析出任何有效样本。")
    _write_jsonl(records, output_path)
    return len(records)


def _manifest_relative_path(manifest_path, dataset_path):
    manifest_parent = Path(manifest_path).parent.resolve()
    return Path(os.path.relpath(Path(dataset_path).resolve(), manifest_parent)).as_posix()


def update_chat_sft_manifest(manifest_path, output_entries, *, weight):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError(f"{manifest_path} 缺少 datasets 列表。")

    existing_by_name = {entry.get("name"): entry for entry in datasets if isinstance(entry, dict)}
    for entry in output_entries:
        manifest_entry = {
            "name": entry["name"],
            "path": _manifest_relative_path(manifest_path, entry["path"]),
            "weight": weight,
        }
        if entry["name"] in existing_by_name:
            existing_by_name[entry["name"]].update(manifest_entry)
        else:
            datasets.append(manifest_entry)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_default_datasets(
    *,
    data_root,
    output_dir,
    manifest_path=None,
    update_manifest=False,
    manifest_weight=1.0,
):
    output_entries = []

    for dataset in LOGIC_DATASETS:
        input_path = _first_existing_path(data_root, dataset["input"])
        output_path = Path(output_dir) / dataset["output"]
        count = convert_logic_dataset(
            input_path,
            output_path,
            source_name=dataset["source"],
            language=dataset["language"],
        )
        output_entries.append({"name": dataset["source"], "path": output_path, "count": count})
        print(f"converted {dataset['source']}: count={count} output={output_path}")

    input_path = _first_existing_path(data_root, DOLLY_GLD_DATASET["input"])
    output_path = Path(output_dir) / DOLLY_GLD_DATASET["output"]
    count = convert_dolly_gld_dataset(
        input_path,
        output_path,
        source_name=DOLLY_GLD_DATASET["source"],
        language=DOLLY_GLD_DATASET["language"],
    )
    output_entries.append({"name": DOLLY_GLD_DATASET["source"], "path": output_path, "count": count})
    print(f"converted {DOLLY_GLD_DATASET['source']}: count={count} output={output_path}")

    if update_manifest:
        if manifest_path is None:
            raise ValueError("update_manifest=True 时必须提供 manifest_path。")
        update_chat_sft_manifest(manifest_path, output_entries, weight=manifest_weight)
        print(f"manifest_updated={manifest_path}")

    return output_entries


def build_argument_parser():
    parser = ArgumentParser(description="把带思考链的推理 JSONL 转为结构化 chat SFT 数据。")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data", help="原始 JSONL 所在目录。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "structured" / "chat_sft",
        help="结构化 chat SFT 输出目录。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "chat_sft.json",
        help="chat_sft manifest 路径。",
    )
    parser.add_argument("--update-manifest", action="store_true", help="转换后写入 chat_sft manifest。")
    parser.add_argument("--manifest-weight", type=float, default=1.0, help="写入 manifest 的默认 weight。")
    return parser


def main():
    args = build_argument_parser().parse_args()
    convert_default_datasets(
        data_root=args.data_root,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        update_manifest=args.update_manifest,
        manifest_weight=args.manifest_weight,
    )


if __name__ == "__main__":
    main()
