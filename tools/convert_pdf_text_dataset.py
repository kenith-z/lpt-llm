"""把 pending 目录中的 PDF 文件转换为结构化 text JSONL。"""

from __future__ import annotations
import pdfplumber
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data.schema import normalize_dataset_record
from lpt_protocol import DS_EOS_TOKEN


# 待转换数据类型。PDF 原文只转成 text 样本；chat_sft/chat_lora 仅保留目录映射，避免手写路径。
DATASET_KIND = "text"
# 待处理 PDF 目录。命令行 --input-dir 可临时覆盖。
PENDING_INPUT_DIR = PROJECT_ROOT / "data" / "z-pending-data" / "pdf"
# 输出 JSONL 路径。None 表示输出到 data/structured/<阶段目录>/<目录名>.text.jsonl。
OUTPUT_PATH = None
# JSONL 写入模式。True 表示追加到已有文件末尾；需要重建文件时用命令行 --overwrite 显式覆盖。
APPEND_OUTPUT = True
# source 字段。None 表示使用输入目录名，保留来源但不写入不参与训练的额外元数据。
SOURCE_NAME = None
# 已处理原始 PDF 的归档根目录；真实归档路径会追加类型目录和输入目录名。
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "z-old-data"
# 转换成功后是否移动已提取出文字的 PDF 到 z-old-data。图片 PDF 会跳过并保留在 pending。
MOVE_AFTER_CONVERT = True
# 训练最大序列长度，默认与 GlobalConfig.train_max_sequence_length 保持一致；转换时会用 tokenizer 预分块。
TRAIN_MAX_SEQUENCE_LENGTH = 4096
# 本地 DS tokenizer 目录。用于按 train_max_sequence_length 做 token 级分块。
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "lpt_model" / "ds_tokenizer"


@dataclass(frozen=True)
class DatasetKindConfig:
    """描述外部数据类型与 v2 结构化目录之间的映射。"""

    kind: str
    record_type: str
    structured_dir: str
    archive_dir: str
    output_suffix: str


DATASET_KIND_CONFIGS = {
    "text": DatasetKindConfig(
        kind="text",
        record_type="text",
        structured_dir="text_pretrain",
        archive_dir="text",
        output_suffix="text",
    ),
    "chat_sft": DatasetKindConfig(
        kind="chat_sft",
        record_type="chat",
        structured_dir="chat_sft",
        archive_dir="chat_sft",
        output_suffix="chat.sft",
    ),
    "chat_lora": DatasetKindConfig(
        kind="chat_lora",
        record_type="chat",
        structured_dir="chat_lora",
        archive_dir="chat_lora",
        output_suffix="chat.lora",
    ),
}


def _resolve_kind_config(dataset_kind):
    """校验数据类型，并返回输出目录配置。"""
    try:
        return DATASET_KIND_CONFIGS[dataset_kind]
    except KeyError as exc:
        allowed = ", ".join(sorted(DATASET_KIND_CONFIGS))
        raise ValueError(f"不支持的数据类型: {dataset_kind}，可选值: {allowed}") from exc


def _ensure_pdf_to_text_kind(kind_config):
    """PDF 原文转换只能生成 text 样本，避免伪造 chat 监督数据。"""
    if kind_config.record_type != "text":
        raise ValueError(
            "PDF 原文转换当前只支持 dataset_kind='text'。"
            "chat_sft/chat_lora 需要 question/answer 或 instruction/response 结构化字段。"
        )


def _iter_pdf_files(input_dir):
    """递归枚举目录下可处理的 PDF 文件。"""
    return sorted(
        path
        for path in Path(input_dir).rglob("*.pdf")
        if path.is_file() and not path.name.startswith("~$")
    )


def _load_tokenizer(tokenizer_path):
    """按需加载本地 tokenizer，避免把转换脚本绑定到训练入口依赖。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )


def _chunk_token_ids(token_ids, max_tokens):
    """把 token id 序列切成若干固定长度块。"""
    for begin in range(0, len(token_ids), max_tokens):
        yield token_ids[begin : begin + max_tokens]


def _chunk_text_by_lines(text, tokenizer, max_tokens):
    """优先按 PDF 行边界保留结构，再在 token 超长时回退切分。"""
    if max_tokens is None:
        return [text] if text else []
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须为正整数。")

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


def _text_token_budget(train_max_sequence_length, tokenizer):
    """计算 text 字段预算，给训练渲染阶段追加的 EOS token 预留空间。"""
    if train_max_sequence_length is None:
        return None

    max_sequence_length = int(train_max_sequence_length)
    if max_sequence_length <= 1:
        raise ValueError("train_max_sequence_length 必须大于 1，才能容纳正文和 EOS。")

    eos_length = len(tokenizer(DS_EOS_TOKEN, add_special_tokens=False)["input_ids"])
    text_budget = max_sequence_length - eos_length
    if text_budget <= 0:
        raise ValueError("train_max_sequence_length 小于等于 EOS token 长度，无法生成有效 text 样本。")
    return text_budget


def _normalize_extracted_text(text):
    """清理 PDF 抽取出的文本，空文本表示图片型或无法抽取文字。"""
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    normalized_lines = []
    previous_blank = False
    for line in lines:
        if not line:
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    return "\n".join(normalized_lines).strip()


def _escape_markdown_table_cell(value):
    """清理并转义 Markdown 表格单元格。"""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = text.replace("|", "\\|")
    return "<br>".join(line.strip() for line in text.splitlines() if line.strip())


def table_to_markdown(table):
    """把 pdfplumber 抽取出的二维表格转为 Markdown 表格。"""
    if not table:
        return ""

    normalized_rows = []
    max_columns = 0
    for row in table:
        if row is None:
            continue
        normalized_row = [_escape_markdown_table_cell(cell) for cell in row]
        if not any(normalized_row):
            continue
        normalized_rows.append(normalized_row)
        max_columns = max(max_columns, len(normalized_row))

    if not normalized_rows or max_columns == 0:
        return ""

    rows = [row + [""] * (max_columns - len(row)) for row in normalized_rows]
    header = rows[0]
    if not any(header):
        header = [f"列{index}" for index in range(1, max_columns + 1)]
        body_rows = rows[1:]
    else:
        body_rows = rows[1:]

    def markdown_row(row):
        return "| " + " | ".join(row) + " |"

    lines = [
        markdown_row(header),
        "| " + " | ".join(["---"] * max_columns) + " |",
    ]
    lines.extend(markdown_row(row) for row in body_rows)
    return "\n".join(lines)


def _extract_with_pdfplumber(pdf_path):
    """使用 pdfplumber 抽取页面文字，并把页面表格转为 Markdown 表格。"""


    page_texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_parts = []
            text = page.extract_text() or ""
            if text.strip():
                page_parts.append(text.strip())

            tables = page.extract_tables() or []
            markdown_tables = [table_to_markdown(table) for table in tables]
            page_parts.extend(table for table in markdown_tables if table)

            if page_parts:
                page_texts.append("\n\n".join(page_parts))
    return "\n\n".join(page_texts)


def extract_pdf_text(pdf_path):
    """仅使用 pdfplumber 提取 PDF 文字；图片型 PDF 会返回空字符串。"""
    return _normalize_extracted_text(_extract_with_pdfplumber(pdf_path))


def _resolve_output_path(input_dir, kind_config, output_path):
    """按“一个目录一个 JSONL”规则推导默认输出路径。"""
    if output_path is not None:
        return Path(output_path)
    input_dir = Path(input_dir)
    filename = f"{input_dir.name}.{kind_config.output_suffix}.jsonl"
    return PROJECT_ROOT / "data" / "structured" / kind_config.structured_dir / filename


def _resolve_source_name(input_dir, source_name):
    """生成 JSONL 记录的 source 字段。"""
    if source_name:
        return source_name
    return Path(input_dir).name


def _record_id(source_name, file_index, chunk_index, chunk_count):
    """生成稳定记录 ID，分块时追加 chunk 后缀。"""
    base_id = f"{source_name}-{file_index:06d}"
    if chunk_count == 1:
        return base_id
    return f"{base_id}-{chunk_index:04d}"


def _count_existing_jsonl_records(output_path):
    """统计已有 JSONL 非空行数，作为追加写入的新 ID 起点。"""
    output_path = Path(output_path)
    if not output_path.is_file():
        return 0
    with output_path.open("r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def _build_pdf_text_records(
    pdf_file,
    *,
    source_name,
    file_index,
    tokenizer=None,
    text_token_budget=None,
):
    """把单个 PDF 文件转换为结构化 text 记录；图片 PDF 返回空列表。"""
    text = extract_pdf_text(pdf_file)
    if not text:
        return []

    text_chunks = [text]
    if tokenizer is not None:
        text_chunks = _chunk_text_by_lines(text, tokenizer, max_tokens=text_token_budget)
    if not text_chunks:
        return []

    records = []
    for chunk_index, text_chunk in enumerate(text_chunks, start=1):
        records.append(
            normalize_dataset_record(
                {
                    "id": _record_id(source_name, file_index, chunk_index, len(text_chunks)),
                    "type": "text",
                    "text": text_chunk,
                    "source": source_name,
                }
            )
        )
    return records


def _prepare_pdf_conversion_context(
    *,
    train_max_sequence_length=TRAIN_MAX_SEQUENCE_LENGTH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
):
    """准备 PDF 转换所需 tokenizer 和 text token 预算。"""
    if train_max_sequence_length is not None and tokenizer_path is None:
        raise ValueError("启用按 token 分块时必须提供 tokenizer_path。")

    tokenizer = None if train_max_sequence_length is None else _load_tokenizer(tokenizer_path)
    text_token_budget = None if tokenizer is None else _text_token_budget(train_max_sequence_length, tokenizer)
    return tokenizer, text_token_budget


def _write_jsonl(records, output_path):
    """以 UTF-8 无 BOM 和 LF 换行写入 JSONL。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _jsonl_needs_leading_newline(output_path):
    """追加写入前检查文件末尾是否已经以换行收尾。"""
    output_path = Path(output_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False

    with output_path.open("rb") as input_file:
        input_file.seek(-1, 2)
        return input_file.read(1) != b"\n"


def _append_jsonl(records, output_path):
    """把记录追加到已有 JSONL 末尾，必要时先补一个换行。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = _jsonl_needs_leading_newline(output_path)
    with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
        if needs_newline:
            output_file.write("\n")
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _next_available_path(path):
    """避免归档时覆盖同名旧文件。"""
    path = Path(path)
    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}-{index:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"无法为归档文件生成非冲突路径: {path}")


def archive_pdf_files(pdf_files, input_dir, *, archive_root, dataset_kind):
    """把已成功提取文字的 PDF 移入 z-old-data 对应类型目录。"""
    kind_config = _resolve_kind_config(dataset_kind)
    input_dir = Path(input_dir)
    archive_dir = Path(archive_root) / kind_config.archive_dir / input_dir.name
    moved_paths = []

    for pdf_file in pdf_files:
        relative_path = pdf_file.relative_to(input_dir)
        target_path = _next_available_path(archive_dir / relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_file.replace(target_path)
        moved_paths.append(target_path)

    return moved_paths


def convert_pdf_directory_to_jsonl(
    input_dir,
    output_path=None,
    *,
    dataset_kind="text",
    source_name=None,
    archive_root=ARCHIVE_ROOT,
    move_to_archive=True,
    append_output=APPEND_OUTPUT,
    train_max_sequence_length=TRAIN_MAX_SEQUENCE_LENGTH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
):
    """把一个目录下可提取文字的 PDF 合并为结构化 JSONL。"""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    kind_config = _resolve_kind_config(dataset_kind)
    _ensure_pdf_to_text_kind(kind_config)

    pdf_files = _iter_pdf_files(input_dir)
    if not pdf_files:
        raise FileNotFoundError(f"未找到 PDF 文件: {input_dir}")

    source_name = _resolve_source_name(input_dir, source_name)
    output_path = _resolve_output_path(input_dir, kind_config, output_path)
    existing_record_count = _count_existing_jsonl_records(output_path) if append_output else 0
    tokenizer, text_token_budget = _prepare_pdf_conversion_context(
        train_max_sequence_length=train_max_sequence_length,
        tokenizer_path=tokenizer_path,
    )
    converted_count = 0
    converted_file_count = 0
    skipped_count = 0
    archived_count = 0
    next_file_index = existing_record_count + 1
    first_write = True

    for pdf_file in pdf_files:
        try:
            records = _build_pdf_text_records(
                pdf_file,
                source_name=source_name,
                file_index=next_file_index,
                tokenizer=tokenizer,
                text_token_budget=text_token_budget,
            )
        except Exception as exc:
            skipped_count += 1
            print(f"跳过 PDF: path={pdf_file}, reason={exc}", file=sys.stderr)
            continue

        if not records:
            skipped_count += 1
            print(f"跳过 PDF: path={pdf_file}, reason=未提取到可训练文字", file=sys.stderr)
            continue

        # 每个 PDF 成功后立即落盘。覆盖模式只用于第一批写入，后续文件继续追加，保证长任务可断点续跑。
        if append_output or not first_write:
            _append_jsonl(records, output_path)
        else:
            _write_jsonl(records, output_path)
        first_write = False

        converted_count += len(records)
        converted_file_count += 1
        next_file_index += 1

        if move_to_archive:
            archive_pdf_files(
                [pdf_file],
                input_dir,
                archive_root=archive_root,
                dataset_kind=dataset_kind,
            )
            archived_count += 1

    summary = {
        "dataset_kind": dataset_kind,
        "source": source_name,
        "input_dir": str(input_dir),
        "output": str(output_path),
        "input_files": len(pdf_files),
        "converted": converted_count,
        "converted_files": converted_file_count,
        "skipped": skipped_count,
        "archived": archived_count,
        "archive_root": str(archive_root),
        "output_mode": "append" if append_output else "overwrite",
        "existing_records": existing_record_count,
        "train_max_sequence_length": train_max_sequence_length,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_argument_parser():
    parser = ArgumentParser(description="把 pending 目录中的 PDF 文本转换为结构化 text JSONL。")
    parser.add_argument("--input-dir", type=Path, default=PENDING_INPUT_DIR, help="待转换 PDF 所在目录。")
    parser.add_argument(
        "--dataset-kind",
        choices=sorted(DATASET_KIND_CONFIGS),
        default=DATASET_KIND,
        help="数据类型配置；PDF 原文转换本次使用 text。",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出 JSONL 路径；默认按目录名推导。")
    parser.add_argument("--source", default=SOURCE_NAME, help="写入 source 字段；默认使用输入目录名。")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT, help="z-old-data 根目录。")
    parser.add_argument("--keep-source", action="store_true", help="只转换，不移动已提取出文字的 PDF 到 z-old-data。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 JSONL；默认是追加。")
    parser.add_argument(
        "--train-max-sequence-length",
        "--max-tokens",
        dest="train_max_sequence_length",
        type=int,
        default=TRAIN_MAX_SEQUENCE_LENGTH,
        help="训练最大序列长度；会用 tokenizer 预分块，并为 text EOS 预留空间。",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="按 token 分块时使用的本地 tokenizer 路径。",
    )
    return parser


def main(args=None):
    parser = build_argument_parser()
    parsed_args = parser.parse_args(args)
    convert_pdf_directory_to_jsonl(
        parsed_args.input_dir,
        parsed_args.output,
        dataset_kind=parsed_args.dataset_kind,
        source_name=parsed_args.source,
        archive_root=parsed_args.archive_root,
        move_to_archive=not parsed_args.keep_source and MOVE_AFTER_CONVERT,
        append_output=not parsed_args.overwrite and APPEND_OUTPUT,
        train_max_sequence_length=parsed_args.train_max_sequence_length,
        tokenizer_path=parsed_args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
