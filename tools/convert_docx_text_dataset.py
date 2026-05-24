"""把 pending 目录中的 DOC/DOCX 文件合并转换为一个结构化 text JSONL。"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data.schema import normalize_dataset_record
from lpt_protocol import DS_EOS_TOKEN


# 待转换数据类型。本脚本当前只把 DOC/DOCX 原文转成 text 样本；chat_sft/chat_lora 仅保留目录映射，避免手写路径。
DATASET_KIND = "text"
# 待处理 DOC/DOCX 目录。命令行 --input-dir 可临时覆盖。
PENDING_INPUT_DIR = PROJECT_ROOT / "data" / "z-pending-data" / "法律"
# 输出 JSONL 路径。None 表示按“一个目录一个 JSONL”规则输出到 data/structured/<阶段目录>/<目录名>.text.jsonl。
OUTPUT_PATH = None
# JSONL 写入模式。True 表示追加到已有文件末尾；需要重建文件时用命令行 --overwrite 显式覆盖。
APPEND_OUTPUT = True
# source 字段。None 表示使用输入目录名，保留来源但不写入不参与训练的额外元数据。
SOURCE_NAME = None
# 已处理原始 DOC/DOCX 的归档根目录；真实归档路径会追加类型目录和输入目录名。
ARCHIVE_ROOT = PROJECT_ROOT / "data" / "z-old-data"
# 转换成功后是否移动 DOC/DOCX 到 z-old-data。重跑已归档数据时可用 --keep-source 关闭。
MOVE_AFTER_CONVERT = True
# 训练最大序列长度，默认与 GlobalConfig.train_max_sequence_length 保持一致；转换时会用 tokenizer 预分块。
TRAIN_MAX_SEQUENCE_LENGTH = 4096
# 本地 DS tokenizer 目录。用于按 train_max_sequence_length 做 token 级分块。
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "lpt_model" / "ds_tokenizer"
DOC_CONVERT_TIMEOUT_SECONDS = 120

_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WORD_TAG = f"{{{_WORD_NAMESPACE}}}"
_SUPPORTED_WORD_SUFFIXES = frozenset({".doc", ".docx"})
_LIBREOFFICE_COMMAND_CANDIDATES = (
    "soffice",
    "libreoffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)


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


def _ensure_word_to_text_kind(kind_config):
    """Word 原文转换只能生成 text 样本，避免伪造 chat 监督数据。"""
    if kind_config.record_type != "text":
        raise ValueError(
            "DOC/DOCX 原文转换当前只支持 dataset_kind='text'。"
            "chat_sft/chat_lora 需要 question/answer 或 instruction/response 结构化字段。"
        )


def _iter_word_files(input_dir):
    """递归枚举目录下可处理的 DOC/DOCX 文件，跳过 Word 临时锁文件。"""
    return sorted(
        path
        for path in Path(input_dir).rglob("*")
        if path.is_file()
        and path.suffix.lower() in _SUPPORTED_WORD_SUFFIXES
        and not path.name.startswith("~$")
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
    """优先按段落和换行保留结构，再在 token 超长时回退切分。"""
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


def _paragraph_text(paragraph):
    """从 w:p 节点提取可见文本，保留制表符和软换行。"""
    pieces = []
    for node in paragraph.iter():
        if node.tag == f"{_WORD_TAG}t":
            pieces.append(node.text or "")
        elif node.tag == f"{_WORD_TAG}tab":
            pieces.append("\t")
        elif node.tag in {f"{_WORD_TAG}br", f"{_WORD_TAG}cr"}:
            pieces.append("\n")
    return "".join(pieces).strip()


def extract_docx_text(docx_path):
    """从 DOCX 的 word/document.xml 中提取正文段落。"""
    docx_path = Path(docx_path)
    try:
        with zipfile.ZipFile(docx_path) as archive:
            document_xml = archive.read("word/document.xml")
    except KeyError as exc:
        raise ValueError(f"{docx_path} 缺少 word/document.xml，不能按 DOCX 解析。") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{docx_path} 不是有效的 DOCX/ZIP 文件。") from exc

    root = ET.fromstring(document_xml)
    paragraphs = [
        text
        for text in (_paragraph_text(paragraph) for paragraph in root.iter(f"{_WORD_TAG}p"))
        if text
    ]
    return "\n\n".join(paragraphs).strip()


def _find_libreoffice_command():
    """查找可用于 .doc 转换的 LibreOffice/soffice 命令。"""
    for candidate in _LIBREOFFICE_COMMAND_CANDIDATES:
        resolved = shutil.which(candidate) if "\\" not in candidate else candidate
        if resolved and Path(resolved).exists():
            return resolved
    return None


def _convert_doc_to_docx_with_libreoffice(doc_path, output_dir):
    """用 LibreOffice headless 把旧 .doc 转成临时 .docx。"""
    command = _find_libreoffice_command()
    if command is None:
        raise RuntimeError("未找到 LibreOffice/soffice。")

    output_dir = Path(output_dir)
    result = subprocess.run(
        [
            command,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            str(output_dir),
            str(doc_path),
        ],
        capture_output=True,
        text=True,
        timeout=DOC_CONVERT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice 转换失败: stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )

    converted_path = output_dir / f"{Path(doc_path).stem}.docx"
    if not converted_path.is_file() or converted_path.stat().st_size <= 0:
        raise RuntimeError(f"LibreOffice 未生成有效 docx: {converted_path}")
    return converted_path


_WORD_COM_CONVERT_SCRIPT = r"""
param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $document = $word.Documents.Open($InputPath, $false, $true)
    try {
        $docxFormat = 16
        $document.SaveAs([ref]$OutputPath, [ref]$docxFormat)
    }
    finally {
        $document.Close([ref]$false)
    }
}
finally {
    $word.Quit()
}
"""


def _find_powershell_command():
    """查找可执行 Word COM 自动化的 PowerShell。"""
    return shutil.which("pwsh") or shutil.which("powershell")


def _convert_doc_to_docx_with_word_com(doc_path, output_dir):
    """在 Windows 上用本机 Microsoft Word COM 把旧 .doc 转成临时 .docx。"""
    if platform.system().lower() != "windows":
        raise RuntimeError("Word COM 只支持 Windows。")

    powershell = _find_powershell_command()
    if powershell is None:
        raise RuntimeError("未找到 PowerShell，无法调用 Word COM。")

    output_path = Path(output_dir) / f"{Path(doc_path).stem}.docx"
    script_path = Path(output_dir) / "convert_doc_to_docx.ps1"
    script_path.write_text(_WORD_COM_CONVERT_SCRIPT, encoding="utf-8")

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script_path),
            str(Path(doc_path).resolve()),
            str(output_path.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=DOC_CONVERT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Word COM 转换失败: stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Word COM 未生成有效 docx: {output_path}")
    return output_path


def extract_doc_text(doc_path):
    """把旧版 .doc 转成临时 .docx 后复用 OpenXML 抽取逻辑。"""
    errors = []
    with tempfile.TemporaryDirectory(prefix="lpt_doc_convert_") as temp_dir:
        for converter in (_convert_doc_to_docx_with_libreoffice, _convert_doc_to_docx_with_word_com):
            try:
                converted_path = converter(doc_path, temp_dir)
                return extract_docx_text(converted_path)
            except Exception as exc:
                errors.append(f"{converter.__name__}: {exc}")

    details = "\n".join(errors)
    raise RuntimeError(
        f"无法转换 DOC 文件: {doc_path}\n"
        "请安装 LibreOffice/soffice，或在 Windows 上安装 Microsoft Word 以启用 COM 转换。\n"
        f"{details}"
    )


def extract_word_text(word_path):
    """按扩展名抽取 DOC/DOCX 正文。"""
    word_path = Path(word_path)
    suffix = word_path.suffix.lower()
    if suffix == ".docx":
        return extract_docx_text(word_path)
    if suffix == ".doc":
        return extract_doc_text(word_path)
    raise ValueError(f"不支持的 Word 文件类型: {word_path}")


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


def _build_text_records(
    word_files,
    input_dir,
    *,
    source_name,
    start_file_index=1,
    train_max_sequence_length=TRAIN_MAX_SEQUENCE_LENGTH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
):
    """把多个 DOC/DOCX 文件转换为结构化 text 记录列表。"""
    if train_max_sequence_length is not None and tokenizer_path is None:
        raise ValueError("启用按 token 分块时必须提供 tokenizer_path。")

    tokenizer = None if train_max_sequence_length is None else _load_tokenizer(tokenizer_path)
    text_token_budget = None if tokenizer is None else _text_token_budget(train_max_sequence_length, tokenizer)
    records = []
    skipped_count = 0

    for file_index, word_file in enumerate(word_files, start=start_file_index):
        text = extract_word_text(word_file)
        text_chunks = [text]
        if tokenizer is not None and text:
            text_chunks = _chunk_text_by_lines(text, tokenizer, max_tokens=text_token_budget)

        if not text_chunks:
            skipped_count += 1
            continue

        for chunk_index, text_chunk in enumerate(text_chunks, start=1):
            try:
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
            except Exception:
                skipped_count += 1

    return records, skipped_count


def _write_jsonl(records, output_path):
    """以 UTF-8 无 BOM 和 LF 换行写入 JSONL。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def archive_word_files(word_files, input_dir, *, archive_root, dataset_kind):
    """把已成功转换的 DOC/DOCX 移入 z-old-data 对应类型目录。"""
    kind_config = _resolve_kind_config(dataset_kind)
    input_dir = Path(input_dir)
    archive_dir = Path(archive_root) / kind_config.archive_dir / input_dir.name
    moved_paths = []

    for word_file in word_files:
        relative_path = word_file.relative_to(input_dir)
        target_path = _next_available_path(archive_dir / relative_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        word_file.replace(target_path)
        moved_paths.append(target_path)

    return moved_paths


def convert_docx_directory_to_jsonl(
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
    """把一个目录下的全部 DOC/DOCX 合并为一个结构化 JSONL。"""
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    kind_config = _resolve_kind_config(dataset_kind)
    _ensure_word_to_text_kind(kind_config)

    word_files = _iter_word_files(input_dir)
    if not word_files:
        raise FileNotFoundError(f"未找到 DOC/DOCX 文件: {input_dir}")

    source_name = _resolve_source_name(input_dir, source_name)
    output_path = _resolve_output_path(input_dir, kind_config, output_path)
    existing_record_count = _count_existing_jsonl_records(output_path) if append_output else 0
    records, skipped_count = _build_text_records(
        word_files,
        input_dir,
        source_name=source_name,
        start_file_index=existing_record_count + 1,
        train_max_sequence_length=train_max_sequence_length,
        tokenizer_path=tokenizer_path,
    )
    if not records:
        raise ValueError(f"未从 {input_dir} 解析出任何有效 text 样本。")

    if append_output:
        _append_jsonl(records, output_path)
    else:
        _write_jsonl(records, output_path)
    moved_paths = []
    if move_to_archive:
        moved_paths = archive_word_files(
            word_files,
            input_dir,
            archive_root=archive_root,
            dataset_kind=dataset_kind,
        )

    summary = {
        "dataset_kind": dataset_kind,
        "source": source_name,
        "input_dir": str(input_dir),
        "output": str(output_path),
        "input_files": len(word_files),
        "converted": len(records),
        "skipped": skipped_count,
        "archived": len(moved_paths),
        "archive_root": str(archive_root),
        "output_mode": "append" if append_output else "overwrite",
        "existing_records": existing_record_count,
        "train_max_sequence_length": train_max_sequence_length,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_argument_parser():
    parser = ArgumentParser(description="把 pending 目录中的 DOC/DOCX 文件合并转换为一个结构化 text JSONL。")
    parser.add_argument("--input-dir", type=Path, default=PENDING_INPUT_DIR, help="待转换 DOC/DOCX 所在目录。")
    parser.add_argument(
        "--dataset-kind",
        choices=sorted(DATASET_KIND_CONFIGS),
        default=DATASET_KIND,
        help="数据类型配置；DOC/DOCX 原文转换本次使用 text。",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出 JSONL 路径；默认按目录名推导。")
    parser.add_argument("--source", default=SOURCE_NAME, help="写入 source 字段；默认使用输入目录名。")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT, help="z-old-data 根目录。")
    parser.add_argument("--keep-source", action="store_true", help="只转换，不移动 DOC/DOCX 到 z-old-data。")
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
    convert_docx_directory_to_jsonl(
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
