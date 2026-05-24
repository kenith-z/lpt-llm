import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_protocol import DS_EOS_TOKEN
from tools.convert_docx_text_dataset import convert_docx_directory_to_jsonl, extract_docx_text


class DummyTokenizer:
    """测试用字符级 tokenizer，把 EOS 视为单个 token。"""

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("测试 tokenizer 不支持 add_special_tokens=True。")
        if text == DS_EOS_TOKEN:
            return {"input_ids": ["<eos>"]}
        return {"input_ids": list(text)}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(token_ids)


def _build_minimal_docx_xml(paragraphs):
    """生成最小可解析的 DOCX 主文档 XML。"""
    body_parts = []
    for paragraph in paragraphs:
        body_parts.append(
            """
            <w:p>
              <w:r>
                <w:t>{text}</w:t>
              </w:r>
            </w:p>
            """.format(text=paragraph)
        )
    body_parts.append("<w:sectPr/>")
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
  </w:body>
</w:document>
""".format(body="".join(body_parts))


def _write_minimal_docx(path, paragraphs):
    """把最小 DOCX 直接写成 ZIP 包，避免依赖 python-docx。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", _build_minimal_docx_xml(paragraphs))


def _load_jsonl(path):
    """读取测试输出 JSONL，避免把训练批处理依赖带入转换测试。"""
    with Path(path).open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


class TestConvertDocxTextDataset(unittest.TestCase):
    def test_extract_docx_text_keeps_paragraph_boundaries(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            docx_path = Path(temp_dir) / "sample.docx"
            _write_minimal_docx(docx_path, ["第一段", "第二段"])

            text = extract_docx_text(docx_path)

        self.assertEqual(text, "第一段\n\n第二段")

    def test_convert_docx_directory_to_jsonl_moves_source_files(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "论文"
            archive_root = temp_root / "data" / "z-old-data"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "论文.text.jsonl"

            _write_minimal_docx(pending_dir / "试论AI技术在IT审计中应用.docx", ["第一段", "第二段"])
            _write_minimal_docx(pending_dir / "AI技术在IT审计中的应用研究.docx", ["第三段"])

            summary = convert_docx_directory_to_jsonl(
                pending_dir,
                output_path,
                dataset_kind="text",
                source_name="论文",
                archive_root=archive_root,
                move_to_archive=True,
                train_max_sequence_length=None,
                tokenizer_path=None,
            )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["dataset_kind"], "text")
            self.assertEqual(summary["converted"], 2)
            self.assertEqual(summary["archived"], 2)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(set(record) == {"id", "type", "text", "source"} for record in records))
            self.assertTrue(all(record["source"] == "论文" for record in records))
            self.assertEqual({record["type"] for record in records}, {"text"})
            self.assertEqual({record["text"] for record in records}, {"第一段\n\n第二段", "第三段"})
            self.assertFalse((pending_dir / "试论AI技术在IT审计中应用.docx").exists())
            self.assertFalse((pending_dir / "AI技术在IT审计中的应用研究.docx").exists())
            self.assertTrue((archive_root / "text" / "论文" / "试论AI技术在IT审计中应用.docx").exists())
            self.assertTrue((archive_root / "text" / "论文" / "AI技术在IT审计中的应用研究.docx").exists())

    def test_convert_directory_accepts_legacy_doc_files(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "法律"
            archive_root = temp_root / "data" / "z-old-data"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "法律.text.jsonl"
            doc_path = pending_dir / "legacy.doc"
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_bytes(b"legacy-doc-placeholder")

            with patch("tools.convert_docx_text_dataset.extract_doc_text", return_value="旧 DOC 正文"):
                summary = convert_docx_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="法律",
                    archive_root=archive_root,
                    move_to_archive=True,
                    train_max_sequence_length=None,
                    tokenizer_path=None,
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["input_files"], 1)
            self.assertEqual(records, [{"id": "法律-000001", "type": "text", "text": "旧 DOC 正文", "source": "法律"}])
            self.assertFalse(doc_path.exists())
            self.assertTrue((archive_root / "text" / "法律" / "legacy.doc").exists())

    def test_convert_docx_directory_appends_to_existing_jsonl(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "论文"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "论文.text.jsonl"
            pending_doc = pending_dir / "append.docx"
            _write_minimal_docx(pending_doc, ["追加正文"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "id": "论文-000001",
                        "type": "text",
                        "text": "旧正文",
                        "source": "论文",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = convert_docx_directory_to_jsonl(
                pending_dir,
                output_path,
                dataset_kind="text",
                source_name="论文",
                move_to_archive=False,
                train_max_sequence_length=None,
                tokenizer_path=None,
            )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["output_mode"], "append")
            self.assertEqual(summary["existing_records"], 1)
            self.assertEqual([record["id"] for record in records], ["论文-000001", "论文-000002"])
            self.assertEqual([record["text"] for record in records], ["旧正文", "追加正文"])

    def test_convert_docx_directory_chunks_by_train_max_sequence_length(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "论文"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "论文.text.jsonl"
            _write_minimal_docx(pending_dir / "long.docx", ["abcdefghi"])

            with patch("tools.convert_docx_text_dataset._load_tokenizer", return_value=DummyTokenizer()):
                summary = convert_docx_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="论文",
                    move_to_archive=False,
                    train_max_sequence_length=5,
                    tokenizer_path=PROJECT_ROOT / "lpt_model" / "ds_tokenizer",
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["train_max_sequence_length"], 5)
            self.assertEqual(summary["converted"], 3)
            self.assertEqual([record["text"] for record in records], ["abcd", "efgh", "i"])
            self.assertTrue(all(len(DummyTokenizer()(record["text"])["input_ids"]) <= 4 for record in records))
            self.assertTrue(all(set(record) == {"id", "type", "text", "source"} for record in records))


if __name__ == "__main__":
    unittest.main()
