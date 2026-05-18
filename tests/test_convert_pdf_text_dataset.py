import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_protocol import DS_EOS_TOKEN
from tools.convert_pdf_text_dataset import convert_pdf_directory_to_jsonl, extract_pdf_text, table_to_markdown


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


def _write_dummy_pdf(path):
    """写入占位 PDF 文件；内容提取在测试中通过 mock 控制。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")


def _load_jsonl(path):
    """读取测试输出 JSONL。"""
    with Path(path).open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


class TestConvertPdfTextDataset(unittest.TestCase):
    def test_table_to_markdown_escapes_cells(self):
        markdown = table_to_markdown(
            [
                ["字段", "说明"],
                ["A|B", "第一行\n第二行"],
            ]
        )

        self.assertEqual(
            markdown,
            "| 字段 | 说明 |\n"
            "| --- | --- |\n"
            "| A\\|B | 第一行<br>第二行 |",
        )

    def test_extract_pdf_text_uses_pdfplumber_tables_as_markdown(self):
        class FakePage:
            def extract_text(self):
                return "页面正文"

            def extract_tables(self):
                return [[["字段", "值"], ["名称", "灵预"]]]

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_pdfplumber = SimpleNamespace(open=lambda path: FakePdf())
        with patch.dict(sys.modules, {"pdfplumber": fake_pdfplumber}):
            text = extract_pdf_text(PROJECT_ROOT / "unused.pdf")

        self.assertIn("页面正文", text)
        self.assertIn("| 字段 | 值 |", text)
        self.assertIn("| 名称 | 灵预 |", text)

    def test_extract_pdf_text_reads_pdfplumber_text(self):
        class FakePage:
            def extract_text(self):
                return "Hello PDF"

            def extract_tables(self):
                return []

        class FakePdf:
            pages = [FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        fake_pdfplumber = SimpleNamespace(open=lambda path: FakePdf())
        with patch.dict(sys.modules, {"pdfplumber": fake_pdfplumber}):
            text = extract_pdf_text(PROJECT_ROOT / "unused.pdf")

        self.assertEqual(text, "Hello PDF")

    def test_convert_pdf_directory_skips_image_only_pdf(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "pending-data" / "pdf"
            archive_root = temp_root / "data" / "old-data"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "pdf.text.jsonl"
            text_pdf = pending_dir / "text.pdf"
            image_pdf = pending_dir / "image.pdf"
            _write_dummy_pdf(text_pdf)
            _write_dummy_pdf(image_pdf)

            def fake_extract(path):
                return "PDF 正文" if Path(path).name == "text.pdf" else ""

            with patch("tools.convert_pdf_text_dataset.extract_pdf_text", side_effect=fake_extract):
                summary = convert_pdf_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="pdf",
                    archive_root=archive_root,
                    move_to_archive=True,
                    train_max_sequence_length=None,
                    tokenizer_path=None,
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["input_files"], 2)
            self.assertEqual(summary["converted"], 1)
            self.assertEqual(summary["converted_files"], 1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(summary["archived"], 1)
            self.assertEqual(records, [{"id": "pdf-000001", "type": "text", "text": "PDF 正文", "source": "pdf"}])
            self.assertFalse(text_pdf.exists())
            self.assertTrue(image_pdf.exists())
            self.assertTrue((archive_root / "text" / "pdf" / "text.pdf").exists())

    def test_convert_pdf_directory_persists_each_file_before_next_pdf(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "pending-data" / "pdf"
            archive_root = temp_root / "data" / "old-data"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "pdf.text.jsonl"
            first_pdf = pending_dir / "first.pdf"
            second_pdf = pending_dir / "second.pdf"
            _write_dummy_pdf(first_pdf)
            _write_dummy_pdf(second_pdf)

            def fake_extract(path):
                if Path(path).name == "second.pdf":
                    raise RuntimeError("模拟第二个 PDF 抽取失败")
                return "第一份"

            with patch("tools.convert_pdf_text_dataset.extract_pdf_text", side_effect=fake_extract):
                summary = convert_pdf_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="pdf",
                    archive_root=archive_root,
                    move_to_archive=True,
                    train_max_sequence_length=None,
                    tokenizer_path=None,
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["converted"], 1)
            self.assertEqual(summary["converted_files"], 1)
            self.assertEqual(summary["skipped"], 1)
            self.assertEqual(records, [{"id": "pdf-000001", "type": "text", "text": "第一份", "source": "pdf"}])
            self.assertFalse(first_pdf.exists())
            self.assertTrue(second_pdf.exists())
            self.assertTrue((archive_root / "text" / "pdf" / "first.pdf").exists())

    def test_convert_pdf_directory_appends_to_existing_jsonl(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "pending-data" / "pdf"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "pdf.text.jsonl"
            _write_dummy_pdf(pending_dir / "append.pdf")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "id": "pdf-000001",
                        "type": "text",
                        "text": "旧 PDF",
                        "source": "pdf",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("tools.convert_pdf_text_dataset.extract_pdf_text", return_value="追加 PDF"):
                summary = convert_pdf_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="pdf",
                    move_to_archive=False,
                    train_max_sequence_length=None,
                    tokenizer_path=None,
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["output_mode"], "append")
            self.assertEqual(summary["existing_records"], 1)
            self.assertEqual([record["id"] for record in records], ["pdf-000001", "pdf-000002"])
            self.assertEqual([record["text"] for record in records], ["旧 PDF", "追加 PDF"])

    def test_convert_pdf_directory_chunks_by_train_max_sequence_length(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "pending-data" / "pdf"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "pdf.text.jsonl"
            _write_dummy_pdf(pending_dir / "long.pdf")

            with patch("tools.convert_pdf_text_dataset._load_tokenizer", return_value=DummyTokenizer()), patch(
                "tools.convert_pdf_text_dataset.extract_pdf_text",
                return_value="abcdefghi",
            ):
                summary = convert_pdf_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="pdf",
                    move_to_archive=False,
                    train_max_sequence_length=5,
                    tokenizer_path=PROJECT_ROOT / "lpt_model" / "ds_tokenizer",
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["train_max_sequence_length"], 5)
            self.assertEqual(summary["converted"], 3)
            self.assertEqual([record["text"] for record in records], ["abcd", "efgh", "i"])
            self.assertTrue(all(set(record) == {"id", "type", "text", "source"} for record in records))


if __name__ == "__main__":
    unittest.main()
