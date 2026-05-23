import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_protocol import DS_EOS_TOKEN
from tools.convert_epub_text_dataset import convert_epub_directory_to_jsonl, extract_epub_text, html_to_text


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


class FakeEpubItem:
    """模拟 ebooklib 的 EPUB 文档项。"""

    def __init__(self, item_id, name, content, item_type, properties=None):
        self._item_id = item_id
        self._name = name
        self._content = content
        self._item_type = item_type
        self.properties = properties or []

    def get_id(self):
        return self._item_id

    def get_name(self):
        return self._name

    def get_type(self):
        return self._item_type

    def get_content(self):
        return self._content


class FakeEpubBook:
    """模拟 ebooklib 读取出的 EPUB 书籍对象。"""

    def __init__(self, items, spine):
        self._items = {item.get_id(): item for item in items}
        self.spine = spine

    def get_item_with_id(self, item_id):
        return self._items.get(item_id)

    def get_items_of_type(self, item_type):
        return [item for item in self._items.values() if item.get_type() == item_type]


def _write_dummy_epub(path):
    """写入占位 EPUB 文件；内容提取在测试中通过 mock 控制。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"EPUB")


def _load_jsonl(path):
    """读取测试输出 JSONL。"""
    with Path(path).open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


class TestConvertEpubTextDataset(unittest.TestCase):
    def test_html_to_text_keeps_block_boundaries(self):
        text = html_to_text(
            """
            <html>
              <head><title>忽略标题</title><style>.x { color: red; }</style></head>
              <body>
                <h1>第一章</h1>
                <p>第一 <b>段</b></p>
                <ul><li>条目A</li><li>条目B</li></ul>
              </body>
            </html>
            """
        )

        self.assertEqual(text, "第一章\n\n第一 段\n\n- 条目A\n\n- 条目B")

    def test_extract_epub_text_uses_ebooklib_spine_order_and_skips_nav(self):
        item_type = 9
        items = [
            FakeEpubItem(
                "nav",
                "nav.xhtml",
                "<html><body><nav><p>目录页</p></nav></body></html>".encode("utf-8"),
                item_type,
                properties=["nav"],
            ),
            FakeEpubItem(
                "chapter-1",
                "chapter1.xhtml",
                "<html><body><p>第一章</p></body></html>".encode("utf-8"),
                item_type,
            ),
            FakeEpubItem(
                "chapter-2",
                "chapter2.xhtml",
                "<html><body><p>第二章</p></body></html>".encode("utf-8"),
                item_type,
            ),
        ]
        book = FakeEpubBook(items, spine=[("nav", "yes"), ("chapter-2", "yes"), ("chapter-1", "yes")])
        fake_ebooklib = SimpleNamespace(
            ITEM_DOCUMENT=item_type,
            epub=SimpleNamespace(read_epub=lambda path: book),
        )

        with patch.dict(sys.modules, {"ebooklib": fake_ebooklib}):
            text = extract_epub_text(PROJECT_ROOT / "unused.epub")

        self.assertEqual(text, "第二章\n\n第一章")

    def test_convert_epub_directory_appends_and_archives_each_successful_file(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "epub"
            archive_root = temp_root / "data" / "z-old-data"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "epub.text.jsonl"
            text_epub = pending_dir / "book.epub"
            empty_epub = pending_dir / "empty.epub"
            _write_dummy_epub(text_epub)
            _write_dummy_epub(empty_epub)

            def fake_extract(path):
                return "EPUB 正文" if Path(path).name == "book.epub" else ""

            with patch("tools.convert_epub_text_dataset.extract_epub_text", side_effect=fake_extract):
                summary = convert_epub_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="epub",
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
            self.assertEqual(records, [{"id": "epub-000001", "type": "text", "text": "EPUB 正文", "source": "epub"}])
            self.assertFalse(text_epub.exists())
            self.assertTrue(empty_epub.exists())
            self.assertTrue((archive_root / "text" / "epub" / "book.epub").exists())

    def test_convert_epub_directory_appends_to_existing_jsonl(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "epub"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "epub.text.jsonl"
            _write_dummy_epub(pending_dir / "append.epub")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "id": "epub-000001",
                        "type": "text",
                        "text": "旧 EPUB",
                        "source": "epub",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("tools.convert_epub_text_dataset.extract_epub_text", return_value="追加 EPUB"):
                summary = convert_epub_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="epub",
                    move_to_archive=False,
                    train_max_sequence_length=None,
                    tokenizer_path=None,
                )
            records = _load_jsonl(output_path)

            self.assertEqual(summary["output_mode"], "append")
            self.assertEqual(summary["existing_records"], 1)
            self.assertEqual([record["id"] for record in records], ["epub-000001", "epub-000002"])
            self.assertEqual([record["text"] for record in records], ["旧 EPUB", "追加 EPUB"])

    def test_convert_epub_directory_chunks_by_train_max_sequence_length(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temp_dir:
            temp_root = Path(temp_dir)
            pending_dir = temp_root / "data" / "z-pending-data" / "epub"
            output_path = temp_root / "data" / "structured" / "text_pretrain" / "epub.text.jsonl"
            _write_dummy_epub(pending_dir / "long.epub")

            with patch("tools.convert_epub_text_dataset._load_tokenizer", return_value=DummyTokenizer()), patch(
                "tools.convert_epub_text_dataset.extract_epub_text",
                return_value="abcdefghi",
            ):
                summary = convert_epub_directory_to_jsonl(
                    pending_dir,
                    output_path,
                    dataset_kind="text",
                    source_name="epub",
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
