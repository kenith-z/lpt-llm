import re
import sys
import tempfile
import unittest
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_data import build_streaming_manifest_dataset, load_dataset_manifest, load_dataset_records
from lpt_config import GlobalConfig
from lpt_protocol import (
    DS_BOS_TOKEN,
    DS_EOS_TOKEN,
    DS_PAD_TOKEN,
    render_prompt_from_messages,
    render_training_segments,
)
from lpt_training import (
    build_packed_training_batch,
    build_training_batch,
    encode_training_sample,
    prepare_tokenizer,
)
from tools.convert_instruction_chat_jsonl import convert_instruction_chat_jsonl
from tools.convert_parquet_text_dataset import convert_parquet_text_dataset
from tools.convert_raw_text_jsonl import convert_raw_text_jsonl


class DummyTokenizer:
    def __init__(self):
        self.special_tokens = (
            DS_BOS_TOKEN,
            DS_PAD_TOKEN,
            DS_EOS_TOKEN,
        )
        self.pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        )
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.special_tokens, start=1)
        }
        self.next_id = len(self.token_to_id) + 1
        self.bos_token = DS_BOS_TOKEN
        self.eos_token = DS_EOS_TOKEN
        self.pad_token = DS_PAD_TOKEN
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.name_or_path = "dummy-tokenizer"

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token)

    def _split_text(self, text):
        parts = []
        cursor = 0
        for match in self.pattern.finditer(text):
            if match.start() > cursor:
                parts.extend(list(text[cursor:match.start()]))
            parts.append(match.group(0))
            cursor = match.end()
        if cursor < len(text):
            parts.extend(list(text[cursor:]))
        return [part for part in parts if part]

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("DummyTokenizer 不支持 add_special_tokens=True。")

        input_ids = []
        for piece in self._split_text(text):
            if piece not in self.token_to_id:
                self.token_to_id[piece] = self.next_id
                self.next_id += 1
            input_ids.append(self.token_to_id[piece])
        return {"input_ids": input_ids}


class TestStructuredDataPipeline(unittest.TestCase):
    def setUp(self):
        self.tokenizer = prepare_tokenizer(DummyTokenizer())

    def test_prepare_tokenizer_loads_ds_tokenizer(self):
        tokenizer = prepare_tokenizer(
            AutoTokenizer.from_pretrained(
                str(PROJECT_ROOT / "lpt_model" / "ds_tokenizer"),
                trust_remote_code=True,
                local_files_only=True,
            )
        )

        self.assertEqual(tokenizer.bos_token, DS_BOS_TOKEN)
        self.assertEqual(tokenizer.eos_token, DS_EOS_TOKEN)
        self.assertEqual(tokenizer.pad_token, DS_PAD_TOKEN)
        self.assertNotEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)
        encoded = tokenizer(
            render_prompt_from_messages(
                [{"role": "user", "content": "你好"}],
                template_version=GlobalConfig.chat_template_version,
                add_generation_prompt=True,
            ),
            add_special_tokens=False,
        )
        self.assertTrue(encoded["input_ids"])

    def test_render_prompt_from_messages_uses_versioned_ds_template(self):
        prompt = render_prompt_from_messages(
            [
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "你好"},
            ],
            template_version=GlobalConfig.chat_template_version,
            add_generation_prompt=True,
        )
        self.assertEqual(
            prompt,
            f"{DS_BOS_TOKEN}System: 你是助手\n\nUser: 你好\n\nAssistant: ",
        )

    def test_render_training_segments_marks_only_assistant_reply_as_supervised(self):
        segments = render_training_segments(
            {
                "id": "chat-001",
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "世界"},
                ],
            },
            template_version=GlobalConfig.chat_template_version,
        )

        self.assertEqual(
            [(segment.text, segment.supervise) for segment in segments],
            [
                (DS_BOS_TOKEN, False),
                ("\n\nUser: ", False),
                ("你好", False),
                ("\n\nAssistant: ", False),
                ("世界", True),
                (DS_EOS_TOKEN, True),
            ],
        )

    def test_build_training_batch_masks_prompt_and_padding_only(self):
        chat_sample = {
            "id": "chat-001",
            "type": "chat",
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "世界"},
            ],
        }
        text_sample = {
            "id": "text-001",
            "type": "text",
            "text": "知识",
        }

        input_ids, labels, attention_mask = build_training_batch(
            [chat_sample, text_sample],
            self.tokenizer,
        )

        chat_segments = render_training_segments(chat_sample, GlobalConfig.chat_template_version)
        chat_expected_labels = []
        for segment in chat_segments:
            segment_ids = self.tokenizer(segment.text, add_special_tokens=False)["input_ids"]
            if segment.supervise:
                chat_expected_labels.extend(segment_ids)
            else:
                chat_expected_labels.extend([-100] * len(segment_ids))

        text_segments = render_training_segments(text_sample, GlobalConfig.chat_template_version)
        text_expected_labels = []
        for segment in text_segments:
            segment_ids = self.tokenizer(segment.text, add_special_tokens=False)["input_ids"]
            if segment.supervise:
                text_expected_labels.extend(segment_ids)
            else:
                text_expected_labels.extend([-100] * len(segment_ids))

        chat_length = len(chat_expected_labels)
        text_length = len(text_expected_labels)

        self.assertEqual(labels[0, :chat_length].tolist(), chat_expected_labels)
        self.assertTrue(all(value == -100 for value in labels[0, chat_length:].tolist()))
        self.assertEqual(attention_mask[0, :chat_length].tolist(), [1] * chat_length)
        self.assertTrue(all(value == 0 for value in attention_mask[0, chat_length:].tolist()))

        self.assertEqual(labels[1, :text_length].tolist(), text_expected_labels)
        self.assertTrue(all(value == -100 for value in labels[1, text_length:].tolist()))
        self.assertEqual(attention_mask[1, :text_length].tolist(), [1] * text_length)
        self.assertTrue(all(value == 0 for value in attention_mask[1, text_length:].tolist()))
        self.assertEqual(input_ids[1, text_length:].tolist(), [self.tokenizer.pad_token_id] * (input_ids.size(1) - text_length))

    def test_build_packed_training_batch_resets_positions_and_keeps_segment_boundaries(self):
        chat_sample = {
            "id": "chat-001",
            "type": "chat",
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "世界"},
            ],
        }
        text_sample = {
            "id": "text-001",
            "type": "text",
            "text": "知识",
        }

        (
            input_ids,
            labels,
            attention_mask,
            position_ids,
            segment_ids,
            sample_count,
        ) = build_packed_training_batch(
            [chat_sample, text_sample],
            self.tokenizer,
            max_length=64,
        )

        chat_encoded = encode_training_sample(chat_sample, self.tokenizer, max_length=64)
        text_encoded = encode_training_sample(text_sample, self.tokenizer, max_length=64)
        chat_length = chat_encoded.length
        text_length = text_encoded.length

        self.assertEqual(sample_count, 2)
        self.assertEqual(input_ids.size(0), 1)
        self.assertEqual(attention_mask[0, :chat_length + text_length].tolist(), [1] * (chat_length + text_length))
        self.assertEqual(position_ids[0, :chat_length].tolist(), list(range(chat_length)))
        self.assertEqual(
            position_ids[0, chat_length:chat_length + text_length].tolist(),
            list(range(text_length)),
        )
        self.assertEqual(segment_ids[0, :chat_length].tolist(), [1] * chat_length)
        self.assertEqual(
            segment_ids[0, chat_length:chat_length + text_length].tolist(),
            [2] * text_length,
        )
        self.assertEqual(labels[0, :chat_length].tolist(), list(chat_encoded.labels))
        self.assertEqual(
            labels[0, chat_length:chat_length + text_length].tolist(),
            list(text_encoded.labels),
        )

    def test_load_dataset_records_supports_structured_text_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "sample.text.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "text-001",
                        "type": "text",
                        "text": "教材内容",
                        "source": "unit-test",
                        "split": "train",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_dataset_records(dataset_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "text")
        self.assertEqual(records[0]["text"], "教材内容")
        self.assertEqual(records[0]["source"], "unit-test")

    def test_convert_raw_text_jsonl_preserves_extra_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "textbook.jsonl"
            structured_path = Path(temp_dir) / "textbook.text.jsonl"
            raw_path.write_text(
                json.dumps(
                    {
                        "id": "book-001",
                        "content": "第一课",
                        "data_type": "markdown",
                        "language": "zh",
                        "subset": "xiaoxue",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            convert_raw_text_jsonl(raw_path, structured_path, source_name="textbook")
            records = load_dataset_records(structured_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "text")
        self.assertEqual(records[0]["text"], "第一课")
        self.assertEqual(records[0]["source"], "textbook")
        self.assertEqual(records[0]["data_type"], "markdown")
        self.assertEqual(records[0]["subset"], "xiaoxue")

    def test_load_dataset_manifest_supports_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "structured"
            manifest_dir = temp_root / "manifests"
            dataset_dir.mkdir()
            manifest_dir.mkdir()

            dataset_path = dataset_dir / "sample.text.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "text-001",
                        "type": "text",
                        "text": "相对路径样本",
                        "source": "unit-test",
                        "split": "train",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path = manifest_dir / "text_pretrain.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "sample",
                                "path": "../structured/sample.text.jsonl",
                                "weight": 1.0,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            records, loaded_datasets = load_dataset_manifest(
                manifest_path,
                expected_types={"text"},
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["text"], "相对路径样本")
        self.assertEqual(loaded_datasets[0]["name"], "sample")

    def test_load_dataset_manifest_applies_weight_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_path = temp_root / "sample.text.jsonl"
            manifest_path = temp_root / "manifest.json"
            dataset_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"text-{index:03d}",
                            "type": "text",
                            "text": f"样本{index}",
                            "source": "unit-test",
                            "split": "train",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index in range(4)
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "sample",
                                "path": "sample.text.jsonl",
                                "weight": 0.5,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            records, loaded_datasets = load_dataset_manifest(
                manifest_path,
                expected_types={"text"},
            )

        self.assertEqual(len(records), 2)
        self.assertEqual(loaded_datasets[0]["count"], 2)

    def test_build_streaming_manifest_dataset_supports_zero_weight_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_a = temp_root / "sample_a.text.jsonl"
            dataset_b = temp_root / "sample_b.text.jsonl"
            manifest_path = temp_root / "manifest.json"

            dataset_a.write_text(
                json.dumps(
                    {
                        "id": "text-a",
                        "type": "text",
                        "text": "不会被加载",
                        "source": "unit-test-a",
                        "split": "train",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            dataset_b.write_text(
                json.dumps(
                    {
                        "id": "text-b",
                        "type": "text",
                        "text": "会被加载",
                        "source": "unit-test-b",
                        "split": "train",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "name": "disabled-by-weight",
                                "path": "sample_a.text.jsonl",
                                "weight": 0,
                            },
                            {
                                "name": "enabled",
                                "path": "sample_b.text.jsonl",
                                "weight": 1.0,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            dataset = build_streaming_manifest_dataset(
                manifest_path,
                expected_types={"text"},
                shuffle_buffer_size=1,
                seed=7,
            )
            records = list(dataset)

        self.assertEqual(len(dataset), 1)
        self.assertEqual([record["id"] for record in records], ["text-b"])
        self.assertEqual(dataset.loaded_datasets[0]["count"], 0)
        self.assertEqual(dataset.loaded_datasets[1]["count"], 1)
        self.assertEqual(dataset.summary_types, {"text": 1})
        self.assertEqual(dataset.summary_sources, {"unit-test-b": 1})

    def test_convert_instruction_chat_jsonl_splits_lora_subset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "chat.jsonl"
            sft_output = temp_root / "chat.sft.jsonl"
            lora_output = temp_root / "chat.lora.jsonl"
            input_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "instruction": f"问题{index}",
                            "context": "" if index % 2 == 0 else f"上下文{index}",
                            "response": f"回答{index}",
                            "category": "test",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index in range(10)
                ),
                encoding="utf-8",
            )

            convert_instruction_chat_jsonl(
                input_path,
                sft_output,
                source_name="chat-test",
                lora_output_path=lora_output,
                lora_count=3,
                seed=7,
            )
            sft_records = load_dataset_records(sft_output)
            lora_records = load_dataset_records(lora_output)

        self.assertEqual(len(sft_records), 7)
        self.assertEqual(len(lora_records), 3)
        self.assertTrue(all(record["type"] == "chat" for record in sft_records + lora_records))
        self.assertTrue(all(record["messages"][0]["role"] == "user" for record in sft_records + lora_records))
        self.assertTrue(all(record["messages"][1]["role"] == "assistant" for record in sft_records + lora_records))

    def test_convert_parquet_text_dataset_generates_structured_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            parquet_path = temp_root / "sample.parquet"
            output_path = temp_root / "sample.text.jsonl"

            table = pa.table(
                {
                    "text": [
                        "第一行\n第二行",
                        "这是另一段教材内容",
                    ],
                    "score": [0.1, 0.2],
                    "source": ["source-a", "source-a"],
                }
            )
            pq.write_table(table, parquet_path)

            convert_parquet_text_dataset(
                parquet_path,
                output_path,
                source_name="sample",
                max_tokens=64,
                tokenizer_path=PROJECT_ROOT / "lpt_model" / "ds_tokenizer",
            )
            records = load_dataset_records(output_path)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["type"] == "text" for record in records))
        self.assertEqual(records[0]["text"], "第一行\n第二行")
        self.assertEqual(records[0]["source"], "sample")
