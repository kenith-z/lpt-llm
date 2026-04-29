import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_evaluation.long_context import (
    build_needle_case,
    build_retrieval_case,
    build_text_ppl_windows,
    format_long_context_report_markdown,
    save_long_context_report,
    summarize_longrope_factors,
)


class SimpleTokenizer:
    def __init__(self):
        self.token_to_id = {"<eos>": 0, "<pad>": 1}
        self.id_to_token = {0: "<eos>", 1: "<pad>"}
        self.next_id = 2
        self.eos_token_id = 0
        self.pad_token_id = 1

    def _encode(self, text):
        pieces = [piece for piece in text.replace("\n", " \n ").split(" ") if piece]
        token_ids = []
        for piece in pieces:
            if piece not in self.token_to_id:
                self.token_to_id[piece] = self.next_id
                self.id_to_token[self.next_id] = piece
                self.next_id += 1
            token_ids.append(self.token_to_id[piece])
        return token_ids

    def __call__(self, texts, add_special_tokens=False, return_tensors=None, return_attention_mask=False):
        if isinstance(texts, str):
            encoded = self._encode(texts)
            result = {"input_ids": encoded}
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor([encoded], dtype=torch.long)
                if return_attention_mask:
                    result["attention_mask"] = torch.ones((1, len(encoded)), dtype=torch.long)
            return result
        raise TypeError("SimpleTokenizer 仅支持单条字符串输入。")

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = []
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in {"<eos>", "<pad>"}:
                continue
            pieces.append(token)
        return " ".join(pieces)


class TestLongContextEvaluationHelpers(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizer()

    def test_summarize_longrope_factors_distinguishes_uniform_and_per_dimension(self):
        uniform_summary = summarize_longrope_factors([2.0, 2.0, 2.0])
        per_dimension_summary = summarize_longrope_factors([1.5, 2.0, 2.5])

        self.assertEqual(uniform_summary["factor_mode"], "uniform")
        self.assertEqual(uniform_summary["uniform_factor"], 2.0)
        self.assertEqual(per_dimension_summary["factor_mode"], "per_dimension")
        self.assertEqual(per_dimension_summary["unique_factor_count"], 3)

    def test_build_needle_case_reaches_target_token_length(self):
        case = build_needle_case(self.tokenizer, 80, 0.5, seed=7)

        token_count = len(self.tokenizer(case["prompt_text"], add_special_tokens=False)["input_ids"])
        self.assertGreaterEqual(token_count, 80)
        self.assertIn(case["expected_answer"], case["prompt_text"])

    def test_build_retrieval_case_reaches_target_token_length(self):
        case = build_retrieval_case(self.tokenizer, 90, seed=9)

        token_count = len(self.tokenizer(case["prompt_text"], add_special_tokens=False)["input_ids"])
        self.assertGreaterEqual(token_count, 90)
        self.assertIn("密钥", case["prompt_text"])

    def test_build_text_ppl_windows_packs_fixed_size_windows(self):
        records = [
            {"type": "text", "text": "甲 乙 丙 丁 戊 己 庚 辛"},
            {"type": "text", "text": "壬 癸 子 丑 寅 卯 辰 巳"},
        ]

        windows = build_text_ppl_windows(records, self.tokenizer, 6, max_windows=2)

        self.assertEqual(len(windows), 2)
        self.assertTrue(all(len(window) == 6 for window in windows))

    def test_format_and_save_report_outputs_json_and_markdown(self):
        report = {
            "checkpoint": {
                "checkpoint_path": "artifacts/example/latest.pth",
                "checkpoint_schema_version": 1,
                "model_config_schema_version": 1,
                "training_stage": "chat_sft",
                "training_mode": "full",
                "source_manifest": "data/manifests/chat_sft.json",
                "model_config": {"num_layers": 2},
                "longrope_factor_summary": {
                    "factor_mode": "uniform",
                    "factor_count": 32,
                    "unique_factor_count": 1,
                    "min_factor": 2.0,
                    "max_factor": 2.0,
                },
            },
            "runtime": {
                "device": "cpu",
                "cache_strategy": "session_rebuild",
                "total_latency_sec": 1.23,
            },
            "tasks": {
                "needle_in_a_haystack": {
                    "aggregate": {
                        "sample_count": 1,
                        "success_count": 1,
                        "error_count": 0,
                        "exact_match_rate": 1.0,
                        "average_latency_sec": 0.1,
                    },
                    "samples": [
                        {
                            "target_sequence_length": 1024,
                            "needle_depth": 0.5,
                            "status": "ok",
                            "exact_match": True,
                            "latency_sec": 0.1,
                            "prediction": "NIAH-0001",
                        }
                    ],
                },
                "long_text_ppl": {
                    "aggregate": [
                        {
                            "window_size": 1024,
                            "window_count": 2,
                            "effective_token_count": 2046,
                            "perplexity": 12.34,
                        }
                    ]
                },
                "qa_retrieval": {
                    "aggregate": {
                        "sample_count": 1,
                        "success_count": 1,
                        "error_count": 0,
                        "exact_match_rate": 1.0,
                        "average_latency_sec": 0.2,
                    },
                    "samples": [
                        {
                            "target_sequence_length": 2048,
                            "status": "ok",
                            "exact_match": True,
                            "latency_sec": 0.2,
                            "prediction": "KEY-0001-001",
                        }
                    ],
                },
            },
        }

        markdown = format_long_context_report_markdown(report)
        self.assertIn("长上下文评测报告", markdown)
        self.assertIn("Needle In A Haystack", markdown)
        self.assertIn("Long Text PPL", markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_long_context_report(
                report,
                checkpoint_root=Path(temp_dir) / "latest",
                output_dir=temp_dir,
                output_format="both",
            )

            self.assertIn("json", saved_paths)
            self.assertIn("markdown", saved_paths)

            json_payload = json.loads(Path(saved_paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(json_payload["runtime"]["cache_strategy"], "session_rebuild")
            self.assertIn("长上下文评测报告", Path(saved_paths["markdown"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
