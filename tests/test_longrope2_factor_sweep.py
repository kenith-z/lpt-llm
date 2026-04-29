import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import ModelConfig
from lpt_evaluation.longrope2_factor_sweep import (
    LongRoPE2FactorSweepConfig,
    build_longrope2_factor_candidates,
    format_longrope2_factor_sweep_report_markdown,
    save_longrope2_factor_sweep_report,
)


class TestLongRoPE2FactorSweep(unittest.TestCase):
    def build_model_config(self):
        return ModelConfig(
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            hidden_size=16,
            layer_block_types=("attention", "attention"),
            original_max_len=4,
            longrope2_target_length=16,
            longrope2_long_factors=(2.0, 2.0, 2.0, 2.0),
            longrope2_factor_max_sequence_length=12,
        )

    def test_build_factor_candidates_combines_current_bootstrap_uniform_and_file(self):
        model_config = self.build_model_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            factors_path = Path(temp_dir) / "searched.csv"
            factors_path.write_text("1.5,2.0,2.5,3.0", encoding="utf-8")

            candidates = build_longrope2_factor_candidates(
                model_config,
                LongRoPE2FactorSweepConfig(
                    uniform_factor_candidates=(("uniform-3x", 3.0),),
                    factor_file_candidates=(("searched", str(factors_path)),),
                ),
            )

        self.assertEqual(
            [candidate.name for candidate in candidates],
            ["current", "bootstrap", "uniform-3x", "searched"],
        )
        self.assertEqual(candidates[0].long_factors, (2.0, 2.0, 2.0, 2.0))
        self.assertEqual(candidates[1].factor_max_sequence_length, 12)
        self.assertEqual(candidates[2].long_factors, (3.0, 3.0, 3.0, 3.0))
        self.assertEqual(candidates[3].long_factors, (1.5, 2.0, 2.5, 3.0))

    def test_build_factor_candidates_rejects_duplicate_names(self):
        model_config = self.build_model_config()

        with self.assertRaisesRegex(ValueError, "名称重复"):
            build_longrope2_factor_candidates(
                model_config,
                LongRoPE2FactorSweepConfig(
                    uniform_factor_candidates=(("current", 3.0),),
                ),
            )

    def test_format_and_save_factor_sweep_report(self):
        report = {
            "report_type": "longrope2_factor_sweep",
            "checkpoint": {
                "checkpoint_path": "artifacts/example/latest.pth",
                "training_stage": "chat_sft",
                "source_manifest": "data/manifests/chat_sft.json",
            },
            "runtime": {
                "device": "cpu",
                "cache_strategy": "session_rebuild",
                "total_latency_sec": 1.25,
            },
            "candidates": [
                {
                    "name": "current",
                    "source": "checkpoint:model_config",
                    "status": "ok",
                    "longrope_factor_summary": {
                        "factor_mode": "uniform",
                        "min_factor": 2.0,
                        "max_factor": 2.0,
                    },
                    "tasks": {
                        "needle_in_a_haystack": {
                            "aggregate": {"exact_match_rate": 0.5},
                        },
                        "qa_retrieval": {
                            "aggregate": {"exact_match_rate": 1.0},
                        },
                        "long_text_ppl": {
                            "aggregate": [
                                {"window_size": 1024, "perplexity": 12.0},
                            ],
                        },
                    },
                    "runtime": {"latency_sec": 0.8},
                }
            ],
        }

        markdown = format_longrope2_factor_sweep_report_markdown(report)
        self.assertIn("LongRoPE2 候选因子 Sweep 报告", markdown)
        self.assertIn("current", markdown)
        self.assertIn("1024:12", markdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = save_longrope2_factor_sweep_report(
                report,
                checkpoint_root=Path(temp_dir) / "latest",
                output_dir=temp_dir,
                output_format="both",
            )

            self.assertIn("json", saved_paths)
            self.assertIn("markdown", saved_paths)
            json_payload = json.loads(Path(saved_paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(json_payload["report_type"], report.get("report_type"))
            self.assertIn(
                "LongRoPE2 候选因子 Sweep 报告",
                Path(saved_paths["markdown"]).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
