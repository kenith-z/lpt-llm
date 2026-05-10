import sys
import unittest
from pathlib import Path
import uuid

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_eval import (
    run_lpt_v2_baselines,
    run_lpt_v2_forward_smoke_report,
    run_lpt_v2_long_context_admission,
    run_lpt_v2_resource_report,
)
from lpt_config import GlobalConfig, ModelConfig
from lpt_model import LPTV2, save_lpt_v2_checkpoint


def build_workspace_tmp_dir(name):
    path = PROJECT_ROOT / ".tmp_unittest" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_tiny_config(**overrides):
    payload = {
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "layer_block_types": ("attention", "attention"),
        "attention_window_size": 4,
        "page_block_size": 2,
        "retnet_assist_layers": "all_layers",
        "retnet_state_dim": 4,
        "retnet_adapter_rank": 2,
        "moe_num_experts": 2,
        "moe_top_k": 1,
        "original_max_len": 4,
        "longrope2_target_length": 12,
    }
    payload.update(overrides)
    return ModelConfig.from_preset("lpt_v2_dev_tiny", **payload)


class TestLPTV2EvalReports(unittest.TestCase):
    def test_baseline_report_covers_profiles_and_markdown(self):
        report = run_lpt_v2_baselines(
            profiles="lpt_v2_bootstrap,lpt_v2_assist,lpt_v2_memory",
            vocabulary_size=32,
            sequence_length=4,
            decode_steps=1,
            device="cpu",
            dtype="fp32",
        )

        payload = report.to_dict()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["results"]), 3)
        self.assertIn("lpt_v2_assist", report.to_markdown())
        self.assertTrue(all(result["logits_shape"] == [1, 4, 32] for result in payload["results"]))

    def test_long_context_admission_reports_mechanism_and_quality_decision(self):
        report = run_lpt_v2_long_context_admission(
            vocabulary_size=32,
            attention_window_size=4,
            sequence_length=10,
            device="cpu",
            dtype="fp32",
        )

        payload = report.to_dict()
        self.assertEqual(payload["metrics"]["mechanism"]["assist_retnet_token_count"], 10)
        self.assertEqual(payload["metrics"]["mechanism"]["paged_kv_window_token_count"], 4)
        self.assertIn(payload["metrics"]["quality_decision"]["status"], {
            "admit_quality_benefit",
            "admit_instrumentation_only",
            "close_or_debug",
        })
        self.assertIn("Long Context", report.to_markdown())

    def test_long_context_admission_loads_real_checkpoint(self):
        GlobalConfig.parameter_dtype = torch.float32
        model = LPTV2(32, build_tiny_config())
        checkpoint_path = build_workspace_tmp_dir("long_context_checkpoint") / "model.pt"
        save_lpt_v2_checkpoint(
            model,
            checkpoint_path,
            extra_metadata={
                "training_stage": "unit_text_pretrain",
                "global_step": 1,
                "tokenizer_metadata": {"vocab_size": 32, "chat_template_version": "lpt-ds-v1"},
            },
        )

        report = run_lpt_v2_long_context_admission(
            checkpoint_path=checkpoint_path,
            attention_window_size=4,
            sequence_length=10,
            device="cpu",
            dtype="fp32",
        )

        payload = report.to_dict()
        self.assertEqual(payload["checkpoint_metadata"]["training_stage"], "unit_text_pretrain")
        self.assertEqual(payload["metrics"]["mechanism"]["assist_retnet_token_count"], 10)
        self.assertEqual(payload["metrics"]["long_text_ppl"]["no_assist_ppl"], None)
        self.assertIn("checkpoint:", report.to_markdown())

    def test_forward_smoke_loads_real_checkpoint(self):
        GlobalConfig.parameter_dtype = torch.float32
        model = LPTV2(
            32,
            build_tiny_config(
                xlstm_memory_enabled=True,
                xlstm_memory_layers="all_layers",
                xlstm_memory_state_dim=4,
                xlstm_memory_adapter_rank=2,
                moe_router_input_mode="memory_augmented_input",
            ),
        )
        checkpoint_path = build_workspace_tmp_dir("forward_smoke_checkpoint") / "model.pt"
        save_lpt_v2_checkpoint(
            model,
            checkpoint_path,
            extra_metadata={"training_stage": "unit_forward_smoke", "global_step": 1},
        )

        report = run_lpt_v2_forward_smoke_report(
            checkpoint_path=checkpoint_path,
            sequence_length=4,
            device="cpu",
            dtype="fp32",
        )

        metrics = report.to_dict()["metrics"]
        self.assertTrue(metrics["forward_ok"])
        self.assertEqual(metrics["logits_shape"], [1, 4, 32])
        self.assertEqual(metrics["state_count"], 2)
        self.assertEqual(metrics["xlstm_state_count"], 2)
        self.assertEqual(metrics["expected_xlstm_state_count"], 2)
        self.assertEqual(metrics["paged_kv_page_count"], 0)
        self.assertIn("Forward Smoke", report.to_markdown())

    def test_resource_report_contains_runtime_metrics(self):
        report = run_lpt_v2_resource_report(
            profile="lpt_v2_assist",
            vocabulary_size=32,
            sequence_length=4,
            decode_steps=2,
            device="cpu",
            dtype="fp32",
            attention_window_size=4,
            page_block_size=2,
            original_max_len=4,
            longrope2_target_length=8,
        )

        metrics = report.to_dict()["metrics"]
        self.assertGreater(metrics["prefill_tokens_per_sec"], 0.0)
        self.assertGreater(metrics["decode_tokens_per_sec"], 0.0)
        self.assertEqual(metrics["paged_kv_runtime_metadata"]["allocated_page_count"], 8)
        self.assertEqual(len(metrics["per_layer_ms"]), 4)
        self.assertIn("Resource Report", report.to_markdown())


if __name__ == "__main__":
    unittest.main()
