import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import ModelConfig
from lpt_eval import run_lpt_v2_memory_assist_report
from lpt_model import LPTV2


def build_memory_config(**overrides):
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
        "longrope2_target_length": 8,
        "xlstm_memory_enabled": True,
        "xlstm_memory_layers": "all_layers",
        "xlstm_memory_state_dim": 4,
        "xlstm_memory_adapter_rank": 2,
        "xlstm_memory_state_decay_interval": 2,
        "xlstm_memory_state_decay_factor": 0.5,
        "xlstm_memory_boundary_token_ids": (31,),
        "moe_router_input_mode": "memory_augmented_input",
    }
    payload.update(overrides)
    return ModelConfig.from_preset("lpt_v2_dev_tiny", **payload)


class TestLPTV2XLSTMMemory(unittest.TestCase):
    def test_xlstm_prefill_decode_decay_and_reset_triggers(self):
        model = LPTV2(32, build_memory_config())

        _, states = model.prefill(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), request_id="mem")
        memory_state = states[0].xlstm_memory
        self.assertEqual(memory_state.token_count, 4)
        self.assertEqual(memory_state.decay_count, 2)
        self.assertEqual(memory_state.last_decay_token_count, 4)
        self.assertAlmostEqual(memory_state.effective_beta, 1e-4, places=7)
        self.assertGreater(memory_state.adapter_delta_norm, 0.0)
        self.assertFalse(model.config.xlstm_memory_as_expert)
        self.assertFalse(model.config.xlstm_memory_as_router_target)

        _, decoded = model.decode(
            torch.tensor([[5]], dtype=torch.long),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            layer_states=states,
            request_id="mem",
        )
        self.assertEqual(decoded[0].xlstm_memory.token_count, 5)
        self.assertEqual(model.xlstm_memory_state_pool.to_runtime_metadata()["requests"]["mem"]["phase"], "decode")

        _, boundary_states = model.decode(
            torch.tensor([[6]], dtype=torch.long),
            attention_mask=torch.ones(1, 6, dtype=torch.long),
            layer_states=decoded,
            memory_boundary_metadata={"boundary_type": "document"},
            request_id="mem",
        )
        self.assertEqual(boundary_states[0].xlstm_memory.reset_count, 1)
        self.assertEqual(boundary_states[0].xlstm_memory.last_reset_reason, "boundary:document")

        _, special_states = model.decode(
            torch.tensor([[31]], dtype=torch.long),
            attention_mask=torch.ones(1, 7, dtype=torch.long),
            layer_states=boundary_states,
            request_id="mem",
        )
        self.assertEqual(special_states[0].xlstm_memory.reset_count, 2)
        self.assertEqual(special_states[0].xlstm_memory.last_reset_reason, "special_token")

        _, session_states = model.decode(
            torch.tensor([[7]], dtype=torch.long),
            attention_mask=torch.ones(1, 8, dtype=torch.long),
            layer_states=special_states,
            session_event="session_reset",
            request_id="mem",
        )
        self.assertEqual(session_states[0].xlstm_memory.reset_count, 3)
        self.assertEqual(session_states[0].xlstm_memory.last_reset_reason, "session_event:session_reset")

        released = model.release_xlstm_memory_state("mem")
        self.assertEqual(len(released), 2)
        self.assertIsNone(model.xlstm_memory_state_pool.get("mem", 0))

    def test_ffn_norm_only_eval_keeps_state_but_bypasses_memory_input(self):
        torch.manual_seed(20260503)
        memory_model = LPTV2(32, build_memory_config())
        eval_model = LPTV2(32, build_memory_config(moe_router_input_mode="ffn_norm_only_eval"))
        eval_model.load_state_dict(memory_model.state_dict(), strict=True)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        memory_logits, memory_states = memory_model.prefill(input_ids, request_id="mem-on")
        eval_logits, eval_states = eval_model.prefill(input_ids, request_id="mem-eval")

        self.assertIsNotNone(eval_states[0].xlstm_memory)
        self.assertEqual(eval_states[0].xlstm_memory.token_count, 3)
        self.assertGreater(eval_states[0].xlstm_memory.adapter_delta_norm, 0.0)
        self.assertGreater(
            (memory_logits.float() - eval_logits.float()).pow(2).mean().sqrt().item(),
            0.0,
        )

    def test_xlstm_layer_granularity_controls_state_updates(self):
        config = build_memory_config(
            num_layers=4,
            layer_block_types=("attention", "attention", "attention", "attention"),
            xlstm_memory_layers="selected_layers",
            xlstm_memory_selected_layers=(1, 3),
        )
        model = LPTV2(32, config)

        _, states = model.prefill(torch.tensor([[1, 2, 3]], dtype=torch.long), request_id="granularity")

        self.assertIsNone(states[0].xlstm_memory)
        self.assertIsNotNone(states[1].xlstm_memory)
        self.assertIsNone(states[2].xlstm_memory)
        self.assertIsNotNone(states[3].xlstm_memory)
        self.assertEqual(model.xlstm_memory_state_pool.to_runtime_metadata()["state_count"], 2)

    def test_memory_assist_report_observes_state_tracking(self):
        report = run_lpt_v2_memory_assist_report(
            vocabulary_size=32,
            sequence_length=4,
            device="cpu",
            dtype="fp32",
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            layer_block_types=("attention", "attention"),
            attention_window_size=4,
            page_block_size=2,
            retnet_state_dim=4,
            retnet_adapter_rank=2,
            original_max_len=4,
            longrope2_target_length=8,
        )

        metrics = report.to_dict()["metrics"]
        self.assertEqual(metrics["prefill_token_count"], 4)
        self.assertEqual(metrics["decode_token_count"], 5)
        self.assertEqual(metrics["boundary_reset_count"], 1)
        self.assertEqual(metrics["special_token_reset_count"], 2)
        self.assertTrue(metrics["special_token_reset_configured"])
        self.assertTrue(metrics["special_token_reset_ready"])
        self.assertEqual(metrics["session_reset_count"], 3)
        self.assertEqual(metrics["decision"]["status"], "admit_instrumentation_only")

    def test_memory_assist_report_treats_unconfigured_special_reset_as_not_required(self):
        report = run_lpt_v2_memory_assist_report(
            vocabulary_size=32,
            sequence_length=4,
            device="cpu",
            dtype="fp32",
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            layer_block_types=("attention", "attention"),
            attention_window_size=4,
            page_block_size=2,
            retnet_state_dim=4,
            retnet_adapter_rank=2,
            original_max_len=4,
            longrope2_target_length=8,
            xlstm_memory_boundary_token_ids=(),
        )

        metrics = report.to_dict()["metrics"]
        self.assertEqual(metrics["boundary_reset_count"], 1)
        self.assertEqual(metrics["special_token_reset_count"], 1)
        self.assertFalse(metrics["special_token_reset_configured"])
        self.assertTrue(metrics["special_token_reset_ready"])
        self.assertEqual(metrics["session_reset_count"], 2)
        self.assertEqual(metrics["decision"]["status"], "admit_instrumentation_only")


if __name__ == "__main__":
    unittest.main()
