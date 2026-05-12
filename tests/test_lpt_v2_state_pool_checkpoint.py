import sys
import unittest
import uuid
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import ModelConfig
from lpt_model import (
    LPTV2,
    LPT_V2_CHECKPOINT_SCHEMA_VERSION,
    build_lpt_v2_checkpoint_payload,
    load_lpt_v2_checkpoint,
    save_lpt_v2_checkpoint,
    validate_lpt_v2_checkpoint_payload,
)


def build_tiny_config():
    return ModelConfig.from_preset(
        "lpt_v2_dev_tiny",
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        layer_block_types=("attention", "attention"),
        attention_window_size=4,
        page_block_size=2,
        retnet_assist_layers="all_layers",
        retnet_state_sharing="per_layer",
        retnet_state_dim=4,
        retnet_adapter_rank=2,
        moe_num_experts=2,
        moe_top_k=1,
        original_max_len=4,
        longrope2_target_length=8,
    )


def build_workspace_tmp_dir(name):
    path = PROJECT_ROOT / ".tmp_unittest" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TestLPTV2StatePoolCheckpoint(unittest.TestCase):
    def test_retnet_state_pool_keeps_mixed_requests_isolated(self):
        model = LPTV2(32, build_tiny_config())

        _, states_a = model.prefill(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), request_id="req-a")
        _, states_b = model.prefill(torch.tensor([[5, 6]], dtype=torch.long), request_id="req-b")
        model.retnet_state_pool.mark_preempted("req-a")

        _, decoded_b = model.decode(
            torch.tensor([[7]], dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            layer_states=states_b,
            request_id="req-b",
        )

        metadata = model.retnet_state_pool.to_runtime_metadata()["requests"]
        self.assertEqual(metadata["req-a"]["phase"], "preempted")
        self.assertEqual(metadata["req-a"]["token_count"], 4)
        self.assertEqual(metadata["req-b"]["phase"], "decode")
        self.assertEqual(metadata["req-b"]["token_count"], 3)
        self.assertEqual(decoded_b[0].retnet_assist.token_count, 3)

        released_b = model.release_retnet_assist_state("req-b")
        self.assertEqual(len(released_b), 2)
        self.assertEqual(model.retnet_state_pool.get("req-b", 0), None)
        self.assertIsNotNone(model.retnet_state_pool.get("req-a", 0))
        self.assertEqual(states_a[0].retnet_assist.token_count, 4)

    def test_lpt_v2_checkpoint_schema_round_trip_and_strict_reject(self):
        model = LPTV2(32, build_tiny_config())
        model.prefill(torch.tensor([[1, 2, 3]], dtype=torch.long), request_id="ckpt")
        payload = build_lpt_v2_checkpoint_payload(model)

        config = validate_lpt_v2_checkpoint_payload(payload)
        self.assertEqual(config.architecture_version, "lpt_v2")
        self.assertEqual(payload["checkpoint_schema_version"], LPT_V2_CHECKPOINT_SCHEMA_VERSION)
        self.assertIn("paged_kv", payload["runtime_metadata"]["cache_backend"])
        self.assertEqual(payload["runtime_metadata"]["state_schema"]["layer_state_schema"], "LayerStateV2")

        bad_architecture = dict(payload)
        bad_architecture["architecture_version"] = "lpt_v1"
        with self.assertRaises(ValueError):
            validate_lpt_v2_checkpoint_payload(bad_architecture)

        bad_schema = dict(payload)
        bad_schema["checkpoint_schema_version"] = LPT_V2_CHECKPOINT_SCHEMA_VERSION - 1
        with self.assertRaises(ValueError):
            validate_lpt_v2_checkpoint_payload(bad_schema)

        checkpoint_path = build_workspace_tmp_dir("state_pool_checkpoint") / "model.pt"
        save_lpt_v2_checkpoint(model, checkpoint_path)
        loaded = load_lpt_v2_checkpoint(checkpoint_path, strict=True)

        self.assertEqual(loaded.model.config, model.config)
        self.assertEqual(tuple(loaded.model.token_embedding.weight.shape), (32, model.config.hidden_size))
        self.assertFalse(loaded.missing_keys)
        self.assertFalse(loaded.unexpected_keys)


if __name__ == "__main__":
    unittest.main()
