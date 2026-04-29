import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import GlobalConfig, MODEL_CONFIG_SCHEMA_VERSION
from lpt_model import ATTENTION_BLOCK_TYPE, LPT, ModelConfig, get_model_architecture_metadata, list_architecture_mismatches
from lpt_training import CURRENT_CHECKPOINT_SCHEMA_VERSION, load_checkpoint
from lpt_training.train import _save_full_checkpoint, _save_model_config_snapshot


class DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    eos_token = "<eos>"
    pad_token = "<pad>"


def build_tiny_attention_config():
    return ModelConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cla_share_every_n_layers=1,
        layer_block_types=(ATTENTION_BLOCK_TYPE, ATTENTION_BLOCK_TYPE),
    )


class TestCheckpointSchema(unittest.TestCase):
    def setUp(self):
        self.original_device = GlobalConfig.device
        self.original_training_stage = GlobalConfig.training_stage
        GlobalConfig.device = torch.device("cpu")
        GlobalConfig.training_stage = "text_pretrain"

    def tearDown(self):
        GlobalConfig.device = self.original_device
        GlobalConfig.training_stage = self.original_training_stage

    def _build_model(self):
        model = LPT(vocabulary_size=32, config=build_tiny_attention_config())
        model.eval()
        return model

    def test_save_full_checkpoint_includes_schema_version_snapshot_and_manifest(self):
        model = self._build_model()
        tokenizer = DummyTokenizer()
        expected_metadata = get_model_architecture_metadata(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "latest"
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            checkpoint = _save_full_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_root=checkpoint_root,
                manifest_path=Path("data/manifests/text_pretrain.json"),
                eval_manifest_path=None,
                loss_value=1.25,
                eval_loss=1.1,
                eval_ppl=3.0,
                epoch_index=1,
                total_epochs=3,
                batch_size=2,
                learning_rate=3e-4,
                warmup_ratio=0.1,
                weight_decay=0.1,
                gradient_accumulation_steps=2,
                max_grad_norm=1.0,
                random_seed=7,
                deterministic_algorithms=True,
                log_interval_steps=5,
                eval_interval_steps=10,
                optimizer=optimizer,
                global_step=4,
                optimizer_step=4,
                tokens_seen=128,
                samples_seen=8,
                run_id="unit-test-run",
                optimizer_group_summary={
                    "weight_decay": 0.1,
                    "decay_parameter_count": 2,
                    "no_decay_parameter_count": 3,
                    "decay_parameter_names": ("layers.0.sequence_mixer.q_proj.weight",),
                    "no_decay_parameter_names": ("token_embedding.weight",),
                },
                longrope2_training_strategy={
                    "original_window": 2048,
                    "target_window": 32768,
                    "train_embedding_mode": "dynamic",
                    "inference_embedding_mode": "static",
                },
            )

            self.assertEqual(checkpoint["checkpoint_schema_version"], CURRENT_CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(checkpoint["model_config_schema_version"], MODEL_CONFIG_SCHEMA_VERSION)
            self.assertEqual(checkpoint["model_config"], model.config.to_dict())
            self.assertEqual(checkpoint["model_architecture_metadata"], expected_metadata)
            self.assertNotIn("num_layers", checkpoint)
            self.assertNotIn("layer_block_types", checkpoint)
            self.assertEqual(checkpoint["source_manifest"], "data\\manifests\\text_pretrain.json")
            self.assertIsNone(checkpoint["eval_manifest"])
            self.assertEqual(checkpoint["training_stage"], "text_pretrain")
            self.assertEqual(checkpoint["gradient_accumulation_steps"], 2)
            self.assertEqual(checkpoint["global_step"], 4)
            self.assertEqual(checkpoint["run_id"], "unit-test-run")
            self.assertEqual(checkpoint["latest_eval_loss"], 1.1)
            self.assertEqual(checkpoint["longrope2_training_strategy"]["train_embedding_mode"], "dynamic")

            loaded_checkpoint = load_checkpoint(checkpoint_root)
            self.assertEqual(loaded_checkpoint["checkpoint_schema_version"], CURRENT_CHECKPOINT_SCHEMA_VERSION)
            self.assertEqual(loaded_checkpoint["model_config_schema_version"], MODEL_CONFIG_SCHEMA_VERSION)
            self.assertEqual(loaded_checkpoint["model_config"], model.config.to_dict())
            self.assertEqual(loaded_checkpoint["model_architecture_metadata"], expected_metadata)
            self.assertEqual(loaded_checkpoint["source_manifest"], "data\\manifests\\text_pretrain.json")
            self.assertEqual(loaded_checkpoint["global_step"], 4)

    def test_save_model_config_snapshot_writes_artifact_json(self):
        model = self._build_model()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "checkpoints" / "latest"
            checkpoint_root.parent.mkdir(parents=True, exist_ok=True)

            snapshot_path = _save_model_config_snapshot(model, checkpoint_root)

            self.assertTrue(snapshot_path.exists())
            snapshot_text = snapshot_path.read_text(encoding="utf-8")
            self.assertIn("\"model_config_schema_version\": 1", snapshot_text)
            self.assertIn("\"num_layers\": 2", snapshot_text)

    def test_load_checkpoint_rejects_missing_required_schema_fields(self):
        model = self._build_model()
        invalid_checkpoint = {
            "model_abbr": "LPT",
            "model_state_dict": model.state_dict(),
            "training_stage": "text_pretrain",
            "source_manifest": None,
            "epoch": 2,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "invalid"
            torch.save(invalid_checkpoint, checkpoint_root.with_suffix(".pth"))

            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint_root)

    def test_list_architecture_mismatches_reads_nested_snapshot(self):
        model = self._build_model()
        checkpoint = {
            "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "model_architecture_metadata": get_model_architecture_metadata(model),
        }

        self.assertEqual(list_architecture_mismatches(checkpoint, model), [])

    def test_load_checkpoint_rejects_unknown_schema_version(self):
        model = self._build_model()
        unsupported_checkpoint = {
            "checkpoint_schema_version": CURRENT_CHECKPOINT_SCHEMA_VERSION + 1,
            "model_state_dict": model.state_dict(),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "unsupported"
            torch.save(unsupported_checkpoint, checkpoint_root.with_suffix(".pth"))

            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint_root)


if __name__ == "__main__":
    unittest.main()
