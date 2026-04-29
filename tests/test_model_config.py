import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import (
    MODEL_CONFIG_SCHEMA_VERSION,
    ModelConfig,
    build_model_config_from_checkpoint,
    load_model_config_json,
    model_config_snapshot_path,
    normalize_model_config,
)


def build_tiny_attention_config():
    return ModelConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cla_share_every_n_layers=1,
        layer_block_types=("attention", "attention"),
    )


class TestModelConfig(unittest.TestCase):
    def test_model_config_defaults_to_mixed_longrope2_embedding(self):
        config = ModelConfig()

        self.assertEqual(config.longrope2_train_embedding_mode, "mixed")
        self.assertEqual(config.longrope2_inference_embedding_mode, "mixed")

    def test_model_config_normalizes_scalar_longrope2_factor_to_array(self):
        config = ModelConfig(
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            hidden_size=16,
            longrope2_long_factors=2.0,
        )

        self.assertEqual(config.longrope2_long_factors, (2.0, 2.0, 2.0, 2.0))

    def test_model_config_json_round_trip(self):
        config = ModelConfig(
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=16,
            cla_share_every_n_layers=1,
            layer_block_types=("attention", "attention", "retnet", "retnet"),
            longrope2_long_factors=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            longrope2_factor_max_sequence_length=4096,
            longrope2_train_embedding_mode="mixed",
            longrope2_inference_embedding_mode="static",
            longrope2_mixed_original_window=1024,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "config" / "model_config.json"
            config.save_json(snapshot_path)
            loaded_config = load_model_config_json(snapshot_path)

        self.assertEqual(loaded_config, config)

    def test_model_config_rejects_invalid_longrope2_embedding_mode(self):
        with self.assertRaises(ValueError):
            ModelConfig(longrope2_train_embedding_mode="unknown")

    def test_normalize_model_config_rejects_legacy_class_style(self):
        class LegacyModelConfig(ModelConfig):
            num_layers = 2

        with self.assertRaises(TypeError):
            normalize_model_config(LegacyModelConfig)

    def test_build_model_config_from_checkpoint_requires_nested_snapshot(self):
        config = build_tiny_attention_config()
        checkpoint = {
            "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
            "model_config": config.to_dict(),
        }

        loaded_config = build_model_config_from_checkpoint(checkpoint)

        self.assertEqual(loaded_config, config)

    def test_load_model_config_json_rejects_unwrapped_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = Path(temp_dir) / "config" / "model_config.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text("{\"num_layers\": 2}", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_model_config_json(snapshot_path)

    def test_model_config_snapshot_path_follows_artifact_convention(self):
        snapshot_path = model_config_snapshot_path("artifacts/lpt_ds_v1/chat_sft")
        self.assertTrue(str(snapshot_path).endswith("artifacts\\lpt_ds_v1\\chat_sft\\config\\model_config.json"))


if __name__ == "__main__":
    unittest.main()
