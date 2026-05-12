import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import (
    DEFAULT_ATTENTION_BACKEND_PRIORITY,
    DEFAULT_MODEL_SIZE_PRESET,
    LPT_V2_BASE_PRESET,
    LPT_V2_ARCHITECTURE_VERSION,
    LPT_V2_BLOCK_TYPE,
    LPT_V2_DEV_TINY_PRESET,
    LPT_V2_LARGE_PRESET,
    LPT_V2_MODEL_SIZE_PRESETS,
    LPT_V2_SEQUENCE_MIXER_MODE,
    LPT_V2_SMALL_PRESET,
    MODEL_CONFIG_SCHEMA_VERSION,
    ModelConfig,
    PARAMETER_COUNT_MODES,
    build_lpt_v2_model_config_preset,
    build_model_config_from_checkpoint,
    count_retnet_assist_enabled_layers,
    expand_lpt_v2_model_config_preset,
    is_retnet_assist_enabled_for_layer,
    load_model_config_json,
    model_config_snapshot_path,
    normalize_model_config,
)


def build_workspace_tmp_dir(name):
    path = PROJECT_ROOT / ".tmp_tests" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        self.assertEqual(config.architecture_version, LPT_V2_ARCHITECTURE_VERSION)
        self.assertEqual(config.block_type, LPT_V2_BLOCK_TYPE)
        self.assertEqual(config.sequence_mixer_mode, LPT_V2_SEQUENCE_MIXER_MODE)
        self.assertEqual(config.model_size_preset, DEFAULT_MODEL_SIZE_PRESET)
        self.assertEqual(config.attention_backend_priority, DEFAULT_ATTENTION_BACKEND_PRIORITY)
        self.assertEqual(config.attention_backend_priority, ("sdpa",))
        self.assertEqual(config.cla_share_every_n_layers, 1)
        default_shape = LPT_V2_MODEL_SIZE_PRESETS[DEFAULT_MODEL_SIZE_PRESET]
        self.assertEqual(config.num_layers, default_shape["num_layers"])
        self.assertEqual(config.hidden_size, default_shape["num_heads"] * default_shape["head_dim"])
        self.assertEqual(config.num_heads, default_shape["num_heads"])
        self.assertEqual(config.num_kv_heads, default_shape["num_kv_heads"])
        self.assertEqual(config.moe_num_experts, default_shape["moe_num_experts"])
        self.assertEqual(config.moe_top_k, default_shape["moe_top_k"])
        self.assertEqual(config.attention_window_size, default_shape["attention_window_size"])
        self.assertEqual(config.parameter_count_modes, PARAMETER_COUNT_MODES)
        self.assertTrue(all(block_type == "attention" for block_type in config.layer_block_types))

    def test_lpt_v2_model_size_presets_expand_to_complete_model_config(self):
        expected_shapes = {
            LPT_V2_DEV_TINY_PRESET: (4, 256, 4, 2, 2, 1, 512),
            LPT_V2_SMALL_PRESET: (12, 768, 12, 4, 4, 2, 2048),
            LPT_V2_BASE_PRESET: (24, 1536, 16, 4, 8, 2, 4096),
            LPT_V2_LARGE_PRESET: (32, 2048, 32, 8, 8, 2, 4096),
        }

        for preset, expected_shape in expected_shapes.items():
            with self.subTest(preset=preset):
                payload = expand_lpt_v2_model_config_preset(preset)
                config = build_lpt_v2_model_config_preset(preset)
                expected_layers, expected_hidden, expected_heads, expected_kv_heads, expected_experts, expected_top_k, expected_window = expected_shape

                self.assertEqual(payload["model_size_preset"], preset)
                self.assertEqual(config.num_layers, expected_layers)
                self.assertEqual(config.hidden_size, expected_hidden)
                self.assertEqual(config.num_heads, expected_heads)
                self.assertEqual(config.num_kv_heads, expected_kv_heads)
                self.assertEqual(config.moe_num_experts, expected_experts)
                self.assertEqual(config.moe_top_k, expected_top_k)
                self.assertEqual(config.attention_window_size, expected_window)
                self.assertEqual(len(config.layer_block_types), expected_layers)
                self.assertTrue(all(block_type == "attention" for block_type in config.layer_block_types))

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
            layer_block_types=("attention", "attention", "attention", "attention"),
            longrope2_long_factors=(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0),
            longrope2_factor_max_sequence_length=4096,
            longrope2_train_embedding_mode="mixed",
            longrope2_inference_embedding_mode="static",
            longrope2_mixed_original_window=1024,
        )

        snapshot_path = build_workspace_tmp_dir("model_config_json_round_trip") / "config" / "model_config.json"
        config.save_json(snapshot_path)
        loaded_config = load_model_config_json(snapshot_path)

        self.assertEqual(loaded_config, config)

    def test_model_config_v2_rejects_cla_sharing(self):
        with self.assertRaises(ValueError):
            ModelConfig(cla_share_every_n_layers=2)

    def test_model_config_v2_rejects_retnet_main_block(self):
        with self.assertRaises(ValueError):
            ModelConfig(
                num_layers=2,
                layer_block_types=("attention", "retnet"),
            )

    def test_model_config_rejects_incompatible_schema_version(self):
        payload = {
            "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION - 1,
            "model_config": ModelConfig().to_dict(),
        }

        with self.assertRaises(ValueError):
            ModelConfig.from_json_payload(payload)

    def test_model_config_validates_xlstm_router_input_contract(self):
        config = ModelConfig(
            xlstm_memory_enabled=True,
            xlstm_memory_layers="every_n_layers",
            moe_router_input_mode="ffn_norm_only_eval",
        )
        self.assertEqual(config.moe_router_input_mode, "ffn_norm_only_eval")
        with self.assertRaises(ValueError):
            ModelConfig(
                xlstm_memory_enabled=False,
                xlstm_memory_layers="disabled",
                moe_router_input_mode="memory_augmented_input",
            )

    def test_model_config_validates_xlstm_selected_layers(self):
        config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            xlstm_memory_enabled=True,
            xlstm_memory_layers="selected_layers",
            xlstm_memory_selected_layers=[3, 1],
            moe_router_input_mode="memory_augmented_input",
        )

        self.assertEqual(config.xlstm_memory_selected_layers, (1, 3))
        self.assertEqual(
            ModelConfig.from_dict(config.to_dict()).xlstm_memory_selected_layers,
            (1, 3),
        )

        invalid_payloads = (
            {"xlstm_memory_selected_layers": ()},
            {"xlstm_memory_selected_layers": (1, 1)},
            {"xlstm_memory_selected_layers": (-1,)},
            {"xlstm_memory_selected_layers": (4,)},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    ModelConfig.from_preset(
                        LPT_V2_DEV_TINY_PRESET,
                        xlstm_memory_enabled=True,
                        xlstm_memory_layers="selected_layers",
                        moe_router_input_mode="memory_augmented_input",
                        **payload,
                    )

        with self.assertRaises(ValueError):
            ModelConfig.from_preset(
                LPT_V2_DEV_TINY_PRESET,
                xlstm_memory_enabled=True,
                xlstm_memory_layers="every_0_layers",
                moe_router_input_mode="memory_augmented_input",
            )

    def test_xlstm_memory_gate_requires_enabled_memory(self):
        with self.assertRaises(ValueError):
            ModelConfig.from_preset(
                LPT_V2_DEV_TINY_PRESET,
                xlstm_memory_gate_enabled=True,
            )

        config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            xlstm_memory_enabled=True,
            xlstm_memory_layers="all_layers",
            moe_router_input_mode="memory_augmented_input",
            xlstm_memory_gate_enabled=True,
        )

        self.assertTrue(config.xlstm_memory_gate_enabled)

    def test_retnet_qk_adapter_requires_consistent_mode_and_target(self):
        config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            retnet_assist_mode="qk_adapter",
            retnet_adapter_target=("k", "q"),
            retnet_k_adapter_enabled=True,
        )

        self.assertEqual(config.retnet_adapter_target, ("q", "k"))
        self.assertTrue(config.retnet_k_adapter_enabled)

        invalid_payloads = (
            {"retnet_assist_mode": "q_adapter", "retnet_adapter_target": ("q", "k"), "retnet_k_adapter_enabled": True},
            {"retnet_assist_mode": "qk_adapter", "retnet_adapter_target": ("q",), "retnet_k_adapter_enabled": False},
            {"retnet_assist_mode": "qk_adapter", "retnet_adapter_target": ("k",), "retnet_k_adapter_enabled": True},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    ModelConfig.from_preset(LPT_V2_DEV_TINY_PRESET, **payload)

    def test_retnet_sharing_config_accepts_per_layer_and_rejects_bad_group_size(self):
        config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            retnet_parameter_sharing="per_layer",
            retnet_state_sharing="per_layer",
            retnet_sharing_group_size=2,
        )

        self.assertEqual(config.retnet_parameter_sharing, "per_layer")
        self.assertEqual(config.retnet_state_sharing, "per_layer")
        self.assertEqual(config.retnet_sharing_group_size, 2)

        with self.assertRaises(ValueError):
            ModelConfig.from_preset(LPT_V2_DEV_TINY_PRESET, retnet_sharing_group_size=0)

    def test_retnet_selected_layers_are_validated_and_counted(self):
        config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            retnet_assist_layers="selected_layers",
            retnet_assist_selected_layers=(3, 1),
        )

        self.assertEqual(config.retnet_assist_selected_layers, (1, 3))
        self.assertEqual(count_retnet_assist_enabled_layers(config), 2)
        self.assertFalse(is_retnet_assist_enabled_for_layer(config, 0))
        self.assertTrue(is_retnet_assist_enabled_for_layer(config, 1))

        with self.assertRaises(ValueError):
            ModelConfig.from_preset(
                LPT_V2_DEV_TINY_PRESET,
                retnet_assist_layers="selected_layers",
            )
        with self.assertRaises(ValueError):
            ModelConfig.from_preset(
                LPT_V2_DEV_TINY_PRESET,
                retnet_assist_layers="all_layers",
                retnet_assist_selected_layers=(1,),
            )

    def test_model_config_rejects_invalid_longrope2_embedding_mode(self):
        with self.assertRaises(ValueError):
            ModelConfig(longrope2_train_embedding_mode="unknown")

    def test_normalize_model_config_rejects_class_objects(self):
        class ClassStyleModelConfig(ModelConfig):
            num_layers = 2

        with self.assertRaises(TypeError):
            normalize_model_config(ClassStyleModelConfig)

    def test_build_model_config_from_checkpoint_requires_nested_snapshot(self):
        config = build_tiny_attention_config()
        checkpoint = {
            "model_config_schema_version": MODEL_CONFIG_SCHEMA_VERSION,
            "model_config": config.to_dict(),
        }

        loaded_config = build_model_config_from_checkpoint(checkpoint)

        self.assertEqual(loaded_config, config)

    def test_load_model_config_json_rejects_unwrapped_payload(self):
        snapshot_path = build_workspace_tmp_dir("model_config_unwrapped_payload") / "config" / "model_config.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text("{\"num_layers\": 2}", encoding="utf-8")

        with self.assertRaises(ValueError):
            load_model_config_json(snapshot_path)

    def test_model_config_snapshot_path_follows_artifact_convention(self):
        snapshot_path = model_config_snapshot_path("artifacts/lpt_v2/base")
        self.assertTrue(str(snapshot_path).endswith("artifacts\\lpt_v2\\base\\config\\model_config.json"))


if __name__ == "__main__":
    unittest.main()
