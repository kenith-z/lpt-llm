import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import (
    LPT_V2_BASE_PRESET,
    LPT_V2_DEV_TINY_PRESET,
    ModelConfig,
    build_lpt_v2_model_config_preset,
)
from lpt_model import estimate_moe_aware_parameter_counts


class TestMoEAwareParameterCount(unittest.TestCase):
    def test_tiny_report_counts_all_experts_but_only_top_k_active(self):
        config = ModelConfig.from_preset(LPT_V2_DEV_TINY_PRESET)

        report = estimate_moe_aware_parameter_counts(config)

        self.assertEqual(report.model_size_preset, "lpt_v2_dev_tiny")
        self.assertGreater(report.total_physical_params, report.active_params_per_token)
        self.assertEqual(
            report.module_breakdown["swiglu_experts"],
            report.active_module_breakdown["swiglu_experts"] * config.moe_num_experts // config.moe_top_k,
        )
        self.assertEqual(
            report.total_physical_params,
            report.shared_params + report.expert_params + report.router_params + report.adapter_params,
        )
        self.assertIn("attention", report.module_breakdown)
        self.assertIn("retnet_assist_core", report.module_breakdown)
        self.assertIn("moe_router", report.module_breakdown)

    def test_base_report_uses_moe_top_k_for_active_params(self):
        config = build_lpt_v2_model_config_preset(LPT_V2_BASE_PRESET)

        report = estimate_moe_aware_parameter_counts(config)

        self.assertEqual(config.moe_num_experts, 8)
        self.assertEqual(config.moe_top_k, 2)
        self.assertEqual(
            report.module_breakdown["swiglu_experts"],
            report.active_module_breakdown["swiglu_experts"] * 4,
        )

    def test_xlstm_memory_params_and_state_bytes_are_reported_when_enabled(self):
        config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            xlstm_memory_enabled=True,
            xlstm_memory_layers="all_layers",
            moe_router_input_mode="memory_augmented_input",
        )

        report = estimate_moe_aware_parameter_counts(config)

        self.assertGreater(report.module_breakdown["xlstm_memory_core"], 0)
        self.assertGreater(report.module_breakdown["xlstm_memory_adapter"], 0)
        self.assertGreater(report.state_runtime_bytes, config.retnet_state_dim * 4)
        self.assertEqual(report.to_dict()["state_runtime_bytes"], report.state_runtime_bytes)

    def test_xlstm_memory_layer_granularity_changes_parameter_count(self):
        all_layers_config = ModelConfig.from_preset(
            LPT_V2_DEV_TINY_PRESET,
            xlstm_memory_enabled=True,
            xlstm_memory_layers="all_layers",
            moe_router_input_mode="memory_augmented_input",
        )
        every_two_config = all_layers_config.with_overrides(xlstm_memory_layers="every_2_layers")
        selected_config = all_layers_config.with_overrides(
            xlstm_memory_layers="selected_layers",
            xlstm_memory_selected_layers=(1, 3),
        )

        all_layers_report = estimate_moe_aware_parameter_counts(all_layers_config)
        every_two_report = estimate_moe_aware_parameter_counts(every_two_config)
        selected_report = estimate_moe_aware_parameter_counts(selected_config)

        self.assertEqual(
            every_two_report.module_breakdown["xlstm_memory_core"],
            all_layers_report.module_breakdown["xlstm_memory_core"] // 2,
        )
        self.assertEqual(
            selected_report.module_breakdown["xlstm_memory_adapter"],
            all_layers_report.module_breakdown["xlstm_memory_adapter"] // 2,
        )
        self.assertLess(every_two_report.state_runtime_bytes, all_layers_report.state_runtime_bytes)

    def test_retnet_qk_adapter_adds_k_adapter_params(self):
        q_config = ModelConfig.from_preset(LPT_V2_DEV_TINY_PRESET)
        qk_config = q_config.with_overrides(
            retnet_assist_mode="qk_adapter",
            retnet_adapter_target=("q", "k"),
            retnet_k_adapter_enabled=True,
        )

        q_report = estimate_moe_aware_parameter_counts(q_config)
        qk_report = estimate_moe_aware_parameter_counts(qk_config)

        self.assertEqual(q_report.module_breakdown["retnet_k_adapter"], 0)
        self.assertGreater(qk_report.module_breakdown["retnet_k_adapter"], 0)
        self.assertGreater(qk_report.adapter_params, q_report.adapter_params)


if __name__ == "__main__":
    unittest.main()
