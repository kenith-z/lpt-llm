import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import DENSE_KV_CACHE_BACKEND, ModelConfig
from lpt_model import LPTV2, LayerStateV2, QOnlyRetNetAdapter, RetNetContextAdapter
from lpt_model.model_v2 import SwiGLUMoE


def build_tiny_v2_config(**overrides):
    payload = {
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "layer_block_types": ("attention", "attention"),
        "attention_window_size": 3,
        "page_block_size": 2,
        "retnet_assist_layers": "all_layers",
        "retnet_state_dim": 4,
        "retnet_adapter_rank": 2,
        "moe_num_experts": 2,
        "moe_top_k": 1,
        "original_max_len": 4,
        "longrope2_target_length": 8,
    }
    payload.update(overrides)
    return ModelConfig.from_preset("lpt_v2_dev_tiny", **payload)


class TestLPTV2Model(unittest.TestCase):
    def test_lpt_v2_prefill_updates_layer_states(self):
        model = LPTV2(32, build_tiny_v2_config())

        logits, states = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

        self.assertEqual(tuple(logits.shape), (1, 4, 32))
        self.assertEqual(len(states), 2)
        self.assertTrue(all(isinstance(state, LayerStateV2) for state in states))
        self.assertEqual(states[0].attention.paged_kv_ref.window_token_count, 3)
        self.assertEqual(states[0].retnet_assist.token_count, 4)
        self.assertEqual(sum(states[0].moe.expert_token_counts), 4)

    def test_lpt_v2_decode_uses_full_attention_mask_tail(self):
        model = LPTV2(32, build_tiny_v2_config())
        _, prefill_states = model(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            request_id="decode-req",
        )

        decode_logits, decode_states = model(
            torch.tensor([[5]], dtype=torch.long),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            layer_states=prefill_states,
            request_id="decode-req",
        )

        self.assertEqual(tuple(decode_logits.shape), (1, 1, 32))
        self.assertEqual(decode_states[0].attention.paged_kv_ref.token_count, 5)
        self.assertEqual(decode_states[0].attention.paged_kv_ref.window_token_count, 3)
        self.assertEqual(decode_states[0].retnet_assist.token_count, 5)
        self.assertEqual(model.paged_kv_cache.allocated_page_count, 4)

    def test_paged_kv_window_trim_does_not_reset_retnet_assist(self):
        model = LPTV2(32, build_tiny_v2_config())

        _, states = model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))

        self.assertEqual(states[0].attention.paged_kv_ref.window_token_count, 3)
        self.assertEqual(states[0].attention.paged_kv_ref.token_count, 5)
        self.assertEqual(states[0].retnet_assist.token_count, 5)
        self.assertIsNotNone(states[0].retnet_assist.summary)

    def test_reset_request_state_releases_only_paged_kv_pages(self):
        model = LPTV2(32, build_tiny_v2_config())
        _, states = model(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            request_id="req-a",
        )
        retnet_state = states[0].retnet_assist

        self.assertGreater(model.paged_kv_cache.allocated_page_count, 0)
        model.reset_request_state("req-a")

        self.assertEqual(model.paged_kv_cache.allocated_page_count, 0)
        self.assertEqual(retnet_state.token_count, 4)
        self.assertFalse(retnet_state.release_metadata.released)

    def test_dense_kv_cache_backend_keeps_windowed_state(self):
        model = LPTV2(
            32,
            build_tiny_v2_config(cache_backend=DENSE_KV_CACHE_BACKEND),
        )

        logits, states = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

        self.assertEqual(tuple(logits.shape), (1, 4, 32))
        dense_k, dense_v = states[0].attention.dense_kv_state
        self.assertEqual(dense_k.size(2), 3)
        self.assertEqual(dense_v.size(2), 3)
        self.assertIsNone(states[0].attention.paged_kv_ref)

    def test_training_forward_can_disable_kv_cache(self):
        model = LPTV2(32, build_tiny_v2_config())
        model.train()

        logits, states = model(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            rope_cache_scope="train",
            request_id="train",
            use_kv_cache=False,
        )
        logits.float().sum().backward()

        self.assertEqual(tuple(logits.shape), (1, 4, 32))
        self.assertEqual(model.paged_kv_cache.allocated_page_count, 0)
        self.assertTrue(all(state.attention is None for state in states))
        self.assertTrue(all(state.retnet_assist is not None for state in states))
        self.assertIsNotNone(model.layers[0].attention_mixer.k_proj.weight.grad)
        self.assertIsNotNone(model.layers[0].attention_mixer.v_proj.weight.grad)

    def test_retnet_assist_is_shared_and_adapter_is_q_only_fp32_scale(self):
        model = LPTV2(32, build_tiny_v2_config())

        self.assertIs(
            model.layers[0].attention_mixer.retnet_assist,
            model.layers[1].attention_mixer.retnet_assist,
        )
        self.assertIs(
            model.layers[0].attention_mixer.q_adapter,
            model.layers[1].attention_mixer.q_adapter,
        )
        q_adapter = model.layers[0].attention_mixer.q_adapter
        self.assertEqual(q_adapter.alpha_q.dtype, torch.float32)
        self.assertFalse(model.config.retnet_k_adapter_enabled)
        self.assertFalse(model.config.retnet_enters_paged_kv)

        standalone_adapter = QOnlyRetNetAdapter(model.config).to(dtype=torch.float16)
        self.assertEqual(standalone_adapter.alpha_q.dtype, torch.float32)

    def test_retnet_parameter_sharing_controls_module_identity(self):
        group_config = build_tiny_v2_config(
            num_layers=4,
            layer_block_types=("attention", "attention", "attention", "attention"),
            retnet_parameter_sharing="group",
            retnet_sharing_group_size=2,
        )
        group_model = LPTV2(32, group_config)

        self.assertIs(
            group_model.layers[0].attention_mixer.retnet_assist,
            group_model.layers[1].attention_mixer.retnet_assist,
        )
        self.assertIsNot(
            group_model.layers[0].attention_mixer.retnet_assist,
            group_model.layers[2].attention_mixer.retnet_assist,
        )
        self.assertIs(
            group_model.layers[0].attention_mixer.q_adapter,
            group_model.layers[1].attention_mixer.q_adapter,
        )
        self.assertIsNot(
            group_model.layers[0].attention_mixer.q_adapter,
            group_model.layers[2].attention_mixer.q_adapter,
        )

        per_layer_model = LPTV2(
            32,
            group_config.with_overrides(retnet_parameter_sharing="per_layer"),
        )
        self.assertIsNot(
            per_layer_model.layers[0].attention_mixer.retnet_assist,
            per_layer_model.layers[1].attention_mixer.retnet_assist,
        )
        self.assertIsNot(
            per_layer_model.layers[0].attention_mixer.q_adapter,
            per_layer_model.layers[1].attention_mixer.q_adapter,
        )

    def test_retnet_state_sharing_uses_group_slots(self):
        config = build_tiny_v2_config(
            num_layers=4,
            layer_block_types=("attention", "attention", "attention", "attention"),
            retnet_state_sharing="group",
            retnet_sharing_group_size=2,
        )
        model = LPTV2(32, config)

        _, states = model.prefill(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), request_id="group-state")

        self.assertEqual([state.retnet_assist.state_slot for state in states], [0, 0, 1, 1])
        self.assertEqual([state.retnet_assist.token_count for state in states], [4, 4, 4, 4])
        metadata = model.retnet_state_pool.to_runtime_metadata()
        self.assertEqual(metadata["state_slot_count"], 2)
        self.assertEqual(metadata["requests"]["group-state"]["state_count"], 2)

    def test_retnet_sparse_layers_do_not_keep_unused_adapter_state(self):
        config = build_tiny_v2_config(
            num_layers=4,
            layer_block_types=("attention", "attention", "attention", "attention"),
            retnet_assist_layers="every_2_layers",
            retnet_state_sharing="per_layer",
        )
        model = LPTV2(32, config)

        self.assertIsNone(model.layers[1].attention_mixer.retnet_assist)
        self.assertIsNone(model.layers[1].attention_mixer.q_adapter)

        _, states = model.prefill(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), request_id="sparse-retnet")

        self.assertEqual(
            [state.retnet_assist is not None for state in states],
            [True, False, True, False],
        )
        metadata = model.retnet_state_pool.to_runtime_metadata()
        self.assertEqual(metadata["requests"]["sparse-retnet"]["state_count"], 2)

    def test_retnet_selected_layers_only_update_selected_state(self):
        config = build_tiny_v2_config(
            num_layers=4,
            layer_block_types=("attention", "attention", "attention", "attention"),
            retnet_assist_layers="selected_layers",
            retnet_assist_selected_layers=(1, 3),
        )
        model = LPTV2(32, config)

        _, states = model.prefill(torch.tensor([[1, 2, 3, 4]], dtype=torch.long), request_id="selected-retnet")

        self.assertEqual(
            [state.retnet_assist is not None for state in states],
            [False, True, False, True],
        )
        self.assertEqual(
            [None if state.retnet_assist is None else state.retnet_assist.state_slot for state in states],
            [None, 0, None, 0],
        )

    def test_retnet_qk_adapter_updates_key_without_touching_value(self):
        config = build_tiny_v2_config(
            retnet_assist_mode="qk_adapter",
            retnet_adapter_target=("q", "k"),
            retnet_k_adapter_enabled=True,
        )
        model = LPTV2(32, config)
        model.train()

        logits, _ = model(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            rope_cache_scope="train",
            request_id="qk-train",
            use_kv_cache=False,
        )
        logits.float().sum().backward()

        adapter = model.layers[0].attention_mixer.q_adapter
        self.assertEqual(adapter.alpha_q.dtype, torch.float32)
        self.assertEqual(adapter.alpha_k.dtype, torch.float32)
        self.assertIsNotNone(adapter.k_down_projection.weight.grad)
        self.assertIsNotNone(adapter.k_up_projection.weight.grad)
        self.assertFalse(hasattr(adapter, "v_down_projection"))

        model.eval()
        with torch.no_grad():
            _, states = model(
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                rope_cache_scope="inference",
                request_id="qk-eval",
                use_kv_cache=False,
            )
        self.assertGreater(states[0].retnet_assist.q_adapter_delta_norm, 0.0)
        self.assertGreater(states[0].retnet_assist.k_adapter_delta_norm, 0.0)

    def test_retnet_context_adapter_reuses_summary_without_extra_state(self):
        config = build_tiny_v2_config(
            retnet_context_adapter_enabled=True,
            retnet_context_adapter_alpha=1e-4,
        )
        model = LPTV2(32, config)

        self.assertIsInstance(model.layers[0].attention_mixer.context_adapter, RetNetContextAdapter)
        self.assertIs(
            model.layers[0].attention_mixer.context_adapter,
            model.layers[1].attention_mixer.context_adapter,
        )
        context_adapter = model.layers[0].attention_mixer.context_adapter
        self.assertEqual(context_adapter.alpha_context.dtype, torch.float32)

        model.train()
        logits, states = model(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            rope_cache_scope="train",
            request_id="context-train",
            use_kv_cache=False,
        )
        logits.float().sum().backward()

        self.assertIsNotNone(context_adapter.down_projection.weight.grad)
        self.assertIsNotNone(context_adapter.up_projection.weight.grad)
        self.assertIsNotNone(context_adapter.alpha_context.grad)
        self.assertEqual(states[0].retnet_assist.token_count, 4)

        model.eval()
        with torch.no_grad():
            _, eval_states = model(
                torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                rope_cache_scope="inference",
                request_id="context-eval",
                use_kv_cache=False,
            )

        self.assertGreater(eval_states[0].retnet_assist.context_adapter_delta_norm, 0.0)
        self.assertAlmostEqual(eval_states[0].retnet_assist.alpha_context, 1e-4)

    def test_swiglu_moe_records_router_statistics(self):
        model = LPTV2(32, build_tiny_v2_config(moe_num_experts=4, moe_top_k=2))

        _, states = model(torch.tensor([[1, 2, 3]], dtype=torch.long))

        moe_state = states[0].moe
        self.assertEqual(len(moe_state.expert_token_counts), 4)
        self.assertEqual(sum(moe_state.expert_token_counts), 6)
        self.assertIsNotNone(moe_state.router_entropy)
        self.assertIsNotNone(moe_state.load_balance_loss)
        self.assertTrue(
            all(
                expert.__class__.__name__ == "SwiGLU"
                for expert in model.layers[0].feed_forward.experts
            )
        )

    def test_swiglu_moe_only_runs_routed_experts(self):
        class CountingExpert(torch.nn.Module):
            def __init__(self, hidden_size, value):
                super().__init__()
                self.hidden_size = int(hidden_size)
                self.value = float(value)
                self.calls = 0
                self.tokens = 0

            def forward(self, x):
                self.calls += 1
                self.tokens += int(x.size(0))
                return torch.full_like(x, self.value)

        moe = SwiGLUMoE(build_tiny_v2_config(moe_num_experts=4, moe_top_k=1), layer_index=0)
        moe.experts = torch.nn.ModuleList(
            [CountingExpert(moe.hidden_size, value=index + 1) for index in range(4)]
        )
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[0, 0] = 1.0
            moe.router.weight[1:, 0] = -1.0

        x_ffn = torch.ones(2, 3, moe.hidden_size)
        output, state = moe(x_ffn)

        self.assertTrue(torch.allclose(output, torch.ones_like(output)))
        self.assertEqual(moe.experts[0].calls, 1)
        self.assertEqual(moe.experts[0].tokens, 6)
        self.assertTrue(all(expert.calls == 0 for expert in moe.experts[1:]))
        self.assertEqual(state.expert_token_counts, (6, 0, 0, 0))

    def test_xlstm_memory_gate_participates_in_training_graph(self):
        model = LPTV2(
            32,
            build_tiny_v2_config(
                xlstm_memory_enabled=True,
                xlstm_memory_layers="all_layers",
                moe_router_input_mode="memory_augmented_input",
                xlstm_memory_gate_enabled=True,
            ),
        )
        model.train()

        logits, states = model(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            rope_cache_scope="train",
            use_kv_cache=False,
        )
        logits.float().sum().backward()

        memory_gate = model.layers[0].xlstm_memory.memory_gate
        self.assertIsNotNone(memory_gate)
        self.assertIsNotNone(memory_gate.weight.grad)
        self.assertIsNotNone(states[0].xlstm_memory)
        self.assertGreaterEqual(states[0].xlstm_memory.adapter_delta_norm, 0.0)

    def test_current_paged_kv_forward_keeps_attention_projection_gradients(self):
        model = LPTV2(32, build_tiny_v2_config())
        model.train()

        logits, _ = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
        logits.float().sum().backward()

        self.assertIsNotNone(model.layers[0].attention_mixer.k_proj.weight.grad)
        self.assertIsNotNone(model.layers[0].attention_mixer.v_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
