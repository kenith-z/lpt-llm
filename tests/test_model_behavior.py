import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import GenerationConfig, GlobalConfig, load_longrope2_factors_file
from lpt_model.longrope import MixedLongRoPEScaledRotaryEmbedding
from lpt_model import (
    ATTENTION_BLOCK_TYPE,
    LPT,
    ModelConfig,
    RETNET_BLOCK_TYPE,
    RetNetBlock,
    list_architecture_mismatches,
)


class TestLPTModelBehavior(unittest.TestCase):
    def setUp(self):
        self.original_train_rope_cache_max_sequence_length = GlobalConfig.train_rope_cache_max_sequence_length
        self.original_inference_rope_cache_max_sequence_length = (
            GlobalConfig.inference_rope_cache_max_sequence_length
        )
        self.model = LPT(vocabulary_size=32)

    def tearDown(self):
        GlobalConfig.train_rope_cache_max_sequence_length = (
            self.original_train_rope_cache_max_sequence_length
        )
        GlobalConfig.inference_rope_cache_max_sequence_length = (
            self.original_inference_rope_cache_max_sequence_length
        )

    @staticmethod
    def _build_pure_retnet_config(chunk_size):
        return ModelConfig(
            num_layers=4,
            cla_share_every_n_layers=1,
            layer_block_types=(RETNET_BLOCK_TYPE,) * 4,
            retnet_chunk_size=chunk_size,
        )

    @staticmethod
    def _build_pure_attention_config():
        return ModelConfig(
            num_layers=6,
            cla_share_every_n_layers=2,
            layer_block_types=(ATTENTION_BLOCK_TYPE,) * 6,
        )

    @staticmethod
    def _build_longrope2_attention_config():
        return ModelConfig(
            num_layers=6,
            cla_share_every_n_layers=2,
            layer_block_types=(ATTENTION_BLOCK_TYPE,) * 6,
            original_max_len=4,
            longrope2_target_length=8,
            longrope2_long_factors=2.0,
        )

    @staticmethod
    def _build_longrope2_attention_factors_config(factors):
        return ModelConfig(
            num_layers=6,
            cla_share_every_n_layers=2,
            layer_block_types=(ATTENTION_BLOCK_TYPE,) * 6,
            original_max_len=4,
            longrope2_target_length=8,
            longrope2_long_factors=factors,
        )

    def test_hybrid_layout_builds_expected_slots(self):
        self.assertEqual(self.model.num_state_slots, 12)
        self.assertEqual(
            self.model.layer_block_types,
            (
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
                RETNET_BLOCK_TYPE,
                ATTENTION_BLOCK_TYPE,
            ),
        )
        self.assertEqual(self.model.layer_state_group_ids, (4, 0, 5, 0, 6, 1, 7, 1, 8, 2, 9, 2, 10, 3, 11, 3))
        self.assertEqual(self.model.layer_to_state_slot, [0, 1, 2, 1, 3, 4, 5, 4, 6, 7, 8, 7, 9, 10, 11, 10])

    def test_hybrid_layout_contains_retnet_layers(self):
        retnet_mixers = [
            layer.sequence_mixer
            for layer in self.model.layers
            if isinstance(layer.sequence_mixer, RetNetBlock)
        ]
        self.assertEqual(len(retnet_mixers), 8)

    def test_embedding_and_lm_head_share_weights(self):
        self.assertEqual(
            self.model.token_embedding.weight.data_ptr(),
            self.model.lm_head.weight.data_ptr(),
        )

    def test_repetition_penalty_ignores_masked_history(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
        token_history = torch.tensor([[0, 1], [0, 2]])
        history_mask = torch.tensor([[False, True], [False, False]])

        adjusted = self.model._apply_repetition_penalty_vectorized(
            logits.clone(),
            token_history,
            penalty=2.0,
            history_mask=history_mask,
        )

        self.assertAlmostEqual(adjusted[0, 0].item(), logits[0, 0].item())
        self.assertAlmostEqual(adjusted[0, 1].item(), logits[0, 1].item() / 2.0)
        self.assertTrue(torch.allclose(adjusted[1], logits[1]))

    def test_top_k_filters_to_argmax_when_k_is_one(self):
        logits = torch.tensor([[0.1, 2.0, 1.2]])
        config = GenerationConfig(
            do_sample=False,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            max_length=1,
        )

        probs = self.model._temperature_and_top_p(logits, config)
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        self.assertTrue(torch.allclose(probs, expected))

    def test_generate_greedy_decode_skips_sampling_pipeline(self):
        prompt_tokens = torch.tensor([[1, 2]])
        config = GenerationConfig(
            do_sample=False,
            temperature=0.0,
            top_k=1,
            top_p=0.1,
            max_length=1,
        )
        original_temperature_and_top_p = self.model._temperature_and_top_p

        def fail_if_called(*args, **kwargs):
            raise AssertionError("贪心解码不应进入采样概率计算。")

        self.model._temperature_and_top_p = fail_if_called
        try:
            output = self.model.generate(prompt_tokens, config)
        finally:
            self.model._temperature_and_top_p = original_temperature_and_top_p

        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(torch.equal(output[:, :2], prompt_tokens))

    def test_forward_supports_padding_mask_and_layer_states(self):
        input_ids = torch.tensor([
            [0, 0, 5, 7],
            [0, 3, 4, 8],
        ])
        attention_mask = torch.tensor([
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ])
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        logits, layer_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        self.assertEqual(logits.shape, (2, 4, 32))
        self.assertEqual(len(layer_states), self.model.num_state_slots)

        next_tokens = torch.tensor([[9], [10]])
        next_attention_mask = torch.tensor([
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ])
        next_position_ids = next_attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)[:, -1:]

        next_logits, next_layer_states = self.model(
            next_tokens,
            position_ids=next_position_ids,
            attention_mask=next_attention_mask,
            layer_states=layer_states,
        )

        self.assertEqual(next_logits.shape, (2, 1, 32))
        self.assertEqual(len(next_layer_states), self.model.num_state_slots)

    def test_forward_with_segment_ids_matches_independent_sequences_when_packed(self):
        self.model.eval()

        separate_input_ids = torch.tensor([
            [1, 2, 3],
            [4, 5, 0],
        ])
        separate_attention_mask = torch.tensor([
            [1, 1, 1],
            [1, 1, 0],
        ])
        separate_position_ids = separate_attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        separate_logits, _ = self.model(
            separate_input_ids,
            position_ids=separate_position_ids,
            attention_mask=separate_attention_mask,
        )

        packed_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        packed_attention_mask = torch.ones(1, 5, dtype=torch.long)
        packed_position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        packed_segment_ids = torch.tensor([[1, 1, 1, 2, 2]])

        packed_logits, _ = self.model(
            packed_input_ids,
            position_ids=packed_position_ids,
            attention_mask=packed_attention_mask,
            segment_ids=packed_segment_ids,
        )

        self.assertTrue(
            torch.allclose(
                packed_logits[0, :3],
                separate_logits[0, :3],
                atol=5e-4,
                rtol=5e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                packed_logits[0, 3:5],
                separate_logits[1, :2],
                atol=5e-4,
                rtol=5e-4,
            )
        )

    def test_rope_cache_scope_split_uses_train_and_inference_limits(self):
        GlobalConfig.train_rope_cache_max_sequence_length = 4
        GlobalConfig.inference_rope_cache_max_sequence_length = 8
        model = LPT(vocabulary_size=32, config=self._build_pure_attention_config())
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention_mask = torch.ones_like(input_ids)
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        with self.assertRaisesRegex(ValueError, "train RoPE 缓存上限"):
            model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                rope_cache_scope="train",
            )

        default_logits, _ = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        self.assertEqual(default_logits.shape, (1, 6, 32))

        inference_logits, _ = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            rope_cache_scope="inference",
        )
        self.assertEqual(inference_logits.shape, (1, 6, 32))
        self.assertEqual(model.get_rope_cache("train").max_seq_len, 4)
        self.assertEqual(model.get_rope_cache("inference").max_seq_len, 8)

    def test_architecture_mismatch_detects_different_grouping(self):
        alternate_model = LPT(
            vocabulary_size=32,
            config=ModelConfig().with_overrides(cla_share_every_n_layers=1),
        )
        alternate_checkpoint = {
            "model_architecture_metadata": {
                "hidden_size": alternate_model.config.hidden_size,
                "num_heads": alternate_model.config.num_heads,
                "num_kv_heads": alternate_model.config.num_kv_heads,
                "head_dim": alternate_model.config.head_dim,
                "num_layers": alternate_model.config.num_layers,
                "cla_share_every_n_layers": alternate_model.config.cla_share_every_n_layers,
                "layer_block_types": alternate_model.layer_block_types,
                "layer_state_group_ids": alternate_model.layer_state_group_ids,
                "retnet_value_factor": alternate_model.config.retnet_value_factor,
                "retnet_gate_fn": alternate_model.config.retnet_gate_fn,
                "retnet_chunk_size": alternate_model.config.retnet_chunk_size,
                "rope_base": alternate_model.config.rope_base,
                "original_max_len": alternate_model.config.original_max_len,
                "longrope2_target_length": alternate_model.config.longrope2_target_length,
                "longrope2_long_factors": alternate_model.longrope2_long_factors,
                "longrope2_factor_max_sequence_length": alternate_model.config.longrope2_factor_max_sequence_length,
                "longrope2_magnitude_scaling_policy": alternate_model.config.longrope2_magnitude_scaling_policy,
                "longrope2_mscale_factors": alternate_model.config.longrope2_mscale_factors,
                "num_state_slots": alternate_model.num_state_slots,
            },
        }

        mismatches = list_architecture_mismatches(alternate_checkpoint, self.model)
        mismatch_keys = {key for key, _, _ in mismatches}

        self.assertIn("cla_share_every_n_layers", mismatch_keys)
        self.assertIn("layer_state_group_ids", mismatch_keys)
        self.assertIn("num_state_slots", mismatch_keys)

    def test_architecture_mismatch_rejects_missing_nested_metadata(self):
        with self.assertRaises(ValueError):
            list_architecture_mismatches({"num_layers": self.model.config.num_layers}, self.model)

    def test_pure_attention_recurrent_matches_parallel(self):
        model = LPT(vocabulary_size=32, config=self._build_pure_attention_config())
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attention_mask = torch.ones(1, input_ids.size(1), dtype=torch.long)
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        parallel_logits, _ = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        recurrent_logits = []
        layer_states = None
        for step_index in range(input_ids.size(1)):
            step_input_ids = input_ids[:, step_index:step_index + 1]
            step_attention_mask = attention_mask[:, :step_index + 1]
            step_position_ids = position_ids[:, step_index:step_index + 1]

            step_logits, layer_states = model(
                step_input_ids,
                position_ids=step_position_ids,
                attention_mask=step_attention_mask,
                layer_states=layer_states,
            )
            recurrent_logits.append(step_logits)

        recurrent_logits = torch.cat(recurrent_logits, dim=1)
        self.assertTrue(torch.allclose(parallel_logits, recurrent_logits, atol=5e-4, rtol=5e-4))

    def test_retnet_hybrid_generate_reuses_layer_states(self):
        prompt_tokens = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        _, layer_states = self.model(
            prompt_tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        next_tokens = torch.tensor([[4]])
        next_attention_mask = torch.tensor([[1, 1, 1, 1]])
        next_position_ids = next_attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)[:, -1:]

        logits, next_layer_states = self.model(
            next_tokens,
            position_ids=next_position_ids,
            attention_mask=next_attention_mask,
            layer_states=layer_states,
        )

        self.assertEqual(logits.shape, (1, 1, 32))
        self.assertEqual(len(next_layer_states), self.model.num_state_slots)

    def test_hybrid_recurrent_matches_parallel(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        attention_mask = torch.ones(1, input_ids.size(1), dtype=torch.long)
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        parallel_logits, _ = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        recurrent_logits = []
        layer_states = None
        for step_index in range(input_ids.size(1)):
            step_input_ids = input_ids[:, step_index:step_index + 1]
            step_attention_mask = attention_mask[:, :step_index + 1]
            step_position_ids = position_ids[:, step_index:step_index + 1]

            step_logits, layer_states = self.model(
                step_input_ids,
                position_ids=step_position_ids,
                attention_mask=step_attention_mask,
                layer_states=layer_states,
            )
            recurrent_logits.append(step_logits)

        recurrent_logits = torch.cat(recurrent_logits, dim=1)
        self.assertTrue(torch.allclose(parallel_logits, recurrent_logits, atol=5e-4, rtol=5e-4))

    def test_longrope2_attention_generate_crosses_threshold(self):
        model = LPT(vocabulary_size=32, config=self._build_longrope2_attention_config())
        model.eval()

        prompt_tokens = torch.tensor([[1, 2, 3, 4]])
        config = GenerationConfig(
            do_sample=False,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            max_length=3,
        )
        output = model.generate(prompt_tokens, config)
        self.assertEqual(output.shape, (1, 7))

    def test_longrope2_manual_layer_states_require_rebuild_after_threshold(self):
        model = LPT(vocabulary_size=32, config=self._build_longrope2_attention_config())
        model.eval()

        prompt_tokens = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.ones(1, prompt_tokens.size(1), dtype=torch.long)
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        _, layer_states = model(
            prompt_tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        next_tokens = torch.tensor([[5]])
        next_attention_mask = torch.ones(1, 5, dtype=torch.long)
        next_position_ids = next_attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)[:, -1:]

        with self.assertRaises(ValueError):
            model(
                next_tokens,
                position_ids=next_position_ids,
                attention_mask=next_attention_mask,
                layer_states=layer_states,
            )

    def test_longrope2_factors_file_imports_to_config_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factors_path = Path(temp_dir) / "result_final.csv"
            factors_path.write_text("\n".join(["2.0"] * 32), encoding="utf-8")

            config = self._build_longrope2_attention_factors_config(
                load_longrope2_factors_file(factors_path)
            )
            model = LPT(vocabulary_size=32, config=config)

        self.assertEqual(model.longrope2_long_factors, tuple([2.0] * 32))
        self.assertEqual(config.longrope2_long_factors, tuple([2.0] * 32))

    def test_longrope2_factors_array_validates_dimension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            factors_path = Path(temp_dir) / "result_final.csv"
            factors_path.write_text("\n".join(["2.0"] * 31), encoding="utf-8")

            config = self._build_longrope2_attention_factors_config(
                load_longrope2_factors_file(factors_path)
            )
            with self.assertRaises(ValueError):
                LPT(vocabulary_size=32, config=config)

    def test_longrope2_embedding_mode_is_scope_specific(self):
        config = ModelConfig(
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cla_share_every_n_layers=1,
            layer_block_types=(ATTENTION_BLOCK_TYPE, ATTENTION_BLOCK_TYPE),
            original_max_len=4,
            longrope2_target_length=8,
            longrope2_long_factors=2.0,
            longrope2_train_embedding_mode="dynamic",
            longrope2_inference_embedding_mode="static",
        )
        model = LPT(vocabulary_size=32, config=config)

        self.assertEqual(model.get_rope_cache("train").embedding_mode, "dynamic")
        self.assertEqual(model.get_rope_cache("inference").embedding_mode, "static")

    def test_longrope2_mixed_embedding_respects_reset_position_ids(self):
        config = ModelConfig(
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cla_share_every_n_layers=1,
            layer_block_types=(ATTENTION_BLOCK_TYPE, ATTENTION_BLOCK_TYPE),
            original_max_len=2,
            longrope2_target_length=8,
            longrope2_long_factors=2.0,
            longrope2_train_embedding_mode="mixed",
            longrope2_mixed_original_window=2,
        )
        model = LPT(vocabulary_size=32, config=config)
        rope_cache = model.get_rope_cache("train")
        q = torch.randn(1, config.num_heads, 4, config.head_dim)
        position_ids = torch.tensor([[0, 1, 0, 2]])

        cos, sin = rope_cache._lookup_cos_sin(q, position_ids)
        original_cos, original_sin = rope_cache._forward_cos_sin(
            rope_cache.original_embedding,
            q,
            position_ids,
        )
        long_cos, long_sin = rope_cache._forward_cos_sin(
            rope_cache.long_embedding,
            q,
            position_ids,
        )

        position_mask = position_ids.lt(2).unsqueeze(-1)
        expected_cos = torch.where(position_mask, original_cos, long_cos)
        expected_sin = torch.where(position_mask, original_sin, long_sin)
        self.assertTrue(torch.allclose(cos.squeeze(1), expected_cos))
        self.assertTrue(torch.allclose(sin.squeeze(1), expected_sin))
        self.assertTrue(torch.allclose(cos.squeeze(1)[0, 2], original_cos[0, 2]))

    def test_official_mixed_longrope_embedding_can_be_reused(self):
        embedding = MixedLongRoPEScaledRotaryEmbedding(
            dim=8,
            rescale_factors=[2.0, 2.0, 2.0, 2.0],
            start_token_idx=1,
            original_embeddings=(torch.ones(1, 3, 8), torch.ones(1, 3, 8)),
            max_position_embeddings=8,
            original_max_position_embeddings=4,
            model_type="LPT",
            device=torch.device("cpu"),
        )
        x = torch.randn(1, 2, 3, 8)
        position_ids = torch.tensor([[0, 1, 2]])

        first_cos, first_sin = embedding(x, position_ids)
        second_cos, second_sin = embedding(x, position_ids)

        self.assertEqual(first_cos.shape, (1, 3, 8))
        self.assertEqual(first_sin.shape, (1, 3, 8))
        self.assertTrue(torch.allclose(first_cos, second_cos))
        self.assertTrue(torch.allclose(first_sin, second_sin))

    def test_pure_retnet_chunkwise_matches_parallel(self):
        parallel_config = self._build_pure_retnet_config(chunk_size=32)
        chunkwise_config = self._build_pure_retnet_config(chunk_size=2)

        parallel_model = LPT(vocabulary_size=32, config=parallel_config)
        chunkwise_model = LPT(vocabulary_size=32, config=chunkwise_config)
        chunkwise_model.load_state_dict(parallel_model.state_dict())
        parallel_model.eval()
        chunkwise_model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        parallel_logits, parallel_states = parallel_model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        chunkwise_logits, chunkwise_states = chunkwise_model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        self.assertTrue(torch.allclose(parallel_logits, chunkwise_logits, atol=1e-4, rtol=1e-4))
        for parallel_state, chunkwise_state in zip(parallel_states, chunkwise_states):
            self.assertIsNotNone(parallel_state)
            self.assertIsNotNone(chunkwise_state)
            self.assertEqual(parallel_state.state_type, chunkwise_state.state_type)
            self.assertTrue(
                torch.allclose(
                    parallel_state.tensors[0],
                    chunkwise_state.tensors[0],
                    atol=2e-4,
                    rtol=2e-4,
                )
            )

    def test_pure_retnet_recurrent_matches_parallel(self):
        pure_retnet_config = self._build_pure_retnet_config(chunk_size=32)
        model = LPT(vocabulary_size=32, config=pure_retnet_config)
        model.eval()

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        parallel_logits, _ = model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        recurrent_logits = []
        layer_states = None
        for step_index in range(input_ids.size(1)):
            step_input_ids = input_ids[:, step_index:step_index + 1]
            step_attention_mask = attention_mask[:, :step_index + 1]
            step_position_ids = position_ids[:, step_index:step_index + 1]

            step_logits, layer_states = model(
                step_input_ids,
                position_ids=step_position_ids,
                attention_mask=step_attention_mask,
                layer_states=layer_states,
            )
            recurrent_logits.append(step_logits)

        recurrent_logits = torch.cat(recurrent_logits, dim=1)
        self.assertTrue(torch.allclose(parallel_logits, recurrent_logits, atol=1e-4, rtol=1e-4))

    def test_training_backward_supports_gradient_checkpointing(self):
        original_flag = GlobalConfig.gradient_checkpointing_enabled
        GlobalConfig.gradient_checkpointing_enabled = True
        try:
            model = LPT(vocabulary_size=32)
            model.train()

            input_ids = torch.tensor([[1, 2, 3, 4]])
            attention_mask = torch.tensor([[1, 1, 1, 1]])
            position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

            logits, _ = model(
                input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].transpose(1, 2),
                input_ids[:, 1:],
            )
            loss.backward()

            self.assertIsNotNone(model.token_embedding.weight.grad)
            self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0.0)
        finally:
            GlobalConfig.gradient_checkpointing_enabled = original_flag


if __name__ == "__main__":
    unittest.main()
