import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import GlobalConfig
from lpt_inference import InferenceSession
from lpt_inference.inference import count_text_tokens, generate_responses_with_token_counts, run_chat_session
from lpt_model import ATTENTION_BLOCK_TYPE, LPT, ModelConfig
from lpt_protocol import DS_BOS_TOKEN, DS_EOS_TOKEN, DS_PAD_TOKEN, render_prompt_from_messages


def build_tiny_attention_config():
    return ModelConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cla_share_every_n_layers=1,
        layer_block_types=(ATTENTION_BLOCK_TYPE, ATTENTION_BLOCK_TYPE),
    )


def build_tiny_longrope_attention_config():
    return build_tiny_attention_config().with_overrides(
        original_max_len=4,
        longrope2_target_length=8,
        longrope2_long_factors=2.0,
    )


class DummyBatchTokenizer:
    def __init__(self):
        self.special_tokens = (
            DS_PAD_TOKEN,
            DS_BOS_TOKEN,
            DS_EOS_TOKEN,
        )
        self.pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        )
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.special_tokens)
        }
        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }
        self.next_id = len(self.token_to_id)
        self.bos_token = DS_BOS_TOKEN
        self.pad_token = DS_PAD_TOKEN
        self.eos_token = DS_EOS_TOKEN
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]

    def _split_text(self, text):
        parts = []
        cursor = 0
        for match in self.pattern.finditer(text):
            if match.start() > cursor:
                parts.extend(list(text[cursor:match.start()]))
            parts.append(match.group(0))
            cursor = match.end()
        if cursor < len(text):
            parts.extend(list(text[cursor:]))
        return [part for part in parts if part]

    def _encode_text(self, text):
        token_ids = []
        for piece in self._split_text(text):
            if piece not in self.token_to_id:
                self.token_to_id[piece] = self.next_id
                self.id_to_token[self.next_id] = piece
                self.next_id += 1
            token_ids.append(self.token_to_id[piece])
        return token_ids

    def __call__(
        self,
        texts,
        add_special_tokens=False,
        padding=False,
        padding_side="right",
        return_tensors=None,
        return_attention_mask=False,
    ):
        if add_special_tokens:
            raise AssertionError("DummyBatchTokenizer 不支持 add_special_tokens=True。")

        if isinstance(texts, str):
            return {"input_ids": self._encode_text(texts)}

        encoded_rows = [self._encode_text(text) for text in texts]
        if not padding:
            return {"input_ids": encoded_rows}

        max_width = max(len(row) for row in encoded_rows)
        padded_rows = []
        attention_masks = []
        for row in encoded_rows:
            pad_width = max_width - len(row)
            pad_values = [self.pad_token_id] * pad_width
            if padding_side == "left":
                padded_row = pad_values + row
                attention_mask = [0] * pad_width + [1] * len(row)
            else:
                padded_row = row + pad_values
                attention_mask = [1] * len(row) + [0] * pad_width
            padded_rows.append(padded_row)
            attention_masks.append(attention_mask)

        result = {"input_ids": padded_rows}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
            if return_attention_mask:
                result["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
        elif return_attention_mask:
            result["attention_mask"] = attention_masks
        return result

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = []
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if token == self.pad_token:
                continue
            if skip_special_tokens and token in self.special_tokens:
                continue
            pieces.append(token)
        return "".join(pieces)


class StubGenerateModel:
    def __init__(self, output_sequence):
        self.output_sequence = output_sequence

    def generate(self, prompt_tokens, config=None, attention_mask=None, pad_token_id=None, eos_token_id=None):
        return self.output_sequence.to(prompt_tokens.device)


class TestInferenceTokenCounting(unittest.TestCase):
    def setUp(self):
        self.original_device = GlobalConfig.device
        self.tokenizer = DummyBatchTokenizer()
        GlobalConfig.device = torch.device("cpu")

    def tearDown(self):
        GlobalConfig.device = self.original_device

    def _build_stub_model(self, conversations):
        prompts = [
            render_prompt_from_messages(
                messages,
                template_version=GlobalConfig.chat_template_version,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]
        encoded_batch = self.tokenizer(
            prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_attention_mask=True,
        )

        first_output_ids = self.tokenizer("好", add_special_tokens=False)["input_ids"]
        second_output_ids = self.tokenizer("回答", add_special_tokens=False)["input_ids"]

        output_rows = [
            torch.tensor(
                encoded_batch["input_ids"][0].tolist()
                + first_output_ids
                + [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
                dtype=torch.long,
            ),
            torch.tensor(
                encoded_batch["input_ids"][1].tolist()
                + second_output_ids
                + [self.tokenizer.eos_token_id],
                dtype=torch.long,
            ),
        ]
        return StubGenerateModel(torch.stack(output_rows)), encoded_batch

    def test_count_text_tokens_uses_current_tokenizer_rules(self):
        self.assertEqual(count_text_tokens(self.tokenizer, "你好"), 2)
        self.assertEqual(count_text_tokens(self.tokenizer, f"{DS_EOS_TOKEN}好"), 2)

    def test_generate_responses_with_token_counts_tracks_input_and_output_tokens(self):
        conversations = [
            [{"role": "user", "content": "短问题"}],
            [{"role": "user", "content": "这是一个更长的问题"}],
        ]
        model, encoded_batch = self._build_stub_model(conversations)

        results = generate_responses_with_token_counts(
            model,
            self.tokenizer,
            conversations=conversations,
            config=None,
        )

        self.assertEqual([result.text for result in results], ["好", "回答"])
        self.assertEqual([result.output_token_count for result in results], [1, 2])
        self.assertEqual(
            [result.input_token_count for result in results],
            encoded_batch["attention_mask"].sum(dim=1).tolist(),
        )

    def test_run_chat_session_prints_token_counts_for_batch_outputs(self):
        conversations = [
            [{"role": "user", "content": "短问题"}],
            [{"role": "user", "content": "这是一个更长的问题"}],
        ]
        model, _ = self._build_stub_model(conversations)

        with io.StringIO() as buffer, redirect_stdout(buffer):
            outputs = run_chat_session(
                model=model,
                tokenizer=self.tokenizer,
                conversations=conversations,
                config=None,
            )
            printed = buffer.getvalue()

        self.assertEqual(outputs, ["好", "回答"])
        self.assertIn("输入 token 数:", printed)
        self.assertIn("输出 token 数:", printed)


class TestInferenceSession(unittest.TestCase):
    def setUp(self):
        self.original_device = GlobalConfig.device
        self.original_train_rope_cache_max_sequence_length = GlobalConfig.train_rope_cache_max_sequence_length
        self.original_inference_rope_cache_max_sequence_length = (
            GlobalConfig.inference_rope_cache_max_sequence_length
        )
        GlobalConfig.device = torch.device("cpu")

    def tearDown(self):
        GlobalConfig.device = self.original_device
        GlobalConfig.train_rope_cache_max_sequence_length = (
            self.original_train_rope_cache_max_sequence_length
        )
        GlobalConfig.inference_rope_cache_max_sequence_length = (
            self.original_inference_rope_cache_max_sequence_length
        )

    def test_inference_session_tracks_left_padding_masks_and_positions(self):
        model = LPT(vocabulary_size=32, config=build_tiny_attention_config())
        model.eval()
        session = InferenceSession(model)

        prompt_tokens = torch.tensor([
            [0, 0, 5, 7],
            [0, 3, 4, 8],
        ])
        attention_mask = torch.tensor([
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ])

        logits = session.prefill(prompt_tokens, attention_mask=attention_mask)
        state = session.export_state()

        self.assertEqual(logits.shape, (2, 4, 32))
        self.assertIsNotNone(state)
        self.assertTrue(torch.equal(state.attention_mask, attention_mask))
        self.assertTrue(
            torch.equal(
                state.position_ids,
                attention_mask.cumsum(dim=-1).sub(1).clamp_min(0),
            )
        )
        self.assertEqual(len(state.layer_states), model.num_state_slots)

        next_tokens = torch.tensor([[9], [10]])
        next_logits = session.append(next_tokens)
        next_state = session.export_state()

        self.assertEqual(next_logits.shape, (2, 1, 32))
        self.assertTrue(
            torch.equal(
                next_state.attention_mask,
                torch.tensor([
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                ]),
            )
        )

    def test_inference_session_rebuilds_longrope_cache_on_threshold_switch(self):
        model = LPT(vocabulary_size=32, config=build_tiny_longrope_attention_config())
        model.eval()
        session = InferenceSession(model)

        prompt_tokens = torch.tensor([[1, 2, 3, 4]])
        logits = session.prefill(prompt_tokens)
        state = session.export_state()

        self.assertEqual(logits.shape, (1, 4, 32))
        self.assertFalse(state.using_rescaled_rope)

        next_logits = session.append(torch.tensor([[5]]))
        next_state = session.export_state()

        self.assertEqual(next_logits.shape, (1, 1, 32))
        self.assertTrue(next_state.using_rescaled_rope)
        self.assertEqual(next_state.token_ids.shape, (1, 5))
        self.assertIsNone(session.rebuild_on_switch())

    def test_inference_session_uses_inference_rope_cache_limit(self):
        GlobalConfig.train_rope_cache_max_sequence_length = 4
        GlobalConfig.inference_rope_cache_max_sequence_length = 8

        model = LPT(vocabulary_size=32, config=build_tiny_attention_config())
        model.eval()
        session = InferenceSession(model)

        prompt_tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        logits = session.prefill(prompt_tokens)

        self.assertEqual(logits.shape, (1, 6, 32))
        self.assertEqual(model.get_rope_cache("train").max_seq_len, 4)
        self.assertEqual(model.get_rope_cache("inference").max_seq_len, 8)

    def test_inference_session_reset_clears_cached_state(self):
        model = LPT(vocabulary_size=32, config=build_tiny_attention_config())
        model.eval()
        session = InferenceSession(model)

        session.prefill(torch.tensor([[1, 2, 3]]))
        self.assertIsNotNone(session.export_state())

        session.reset()
        self.assertIsNone(session.export_state())


if __name__ == "__main__":
    unittest.main()
