import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import GlobalConfig
from lpt_model import ATTENTION_BLOCK_TYPE, LPT, ModelConfig
from lpt_protocol import DS_BOS_TOKEN, DS_EOS_TOKEN, DS_PAD_TOKEN
from lpt_training import configure_training_runtime, load_checkpoint, prepare_tokenizer, train
from lpt_training.train import (
    _build_longrope2_window_sampler,
    _build_optimizer_parameter_groups,
    _ensure_longrope2_dataset_factors,
    _forward_batch,
)


class DummyTokenizer:
    def __init__(self):
        self.special_tokens = (
            DS_BOS_TOKEN,
            DS_PAD_TOKEN,
            DS_EOS_TOKEN,
        )
        self.pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        )
        self.token_to_id = {
            token: index
            for index, token in enumerate(self.special_tokens, start=1)
        }
        self.next_id = len(self.token_to_id) + 1
        self.bos_token = DS_BOS_TOKEN
        self.eos_token = DS_EOS_TOKEN
        self.pad_token = DS_PAD_TOKEN
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.name_or_path = "dummy-tokenizer"

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token)

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

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("DummyTokenizer 不支持 add_special_tokens=True。")

        input_ids = []
        for piece in self._split_text(text):
            if piece not in self.token_to_id:
                self.token_to_id[piece] = self.next_id
                self.next_id += 1
            input_ids.append(self.token_to_id[piece])
        return {"input_ids": input_ids}


def build_tiny_attention_config():
    return ModelConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cla_share_every_n_layers=1,
        layer_block_types=(ATTENTION_BLOCK_TYPE, ATTENTION_BLOCK_TYPE),
    )


def build_chat_samples():
    return [
        {
            "id": "chat-001",
            "type": "chat",
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "世界"},
            ],
        },
        {
            "id": "chat-002",
            "type": "chat",
            "messages": [
                {"role": "user", "content": "请介绍你自己"},
                {"role": "assistant", "content": "我是一个测试助手"},
            ],
        },
    ]


class SingleEpochTrainingProfile:
    batch_size = 2
    target_total_epochs = 1
    learning_rate = 3e-4
    warmup_ratio = 0.0
    save_optimizer = True
    save_scheduler = True
    key_checkpoints = ()
    weight_decay = 0.1
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    random_seed = 7
    deterministic_algorithms = True
    tensorboard_enabled = False
    log_interval_steps = 1
    eval_interval_steps = 1
    eval_batch_size = 1
    eval_max_batches = 1


class TwoEpochTrainingProfile(SingleEpochTrainingProfile):
    target_total_epochs = 2


class TwoEpochTrainingProfileWithKeyCheckpoint(TwoEpochTrainingProfile):
    key_checkpoints = (1,)


class TensorBoardTrainingProfile(SingleEpochTrainingProfile):
    tensorboard_enabled = True
    eval_interval_steps = None


class LongRoPE2WindowSamplingProfile(SingleEpochTrainingProfile):
    longrope2_window_sampling_enabled = True
    longrope2_window_lengths = (4, 8)
    longrope2_window_sampling_weights = (0.0, 1.0)


class TestTrainingRecipe(unittest.TestCase):
    def setUp(self):
        self.original_device = GlobalConfig.device
        self.original_training_stage = GlobalConfig.training_stage
        self.original_gradient_checkpointing_enabled = GlobalConfig.gradient_checkpointing_enabled
        self.original_train_max_sequence_length = GlobalConfig.train_max_sequence_length
        self.original_train_rope_cache_max_sequence_length = GlobalConfig.train_rope_cache_max_sequence_length
        self.original_inference_rope_cache_max_sequence_length = (
            GlobalConfig.inference_rope_cache_max_sequence_length
        )
        self.original_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        self.original_cudnn_deterministic = getattr(torch.backends.cudnn, "deterministic", False)
        self.original_cudnn_benchmark = getattr(torch.backends.cudnn, "benchmark", False)

        GlobalConfig.device = torch.device("cpu")
        GlobalConfig.training_stage = "chat_sft"
        GlobalConfig.gradient_checkpointing_enabled = False
        self.tokenizer = prepare_tokenizer(DummyTokenizer())

    def tearDown(self):
        GlobalConfig.device = self.original_device
        GlobalConfig.training_stage = self.original_training_stage
        GlobalConfig.gradient_checkpointing_enabled = self.original_gradient_checkpointing_enabled
        GlobalConfig.train_max_sequence_length = self.original_train_max_sequence_length
        GlobalConfig.train_rope_cache_max_sequence_length = (
            self.original_train_rope_cache_max_sequence_length
        )
        GlobalConfig.inference_rope_cache_max_sequence_length = (
            self.original_inference_rope_cache_max_sequence_length
        )
        torch.use_deterministic_algorithms(self.original_deterministic_algorithms)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = self.original_cudnn_deterministic
            torch.backends.cudnn.benchmark = self.original_cudnn_benchmark

    def _build_model(self, *, seed=7):
        configure_training_runtime(seed=seed, deterministic_algorithms=True)
        model = LPT(vocabulary_size=256, config=build_tiny_attention_config())
        model.to(GlobalConfig.device)
        return model

    def test_optimizer_parameter_groups_split_weight_decay_and_no_decay(self):
        model = self._build_model()
        _, group_summary = _build_optimizer_parameter_groups(model, weight_decay=0.1)

        self.assertIn("token_embedding.weight", group_summary["no_decay_parameter_names"])
        self.assertIn(
            "layers.0.sequence_mixer.q_proj.weight",
            group_summary["decay_parameter_names"],
        )
        self.assertGreater(group_summary["decay_parameter_count"], 0)
        self.assertGreater(group_summary["no_decay_parameter_count"], 0)

    def test_forward_batch_uses_train_rope_cache_limit_even_in_eval_mode(self):
        GlobalConfig.train_rope_cache_max_sequence_length = 4
        GlobalConfig.inference_rope_cache_max_sequence_length = 8

        model = self._build_model()
        model.eval()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels = input_ids.clone()
        attention_mask = torch.ones_like(input_ids)
        position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)

        with self.assertRaisesRegex(ValueError, "train RoPE 缓存上限"):
            _forward_batch(
                model,
                input_ids,
                labels,
                attention_mask,
                position_ids=position_ids,
            )

    def test_longrope2_window_sampler_uses_configured_weights(self):
        GlobalConfig.train_max_sequence_length = 8
        GlobalConfig.train_rope_cache_max_sequence_length = 8
        model = self._build_model()

        sampler = _build_longrope2_window_sampler(
            model.config,
            LongRoPE2WindowSamplingProfile,
            seed=7,
            randomize=True,
        )

        self.assertEqual(sampler.next_length(), 8)
        self.assertEqual(sampler.to_dict()["window_lengths"], (4, 8))

    def test_longrope2_factor_refresh_reuses_recorded_dataset_max(self):
        configure_training_runtime(seed=7, deterministic_algorithms=True)
        config = build_tiny_attention_config().with_overrides(
            longrope2_long_factors=2.0,
            longrope2_factor_max_sequence_length=128,
        )
        model = LPT(vocabulary_size=256, config=config)
        original_factors = model.config.longrope2_long_factors

        length_summary = _ensure_longrope2_dataset_factors(
            model,
            build_chat_samples(),
            self.tokenizer,
            enabled=True,
        )

        self.assertLessEqual(length_summary["max_sequence_length"], 128)
        self.assertEqual(model.config.longrope2_factor_max_sequence_length, 128)
        self.assertEqual(model.config.longrope2_long_factors, original_factors)

    def test_train_logs_eval_metrics_and_checkpoint_recipe_fields(self):
        model = self._build_model()
        train_dataset = build_chat_samples()
        eval_dataset = train_dataset[:1]

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "checkpoints" / "latest"
            inference_weight_path = Path(temp_dir) / "weights" / "model_weights.pth"

            train(
                model=model,
                dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                save_path=str(checkpoint_root),
                manifest_path=Path("data/manifests/chat_sft.json"),
                eval_manifest_path=Path("data/manifests/chat_sft_eval.json"),
                training_profile=SingleEpochTrainingProfile,
                inference_weight_path=str(inference_weight_path),
            )

            checkpoint = load_checkpoint(checkpoint_root)
            self.assertEqual(checkpoint["gradient_accumulation_steps"], 1)
            self.assertEqual(checkpoint["weight_decay"], 0.1)
            self.assertEqual(checkpoint["random_seed"], 7)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertEqual(checkpoint["optimizer_step"], 1)
            self.assertIsNotNone(checkpoint["latest_eval_loss"])
            self.assertIsNotNone(checkpoint["latest_eval_ppl"])
            self.assertEqual(
                checkpoint["eval_manifest"],
                "data\\manifests\\chat_sft_eval.json",
            )
            self.assertEqual(
                checkpoint["longrope2_training_strategy"]["train_embedding_mode"],
                "mixed",
            )
            self.assertEqual(
                checkpoint["longrope2_training_strategy"]["inference_embedding_mode"],
                "mixed",
            )
            self.assertIsNotNone(checkpoint["model_config"]["longrope2_long_factors"])
            self.assertIsNotNone(checkpoint["model_config"]["longrope2_factor_max_sequence_length"])
            self.assertEqual(
                checkpoint["longrope2_training_strategy"]["factor_max_sequence_length"],
                checkpoint["model_config"]["longrope2_factor_max_sequence_length"],
            )

            metrics_path = Path(temp_dir) / "logs" / f"{checkpoint['run_id']}.training_metrics.jsonl"
            self.assertTrue(metrics_path.exists())
            metric_records = [
                json.loads(line)
                for line in metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            train_records = [record for record in metric_records if record["phase"] == "train"]
            eval_records = [record for record in metric_records if record["phase"] == "eval"]
            self.assertTrue(train_records)
            self.assertTrue(eval_records)
            self.assertIn("loss", train_records[0])
            self.assertIn("lr", train_records[0])
            self.assertIn("tokens_per_sec", train_records[0])
            self.assertIn("grad_norm", train_records[0])
            self.assertIn("eval_loss", eval_records[0])
            self.assertIn("eval_ppl", eval_records[0])
            self.assertTrue(inference_weight_path.exists())

    def test_train_prefers_tensorboard_when_available(self):
        model = self._build_model()
        train_dataset = build_chat_samples()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_root = Path(temp_dir) / "checkpoints" / "latest"
            train(
                model=model,
                dataset=train_dataset,
                tokenizer=self.tokenizer,
                save_path=str(checkpoint_root),
                manifest_path=Path("data/manifests/chat_sft.json"),
                training_profile=TensorBoardTrainingProfile,
            )

            checkpoint = load_checkpoint(checkpoint_root)
            tensorboard_dir = Path(temp_dir) / "logs" / "tensorboard" / checkpoint["run_id"]
            self.assertTrue(tensorboard_dir.exists())
            self.assertTrue(any(tensorboard_dir.iterdir()))

    def test_resume_run_matches_full_run_with_fixed_seed(self):
        dataset = build_chat_samples()

        with tempfile.TemporaryDirectory() as full_run_dir, tempfile.TemporaryDirectory() as split_run_dir:
            full_checkpoint_root = Path(full_run_dir) / "checkpoints" / "latest"
            split_checkpoint_root = Path(split_run_dir) / "checkpoints" / "latest"
            epoch_one_checkpoint_root = Path(full_run_dir) / "checkpoints" / "epoch_1"

            full_model = self._build_model()
            train(
                model=full_model,
                dataset=dataset,
                tokenizer=self.tokenizer,
                save_path=str(full_checkpoint_root),
                manifest_path=Path("data/manifests/chat_sft.json"),
                training_profile=TwoEpochTrainingProfileWithKeyCheckpoint,
            )
            full_checkpoint = load_checkpoint(full_checkpoint_root)

            resumed_model = self._build_model()
            train(
                model=resumed_model,
                dataset=dataset,
                tokenizer=self.tokenizer,
                save_path=str(split_checkpoint_root),
                manifest_path=Path("data/manifests/chat_sft.json"),
                training_profile=TwoEpochTrainingProfile,
                resume_checkpoint_path=str(epoch_one_checkpoint_root),
            )
            resumed_checkpoint = load_checkpoint(split_checkpoint_root)

            self.assertEqual(full_checkpoint["global_step"], resumed_checkpoint["global_step"])
            self.assertEqual(full_checkpoint["optimizer_step"], resumed_checkpoint["optimizer_step"])
            self.assertAlmostEqual(
                full_checkpoint["current_learning_rate"],
                resumed_checkpoint["current_learning_rate"],
                places=12,
            )

            for parameter_name, full_tensor in full_checkpoint["model_state_dict"].items():
                resumed_tensor = resumed_checkpoint["model_state_dict"][parameter_name]
                self.assertTrue(
                    torch.allclose(full_tensor, resumed_tensor, atol=0.0, rtol=0.0),
                    msg=f"参数不一致: {parameter_name}",
                )


if __name__ == "__main__":
    unittest.main()
