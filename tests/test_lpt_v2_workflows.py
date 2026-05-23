import io
import importlib
import json
import re
import subprocess
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import (
    ChatLoRATrainingConfig,
    ChatSFTTrainingConfig,
    GlobalConfig,
    ModelConfig,
    TextPretrainingConfig,
)
from lpt_inference import InferenceSession, display_model_parameter_summary
from lpt_lora import (
    LoRAConfig,
    attach_lora_adapters,
    collect_lora_adapter_state,
    load_lora_adapter_config,
    load_lora_adapter_state,
    save_lora_adapter_state,
)
from lpt_model import LPTV2
from lpt_protocol import DS_BOS_TOKEN, DS_EOS_TOKEN, DS_PAD_TOKEN
from lpt_data import build_streaming_manifest_dataset
from lpt_training import (
    TrainingRunConfig,
    has_complete_training_state,
    resolve_latest_training_checkpoint,
    train,
)
from lpt_training.train import _compute_lm_loss, _save_checkpoint, _write_tensorboard_metrics
from lpt_workflows.chat_lora import build_parser as build_chat_lora_parser
from lpt_workflows.chat_sft import build_parser as build_chat_sft_parser
from lpt_workflows.text_pretrain import build_parser as build_text_pretrain_parser


class DummyTokenizer:
    def __init__(self):
        self.special_tokens = (DS_BOS_TOKEN, DS_PAD_TOKEN, DS_EOS_TOKEN)
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
        self.pad_token_id = self.token_to_id[DS_PAD_TOKEN]

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token)

    def __call__(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("测试 tokenizer 不支持 add_special_tokens=True。")
        pieces = []
        cursor = 0
        for match in self.pattern.finditer(text):
            if match.start() > cursor:
                pieces.extend(list(text[cursor:match.start()]))
            pieces.append(match.group(0))
            cursor = match.end()
        if cursor < len(text):
            pieces.extend(list(text[cursor:]))
        input_ids = []
        for piece in pieces:
            if piece not in self.token_to_id:
                self.token_to_id[piece] = self.next_id
                self.next_id += 1
            input_ids.append(self.token_to_id[piece])
        return {"input_ids": input_ids}


def build_tiny_config(**overrides):
    payload = {
        "num_layers": 2,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "layer_block_types": ("attention", "attention"),
        "attention_window_size": 8,
        "page_block_size": 4,
        "retnet_assist_layers": "all_layers",
        "retnet_state_dim": 4,
        "retnet_adapter_rank": 2,
        "moe_num_experts": 2,
        "moe_top_k": 1,
        "original_max_len": 8,
        "longrope2_target_length": 16,
    }
    payload.update(overrides)
    return ModelConfig.from_preset("lpt_v2_dev_tiny", **payload)


def build_workspace_tmp_dir(name):
    path = PROJECT_ROOT / ".tmp_unittest" / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TestLPTV2Workflows(unittest.TestCase):
    def test_stage_training_recipes_drive_parser_defaults(self):
        text_args = build_text_pretrain_parser().parse_args([])
        sft_args = build_chat_sft_parser().parse_args([])
        lora_args = build_chat_lora_parser().parse_args([])

        self.assertEqual(text_args.batch_size, TextPretrainingConfig().batch_size)
        self.assertEqual(text_args.learning_rate, TextPretrainingConfig().learning_rate)
        self.assertEqual(sft_args.batch_size, ChatSFTTrainingConfig().batch_size)
        self.assertEqual(sft_args.epochs, ChatSFTTrainingConfig().target_total_epochs)
        self.assertEqual(lora_args.batch_size, ChatLoRATrainingConfig().batch_size)
        self.assertEqual(lora_args.lora_rank, ChatLoRATrainingConfig().lora_rank)
        self.assertTrue(text_args.deterministic_algorithms)
        self.assertFalse(text_args.no_sequence_packing)
        self.assertIsNone(text_args.max_steps)
        self.assertIsNone(text_args.eval_max_batches)
        self.assertEqual(text_args.latest_save_interval, TextPretrainingConfig().latest_save_interval_steps)
        self.assertFalse(text_args.save_best_checkpoint)

    def test_chunked_lm_loss_matches_reference_cross_entropy(self):
        logits = torch.randn(2, 5, 17, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor(
            [
                [1, 2, -100, 4, 5],
                [6, 7, 8, -100, 10],
            ],
            dtype=torch.long,
        )

        chunked_loss, valid_targets = _compute_lm_loss(logits, labels, chunk_tokens=2)
        reference_loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().float().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        chunked_loss.backward()

        self.assertEqual(valid_targets, 6)
        self.assertTrue(torch.allclose(chunked_loss.detach(), reference_loss.detach(), atol=1e-6))
        self.assertIsNotNone(logits.grad)

    def test_training_loop_without_max_steps_runs_all_epoch_batches(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        tokenizer = DummyTokenizer()
        artifact_dir = build_workspace_tmp_dir("workflow_train_all_batches")
        dataset = [
            {"id": "text-001", "type": "text", "text": "知识点一"},
            {"id": "text-002", "type": "text", "text": "知识点二"},
            {"id": "text-003", "type": "text", "text": "知识点三"},
        ]

        trainer_state = train(
            model,
            tokenizer,
            dataset,
            config=TrainingRunConfig(
                training_stage="unit_text_pretrain_all_batches",
                artifact_dir=artifact_dir,
                checkpoint_dir=artifact_dir / "checkpoints" / "latest",
                inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
                batch_size=2,
                epochs=2,
                max_steps=None,
                warmup_ratio=0.0,
                max_sequence_length=16,
                seed=11,
                save_optimizer=False,
                save_scheduler=False,
                tensorboard_enabled=False,
            ),
        )

        self.assertEqual(trainer_state["global_step"], 4)
        self.assertIsNone(trainer_state["training_config"]["max_steps"])

    def test_root_entrypoints_expose_help(self):
        for script_name in ("main.py", "main-pretrain.py", "main-sft.py", "main-LoRA.py"):
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / script_name), "--help"],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_training_loop_saves_v2_checkpoint_and_inference_weight(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        tokenizer = DummyTokenizer()
        artifact_dir = build_workspace_tmp_dir("workflow_train")
        dataset = [
            {
                "id": "text-001",
                "type": "text",
                "text": "知识点",
            }
        ]

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            trainer_state = train(
                model,
                tokenizer,
                dataset,
                config=TrainingRunConfig(
                    training_stage="unit_text_pretrain",
                    artifact_dir=artifact_dir,
                    checkpoint_dir=artifact_dir / "checkpoints" / "latest",
                    inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
                    batch_size=1,
                    epochs=1,
                    max_steps=1,
                    log_interval=1,
                    warmup_ratio=0.1,
                    eval_batch_size=1,
                    eval_max_batches=2,
                    key_checkpoints=(1,),
                    max_sequence_length=16,
                    seed=7,
                    longrope2_window_lengths=(8, 16),
                    longrope2_window_weights=(1.0, 1.0),
                    tokenizer_metadata={"tokenizer_path": "dummy", "vocab_size": 128},
                ),
            )

        self.assertEqual(trainer_state["global_step"], 1)
        self.assertEqual(trainer_state["tokenizer_metadata"]["tokenizer_path"], "dummy")
        self.assertEqual(trainer_state["warmup_ratio"], 0.1)
        self.assertEqual(trainer_state["eval_max_batches"], 2)
        self.assertEqual(trainer_state["longrope2_training_strategy"]["window_lengths"], [8, 16])
        self.assertIn("optimizer_group_summary", trainer_state)
        self.assertTrue((artifact_dir / "checkpoints" / "latest" / "model.pt").exists())
        self.assertTrue((artifact_dir / "checkpoints" / "step_000001" / "model.pt").exists())
        self.assertTrue((artifact_dir / "weights" / "model_weights.pth").exists())
        self.assertTrue((artifact_dir / "config" / "model_config.json").exists())
        self.assertIn("Token吞吐(tokens_per_sec)", stdout.getvalue())
        self.assertIn("梯度范数(grad_norm)", stdout.getvalue())
        self.assertIn("序列长度(sequence_length)", stdout.getvalue())
        metric_line = (artifact_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("tokens_per_sec", metric_line)
        self.assertIn("grad_norm", metric_line)
        self.assertEqual(json.loads(metric_line)["sequence_length"], 4)

    def test_manifest_data_progress_resumes_new_entries_and_skips_weight_zero(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        tokenizer = DummyTokenizer()
        artifact_dir = build_workspace_tmp_dir("workflow_data_progress")
        dataset_dir = artifact_dir / "data"
        dataset_dir.mkdir()
        manifest_path = artifact_dir / "manifest.json"
        dataset_a = dataset_dir / "a.text.jsonl"
        dataset_b = dataset_dir / "b.text.jsonl"
        dataset_a.write_text(
            json.dumps({"id": "a-1", "type": "text", "text": "旧数据", "source": "a"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        dataset_b.write_text(
            json.dumps({"id": "b-1", "type": "text", "text": "新增数据", "source": "b"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {"datasets": [{"name": "old", "path": "data/a.text.jsonl", "weight": 1}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        initial_dataset = build_streaming_manifest_dataset(
            manifest_path,
            expected_types={"text"},
            shuffle_buffer_size=1,
            seed=3,
        )
        initial_state = train(
            model,
            tokenizer,
            initial_dataset,
            config=TrainingRunConfig(
                training_stage="unit_data_progress",
                artifact_dir=artifact_dir,
                checkpoint_dir=artifact_dir / "checkpoints" / "latest",
                inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
                save_inference_weights=False,
                batch_size=1,
                epochs=1,
                max_steps=None,
                latest_save_interval=0,
                warmup_ratio=0.0,
                max_sequence_length=16,
                seed=3,
                source_manifest=manifest_path,
                save_scheduler=False,
                tensorboard_enabled=False,
            ),
        )
        self.assertEqual(initial_state["global_step"], 1)

        manifest_path.write_text(
            json.dumps(
                {
                    "datasets": [
                        {"name": "old", "path": "data/a.text.jsonl", "weight": 0},
                        {"name": "new", "path": "data/b.text.jsonl", "weight": 1},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        resumed_dataset = build_streaming_manifest_dataset(
            manifest_path,
            expected_types={"text"},
            shuffle_buffer_size=1,
            seed=3,
        )
        resumed_state = train(
            model,
            tokenizer,
            resumed_dataset,
            config=TrainingRunConfig(
                training_stage="unit_data_progress",
                artifact_dir=artifact_dir,
                checkpoint_dir=artifact_dir / "checkpoints" / "latest",
                inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
                save_inference_weights=False,
                batch_size=1,
                epochs=1,
                max_steps=None,
                latest_save_interval=0,
                warmup_ratio=0.0,
                max_sequence_length=16,
                seed=3,
                resume_checkpoint=artifact_dir / "checkpoints" / "latest",
                source_manifest=manifest_path,
                save_scheduler=False,
                tensorboard_enabled=False,
            ),
        )

        progress_path = artifact_dir / "checkpoints" / "latest" / "data_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        entries_by_name = {entry["entry_name"]: entry for entry in progress["entries"]}
        self.assertEqual(resumed_state["global_step"], 2)
        self.assertFalse(Path(progress["manifest_path"]).is_absolute())
        self.assertEqual(progress["active_entry_count"], 1)
        self.assertFalse(entries_by_name["old"]["active"])
        self.assertEqual(entries_by_name["old"]["total_consumed_samples"], 1)
        self.assertFalse(Path(entries_by_name["new"]["path"]).is_absolute())
        self.assertFalse(Path(entries_by_name["new"]["entry_key"]).is_absolute())
        self.assertEqual(entries_by_name["new"]["completed_epochs"], 1)
        self.assertEqual(entries_by_name["new"]["total_consumed_samples"], 1)
        self.assertTrue((artifact_dir / "config" / "data_progress.json").exists())

    def test_training_loop_can_skip_inference_weight_export(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        tokenizer = DummyTokenizer()
        artifact_dir = build_workspace_tmp_dir("workflow_train_no_inference_weight")
        dataset = [{"id": "text-001", "type": "text", "text": "知识点"}]

        trainer_state = train(
            model,
            tokenizer,
            dataset,
            config=TrainingRunConfig(
                training_stage="unit_text_pretrain_no_inference_weight",
                artifact_dir=artifact_dir,
                checkpoint_dir=artifact_dir / "checkpoints" / "latest",
                inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
                save_inference_weights=False,
                batch_size=1,
                epochs=1,
                max_steps=1,
                latest_save_interval=0,
                warmup_ratio=0.0,
                max_sequence_length=16,
                seed=19,
                save_optimizer=False,
                save_scheduler=False,
                tensorboard_enabled=False,
            ),
        )

        self.assertEqual(trainer_state["global_step"], 1)
        self.assertFalse(trainer_state["save_inference_weights"])
        self.assertFalse((artifact_dir / "weights" / "model_weights.pth").exists())
        self.assertFalse((artifact_dir / "weights" / "model_checkpoint.pt").exists())
        self.assertTrue((artifact_dir / "checkpoints" / "latest" / "model.pt").exists())

    def test_training_loop_saves_single_best_loss_checkpoint(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        tokenizer = DummyTokenizer()
        artifact_dir = build_workspace_tmp_dir("workflow_train_best_loss")
        dataset = [
            {"id": "text-001", "type": "text", "text": "知识点一"},
            {"id": "text-002", "type": "text", "text": "知识点二"},
        ]

        trainer_state = train(
            model,
            tokenizer,
            dataset,
            config=TrainingRunConfig(
                training_stage="unit_text_pretrain_best_loss",
                artifact_dir=artifact_dir,
                checkpoint_dir=artifact_dir / "checkpoints" / "latest",
                inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
                batch_size=1,
                epochs=1,
                max_steps=2,
                log_interval=1,
                latest_save_interval=1,
                save_best_checkpoint=True,
                best_checkpoint_metric="loss",
                warmup_ratio=0.0,
                max_sequence_length=16,
                seed=17,
                save_optimizer=False,
                save_scheduler=False,
                tensorboard_enabled=False,
            ),
        )

        best_root = artifact_dir / "checkpoints" / "best_loss"
        self.assertTrue((best_root / "model.pt").exists())
        best_state = json.loads((best_root / "trainer_state.json").read_text(encoding="utf-8"))
        self.assertEqual(best_state["best_checkpoint"]["metric"], "loss")
        self.assertEqual(best_state["best_checkpoint"]["path"], str(best_root))
        self.assertEqual(trainer_state["best_checkpoint"]["path"], str(best_root))
        self.assertFalse((artifact_dir / "checkpoints" / "step_000001").exists())

    def test_checkpoint_publish_keeps_latest_when_staging_save_fails(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        artifact_dir = build_workspace_tmp_dir("workflow_checkpoint_failed_publish")
        config = TrainingRunConfig(
            training_stage="unit_checkpoint_failure",
            artifact_dir=artifact_dir,
            checkpoint_dir=artifact_dir / "checkpoints" / "latest",
            inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
            save_scheduler=False,
            tensorboard_enabled=False,
        )
        state = {
            "training_stage": "unit_checkpoint_failure",
            "run_id": "unit",
            "global_step": 1,
            "optimizer_step": 1,
            "lora_mode": False,
            "save_optimizer": True,
            "save_scheduler": False,
            "training_config": {"save_optimizer": True, "save_scheduler": False},
        }
        _save_checkpoint(model, optimizer, None, config, state, is_latest=True)

        train_module = importlib.import_module("lpt_training.train")
        original_save = train_module.save_lpt_v2_checkpoint
        try:
            def _fail_model_save(*_args, **_kwargs):
                raise RuntimeError("模拟 checkpoint 写入失败")

            train_module.save_lpt_v2_checkpoint = _fail_model_save
            failed_state = dict(state)
            failed_state.update({"global_step": 2, "optimizer_step": 2})
            with self.assertRaisesRegex(RuntimeError, "模拟 checkpoint 写入失败"):
                _save_checkpoint(model, optimizer, None, config, failed_state, is_latest=True)
        finally:
            train_module.save_lpt_v2_checkpoint = original_save

        latest_root = artifact_dir / "checkpoints" / "latest"
        latest_state = json.loads((latest_root / "trainer_state.json").read_text(encoding="utf-8"))
        self.assertEqual(latest_state["global_step"], 1)
        self.assertTrue(has_complete_training_state(latest_root, lora_mode=False))
        staging_roots = list((artifact_dir / "checkpoints").glob("latest.staging.*"))
        self.assertEqual(staging_roots, [])

    def test_resume_checkpoint_falls_back_to_previous_latest_when_latest_is_corrupt(self):
        GlobalConfig.parameter_dtype = torch.float32
        GlobalConfig.device = torch.device("cpu")
        model = LPTV2(128, build_tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        artifact_dir = build_workspace_tmp_dir("workflow_checkpoint_fallback")
        config = TrainingRunConfig(
            training_stage="unit_checkpoint_fallback",
            artifact_dir=artifact_dir,
            checkpoint_dir=artifact_dir / "checkpoints" / "latest",
            inference_weight_path=artifact_dir / "weights" / "model_weights.pth",
            save_scheduler=False,
            tensorboard_enabled=False,
        )
        state = {
            "training_stage": "unit_checkpoint_fallback",
            "run_id": "unit",
            "global_step": 1,
            "optimizer_step": 1,
            "lora_mode": False,
            "save_optimizer": True,
            "save_scheduler": False,
            "training_config": {"save_optimizer": True, "save_scheduler": False},
        }
        _save_checkpoint(model, optimizer, None, config, state, is_latest=True)
        next_state = dict(state)
        next_state.update({"global_step": 2, "optimizer_step": 2})
        _save_checkpoint(model, optimizer, None, config, next_state, is_latest=True)

        latest_root = artifact_dir / "checkpoints" / "latest"
        previous_root = artifact_dir / "checkpoints" / "latest_previous"
        (latest_root / "model.pt").write_text("broken checkpoint", encoding="utf-8")

        self.assertFalse(has_complete_training_state(latest_root, lora_mode=False))
        self.assertTrue(has_complete_training_state(previous_root, lora_mode=False))
        self.assertEqual(resolve_latest_training_checkpoint(latest_root, lora_mode=False), previous_root)

    def test_lora_adapter_roundtrip_uses_checkpoint_config(self):
        GlobalConfig.parameter_dtype = torch.float32
        base_model = LPTV2(128, build_tiny_config())
        lora_config = LoRAConfig(rank=2, alpha=4.0, dropout_p=0.0)
        attach_lora_adapters(base_model, lora_config)
        adapter_path = build_workspace_tmp_dir("workflow_lora") / "adapter_weights.pth"
        save_lora_adapter_state(base_model, adapter_path, config=lora_config)

        loaded_config = load_lora_adapter_config(adapter_path)
        restored_model = LPTV2(128, build_tiny_config())
        attach_lora_adapters(restored_model, loaded_config)
        load_lora_adapter_state(restored_model, adapter_path, strict=True)

        self.assertEqual(loaded_config.rank, 2)
        self.assertEqual(
            sorted(collect_lora_adapter_state(base_model)),
            sorted(collect_lora_adapter_state(restored_model)),
        )

    def test_inference_session_can_rebuild_current_context(self):
        GlobalConfig.parameter_dtype = torch.float32
        model = LPTV2(128, build_tiny_config())
        session = InferenceSession(model, request_id="unit-rebuild")

        session.prefill([1, 2, 3])
        session.append(4)
        logits = session.rebuild_on_switch()

        self.assertEqual(tuple(logits.shape), (1, 4, 128))
        self.assertEqual(session.export_state()["token_count"], 4)

    def test_tensorboard_metric_names_use_chinese_english_labels(self):
        class FakeWriter:
            def __init__(self):
                self.tags = []

            def add_scalar(self, tag, value, step):
                self.tags.append((tag, value, step))

        writer = FakeWriter()
        _write_tensorboard_metrics(
            writer,
            "train",
            {"global_step": 3, "loss": 1.25, "tokens_per_sec": 42.0, "stage": "unit"},
        )

        tags = [tag for tag, _value, _step in writer.tags]
        self.assertIn("训练(train)/损失(loss)", tags)
        self.assertIn("训练(train)/Token吞吐(tokens_per_sec)", tags)

    def test_tensorboard_write_failure_does_not_abort_training_metrics(self):
        class BrokenWriter:
            def add_scalar(self, *_args, **_kwargs):
                raise FileNotFoundError("tensorboard 目录不存在")

        _write_tensorboard_metrics(
            BrokenWriter(),
            "train",
            {"global_step": 3, "loss": 1.25, "tokens_per_sec": 42.0, "stage": "unit"},
        )

    def test_parameter_summary_uses_chinese_english_labels(self):
        model = LPTV2(128, build_tiny_config())
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            payload = display_model_parameter_summary(model)

        output = stdout.getvalue()
        self.assertIn("物理总参数(total_physical_params)", output)
        self.assertIn("每Token激活参数(active_params_per_token)", output)
        self.assertIn("路由参数(router_params)", output)
        self.assertIn("total_physical_params", payload)


if __name__ == "__main__":
    unittest.main()
