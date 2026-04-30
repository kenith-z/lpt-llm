"""真实 GPU 训练吞吐基准：对比 sequence packing 开/关。"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import GlobalConfig
from lpt_model import LPT
from lpt_training.train import (
    _build_dataloader,
    _build_optimizer,
    _build_scheduler,
    _forward_batch,
    _unpack_training_batch,
    configure_training_runtime,
)
from lpt_workflows.common import (
    TOKENIZER_PATH,
    build_local_tokenizer,
    build_tokenizer_metadata,
    load_dataset_from_manifest,
)


@dataclass(frozen=True)
class BenchmarkResult:
    sequence_packing_enabled: bool
    warmup_steps: int
    measured_steps: int
    batch_size: int
    raw_samples: int
    packed_rows: int
    active_tokens: int
    padded_tokens: int
    token_utilization: float
    wall_clock_seconds: float
    avg_step_ms: float
    active_tokens_per_sec: float
    padded_tokens_per_sec: float
    raw_samples_per_sec: float
    peak_memory_allocated_gb: float
    peak_memory_reserved_gb: float
    device_name: str
    manifest_path: str

    def to_dict(self):
        return {
            "sequence_packing_enabled": self.sequence_packing_enabled,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "batch_size": self.batch_size,
            "raw_samples": self.raw_samples,
            "packed_rows": self.packed_rows,
            "active_tokens": self.active_tokens,
            "padded_tokens": self.padded_tokens,
            "token_utilization": round(self.token_utilization, 6),
            "wall_clock_seconds": round(self.wall_clock_seconds, 6),
            "avg_step_ms": round(self.avg_step_ms, 3),
            "active_tokens_per_sec": round(self.active_tokens_per_sec, 3),
            "padded_tokens_per_sec": round(self.padded_tokens_per_sec, 3),
            "raw_samples_per_sec": round(self.raw_samples_per_sec, 3),
            "peak_memory_allocated_gb": round(self.peak_memory_allocated_gb, 3),
            "peak_memory_reserved_gb": round(self.peak_memory_reserved_gb, 3),
            "device_name": self.device_name,
            "manifest_path": self.manifest_path,
        }


def _parse_expected_types(kind: str):
    if kind == "text":
        return {"text"}
    if kind == "chat":
        return {"chat"}
    raise ValueError(f"未支持的 manifest kind: {kind}")


def _set_cuda_device():
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA，无法执行真实 GPU 基准。")
    GlobalConfig.device = torch.device("cuda")


def _move_batch_to_device(batch):
    input_ids, labels, attention_mask, position_ids, segment_ids, sample_count = _unpack_training_batch(batch)
    input_ids = input_ids.to(GlobalConfig.device, non_blocking=True)
    labels = labels.to(GlobalConfig.device, non_blocking=True)
    attention_mask = attention_mask.to(GlobalConfig.device, non_blocking=True)
    if position_ids is not None:
        position_ids = position_ids.to(GlobalConfig.device, non_blocking=True)
    if segment_ids is not None:
        segment_ids = segment_ids.to(GlobalConfig.device, non_blocking=True)
    return input_ids, labels, attention_mask, position_ids, segment_ids, sample_count


def _run_single_benchmark(
    *,
    dataset,
    tokenizer,
    batch_size,
    warmup_steps,
    measured_steps,
    learning_rate,
    weight_decay,
    seed,
    sequence_packing_enabled,
):
    configure_training_runtime(seed=seed, deterministic_algorithms=False)
    model = LPT(vocabulary_size=len(tokenizer))
    model.to(GlobalConfig.device)
    model.train()

    dataloader = _build_dataloader(
        dataset,
        tokenizer,
        batch_size,
        shuffle=True,
        pack_sequences=sequence_packing_enabled,
    )
    optimizer, _ = _build_optimizer(model, learning_rate, weight_decay)
    scheduler = _build_scheduler(
        optimizer,
        num_batches=warmup_steps + measured_steps,
        total_epochs=1,
        warmup_ratio=0.0,
        gradient_accumulation_steps=1,
    )
    optimizer.zero_grad(set_to_none=True)

    batch_iterator = itertools.cycle(dataloader)

    def run_step(batch):
        input_ids, labels, attention_mask, position_ids, segment_ids, sample_count = _move_batch_to_device(batch)
        loss, _ = _forward_batch(
            model,
            input_ids,
            labels,
            attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        packed_rows = int(input_ids.size(0))
        active_tokens = int(attention_mask.sum().item())
        padded_tokens = int(input_ids.numel())
        return {
            "sample_count": int(sample_count),
            "packed_rows": packed_rows,
            "active_tokens": active_tokens,
            "padded_tokens": padded_tokens,
        }

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for _ in range(warmup_steps):
        run_step(next(batch_iterator))
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    measured_sample_count = 0
    measured_packed_rows = 0
    measured_active_tokens = 0
    measured_padded_tokens = 0

    started_at = time.perf_counter()
    for _ in range(measured_steps):
        step_stats = run_step(next(batch_iterator))
        measured_sample_count += step_stats["sample_count"]
        measured_packed_rows += step_stats["packed_rows"]
        measured_active_tokens += step_stats["active_tokens"]
        measured_padded_tokens += step_stats["padded_tokens"]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    utilization = measured_active_tokens / max(1, measured_padded_tokens)

    return BenchmarkResult(
        sequence_packing_enabled=sequence_packing_enabled,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        batch_size=batch_size,
        raw_samples=measured_sample_count,
        packed_rows=measured_packed_rows,
        active_tokens=measured_active_tokens,
        padded_tokens=measured_padded_tokens,
        token_utilization=utilization,
        wall_clock_seconds=elapsed,
        avg_step_ms=elapsed * 1000.0 / max(1, measured_steps),
        active_tokens_per_sec=measured_active_tokens / max(elapsed, 1e-9),
        padded_tokens_per_sec=measured_padded_tokens / max(elapsed, 1e-9),
        raw_samples_per_sec=measured_sample_count / max(elapsed, 1e-9),
        peak_memory_allocated_gb=peak_allocated,
        peak_memory_reserved_gb=peak_reserved,
        device_name=torch.cuda.get_device_name(0),
        manifest_path="",
    )


def build_argument_parser():
    parser = ArgumentParser(description="对比 sequence packing 开/关的真实 GPU 训练吞吐。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/text_pretrain.json"),
        help="基准使用的 manifest 路径。",
    )
    parser.add_argument(
        "--manifest-kind",
        choices=("text", "chat"),
        default="text",
        help="manifest 对应的样本类型。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="每步送入 dataloader 的原始样本数。",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="预热步数，不计入最终统计。",
    )
    parser.add_argument(
        "--measured-steps",
        type=int,
        default=60,
        help="计入统计的训练步数。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="基准中的学习率。",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="基准中的 weight decay。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="随机种子。",
    )
    parser.add_argument(
        "--train-max-sequence-length",
        type=int,
        default=None,
        help="可选：仅基准时覆盖 GlobalConfig.train_max_sequence_length。",
    )
    parser.add_argument(
        "--packing-mode",
        choices=("both", "off", "on"),
        default="both",
        help="选择基准模式：同时测试开关，或只测试其中一种。",
    )
    parser.add_argument(
        "--train-rope-cache-max-sequence-length",
        type=int,
        default=None,
        help="可选：仅基准时覆盖 GlobalConfig.train_rope_cache_max_sequence_length。",
    )
    parser.add_argument(
        "--inference-rope-cache-max-sequence-length",
        type=int,
        default=None,
        help="可选：仅基准时覆盖 GlobalConfig.inference_rope_cache_max_sequence_length。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选：把结果写入 JSON 文件。",
    )
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    _set_cuda_device()
    if args.train_max_sequence_length is not None:
        GlobalConfig.train_max_sequence_length = int(args.train_max_sequence_length)
        if args.train_rope_cache_max_sequence_length is None:
            GlobalConfig.train_rope_cache_max_sequence_length = int(args.train_max_sequence_length)
    if args.train_rope_cache_max_sequence_length is not None:
        GlobalConfig.train_rope_cache_max_sequence_length = int(args.train_rope_cache_max_sequence_length)
    if args.inference_rope_cache_max_sequence_length is not None:
        GlobalConfig.inference_rope_cache_max_sequence_length = int(
            args.inference_rope_cache_max_sequence_length
        )
    expected_types = _parse_expected_types(args.manifest_kind)
    tokenizer = build_local_tokenizer()
    dataset = load_dataset_from_manifest(
        args.manifest,
        expected_types=expected_types,
        seed=args.seed,
    )

    results = []
    if args.packing_mode == "both":
        packing_modes = (False, True)
    elif args.packing_mode == "off":
        packing_modes = (False,)
    else:
        packing_modes = (True,)

    for sequence_packing_enabled in packing_modes:
        result = _run_single_benchmark(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            sequence_packing_enabled=sequence_packing_enabled,
        )
        results.append(
            BenchmarkResult(
                **{
                    **result.__dict__,
                    "manifest_path": str(args.manifest),
                }
            )
        )

    payload = {
        "config": {
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "packing_mode": args.packing_mode,
            "train_max_sequence_length": int(GlobalConfig.train_max_sequence_length),
            "train_rope_cache_max_sequence_length": int(GlobalConfig.train_rope_cache_max_sequence_length),
            "inference_rope_cache_max_sequence_length": int(GlobalConfig.inference_rope_cache_max_sequence_length),
        },
        "tokenizer": build_tokenizer_metadata(tokenizer, TOKENIZER_PATH),
        "benchmark": [result.to_dict() for result in results],
    }
    if len(results) == 2:
        plain_result, packed_result = results
        speedup = packed_result.active_tokens_per_sec / max(plain_result.active_tokens_per_sec, 1e-9)
        utilization_gain = packed_result.token_utilization - plain_result.token_utilization
        payload["comparison"] = {
            "active_tokens_per_sec_speedup": round(speedup, 4),
            "token_utilization_gain": round(utilization_gain, 4),
            "wall_clock_ratio": round(
                plain_result.wall_clock_seconds / max(packed_result.wall_clock_seconds, 1e-9),
                4,
            ),
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
