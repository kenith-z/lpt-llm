"""LPT v2 sequence packing 训练吞吐与显存基准。"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_config import (
    CHAT_SFT_MANIFEST_PATH,
    DEFAULT_MODEL_SIZE_PRESET,
    DEFAULT_PROFILE,
    GlobalConfig,
    ChatSFTTrainingConfig,
    build_lpt_v2_profile_config,
)
from lpt_eval.utils import write_json_report, write_text_report
from lpt_model import LPTV2, load_lpt_v2_checkpoint
from lpt_training.train import (
    TrainingRunConfig,
    _autocast_enabled,
    _batch_iterator,
    _build_batch_tensors,
    _build_optimizer,
    _collect_cuda_memory_metrics,
    _compute_lm_loss,
    _forward_batch,
    _move_batch,
    _reset_cuda_peak_memory_stats,
    configure_training_runtime,
)
from lpt_workflows.common import (
    build_local_tokenizer,
    load_dataset_from_manifest,
    resolve_checkpoint_file,
    resolve_torch_device,
    resolve_torch_dtype,
)


DEFAULT_RECIPE = ChatSFTTrainingConfig()


def _expected_types(manifest_kind):
    if manifest_kind == "chat":
        return {"chat"}
    if manifest_kind == "text":
        return {"text"}
    if manifest_kind == "mixed":
        return {"chat", "text"}
    raise ValueError(f"不支持的 manifest_kind: {manifest_kind}")


def _collect_batches(dataset, *, batch_size, step_count):
    batches = list(
        _batch_iterator(
            dataset,
            batch_size=int(batch_size),
            epochs=1,
            max_steps=int(step_count),
        )
    )
    if not batches:
        raise ValueError("manifest 未产生可用于 benchmark 的 batch。")
    return batches


def _resolve_benchmark_dtype(raw_dtype, *, device):
    dtype = resolve_torch_dtype(raw_dtype, device=device)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU sequence packing benchmark 仅支持 fp32；请使用 --dtype fp32。")
    return dtype


def _build_model(args, tokenizer, *, device, dtype):
    GlobalConfig.device = device
    GlobalConfig.parameter_dtype = dtype
    GlobalConfig.autocast_dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.float32
    if args.checkpoint is not None:
        loaded = load_lpt_v2_checkpoint(resolve_checkpoint_file(args.checkpoint), map_location="cpu", strict=True)
        model = loaded.model
    else:
        config = build_lpt_v2_profile_config(args.profile, preset=args.preset)
        model = LPTV2(len(tokenizer), config)
    return model.to(device=device, dtype=dtype)


def _build_training_config(args, *, mode):
    return TrainingRunConfig(
        training_stage=f"sequence_packing_benchmark_{mode}",
        artifact_dir=Path(".tmp_sequence_packing_benchmark") / mode,
        checkpoint_dir=Path(".tmp_sequence_packing_benchmark") / mode / "checkpoints" / "latest",
        inference_weight_path=Path(".tmp_sequence_packing_benchmark") / mode / "weights" / "model_weights.pth",
        save_inference_weights=False,
        batch_size=args.batch_size,
        epochs=1,
        max_steps=args.warmup_steps + args.measured_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=1,
        max_sequence_length=args.train_max_sequence_length,
        sequence_packing=(mode == "on"),
        seed=args.seed,
        deterministic_algorithms=False,
        save_interval=0,
        latest_save_interval=0,
        save_best_checkpoint=False,
        save_optimizer=False,
        save_scheduler=False,
        tensorboard_enabled=False,
    )


def _synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_mode(args, tokenizer, batches, *, mode, device, dtype):
    configure_training_runtime(args.seed, deterministic=False)
    model = _build_model(args, tokenizer, device=device, dtype=dtype).train()
    optimizer, _summary = _build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config = _build_training_config(args, mode=mode)
    metrics = {
        "mode": mode,
        "sequence_packing": mode == "on",
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": 0,
        "raw_samples": 0,
        "sample_count": 0,
        "batch_rows": 0,
        "active_tokens": 0,
        "padded_tokens": 0,
        "target_tokens": 0,
        "loss_sum": 0.0,
    }
    measured_started = False
    start_time = None
    for step_index, samples in enumerate(batches):
        is_measured = step_index >= int(args.warmup_steps)
        if is_measured and not measured_started:
            _synchronize_if_needed(device)
            _reset_cuda_peak_memory_stats(device)
            start_time = perf_counter()
            measured_started = True

        batch = _build_batch_tensors(samples, tokenizer, config)
        if is_measured:
            metrics["raw_samples"] += len(samples)
            metrics["sample_count"] += int(batch["sample_count"])
            metrics["batch_rows"] += int(batch["input_ids"].size(0))
            metrics["active_tokens"] += int(batch["attention_mask"].sum().item())
            metrics["padded_tokens"] += int(batch["input_ids"].numel())

        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=GlobalConfig.autocast_dtype,
            enabled=_autocast_enabled(device),
        ):
            logits, _states = _forward_batch(model, batch)
            loss, valid_targets = _compute_lm_loss(logits, batch["labels"])
        if loss is None:
            continue
        loss.backward()
        optimizer.step()

        if is_measured:
            metrics["measured_steps"] += 1
            metrics["target_tokens"] += int(valid_targets)
            metrics["loss_sum"] += float(loss.detach().cpu())

    if not measured_started or metrics["measured_steps"] <= 0:
        raise ValueError("没有完成任何 measured step；请增加数据量或降低 --warmup-steps。")
    _synchronize_if_needed(device)
    wall_seconds = max(perf_counter() - start_time, 1e-9)
    metrics["wall_seconds"] = wall_seconds
    metrics["avg_step_ms"] = wall_seconds * 1000.0 / max(1, metrics["measured_steps"])
    metrics["avg_loss"] = metrics["loss_sum"] / max(1, metrics["measured_steps"])
    metrics["token_utilization"] = metrics["active_tokens"] / max(1, metrics["padded_tokens"])
    metrics["active_tokens_per_sec"] = metrics["active_tokens"] / wall_seconds
    metrics["padded_tokens_per_sec"] = metrics["padded_tokens"] / wall_seconds
    metrics["raw_samples_per_sec"] = metrics["raw_samples"] / wall_seconds
    metrics["avg_rows_per_step"] = metrics["batch_rows"] / max(1, metrics["measured_steps"])
    metrics["avg_samples_per_step"] = metrics["sample_count"] / max(1, metrics["measured_steps"])
    metrics["avg_active_tokens_per_step"] = metrics["active_tokens"] / max(1, metrics["measured_steps"])
    metrics.update(_collect_cuda_memory_metrics(device))
    return metrics


def _build_comparison(results):
    by_mode = {result["mode"]: result for result in results}
    if "on" not in by_mode or "off" not in by_mode:
        return {}
    on = by_mode["on"]
    off = by_mode["off"]
    return {
        "active_tokens_per_sec_speedup": on["active_tokens_per_sec"] / max(off["active_tokens_per_sec"], 1e-9),
        "avg_step_ms_ratio": on["avg_step_ms"] / max(off["avg_step_ms"], 1e-9),
        "padded_token_reduction": 1.0 - (on["padded_tokens"] / max(off["padded_tokens"], 1)),
        "token_utilization_delta": on["token_utilization"] - off["token_utilization"],
    }


def _format_float(value, digits=4):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _to_markdown(payload):
    lines = [
        "# LPT v2 Sequence Packing Benchmark",
        "",
        f"- manifest: `{payload['manifest']}`",
        f"- manifest_kind: `{payload['manifest_kind']}`",
        f"- device: `{payload['device']}`",
        f"- dtype: `{payload['dtype']}`",
        f"- batch_size: `{payload['batch_size']}`",
        f"- train_max_sequence_length: `{payload['train_max_sequence_length']}`",
        f"- checkpoint: `{payload.get('checkpoint')}`",
        "",
        "| mode | steps | active_tok/s | padded_tok/s | util | avg_step_ms | rows/step | samples/step | peak_mib |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        lines.append(
            "| {mode} | {steps} | {active} | {padded} | {util} | {step_ms} | {rows} | {samples} | {peak} |".format(
                mode=result["mode"],
                steps=result["measured_steps"],
                active=_format_float(result["active_tokens_per_sec"], 2),
                padded=_format_float(result["padded_tokens_per_sec"], 2),
                util=_format_float(result["token_utilization"]),
                step_ms=_format_float(result["avg_step_ms"], 2),
                rows=_format_float(result["avg_rows_per_step"], 2),
                samples=_format_float(result["avg_samples_per_step"], 2),
                peak=_format_float(result.get("cuda_peak_memory_allocated_mib"), 2),
            )
        )
    if payload["comparison"]:
        comparison = payload["comparison"]
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                f"- active_tokens_per_sec_speedup: `{_format_float(comparison['active_tokens_per_sec_speedup'])}`",
                f"- avg_step_ms_ratio: `{_format_float(comparison['avg_step_ms_ratio'])}`",
                f"- padded_token_reduction: `{_format_float(comparison['padded_token_reduction'])}`",
                f"- token_utilization_delta: `{_format_float(comparison['token_utilization_delta'])}`",
            ]
        )
    return "\n".join(lines) + "\n"


def run_benchmark(args):
    device = resolve_torch_device(args.device)
    dtype = _resolve_benchmark_dtype(args.dtype, device=device)
    GlobalConfig.device = device
    tokenizer = build_local_tokenizer()
    dataset = load_dataset_from_manifest(
        args.manifest,
        expected_types=_expected_types(args.manifest_kind),
        seed=args.seed,
    )
    total_steps = int(args.warmup_steps) + int(args.measured_steps)
    batches = _collect_batches(dataset, batch_size=args.batch_size, step_count=total_steps)
    modes = ("off", "on") if args.packing_mode == "both" else (args.packing_mode,)
    results = [
        _run_mode(args, tokenizer, batches, mode=mode, device=device, dtype=dtype)
        for mode in modes
    ]
    device_name = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "report_type": "lpt_v2_sequence_packing_benchmark",
        "manifest": str(args.manifest),
        "manifest_kind": args.manifest_kind,
        "profile": args.profile,
        "preset": args.preset,
        "checkpoint": None if args.checkpoint is None else str(resolve_checkpoint_file(args.checkpoint)),
        "device": str(device),
        "device_name": device_name,
        "dtype": str(dtype).removeprefix("torch."),
        "batch_size": int(args.batch_size),
        "train_max_sequence_length": args.train_max_sequence_length,
        "warmup_steps": int(args.warmup_steps),
        "measured_steps": int(args.measured_steps),
        "results": results,
        "comparison": _build_comparison(results),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark LPT v2 sequence packing 训练吞吐与显存。")
    parser.add_argument("--manifest", type=Path, default=CHAT_SFT_MANIFEST_PATH, help="训练 manifest。")
    parser.add_argument("--manifest-kind", choices=("chat", "text", "mixed"), default="chat", help="manifest 样本类型。")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="未指定 checkpoint 时使用的 v2 profile。")
    parser.add_argument("--preset", default=DEFAULT_MODEL_SIZE_PRESET, help="未指定 checkpoint 时使用的模型规格。")
    parser.add_argument("--checkpoint", type=Path, default=None, help="真实 v2 checkpoint；指定后使用 checkpoint 模型结构和权重。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RECIPE.batch_size, help="原始样本 batch size。")
    parser.add_argument("--train-max-sequence-length", type=int, default=DEFAULT_RECIPE.train_max_sequence_length, help="训练截断长度；为空时使用 GlobalConfig。")
    parser.add_argument("--warmup-steps", type=int, default=2, help="预热 step 数，不计入结果。")
    parser.add_argument("--measured-steps", type=int, default=10, help="计入 benchmark 的 step 数。")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_RECIPE.learning_rate, help="AdamW 学习率。")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_RECIPE.weight_decay, help="权重衰减。")
    parser.add_argument("--packing-mode", choices=("both", "on", "off"), default="both", help="测试 packing 开/关或两者。")
    parser.add_argument("--device", default=DEFAULT_RECIPE.device, help="auto/cpu/cuda/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_RECIPE.dtype, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=DEFAULT_RECIPE.random_seed, help="随机种子。")
    parser.add_argument("--output-json", help="JSON 报告输出路径。")
    parser.add_argument("--output-md", help="Markdown 报告输出路径。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = run_benchmark(args)
    if args.output_json:
        write_json_report(args.output_json, payload)
    if args.output_md:
        write_text_report(args.output_md, _to_markdown(payload))
    if not args.output_json and not args.output_md:
        print(_to_markdown(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
