"""LPT v2 最小训练 smoke。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from lpt_config import TrainingSmokeConfig
from lpt_eval import build_lpt_v2_profile_config
from lpt_eval.utils import build_deterministic_input, resolve_eval_device, resolve_eval_dtype, set_eval_seed
from lpt_model import LPTV2, save_lpt_v2_checkpoint


DEFAULT_CONFIG = TrainingSmokeConfig()


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 最小训练 smoke。")
    parser.add_argument("--profile", default=DEFAULT_CONFIG.profile, help="运行 profile。")
    parser.add_argument("--preset", default=DEFAULT_CONFIG.preset, help="模型规格 preset。")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocabulary_size, help="词表大小。")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size, help="batch size。")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.sequence_length, help="训练序列长度。")
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.steps, help="训练步数。")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG.learning_rate, help="AdamW 学习率。")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG.dtype, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="随机种子。")
    parser.add_argument("--save-checkpoint", help="训练后保存 checkpoint 路径。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_eval_seed(args.seed)
    device = resolve_eval_device(args.device)
    dtype = resolve_eval_dtype(args.dtype, device=device)
    config = build_lpt_v2_profile_config(args.profile, preset=args.preset)
    model = LPTV2(args.vocab_size, config).to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    last_loss = None
    for step in range(int(args.steps)):
        input_ids = build_deterministic_input(
            args.vocab_size,
            args.batch_size,
            args.seq_len,
            offset=1 + step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _states = model(input_ids, request_id=f"train-smoke-{step}")
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, args.vocab_size), input_ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        print(f"step={step} loss={last_loss:.6f}")

    if args.save_checkpoint:
        save_lpt_v2_checkpoint(
            model,
            args.save_checkpoint,
            extra_metadata={"source": "tools/train_lpt_v2_smoke.py", "steps": int(args.steps), "last_loss": last_loss},
        )
        print(f"checkpoint_saved={args.save_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
