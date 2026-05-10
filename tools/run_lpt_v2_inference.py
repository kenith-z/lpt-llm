"""LPT v2 token-id 贪心推理。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from lpt_config import InferenceSmokeConfig
from lpt_eval import build_lpt_v2_profile_config
from lpt_eval.utils import resolve_eval_device, resolve_eval_dtype, set_eval_seed
from lpt_model import LPTV2, load_lpt_v2_checkpoint


DEFAULT_CONFIG = InferenceSmokeConfig()


def _parse_prompt_ids(raw_prompt_ids):
    values = [value.strip() for value in str(raw_prompt_ids).split(",") if value.strip()]
    if not values:
        raise ValueError("--prompt-ids 不能为空。")
    return [int(value) for value in values]


def build_parser():
    parser = argparse.ArgumentParser(description="运行 LPT v2 token-id 贪心推理。")
    parser.add_argument("--checkpoint", help="可选 LPT v2 checkpoint。")
    parser.add_argument("--profile", default=DEFAULT_CONFIG.profile, help="无 checkpoint 时使用的运行 profile。")
    parser.add_argument("--preset", default=DEFAULT_CONFIG.preset, help="无 checkpoint 时使用的模型规格 preset。")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG.vocabulary_size, help="词表大小。")
    parser.add_argument("--prompt-ids", default=DEFAULT_CONFIG.prompt_ids, help="逗号分隔 token id。")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_CONFIG.max_new_tokens, help="生成 token 数。")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device, help="auto/cpu/cuda:0。")
    parser.add_argument("--dtype", default=DEFAULT_CONFIG.dtype, help="auto/fp32/fp16/bf16。")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="随机种子。")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_eval_seed(args.seed)
    device = resolve_eval_device(args.device)
    dtype = resolve_eval_dtype(args.dtype, device=device)
    if args.checkpoint:
        loaded = load_lpt_v2_checkpoint(args.checkpoint, map_location="cpu", strict=True)
        model = loaded.model
        vocab_size = loaded.model.vocabulary_size
    else:
        config = build_lpt_v2_profile_config(args.profile, preset=args.preset)
        model = LPTV2(args.vocab_size, config)
        vocab_size = args.vocab_size
    model.to(device=device, dtype=dtype)
    model.eval()

    generated_ids = _parse_prompt_ids(args.prompt_ids)
    with torch.no_grad():
        input_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
        logits, states = model.prefill(input_ids, request_id="inference")
        for step in range(int(args.max_new_tokens)):
            next_id = int(torch.argmax(logits[:, -1].float(), dim=-1).item()) % vocab_size
            generated_ids.append(next_id)
            next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
            full_mask = torch.ones(1, len(generated_ids), dtype=torch.long, device=device)
            logits, states = model.decode(
                next_input,
                attention_mask=full_mask,
                layer_states=states,
                request_id="inference",
            )
    print(",".join(str(value) for value in generated_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
