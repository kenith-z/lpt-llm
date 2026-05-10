import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_runtime import resolve_attention_backend


class TestAttentionBackend(unittest.TestCase):
    def test_auto_backend_uses_sdpa_as_current_default(self):
        decision = resolve_attention_backend(
            "auto",
            available_backends=("sdpa",),
            dtype=torch.float32,
            platform="windows",
        )

        self.assertEqual(decision.selected_backend, "sdpa")
        self.assertIsNone(decision.fallback_reason)
        self.assertEqual(decision.to_log_dict()["selected_backend"], "sdpa")
        self.assertEqual(
            [attempt["backend"] for attempt in decision.to_log_dict()["attempted_backends"]],
            ["sdpa"],
        )

    def test_fixed_backend_does_not_silently_fallback(self):
        with self.assertRaises(ValueError):
            resolve_attention_backend(
                "flash_attention_3",
                available_backends=("sdpa",),
                dtype=torch.float16,
                platform="linux",
            )

    def test_required_capability_filters_backend(self):
        decision = resolve_attention_backend(
            "auto",
            priority=("flash_attention_3", "flash_attention_2", "sdpa"),
            available_backends=("flash_attention_3", "flash_attention_2", "sdpa"),
            required_capabilities=("paged_kv",),
            dtype=torch.float16,
            platform="linux",
        )

        self.assertEqual(decision.selected_backend, "flash_attention_3")


if __name__ == "__main__":
    unittest.main()
