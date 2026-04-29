import json
import tempfile
import unittest
from pathlib import Path


from lpt_runtime import (
    DeviceInfo,
    ExecutionConfig,
    describe_execution_plan,
    parse_cuda_visible_devices,
    resolve_execution_plan,
)


def build_visible_devices(*memory_values):
    return tuple(
        DeviceInfo(
            logical_index=index,
            logical_name=f"cuda:{index}",
            visible_device=str(index + 2),
            name=f"Fake GPU {index}",
            total_memory_bytes=memory,
        )
        for index, memory in enumerate(memory_values)
    )


class TestRuntimeExecution(unittest.TestCase):
    def test_parse_cuda_visible_devices_distinguishes_unset_and_hidden(self):
        self.assertIsNone(parse_cuda_visible_devices(None))
        self.assertEqual(parse_cuda_visible_devices(""), ())
        self.assertEqual(parse_cuda_visible_devices("-1"), ())
        self.assertEqual(parse_cuda_visible_devices("2, 3"), ("2", "3"))

    def test_auto_model_parallel_uses_visible_logical_devices(self):
        devices = build_visible_devices(8 * 1024**3, 8 * 1024**3)
        plan = resolve_execution_plan(
            ExecutionConfig(mode="model_parallel", device_map="auto"),
            num_layers=4,
            visible_cuda_devices=devices,
        )

        self.assertEqual(plan.mode, "model_parallel")
        self.assertEqual(plan.primary_device, "cuda:0")
        self.assertEqual(plan.layer_devices, ("cuda:0", "cuda:0", "cuda:1", "cuda:1"))
        self.assertEqual(plan.visible_cuda_devices[0].visible_device, "2")
        self.assertIn("layers 0-1 -> cuda:0", describe_execution_plan(plan))

    def test_auto_execution_mode_keeps_single_device_when_only_one_cuda_is_visible(self):
        devices = build_visible_devices(8 * 1024**3)
        plan = resolve_execution_plan(
            ExecutionConfig(mode="auto", device_map="auto"),
            num_layers=4,
            visible_cuda_devices=devices,
        )

        self.assertEqual(plan.mode, "single")
        self.assertEqual(plan.primary_device, "cuda:0")

    def test_manual_device_map_file_is_validated_against_visible_devices(self):
        devices = build_visible_devices(8 * 1024**3, 8 * 1024**3)
        with tempfile.TemporaryDirectory() as temp_dir:
            map_path = Path(temp_dir) / "device_map.json"
            map_path.write_text(
                json.dumps({"layers": ["cuda:0", "cuda:1", "cuda:1"]}),
                encoding="utf-8",
            )

            plan = resolve_execution_plan(
                ExecutionConfig(mode="model_parallel", device_map=map_path),
                num_layers=3,
                visible_cuda_devices=devices,
            )

        self.assertEqual(plan.layer_devices, ("cuda:0", "cuda:1", "cuda:1"))

    def test_model_parallel_rejects_invisible_manual_device(self):
        devices = build_visible_devices(8 * 1024**3, 8 * 1024**3)
        with self.assertRaisesRegex(ValueError, "不可见"):
            resolve_execution_plan(
                ExecutionConfig(mode="model_parallel", device_map=["cuda:0", "cuda:2"]),
                num_layers=2,
                visible_cuda_devices=devices,
            )

    def test_model_parallel_requires_multiple_visible_cuda_devices(self):
        with self.assertRaisesRegex(ValueError, "至少需要 2 张"):
            resolve_execution_plan(
                ExecutionConfig(mode="model_parallel", device_map="auto"),
                num_layers=2,
                visible_cuda_devices=build_visible_devices(8 * 1024**3),
            )


if __name__ == "__main__":
    unittest.main()
