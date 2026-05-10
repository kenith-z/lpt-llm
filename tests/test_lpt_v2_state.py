import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lpt_model import (
    AttentionLayerState,
    LayerStateV2,
    MoELayerState,
    PagedKVReference,
    RetNetAssistState,
    xLSTMMemoryState,
)


class TestLPTV2State(unittest.TestCase):
    def build_layer_state(self):
        paged_ref = PagedKVReference(
            request_id="req-1",
            layer_index=3,
            page_ids=(10, 11, 12),
            token_count=768,
            window_token_count=768,
        )
        return LayerStateV2(
            attention=AttentionLayerState(
                request_id="req-1",
                layer_index=3,
                paged_kv_ref=paged_ref,
            ),
            retnet_assist=RetNetAssistState(
                request_id="req-1",
                layer_index=3,
                token_count=768,
            ),
            moe=MoELayerState(
                request_id="req-1",
                layer_index=3,
                expert_token_counts=(128, 128),
            ),
            xlstm_memory=xLSTMMemoryState(
                request_id="req-1",
                layer_index=3,
                token_count=768,
            ),
        )

    def test_paged_kv_trim_does_not_release_assist_states(self):
        layer_state = self.build_layer_state()

        trimmed_state = layer_state.trim_paged_kv(
            kept_page_ids=(12,),
            token_count=256,
            window_token_count=256,
        )

        self.assertEqual(trimmed_state.attention.paged_kv_ref.page_ids, (12,))
        self.assertEqual(trimmed_state.attention.paged_kv_ref.window_token_count, 256)
        self.assertIs(trimmed_state.retnet_assist, layer_state.retnet_assist)
        self.assertIs(trimmed_state.xlstm_memory, layer_state.xlstm_memory)
        self.assertFalse(trimmed_state.retnet_assist.release_metadata.released)
        self.assertFalse(trimmed_state.xlstm_memory.release_metadata.released)

    def test_release_assist_states_keeps_attention_lifecycle_separate(self):
        layer_state = self.build_layer_state()

        released_state = layer_state.release_assist_states("request_finished", token_count=800)

        self.assertIs(released_state.attention, layer_state.attention)
        self.assertTrue(released_state.retnet_assist.release_metadata.released)
        self.assertTrue(released_state.xlstm_memory.release_metadata.released)
        self.assertEqual(
            released_state.retnet_assist.release_metadata.release_reason,
            "request_finished",
        )

    def test_layer_state_v2_rejects_cross_request_state_mix(self):
        with self.assertRaises(ValueError):
            LayerStateV2(
                attention=AttentionLayerState(
                    request_id="req-1",
                    layer_index=0,
                ),
                retnet_assist=RetNetAssistState(
                    request_id="req-2",
                    layer_index=0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
