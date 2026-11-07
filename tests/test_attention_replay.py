"""Pure live-slot/mixed-query contracts, not GPU replay validation."""
import unittest

from experiments.attention_replay import live_page_plan, query_plan


class AttentionReplayPlanTests(unittest.TestCase):
    def test_mixed_prefill_target_uses_packed_offset_not_request_index(self):
        offset, counts = query_plan([0, 1, 513, 514, 1026, 1541], [513,512,513,512,515], 2, 1541)
        self.assertEqual(offset,513)
        self.assertEqual(counts,[1,512,1,512,515])
        with self.assertRaises(ValueError):
            query_plan([0,1,513],[513,512],1,513)

    def test_live_page_tail_is_explicit_and_unreachable_padding_excluded(self):
        used = live_page_plan([[4,8],[4]], [17,16], 16, 10)
        self.assertEqual(used,{4:16,8:1})
        with self.assertRaises(ValueError):
            live_page_plan([[4]], [17], 16, 10)
        with self.assertRaises(ValueError):
            live_page_plan([[10]], [1], 16, 10)
