"""Replay-layout guards only; no GPU numerical validation."""
import unittest

from experiments.unbiased_linear_microprobe import replay_layouts


class UnbiasedReplayLayoutTests(unittest.TestCase):
    def test_captured_offsets_and_shape_only_comparisons_preserved(self):
        layouts = replay_layouts([{'M':8,'target_offset':2},{'M':1,'target_offset':0}])
        self.assertIn((8,2),layouts)
        self.assertIn((8,0),layouts)
        self.assertIn((1,0),layouts)
        self.assertEqual({m for m,_ in layouts},{1,8,16,128,256,2048})
        self.assertEqual(len(layouts),len(set(layouts)))

    def test_invalid_observed_offset_rejected(self):
        with self.assertRaises(ValueError):
            replay_layouts([{'M':8,'target_offset':8}])
