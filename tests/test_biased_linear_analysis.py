"""Synthetic classification tests; not GPU evidence."""
import unittest

from experiments.biased_linear_analysis import METHODS, layout_effects


class LayoutEffectsTests(unittest.TestCase):
    def test_separates_row_count_and_offset(self):
        values = {(8,2): [1,2], (8,0): [1,2], (1,0): [1,3]}
        replays = [{"rows": rows, "offset": offset, "method": method, "values": vector}
                   for (rows,offset),vector in values.items() for method in METHODS]
        for result in layout_effects(replays,8,2,1,0).values():
            self.assertEqual(result["offset_only"]["changed_elements"], 0)
            self.assertEqual(result["row_count_only"]["changed_elements"], 1)
        with self.assertRaises(ValueError):
            layout_effects(replays[:-1],8,2,1,0)
        with self.assertRaises(ValueError):
            layout_effects(replays+replays[:1],8,2,1,0)
