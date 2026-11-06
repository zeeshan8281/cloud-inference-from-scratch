"""Pure layout/statistics self-checks, not GPU replay validation."""
import unittest

from experiments.biased_linear_probe import layout_variants, vector_metrics


class BiasedProbeTests(unittest.TestCase):
    def test_separates_observed_shape_and_offset(self):
        self.assertEqual(layout_variants(2048, 767, 256, 255), [(2048, 767), (2048, 255), (256, 255)])
        self.assertEqual(layout_variants(8, 2, 1, 0), [(8, 2), (8, 0), (1, 0)])
        self.assertEqual(layout_variants(1, 0, 1, 0), [(1, 0)])
        with self.assertRaises(ValueError):
            layout_variants(1, 1, 1, 0)

    def test_vector_difference(self):
        self.assertEqual(vector_metrics([1, 2], [1, 3]), {"changed_elements": 1, "max_abs": 1, "mean_abs": 0.5})
