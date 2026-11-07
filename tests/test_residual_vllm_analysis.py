"""Pure comparison logic tests, not GPU validation."""
import unittest

from experiments.residual_vllm_analysis import operator_difference


def call(name, values, prefix=None):
    return {'request_id': 'diag-2', 'target_token_position': 512,
            'output_tokens_before': [279] if prefix is None else prefix,
            'target_input_token': 279, 'prompt_length': 512, 'module': name,
            'vectors': [{'path': 'output', 'dtype': 'torch.float16', 'shape': [8, 2], 'values': values}]}


class ResidualAnalysisTests(unittest.TestCase):
    def test_first_observed_difference_preserves_order(self):
        left = [call('norm', [1., 2.]), call('linear', [3., 4.])]
        right = [call('norm', [1., 2.]), call('linear', [3., 4.5])]
        result = operator_difference(left, right)
        self.assertEqual(result['first_observed_difference']['module'], 'linear')
        self.assertEqual(result['first_observed_difference']['max_abs'], .5)

    def test_prefix_and_nonfinite_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'output_tokens_before'):
            operator_difference([call('norm', [1., 2.])], [call('norm', [1., 2.], [3015])])
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            operator_difference([call('norm', [1., 2.])], [call('norm', [1., float('nan')])])
