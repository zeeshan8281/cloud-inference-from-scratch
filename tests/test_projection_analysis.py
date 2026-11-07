"""Synthetic tests of evidence alignment, not GPU validation."""
import copy
import unittest

from experiments.projection_analysis import compare


def fixture(token=1401):
    row = {'hidden': [1.0, 2.0], 'candidate_ids': [323, 1401],
           'native_logits': [3.0, 3.0], 'fp32_projection': [3.0, 3.01],
           'native_argmax': token}
    return {'exit_code': 0, 'gpu': {'uuid': 'fixture'}, 'capture': {
        'config': {'target_index': 0, 'batch_ids': [[1, 2]]}, 'result': {'output_tokens': [token], 'verified': True},
        'projection_calls': [{'rows': [row]}]}}


class ProjectionEvidenceTests(unittest.TestCase):
    def test_identical_hidden_can_have_different_selected_tokens(self):
        result = compare(fixture(323), fixture())
        self.assertTrue(result['hidden_exact_equal'])
        self.assertEqual(result['original']['native_1401_minus_323'], 0)
        self.assertGreater(result['original']['fp32_1401_minus_323'], 0)

    def test_rejects_changed_gpu_or_misaligned_output(self):
        for mutation in ('gpu', 'output', 'nonfinite'):
            right = copy.deepcopy(fixture())
            if mutation == 'gpu':
                right['gpu']['uuid'] = 'other'
            elif mutation == 'output':
                right['capture']['result']['output_tokens'] = [323]
            else:
                right['capture']['projection_calls'][0]['rows'][0]['hidden'][0] = float('nan')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare(fixture(), right)

    def test_rejects_invalid_candidate_support_and_unverified_capture(self):
        for mutation in ('duplicate', 'truncated', 'nonmaximal', 'unverified', 'crashed'):
            right = fixture()
            row = right['capture']['projection_calls'][0]['rows'][0]
            if mutation == 'duplicate':
                row['candidate_ids'] = [1401, 1401]
            elif mutation == 'truncated':
                row['fp32_projection'] = [3.0]
            elif mutation == 'nonmaximal':
                row['native_logits'] = [4.0, 3.0]
            elif mutation == 'unverified':
                right['capture']['result']['verified'] = False
            else:
                right['capture'] = None
                right['exit_code'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare(fixture(), right)

    def test_identifies_worker_rows_and_checks_prefix(self):
        left, right = fixture(), fixture()
        for payload in (left, right):
            row = payload['capture']['projection_calls'][0]['rows'][0]
            row.update(request_id='diag-0', output_tokens_before=[])
            row.update(permuted_argmax=1401, permuted_logits=[3.0, 3.0])
        result = compare(left, right)
        self.assertEqual(result['alignment'], 'request ID and emitted prefix')
        self.assertTrue(result['original']['permuted_candidate_logits_equal'])
        right['capture']['projection_calls'][0]['rows'][0]['output_tokens_before'] = [7]
        with self.assertRaisesRegex(ValueError, 'shared prefix'):
            compare(left, right)
