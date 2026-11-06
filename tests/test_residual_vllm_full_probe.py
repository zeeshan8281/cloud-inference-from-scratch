"""CPU configuration guards; actual worker/observer reproduction needs the GPU."""
import unittest

from experiments.residual_vllm_full_probe import validate_config


class FullResidualConfigTests(unittest.TestCase):
    def config(self):
        return {'trace_case': 'trace', 'candidate': 'fixed_tile_qkv_fp32_head',
                'mode': 'resource_normalized', 'implementation': 'vllm',
                'target_index': 2, 'output_tokens': 128,
                'workload': {'output_tokens': 128, 'concurrency': 8, 'input_token_ids': [[1]*512 for _ in range(8)]}}

    def test_all_control_modes_preserve_full_workload(self):
        for case in ('control', 'trace', 'repeat'):
            config = self.config()
            config['trace_case'] = case
            self.assertIs(validate_config(config), config['workload']['input_token_ids'])

    def test_rejects_short_window_or_wrong_candidate(self):
        for key, value in (('output_tokens', 2), ('target_index', 6),
                           ('mode', 'complete_system'), ('candidate', 'fp32_qkv_and_head')):
            config = self.config()
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)
