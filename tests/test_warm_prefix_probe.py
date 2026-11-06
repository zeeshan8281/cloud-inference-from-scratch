"""Diagnostic planning guards only; no GPU validation."""
import unittest

from experiments.custom_residual_probe import phase_plan
from experiments.warm_prefix_probe import configure


class WarmPrefixPlanTests(unittest.TestCase):
    def config(self, graph=False, observe=True):
        return configure({'batch_ids': [[i]*512 for i in range(8)], 'target_index': 2,
                          'graph_enabled': graph, 'observe': observe})

    def test_full_cold_warm_sequence_and_explicit_eager_graph_axis(self):
        for graph, observe in ((True,False),(False,False),(False,True)):
            config=self.config(graph,observe)
            self.assertEqual(config['output_tokens'],128)
            self.assertEqual(config['mode'],'complete_system')
            self.assertEqual(phase_plan(config),[('batch','batch-cold','cold'),('batch','batch-warm','warm'),
                                                ('solo','solo-cold','cold'),('solo','solo-warm','warm')])

    def test_graph_observation_and_truncated_windows_rejected(self):
        with self.assertRaises(ValueError):
            self.config(True,True)
        config=self.config()
        config['output_tokens']=2
        with self.assertRaises(ValueError):
            phase_plan(config)
        config=self.config()
        config['cases']=['solo','batch']
        with self.assertRaises(ValueError):
            phase_plan(config)

    def test_fixed_candidate_is_preserved_and_unknown_candidate_rejected(self):
        original = self.config()
        self.assertEqual(original['candidate'], 'fp32_qkv_and_head')
        fixed = configure({**original, 'candidate': 'fixed_tile_qkv_fp32_head'})
        self.assertEqual(fixed['candidate'], 'fixed_tile_qkv_fp32_head')
        self.assertEqual(phase_plan(fixed), phase_plan(original))
        self.assertEqual(fixed['output_tokens'], 128)
        with self.assertRaises(ValueError):
            configure({**original, 'candidate': 'unknown'})

    def test_legacy_resource_default_plan_unchanged(self):
        self.assertEqual(phase_plan({}),[(name,name,None) for name in
            ('batch_control','solo_control','batch','solo','batch_repeat','solo_repeat')])
