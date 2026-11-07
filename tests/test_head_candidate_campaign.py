"""CPU-only head diagnostic host checks; not GPU validation."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(names, **scope):
    nodes = [node for node in ast.parse((ROOT/'qkv_candidate_app.py').read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    scope.update(json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'head-host', 'exec'), scope)
    return scope


class HeadOnlyCampaignTests(unittest.TestCase):
    def test_only_micro_runs_and_raw_is_retained_before_guard(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fullmodel = mock.Mock(side_effect=AssertionError('head-only cannot run a fullmodel'))
                micro = mock.Mock(return_value={'capture': {}})
                def guard(*args, root=root, fails=fails):
                    self.assertTrue((root/('a'*32)/'microprobe.json').exists())
                    self.assertTrue((root/('a'*32)/'micro-accuracy.json').exists())
                    if fails:
                        raise ValueError('micro evidence rejected')
                ns = load({'fixed_campaign'}, Path=lambda value, root=root: root if value == '/evidence' else Path(value),
                          evidence_volume=NS(commit=lambda: None), capture=NS(local=fullmodel),
                          capture_micro=NS(local=micro), validate_fixed_payload=lambda *args, **kwargs: None,
                          assess_head_micro=lambda *args, fails=fails: {'passes': not fails}, validate_head_micro=guard)
                with mock.patch.dict('sys.modules', {'fix_holdout_app_source': NS(_child_hard_failures=None)}):
                    result = ns['fixed_campaign']('a'*32, {}, '{}', 'source', head_only=True)
                self.assertEqual(result['complete'], not fails)
                self.assertFalse(result['gate_cleared'])
                self.assertTrue(result['head_only'])
                self.assertEqual(micro.call_args.args[1], 'fixed_tile_fp32_head_microprobe')
                self.assertEqual(micro.call_count, 1)
                fullmodel.assert_not_called()

class HeadEvidenceAuditTests(unittest.TestCase):
    def test_full_vector_accuracy_and_historical_subset_are_separate_and_strict(self):
        import base64
        import struct
        def vector(size, value, dtype):
            raw = struct.pack('<'+('f' if dtype == 'torch.float32' else 'd')*size, *([value]*size))
            return {'shape': [size], 'dtype': dtype, 'sha256': hashlib.sha256(raw).hexdigest(),
                    'bytes_base64': base64.b64encode(raw).decode()}
        audit = load({'assess_head_micro'})['assess_head_micro']
        score = vector(20, 1, 'torch.float32')
        full = vector(151936, 1, 'torch.float32')
        fixture = {'pinned_model_contract': {'original_vocab_size': 151936, 'hidden_size': 2048},
                   'states': [{'selection': name, 'M': m, 'projection_row': offset,
                               'observed_token_ids': list(range(20)), 'observed_fp32_logits': score}
                              for name,m,offset in [('batch',5,2),('solo',1,0)]]}
        history = [{'selection': row['selection'], 'm': row['M'], 'offset': row['projection_row'],
                    'token_ids': row['observed_token_ids'], 'captured': score, 'replay': score, 'exact': True}
                   for row in fixture['states']]
        layouts = [{'m': m, 'offset': offset, 'baseline': full, 'candidate': full,
                    'baseline_fp64_error': {'max_abs': 0}, 'candidate_fp64_error': {'max_abs': 0}, 'repeat_exact': True}
                   for m,offset in sorted({(m,o) for m in (1,5,8,16,32) for o in (0,m-1)}|{(5,2)})]
        result = {'historical_reproduction': history, 'layouts': layouts,
                  'full_fp64_reference': vector(151936, 1, 'torch.float64')}
        self.assertTrue(audit(result, fixture)['passes'])
        layouts[0]['baseline'] = vector(151936, 0, 'torch.float32')
        layouts[0]['baseline_fp64_error'] = {'max_abs': 1}
        with self.assertRaisesRegex(ValueError, 'full native baseline'):
            audit(result, fixture)
        layouts[0]['baseline'], layouts[0]['baseline_fp64_error'] = full, {'max_abs': 0}
        zero = vector(151936, 0, 'torch.float32')
        for row in layouts:
            row.update(candidate=zero, candidate_fp64_error={'max_abs': 1})
        self.assertFalse(audit(result, fixture)['passes'])  # Invariant zero is not accurate.
        history[0]['replay'] = vector(20, 0, 'torch.float32')
        with self.assertRaisesRegex(ValueError, 'historical twenty'):
            audit(result, fixture)
        history[0]['replay'] = score
        zero['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            audit(result, fixture)

    def test_missing_compile_or_graph_layout_cannot_pass(self):
        validate = load({'validate_head_micro'})['validate_head_micro']
        result = {'schema': 'fixed-fp32-head-microprobe-v1', 'fixture_sha256': 'fixture',
                  'precision': {'allow_fp16_reduced_precision_reduction': False, 'allow_tf32': False,
                                'float32_matmul_precision': 'highest'},
                  'baseline_reproduces_capture': True, 'candidate_layouts_exact': True,
                  'accuracy_no_worse': True, 'execution_exact': True,
                  'execution': [{'m': m, 'compiled_fullgraph_exact': True, 'cuda_graph_replay_exact': True} for m in (1,5,32)]}
        validate(result, 'fixture', {'passes': True})
        result['execution'].pop()
        with self.assertRaisesRegex(ValueError, 'compiled/graph'):
            validate(result, 'fixture', {'passes': True})
