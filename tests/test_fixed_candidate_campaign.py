"""Host/campaign control-flow checks only; no GPU or numerical validation."""
import ast
import base64
import hashlib
import io
import json
import struct
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


def load(names, **scope):
    if names & {"validate_micro", "fixed_campaign"}:
        names = names | {"assess_micro_accuracy"}
    nodes = [node for node in ast.parse((REPO / 'qkv_candidate_app.py').read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    scope.update(json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'fixed_campaign', 'exec'), scope)
    return scope


def vector(width, value, dtype):
    raw = struct.pack("<" + ("e" if dtype == "torch.float16" else "d") * width, *([value] * width))
    return {"shape": [width], "dtype": dtype, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes_base64": base64.b64encode(raw).decode()}


def micro_result():
    result = {'precision': {'allow_fp16_reduced_precision_reduction': False, 'allow_tf32': False,
                           'float32_matmul_precision': 'highest'}, 'schema': 'fixed-qkv-microprobe-v1',
              'fixture_sha256': hashlib.sha256(b'{}').hexdigest(),
              'baseline_reproduces_capture': True, 'candidate_layouts_exact': True, 'fused_separate_exact': True,
              'execution': [{'m': m, 'compiled_fullgraph_exact': True, 'cuda_graph_replay_exact': True} for m in (16, 256)],
              'projections': {}}
    for name, width in (("q", 2048), ("k", 256), ("v", 256)):
        result['projections'][name] = {"fp64_reference": vector(width, 1, "torch.float64"), "layouts": [
            {"m": m, "offset": offset, "baseline": vector(width, 1, "torch.float16"),
             "candidate": vector(width, 1, "torch.float16"), "baseline_fp64_error": {"max_abs": 0.0},
             "candidate_fp64_error": {"max_abs": 0.0}}
            for m, offset in sorted({(m, offset) for m in (1, 8, 16, 128, 256, 2048) for offset in (0, m-1)})]}
    return result


class FixedCampaignTests(unittest.TestCase):
    def test_micro_requires_all_reproduction_layout_and_execution_flags(self):
        validate = load({'validate_micro'})['validate_micro']
        result = micro_result()
        validate(result, result['fixture_sha256'])
        for key in ('baseline_reproduces_capture', 'candidate_layouts_exact', 'fused_separate_exact'):
            bad = {**result, key: False}
            with self.assertRaises(ValueError):
                validate(bad, result['fixture_sha256'])
        with self.assertRaises(ValueError):
            validate({**result, 'execution': result['execution'][:1]}, result['fixture_sha256'])

    def test_invariant_zero_kernel_fails_fp64_accuracy_without_tolerance(self):
        ns = load({'validate_micro'})
        result = micro_result()
        for name, width in (("q", 2048), ("k", 256), ("v", 256)):
            for row in result['projections'][name]['layouts']:
                row['candidate'] = vector(width, 0, "torch.float16")
                row['candidate_fp64_error']['max_abs'] = 1.0
        audit = ns['assess_micro_accuracy'](result)
        self.assertFalse(audit['passes'])
        self.assertEqual(audit['added_tolerance'], 0)
        with self.assertRaisesRegex(ValueError, "accuracy worse"):
            ns['validate_micro'](result, result['fixture_sha256'])
        result['projections']['q']['layouts'] = []
        with self.assertRaisesRegex(ValueError, "layouts"):
            ns['assess_micro_accuracy'](result)

    def test_micro_fail_stops_children_and_pair_failure_stops_next_mode(self):
        for micro_passes in (False, True):
            with self.subTest(micro_passes=micro_passes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = micro_result()
                result['candidate_layouts_exact'] = micro_passes
                micro = mock.Mock(return_value={'capture': result})
                child = mock.Mock(return_value={'capture': {}})
                commits = []
                ns = load({'fixed_campaign', 'validate_micro'},
                          Path=lambda value, root=root: root if value == '/evidence' else Path(value),
                          evidence_volume=NS(commit=lambda commits=commits, root=root: commits.append(list(root.rglob('*.json')))),
                          capture_micro=NS(local=micro), capture=NS(local=child),
                          validate_fixed_payload=lambda *args: None, validate_candidate_capture=lambda *args: None)
                with mock.patch.dict('sys.modules', {'fix_holdout_app_source': NS(_child_hard_failures=lambda *args: [])}), \
                        mock.patch('experiments.fix_holdout.summarize', return_value={'passes': False}):
                    index = ns['fixed_campaign']('b'*32, {'resource_normalized': {}, 'complete_system': {}}, '{}', 'source')
                self.assertFalse(index['complete'])
                self.assertTrue(index['error'])
                self.assertEqual(child.call_count, 2 if micro_passes else 0)
                self.assertIn('microprobe.json', index['artifacts'])
                self.assertTrue(commits)
                if micro_passes:
                    self.assertTrue(all(call.args[1] == 'resource_normalized' for call in child.call_args_list))
                    self.assertIn('resource_normalized-summary.json', index['artifacts'])

    def test_recovery_keeps_original_transport_error_and_never_spawns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dispatch = {'call_id': 'fc-existing', 'campaign_id': 'existing'}
            (root/'dispatch.json').write_text(json.dumps(dispatch))
            original = ConnectionError('transport unavailable')
            call = NS(get=mock.Mock(side_effect=original))
            spawn = mock.Mock()
            ns = load({'durable_fixed'}, Path=Path, fixed_campaign=NS(spawn=spawn),
                      modal=NS(FunctionCall=NS(from_id=lambda _: call)),
                      recover_artifacts=mock.Mock(side_effect=FileNotFoundError('pending')))
            with redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError) as caught:
                ns['durable_fixed'](temporary, 'unused', 'unused', recover=True)
            self.assertIs(caught.exception, original)
            self.assertEqual(json.loads((root/'dispatch.json').read_text()), dispatch)
            spawn.assert_not_called()


class ResidualFollowupTests(unittest.TestCase):
    def test_signature_ignores_addresses_but_includes_full_schedule_and_output(self):
        signature = load({'trace_control_signature'})['trace_control_signature']
        result = {'cases': {'batch': {'capture': {
            'records': [{'output_tokens': [1, 2, 3]}],
            'identity': {'forward_traces': [{'req_ids': ['r'], 'query_lengths': [1],
                'query_starts': [0, 1], 'computed_tokens': [4], 'packed_input_sha256': 'input',
                'kv_table_hashes': 'address-one'}]}}}}}
        before = signature(result)
        result['cases']['batch']['capture']['identity']['forward_traces'][0]['kv_table_hashes'] = 'address-two'
        self.assertEqual(signature(result), before)
        result['cases']['batch']['capture']['identity']['forward_traces'][0]['computed_tokens'] = [5]
        self.assertNotEqual(signature(result), before)

    def test_numerical_failure_retained_and_observer_change_stops_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def child(*args, trace_case=None):
                return {'capture': {'case': trace_case}}
            capture = mock.Mock(side_effect=child)
            ns = load({'fixed_followup_campaign'},
                      Path=lambda value: root if value == '/evidence' else Path(value),
                      evidence_volume=NS(commit=lambda: None), capture=NS(local=capture),
                      validate_fixed_payload=lambda *args: None, validate_candidate_capture=lambda *args: None,
                      validate_trace_capture=lambda *args: None,
                      trace_control_signature=lambda result: 'changed' if result['case'] == 'trace' else 'same')
            with mock.patch.dict('sys.modules', {'fix_holdout_app_source': NS(_child_hard_failures=lambda *args: ['numeric mismatch'])}):
                index = ns['fixed_followup_campaign']('a'*32, {'complete_system': {}, 'resource_normalized': {}}, 'source')
            self.assertEqual(capture.call_count, 3)
            self.assertFalse(index['complete'])
            self.assertIn('observer changed', index['error'])
            self.assertEqual(json.loads((root/('a'*32)/'complete_system-custom-failures.json').read_text()), ['numeric mismatch'])
            self.assertIn('vllm-trace.json', index['artifacts'])
            self.assertNotIn('vllm-repeat.json', index['artifacts'])
            self.assertFalse(index['gate_cleared'])


class AttentionFollowupTests(unittest.TestCase):
    def test_attention_campaign_runs_only_three_vllm_cases_and_retains_before_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def child(*args, trace_case=None, attention_replay=False):
                return {'environment': {'attention_replay': attention_replay},
                        'capture': {'config': {'attention_probe': attention_replay}, 'case': trace_case}}
            capture = mock.Mock(side_effect=child)
            def validate(payload, *args):
                self.assertTrue((root/('a'*32)/('vllm-'+payload['capture']['case']+'.json')).exists())
            ns = load({'fixed_followup_campaign'},
                      Path=lambda value: root if value == '/evidence' else Path(value),
                      evidence_volume=NS(commit=lambda: None), capture=NS(local=capture),
                      validate_fixed_payload=validate, validate_candidate_capture=lambda *args: None,
                      validate_trace_capture=lambda *args: None, validate_attention_capture=lambda *args: None,
                      trace_control_signature=lambda result: 'same')
            with mock.patch.dict('sys.modules', {'fix_holdout_app_source': NS(_child_hard_failures=mock.Mock(side_effect=AssertionError('custom must be skipped')))}):
                index = ns['fixed_followup_campaign']('a'*32, {'resource_normalized': {}}, 'source', True)
            self.assertTrue(index['complete'])
            self.assertTrue(index['attention_replay'])
            self.assertEqual([call.kwargs['trace_case'] for call in capture.call_args_list], ['control', 'trace', 'repeat'])
            self.assertTrue(all(call.args[0] == 'vllm' and call.kwargs['attention_replay'] is True for call in capture.call_args_list))
            self.assertNotIn('complete_system-custom.json', index['artifacts'])

    def test_explicit_attention_selector_detaches_without_unused_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'out'
            followup = mock.Mock(return_value=NS(object_id='fc-attention', get=lambda: None))
            fixed = mock.Mock(side_effect=AssertionError('micro campaign must be skipped'))
            ns = load({'durable_fixed'}, Path=Path, __file__=str(REPO/'qkv_candidate_app.py'),
                      app=NS(run=lambda **kwargs: nullcontext(), app_id='ap-attention'),
                      fixed_followup_campaign=NS(spawn=followup), fixed_campaign=NS(spawn=fixed),
                      candidate_sources=lambda _: {}, recover_artifacts=lambda *args: {'complete': True})
            with mock.patch('experiments.fix_holdout.computational_source', return_value={}), redirect_stdout(io.StringIO()):
                ns['durable_fixed'](str(destination), str(REPO/'experiments/fix-validation-smoke-01/manifest.json'),
                                    'missing-unused-fixture.json', attention_followup=True)
            self.assertIs(followup.call_args.args[3], True)
            self.assertEqual(followup.call_count, 1)
            self.assertFalse((destination/'fixture.json').exists())
            self.assertTrue(json.loads((destination/'dispatch.json').read_text())['attention_followup'])
            fixed.assert_not_called()

    def test_attention_guard_binds_raw_and_rejects_failed_native_replay(self):
        validate = load({'validate_attention_capture'}, Path=Path)['validate_attention_capture']
        flags = ('past_kv_unchanged', 'new_kv_matches_input', 'native_replay_exact',
                 'full_replay_repeat_exact', 'counterfactual_executed', 'target_inputs_fixed',
                 'singleton_repeat_exact', 'singleton_preserve_global_max_repeat_exact', 'singleton_max2_repeat_exact')
        record = dict.fromkeys(flags, True)
        record.update(layer_index=0, target_position=512, computed_tokens=512,
                      output_tokens_before=[279], singleton_equals_native=False, request_id="r",
                      actual_fa_version=2, unused_page_contents_serialized=False)
        payload = {'capture': {'config': {'trace_case': 'trace'}, 'cases': {}}, 'raw_traces': {}}
        def update():
            raw = json.dumps(record)+'\n'
            for selection in ('batch', 'solo'):
                name = selection+'.attention.jsonl'
                payload['raw_traces'][name] = raw
                payload['capture']['cases'][selection] = {
                    'attention_path': '/tmp/'+name, 'attention_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                    'attention_calls': [dict(record)], 'target_local_index': 0,
                    'capture': {'records': [{'output_tokens': [279, 3015], 'request_id': 'r'}]}}
        update()
        validate(payload, 'trace')  # Singleton differences are scientific observations.
        record['native_replay_exact'] = False
        update()
        with self.assertRaisesRegex(ValueError, 'integrity'):
            validate(payload, 'trace')
        record['native_replay_exact'] = True
        update()
        payload['raw_traces']['batch.attention.jsonl'] += ' '
        with self.assertRaisesRegex(ValueError, 'raw trace'):
            validate(payload, 'trace')

class AllLinearCampaignTests(unittest.TestCase):
    def test_micro_then_only_custom_modes_and_raw_retained_before_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = 'fixed_tile_linears_fp32_head'
            capture = mock.Mock(return_value={'capture': {}})
            micro = mock.Mock(return_value={'capture': {}})
            def guard(payload, *args, **kwargs):
                self.assertEqual(kwargs['candidate'], candidate)
                self.assertTrue((root/('a'*32)/'microprobe.json').exists())
            ns = load({'fixed_campaign'}, Path=lambda value: root if value == '/evidence' else Path(value),
                      evidence_volume=NS(commit=lambda: None), capture=NS(local=capture),
                      capture_micro=NS(local=micro), validate_fixed_payload=guard,
                      assess_unbiased_micro=lambda *args: {'passes': True}, validate_unbiased_micro=lambda *args: None,
                      validate_candidate_capture=lambda *args: None)
            with mock.patch.dict('sys.modules', {'fix_holdout_app_source': NS(_child_hard_failures=lambda *args: [])}):
                result = ns['fixed_campaign']('a'*32, {'resource_normalized': {}, 'complete_system': {}}, '{}', 'source', True)
            self.assertTrue(result['complete'])
            self.assertFalse(result['gate_cleared'])
            self.assertEqual(micro.call_args.args[1], candidate)
            self.assertEqual([call.args[:2] for call in capture.call_args_list],
                             [('custom', 'resource_normalized'), ('custom', 'complete_system')])

    def test_unbiased_accuracy_rejects_invariant_zero_and_missing_layout(self):
        assess = load({'assess_unbiased_micro'})['assess_unbiased_micro']
        fixture = {'cases': [{'name': name, 'weight_name': name, 'states': [{'phase': 'cold', 'M': 16, 'target_offset': 15}]} for name in ('down_proj', 'o_proj')]}
        operators = {}
        for name in ('down_proj', 'o_proj'):
            layouts = [{'m': m, 'offset': offset, 'repeat_exact': True,
                        'baseline': vector(2048, 1, 'torch.float16'), 'candidate': vector(2048, 0, 'torch.float16'),
                        'baseline_fp64_error': {'max_abs': 0}, 'candidate_fp64_error': {'max_abs': 1}}
                       for m, offset in sorted({(m,o) for m in (1,8,16,128,256,2048) for o in (0,m-1)})]
            operators[name] = {'weight_name': name, 'fp64_reference': vector(2048, 1, 'torch.float64'),
                               'layouts': layouts, 'all_layouts_exact': True, 'baseline_native_matches': {'cold': {'exact': True}}}
        self.assertFalse(assess({'operators': operators}, fixture)['passes'])
        operators['down_proj']['layouts'].pop()
        with self.assertRaisesRegex(ValueError, 'layouts'):
            assess({'operators': operators}, fixture)
