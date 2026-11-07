"""Candidate selection guards only; no mocked GPU validation."""
import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.qkv_policy_probe import (
    eligible,
    fp32_linear,
    head_eligible,
    install_custom,
    load_candidate,
    require_fixed_bias,
    select_output_projection,
)


class QKVPolicyScopeTests(unittest.TestCase):
    def test_type_bias_and_dtype_required(self):
        class Target:
            weight = SimpleNamespace(dtype='fp16')
        layer = Target()
        fp16 = SimpleNamespace(dtype='fp16')
        self.assertTrue(eligible(layer, fp16, fp16, Target, 'fp16'))
        self.assertFalse(eligible(layer, fp16, None, Target, 'fp16'))
        self.assertFalse(eligible(layer, SimpleNamespace(dtype='bf16'), fp16, Target, 'fp16'))
        self.assertFalse(eligible(SimpleNamespace(weight=fp16), fp16, fp16, Target, 'fp16'))


class CandidateLifecycleTests(unittest.TestCase):
    def test_installed_before_loaded_model_reaches_runner(self):
        events = []
        model = SimpleNamespace(installed=False)
        def loader():
            events.append('weights_loaded')
            return model, 'metadata'
        def installer(loaded):
            self.assertIs(loaded, model)
            loaded.installed = True
            events.append('candidate_installed')
            return 'status_callback'
        result, status = load_candidate(loader, installer)
        self.assertTrue(result[0].installed)
        events.append('runner_can_capture')
        self.assertEqual(events, ['weights_loaded', 'candidate_installed', 'runner_can_capture'])
        self.assertEqual(status, 'status_callback')


class HeadScopeTests(unittest.TestCase):
    def test_head_type_and_actual_fp16_inputs_required_without_bias(self):
        class Head:
            weight = SimpleNamespace(dtype='fp16')
        fp16 = SimpleNamespace(dtype='fp16')
        self.assertTrue(head_eligible(Head(), fp16, Head, 'fp16'))
        self.assertFalse(head_eligible(SimpleNamespace(weight=fp16), fp16, Head, 'fp16'))
        self.assertFalse(head_eligible(Head(), SimpleNamespace(dtype='fp32'), Head, 'fp16'))


class TiedHeadSelectionTests(unittest.TestCase):
    def test_tied_and_untied_select_actual_output_object(self):
        class Embedding:
            weight = SimpleNamespace(dtype='fp16')
        class UntiedHead(Embedding):
            pass
        embedding = Embedding()
        for head, expected_tied in [(embedding, True), (UntiedHead(), False)]:
            model = SimpleNamespace(lm_head=head, model=SimpleNamespace(embed_tokens=embedding))
            actual, tied = select_output_projection(model, Embedding, 'fp16')
            self.assertIs(actual, head)
            self.assertEqual(tied, expected_tied)
        with self.assertRaises(ValueError):
            select_output_projection(SimpleNamespace(lm_head=SimpleNamespace(weight=embedding.weight)), Embedding, 'fp16')


class CompiledQKVMathTests(unittest.TestCase):
    def test_math_helper_has_no_runtime_policy_query_or_graph_bypass(self):
        tree = ast.parse(inspect.getsource(fp32_linear))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertNotIn('policy_state', names)
        self.assertTrue({'linear', 'float', 'to'} <= attributes)
        self.assertFalse({'backends', 'disable', 'graph_break'} & attributes)


class FixedTileIntegrationTests(unittest.TestCase):
    def test_custom_installs_same_operator_without_qkv_hotpath_counters(self):
        class Linear:
            weight = SimpleNamespace(dtype='fp16')
            bias = SimpleNamespace(dtype='fp16')
        class Hooks:
            def __init__(self):
                self._forward_hooks = {}
            def register_forward_hook(self, hook):
                self._forward_hooks[1] = hook
                return SimpleNamespace(id=1)
        modules = [(f'layers.0.self_attn.{name}', Linear()) for name in ('q_proj', 'k_proj', 'v_proj')]
        model = Hooks()
        model.norm = Hooks()
        model.named_modules = lambda: modules
        model.dims = SimpleNamespace(num_layers=1)
        model.lm_head = None
        model.embed_tokens = Linear()
        kernel = mock.Mock(return_value='kernel-result')
        with mock.patch.dict('sys.modules', {'torch': SimpleNamespace(nn=SimpleNamespace(Linear=Linear), float16='fp16')}), \
                mock.patch('cloud_engine.batch_invariant_linear.batch_invariant_linear', kernel), \
                mock.patch('experiments.qkv_policy_probe.policy_state', return_value={}):
            state = install_custom(model, 'fixed_tile_qkv_fp32_head')
            self.assertEqual(modules[0][1].forward('input'), 'kernel-result')
            kernel.assert_called_once_with('input', modules[0][1].weight, modules[0][1].bias)
            self.assertEqual(state()['calls'], {})
            self.assertTrue(state()['fixed_tile_policy']['operator_active'])
            modules[0][1].forward = lambda x: x
            self.assertFalse(state()['wrapper_active'])

    def test_fixed_worker_rejects_skipped_or_unknown_bias_at_load(self):
        require_fixed_bias(SimpleNamespace(skip_bias_add=False), 'fixed_tile_qkv_fp32_head')
        for layer in (SimpleNamespace(skip_bias_add=True), SimpleNamespace()):
            with self.assertRaises(ValueError):
                require_fixed_bias(layer, 'fixed_tile_qkv_fp32_head')
        require_fixed_bias(SimpleNamespace(skip_bias_add=True), 'fp32_qkv_and_head')

    def test_vllm_fixed_operator_is_scoped_and_bias_none_preserves_native(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'experiments/qkv_policy_worker.py').read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'apply')
        class QKV:
            weight = 'weight'
        layer = QKV()
        native, kernel = mock.Mock(return_value='native'), mock.Mock(return_value='fixed')
        scope = {'QKVParallelLinear': QKV, '_apply': native, '_fixed_linear': kernel,
                 'eligible': lambda *args: True, 'torch': SimpleNamespace(float16='fp16'),
                 '_names': {id(layer): 'qkv'}, '_calls': {}, '_all_linears': False}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'worker_scope', 'exec'), scope)
        self.assertEqual(scope['apply']('method', layer, 'x', 'bias'), 'fixed')
        kernel.assert_called_once_with('x', 'weight', 'bias')
        self.assertEqual(scope['_calls'], {})
        self.assertEqual(scope['apply']('method', layer, 'x', None), 'native')
        self.assertEqual(scope['apply']('method', object(), 'x', 'bias'), 'native')

    def test_all_linear_custom_coverage_and_unbiased_operator_dispatch(self):
        class Linear:
            weight = SimpleNamespace(dtype='fp16')
            def __init__(self, bias):
                self.bias = SimpleNamespace(dtype='fp16') if bias else None
        class Hooks:
            def __init__(self):
                self._forward_hooks = {}
            def register_forward_hook(self, hook):
                self._forward_hooks[1] = hook
                return SimpleNamespace(id=1)
        modules = [(f'layers.0.self_attn.{name}', Linear(name != 'o_proj'))
                   for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
        modules += [(f'layers.0.mlp.{name}', Linear(False)) for name in ('gate_proj', 'up_proj', 'down_proj')]
        model = Hooks()
        model.norm = Hooks()
        model.named_modules = lambda: modules
        model.dims = SimpleNamespace(num_layers=1)
        model.lm_head = None
        model.embed_tokens = Linear(False)
        kernel = mock.Mock(return_value='fixed')
        with mock.patch.dict('sys.modules', {'torch': SimpleNamespace(nn=SimpleNamespace(Linear=Linear), float16='fp16')}), \
                mock.patch('cloud_engine.batch_invariant_linear.fixed_tile_linear', kernel), \
                mock.patch('experiments.qkv_policy_probe.policy_state', return_value={}):
            state = install_custom(model, 'fixed_tile_linears_fp32_head')
            self.assertEqual(state()['module_count'], 7)
            self.assertEqual(state()['calls'], {})
            self.assertTrue(state()['fixed_tile_policy']['operator_active'])
            for _, module in modules:
                self.assertEqual(module.forward('x'), 'fixed')
            self.assertEqual(kernel.call_count, 7)
            self.assertIsNone(kernel.call_args.args[2])
            modules[0][1].bias = None
            with self.assertRaisesRegex(ValueError, 'bias'):
                install_custom(model, 'fixed_tile_linears_fp32_head')

    def test_all_linear_worker_scoped_identity_and_no_bias_fallback(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'experiments/qkv_policy_worker.py').read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == 'apply')
        layer = SimpleNamespace(weight='weight')
        native, kernel = mock.Mock(return_value='native'), mock.Mock(return_value='fixed')
        scope = {'_all_linears': True, '_names': {id(layer): 'down'}, '_apply': native, '_fixed_linear': kernel}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'worker', 'exec'), scope)
        self.assertEqual(scope['apply']('method', layer, 'x', None), 'fixed')
        kernel.assert_called_once_with('x', 'weight', None)
        self.assertEqual(scope['apply']('method', object(), 'x', None), 'native')
