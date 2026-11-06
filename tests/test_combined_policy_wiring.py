"""Pure lifecycle/guard checks; no GPU or numerical validation."""
import ast
import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, **scope):
    nodes = [node for node in ast.parse((ROOT/path).read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return scope


def precision_module():
    spec = importlib.util.spec_from_file_location('draft_precision', ROOT/'src/cloud_engine/precision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status(implementation='custom'):
    count = 252 if implementation == 'custom' else 144
    return {'installed': True, 'policy_id': 'fixed_tile_linears_fp32_head_fa2_min_query2_v1',
            'implementation': implementation, 'linears_count': count, 'expected_linears': count,
            'head_installed': True, 'wrappers_active': True, 'parameters_preserved': True,
            'storage_dtype': 'torch.float16', 'output_dtype': 'torch.float32',
            'precision': {'allow_fp16_reduced_precision_reduction': False, 'allow_tf32': False,
                          'float32_matmul_precision': 'highest'},
            'metadata_floor': {'installed': True, 'active': True, 'pinned_version': '0.10.0', 'fa_version': 2}}


class LifecycleTests(unittest.TestCase):
    def test_worker_install_after_load_once_without_extension_method_conflict(self):
        events = []
        class Worker:
            model_runner = NS(model='actual', model_config='config', vllm_config=NS(parallel_config='parallel'))
            def load_model(self):
                events.append('loaded')
                return 'result'
        def install(*args):
            self.assertEqual(args, ('actual', 'config', 'parallel'))
            events.append('installed')
        tree = ast.parse((ROOT/'src/cloud_engine/vllm_precision_worker.py').read_text())
        extension = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PrecisionWorker')
        self.assertNotIn('load_model', [node.name for node in extension.body if isinstance(node, ast.FunctionDef)])
        with mock.patch.dict('sys.modules', {'vllm.v1.worker.gpu_worker': NS(Worker=Worker),
                                            'cloud_engine.numerical_policy': NS(install_vllm_policy=install)}):
            ns = definitions('src/cloud_engine/vllm_precision_worker.py', {'_install_model_policy_loader'})
            ns['_install_model_policy_loader']()
            wrapper = Worker.load_model
            ns['_install_model_policy_loader']()
            self.assertIs(Worker.load_model, wrapper)
            self.assertEqual(Worker().load_model(), 'result')
            events.append('warmup_can_begin')
        self.assertEqual(events, ['loaded', 'installed', 'warmup_can_begin'])

    def test_custom_strict_weights_precede_install_and_failed_load_does_not_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'config.json').write_text('{}')
            events = []
            class Model:
                def __init__(self, *args, **kwargs):
                    pass
                def load_state_dict(self, state, strict):
                    self_outer.assertTrue(strict)
                    events.append('strict_load')
                    return NS(unexpected_keys=[], missing_keys=[])
                def eval(self):
                    events.append('eval')
                def requires_grad_(self, value):
                    pass
            self_outer = self
            ns = definitions('src/cloud_engine/weights.py', {'load_model'}, Path=Path, Any=object, json=json,
                             Qwen2CausalLM=Model, ModelDims=NS(from_hf_config=lambda _: 'dims'),
                             load_state_dict=lambda *a, **k: {})
            with mock.patch.dict('sys.modules', {'torch': NS(float16='fp16'),
                    'cloud_engine.numerical_policy': NS(install_custom_policy=lambda model: events.append('installed'))}):
                ns['load_model'](root, 'attention')
                self.assertEqual(events, ['strict_load', 'eval', 'installed'])
                events.clear()
                with mock.patch.object(Model, 'load_state_dict', return_value=NS(unexpected_keys=['bad'], missing_keys=[])):
                    with self.assertRaises(RuntimeError):
                        ns['load_model'](root, 'attention')
                self.assertNotIn('installed', events)


class PolicyGuardTests(unittest.TestCase):
    def test_actual_rpc_and_installation_or_arithmetic_changes_rejected(self):
        module = precision_module()
        engine = NS(collective_rpc=mock.Mock(return_value=[status('vllm')]))
        self.assertEqual(module.numerical_policy_snapshot(engine, 'vllm')['linears_count'], 144)
        engine.collective_rpc.assert_called_once_with('numerical_policy_status')
        for key, value in [('installed', False), ('wrappers_active', False), ('linears_count', 143),
                           ('head_installed', False), ('parameters_preserved', False)]:
            altered = status('vllm')
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.require_numerical_policy(altered, 'vllm')
        altered = status('vllm')
        altered['metadata_floor']['active'] = False
        with self.assertRaises(ValueError):
            module.require_numerical_policy(altered, 'vllm')
        altered = status()
        altered['precision']['allow_tf32'] = True
        with self.assertRaises(ValueError):
            module.require_numerical_policy(altered, 'custom')

    def test_pilot_postphase_policy_failure_keeps_measured_records(self):
        class Stop(Exception):
            def __init__(self, kind, detail):
                self.kind, self.detail = kind, detail
        phase = {'records': [{'output_token_ids': [1,2]}]}
        async def run(*args):
            return phase
        policy = NS(numerical_policy_snapshot=mock.Mock(side_effect=[status(), ValueError('removed wrapper')]))
        engine = NS(runner=NS(cuda_graph_captures=1, cuda_graph_replays=1))
        ns = definitions('experiments/sentinel_pilot.py', {'_run_phase'}, Any=object,
                         _run_custom_phase=run, StopPilot=Stop,
                         _graph_counters=lambda *args: {'unresolved': False})
        with mock.patch.dict('sys.modules', {'cloud_engine.precision': policy}):
            with self.assertRaises(Stop) as caught:
                asyncio.run(ns['_run_phase']('custom', engine, [], 'unit'))
        self.assertIs(caught.exception.detail['phase_result'], phase)
        self.assertEqual(phase['records'][0]['output_token_ids'], [1,2])
        self.assertIn('removed wrapper', phase['numerical_policy']['after']['error'])

    def test_combined_arithmetic_configuration_is_separate_from_legacy_paths(self):
        module = precision_module()
        matmul = NS(allow_fp16_reduced_precision_reduction=True, allow_tf32=True)
        setting = ['medium']
        torch = NS(backends=NS(cuda=NS(matmul=matmul)),
                   set_float32_matmul_precision=lambda value: setting.__setitem__(0, value),
                   get_float32_matmul_precision=lambda: setting[0])
        with mock.patch.dict('sys.modules', {'torch': torch}):
            module.configure_inference_precision()
            self.assertTrue(matmul.allow_tf32)
            self.assertEqual(setting[0], 'medium')
            module.configure_numerical_policy_precision()
            self.assertEqual(module.precision_status(), status()['precision'])
