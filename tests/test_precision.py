"""Policy and worker wiring unit tests; these are not GPU validation."""
import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cloud_engine.precision import (
    configure_inference_precision,
    precision_status,
    require_vllm_precision,
)
from experiments.controlled import _vllm_engine
from experiments.sentinel_pilot import _build_vllm_engine


class PrecisionTests(unittest.TestCase):
    def test_policy_changes_only_fp16_reduction_and_worker_applies_it(self):
        matmul = SimpleNamespace(allow_fp16_reduced_precision_reduction=True,
                                 allow_bf16_reduced_precision_reduction=True, allow_tf32=False)
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
                                get_float32_matmul_precision=lambda: 'highest')
        with patch.dict(sys.modules, {'torch': torch}):
            configure_inference_precision()
            self.assertEqual(precision_status(), {'allow_fp16_reduced_precision_reduction': False,
                                                 'allow_tf32': False, 'float32_matmul_precision': 'highest'})
            self.assertTrue(matmul.allow_bf16_reduced_precision_reduction)
            self.assertFalse(matmul.allow_tf32)
            matmul.allow_fp16_reduced_precision_reduction = True
            # Import-time configuration is what the spawned worker executes.
            name = 'cloud_engine.vllm_precision_worker'
            with patch.dict(sys.modules, {'vllm.v1.worker.gpu_worker': SimpleNamespace(Worker=type('Worker', (), {'load_model': lambda self: None}))}):
                sys.modules.pop(name, None)
                worker = importlib.import_module(name).PrecisionWorker()
                self.assertIs(worker.precision_status()['allow_fp16_reduced_precision_reduction'], False)

    def test_worker_verification_rejects_missing_or_wrong_actual_policy(self):
        for states in ([], [{}], [{'allow_fp16_reduced_precision_reduction': True}],
                       [{'allow_fp16_reduced_precision_reduction': False}, {}]):
            engine = SimpleNamespace(collective_rpc=lambda method, states=states: states)
            with self.subTest(states=states), self.assertRaises(RuntimeError):
                require_vllm_precision(engine)

    def test_vllm_builders_wire_and_query_actual_worker(self):
        calls = []
        def build(args):
            calls.append(args)
            def rpc(method):
                self.assertEqual(method, 'precision_status')
                return [{'allow_fp16_reduced_precision_reduction': False}]
            return SimpleNamespace(collective_rpc=rpc)
        vllm = SimpleNamespace(EngineArgs=lambda **kwargs: kwargs,
                               LLMEngine=SimpleNamespace(from_engine_args=build))
        with patch.dict(sys.modules, {'vllm': vllm}):
            _vllm_engine('model', {'max_model_len': 4096}, False)
            _build_vllm_engine('model', {'max_model_len': 4096}, 'resource_normalized', SimpleNamespace(num_blocks=64))
            _build_vllm_engine('model', {'max_model_len': 4096}, 'complete_system', None)
        self.assertEqual(len(calls), 3)
        for args in calls:
            self.assertEqual(args['worker_extension_cls'], 'cloud_engine.vllm_precision_worker.PrecisionWorker')


class GraphEvidenceTests(unittest.TestCase):
    def test_phase_deltas_do_not_attribute_warmup_graphs_to_eager_phase(self):
        from experiments.sentinel_pilot import _custom_graph_counters, _vllm_graph_counters

        before = {'observer_active': True, 'capture_count': 9, 'replay_count': 100}
        engine = SimpleNamespace(_cie_phase_graph_observation=([before], [dict(before)]))
        self.assertEqual(_vllm_graph_counters(engine, 'complete_system')['path'], 'eager_fallback')
        engine._cie_phase_graph_observation[1][0]['replay_count'] += 1
        self.assertEqual(_vllm_graph_counters(engine, 'complete_system')['path'], 'capture_or_replay')
        engine._cie_phase_graph_observation[1][0]['observer_active'] = False
        self.assertTrue(_vllm_graph_counters(engine, 'complete_system')['unresolved'])
        counters = {'cuda_graph_captures': 5, 'cuda_graph_replays': 80}
        engine._cie_phase_graph_observation = (counters, dict(counters))
        self.assertEqual(_custom_graph_counters(engine)['path'], 'eager_fallback')

    def test_unresolved_phase_stops_and_retains_records(self):
        import asyncio

        from experiments import sentinel_pilot

        engine = SimpleNamespace(collective_rpc=lambda method: [{'observer_active': False}])
        phase = {'records': [{'request_index': 0, 'output_token_ids': [1], 'error': None}]}
        with patch.object(sentinel_pilot, '_run_vllm_phase', return_value=phase), patch(
                'cloud_engine.precision.numerical_policy_snapshot', return_value={'unit': 'graph test isolates policy'}):
            with self.assertRaises(sentinel_pilot.StopPilot) as caught:
                asyncio.run(sentinel_pilot._run_phase('vllm', engine, [{'request_index': 0, 'output_tokens': 1}], 'test'))
        self.assertEqual(caught.exception.kind, 'graph_path_unresolved')
        self.assertIs(caught.exception.detail['phase_result'], phase)


class CapacityEvidenceTests(unittest.TestCase):
    def test_worker_capacity_is_used_when_v1_driver_value_is_unset(self):
        from experiments.sentinel_pilot import _vllm_kv_capacity

        actual = {'num_gpu_blocks': 64, 'block_sizes': [16], 'tensor_parallel_size': 1}
        engine = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks=None, num_gpu_blocks_override=999),
                                 collective_rpc=lambda method: [actual])
        self.assertEqual(_vllm_kv_capacity(engine)['num_gpu_blocks'], 64)
        engine.cache_config.num_gpu_blocks = 999
        with self.assertRaisesRegex(ValueError, 'disagree'):
            _vllm_kv_capacity(engine)

    def test_missing_actual_capacity_cannot_be_replaced_by_requested_override(self):
        from experiments.sentinel_pilot import _vllm_kv_capacity

        for workers in ([], [{}], [{'num_gpu_blocks': None, 'block_sizes': [16], 'tensor_parallel_size': 1}],
                        [{'num_gpu_blocks': 64, 'block_sizes': [32], 'tensor_parallel_size': 1}]):
            engine = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=64),
                                     collective_rpc=lambda method, workers=workers: workers)
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                _vllm_kv_capacity(engine)
