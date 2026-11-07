"""CPU-only guard tests with stubbed imports; not GPU/cache validation."""
import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class PreemptionGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner = type('Runner', (), {'execute_model': lambda *args: None, '_prepare_inputs': lambda *args: None})
        processor = type('Processor', (), {'_get_logits': lambda *args: None})
        imports = {
            'torch': SimpleNamespace(),
            'vllm.model_executor.layers.logits_processor': SimpleNamespace(LogitsProcessor=processor),
            'vllm.v1.worker.gpu_model_runner': SimpleNamespace(GPUModelRunner=runner),
            'cloud_engine.vllm_precision_worker': SimpleNamespace(PrecisionWorker=type('PrecisionWorker', (), {})),
        }
        with patch.dict(sys.modules, imports):
            sys.modules.pop('experiments.fix_holdout_worker', None)
            cls.worker = importlib.import_module('experiments.fix_holdout_worker')

    def test_resumed_preemption_is_explicitly_unsupported(self):
        schedule = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(
            req_ids=['a', 'b'], resumed_from_preemption=[False, True]))
        with self.assertRaisesRegex(self.worker.UnsupportedPreemption, 'b'):
            self.worker.require_no_preemption(schedule)
        schedule.scheduled_cached_reqs.resumed_from_preemption = [False, False]
        self.worker.require_no_preemption(schedule)

    def test_stale_unscheduled_owner_is_not_classified_as_corruption(self):
        with self.assertRaises(self.worker.UnsupportedPreemption):
            self.worker.require_writable_owner([('active',), ('paused',)], {'active'})
        with self.assertRaises(AssertionError):
            self.worker.require_writable_owner([('active',), ('other',)], {'active', 'other'})
        self.worker.require_writable_owner([('active',)], {'active'})
