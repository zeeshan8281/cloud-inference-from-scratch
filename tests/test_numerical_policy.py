import gc
import importlib.util
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('combined_policy', Path(__file__).resolve().parents[1] / 'src/cloud_engine/numerical_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class NumericalPolicyTests(unittest.TestCase):
    def test_floor_changes_only_legal_upper_bound(self):
        offsets, cache = object(), object()
        metadata = SimpleNamespace(num_actual_tokens=8, max_query_len=1,
                                   query_start_loc=offsets, block_table=cache, max_seq_len=513)
        self.assertIs(policy.floor_metadata(metadata, True), metadata)
        self.assertEqual(metadata.max_query_len, 2)
        self.assertIs(metadata.query_start_loc, offsets)
        self.assertIs(metadata.block_table, cache)
        self.assertEqual(metadata.max_seq_len, 513)
        for eligible, tokens, maximum in ((False, 8, 1), (True, 0, 1), (True, 8, 512), (True, 8, 2)):
            metadata = SimpleNamespace(num_actual_tokens=tokens, max_query_len=maximum)
            policy.floor_metadata(metadata, eligible)
            self.assertEqual(metadata.max_query_len, maximum)

    def test_dimensions_fail_closed(self):
        config = SimpleNamespace(hidden_size=2048, intermediate_size=11008, vocab_size=151936,
                                 num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
                                 tie_word_embeddings=True)
        self.assertTrue(policy.pinned_dimensions(config))
        config.vocab_size = 152064
        self.assertFalse(policy.pinned_dimensions(config))

    def test_vllm_scope_excludes_later_bf16_and_quantized_builders(self):
        config = SimpleNamespace(hidden_size=2048, intermediate_size=11008, vocab_size=151936,
                                 num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
                                 tie_word_embeddings=True, model_type='qwen2')
        model = SimpleNamespace(hf_config=config, dtype='torch.float16', quantization=None)
        parallel = SimpleNamespace(tensor_parallel_size=1)
        self.assertTrue(policy.vllm_scope(model, parallel))
        model.dtype = 'torch.bfloat16'
        self.assertFalse(policy.vllm_scope(model, parallel))
        model.dtype, model.quantization = 'torch.float16', 'awq'
        self.assertFalse(policy.vllm_scope(model, parallel))

    def test_status_detects_replaced_method_and_parameter(self):
        model = SimpleNamespace()
        def function():
            return None
        method = SimpleNamespace(apply=function)
        parameter = SimpleNamespace(dtype='torch.float16')
        owner = SimpleNamespace(quant_method=method, weight=parameter)
        policy._record(model, 'vllm', [(method, 'apply', function)], [(owner, 'weight', parameter)], 0, [(owner, method)])
        self.assertTrue(policy.policy_status(model)['wrappers_active'])
        owner.quant_method = object()
        self.assertFalse(policy.policy_status(model)['wrappers_active'])
        owner.quant_method = method
        owner.weight = SimpleNamespace(dtype='torch.float16')
        self.assertFalse(policy.policy_status(model)['parameters_preserved'])


    def test_record_is_model_owned_and_reclaimable_with_head_cycle(self):
        class Owner:
            pass

        def make_installed():
            model, parameter = Owner(), Owner()
            parameter.dtype = 'torch.float16'
            model.weight = parameter
            def head(hidden, owner=model):
                return owner.weight
            model.project_logits = head
            policy._record(model, 'custom', [(model, 'project_logits', head)],
                           [(model, 'weight', parameter)], 0)
            self.assertTrue(policy.policy_status(model)['wrappers_active'])
            # Idempotence returns the existing record before importing GPU libraries.
            record = getattr(model, policy._STATE_ATTRIBUTE)
            self.assertEqual(policy.install_custom_policy(model), policy.policy_status(model))
            self.assertIs(getattr(model, policy._STATE_ATTRIBUTE), record)
            return weakref.ref(model), weakref.ref(parameter)

        model_ref, parameter_ref = make_installed()
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(parameter_ref())

    def test_existing_installation_reports_callable_tampering(self):
        model = SimpleNamespace()
        def original(hidden):
            return hidden
        model.project_logits = original
        policy._record(model, 'custom', [(model, 'project_logits', original)], [], 0)
        model.project_logits = object()
        self.assertFalse(policy.install_custom_policy(model)['wrappers_active'])


if __name__ == '__main__':
    unittest.main()
