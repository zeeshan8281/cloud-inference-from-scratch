"""CPU callable/alias/contract checks only; real compiled CUDA proof is separate."""
import gc
import importlib.util
import struct
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('policy', ROOT/'src/cloud_engine/numerical_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class HalfBuffer:
    def __init__(self, value):
        self.value = struct.unpack('<e', struct.pack('<e',value))[0]
    def __add__(self, other):
        return HalfBuffer(self.value+other.value)
    def copy_(self, other):
        self.value = other.value
        return self


class Norm:
    def __init__(self):
        self.weight = NS(shape=(2048,),dtype='torch.float16',is_cuda=True,device='cuda:0',is_contiguous=lambda:True)
        self.variance_epsilon = 1e-6
        self.hidden_size, self.variance_size_override, self.has_weight = 2048,None,True
    def forward(self, x, residual=None):
        return x


class CommonNormTests(unittest.TestCase):
    def test_pure_preserves_input_and_fused_keeps_aliases_after_half_add(self):
        observed = []
        def normalize(x,w,eps):
            observed.append(x.value)
            return HalfBuffer(12)
        fn = policy._norm_forward(normalize)
        module = NS(weight=object(),variance_epsilon=1e-6)
        x = HalfBuffer(1)
        result = fn(module,x)
        self.assertIsNot(result,x)
        self.assertEqual(x.value,1)
        residual = HalfBuffer(2**-11)
        left,right = fn(module,x,residual)
        self.assertIs(left,x)
        self.assertIs(right,residual)
        self.assertEqual(residual.value,1)  # round-to-even at FP16 addition
        self.assertEqual(observed,[1,1])
        self.assertEqual(x.value,12)

    def test_contract_rejects_last_bad_module_without_changing_any_forward(self):
        norms = [Norm() for _ in range(73)]
        root = NS(layers=[NS(input_layernorm=norms[2*i],post_attention_layernorm=norms[2*i+1]) for i in range(36)],
                  norm=norms[-1],embed_tokens=NS(weight=NS(device='cuda:0')))
        with mock.patch.dict('sys.modules',{'vllm.model_executor.layers.layernorm':NS(RMSNorm=Norm)}):
            result,_ = policy._validate_norms(NS(model=root),'vllm',1e-6,NS(float16='torch.float16'))
            self.assertEqual(result,norms)
            norms[-1].variance_size_override = 1024
            with self.assertRaises(ValueError):
                policy._validate_norms(NS(model=root),'vllm',1e-6,NS(float16='torch.float16'))
        self.assertTrue(all('forward' not in n.__dict__ for n in norms))

    def test_status_detects_norm_method_parameter_epsilon_and_does_not_change_linear_count(self):
        norms = [Norm() for _ in range(73)]
        records = policy._normalization_records(norms,'variance_epsilon',lambda x,w,e:x,replace=True)
        owner = NS()
        policy._record(owner,'vllm',[(NS(apply=None),'apply',None)]*145,[],144,norms=records)
        self.assertEqual(policy.policy_status(owner)['linears_count'],144)
        self.assertTrue(policy.policy_status(owner)['normalization']['active'])
        old = norms[0].forward
        norms[0].forward = object()
        self.assertFalse(policy.policy_status(owner)['normalization']['active'])
        norms[0].forward = old
        norms[0].variance_epsilon = 1e-5
        self.assertFalse(policy.policy_status(owner)['normalization']['active'])
        norms[0].variance_epsilon = 1e-6
        norms[0].weight = NS(dtype='torch.float16')
        self.assertFalse(policy.policy_status(owner)['normalization']['parameter_identity_preserved'])

    def test_custom_same_class_method_rebound_to_other_module_is_rejected(self):
        norms = [Norm() for _ in range(73)]
        def normalizer(x, w, e):
            return x
        with mock.patch.dict(Norm.forward.__globals__, {'rms_norm':normalizer}):
            records = policy._normalization_records(norms,'variance_epsilon',normalizer)
            self.assertTrue(policy._normalization_status(records)['active'])
            norms[0].forward = norms[1].forward
            self.assertFalse(policy._normalization_status(records)['active'])

    def test_model_owned_norm_cycles_are_collectible(self):
        class Model:
            pass
        model = Model()
        modules = [Norm() for _ in range(73)]
        model.layers = modules
        records = policy._normalization_records(modules,'variance_epsilon',lambda x,w,e:x,replace=True)
        policy._record(model,'vllm',[],[],0,norms=records)
        model_ref, norm_ref = weakref.ref(model), weakref.ref(modules[0])
        del model, modules, records
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(norm_ref())


if __name__ == '__main__':
    unittest.main()
