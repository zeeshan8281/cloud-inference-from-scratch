"""CPU registration/identity tests; opaque compile/CUDA behavior requires real GPU."""
import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]


class OpaqueNormalizationTests(unittest.TestCase):
    def test_cuda_only_nonmutating_body_and_fresh_contiguous_fake(self):
        registration, calls = [], []
        class Operator:
            def __init__(self, body):
                self.body = body
            def __call__(self, *args):
                return self.body(*args)
            def register_fake(self, fn):
                self.fake = fn
                return fn
        def custom_op(name, **kwargs):
            registration.append((name, kwargs))
            return Operator
        result = NS()
        result.contiguous = lambda: result
        def native(*args):
            calls.append(args)
            return result
        scope = {'torch':NS(Tensor=object,library=NS(custom_op=custom_op)), 'rms_norm':native}
        tree = ast.parse((ROOT/'src/cloud_engine/opaque_normalization.py').read_text())
        nodes = [node for node in tree.body if not isinstance(node,(ast.Import,ast.ImportFrom))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'opaque-body','exec'),scope)
        op = scope['opaque_rms_norm']
        x, weight = object(), object()
        self.assertIs(op(x,weight,1e-6),result)
        self.assertEqual(calls,[(x,weight,1e-6)])
        self.assertEqual(registration,[('cloud_engine::common_rms_norm',{'mutates_args':(),'device_types':'cuda'})])
        allocated = []
        def empty(shape):
            value = NS(shape=shape)
            allocated.append(value)
            return value
        fake_x = NS(shape=(0,2048),new_empty=empty)
        fake = op.fake(fake_x,weight,1e-6)
        self.assertIs(fake,allocated[0])
        self.assertIsNot(fake,fake_x)
        self.assertEqual(fake.shape,(0,2048))

    def test_status_rejects_replaced_operator_or_eager_body(self):
        spec = importlib.util.spec_from_file_location('policy',ROOT/'src/cloud_engine/numerical_policy.py')
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        def eager(x,w,e):
            return x
        def opaque(x,w,e):
            return x
        owner = NS(opaque_rms_norm=opaque,rms_norm=eager)
        modules = [NS(weight=NS(dtype='torch.float16'),variance_epsilon=1e-6) for _ in range(73)]
        records = policy._normalization_records(modules,'variance_epsilon',opaque,replace=True)
        model = NS()
        policy._record(model,'vllm',[],[],0,norms=records,norm_operator=(owner,opaque,eager))
        self.assertTrue(policy.policy_status(model)['normalization']['operator_active'])
        owner.opaque_rms_norm = object()
        self.assertFalse(policy.policy_status(model)['normalization']['operator_active'])
        owner.opaque_rms_norm = opaque
        owner.rms_norm = object()
        self.assertFalse(policy.policy_status(model)['normalization']['operator_active'])


if __name__ == '__main__':
    unittest.main()
