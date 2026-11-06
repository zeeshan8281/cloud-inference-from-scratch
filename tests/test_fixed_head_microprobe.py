import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


micro = load('head_micro', ROOT / 'experiments/fixed_head_microprobe.py')
kernel = load('head_kernel', ROOT / 'src/cloud_engine/batch_invariant_linear.py')


class HeadContractTests(unittest.TestCase):
    def test_actual_layouts_and_endpoints(self):
        states = [{'M': 5, 'projection_row': 2}, {'M': 1, 'projection_row': 0}]
        actual = set(micro.replay_layouts(states))
        self.assertIn((5, 2), actual)
        for m in (1, 5, 8, 16, 32):
            self.assertTrue({(m, 0), (m, m-1)} <= actual)
        with self.assertRaises(ValueError):
            micro.replay_layouts([{'M': 8, 'projection_row': 2}])

    def test_head_precision_and_scope(self):
        args = ((5, 2048), (151936, 2048), ((2048, 1), (2048, 1)),
                ('torch.float16',)*2, ('cuda:0',)*2)
        kernel.validate_head_contract(*args)
        self.assertEqual(kernel.head_kernel_policy()['output_dtype'], 'torch.float32')
        self.assertEqual(kernel.kernel_policy()['output_dtype'], 'torch.float16')
        for index, invalid in ((1, (152064, 2048)), (2, ((2560, 1), (2048, 1))),
                               (3, ('torch.float32',)*2), (4, ('cpu',)*2)):
            changed = list(args)
            changed[index] = invalid
            with self.assertRaises(ValueError):
                kernel.validate_head_contract(*changed)

    def test_old_linear_scope_does_not_accept_head(self):
        with self.assertRaises(ValueError):
            kernel.validate_linear_contract((5, 2048), (151936, 2048), None,
                ((2048, 1), (2048, 1)), ('torch.float16',)*2, ('cuda:0',)*2)


if __name__ == '__main__':
    unittest.main()
