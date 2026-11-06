"""CPU arithmetic/scope checks; real CUDA accuracy remains a separate gate."""
import ast
import hashlib
import importlib.util
import struct
import unittest
from fractions import Fraction
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'src/cloud_engine/batch_invariant_linear.py'
spec = importlib.util.spec_from_file_location('head_reduction_contract', PATH)
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


class FixedHeadReductionTests(unittest.TestCase):
    def test_finite_fp16_product_extremes_are_exact_fp32(self):
        values = [0., 2**-24, 2**-14, 1+2**-10, 32.03125, 65504., -65504., -.333251953125]
        for a in values:
            for b in values:
                a, b = (struct.unpack('e', struct.pack('e', value))[0] for value in (a, b))
                fp32 = struct.unpack('f', struct.pack('f', a*b))[0]
                self.assertEqual(Fraction(fp32), Fraction(a)*Fraction(b))

    def test_head_metadata_and_storage_scope(self):
        policy = kernel.head_kernel_policy()
        self.assertEqual(policy['algorithm'], 'fp32_products_fp64_tree_sum')
        self.assertEqual(policy['accumulation_dtype'], 'torch.float64')
        self.assertEqual(policy['product_dtype'], 'torch.float32')
        self.assertEqual(policy['output_dtype'], 'torch.float32')
        self.assertEqual(policy['weight_dtype'], 'torch.float16')
        self.assertEqual((policy['block_m'], policy['block_n'], policy['block_k']), (1, 4, 2048))
        kernel.validate_head_contract((32, 2048), (151936, 2048), ((2048, 1), (2048, 1)),
                                      ('torch.float16',)*2, ('cuda:0',)*2)

    def test_preexisting_transformer_arithmetic_is_unchanged(self):
        # Function source hashes from the executed20 source (62827753...), not generated fixtures.
        expected = {'_fixed_qkv': 'f009e8ec504eb0ba3761a9ba15661a1e70191a49f34c631528366e5d24128ca2', 'batch_invariant_linear': '0972465619502ee28850a2ea2bf5e2b2f0a00cc64568257c951c0940f86246e4', 'fixed_tile_linear': 'ded3ad7c2987b478d4730deceff5db3fc331bb43fb03185cb2785e9eb578aa7b'}
        tree = ast.parse(PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in expected and node.decorator_list:
                digest = hashlib.sha256(ast.get_source_segment(PATH.read_text(), node).encode()).hexdigest()
                self.assertEqual(digest, expected.pop(node.name))
        self.assertEqual(expected, {})


if __name__ == '__main__':
    unittest.main()
