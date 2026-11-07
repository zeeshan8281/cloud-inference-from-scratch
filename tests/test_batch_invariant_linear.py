"""CPU contracts only: actual kernel numerical/compile/graph checks run on L4."""
import unittest

from cloud_engine.batch_invariant_linear import kernel_policy, validate_contract


class FixedQKVContractTests(unittest.TestCase):
    def validate(self, m=16, n=256, strides=None, dtypes=None, devices=None):
        validate_contract((m, 2048), (n, 2048), (n,),
                          strides or ((2048, 1), (2048, 1), (1,)),
                          dtypes or ("torch.float16",) * 3,
                          devices or ("cuda:0",) * 3)

    def test_actual_layouts_share_fixed_output_tile_boundaries(self):
        policy = kernel_policy()
        for m in (1, 8, 16, 17, 128, 256, 2048):
            for n in (256, 2048, 2560):
                self.validate(m=m, n=n)
        self.assertEqual(2048 % policy['block_n'], 0)
        self.assertEqual(256 % policy['block_n'], 0)
        self.assertFalse(policy['split_k'])
        self.assertFalse(policy['autotune'])

    def test_rejects_unsupported_or_noncontiguous_inputs(self):
        cases = ({'n': 255}, {'m': 0}, {'strides': ((1, 16), (2048, 1), (1,))},
                 {'dtypes': ('torch.float32',) * 3}, {'devices': ('cpu',) * 3},
                 {'devices': ('cuda:0', 'cuda:1', 'cuda:0')})
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.validate(**case)


class FixedLinearContractTests(unittest.TestCase):
    def test_actual_unbiased_and_fused_shapes_and_bias_contract(self):
        from cloud_engine.batch_invariant_linear import validate_linear_contract
        for k,n in ((2048,2048),(11008,2048),(2048,11008),(2048,22016)):
            for bias in (None,(n,)):
                extra = ((1,),) if bias is not None else ()
                count = 3 if bias is not None else 2
                validate_linear_contract((17,k),(n,k),bias,((k,1),(k,1))+extra,
                                         ('torch.float16',)*count,('cuda:0',)*count)
        with self.assertRaises(ValueError):
            validate_linear_contract((1,11008),(11008,11008),None,
                                     ((11008,1),(11008,1)),('torch.float16',)*2,('cuda:0',)*2)

    def test_legacy_qkv_contract_stays_narrow(self):
        with self.assertRaises(ValueError):
            validate_contract((1,11008),(2048,11008),(2048,),((11008,1),(11008,1),(1,)),
                              ('torch.float16',)*3,('cuda:0',)*3)
