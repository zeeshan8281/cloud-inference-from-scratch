"""CPU scheduling/identity tests; actual attention and graph validation is separate."""
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

spec = importlib.util.spec_from_file_location('common_attention', Path(__file__).resolve().parents[1]/'src/cloud_engine/common_attention.py')
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def kernel_record():
    kernel = NS(_ragged_attention_direct=object(), _ragged_paged_attention_kernel=object(),
                ragged_attention_direct=object(), ragged_attention_prevalidated=object())
    return {'kernel':kernel, 'internal':kernel._ragged_attention_direct,
            'arithmetic':kernel._ragged_paged_attention_kernel}


class CommonAttentionTests(unittest.TestCase):
    def test_mixed_chunked_prefill_and_decode_positions(self):
        rows = common.metadata_rows([0,1,257,258], [514,512,513], [513,256,512], 258)
        self.assertEqual(rows, [(0,1,0,513),(1,257,1,256),(257,258,2,512)])
        self.assertEqual(common.metadata_rows([0], [], [], 0), [])
        for args in [([0,1],[4097],[4096],1), ([0,1],[512],[0],1),
                     ([0,0],[512],[512],0), ([1,2],[2],[1],2),
                     ([0,1],[1],[],1), ([0,1],[1],[0],2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                common.metadata_rows(*args)

    def test_custom_backend_fallback_owner_and_kernel_tamper(self):
        class Backend:
            def attend(self):
                pass
        backend = Backend()
        backend.allow_reference_fallback = False
        backend.num_heads, backend.num_kv_heads, backend.head_dim, backend.sm_scale = 16,2,128,1.0 / math.sqrt(128)
        record = {**kernel_record(), 'implementation':'custom', 'model':NS(attn_backend=backend),
                  'backend':backend, 'forward':Backend.attend}
        record['launch'] = record['kernel'].ragged_attention_direct
        self.assertTrue(common.status(record)['active'])
        backend.allow_reference_fallback = True
        self.assertFalse(common.status(record)['active'])
        backend.allow_reference_fallback = False
        backend.attend = Backend().attend
        self.assertFalse(common.status(record)['active'])
        del backend.attend
        record['kernel']._ragged_paged_attention_kernel = object()
        self.assertFalse(common.status(record)['active'])

    def test_actual_custom_backend_native_scale_and_owner(self):
        from cloud_engine.attention import RaggedTritonAttentionBackend
        backend = RaggedTritonAttentionBackend(None, 16, 2, 128)
        record = {**kernel_record(), 'implementation':'custom', 'model':NS(attn_backend=backend),
                  'backend':backend, 'forward':backend.attend.__func__}
        record['launch'] = record['kernel'].ragged_attention_direct
        self.assertEqual(backend.sm_scale, 1.0 / math.sqrt(128))
        self.assertNotEqual(backend.sm_scale, 128**-0.5)
        self.assertTrue(common.status(record)['active'])
        # Even an algebraically equivalent but non-native Python value is rejected.
        backend.sm_scale = 128**-0.5
        self.assertFalse(common.status(record)['active'])
        backend.sm_scale = 1.0 / math.sqrt(128)
        backend.allow_reference_fallback = True
        self.assertFalse(common.status(record)['active'])

    def test_vllm_bound_methods_and_metadata_helper_identity(self):
        class Builder:
            pass
        original_builder = common._BUILDER
        try:
            Builder.build = object()
            common._BUILDER = (Builder, Builder.build, common.prepare_metadata, common.metadata_rows)
            records = []
            for _ in range(36):
                impl = NS(num_heads=16,num_kv_heads=2,head_size=128,scale=128**-0.5,
                          vllm_flash_attn_version=2,alibi_slopes=None,sliding_window=(-1,-1),
                          logits_soft_cap=0,kv_sharing_target_layer_name=None,kv_cache_dtype='auto')
                _, function = common.bind_impl(impl)
                records.append((NS(impl=impl),impl,function,common.impl_contract(impl)))
            record = {**kernel_record(),'implementation':'vllm','records':records,
                      'forward':common._shared_forward,'adapter':common.validated_ragged_attention}
            record['launch'] = record['kernel'].ragged_attention_prevalidated
            self.assertTrue(common.status(record)['active'])
            self.assertTrue(common.status(record)['metadata_active'])
            records[0][1].scale = 1.0
            self.assertFalse(common.status(record)['active'])
            records[0][1].scale = 128**-0.5
            records[0][1].forward = records[1][1].forward
            self.assertFalse(common.status(record)['active'])
            Builder.build = object()
            self.assertFalse(common.status(record)['metadata_active'])
        finally:
            common._BUILDER = original_builder


if __name__ == '__main__':
    unittest.main()
