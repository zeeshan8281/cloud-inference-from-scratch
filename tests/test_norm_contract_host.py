"""Host rejection checks only; synthetic result metadata is not GPU evidence."""
import ast
import base64
import copy
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT/'norm_contract_app.py').read_text())
NS = {'json':json,'hashlib':hashlib,'base64':base64}
exec(compile(ast.Module(body=[node for node in TREE.body if isinstance(node,ast.FunctionDef)
    and node.name == 'validate_payload'],type_ignores=[]),'<host>','exec'),NS)


class NormHostTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT/'input.json').read_text())
        state = self.fixture['states']['custom']
        self.expected = {'core':copy.deepcopy(state['source_manifest']['core'])}
        self.expected['core']['files']['src/cloud_engine/opaque_normalization.py'] = 'a'*64
        self.config = {'fixture_path':'/fixture','model_dir':'/model'}
        cases = []
        raw = base64.b64decode(state['output']['bytes_base64'])
        for kind in ('pure','residual'):
            for m in (0,1,16,256,512):
                checksum = hashlib.sha256(raw*m).hexdigest()
                cases.append({'kind':kind,'m':m,**dict.fromkeys(('eager_sha256','compiled_sha256','graph_sha256','expected_sha256'),checksum),
                    **dict.fromkeys(('eager_exact_and_aliases','compiled_fullgraph_exact_and_aliases','poisoned_graph_exact_and_aliases'),True)})
        result = {'schema':'normalization-contract-v1','complete':True,'error':None,'config':self.config,
            'fixture_sha256':'fixture','policy_id':'test-policy','gate_cleared':False,'cases':cases,
            'source_sha256':{'policy':self.expected['core']['files']['src/cloud_engine/numerical_policy.py'],
                             'attention':self.expected['core']['files']['src/cloud_engine/attention.py'],
                             'opaque':self.expected['core']['files']['src/cloud_engine/opaque_normalization.py']},
            'operator':'cloud_engine::common_rms_norm', 'installed_callable':'_norm_forward(opaque_normalization.opaque_rms_norm)',
            'actual_weight_files':dict(list(state['model_files_sha256'].items())[:1]),
            'precision':{'allow_fp16_reduced_precision_reduction':False,'allow_tf32':False,'float32_matmul_precision':'highest'}}
        env = {key:state[key] for key in ('model','model_files_sha256','dependencies')}
        env.update(source_manifest=self.expected,gpu_before={'uuid':'simulated'},gpu_after={'uuid':'simulated'})
        self.payload = {'capture':result,'environment':env,'process':{'returncode':0},'raw_checkpoint':json.dumps(result)}

    def check(self,payload):
        payload['raw_checkpoint'] = json.dumps(payload['capture'])
        module = types.ModuleType('cloud_engine.numerical_policy')
        module.POLICY_ID = 'test-policy'
        with patch.dict(sys.modules,{'cloud_engine.numerical_policy':module}):
            NS['validate_payload'](payload,self.config,self.expected,self.fixture,'fixture')

    def test_valid_synthetic_contract_and_missing_digest(self):
        self.check(self.payload)
        bad = copy.deepcopy(self.payload)
        for key in ('eager_sha256','compiled_sha256','graph_sha256','expected_sha256'):
            bad['capture']['cases'][0][key] = None
        with self.assertRaisesRegex(ValueError,'invalid normative digest'):
            self.check(bad)

    def test_raw_helper_binding_is_rejected(self):
        self.payload['capture']['installed_callable'] = '_norm_forward(attention.rms_norm)'
        with self.assertRaisesRegex(ValueError,'opaque normalization binding missing'):
            self.check(self.payload)

    def test_changed_imported_policy_rejected(self):
        self.payload['capture']['source_sha256']['policy'] = '0'*64
        with self.assertRaisesRegex(ValueError,'imported source mismatch'):
            self.check(self.payload)

    def test_graph_failure_and_missing_case_rejected(self):
        self.payload['capture']['cases'][1]['poisoned_graph_exact_and_aliases'] = False
        with self.assertRaisesRegex(ValueError,'contract failed'):
            self.check(self.payload)
        self.payload['capture']['cases'].pop(1)
        with self.assertRaisesRegex(ValueError,'missing normative cases'):
            self.check(self.payload)


if __name__ == '__main__':
    unittest.main()
