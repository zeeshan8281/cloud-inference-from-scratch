"""Paired parent control-flow only; mocked captures are not GPU validation."""
import ast
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).parents[1]


def paired_function(store, child, guard=lambda *args: []):
    node = next(node for node in ast.parse((REPO/'fix_holdout_app.py').read_text()).body
                if isinstance(node,ast.FunctionDef) and node.name=='capture_pair')
    node.decorator_list=[]
    namespace={'modal':SimpleNamespace(Dict=SimpleNamespace(from_name=lambda *args,**kwargs:store)), '_capture_child':child, '_child_hard_failures':guard}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<paired-test>','exec'),namespace)
    return namespace['capture_pair']


class Store(dict):
    def put(self,key,value):
        self[key]=value
        return True


def payload():
    return {'process':{'returncode':0},'capture':{'complete':True},'environment':{
        'computational_source':{'sha256':'source'},'gpu_before':{'uuid':'GPU-A'},'gpu_after':{'uuid':'GPU-A'}}}


class PairedCaptureTests(unittest.TestCase):
    def test_external_output_reference_does_not_require_repo_membership(self):
        node = next(node for node in ast.parse((REPO/'fix_holdout_app.py').read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name=='evidence_reference')
        namespace={'Path':Path,'REPO':REPO}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<reference-test>','exec'),namespace)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            repo=root/'clean-checkout'
            repo.mkdir()
            external=root/'raw-output'/'summary.json'
            self.assertEqual(namespace['evidence_reference'](external,repo),str(external.resolve()))
            self.assertEqual(namespace['evidence_reference'](repo/'summary.json',repo),'summary.json')
            self.assertEqual(list(repo.iterdir()),[])

    def test_sequential_children_and_immediate_failure_signal(self):
        for fault in (None,'first_failure','uuid_change','already_cancelled'):
            store=Store(campaign=fault=='already_cancelled')
            calls=[]
            def child(mode,implementation,workload,cancelled,calls=calls,fault=fault):
                self.assertFalse(cancelled())
                calls.append(implementation)
                value=copy.deepcopy(payload())
                if fault=='first_failure':
                    value['capture']['complete']=False
                if fault=='uuid_change' and implementation=='vllm':
                    value['environment']['gpu_after']['uuid']='GPU-B'
                return value
            with self.subTest(fault=fault),patch('experiments.fix_holdout.summarize',return_value={'passes':True}):
                result=paired_function(store,child)('resource_normalized',0,{},'campaign','source')
            if fault is None:
                self.assertEqual(calls,['custom','vllm'])
                self.assertIsNone(result['stop'])
                self.assertFalse(store['campaign'])
            else:
                self.assertIsNotNone(result['stop'])
                self.assertTrue(store['campaign'])
                self.assertEqual(calls,[] if fault=='already_cancelled' else ['custom'] if fault=='first_failure' else ['custom','vllm'])

    def test_complete_intraengine_mismatch_stops_before_reference_child(self):
        from tests.test_fix_holdout import capture
        node = next(node for node in ast.parse((REPO/'fix_holdout_app.py').read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name=='_child_hard_failures')
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<guard-test>','exec'),namespace)
        actual={'complete':True,'batch':{'unique':capture()},'solo':[{'unique':capture()}]}
        solo=actual['solo'][0]['unique']['records'][0]
        solo['output_tokens'][0]=1
        solo['steps'][0]['argmax']=1
        solo['steps'][0]['top_k'][1]=(1,0.0)
        store=Store(campaign=False)
        calls=[]
        def child(mode,implementation,workload,cancelled):
            calls.append(implementation)
            result=payload()
            result['capture']=actual
            return result
        workload={'concurrency':1,'input_token_ids':[[3,4]],'output_tokens':2}
        result=paired_function(store,child,namespace['_child_hard_failures'])('resource_normalized',0,workload,'campaign','source')
        self.assertEqual(calls,['custom'])
        self.assertEqual(result['stop'],'intraengine_hard_validation_failure')
        self.assertTrue(store['campaign'])
        self.assertTrue(any(item['reason']=='token_difference' for item in result['child_failures']['failures'][0]['failures']))
        actual['batch']['unique'].pop('graph_counters')
        failures=namespace['_child_hard_failures']('resource_normalized','custom',workload,actual,None)
        self.assertIn('graph evidence',failures[0]['reason'])

    def test_post_gpu_metadata_failure_keeps_child_checkpoint_and_logs(self):
        node = next(node for node in ast.parse((REPO/'fix_holdout_app.py').read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name=='_capture_child')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'child').mkdir()
            (root/'model').mkdir()
            (root/'config.json').write_text(json.dumps({'ragged_model':{'id':'fixture','revision':'pinned'},'kv_cache':{'bytes':16}}))
            def mapped_path(value):
                if value=='/root/engine_config.json':
                    return root/'config.json'
                return REPO/value.removeprefix('/root/') if isinstance(value,str) and value.startswith('/root/') else Path(value)
            namespace={'Path':mapped_path,'json':json,'hashlib':hashlib,'os':os,
                       'tempfile':SimpleNamespace(mkdtemp=lambda **kwargs:str(root/'child'))}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<child-test>','exec'),namespace)
            def process(command,**kwargs):
                Path(command[-1]).write_text(json.dumps({'complete':False,'partial_phases':{'retained':[1,2]}}))
                return {'returncode':0,'stdout':'retained stdout','stderr':'retained stderr','cancelled':False,'timeout':False}
            modules={'torch':SimpleNamespace(version=SimpleNamespace(cuda='fixture')),
                     'cloud_engine.weights':SimpleNamespace(ensure_weights_downloaded=lambda *args:root/'model')}
            with patch.dict(sys.modules,modules),patch('experiments.fix_holdout.computational_source',return_value={'sha256':'fixture'}), \
                    patch('experiments.sentinel_pilot.resolved_dependency_versions',return_value=['fixture']), \
                    patch('experiments.sentinel_pilot.gpu_state_snapshot',side_effect=[{'uuid':'GPU-A'},RuntimeError('fixture GPU failure')]), \
                    patch('experiments.parallel_validation.run_cancellable_process',side_effect=process):
                result=namespace['_capture_child']('resource_normalized','custom',{})
            self.assertEqual(result['capture']['partial_phases'],{'retained':[1,2]})
            self.assertEqual(result['process']['stdout'],'retained stdout')
            self.assertFalse(result['capture']['complete'])
            self.assertIn('fixture GPU failure',result['capture']['post_gpu_metadata_error'])

    def test_second_child_transport_error_retains_first_child(self):
        store=Store(campaign=False)
        def child(mode,implementation,workload,cancelled):
            if implementation=='vllm':
                raise RuntimeError('fixture error')
            return payload()
        result=paired_function(store,child)('resource_normalized',0,{},'campaign','source')
        self.assertEqual(set(result['children']),{'custom'})
        self.assertEqual(result['stop'],'paired_orchestration_error')
        self.assertTrue(store['campaign'])
