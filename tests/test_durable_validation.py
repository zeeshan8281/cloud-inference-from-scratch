"""CPU orchestration guards only; not a real calibration or GPU validation."""
import ast
import hashlib
import io
import json
import os
import tempfile
import unittest
import uuid
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

from experiments.parallel_validation import (
    open_durable_campaign,
    recover_durable,
    save_dispatch,
    wait_durable,
)

REPO = Path(__file__).resolve().parents[1]


def app_functions(filename, names, **scope):
    nodes = [node for node in ast.parse((REPO/filename).read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    scope.update(Path=Path, json=json, os=os, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)
    return scope


class DurableJournalTests(unittest.TestCase):
    def test_commit_binds_raw_before_verdict_and_refuses_duplicate_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            commits = []
            volume = NS(commit=lambda: commits.append(True))
            root, state, checkpoint = open_durable_campaign('a'*32, volume, temporary)
            path = root/'pair.json'
            path.write_text('{"partial":true}')
            checkpoint([path])
            self.assertEqual(state['artifacts']['pair.json'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(commits), 2)
            self.assertFalse(json.loads((root/'index.json').read_text())['complete'])
            with self.assertRaises(FileExistsError):
                open_durable_campaign('a'*32, volume, temporary)

    def test_recovery_checks_hashes_and_never_overwrites_unequal_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b'{"complete":false}'
            state = {'artifacts': {'pairs/pair.json': hashlib.sha256(raw).hexdigest()}, 'complete': False}
            files = {'id/index.json': json.dumps(state).encode(), 'id/pairs/pair.json': raw}
            volume = NS(read_file=lambda path: iter([files[path]]))
            recover_durable(volume, root, 'id')
            self.assertEqual((root/'pairs/pair.json').read_bytes(), raw)
            (root/'pairs/pair.json').write_bytes(b'preserved-original')
            with self.assertRaises(ValueError):
                recover_durable(volume, root, 'id')
            self.assertEqual((root/'pairs/pair.json').read_bytes(), b'preserved-original')

    def test_saved_id_survives_get_failure_and_original_error_is_not_masked(self):
        with tempfile.TemporaryDirectory() as temporary:
            call = NS(object_id='fc-existing', get=mock.Mock(side_effect=ConnectionError('disconnected')))
            dispatch = save_dispatch(temporary, 'id', call, 'ap-existing')
            with mock.patch('experiments.parallel_validation.recover_durable', side_effect=FileNotFoundError('queued')), \
                    redirect_stdout(io.StringIO()), self.assertRaisesRegex(ConnectionError, 'disconnected'):
                wait_durable(call, None, temporary, dispatch)
            self.assertEqual(json.loads((Path(temporary)/'dispatch.json').read_text()), dispatch)


class DetachedStageGuards(unittest.TestCase):
    def test_fresh_remote_claim_prevents_second_coordinator_execution(self):
        class Store(dict):
            def put(self, key, value, skip_if_exists=False):
                if skip_if_exists and key in self:
                    return False
                self[key] = value
                return True
        store = Store()
        executed = []
        source = {'sha256': 'source', 'files': {'fix_holdout_app.py': hashlib.sha256(b'actual-module').hexdigest()}}
        from experiments.fix_holdout import NAMESPACE, digest
        frozen = {'computational_source_sha256': 'source'}
        workload = {'split': 'holdout', 'namespace': NAMESPACE, 'workloads': {},
                    'workloads_sha256': digest({}), 'computational_source': source, 'bounds_sha256': digest(frozen)}
        inputs = {'split': 'holdout', 'workload_manifest': workload, 'source': source, 'frozen': frozen,
                  'source_commit': 'commit', 'calibration_commit': 'head', 'summary_reference': '/external/summary.json',
                  'frozen_input_lineage': {'execution_commit': 'head', 'manifest_repo_path': 'inputs.json', 'bounds_repo_path': 'limits.json'}}
        with tempfile.TemporaryDirectory() as temporary:
            volume = NS(commit=lambda: None)
            original = open_durable_campaign
            def journal(campaign_id, selected_volume):
                return original(campaign_id, selected_volume, temporary)
            ns = app_functions('fix_holdout_app.py', {'validation_coordinator'}, evidence_volume=volume,
                               modal=NS(Dict=NS(from_name=lambda *a, **k: store)),
                               execute_validation=lambda *args, **kwargs: executed.append((args, kwargs)))
            with mock.patch('experiments.parallel_validation.open_durable_campaign', journal), \
                    mock.patch('experiments.fix_holdout.computational_source', return_value=source), \
                    mock.patch('experiments.fix_holdout.validate_manifest'), mock.patch.dict(os.environ, SOURCE_COMMIT='commit'), \
                    mock.patch.object(Path, 'read_bytes', return_value=b'actual-module'):
                first = ns['validation_coordinator']('a'*32, inputs)
                second = ns['validation_coordinator']('b'*32, inputs)
            self.assertTrue(first['complete'])
            self.assertFalse(second['complete'])
            self.assertIn('already attempted', second['error'])
            self.assertEqual(len(executed), 1)

    def test_sentinel_cancel_service_failure_retains_collected_raw(self):
        for failed_pair in (False, True):
            with self.subTest(failed_pair=failed_pair):
                store = NS(get=mock.Mock(side_effect=[False, ConnectionError('cancel service unavailable')]),
                           put=mock.Mock(side_effect=ConnectionError('cancel signal unavailable')))
                raw = {'executed_source_hashes': {}, 'stop': {'kind': 'numerical_failure'} if failed_pair else None,
                       'children': {'custom': {'stdout': 'retained logs', 'partial_checkpoint': {'tokens': [1]}}}}
                ns = app_functions('sentinel_validation_app.py', {'pair'},
                                   modal=NS(Dict=NS(from_name=lambda *a, store=store, **k: store)),
                                   PINNED={'ragged_model': {'id': 'model', 'revision': 'revision'}}, volume=NS(commit=lambda: None))
                with mock.patch.dict('sys.modules', {'cloud_engine.weights': NS(ensure_weights_downloaded=lambda *a: '/model')}), \
                        mock.patch('experiments.sentinel_pilot.hash_file_tree', return_value={'weights': 'hash'}), \
                        mock.patch('experiments.sentinel_pilot.resolved_dependency_versions', return_value=[]), \
                        mock.patch('experiments.sentinel_runtime.run_pair', return_value=raw), \
                        mock.patch.object(Path, 'read_bytes', return_value=b'source'):
                    result = ns['pair']('resource_normalized', 1, 'campaign')
                self.assertEqual(result['children']['custom']['stdout'], 'retained logs')
                self.assertEqual(result['children']['custom']['partial_checkpoint']['tokens'], [1])
                self.assertIn('cancel signal unavailable', result['cancel_signal_error'])
                self.assertEqual(result['stop']['kind'], 'numerical_failure' if failed_pair else 'cancel_state_unresolved')

    def test_calibration_preflight_precedes_detach_and_id_precedes_wait(self):
        from experiments.fix_holdout import digest
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'out'
            events = []
            workload = {'workloads': {}, 'workloads_sha256': digest({}), 'computational_source': {'sha256': 'source'}}
            def preflight(*args):
                events.append('preflight')
                return {'split': 'calibration', 'source': {'sha256': 'source'}, 'source_commit': 'commit',
                        'calibration_commit': 'head', 'frozen': None, 'workload_manifest': None}
            def app_run(**kwargs):
                self.assertEqual(kwargs, {'detach': True})
                self.assertEqual(events, ['preflight'])
                events.append('detach')
                return nullcontext()
            call = NS(object_id='fc-created')
            def wait(actual_call, volume, output, dispatch):
                self.assertIs(actual_call, call)
                self.assertEqual(json.loads((destination/'dispatch.json').read_text())['call_id'], 'fc-created')
                self.assertEqual(events, ['preflight', 'detach', 'preflight'])
                return {'complete': True}
            ns = app_functions('fix_holdout_app.py', {'durable_validation'}, REPO=REPO, uuid=uuid,
                               validation_preflight=preflight, app=NS(run=app_run, app_id='ap-created'),
                               prepare=NS(remote=mock.Mock(return_value=workload)), evidence_volume='volume',
                               validation_coordinator=NS(spawn=mock.Mock(return_value=call)))
            with mock.patch('experiments.fix_holdout.validate_manifest'), \
                    mock.patch('experiments.parallel_validation.wait_durable', side_effect=wait):
                self.assertTrue(ns['durable_validation'](str(destination))['complete'])
            self.assertEqual(ns['validation_coordinator'].spawn.call_count, 1)

    def test_recover_branches_never_run_preflight_or_spawn(self):
        for filename, function in [('fix_holdout_app.py', 'durable_validation'), ('sentinel_validation_app.py', 'durable_pilot')]:
            with self.subTest(function=function), tempfile.TemporaryDirectory() as temporary:
                dispatch = {'call_id': 'fc-saved', 'campaign_id': 'saved'}
                (Path(temporary)/'dispatch.json').write_text(json.dumps(dispatch))
                call = object()
                ns = app_functions(filename, {function}, evidence_volume='volume',
                                   modal=NS(FunctionCall=NS(from_id=mock.Mock(return_value=call))))
                with mock.patch('experiments.parallel_validation.wait_durable', return_value={'complete': True}) as wait:
                    result = ns[function](temporary, recover=True)
                self.assertTrue(result['complete'])
                wait.assert_called_once_with(call, 'volume', Path(temporary).resolve(), dispatch)
