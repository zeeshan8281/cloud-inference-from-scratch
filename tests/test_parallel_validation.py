"""CPU orchestration/subprocess checks, not GPU validation."""
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from experiments.parallel_validation import campaign_cancelled, run_bounded, run_cancellable_process


class FailureDuringRetentionTests(unittest.TestCase):
    def test_failure_completed_during_retention_is_drained_before_refill(self):
        import threading
        from concurrent.futures import Future, ThreadPoolExecutor
        started, futures = [], {}
        release = threading.Event()
        original_submit = ThreadPoolExecutor.submit
        def submit(pool, function, job):
            future = original_submit(pool, function, job)
            futures[job] = future
            return future
        def job(number):
            started.append(number)
            if number == 1:
                release.wait(timeout=5)
                raise RuntimeError('failure while successful peer is being retained')
            return number
        def retain(number, payload):
            if number == 0:
                release.set()
                # Wait until failure is an actual completed Future, not just an event.
                while 1 not in futures:
                    release.wait(0.001)
                try:
                    Future.result(futures[1], timeout=5)
                except RuntimeError:
                    pass
            return not isinstance(payload, dict)
        from unittest import mock
        with mock.patch.object(ThreadPoolExecutor, 'submit', submit):
            result = run_bounded(range(3), job, retain, lambda: None, max_workers=2)
        self.assertEqual(started, [0, 1])
        self.assertEqual(result['submitted'], 2)
        self.assertTrue(result['stopped'])


class ParallelValidationTests(unittest.TestCase):
    def test_first_failure_stops_dispatch_and_retains_inflight(self):
        cancelled = threading.Event()
        barrier = threading.Barrier(4)
        retained = []
        def run(job):
            barrier.wait(timeout=5)
            if job == 0:
                return {"ok": False}
            cancelled.wait(timeout=5)
            return {"cancelled": cancelled.is_set()}
        def retain(job, payload):
            retained.append((job,payload))
            return payload.get("ok", False)
        result = run_bounded(range(20),run,retain,cancelled.set)
        self.assertEqual(result["submitted"],4)
        self.assertEqual(result["completed"],4)
        self.assertEqual({job for job,_ in retained},{0,1,2,3})
        self.assertTrue(result["stopped"])

    def test_success_processes_every_job_once(self):
        retained=[]
        def retain(job,payload):
            retained.append(job)
            return payload == job*2
        result=run_bounded(range(20),lambda job:job*2,retain,lambda:None)
        self.assertFalse(result["stopped"])
        self.assertEqual(sorted(retained),list(range(20)))

    def test_real_child_cancel_retains_logs_and_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            code="import time;from pathlib import Path;print('retained',flush=True);Path('partial.json').write_text('{}');time.sleep(30)"
            # Absolute checkpoint avoids changing global process working directory.
            code=code.replace("'partial.json'",repr(str(root/'partial.json')))
            result=run_cancellable_process([sys.executable,'-c',code],env=os.environ.copy(),
                stdout_path=root/'out',stderr_path=root/'err',cancelled=lambda:(root/'partial.json').exists(),
                timeout_seconds=5,poll_seconds=0.05)
            self.assertTrue(result['cancelled'])
            self.assertNotEqual(result['returncode'],0)
            self.assertIn('retained',result['stdout'])
            self.assertEqual((root/'partial.json').read_text(),'{}')

    def test_cancellation_kills_descendant_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            child_code=f"import time;from pathlib import Path;Path({str(root/'ready')!r}).write_text('ready');time.sleep(0.7);Path({str(root/'escaped')!r}).write_text('bad')"
            parent_code=f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child_code!r}]);time.sleep(30)"
            result=run_cancellable_process([sys.executable,'-c',parent_code],env=os.environ.copy(),
                stdout_path=root/'out',stderr_path=root/'err',cancelled=lambda:(root/'ready').exists(),
                timeout_seconds=5,poll_seconds=0.05)
            self.assertTrue(result['cancelled'])
            time.sleep(0.8)
            self.assertFalse((root/'escaped').exists())

    def test_missing_cancel_state_fails_closed_before_launch(self):
        with self.assertRaises(RuntimeError):
            campaign_cancelled({},'missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            started=time.monotonic()
            result=run_cancellable_process([sys.executable,'-c','raise RuntimeError("must not launch")'],env=os.environ.copy(),
                stdout_path=root/'out',stderr_path=root/'err',cancelled=lambda:campaign_cancelled({},'missing'),timeout_seconds=5)
            self.assertTrue(result['cancelled'])
            self.assertIsNone(result['returncode'])
            self.assertEqual(result['cancellation_reason'],'cancel_check_failed')
            self.assertLess(time.monotonic()-started,2)
