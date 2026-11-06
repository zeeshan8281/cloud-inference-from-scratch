"""Synthetic control-flow checks; not GPU validation."""
import unittest
from types import SimpleNamespace

from experiments.admission_probe import drive


class Engine:
    def __init__(self):
        self.pending = []
        self.waves = []

    def add_request(self, request_id, prompt, params):
        self.pending.append(request_id)

    def step(self):
        wave, self.pending = self.pending, []
        self.waves.append(wave)
        return [SimpleNamespace(request_id=i, finished=True, outputs=[i]) for i in wave]

    def has_unfinished_requests(self):
        return bool(self.pending)


class AdmissionTests(unittest.TestCase):
    def test_only_admission_boundary_changes(self):
        for admission, waves in (("atomic", [["diag-0", "diag-1", "diag-2"]]),
                                 ("staged", [["diag-0"], ["diag-1", "diag-2"]])):
            engine = Engine()
            finished, _ = drive(engine, None, [[1], [2], [3]], admission)
            self.assertEqual(engine.waves, waves)
            self.assertEqual(set(finished), {"diag-0", "diag-1", "diag-2"})

    def test_invalid_admission_rejected_before_submission(self):
        engine = Engine()
        with self.assertRaises(ValueError):
            drive(engine, None, [[1]], "unknown")
        self.assertEqual(engine.pending, [])
