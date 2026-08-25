"""NVTX instrumentation must remain dependency-free and opt-in locally."""

import os
import unittest
from unittest.mock import patch

from cloud_engine.profiling import nvtx_range


class TestNvtxProfiling(unittest.TestCase):
    def test_disabled_range_is_a_noop_without_torch(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with nvtx_range("test.range"):
                value = 42
        self.assertEqual(value, 42)


if __name__ == "__main__":
    unittest.main()
