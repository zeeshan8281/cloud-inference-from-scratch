"""The operator console and pasteable architecture stay wired to the runtime."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class TestOperatorConsole(unittest.TestCase):
    def test_ui_contract(self) -> None:
        html = (ROOT / "ui/index.html").read_text()
        script = (ROOT / "ui/app.js").read_text()
        styles = (ROOT / "ui/styles.css").read_text()
        self.assertIn('id="run-form"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn("/v1/responses", script)
        self.assertIn("AbortController", script)
        self.assertIn('line.startsWith("data:")', script)
        self.assertIn('@import url("/tokens.css")', styles)
        self.assertIn("overflow-x: clip", styles)
        self.assertIn("51.104", html)
        self.assertIn("95.7% of vLLM", html)
        self.assertNotIn("49.281", html)

    def test_architecture_diagram_tracks_distributed_runtime(self) -> None:
        diagram = json.loads((ROOT / "system-architecture.excalidraw.json").read_text())
        elements = {element["id"]: element for element in diagram["elements"]}
        self.assertIn("redis-gate", elements)
        self.assertIn("a-redis-api", elements)
        self.assertIn("CUDA Graph 1/2/4/8/16", elements["scheduler-text"]["text"])
        self.assertIn("95.7% vLLM throughput", elements["metrics-text"]["text"])


if __name__ == "__main__":
    unittest.main()
