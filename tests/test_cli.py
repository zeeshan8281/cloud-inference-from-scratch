"""Tests for the zero-dependency deployed-engine CLI."""

import unittest

from cloud_engine.cli import iter_sse_events, request


class TestCli(unittest.TestCase):
    def test_sse_parser_ignores_event_lines_and_done(self) -> None:
        events = list(
            iter_sse_events(
                [
                    b"event: response.created\n",
                    b'data: {"type":"response.created","sequence_number":0}\n',
                    b"\n",
                    b'data: {"type":"response.output_text.delta","delta":"hi"}\n',
                    b"data: [DONE]\n",
                ]
            )
        )
        self.assertEqual([event["type"] for event in events], ["response.created", "response.output_text.delta"])
        self.assertEqual(events[-1]["delta"], "hi")

    def test_request_keeps_key_in_header(self) -> None:
        req = request("https://engine.example/", "/v1/models", api_key="secret")
        self.assertEqual(req.full_url, "https://engine.example/v1/models")
        self.assertEqual(req.headers["Authorization"], "Bearer secret")
        self.assertNotIn("secret", req.full_url)


if __name__ == "__main__":
    unittest.main()
