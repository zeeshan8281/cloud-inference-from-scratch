"""API validation, error mapping, auth, and SSE event tests — stdlib only.

The FastAPI integration layer runs in a CPU Modal container (PRD §13.2);
these cases cover the pure logic locally without any web dependency.
"""

import unittest

from cloud_engine.api import (
    ApiError,
    authorized,
    build_response_object,
    response_event_sequence,
    validate_payload,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"


class TestValidation(unittest.TestCase):
    def validate(self, payload, **kwargs):
        return validate_payload(
            payload,
            model_id=kwargs.pop("model_id", MODEL_ID),
            default_max_output_tokens=kwargs.pop("default_max_output_tokens", 256),
        )

    def expect_error(self, payload, code, status=None):
        with self.assertRaises(ApiError) as caught:
            self.validate(payload)
        error = caught.exception
        self.assertEqual(error.code, code)
        if status is not None:
            self.assertEqual(error.status, status)

    def test_happy_path_defaults(self) -> None:
        validated = self.validate({"model": MODEL_ID, "input": "Hello there."})
        self.assertEqual(validated.max_output_tokens, 256)
        self.assertEqual(validated.temperature, 0.0)
        self.assertFalse(validated.stream)

    def test_full_explicit_request(self) -> None:
        validated = self.validate(
            {
                "model": MODEL_ID,
                "input": "Write one sentence about paged attention.",
                "max_output_tokens": 64,
                "temperature": 0,
                "stream": True,
            }
        )
        self.assertTrue(validated.stream)
        self.assertEqual(validated.max_output_tokens, 64)

    def test_unknown_parameter_rejected(self) -> None:
        self.expect_error({"model": MODEL_ID, "input": "hi", "top_p": 1}, "unsupported_parameter")

    def test_array_input_rejected(self) -> None:
        self.expect_error(
            {"model": MODEL_ID, "input": [{"role": "user", "content": "hi"}]},
            "unsupported_input_type",
        )

    def test_empty_input_rejected(self) -> None:
        self.expect_error({"model": MODEL_ID, "input": ""}, "unsupported_input_type")
        self.expect_error({"model": MODEL_ID, "input": "   "}, "unsupported_input_type")

    def test_wrong_model_rejected(self) -> None:
        self.expect_error({"model": "gpt-4", "input": "hi"}, "invalid_model")

    def test_missing_model_rejected(self) -> None:
        self.expect_error({"input": "hi"}, "invalid_request")

    def test_temperature_must_be_zero(self) -> None:
        self.expect_error(
            {"model": MODEL_ID, "input": "hi", "temperature": 0.5}, "invalid_temperature"
        )
        # explicit zero accepted as int or float
        self.validate({"model": MODEL_ID, "input": "hi", "temperature": 0})

    def test_max_output_tokens_bounds(self) -> None:
        self.expect_error(
            {"model": MODEL_ID, "input": "hi", "max_output_tokens": 0}, "invalid_max_output_tokens"
        )
        self.expect_error(
            {"model": MODEL_ID, "input": "hi", "max_output_tokens": 257},
            "invalid_max_output_tokens",
        )
        self.expect_error(
            {"model": MODEL_ID, "input": "hi", "max_output_tokens": "32"},
            "invalid_max_output_tokens",
        )
        self.validate({"model": MODEL_ID, "input": "hi", "max_output_tokens": 1})
        self.validate({"model": MODEL_ID, "input": "hi", "max_output_tokens": 256})


class TestAuth(unittest.TestCase):
    def test_constant_time_compare_paths(self) -> None:
        key = "s3cret-value"
        self.assertTrue(authorized(f"Bearer {key}", key))
        # RFC 7235: auth scheme is case-insensitive
        self.assertTrue(authorized(f"bearer {key}", key))
        self.assertFalse(authorized("Bearer wrong", key))
        self.assertFalse(authorized(key, key))  # missing scheme
        self.assertFalse(authorized(None, key))
        self.assertFalse(authorized("Bearer", key))
        self.assertFalse(authorized("", key))


class TestResponseObjectShape(unittest.TestCase):
    def test_matches_prd_schema(self) -> None:
        payload = build_response_object(
            response_id="resp_abc",
            message_id="msg_def",
            model_id=MODEL_ID,
            created_at=1700000000,
            text="paged attention works",
            input_tokens=5,
            output_tokens=3,
            status="completed",
        )
        self.assertEqual(payload["object"], "response")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["created_at"], 1700000000)
        message = payload["output"][0]
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["role"], "assistant")
        content = message["content"][0]
        self.assertEqual(content["type"], "output_text")
        self.assertEqual(content["annotations"], [])
        usage = payload["usage"]
        self.assertEqual(usage["total_tokens"], 8)


class TestStreamingEventSequence(unittest.IsolatedAsyncioTestCase):
    async def collect_events(self, pieces):
        base = build_response_object(
            response_id="resp_s",
            message_id="msg_s",
            model_id=MODEL_ID,
            created_at=1700000000,
            status="in_progress",
            input_tokens=4,
            output_tokens=len(pieces),
        )

        async def deltas():
            for piece in pieces:
                yield piece

        collected = []
        async for event_type, payload in response_event_sequence(base, deltas()):
            collected.append((event_type, payload))
        return collected

    async def test_event_order_and_sequence_numbers(self) -> None:
        events = await self.collect_events(["paged ", "attention ", "rocks"])
        types = [t for t, _ in events]
        expected_prefix = ["response.created"]
        self.assertEqual(types[:1], expected_prefix)
        self.assertEqual(types[-2:], ["response.output_text.done", "response.completed"])
        delta_types = types[1:-2]
        self.assertTrue(delta_types and all(t == "response.output_text.delta" for t in delta_types))

        sequence_numbers = [payload["sequence_number"] for _, payload in events]
        self.assertEqual(sequence_numbers, sorted(sequence_numbers))
        self.assertEqual(len(set(sequence_numbers)), len(sequence_numbers), "must be unique")
        self.assertEqual(sequence_numbers[0], 0)
        for earlier, later in zip(sequence_numbers, sequence_numbers[1:], strict=False):
            self.assertGreater(later, earlier, "sequence numbers strictly increasing")

    async def test_concatenated_deltas_equal_final_text(self) -> None:
        pieces = ["The ", "KV cache ", "stores ", "attention."]
        events = await self.collect_events(pieces)
        deltas = "".join(
            payload["delta"] for t, payload in events if t == "response.output_text.delta"
        )
        done_payload = next(p for t, p in events if t == "response.output_text.done")
        completed_payload = next(p for t, p in events if t == "response.completed")
        self.assertEqual(deltas, "".join(pieces))
        self.assertEqual(done_payload["text"], "".join(pieces))
        self.assertEqual(completed_payload["response"]["output"][0]["content"][0]["text"], deltas)
        self.assertEqual(completed_payload["response"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
