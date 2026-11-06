"""Known synthetic vectors test decoding only, not real GPU validation."""
import base64
import copy
import hashlib
import struct
import unittest

from experiments.custom_residual_analysis import compare_traces, decode_vector


def vector(values, dtype="torch.float16"):
    raw = struct.pack("<"+"e"*len(values), *values)
    return {"bytes_base64": base64.b64encode(raw).decode(), "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": dtype, "shape": [8, len(values)], "row_shape": [len(values)], "path": "output", "finite": True}


class ResidualAnalysisTests(unittest.TestCase):
    def test_known_half_and_bfloat_and_checksum(self):
        value = vector([1.0, -0.5])
        self.assertEqual(decode_vector(value), [1.0, -0.5])
        raw = struct.pack("<HH", 0x3f80, 0xbf00)
        bf = {**value, "dtype": "torch.bfloat16", "sha256": hashlib.sha256(raw).hexdigest(),
              "bytes_base64": base64.b64encode(raw).decode()}
        self.assertEqual(decode_vector(bf), [1.0, -0.5])
        value["sha256"] = "bad"
        with self.assertRaises(ValueError):
            decode_vector(value)

    def test_first_observed_row_and_history_integrity(self):
        left = [{"operator": "layer0", "target_position": 512, "target_input_token": 279,
                 "query_start": 512, "query_end": 513, "target_offset": 2, "packed_tokens": 8,
                 "output_tokens_before": [279], "vectors": [vector([1.0, -0.5])]}]
        right = copy.deepcopy(left)
        right[0]["vectors"] = [vector([1.0, -0.25])]
        result = compare_traces(left, right)["first_observed_unequal_row"]
        self.assertEqual(result["changed_elements"], 1)
        self.assertEqual(result["max_abs"], 0.25)
        self.assertEqual(result["mean_abs"], 0.125)
        right[0]["output_tokens_before"] = [1]
        with self.assertRaises(ValueError):
            compare_traces(left, right)
