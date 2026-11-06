"""Synthetic analysis integrity tests, never GPU validation."""
import copy
import hashlib
import struct
import unittest

from experiments.precision_analysis import linear_comparison


def fixture():
    precise = [1.0007] + [0.0]*2047
    rounded = [struct.unpack("<e", struct.pack("<e", value))[0] for value in precise]
    input_sha = hashlib.sha256(struct.pack("<2e", 1.0, -2.0)).hexdigest()
    record = {"module": "model.layers.0.mlp.down_proj", "weight_dtype": "torch.float16", "weight_shape": [2048, 2],
              "input_values": [1.0, -2.0], "input_sha256": input_sha, "weight_sha256": "c"*64,
              "native_allow_fp16_reduced_precision_reduction": True,
              "restored_allow_fp16_reduced_precision_reduction": True,
              "native_output": rounded, "fp64_reference": precise, "fp64_reference_rounded_fp16": rounded,
              "native_input_shape": [3585, 2], "native_target_offset": 512, "replays": []}
    for flag in (True, False):
        for rows, offset in ((4096, 1023), (3585, 512)):
            output = list(rounded)
            if flag and rows == 4096:
                output[0] = 1.0
            record["replays"].append({"allow_fp16_reduced_precision_reduction": flag,
                                      "input_shape": [rows, 2], "target_offset": offset,
                                      "input_sha256": input_sha, "weight_sha256": "c"*64, "output": output})
    return record


class PrecisionAnalysisTests(unittest.TestCase):
    def test_reports_observations_without_success_requirement(self):
        record = fixture()
        result = linear_comparison(record)
        self.assertFalse(result["shape_drift_reduced_true"]["exact_equal"])
        self.assertTrue(result["shape_drift_reduced_false"]["exact_equal"])
        record["replays"][3]["output"][1] = 0.25
        self.assertFalse(linear_comparison(record)["shape_drift_reduced_false"]["exact_equal"])

    def test_rejects_changed_frozen_state_or_incomplete_vectors(self):
        for field in ("input", "weight", "flag", "vector", "rounding"):
            record = copy.deepcopy(fixture())
            if field == "input":
                record["input_values"][0] = 2.0
            elif field == "weight":
                record["replays"][0]["weight_sha256"] = "d"*64
            elif field == "flag":
                record["restored_allow_fp16_reduced_precision_reduction"] = False
            elif field == "vector":
                record["replays"][0]["output"].pop()
            else:
                record["fp64_reference_rounded_fp16"][0] = 0
            with self.subTest(field=field), self.assertRaises(ValueError):
                linear_comparison(record)
