"""Synthetic host guard tests; do not constitute candidate GPU validation."""
from __future__ import annotations

import ast
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


def policy():
    return {"candidate": "fp32_biased_qkv", "modules": ["qkv"], "module_count": 1,
            "expected_module_count": 1, "calls": {"qkv": 4}, "wrapper_active": True,
            "module_metadata": [{"name": "qkv", "weight_dtype": "torch.float16", "bias_dtype": "torch.float16"}],
            "precision": {"allow_fp16_reduced_precision_reduction": False,
                          "allow_tf32": False, "float32_matmul_precision": "highest"}}


def payload():
    return {"process": {"returncode": 0}, "capture": {"complete": True, "error": None, "config": {"candidate": "fp32_biased_qkv"},
            "batch": {"unique": {"qkv_policy": policy()}}, "solo": []},
            "environment": {"candidate": "fp32_biased_qkv", "gpu_before": {"uuid": "fixture"}, "gpu_after": {"uuid": "fixture"},
                            "computational_source": {"sha256": "core"}, "candidate_sources": {"candidate": "hash"},
                            "model": {"revision": "fixture"}, "model_files_sha256": {"weights": "hash"},
                            "dependencies": ["fixture==1"]}}


def load(remote):
    nodes = [node for node in ast.parse((REPO / "qkv_candidate_app.py").read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in ("run", "validate_candidate_capture", "finish_gpu_metadata")]
    for node in nodes:
        node.decorator_list = []
    scope = {"Path": Path, "json": json, "__file__": str(REPO / "qkv_candidate_app.py"),
             "capture": SimpleNamespace(remote=remote), "candidate_sources": lambda root: {"candidate": "hash"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "candidate-host", "exec"), scope)
    return scope


class TestQKVCandidateApp(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_bytes((REPO / "experiments/fix-validation-smoke-01/manifest.json").read_bytes())
        patch = mock.patch("experiments.fix_holdout.computational_source", return_value={"sha256": "core"})
        patch.start()
        self.addCleanup(patch.stop)

    def test_overwrite_and_fresh_manifest_block_before_remote(self):
        remote = mock.Mock()
        run = load(remote)["run"]
        with self.assertRaisesRegex(SystemExit, "overwrite"):
            run(str(self.root), "missing")
        source = json.loads(self.manifest.read_text())
        source["split"] = "holdout"
        self.manifest.write_text(json.dumps(source))
        with self.assertRaisesRegex(SystemExit, "exposed"):
            run(str(self.root / "fresh"), str(self.manifest))
        remote.assert_not_called()

    def test_child_source_runtime_and_policy_failures_keep_raw(self):
        for failure in ("crash", "source", "candidate_source", "policy", "incomplete", "environment"):
            with self.subTest(failure=failure):
                child = payload()
                if failure == "crash":
                    child["process"]["returncode"] = 1
                elif failure == "source":
                    child["environment"]["computational_source"] = {"sha256": "wrong"}
                elif failure == "candidate_source":
                    child["environment"]["candidate_sources"] = {"candidate": "wrong"}
                elif failure == "environment":
                    child["environment"]["environment_error"] = "unit-only GPU metadata failure"
                    child["environment"]["gpu_after"] = {"uuid": None}
                elif failure == "incomplete":
                    child["capture"]["complete"] = False
                else:
                    child["capture"]["batch"]["unique"].pop("qkv_policy")
                remote = mock.Mock(return_value=child)
                destination = self.root / failure
                with self.assertRaises(SystemExit):
                    load(remote)["run"](str(destination), str(self.manifest))
                self.assertEqual(remote.call_count, 1)
                self.assertEqual(json.loads((destination / "resource_normalized-custom.json").read_text()), child)
        first, second = payload(), payload()
        second["environment"]["model_files_sha256"] = {"weights": "different"}
        remote = mock.Mock(side_effect=[first, second])
        with self.assertRaises(SystemExit):
            load(remote)["run"](str(self.root / "runtime"), str(self.manifest))
        self.assertEqual(remote.call_count, 2)

    def test_strict_summary_failure_stops_before_second_mode_without_bounds(self):
        remote = mock.Mock(side_effect=lambda *args: payload())
        destination = self.root / "strict-fail"
        with mock.patch("experiments.fix_holdout.summarize", return_value={"passes": False}) as summarize:
            with self.assertRaisesRegex(SystemExit, "exact full-window"):
                load(remote)["run"](str(destination), str(self.manifest))
        self.assertEqual(remote.call_count, 2)
        self.assertEqual(summarize.call_args.kwargs, {"partial": True})
        summary = json.loads((destination / "resource_normalized-summary.json").read_text())
        self.assertFalse(summary["gate_cleared"])
        self.assertIsNone(summary["calibration_maxima"])

    def test_actual_policy_rejects_zero_calls_wrong_precision_and_duplicate_modules(self):
        check = load(None)["validate_candidate_capture"]
        check(payload()["capture"])
        for field, value in [("calls", {"qkv": 0}), ("precision", {}),
                             ("modules", ["qkv", "qkv"]), ("candidate", "different"), ("wrapper_active", False), ("module_metadata", [])]:
            record = copy.deepcopy(payload()["capture"])
            record["batch"]["unique"]["qkv_policy"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(record)

    def test_success_is_only_four_child_exposed_candidate_no_bounds(self):
        remote = mock.Mock(side_effect=lambda *args: payload())
        with mock.patch("experiments.fix_holdout.summarize", return_value={"passes": True}), redirect_stdout(io.StringIO()):
            load(remote)["run"](str(self.root / "success"), str(self.manifest))
        self.assertEqual(remote.call_count, 4)
        self.assertFalse((self.root / "success" / "proposed-bounds.json").exists())


    def test_new_candidate_requires_actual_head_evidence_and_is_passed_to_remote(self):
        def child(implementation="custom"):
            result = payload()
            result["environment"]["candidate"] = "fp32_qkv_and_head"
            result["capture"]["config"] = {"candidate": "fp32_qkv_and_head", "implementation": implementation}
            status = result["capture"]["batch"]["unique"]["qkv_policy"]
            status["candidate"] = "fp32_qkv_and_head"
            status["head"] = {"enabled": True, "implementation": "custom_existing_fp32" if implementation == "custom" else "vllm_fp32_output_projection",
                              "module_count": 1, "expected_module_count": 1, "wrapper_active": True,
                              "weight_dtype": "torch.float16", "input_dtypes": ["torch.float16"],
                              "output_dtypes": ["torch.float32"], "calls": 128, "emitted_rows": 128,
                              "all_finite": True, "autocast_enabled": False, "native_counterfactual_calls": 128,
                              "head_class": "VocabParallelEmbedding", "tied_input_embedding": True}
            return result
        check = load(None)["validate_candidate_capture"]
        for implementation in ("custom", "vllm"):
            valid = child(implementation)["capture"]
            check(valid, "fp32_qkv_and_head")
            for field, invalid in [("output_dtypes", ["torch.float16"]), ("weight_dtype", "torch.float32"),
                                   ("calls", 0), ("all_finite", False), ("wrapper_active", False)]:
                broken = copy.deepcopy(valid)
                broken["batch"]["unique"]["qkv_policy"]["head"][field] = invalid
                with self.subTest(implementation=implementation, field=field), self.assertRaises(ValueError):
                    check(broken, "fp32_qkv_and_head")
        valid = child("vllm")["capture"]
        for field, invalid in [("head_class", None), ("tied_input_embedding", None)]:
            broken = copy.deepcopy(valid)
            broken["batch"]["unique"]["qkv_policy"]["head"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                check(broken, "fp32_qkv_and_head")
        valid["batch"]["unique"]["qkv_policy"]["head"].update(
            head_class="ParallelLMHead", tied_input_embedding=False)
        check(valid, "fp32_qkv_and_head")
        remote = mock.Mock(side_effect=lambda implementation, *args: child(implementation))
        with mock.patch("experiments.fix_holdout.summarize", return_value={"passes": True}), redirect_stdout(io.StringIO()):
            load(remote)["run"](str(self.root / "head-candidate"), str(self.manifest), "fp32_qkv_and_head")
        self.assertEqual(remote.call_count, 4)
        self.assertTrue(all(call.args[3] == "fp32_qkv_and_head" for call in remote.call_args_list))
        remote = mock.Mock()
        with self.assertRaisesRegex(SystemExit, "unknown"):
            load(remote)["run"](str(self.root / "invalid"), str(self.manifest), "unknown")
        remote.assert_not_called()

    def test_fixed_tile_requires_installed_operator_and_fp32_head_without_fake_call_counts(self):
        from cloud_engine.batch_invariant_linear import kernel_policy
        candidate = "fixed_tile_qkv_fp32_head"
        item = payload()["capture"]
        item["config"] = {"candidate": candidate, "implementation": "custom"}
        status = item["batch"]["unique"]["qkv_policy"]
        status.update(candidate=candidate, calls={}, fixed_tile_policy={**kernel_policy(), "operator_active": True})
        status["head"] = {"enabled": True, "implementation": "custom_existing_fp32", "wrapper_active": True,
                          "module_count": 1, "expected_module_count": 1, "weight_dtype": "torch.float16",
                          "input_dtypes": ["torch.float16"], "output_dtypes": ["torch.float32"],
                          "calls": 128, "emitted_rows": 128, "all_finite": True, "autocast_enabled": False}
        check = load(None)["validate_candidate_capture"]
        check(item, candidate)
        for key, value in (("operator_active", False), ("block_m", 32), ("autotune", True), ("output_dtype", "torch.float32")):
            broken = copy.deepcopy(item)
            broken["batch"]["unique"]["qkv_policy"]["fixed_tile_policy"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check(broken, candidate)
        broken = copy.deepcopy(item)
        broken["batch"]["unique"]["qkv_policy"]["calls"] = {"qkv": 1}
        with self.assertRaises(ValueError):
            check(broken, candidate)


class GPUPostMetadataRetentionTests(unittest.TestCase):
    def test_post_query_failure_keeps_process_and_capture(self):
        retained = payload()
        before = copy.deepcopy(retained)
        def fails(version):
            raise RuntimeError('unit-only nvidia-smi failure')
        load(None)['finish_gpu_metadata'](retained['environment'], fails, 'fixture', 'finished')
        self.assertEqual(retained['process'], before['process'])
        self.assertEqual(retained['capture'], before['capture'])
        self.assertEqual(retained['environment']['gpu_after'], {'uuid': None})
        self.assertIn('nvidia-smi failure', retained['environment']['environment_error'])

class AllLinearPolicyGuardTests(unittest.TestCase):
    def test_coverage_optional_bias_and_actual_operator(self):
        from cloud_engine.batch_invariant_linear import linear_kernel_policy
        candidate = 'fixed_tile_linears_fp32_head'
        item = payload()['capture']
        item['config'] = {'candidate': candidate, 'implementation': 'custom'}
        state = item['batch']['unique']['qkv_policy']
        names = [f'layers.{index}.{path}' for index in range(36) for path in
                 ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                  'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
        state.update(candidate=candidate, modules=names, module_count=252, expected_module_count=252,
                     calls={}, fixed_tile_policy={**linear_kernel_policy(), 'operator_active': True},
                     module_metadata=[{'name': name, 'weight_dtype': 'torch.float16',
                                       'bias_dtype': 'torch.float16' if name.endswith(('q_proj','k_proj','v_proj')) else None} for name in names])
        state['head'] = {'enabled': True, 'implementation': 'custom_existing_fp32', 'wrapper_active': True,
                         'module_count': 1, 'expected_module_count': 1, 'weight_dtype': 'torch.float16',
                         'input_dtypes': ['torch.float16'], 'output_dtypes': ['torch.float32'],
                         'calls': 128, 'emitted_rows': 128, 'all_finite': True, 'autocast_enabled': False}
        check = load(None)['validate_candidate_capture']
        check(item, candidate)
        state['module_metadata'][0]['bias_dtype'] = None
        with self.assertRaises(ValueError):
            check(item, candidate)
        state['module_metadata'][0]['bias_dtype'] = 'torch.float16'
        state['module_metadata'][-1]['bias_dtype'] = 'torch.float16'
        with self.assertRaises(ValueError):
            check(item, candidate)
