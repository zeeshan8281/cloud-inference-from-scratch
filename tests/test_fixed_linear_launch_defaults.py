"""CPU launch-contract regression for Torch2.7 TTIR mutation analysis, not CUDA validation."""
import ast
import unittest
from pathlib import Path


def launch_arguments(source):
    tree = ast.parse(source)
    kernel = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name == '_fixed_qkv')
    names = [arg.arg for arg in kernel.args.args]
    launches = []
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Subscript):
            continue
        wrapped = node.func.value
        if (isinstance(wrapped,ast.Call) and wrapped.args and isinstance(wrapped.args[0],ast.Name)
                and wrapped.args[0].id == '_fixed_qkv'):
            supplied = set(names[:len(node.args)]) | {kw.arg for kw in node.keywords if kw.arg in names}
            launches.append((node, set(names)-supplied))
    return launches


class FixedLinearLaunchDefaultsTests(unittest.TestCase):
    def test_all_native_parameters_explicit_for_ttir_analysis(self):
        source = (Path(__file__).resolve().parents[1]/'src/cloud_engine/batch_invariant_linear.py').read_text()
        launches = launch_arguments(source)
        self.assertEqual(len(launches),2)
        for node,missing in launches:
            self.assertEqual(missing,set())
            keywords = {kw.arg:kw.value for kw in node.keywords}
            self.assertIs(keywords['OUTPUT_FP32'].value,False)
        # Recreate the previous omitted default: strict Torch TTIR argument coverage fails.
        previous = source.replace('OUTPUT_FP32=False, ','')
        self.assertTrue(all('OUTPUT_FP32' in missing for _,missing in launch_arguments(previous)))


if __name__ == '__main__':
    unittest.main()
