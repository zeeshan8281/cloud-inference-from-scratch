"""CPU control-flow/contract checks; these do not validate CUDA execution."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/cloud_engine/batch_invariant_linear.py'


def definitions():
    tree = ast.parse(SOURCE.read_text())
    contract = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_head_contract')
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fixed_tile_head' and n.decorator_list)
    body.decorator_list = []
    body.returns = None
    for arg in body.args.args:
        arg.annotation = None
    return contract, body


class EmptyHeadTests(unittest.TestCase):
    def test_empty_validates_and_returns_without_kernel_launch(self):
        allocations, launches = [], []
        def empty(shape, **kwargs):
            allocations.append((shape, kwargs))
            return SimpleNamespace(shape=shape, **kwargs)
        def wrap(kernel):
            launches.append(kernel)
            raise AssertionError('zero-row head must not launch')
        env = {'torch': SimpleNamespace(empty=empty, float32='torch.float32', library=SimpleNamespace(wrap_triton=wrap))}
        exec(compile(ast.Module(body=list(definitions()), type_ignores=[]), str(SOURCE), 'exec'), env)
        def tensor(shape, dtype='torch.float16', device='cuda:0'):
            return SimpleNamespace(shape=shape, dtype=dtype, device=device, stride=lambda: (2048, 1))
        result = env['fixed_tile_head'](tensor((0, 2048)), tensor((151936, 2048)))
        self.assertEqual(result.shape, (0, 151936))
        self.assertEqual(result.dtype, 'torch.float32')
        self.assertEqual(len(allocations), 1)
        self.assertEqual(launches, [])
        for bad in (tensor((0, 2047)), tensor((0, 2048), dtype='torch.float32'), tensor((0, 2048), device='cpu')):
            with self.assertRaises(ValueError):
                env['fixed_tile_head'](bad, tensor((151936, 2048)))
        self.assertEqual(len(allocations), 1)

    def test_partial_prefill_model_routes_empty_selection_to_head(self):
        model_path = ROOT / 'src/cloud_engine/model.py'
        tree = ast.parse(model_path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(isinstance(x, ast.FunctionDef) and x.name == 'project_logits' for x in n.body))
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        forward.returns = None
        for arg in forward.args.args:
            arg.annotation = None
        selected = SimpleNamespace(shape=(0, 2048))
        hidden = SimpleNamespace(index_select=lambda dim, indices: selected)
        observed = []
        model = SimpleNamespace(_ensure_rope=lambda device: None, embed_tokens=lambda ids: hidden,
            layers=[], norm=lambda value: value, project_logits=lambda value: observed.append(value))
        env = {'PackedContext': type('PackedContext', (), {}), '_torch': SimpleNamespace(arange=lambda *args, **kwargs: None)}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(model_path), 'exec'), env)
        env['forward'](model, SimpleNamespace(device='cuda:0', shape=(256,)), None, logit_indices=[])
        self.assertEqual(observed, [selected])


if __name__ == '__main__':
    unittest.main()
