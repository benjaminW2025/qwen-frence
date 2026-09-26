"""CPU contracts only: these tests do not validate TMA/WGMMA execution."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


class HopperCandidateContract(unittest.TestCase):
    def test_kv_check_ignores_unwritten_padding_but_detects_written_errors(self):
        import torch
        from experiments.integration.benchmark_integrated_graph import KVWriteCheck
        def pool():
            tensors = [torch.full((1, 16, 2, 128), float('nan'), dtype=torch.float16)
                       for _ in range(2)]
            for tensor in tensors:
                tensor[0, 3].fill_(1.)
            return SimpleNamespace(k_pool=tensors[:1], v_pool=tensors[1:])
        candidate, reference = pool(), pool()
        checker = KVWriteCheck(torch, candidate, reference)
        slots = torch.tensor([3])
        checker(slots, 0)
        self.assertTrue(checker.result()['passed'])
        candidate.k_pool[0][0, 3, 0, 0] = 2.
        checker(slots, 1)
        self.assertFalse(checker.result()['passed'])
        self.assertEqual(checker.result()['elements_outside_tolerance'], 1)
        self.assertEqual(checker.result()['first_failure']['callback'], 1)
        candidate.v_pool[0][0, 3, 0, 0] = float('nan')
        with self.assertRaisesRegex(AssertionError, 'nonfinite written v_pool'):
            checker(slots, 2)

    def test_wrapper_forwards_strided_query_but_rejects_cache_copies(self):
        import torch
        path = ROOT / 'custom_kernels/paged_flash_decode.py'
        spec = importlib.util.spec_from_file_location('hopper_wrapper_contract', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q = torch.zeros(2, 12, 256, dtype=torch.float16)[..., ::2]
        k = torch.zeros(2, 16, 2, 128, dtype=torch.float16)
        table = torch.zeros(2, 1, dtype=torch.int32)
        cu = torch.arange(3, dtype=torch.int32)
        lengths = torch.ones(2, dtype=torch.int32)
        extension = mock.Mock()
        with mock.patch.object(module, '_extension', return_value=extension) as load:
            module.flash_varlen(q, k, k, cu, table, lengths, max_query_len=1)
            forwarded = extension.forward.call_args.args
            self.assertIs(forwarded[0], q)
            self.assertIs(forwarded[1], k)
            self.assertIs(forwarded[2], k)
            load.reset_mock()
            strided_k = torch.zeros(2, 16, 2, 256, dtype=torch.float16)[..., ::2]
            with self.assertRaisesRegex(ValueError, 'must be contiguous'):
                module.flash_varlen(q, k, strided_k, cu, table, lengths, max_query_len=1)
            load.assert_not_called()

    def test_wrapper_has_no_external_attention_import(self):
        source = ROOT / 'custom_kernels/paged_flash_decode.py'
        names = []
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or '')
        self.assertIn('inference_hopper_attention', names)
        self.assertFalse(any('vllm' in name or 'flash_attn' in name for name in names))

    def test_qualification_does_not_promote_microbenchmark_to_full_model(self):
        path = ROOT / 'experiments/decode/qualify_flash_decode.py'
        spec = importlib.util.spec_from_file_location('hopper_qualification_contract', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows = [dict(kind=kind, batch=batch, context=context, correct=True,
                         fixture_sha256=f'{kind}-{batch}-{context}',
                         warm={'median_ms': 1.}, cold={'median_ms': 1.})
                    for kind, batch, context in module.SHAPES]
            payload = dict(status='complete', gpu={'name': 'mock'}, seed=1,
                           repetitions=3, vllm_version='0.30.0', rows=rows)
            for name in ('local', 'reference'):
                (directory / f'{name}.json').write_text(json.dumps(payload))
            with redirect_stdout(io.StringIO()):
                result = module.analyze(directory, 1.05)
            self.assertTrue(result['kernel_parity_passed'])
            self.assertFalse(result['full_model_qualified'])
            # One slow cell must fail the gate, even when the rest pass.
            payload['rows'][-1]['cold']['median_ms'] = 2.
            (directory / 'local.json').write_text(json.dumps(payload))
            with redirect_stdout(io.StringIO()):
                result = module.analyze(directory, 1.05)
            self.assertFalse(result['kernel_parity_passed'])
            payload['rows'][-1]['cold']['median_ms'] = 1.
            payload['rows'][0]['correct'] = False
            (directory / 'local.json').write_text(json.dumps(payload))
            with redirect_stdout(io.StringIO()):
                result = module.analyze(directory, 1.05)
            self.assertFalse(result['kernel_parity_passed'])
            payload['rows'][0]['fixture_sha256'] = 'different'
            (directory / 'local.json').write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'fixtures differ'):
                module.analyze(directory, 1.05)
