import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.quant.attention import oracle
from experiments.quant.key_diagnostics import attention, compare, diagnose, main


class KeyDiagnostics(unittest.TestCase):
    def probe(self):
        g = torch.Generator().manual_seed(92)
        return {"layer": 3,
                "q": torch.randn(2, 6, 3, 16, generator=g).half(),
                "k": torch.randn(2, 2, 19, 16, generator=g).half(),
                "v": torch.randn(2, 2, 19, 16, generator=g).half()}

    def test_reference_matches_gqa_oracle_and_identity(self):
        p = self.probe()
        scores, probabilities, out, visible = attention(p['q'], p['k'], p['v'])
        torch.testing.assert_close(out.half(), oracle(p['q'], p['k'], p['v']), atol=0, rtol=0)
        self.assertTrue((probabilities[..., ~visible] == 0).all())
        metrics = compare(p['q'], p['k'], p['k'], p['v'])
        self.assertEqual(metrics['output_relative_l2'], 0)
        self.assertEqual(metrics['score_max_abs'], 0)
        self.assertEqual(metrics['attention_argmax_agreement'], 1)

    def test_restoring_every_dimension_or_position_eliminates_error(self):
        p = self.probe()
        for scaling in ('token', 'head'):
            result = diagnose(p, scaling, (99,))
            self.assertTrue(result['decoder_matches_torch'])
            self.assertGreater(result['all_keys_quantized']['key_relative_l2'], 0)
            for head in result['heads']:
                for rescue in head['rescues']:
                    self.assertEqual(rescue['key_relative_l2'], 0)
                    self.assertEqual(rescue['output_relative_l2'], 0)

    def test_one_key_head_only_changes_matching_gqa_heads(self):
        p = self.probe()
        original = attention(p['q'], p['k'], p['v'])[2]
        changed = p['k'].clone()
        changed[:, 0, 0] *= 20
        out = attention(p['q'], changed, p['v'])[2]
        torch.testing.assert_close(original[:, 3:], out[:, 3:], atol=0, rtol=0)
        self.assertGreater(float((original[:, :3] - out[:, :3]).abs().max()), 0)

    def test_real_capture_cli_writes_ranked_finite_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'capture.pt', Path(directory)/'report.json'
            torch.save([self.probe()], source)
            main(['--input', str(source), '--output', str(output), '--rescue-counts', '1'])
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['ranking']), 4)
            errors = [row['output_relative_l2'] for row in report['ranking']]
            self.assertEqual(errors, sorted(errors, reverse=True))

    def test_invalid_capture_rejected(self):
        p = self.probe()
        p['q'][0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            diagnose(p)
