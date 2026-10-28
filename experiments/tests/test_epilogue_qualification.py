"""Fail-closed numerical/performance qualification for the epilogue experiment."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch

from experiments.integration import benchmark_epilogue_interventions as ladder
from experiments.integration import ab_gemm_epilogues as ab
from experiments.prefill.benchmark_epilogue_primitives import compare


def row(speed):
    return {'speedup':speed,'numerics':{'finite':True,'non_tie_disagreements':0},'candidate_kv_finite':True}


class EpilogueQualification(unittest.TestCase):
    def test_gains_inside_noise_margin_not_selected(self):
        report = {'status':'pass','prefill':[row(1.01)],'decode':[row(1.01)]}
        self.assertEqual(ladder.select({'gate-up':report})['mode'],'off')

    def test_mixed_regression_blocks_prefill_admission(self):
        report = {'status':'pass','prefill':[row(1.1)],'mixed':[row(.99)],'decode':[row(1.1)]}
        self.assertEqual(ladder.select({'gate-up':report})['mode'],'decode')

    def test_logit_error_detects_changes_even_if_argmax_unchanged(self):
        result = ab.compare_logits(torch.tensor([[1.,2.]]),torch.tensor([[1.,20.]]),.25)
        self.assertEqual(result['greedy_agree'],1)
        self.assertGreater(result['relative_l2'],1)
        self.assertEqual(ab.merge([result])['relative_l2_max'],result['relative_l2'])

    def test_nonfinite_or_out_of_tolerance_primitive_fails(self):
        for candidate in (torch.tensor([float('nan')]),torch.tensor([3.])):
            with self.assertRaises(AssertionError):
                compare(torch.tensor([1.]),candidate,.001,.001)

    def test_failed_primitive_stops_before_model_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'new'
            with mock.patch('sys.argv',['benchmark','--output-dir',str(output)]), \
                 mock.patch.object(ladder.subprocess,'run',return_value=mock.Mock(returncode=1)) as run:
                with self.assertRaises(SystemExit):
                    ladder.main()
                self.assertEqual(run.call_count,1)
                self.assertTrue((output/'report.json').is_file())

    def test_precision_flags_restored_on_failure(self):
        backend = torch.backends.cuda.matmul
        original = backend.allow_fp16_reduced_precision_reduction
        with self.assertRaises(RuntimeError):
            with ab.strict_accumulation() as settings:
                self.assertTrue(all(value is False for value in settings.values()))
                raise RuntimeError('test')
        self.assertEqual(backend.allow_fp16_reduced_precision_reduction,original)
