"""CPU checks for exact-shape GEMM tuning plans and fail-closed selection."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.prefill import tune_plain_gemms as tuner
import fused_gemm as fused


class PlainGemmTuning(unittest.TestCase):
    def test_inventory_has_all_five_weight_shapes_and_correct_head_rows(self):
        rows = tuner.inventory([8,1028,8192],[8,64],list(tuner.SHAPES))
        self.assertEqual(len(rows),14)
        self.assertEqual({row['m'] for row in rows if row['group']=='lm-head'},{8,64})
        for row in rows:
            self.assertEqual((row['n'],row['k'],row['bias']),tuner.SHAPES[row['group']])
            self.assertEqual(row['bias'],row['group']=='qkv')

    def test_default_plan_covers_graph_and_policy_boundary_sizes(self):
        plan = tuner.inventory(tuner.DEFAULT_ROWS,[1,8,64],list(tuner.SHAPES))
        expected = {1,8,64,65,128,1028,1408,1409,2048,8192,8196}
        for group in ('qkv','output','gate-up','down'):
            self.assertTrue(expected <= {row['m'] for row in plan if row['group']==group})
        self.assertEqual(len(plan),67)

    def test_only_correct_finite_positive_candidates_can_win(self):
        rows = [{'config':0,'status':'fail','training_median_ms':.001},
                {'config':1,'status':'pass','training_median_ms':float('nan')},
                {'config':2,'status':'pass','training_median_ms':0},
                {'config':3,'status':'pass','training_median_ms':.02},
                {'config':4,'status':'pass','training_median_ms':.03}]
        self.assertEqual(tuner.best_candidate(rows),3)
        self.assertIsNone(tuner.best_candidate(rows[:3]))

    def test_plan_requires_no_cuda_and_is_not_a_tuning_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'plan'
            with mock.patch.object(torch.cuda,'is_available',side_effect=AssertionError('no CUDA calls')):
                tuner.main(['--plan','--output-dir',str(output),'--rows','8','--head-rows','8'])
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'],'plan')
            self.assertEqual(len(report['problems']),5)
            self.assertEqual(report['cases'],[])

    def test_execution_without_gpu_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with mock.patch.object(torch.cuda,'is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'Hopper'):
                    tuner.main(['--output-dir',str(output),'--rows','8'])
            self.assertEqual(json.loads((output/'report.json').read_text())['status'],'fail')

    def profile(self):
        configs = [{'index':0}]
        report = {'status':'pass','dtype':'float16','source_sha256':'hash','gpu':'H100',
                  'torch':'version','cuda':'cuda','configs':configs,
                  'cases':[{'m':8,'n':2048,'k':1536,'bias':True,'status':'pass',
                            'best_cutlass_config':0,'candidates':[{'config':0,'status':'pass'}],
                            'confirmation':{'0':{'median_ms':.01},'cublas':{'median_ms':.02}}}]}
        return report,configs

    def test_tuning_table_rejects_stale_or_unconfirmed_data(self):
        report,configs = self.profile()
        args = ('hash',configs,'H100','version','cuda')
        self.assertEqual(fused.tuning_table(report,*args),{(8,2048,1536,True):0})
        report['source_sha256'] = 'old'
        with self.assertRaisesRegex(ValueError,'source_sha256'):
            fused.tuning_table(report,*args)
        report['source_sha256'] = 'hash'
        report['cases'][0]['confirmation'] = {}
        with self.assertRaisesRegex(ValueError,'unqualified'):
            fused.tuning_table(report,*args)

    def test_dispatch_requires_exact_shape_and_honors_explicit_configuration(self):
        usage = {}
        with mock.patch.object(fused,'_TUNED_CONFIGS',{(8,2048,1536,True):1}), \
             mock.patch.object(fused,'_TUNING_USAGE',usage):
            x,w = SimpleNamespace(shape=(8,1536)),SimpleNamespace(shape=(2048,1536))
            self.assertEqual(fused._select_config(x,w,-1,True),1)
            self.assertTrue(usage)
            self.assertEqual(fused._select_config(x,w,0,True),0)
            with self.assertRaisesRegex(ValueError,'missing exact shape'):
                fused._select_config(x,w,-1,False)

    def test_existing_results_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                tuner.main(['--plan','--output-dir',directory])
