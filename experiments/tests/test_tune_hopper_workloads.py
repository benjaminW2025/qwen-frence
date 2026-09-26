"""CPU contracts for the bounded H100 tuner; not CUDA-kernel validation."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import json
import tempfile

from experiments.decode import tune_hopper_workloads as tune


class WorkloadTuningTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = tune.make_plan(SimpleNamespace(
            suite_dir=tune.ROOT / 'experiments/results/full-checkpoint-20260916T033540Z', seed=20260914))

    def test_actual_eight_cells_are_covered_without_full_shape_sweep(self):
        self.assertEqual(len(self.plan['shapes']), 8)
        self.assertLessEqual(len(self.plan['cases']), 40)
        self.assertEqual(len({c['id'] for c in self.plan['cases']}), len(self.plan['cases']))
        for shape in self.plan['shapes']:
            for kind in ('prefill', 'decode', 'mixed'):
                tags = [tag for tag in self.plan['coverage'] if tag.startswith(shape + '/') and tag.endswith('/' + kind)]
                self.assertTrue(tags, (shape, kind))
        for case in self.plan['cases']:
            self.assertEqual(len(case['queries']), len(case['lengths']))
            self.assertTrue(all(0 < q <= c for q, c in zip(case['queries'], case['lengths'])))
            self.assertLessEqual(sum(case['queries']), 2112)

    def test_search_only_exposes_implemented_controls(self):
        for case in self.plan['cases']:
            configs = tune.configs(case)
            self.assertLessEqual(len(configs), 10)
            for config in configs:
                self.assertEqual(set(config), {'split_k', 'overlap_qk', 'tile_n', 'register_pv', 'compact'})
                self.assertTrue(1 <= config['split_k'] <= 64)

    def test_architecture_search_and_single_change_attribution(self):
        for kind in ('decode', 'prefill', 'mixed'):
            case = dict(kind=kind, queries=[1, 17], lengths=[128, 64])
            control = tune.configs(case)[0]
            grid = tune.architecture_configs(case, control)
            self.assertEqual(len(grid), 8 if kind == 'mixed' else 4)
            for candidate in grid:
                self.assertEqual(candidate['split_k'], control['split_k'])
                self.assertEqual(candidate['overlap_qk'], control['overlap_qk'])
            for role, candidate in tune.evaluation_configs(case, control, grid[-1]):
                if role.endswith('_only'):
                    changed = {k for k in control if candidate[k] != control[k]}
                    self.assertEqual(changed, {dict(register_pv_only='register_pv',
                        tile_128_only='tile_n', compact_only='compact')[role]})

    def test_fast_incorrect_configuration_cannot_win(self):
        rows = [dict(config={'id': 'bad'}, correctness_error='mismatch', cold={'median_ms': .1}),
                dict(config={'id': 'good'}, correctness_error=None, cold={'median_ms': 1.})]
        self.assertEqual(tune.best_passing(rows), {'id': 'good'})
        with self.assertRaisesRegex(ValueError, 'no correct'):
            tune.best_passing(rows[:1])

    def test_fixture_seeds_are_reproducible_and_evaluation_differs(self):
        import torch
        case = dict(queries=[1, 3], lengths=[17, 5])
        with mock.patch.object(torch.Tensor, 'to', lambda self, *args, **kwargs: self):
            a, ha = tune.fixtures(case, 12)
            b, hb = tune.fixtures(case, 12)
            _, hc = tune.fixtures(case, 13)
        self.assertEqual(ha, hb)
        self.assertNotEqual(ha, hc)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(a[3].tolist(), [0, 1, 4])

    def test_report_does_not_enable_production_or_hide_a_slow_case(self):
        case = self.plan['cases'][0]
        plan = {**self.plan, 'cases': [case]}
        measure = lambda ms: {'median_ms': ms, 'samples_ms': [ms] * 3}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root)
            for arm in ('local', 'reference'):
                tune.write_json(root / f'{arm}-environment.json', dict(gpu='H100', memory=80, uuid='test'))
            row = dict(status='complete', fixture_sha256='same', correctness_passed=True,
                       selected_config={'split_k': 2, 'overlap_qk': True})
            baseline = dict(cold=measure(3.), warm=measure(3.))
            selected = dict(cold=measure(2.), warm=measure(2.))
            evaluation = [dict(baseline, role='baseline'),
                          dict(cold=measure(2.5), warm=measure(2.5), role='tuned_control'),
                          dict(selected, role='register_pv_only', correctness_error=None),
                          dict(selected, role='selected')]
            tune.write_json(root / 'local' / f'{case["id"]}.json', dict(row, evaluation=evaluation))
            tune.write_json(root / 'reference' / f'{case["id"]}.json', dict(row, evaluation=[dict(cold=measure(1.), warm=measure(1.))]))
            tune.analyze(args, plan)
            result = json.loads((root / 'summary.json').read_text())
            self.assertFalse(result['parity_within_5_percent'])
            self.assertFalse(result['full_model_qualified'])
            self.assertEqual(result['rows'][0]['cold_tuning_speedup'], 1.5)
            self.assertEqual(result['rows'][0]['vs_fa3']['warm'], .5)
            self.assertEqual(result['rows'][0]['interventions_vs_tuned_control']
                             ['register_pv_only']['speedup']['cold'], 1.25)
            self.assertFalse(json.loads((root / 'candidate-dispatch.json').read_text())['production_enabled'])
