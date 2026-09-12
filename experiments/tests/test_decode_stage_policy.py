"""Protocol, leakage, resume, and statistics checks; no GPU needed."""
from argparse import Namespace
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'decode'))
from decode_stage_policy import (aggregate_cases, auto_k, evaluate, fit_tree, make_plan,
                                 paired_interval, predict, select_policy, work_features)
from benchmark_decode_stage_policy import analyze, atomic_json, ensure_manifest, load_records


class StageDesignTests(unittest.TestCase):
    def test_full_mechanism_has_exact_full_pages_and_controls(self):
        plan = make_plan()
        ids = [c['id'] for c in plan]
        self.assertEqual(len(ids), len(set(ids)))
        for case in plan:
            self.assertLessEqual(max(case['lengths']), 32768)
            if case['suite'] != 'mechanism':
                continue
            k = case['target_k']
            self.assertEqual(case['context'], case['target_pages_per_program'] * k * 16)
            features = work_features(case, k, 132)
            self.assertEqual(features['pages_per_program_min'], case['target_pages_per_program'])
            self.assertEqual(features['pages_per_program_max'], case['target_pages_per_program'])
            self.assertEqual(case['actions'], [[k, s] for s in (1, 2, 3, 4)])

    def test_dispatch_suites_have_disjoint_shapes_and_common_actions(self):
        cases = [c for c in make_plan() if c['suite'] in ('train', 'validation', 'test')]
        keys = [(c['batch'], c['context']) for c in cases]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(c['actions'] == cases[0]['actions'] for c in cases))
        for c in cases:
            self.assertIn([1, 1], c['actions'])
            self.assertIn([auto_k(c['batch'], c['features']['max_pages'], 132), 1], c['actions'])
        self.assertTrue(any(c['context'] % 16 for c in cases if c['suite'] == 'test'))

    def test_constant_grid_and_footprint_cohort_exists(self):
        cohort = [c for c in make_plan() if c['suite'] == 'mechanism'
                  and c['target_pages_per_program'] == 8 and c['batch'] * c['target_k'] == 32]
        self.assertEqual({(c['batch'], c['target_k']) for c in cohort}, {(1, 32), (8, 4), (32, 1)})
        values = [work_features(c, c['target_k'], 132) for c in cohort]
        self.assertEqual(len({v['kv_bytes'] for v in values}), 1)
        self.assertEqual(len({v['launched_programs_per_sm'] for v in values}), 1)

    def test_ragged_features_account_for_empty_partitions(self):
        case = {'lengths': [1, 33], 'batch': 2}
        f = work_features(case, 4, 132)
        self.assertEqual(f['pages_per_program_min'], 0)
        self.assertEqual(f['pages_per_program_max'], 1)
        self.assertEqual(f['pages_per_program_mean'], .5)
        self.assertEqual(f['empty_program_fraction'], .5)
        self.assertEqual(f['active_programs'], 48)

    def test_intervals_use_trial_pairs_and_do_not_fabricate_confidence(self):
        result = paired_interval([1.25] * 5)
        self.assertAlmostEqual(result['ratio'], 1.25)
        self.assertEqual(result['ci95'], [1.25, 1.25])
        self.assertIsNone(paired_interval([1.1, 1.2])['ci95'])
        for ratios in ([], [0], [float('nan')]):
            with self.assertRaises(ValueError):
                paired_interval(ratios)


class PolicyTests(unittest.TestCase):
    def data(self):
        result = []
        for suite, batches in [('train', [1, 2, 3, 32, 64, 128]), ('validation', [4, 48]), ('test', [8, 96])]:
            for b in batches:
                result.append({'id': f'{suite}-{b}', 'suite': suite, 'features': {'batch': b, 'max_pages': 64},
                               'costs': {(1, 1): 10 if b < 16 else 1, (8, 3): 1 if b < 16 else 10}})
        return result

    def test_tree_learns_simple_threshold(self):
        data = [r for r in self.data() if r['suite'] == 'train']
        tree = fit_tree(data, 1)
        self.assertEqual(predict(tree, {'batch': 1, 'max_pages': 64}), (8, 3))
        self.assertEqual(predict(tree, {'batch': 128, 'max_pages': 64}), (1, 1))
        self.assertEqual(evaluate(data, lambda r: predict(tree, r['features']))['max_regret'], 1)

    def test_test_labels_cannot_change_selected_policy(self):
        data = self.data()
        original = select_policy(data)
        for row in data:
            if row['suite'] == 'test':
                row['costs'] = {(1, 1): 1e9, (8, 3): .00001}
        self.assertEqual(select_policy(data), original)

    def test_fixed_policy_wins_complexity_ties(self):
        data = self.data()
        for row in data:
            row['costs'] = {(1, 1): 1, (8, 3): 1}
        policy = select_policy(data)['selected']
        self.assertEqual(policy['depth'], 0)
        self.assertEqual(policy['tree'], {'action': [1, 1]})

    def test_incomplete_or_duplicate_data_rejected(self):
        c = {'id': 'x', 'suite': 'train', 'features': {'batch': 1, 'max_pages': 1}, 'actions': [[1, 1]]}
        r = {'case_id': 'x', 'action': [1, 1], 'trial': 0, 'cache': 'warm', 'role': 'full', 'samples_ms': [1, 2, 3]}
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            aggregate_cases([c], [r], 2, 'warm')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            aggregate_cases([c], [r, r], 1, 'warm')
        self.assertEqual(aggregate_cases([c], [r], 1, 'warm')[0]['costs'], {(1, 1): 2})

    def test_resume_rejects_changed_protocol_and_ignores_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            initial = ensure_manifest(root, {'trials': 5, 'source': 'v1'})
            self.assertEqual(ensure_manifest(root, {'trials': 5, 'source': 'v1'}), initial)
            with self.assertRaisesRegex(ValueError, 'Resume refused'):
                ensure_manifest(root, {'trials': 5, 'source': 'v2'})
            (root / 'trials').mkdir()
            (root / 'trials' / 'unfinished.json.tmp').write_text('{')
            self.assertEqual(load_records(root), [])

    def test_synthetic_end_to_end_analysis_and_small_trial_gate(self):
        # Smoke is a protocol exercise, never enough evidence for integration.
        plan = make_plan('smoke')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = ensure_manifest(root, {'plan': plan, 'trials': 1, 'seed': 3,
                                              'cache_modes': ['warm'], 'hardware': {'sms': 132}})
            (root / 'trials').mkdir()
            records = []
            for case in plan:
                for k, stages in case['actions']:
                    time = 10 / min(k, 4) / (1 + .1 * (stages - 1))
                    for role in ('full', 'partial'):
                        records.append({'case_id': case['id'], 'action': [k, stages], 'trial': 0,
                                        'cache': 'warm', 'role': role, 'median_ms': time, 'samples_ms': [time] * 3})
                records.append({'case_id': case['id'], 'action': 'production', 'trial': 0,
                                'cache': 'warm', 'role': 'full', 'median_ms': 12, 'samples_ms': [12] * 3})
            atomic_json(root / 'trials' / 'synthetic.json', {'status': 'complete', 'fingerprint': manifest['fingerprint'], 'records': records})
            with patch('builtins.print'):
                analyze(Namespace(output_dir=root, fit_cache='warm'))
            report = json.loads((root / 'policy-report.json').read_text())
            self.assertEqual(report['status'], 'needs_more_work')
            self.assertFalse(report['production_ready'])
            self.assertEqual(set(report['heldout']), {'test:warm', 'ragged:warm'})
            effects = json.loads((root / 'stage-effects.json').read_text())
            self.assertTrue(effects)
            self.assertTrue(all(r['ci95'] is None for r in effects))


if __name__ == '__main__':
    unittest.main()
