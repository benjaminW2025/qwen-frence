"""CPU tests for joint-sweep planning, matched comparisons, and result artifacts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'decode'))
from benchmark_decode_joint_sweep import build_parser, config_key, parse_axis, resolve_plan, summarize_shape, write_results


class JointSweepTests(unittest.TestCase):
    def args(self, *extra):
        return build_parser().parse_args(['--batch-sizes', '16', '--context-lengths', '16384', *extra])

    def test_default_axes_and_group_specific_auto_values(self):
        shape = resolve_plan(self.args(), 132)[0]
        self.assertEqual(shape['auto_k_by_head_size'], {1: 3, 2: 6, 3: 9, 6: 17})
        keys = [config_key(c) for c in shape['configs']]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 4 * 10 * 3)
        # Every auto K is also evaluated on every other head size.
        for h in (1, 2, 3, 6):
            for k in (1, 2, 3, 4, 6, 8, 9, 16, 17, 32):
                for s in (1, 2, 3):
                    self.assertIn((h, k, s), keys)

    def test_subset_adds_controls_and_deduplicates_auto(self):
        shape = resolve_plan(self.args('--heads-per-program', '6', '--k-splits', '1,2,2,auto',
                                       '--pipeline-stages', '3', '--context-lengths', '64'), 132)[0]
        keys = {config_key(c) for c in shape['configs']}
        self.assertEqual(len(keys), 8)
        self.assertEqual(keys, {(h, k, s) for h in (1, 6) for k in (1, 2) for s in (1, 3)})
        for c in shape['configs']:
            self.assertEqual(c['pipelined'], c['num_stages'] > 1)

    def test_explicit_empty_splits_are_retained(self):
        shape = resolve_plan(self.args('--context-lengths', '1', '--k-splits', '32'), None)[0]
        self.assertEqual({c['split_k'] for c in shape['configs']}, {1, 32})

    def test_invalid_axes_and_missing_auto_device_fail(self):
        for value in ('', '0', '-1', '1,,2', '1.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_axis(value)
        with self.assertRaises(ValueError):
            resolve_plan(self.args('--heads-per-program', '4'), 132)
        with self.assertRaises(ValueError):
            resolve_plan(self.args(), None)
        with self.assertRaises(ValueError):
            resolve_plan(self.args('--page-size', '15'), 132)
        with self.assertRaises(ValueError):
            resolve_plan(self.args('--repetitions', '0'), 132)

    def test_ratios_hold_other_axes_fixed_and_best_ungrouped_is_tuned(self):
        times = {(1, 1, 1): 100, (1, 1, 2): 90, (1, 2, 1): 50, (1, 2, 2): 40,
                 (2, 1, 1): 200, (2, 1, 2): 180, (2, 2, 1): 45, (2, 2, 2): 30}
        rows = [{'batch_size': 16, 'context_length': 512, 'heads_per_program': h,
                 'split_k': k, 'num_stages': s, 'median_ms': time}
                for (h, k, s), time in times.items()]
        summary = summarize_shape(rows, 120)
        row = next(r for r in rows if config_key(r) == (2, 2, 2))
        self.assertAlmostEqual(row['grouping_speedup_at_fixed_k_stages'], 40/30)
        self.assertEqual(row['splitk_speedup_at_fixed_heads_stages'], 180/30)
        self.assertEqual(row['pipeline_speedup_at_fixed_heads_k'], 45/30)
        self.assertEqual(row['speedup_vs_production'], 4)
        self.assertEqual(summary['best_heads_per_program'], 2)
        self.assertEqual(summary['best_ungrouped_k'], 2)
        self.assertEqual(summary['best_ungrouped_stages'], 2)
        self.assertAlmostEqual(summary['best_speedup_vs_best_ungrouped'], 40/30)

    def test_failure_and_raw_samples_are_preserved_in_json(self):
        payload = {'status': 'failed', 'failure': {'error': 'compile failed'},
                   'rows': [{'median_ms': 2, 'samples_ms': [1, 2, 3], 'kernel_resources': {'spills': 0}}],
                   'shape_summaries': [{'best_ms': 2}], 'incomplete_shape': {'rows': [{'median_ms': 3}]}}
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'joint-test'
            write_results(prefix, payload)
            self.assertEqual(json.loads(prefix.with_suffix('.json').read_text()), payload)
            self.assertEqual(prefix.with_suffix('.csv').read_text().strip(), 'median_ms\n2')
            self.assertTrue(Path(f'{prefix}-summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
