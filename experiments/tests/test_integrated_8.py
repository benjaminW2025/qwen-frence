"""CPU contracts for the integrated eight-cell session, its gates and the divergence report."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from experiments.integration import ab_gemm_epilogues as ab
from experiments.integration import divergence_report as divergence
from experiments.integration import run_integrated_8 as session

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "experiments/results/full-checkpoint-20260916T033540Z"
ACCEPTED = ROOT / "experiments/results/attention-rerun-4XR8Oi/eight-fa3"


class BudgetChoice(unittest.TestCase):
    def test_smallest_budget_within_noise_of_the_best(self):
        rates = {"1024": 900.0, "2048": 1000.0, "4096": 1005.0, "8192": 1030.0}
        self.assertEqual(session.choose_budget(rates, .01), 8192)
        self.assertEqual(session.choose_budget(rates, .025), 4096)
        self.assertEqual(session.choose_budget(rates, .03), 2048)
        self.assertEqual(session.choose_budget({"2048": 1.0}, .01), 2048)


class Decisions(unittest.TestCase):
    def write(self, root, path, value):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    def args(self, **overrides):
        return SimpleNamespace(**{"gemm_epilogues": "auto", "graph_pool": "auto",
                                  "boundary_buffers": "auto", **overrides})

    def test_each_feature_needs_its_own_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision, _ = session.decide(root, self.args())
            self.assertEqual(decision, {"gemm_epilogues": "off", "graph_pool": "private",
                                        "boundary_buffers": False})
            self.write(root, "micro/layer/report.json", {"rows": [{"correct": True}]})
            self.write(root, "micro/model/report.json", {
                "status": "pass", "prefill": [{"speedup": 1.2}, {"speedup": 1.1}],
                "decode": [{"speedup": 1.05}, {"speedup": .97}]})
            self.write(root, "micro/graph-pool/report.json", {"status": "pass", "rows": []})
            self.write(root, "micro/boundary/report.json", {"status": "complete", "speedup_mixed": 1.02})
            decision, evidence = session.decide(root, self.args())
            # One decode case is slower, so only prefill is admitted.
            self.assertEqual(decision, {"gemm_epilogues": "prefill", "graph_pool": "shared",
                                        "boundary_buffers": True})
            self.assertEqual(evidence["gemm_epilogues"]["decode_speedups"], [1.05, .97])
            self.write(root, "micro/model/report.json", {
                "status": "fail", "prefill": [{"speedup": 1.2}], "decode": [{"speedup": 1.2}]})
            self.assertEqual(session.decide(root, self.args())[0]["gemm_epilogues"], "off")
            forced = session.decide(root, self.args(gemm_epilogues="all", boundary_buffers="off"))[0]
            self.assertEqual((forced["gemm_epilogues"], forced["boundary_buffers"]), ("all", False))

    def test_variant_cli_matches_the_harness_flags(self):
        cli = session.variant_cli({"graph_pool": "shared", "gemm_epilogues": "all", "boundary_buffers": True})
        self.assertEqual(cli, ["--prefill-graph-pool", "shared", "--gemm-epilogues", "all", "--boundary-buffers"])


class Divergence(unittest.TestCase):
    def test_first_divergence(self):
        self.assertIsNone(divergence.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(divergence.first_divergence([1, 2, 3], [1, 5, 3]), 1)
        self.assertEqual(divergence.first_divergence([1, 2], [1, 2, 3]), 2)

    @unittest.skipUnless((ACCEPTED / "summary.json").is_file(), "accepted baseline results not present")
    def test_accepted_baseline_has_four_distinct_divergences(self):
        args = SimpleNamespace(results_dir=ACCEPTED, against=None, suite_dir=SUITE,
                               shape_ids=list(divergence.SHAPES), scenarios=list(divergence.SCENARIOS))
        rows, cells = divergence.divergences(args)
        self.assertEqual(len(cells), 16)
        self.assertEqual(sum(cell["divergent"] for cell in cells.values()), 12)
        distinct = {(tuple(prompt), k, left[k], right[k]) for _, _, _, k, prompt, left, right in rows}
        self.assertEqual(len(distinct), 4)
        # Against itself there is nothing to explain.
        args.against = ACCEPTED
        rows, cells = divergence.divergences(args)
        self.assertEqual(rows, [])

    def test_summary_classifies_by_fp32_margin(self):
        scored = [{"scenario": "burst", "shape": "s", "fp32_margin": m, "left_token": 1, "right_token": 2,
                   "fp32_top_token": 1, "class": c}
                  for m, c in ((.1, "near_tie"), (-.9, "left_worse"))]
        args = SimpleNamespace(results_dir=Path("r"), against=None, reference="fp32", tie_margin=.25)
        report = divergence.summarize({"burst/s": {"divergent": 2}}, scored, args)
        self.assertEqual((report["cells"]["burst/s"]["near_tie"], report["cells"]["burst/s"]["left_worse"]), (1, 1))
        self.assertIn("clearly worse", report["verdict"])


class ModelAB(unittest.TestCase):
    def test_disagreements_are_scored_by_the_control_margin(self):
        control = torch.tensor([[5., 4.9, 0.], [3., 0., 2.], [1., 2., 0.]])
        candidate = torch.tensor([[4.8, 5., 0.], [2., 0., 3.], [1., 2., 0.]])
        row = ab.compare_logits(control, candidate, tie_margin=.25)
        self.assertEqual(row["greedy_agree"], 1)
        self.assertEqual([round(m, 4) for m in row["disagreement_margins"]], [.1, 1.])
        self.assertEqual(row["non_tie_disagreements"], 1)
        merged = ab.merge([row, ab.compare_logits(control, control, .25)])
        self.assertEqual((merged["rows"], merged["greedy_agree"], merged["non_tie_disagreements"]), (6, 4, 1))
        self.assertNotIn("_next", merged)

    def test_cases_parse(self):
        self.assertEqual(ab.parse_case("64x2048"), (64, 2048))
        with self.assertRaises(Exception):
            ab.parse_case("0x5")

    def test_layout_gives_disjoint_pages_and_skips_page_zero(self):
        layout = ab.Layout(torch, "cpu", 3, 40)
        self.assertEqual(layout.table.shape, (3, 3))
        self.assertEqual(int(layout.table.min()), 1)
        slots = layout.slots(torch, torch.tensor([0, 1, 2]), torch.tensor([0, 17, 39]))
        self.assertEqual(slots.tolist(), [16, 5 * 16 + 1, 9 * 16 + 7])


if __name__ == "__main__":
    unittest.main()
