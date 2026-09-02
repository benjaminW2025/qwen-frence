"""CPU-only contracts for immutable scheduler iteration plans."""

import unittest

import _bootstrap  # noqa: F401

from iteration_plan import (
    IterationPlan,
    max_chunk_for_attention_pairs,
    prefill_attention_pairs,
)
from request_state import Request, Status


class SchedulerPlanTests(unittest.TestCase):
    def test_iteration_kinds_and_budget(self):
        mixed = IterationPlan((0, 1), (2, 3), (6, 2))
        self.assertEqual(mixed.kind, "mixed")
        self.assertEqual(mixed.decode_tokens, 2)
        self.assertEqual(mixed.prefill_tokens, 8)
        self.assertEqual(mixed.total_tokens, 10)
        mixed.validate_budget(10)
        with self.assertRaisesRegex(AssertionError, "budget"):
            mixed.validate_budget(9)

        self.assertEqual(IterationPlan((0,), (), ()).kind, "decode_only")
        self.assertEqual(IterationPlan((), (0,), (4,)).kind, "prefill_only")
        self.assertEqual(IterationPlan((), (), ()).kind, "empty")

    def test_invalid_plan_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal size"):
            IterationPlan((), (0,), ())
        with self.assertRaisesRegex(ValueError, "positive"):
            IterationPlan((), (0,), (0,))
        with self.assertRaisesRegex(ValueError, "cannot decode and prefill"):
            IterationPlan((0,), (0,), (1,))

    def test_request_progress_contract(self):
        request = Request(1, [10, 11, 12], max_tokens=2)
        request.status = Status.PREFILLING
        request.num_prompt_tokens_computed = 2
        self.assertEqual(request.remaining_prompt_tokens, 1)
        self.assertEqual(request.next_kv_position, 2)
        self.assertEqual(request.next_rope_position, 2)

    def test_prefill_attention_pair_cost(self):
        self.assertEqual(prefill_attention_pairs(0, 10), 55)
        self.assertEqual(prefill_attention_pairs(20, 3), 66)
        self.assertEqual(prefill_attention_pairs(4096, 2048), 10_486_784)
        self.assertEqual(max_chunk_for_attention_pairs(20, 20, 75), 3)


if __name__ == "__main__":
    unittest.main()
