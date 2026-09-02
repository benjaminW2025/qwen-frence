"""CPU-only contracts for benchmark backend execution modes."""

import inspect
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401

from benchmark_backends import (
    SchedulerBackend,
    SerialEngineBackend,
    scheduler_backend_flags,
)


class BackendModeTests(unittest.TestCase):
    def test_custom_kernel_modes_are_fully_eager(self):
        self.assertEqual(scheduler_backend_flags("custom-kernels"), (False, True))
        self.assertEqual(
            scheduler_backend_flags("regime-dispatched"), (False, True)
        )

    def test_cuda_graphs_are_an_explicit_legacy_backend(self):
        self.assertEqual(
            scheduler_backend_flags("bucketed-cuda-graphs"), (True, False)
        )
        self.assertEqual(
            scheduler_backend_flags("continuous-batching"), (False, False)
        )

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            scheduler_backend_flags("unknown")

    def test_iteration_telemetry_validation_belongs_to_scheduler_backend(self):
        serial_run = inspect.getsource(SerialEngineBackend.run)
        scheduler_run = inspect.getsource(SchedulerBackend.run)
        self.assertNotIn("iteration_fields", serial_run)
        self.assertNotIn("scheduler.iteration_", serial_run)
        self.assertIn("iteration_fields", scheduler_run)
        self.assertIn("scheduler.iteration_kinds", scheduler_run)

    def test_native_decode_does_not_materialize_packed_rope_layouts(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "engine"
            / "graph"
            / "paged_graph_decoder.py"
        ).read_text()
        self.assertNotIn("packed_q =", source)
        self.assertNotIn("apply_packed_rope_kv_write", source)
        self.assertIn("adapting\n        # decode required three materializing", source)

    def test_adaptive_decode_is_resolved_once_and_only_for_long_contexts(self):
        root = Path(__file__).resolve().parents[2]
        dispatch = (root / "engine" / "kvcache" / "paged_decode_attention.py").read_text()
        graph = (root / "engine" / "graph" / "paged_graph_decoder.py").read_text()
        mixed = (root / "engine" / "model_runner" / "mixed_batch.py").read_text()
        self.assertIn("MIN_ADAPTIVE_DECODE_CONTEXT_LENGTH = 1024", dispatch)
        self.assertIn("effective_decode_attention_policy =", graph)
        self.assertIn("effective_decode_attention_policy =", mixed)
        self.assertLess(
            graph.index("effective_decode_attention_policy ="),
            graph.index("for i, layer in enumerate(model.layers)"),
        )
        self.assertLess(
            mixed.index("effective_decode_attention_policy ="),
            mixed.index("for layer_index, layer in enumerate(model.layers)"),
        )


if __name__ == "__main__":
    unittest.main()
