"""Reference processes must not import the local C++/Torch extension."""

from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import builtins
import sys

from experiments.integration import benchmark_current_8_vs_vllm as burst
from experiments.integration import benchmark_current_mixed_8_vs_vllm as mixed
from experiments.integration import benchmark_current_phases_vs_vllm as phases

ROOT = Path(__file__).resolve().parents[2]


class IndependentRoutingTests(TestCase):
    def args(self):
        return SimpleNamespace(suite_dir=ROOT / 'experiments/results/full-checkpoint-20260916T033540Z',
            output_dir=Path('fresh'), model='model', device='cuda:0', seed=20260914,
            warmups=1, repetitions=3, reuse_vllm_from=None, resume_commit=None,
            vllm_python='/separate/reference/python')

    def test_all_arms_route_to_the_requested_environment(self):
        args = self.args()
        for factory in (burst.forwarded, mixed.forward, phases.forwarded):
            self.assertEqual(factory(args, 'run-vllm', burst.SHAPES[0])[0], args.vllm_python)
            self.assertEqual(factory(args, 'run-local', burst.SHAPES[0])[0], sys.executable)
        for factory in (burst.mixed_forwarded, burst.phase_forwarded):
            command = factory(args, burst.SHAPES[0])
            self.assertEqual(command[command.index('--vllm-python') + 1], args.vllm_python)

    def test_reference_workload_is_identical_without_local_extension(self):
        args = self.args()
        original = builtins.__import__
        def guarded(name, *rest, **kwargs):
            if name == 'inference_engine_cpp':
                raise AssertionError('reference tried loading local C++ extension')
            return original(name, *rest, **kwargs)
        for shape in burst.SHAPES:
            local = mixed.mixed_plan(args, shape)
            with mock.patch('builtins.__import__', side_effect=guarded):
                reference = mixed.mixed_plan(args, shape, validate_schedule=False)
            for index in (0, 1, 2, 3, 5, 6):
                self.assertEqual(local[index], reference[index])
            self.assertIsNone(reference[4])

    def test_selected_local_flags_disclose_candidate(self):
        self.assertEqual(burst.ENGINE_FLAGS['decode_attention_policy'], 'flash')
        self.assertEqual(burst.ENGINE_FLAGS['mixed_attention_policy'], 'flash_varlen')
        self.assertIn('unqualified', burst.ENGINE_FLAGS['attention_implementation'])
