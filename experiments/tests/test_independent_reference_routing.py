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
        return SimpleNamespace(attention='project', suite_dir=ROOT / 'experiments/results/full-checkpoint-20260916T033540Z',
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

    def test_fa3_mode_runs_fa3_in_every_phase_and_reaches_every_child(self):
        flags = burst.engine_flags('fa3')
        self.assertEqual(flags['decode_attention_policy'], 'fa3')
        self.assertEqual(flags['prefill_attention_policy'], 'fa3_varlen')
        self.assertEqual(flags['mixed_attention_policy'], 'fa3_varlen')
        self.assertIn('vllm', flags['attention_implementation'])
        self.assertNotEqual(flags, burst.engine_flags('project'))
        case = dict(max_running=8, lengths=[256], outputs=[128])
        for options in (burst.adapter_options(case, [2048], 'fa3'),
                        mixed.adapter_options(case, [2048], 'fa3')):
            # Pure prefill must not fall back to the Triton packed kernel.
            self.assertEqual(options['decode_attention_policy'], 'fa3')
            self.assertEqual(options['prefill_attention_policy'], 'fa3_varlen')
        args = SimpleNamespace(**{**vars(self.args()), 'attention': 'fa3'})
        commands = [burst.forwarded(args, 'run-local', burst.SHAPES[0]),
                    burst.mixed_forwarded(args, burst.SHAPES[0]),
                    burst.phase_forwarded(args, burst.SHAPES[0]),
                    mixed.forward(args, 'run-local', burst.SHAPES[0]),
                    phases.forwarded(args, 'run-local', burst.SHAPES[0])]
        for command in commands:
            self.assertEqual(command[command.index('--attention') + 1], 'fa3')

    def test_mode_specific_keys_come_only_from_attention_modes(self):
        # run_local needs a GPU, so a hard-coded decision key would only fail
        # after a full local run. Only the ATTENTION_MODES table may name one.
        for module in (burst, mixed, phases):
            source = Path(module.__file__).read_text()
            for mode in burst.ATTENTION_MODES.values():
                literal = f'"{mode["decode_decision"]}"'
                self.assertEqual(source.count(literal), int(module is burst),
                                 f'{module.__name__} hard-codes {literal}')

    def test_vllm_default_budget_reaches_every_child_and_leaves_vllm_unpinned(self):
        self.assertEqual(burst.vllm_budget_kwargs("matched"), {"max_num_batched_tokens": 2048})
        self.assertEqual(burst.vllm_budget_kwargs("default"), {})
        with self.assertRaises(ValueError):
            burst.vllm_budget_kwargs("tuned")
        args = SimpleNamespace(**{**vars(self.args()), 'vllm_budget': 'default'})
        commands = [burst.forwarded(args, 'run-vllm', burst.SHAPES[0]),
                    burst.mixed_forwarded(args, burst.SHAPES[0]),
                    burst.phase_forwarded(args, burst.SHAPES[0]),
                    mixed.forward(args, 'run-vllm', burst.SHAPES[0]),
                    phases.forwarded(args, 'run-vllm', burst.SHAPES[0])]
        for command in commands:
            self.assertEqual(command[command.index('--vllm-budget') + 1], 'default')

    def test_a_vllm_result_from_one_budget_mode_is_never_used_as_the_other(self):
        import json
        import tempfile
        from run_benchmarks import workload_fingerprint
        args = self.args()
        case, _, workload, digest, blocks = burst.input_contract(args, burst.SHAPES[0])
        for recorded, requested in (('default', 'matched'), ('matched', 'default'), (None, 'default')):
            configuration = {} if recorded is None else {'vllm_budget': recorded}
            with tempfile.TemporaryDirectory() as directory:
                (Path(directory) / 'result.json').write_text(json.dumps(
                    {'workload': workload.to_dict(), 'configuration': configuration}))
                self.assertEqual(workload_fingerprint(workload), digest)
                with self.assertRaisesRegex(ValueError, 'budget mode'):
                    burst.vllm_result(Path(directory), workload, digest, case, blocks, 'model', 1, 3,
                                      vllm_budget=requested)
