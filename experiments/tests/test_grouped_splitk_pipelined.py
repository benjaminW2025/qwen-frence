"""CPU benchmark controls plus optional CUDA numerical/graph regression tests.

Run: python3 -m unittest discover -s experiments/tests -p 'test_grouped_splitk_pipelined.py' -v
"""

from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

DECODE = Path(__file__).resolve().parents[1] / "decode"
sys.path.insert(0, str(DECODE))
from benchmark_grouped_splitk_pipelined import ablation_configs, build_parser, correctness_preflight
from grouped_splitk_validation import attention_reference, check_output, correctness_cases, make_inputs

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
HAS_CUDA = HAS_TORCH and find_spec("triton") is not None and torch.cuda.is_available()


class AblationProtocolTests(unittest.TestCase):
    def test_each_comparison_changes_only_its_intended_axis(self):
        configs = ablation_configs(3, 7, num_warps=8, pipeline_stages=3)
        changes = lambda a, b: {key for key in a if a[key] != b[key]}
        self.assertEqual(changes(configs["A"], configs["B"]), {"heads_per_program"})
        self.assertEqual(changes(configs["B"], configs["C"]), {"split_k"})
        self.assertEqual(changes(configs["C"], configs["D"]), {"pipelined", "num_stages"})
        self.assertEqual(configs["C"]["split_k"], 7)
        self.assertEqual(configs["D"]["split_k"], 7)

    def test_cli_exposes_fixed_k_and_pipeline_depth_for_sweeps(self):
        args = build_parser().parse_args(["--split-k", "8", "--pipeline-stages", "3", "--dump-ir"])
        self.assertEqual(args.split_k, 8)
        self.assertEqual(args.pipeline_stages, 3)
        self.assertTrue(args.dump_ir)


@unittest.skipUnless(HAS_TORCH, "requires PyTorch")
class ReferenceTests(unittest.TestCase):
    def test_reference_maps_query_heads_to_kv_heads(self):
        q, k, v, table, lengths = make_inputs([0, 1, 17], device="cpu", strided=True, poison_padding=True)
        q.zero_()
        # Every valid token in KV head 0 is 3, and every token in KV head 1 is 9.
        v[:, :, 0] = 3
        v[:, :, 1] = 9
        expected = torch.zeros_like(q)
        expected[1:, :6] = 3
        expected[1:, 6:] = 9
        check_output(attention_reference(q, k, v, table, lengths), expected)

    def test_strides_and_nan_padding_do_not_change_reference(self):
        base = make_inputs([0, 1, 17, 70], device="cpu", seed=42)
        strided = make_inputs([0, 1, 17, 70], device="cpu", seed=42, strided=True, poison_padding=True)
        self.assertNotEqual(strided[1].stride(), strided[2].stride())
        self.assertEqual(strided[-1].stride(0), 2)
        check_output(attention_reference(*strided), attention_reference(*base))

    def test_reference_edge_cases_have_known_answers(self):
        for name, tensors, options in correctness_cases(device="cpu"):
            if name == "ragged_strided":
                continue
            expected = torch.full_like(tensors[0], 128 if name == "unnormalized_overflow" else 1)
            if name == "negative_scores_empty_splits":
                expected[0].zero_()
            check_output(attention_reference(*tensors, **options), expected)

    def test_gate_rejects_wrong_dtype_and_nonfinite_outputs(self):
        expected = torch.ones(1, dtype=torch.float16)
        with self.assertRaises(AssertionError):
            check_output(expected.float(), expected)
        with self.assertRaises(AssertionError):
            check_output(torch.full_like(expected, float("nan")), expected)


@unittest.skipUnless(HAS_CUDA, "requires CUDA and Triton; CPU checks do not validate GPU codegen")
class GroupedSplitKKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paged_decode_grouped_splitk_pipelined import grouped_splitk_attention, compute_split_k
        cls.attention = staticmethod(grouped_splitk_attention)
        cls.compute_k = staticmethod(compute_split_k)
        cls.original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.original_tf32

    def dtypes(self):
        return [torch.float16, torch.bfloat16] if torch.cuda.is_bf16_supported() else [torch.float16]

    def test_ragged_strided_matrix(self):
        for dtype in self.dtypes():
            tensors = make_inputs([0, 1, 15, 16, 17, 33, 81], dtype=dtype,
                                  strided=True, poison_padding=True, seed=42)
            expected = attention_reference(*tensors)
            for hpp in (1, 2, 3, 6):
                for k in (1, 2, 7):
                    for pipeline, stages in ((False, 1), (True, 2), (True, 3)):
                        with self.subTest(dtype=dtype, hpp=hpp, k=k, pipeline=pipeline, stages=stages):
                            actual = self.attention(*tensors, heads_per_program=hpp, split_k=k,
                                                    pipelined=pipeline, num_stages=stages)
                            check_output(actual, expected)

    def test_adversarial_preflight(self):
        for dtype in self.dtypes():
            records = correctness_preflight(self.attention, ablation_configs(6, 2),
                                             page_size=16, dtype=dtype, seed=0)
            self.assertEqual(len(records), 24)

    def test_all_empty_and_empty_page_table(self):
        tensors = list(make_inputs([0, 0]))
        tensors[3] = tensors[3][:, :0]
        for k in (1, 7):
            for pipelined in (False, True):
                with self.subTest(k=k, pipelined=pipelined):
                    actual = self.attention(*tensors, split_k=k, heads_per_program=6, pipelined=pipelined)
                    check_output(actual, torch.zeros_like(tensors[0]))

    def test_graph_replay_reads_updated_inputs_without_host_sync(self):
        tensors = make_inputs([70, 70])
        for k in (1, 7):
            config = dict(heads_per_program=6, split_k=k, pipelined=True)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.attention(*tensors, **config)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = self.attention(*tensors, **config)
            torch.cuda.current_stream().wait_stream(stream)
            tensors[0].mul_(0.5)
            tensors[-1].copy_(torch.tensor([0, 17], device="cuda", dtype=torch.int32))
            graph.replay()
            check_output(actual, attention_reference(*tensors))

    def test_auto_k_and_input_validation(self):
        self.assertEqual(self.compute_k(1, 2, 1, 256, num_sms=132), 128)
        self.assertEqual(self.compute_k(256, 2, 6, 256, num_sms=132), 1)
        self.assertEqual(self.compute_k(1, 2, 1, 0, num_sms=132), 1)
        tensors = make_inputs([17, 81])
        check_output(self.attention(*tensors, split_k=None, heads_per_program=6), attention_reference(*tensors))
        for k in (0, -1, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.attention(*tensors, split_k=k)
        invalid = (tensors[0][:, :11], *tensors[1:])
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.attention(*invalid)


if __name__ == "__main__":
    unittest.main()
