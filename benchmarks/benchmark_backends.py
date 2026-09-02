"""Backend adapters for the shared inference workload driver.

Imports are intentionally lazy: the CPU-only metric tests and workload generation do
not require CUDA, Triton, model weights, or vLLM to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

from benchmark_core import BackendRun, RequestSpec, RequestTrace, Workload


_HERE = Path(__file__).resolve().parent
_ENGINE_ROOT = _HERE.parent
for _path in (
    _ENGINE_ROOT / "baseline",
    _ENGINE_ROOT / "engine" / "kvcache",
    _ENGINE_ROOT / "engine" / "model_runner",
    _ENGINE_ROOT / "engine" / "graph",
    _ENGINE_ROOT / "engine" / "scheduler",
):
    sys.path.insert(0, os.fspath(_path))


class BackendUnavailable(RuntimeError):
    pass


def scheduler_backend_flags(name: str) -> tuple[bool, bool]:
    """Return ``(use_graph, use_custom_kernels)`` for a scheduler backend."""
    if name not in (
        "continuous-batching",
        "bucketed-cuda-graphs",
        "custom-kernels",
        "regime-dispatched",
    ):
        raise ValueError(name)
    return name == "bucketed-cuda-graphs", name in (
        "custom-kernels",
        "regime-dispatched",
    )


class BenchmarkBackend(ABC):
    name: str

    def __init__(self):
        self.setup_time_s: float | None = None
        self.model_load_time_s: float | None = None

    @abstractmethod
    def run(self, workload: Workload) -> BackendRun:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _torch_dtype(torch, dtype: str):
    try:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}; use float16 or bfloat16") from exc


def _matched_kv_cache_bytes(
    model_id: str,
    *,
    dtype: str,
    block_size: int,
    num_blocks: int,
) -> int:
    """Return the physical K/V tensor bytes used by the local paged-cache pool."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    ))
    element_size = {"float16": 2, "bfloat16": 2}[dtype]
    return (
        num_layers
        * 2  # K and V
        * num_kv_heads
        * head_dim
        * element_size
        * block_size
        * num_blocks
    )


def _sync_and_reset_peak(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory(torch, device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def _wait_until(start: float, target_s: float) -> None:
    while True:
        remaining = start + target_s - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


class SerialEngineBackend(BenchmarkBackend):
    """Shared request driver for batch-one PyTorch baseline and paged engines."""

    def __init__(self, *, name: str, model_id: str, device: str, dtype: str, block_size: int):
        super().__init__()
        import torch

        self.name = name
        self.torch = torch
        self.device = device
        self.dtype = _torch_dtype(torch, dtype)
        self.block_size = block_size

        started = time.perf_counter()
        if name == "pytorch-baseline":
            from baseline_engine import BaselineEngine

            self.engine = BaselineEngine(
                model_id=model_id, batch_size=1, device=device, dtype=self.dtype
            )
        elif name == "paged-kv":
            from paged_engine import PagedEngine

            self.engine = PagedEngine(
                model_id=model_id, block_size=block_size, device=device, dtype=self.dtype
            )
        else:
            raise ValueError(name)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.model_load_time_s = time.perf_counter() - started
        self.cache = None

    def _prepare_cache(self, workload: Workload):
        if self.name == "pytorch-baseline":
            return self.engine.cache

        from paged_kv_cache import PagedKVCache

        max_total = max(
            len(request.prompt_ids) + request.max_tokens for request in workload.requests
        )
        num_blocks = (max_total + self.block_size - 1) // self.block_size
        return PagedKVCache(
            self.engine.cfg,
            batch_size=1,
            num_blocks=num_blocks,
            block_size=self.block_size,
            device=self.device,
            dtype=self.dtype,
        )

    def run(self, workload: Workload) -> BackendRun:
        torch = self.torch
        workload.validate()
        if self.cache is None:
            setup_started = time.perf_counter()
            self.cache = self._prepare_cache(workload)
            self.setup_time_s = time.perf_counter() - setup_started
        elif self.setup_time_s is None:
            self.setup_time_s = 0.0

        _sync_and_reset_peak(torch, self.device)
        started = time.perf_counter()
        traces: list[RequestTrace] = []

        with torch.no_grad():
            for request in workload.sorted_requests():
                _wait_until(started, request.arrival_s)
                submitted_s = time.perf_counter() - started
                self.cache.reset()
                seq = SimpleNamespace(prompt_ids=request.prompt_ids)
                logits = self.engine.prefill(seq, self.cache)
                token = self.engine.sample(logits)
                token_times = [time.perf_counter() - started]
                output_ids = [token]

                for _ in range(request.max_tokens - 1):
                    logits = self.engine.decode_step(self.cache, token)
                    token = self.engine.sample(logits)
                    token_times.append(time.perf_counter() - started)
                    output_ids.append(token)

                finished_s = time.perf_counter() - started
                traces.append(
                    RequestTrace(
                        request_id=request.request_id,
                        prompt_tokens=len(request.prompt_ids),
                        requested_output_tokens=request.max_tokens,
                        arrival_s=request.arrival_s,
                        submitted_s=submitted_s,
                        token_times_s=token_times,
                        finished_s=finished_s,
                        output_tokens=len(token_times),
                        output_ids=output_ids,
                    )
                )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - started
        return BackendRun(
            backend=self.name,
            wall_time_s=wall_time,
            traces=traces,
            peak_gpu_memory_bytes=_peak_memory(torch, self.device),
            setup_time_s=self.setup_time_s,
            metadata={
                "model_load_time_s": self.model_load_time_s,
                "execution": "serial-batch-1",
                "block_size": self.block_size if self.name == "paged-kv" else None,
                "components": (
                    ["paged-kv-cache", "paged-decode-attention"]
                    if self.name == "paged-kv" else ["contiguous-kv-cache", "pytorch-sdpa"]
                ),
            },
        )


class SchedulerBackend(BenchmarkBackend):
    def __init__(
        self,
        *,
        name: str,
        workload: Workload,
        model_id: str,
        device: str,
        dtype: str,
        block_size: int,
        max_running: int,
        num_blocks: int | None,
        max_num_batched_tokens: int,
        max_prefill_chunk_size: int | None,
        max_prefill_attention_pairs: int | None = None,
        prefill_tile_policy: str = "static",
        decode_attention_policy: str = "production",
    ):
        super().__init__()
        import torch
        from paged_engine import PagedEngine
        from scheduler import Scheduler

        use_graph, custom_kernels = scheduler_backend_flags(name)
        regime_dispatched = name == "regime-dispatched"
        if regime_dispatched:
            prefill_tile_policy = "adaptive"
            decode_attention_policy = "adaptive"
        self.name = name
        self.torch = torch
        self.device = device
        self.dtype = _torch_dtype(torch, dtype)
        self.block_size = block_size
        self.max_running = max_running

        load_started = time.perf_counter()
        from naive_forward import Qwen2Config

        self.engine = PagedEngine(
            model_id=model_id,
            cfg=Qwen2Config(use_custom_kernels=custom_kernels),
            block_size=block_size,
            device=device,
            dtype=self.dtype,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.model_load_time_s = time.perf_counter() - load_started

        max_total = max(
            len(request.prompt_ids) + request.max_tokens for request in workload.requests
        )
        self.graph_max_blocks = (max_total + block_size - 1) // block_size
        self.num_blocks = num_blocks or (
            max_running * self.graph_max_blocks + max_running
        )

        setup_started = time.perf_counter()
        self.scheduler = Scheduler(
            self.engine.model,
            self.engine.cfg,
            max_running=max_running,
            num_blocks=self.num_blocks,
            block_size=block_size,
            eos_ids=set(),
            device=device,
            dtype=self.dtype,
            # The custom-kernel scheduler is the controlled eager baseline for
            # static/adaptive token-budget experiments. CUDA graphs remain a
            # separate legacy backend rather than changing coverage with batch mix.
            use_graph=use_graph,
            graph_max_blocks=self.graph_max_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            max_prefill_chunk_size=max_prefill_chunk_size,
            max_prefill_attention_pairs=max_prefill_attention_pairs,
            prefill_tile_policy=prefill_tile_policy,
            decode_attention_policy=decode_attention_policy,
            enable_regime_fusions=regime_dispatched,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        self.setup_time_s = time.perf_counter() - setup_started

    def run(self, workload: Workload) -> BackendRun:
        torch = self.torch
        workload.validate()
        scheduler = self.scheduler
        scheduler.reset()

        pending = workload.sorted_requests()
        pending_index = 0
        traces = {
            request.request_id: RequestTrace(
                request_id=request.request_id,
                prompt_tokens=len(request.prompt_ids),
                requested_output_tokens=request.max_tokens,
                arrival_s=request.arrival_s,
            )
            for request in pending
        }
        scheduler_ids: dict[int, RequestSpec] = {}
        request_objects = {}
        observed_tokens: dict[int, int] = {}
        active_batch_sizes: list[int] = []
        selected_buckets: list[int] = []
        iteration_wall_ms: list[float] = []

        real_pick_bucket = None
        if scheduler.graph_decoder is not None:
            real_pick_bucket = scheduler.graph_decoder._pick_bucket

            def record_bucket(n):
                bucket = real_pick_bucket(n)
                selected_buckets.append(bucket)
                return bucket

            scheduler.graph_decoder._pick_bucket = record_bucket

        _sync_and_reset_peak(torch, self.device)
        started = time.perf_counter()
        try:
            while (
                pending_index < len(pending)
                or scheduler.waiting
                or scheduler.prefilling
                or scheduler.running
            ):
                elapsed = time.perf_counter() - started
                while (
                    pending_index < len(pending)
                    and pending[pending_index].arrival_s <= elapsed
                ):
                    request = pending[pending_index]
                    scheduler_id = scheduler.add_request(
                        request.prompt_ids, request.max_tokens
                    )
                    scheduler_ids[scheduler_id] = request
                    request_objects[scheduler_id] = scheduler.waiting[-1]
                    observed_tokens[scheduler_id] = 0
                    traces[request.request_id].submitted_s = elapsed
                    pending_index += 1

                if not scheduler.waiting and not scheduler.prefilling and not scheduler.running:
                    _wait_until(started, pending[pending_index].arrival_s)
                    continue

                will_prefill = bool(scheduler.prefilling) or bool(
                    scheduler.waiting
                    and scheduler.free_slots
                    and scheduler._can_admit(scheduler.waiting[0])
                )
                attempted_batch = len(scheduler.running) if not will_prefill else None
                preemptions_before = scheduler.n_preemptions
                iterations_before = len(scheduler.iteration_kinds)
                step_started = time.perf_counter()
                scheduler.step()
                # Intermediate prefill chunks do not sample and therefore do not
                # otherwise synchronize. Arrival-time and per-iteration measurements
                # must observe GPU completion before planning the next continuous batch.
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                step_wall_ms = (time.perf_counter() - step_started) * 1e3
                if len(scheduler.iteration_kinds) == iterations_before + 1:
                    iteration_wall_ms.append(step_wall_ms)
                step_finished_s = time.perf_counter() - started
                if attempted_batch and scheduler.n_preemptions == preemptions_before:
                    active_batch_sizes.append(attempted_batch)

                for scheduler_id, request in scheduler_ids.items():
                    obj = request_objects[scheduler_id]
                    generated = obj.total_generated()
                    previous = observed_tokens[scheduler_id]
                    if generated > previous:
                        traces[request.request_id].token_times_s.extend(
                            [step_finished_s] * (generated - previous)
                        )
                        observed_tokens[scheduler_id] = generated
                    if (
                        scheduler_id in scheduler.finished
                        and traces[request.request_id].finished_s is None
                    ):
                        trace = traces[request.request_id]
                        trace.finished_s = step_finished_s
                        trace.output_tokens = len(scheduler.finished[scheduler_id])
                        trace.output_ids = list(scheduler.finished[scheduler_id])
        finally:
            if real_pick_bucket is not None:
                scheduler.graph_decoder._pick_bucket = real_pick_bucket

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_time = time.perf_counter() - started
        iteration_fields = (
            scheduler.iteration_decode_token_counts,
            scheduler.iteration_decode_context_lengths,
            scheduler.iteration_prefill_token_counts,
            scheduler.iteration_prefill_attention_pairs,
            scheduler.iteration_prefill_prefix_lengths,
            scheduler.iteration_token_counts,
            scheduler.iteration_kinds,
            iteration_wall_ms,
        )
        if len({len(values) for values in iteration_fields}) != 1:
            raise AssertionError("per-iteration scheduler metadata lost alignment")
        return BackendRun(
            backend=self.name,
            wall_time_s=wall_time,
            traces=[traces[request.request_id] for request in workload.requests],
            peak_gpu_memory_bytes=_peak_memory(torch, self.device),
            setup_time_s=self.setup_time_s,
            metadata={
                "model_load_time_s": self.model_load_time_s,
                "max_running": self.max_running,
                "max_num_batched_tokens": scheduler.max_num_batched_tokens,
                "execution_mode": (
                    "decode-only-cuda-graphs" if scheduler.use_graph else "fully-eager"
                ),
                "components": [
                    "paged-kv-cache",
                    "paged-decode-attention",
                    "continuous-batching",
                ] + (["bucketed-cuda-graphs"] if scheduler.use_graph else [])
                  + (["triton-rmsnorm", "triton-rope"]
                     if self.name in ("custom-kernels", "regime-dispatched") else [])
                  + ([
                      "regime-dispatch",
                      "adaptive-attention-tiles",
                      "fused-swiglu",
                      "fused-rope-kv-write",
                  ] if self.name == "regime-dispatched" else []),
                "num_blocks": self.num_blocks,
                "block_size": self.block_size,
                "preemptions": scheduler.n_preemptions,
                "prefill_attention_backend": scheduler.prefill_attention_backend,
                "prefill_tile_policy": scheduler.prefill_tile_policy,
                "decode_attention_policy": scheduler.decode_attention_policy,
                "enable_regime_fusions": scheduler.enable_regime_fusions,
                "prefill_batch_sizes": list(scheduler.prefill_batch_sizes),
                "prefill_batch_token_counts": list(scheduler.prefill_batch_token_counts),
                "iteration_decode_token_counts": list(
                    scheduler.iteration_decode_token_counts
                ),
                "iteration_decode_context_lengths": list(
                    scheduler.iteration_decode_context_lengths
                ),
                "iteration_prefill_token_counts": list(
                    scheduler.iteration_prefill_token_counts
                ),
                "iteration_prefill_attention_pairs": list(
                    scheduler.iteration_prefill_attention_pairs
                ),
                "iteration_prefill_prefix_lengths": list(
                    scheduler.iteration_prefill_prefix_lengths
                ),
                "iteration_wall_ms": iteration_wall_ms,
                "iteration_token_counts": list(scheduler.iteration_token_counts),
                "iteration_kinds": list(scheduler.iteration_kinds),
                "iteration_type_counts": {
                    kind: scheduler.iteration_kinds.count(kind)
                    for kind in ("decode_only", "prefill_only", "mixed")
                },
                "max_prefill_chunk_size": scheduler.max_prefill_chunk_size,
                "max_prefill_attention_pairs": scheduler.max_prefill_attention_pairs,
                "mean_prefill_batch_size": (
                    sum(scheduler.prefill_batch_sizes) / len(scheduler.prefill_batch_sizes)
                    if scheduler.prefill_batch_sizes else None
                ),
                "max_prefill_batch_size": max(scheduler.prefill_batch_sizes, default=None),
                "decode_batch_sizes": [
                    count for count in scheduler.iteration_decode_token_counts if count
                ],
                "decode_only_graph_batch_sizes": active_batch_sizes,
                "selected_graph_buckets": selected_buckets,
                "mean_decode_batch_size": (
                    sum(scheduler.iteration_decode_token_counts)
                    / sum(bool(count) for count in scheduler.iteration_decode_token_counts)
                    if any(scheduler.iteration_decode_token_counts) else None
                ),
                "max_decode_batch_size": max(
                    scheduler.iteration_decode_token_counts, default=None
                ),
                "graph_batch_utilization": (
                    sum(active_batch_sizes) / sum(selected_buckets)
                    if selected_buckets and sum(selected_buckets) else None
                ),
            },
        )


class VLLMOfflineBackend(BenchmarkBackend):
    """Offline vLLM comparator for burst workloads.

    Staggered arrivals require vLLM's serving benchmark/client and are deliberately
    rejected here rather than mixing HTTP timing with the in-process local backends.
    """

    name = "vllm"

    def __init__(
        self,
        *,
        model_id: str,
        dtype: str,
        seed: int,
        gpu_memory_utilization: float,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        block_size: int,
        kv_cache_memory_bytes: int | None,
        matched_num_blocks: int | None,
    ):
        super().__init__()
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise BackendUnavailable(
                "vLLM is not installed; install it in the GPU environment or omit --backend vllm"
            ) from exc

        self.SamplingParams = SamplingParams
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.matched_num_blocks = matched_num_blocks
        memory_config = (
            {"kv_cache_memory_bytes": kv_cache_memory_bytes}
            if kv_cache_memory_bytes is not None
            else {"gpu_memory_utilization": gpu_memory_utilization}
        )
        started = time.perf_counter()
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            seed=seed,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            generation_config="vllm",
            enforce_eager=False,
            **memory_config,
        )
        self.model_load_time_s = time.perf_counter() - started

    @staticmethod
    def _metric(metrics, name):
        return getattr(metrics, name, None) if metrics is not None else None

    def run(self, workload: Workload) -> BackendRun:
        if workload.has_staggered_arrivals:
            raise BackendUnavailable(
                "offline vLLM only supports burst workloads in this harness; use request_rate=0"
            )

        inputs = [{"prompt_token_ids": request.prompt_ids} for request in workload.requests]
        params = [
            self.SamplingParams(
                temperature=0.0,
                max_tokens=request.max_tokens,
                ignore_eos=True,
                detokenize=False,
            )
            for request in workload.requests
        ]
        started = time.perf_counter()
        outputs = self.llm.generate(inputs, params, use_tqdm=False)
        wall_time = time.perf_counter() - started

        traces = []
        for request, output in zip(workload.requests, outputs):
            token_ids = list(output.outputs[0].token_ids)
            metrics = getattr(output, "metrics", None)
            arrival = self._metric(metrics, "arrival_time")
            first = self._metric(metrics, "first_token_time")
            finished = self._metric(metrics, "finished_time")
            latency_available = all(value is not None for value in (arrival, first, finished))
            ttft_ms = (first - arrival) * 1e3 if latency_available else None
            e2e_ms = (finished - arrival) * 1e3 if latency_available else None
            tpot_ms = (
                (finished - first) * 1e3 / (len(token_ids) - 1)
                if latency_available and len(token_ids) > 1 else None
            )
            queue_s = self._metric(metrics, "time_in_queue")
            traces.append(
                RequestTrace(
                    request_id=request.request_id,
                    prompt_tokens=len(request.prompt_ids),
                    requested_output_tokens=request.max_tokens,
                    arrival_s=0.0,
                    submitted_s=0.0,
                    token_times_s=[],
                    finished_s=wall_time,
                    output_tokens=len(token_ids),
                    output_ids=token_ids,
                    latency_available=latency_available,
                    reported_ttft_ms=ttft_ms,
                    reported_tpot_ms=tpot_ms,
                    reported_e2e_ms=e2e_ms,
                    reported_queue_ms=queue_s * 1e3 if queue_s is not None else None,
                )
            )

        return BackendRun(
            backend=self.name,
            wall_time_s=wall_time,
            traces=traces,
            setup_time_s=0.0,
            metadata={
                "model_load_time_s": self.model_load_time_s,
                "execution": "vllm-offline-burst",
                "per_request_metrics_available": all(t.latency_available for t in traces),
                "max_running": self.max_num_seqs,
                "max_num_seqs": self.max_num_seqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_model_len": self.max_model_len,
                "block_size": self.block_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
                "matched_num_blocks": self.matched_num_blocks,
                "kv_cache_mode": (
                    "matched-local-pool"
                    if self.kv_cache_memory_bytes is not None else "vllm-native"
                ),
                "prefix_caching": False,
                "cuda_graphs": True,
                "generation_config": "vllm",
                "comparison_scope": "concurrency-matched-offline-burst-throughput",
                "sampling": {
                    "strategy": "greedy",
                    "temperature": 0.0,
                    "ignore_eos": True,
                },
            },
        )


def create_backend(
    name: str,
    *,
    workload: Workload,
    model_id: str,
    device: str,
    dtype: str,
    block_size: int,
    max_running: int,
    num_blocks: int | None,
    seed: int,
    vllm_gpu_memory_utilization: float,
    vllm_kv_cache_mode: str,
    max_num_batched_tokens: int = 4096,
    max_prefill_chunk_size: int | None = None,
    max_prefill_attention_pairs: int | None = None,
    prefill_tile_policy: str = "static",
    decode_attention_policy: str = "production",
) -> BenchmarkBackend:
    if name in ("pytorch-baseline", "paged-kv"):
        return SerialEngineBackend(
            name=name,
            model_id=model_id,
            device=device,
            dtype=dtype,
            block_size=block_size,
        )
    if name in (
        "continuous-batching",
        "bucketed-cuda-graphs",
        "custom-kernels",
        "regime-dispatched",
    ):
        return SchedulerBackend(
            name=name,
            workload=workload,
            model_id=model_id,
            device=device,
            dtype=dtype,
            block_size=block_size,
            max_running=max_running,
            num_blocks=num_blocks,
            max_num_batched_tokens=max_num_batched_tokens,
            max_prefill_chunk_size=max_prefill_chunk_size,
            max_prefill_attention_pairs=max_prefill_attention_pairs,
            prefill_tile_policy=prefill_tile_policy,
            decode_attention_policy=decode_attention_policy,
        )
    if name == "vllm":
        if workload.has_staggered_arrivals:
            raise BackendUnavailable(
                "offline vLLM only supports burst workloads in this harness; "
                "use a burst result or request_rate=0"
            )
        max_model_len = max(
            len(request.prompt_ids) + request.max_tokens
            for request in workload.requests
        )
        graph_max_blocks = (max_model_len + block_size - 1) // block_size
        matched_num_blocks = num_blocks or (
            max_running * graph_max_blocks + max_running
        )
        kv_cache_memory_bytes = (
            _matched_kv_cache_bytes(
                model_id,
                dtype=dtype,
                block_size=block_size,
                num_blocks=matched_num_blocks,
            )
            if vllm_kv_cache_mode == "matched" else None
        )
        return VLLMOfflineBackend(
            model_id=model_id,
            dtype=dtype,
            seed=seed,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_num_seqs=max_running,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            block_size=block_size,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            matched_num_blocks=(
                matched_num_blocks if vllm_kv_cache_mode == "matched" else None
            ),
        )
    raise ValueError(f"unknown backend {name!r}")
