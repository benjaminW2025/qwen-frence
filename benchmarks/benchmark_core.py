"""Backend-independent workload and metric primitives for inference benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import statistics
from typing import Any, Iterable


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    arrival_s: float = 0.0

    def validate(self) -> None:
        if not self.prompt_ids:
            raise ValueError(f"{self.request_id}: prompt must contain at least one token")
        if self.max_tokens < 1:
            raise ValueError(f"{self.request_id}: max_tokens must be >= 1")
        if self.arrival_s < 0:
            raise ValueError(f"{self.request_id}: arrival_s must be >= 0")


@dataclass
class Workload:
    name: str
    requests: list[RequestSpec]
    seed: int = 0
    arrival_pattern: str = "burst"

    def validate(self) -> None:
        if not self.requests:
            raise ValueError("workload must contain at least one request")
        ids = [request.request_id for request in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("request IDs must be unique")
        for request in self.requests:
            request.validate()

    @property
    def has_staggered_arrivals(self) -> bool:
        return any(request.arrival_s > 0 for request in self.requests)

    def sorted_requests(self) -> list[RequestSpec]:
        return sorted(self.requests, key=lambda request: (request.arrival_s, request.request_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "arrival_pattern": self.arrival_pattern,
            "requests": [asdict(request) for request in self.requests],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Workload":
        workload = cls(
            name=value["name"],
            seed=int(value.get("seed", 0)),
            arrival_pattern=value.get("arrival_pattern", "custom"),
            requests=[RequestSpec(**request) for request in value["requests"]],
        )
        workload.validate()
        return workload


@dataclass
class RequestTrace:
    request_id: str
    prompt_tokens: int
    requested_output_tokens: int
    arrival_s: float
    submitted_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)
    finished_s: float | None = None
    output_tokens: int = 0
    output_ids: list[int] = field(default_factory=list)
    latency_available: bool = True
    reported_ttft_ms: float | None = None
    reported_tpot_ms: float | None = None
    reported_e2e_ms: float | None = None
    reported_queue_ms: float | None = None

    def ttft_ms(self) -> float | None:
        if self.reported_ttft_ms is not None:
            return self.reported_ttft_ms
        if not self.latency_available:
            return None
        if not self.token_times_s:
            return None
        return (self.token_times_s[0] - self.arrival_s) * 1e3

    def queue_ms(self) -> float | None:
        if self.reported_queue_ms is not None:
            return self.reported_queue_ms
        if not self.latency_available:
            return None
        if self.submitted_s is None:
            return None
        return (self.submitted_s - self.arrival_s) * 1e3

    def e2e_ms(self) -> float | None:
        if self.reported_e2e_ms is not None:
            return self.reported_e2e_ms
        if not self.latency_available:
            return None
        if self.finished_s is None:
            return None
        return (self.finished_s - self.arrival_s) * 1e3

    def tpot_ms(self) -> float | None:
        if self.reported_tpot_ms is not None:
            return self.reported_tpot_ms
        if not self.latency_available:
            return None
        if self.finished_s is None or len(self.token_times_s) < 2:
            return None
        return (self.finished_s - self.token_times_s[0]) * 1e3 / (len(self.token_times_s) - 1)

    def itl_ms(self) -> list[float]:
        return [
            (right - left) * 1e3
            for left, right in zip(self.token_times_s, self.token_times_s[1:])
        ]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            ttft_ms=self.ttft_ms(),
            queue_ms=self.queue_ms(),
            e2e_ms=self.e2e_ms(),
            tpot_ms=self.tpot_ms(),
            itl_ms=self.itl_ms(),
        )
        return value


@dataclass
class BackendRun:
    backend: str
    wall_time_s: float
    traces: list[RequestTrace]
    peak_gpu_memory_bytes: int | None = None
    setup_time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> dict[str, Any]:
        return aggregate_run(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "wall_time_s": self.wall_time_s,
            "setup_time_s": self.setup_time_s,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "metadata": self.metadata,
            "metrics": self.aggregate(),
            "requests": [trace.to_dict() for trace in self.traces],
        }


def parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("expected a comma-separated list of positive integers")
    return parsed


def make_synthetic_workload(
    *,
    name: str,
    num_requests: int,
    prompt_lengths: list[int],
    output_lengths: list[int],
    vocab_size: int,
    seed: int,
    request_rate: float = 0.0,
) -> Workload:
    if num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    if request_rate < 0:
        raise ValueError("request_rate must be >= 0")
    if not prompt_lengths or not output_lengths:
        raise ValueError("prompt_lengths and output_lengths must not be empty")
    if any(length < 1 for length in prompt_lengths + output_lengths):
        raise ValueError("prompt and output lengths must be >= 1")

    rng = random.Random(seed)
    arrival = 0.0
    requests = []
    for index in range(num_requests):
        if index and request_rate > 0:
            arrival += rng.expovariate(request_rate)
        prompt_len = prompt_lengths[index % len(prompt_lengths)]
        output_len = output_lengths[index % len(output_lengths)]
        # Token zero is valid for Qwen, but avoiding it makes synthetic prompts less
        # likely to consist mostly of special/padding tokens across model families.
        prompt_ids = [rng.randrange(1, vocab_size) for _ in range(prompt_len)]
        requests.append(
            RequestSpec(
                request_id=f"request-{index:04d}",
                prompt_ids=prompt_ids,
                max_tokens=output_len,
                arrival_s=arrival,
            )
        )

    workload = Workload(
        name=name,
        requests=requests,
        seed=seed,
        arrival_pattern="poisson" if request_rate > 0 else "burst",
    )
    workload.validate()
    return workload


def percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if not 0 <= q <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": percentile(finite, 95),
        "p99": percentile(finite, 99),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def aggregate_run(run: BackendRun) -> dict[str, Any]:
    completed = [trace for trace in run.traces if trace.finished_s is not None]
    output_tokens = sum(trace.output_tokens for trace in completed)
    input_tokens = sum(trace.prompt_tokens for trace in completed)
    wall_time = run.wall_time_s
    return {
        "requests": len(run.traces),
        "completed_requests": len(completed),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "wall_time_s": wall_time,
        "setup_time_s": run.setup_time_s,
        "request_throughput_rps": len(completed) / wall_time if wall_time > 0 else None,
        "output_throughput_tok_s": output_tokens / wall_time if wall_time > 0 else None,
        "total_throughput_tok_s": (input_tokens + output_tokens) / wall_time if wall_time > 0 else None,
        "ttft_ms": distribution(trace.ttft_ms() for trace in completed),
        "tpot_ms": distribution(trace.tpot_ms() for trace in completed),
        "itl_ms": distribution(value for trace in completed for value in trace.itl_ms()),
        "e2e_ms": distribution(trace.e2e_ms() for trace in completed),
        "queue_ms": distribution(trace.queue_ms() for trace in completed),
        "peak_gpu_memory_bytes": run.peak_gpu_memory_bytes,
    }


def median_aggregate(runs: list[BackendRun]) -> dict[str, Any]:
    if not runs:
        raise ValueError("cannot summarize zero runs")
    aggregates = [run.aggregate() for run in runs]
    scalar_keys = (
        "wall_time_s",
        "setup_time_s",
        "request_throughput_rps",
        "output_throughput_tok_s",
        "total_throughput_tok_s",
        "peak_gpu_memory_bytes",
    )
    summary: dict[str, Any] = {"repetitions": len(runs)}
    for key in scalar_keys:
        values = [aggregate[key] for aggregate in aggregates if aggregate[key] is not None]
        summary[key] = statistics.median(values) if values else None
    for metric in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "queue_ms"):
        summary[metric] = {}
        for statistic in ("mean", "median", "p95", "p99", "min", "max"):
            values = [
                aggregate[metric][statistic]
                for aggregate in aggregates
                if aggregate[metric][statistic] is not None
            ]
            summary[metric][statistic] = statistics.median(values) if values else None
    return summary
