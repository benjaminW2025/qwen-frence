"""Immutable scheduler output for one model iteration."""

from dataclasses import dataclass


def prefill_attention_pairs(prefix_length: int, chunk_length: int) -> int:
    """Causal query-key pairs for a chunk appended after a cached prefix."""
    if prefix_length < 0 or chunk_length < 0:
        raise ValueError("prefix_length and chunk_length must be non-negative")
    return chunk_length * (2 * prefix_length + chunk_length + 1) // 2


def max_chunk_for_attention_pairs(
    prefix_length: int, max_chunk: int, pair_budget: int
) -> int:
    """Largest integer chunk whose causal attention work fits ``pair_budget``."""
    if pair_budget < 0:
        raise ValueError("pair_budget must be non-negative")
    low, high = 0, max_chunk
    while low < high:
        candidate = (low + high + 1) // 2
        if prefill_attention_pairs(prefix_length, candidate) <= pair_budget:
            low = candidate
        else:
            high = candidate - 1
    return low


@dataclass(frozen=True)
class IterationPlan:
    decode_slots: tuple[int, ...]
    prefill_slots: tuple[int, ...]
    prefill_chunk_lengths: tuple[int, ...]

    def __post_init__(self):
        if len(self.prefill_slots) != len(self.prefill_chunk_lengths):
            raise ValueError("prefill slots and chunk lengths must have equal size")
        if any(length < 1 for length in self.prefill_chunk_lengths):
            raise ValueError("prefill chunk lengths must be positive")
        if set(self.decode_slots) & set(self.prefill_slots):
            raise ValueError("a slot cannot decode and prefill in one iteration")

    @property
    def decode_tokens(self) -> int:
        return len(self.decode_slots)

    @property
    def prefill_tokens(self) -> int:
        return sum(self.prefill_chunk_lengths)

    @property
    def total_tokens(self) -> int:
        return self.decode_tokens + self.prefill_tokens

    @property
    def kind(self) -> str:
        if self.decode_tokens and self.prefill_tokens:
            return "mixed"
        if self.decode_tokens:
            return "decode_only"
        if self.prefill_tokens:
            return "prefill_only"
        return "empty"

    def validate_budget(self, max_num_batched_tokens: int) -> None:
        if self.total_tokens > max_num_batched_tokens:
            raise AssertionError(
                f"scheduled {self.total_tokens} tokens with budget "
                f"{max_num_batched_tokens}"
            )
