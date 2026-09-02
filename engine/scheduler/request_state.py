"""Request lifecycle state for continuous batching."""

from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    WAITING = 1
    PREFILLING = 2
    RUNNING = 3
    FINISHED = 4


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    max_tokens: int
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    slot: int | None = None
    n_prompt: int = 0
    num_prompt_tokens_computed: int = 0

    def __post_init__(self):
        # Preemption folds generated tokens into prompt_ids for recompute. Preserve
        # the original boundary so output accounting remains correct afterward.
        self.n_prompt = len(self.prompt_ids)

    def total_generated(self) -> int:
        return (len(self.prompt_ids) - self.n_prompt) + len(self.output_ids)

    @property
    def remaining_prompt_tokens(self) -> int:
        return len(self.prompt_ids) - self.num_prompt_tokens_computed

    @property
    def next_kv_position(self) -> int:
        return self.num_prompt_tokens_computed

    @property
    def next_rope_position(self) -> int:
        return self.num_prompt_tokens_computed

    def full_output(self) -> list[int]:
        return self.prompt_ids[self.n_prompt:] + self.output_ids

    def is_finished(self, eos_ids) -> bool:
        return (
            self.total_generated() >= self.max_tokens
            or (bool(self.output_ids) and self.output_ids[-1] in eos_ids)
        )
