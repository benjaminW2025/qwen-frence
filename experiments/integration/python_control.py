"""Readable Python control matching the C++ prototype's scheduling semantics.

This is not engine/scheduler/scheduler.py: that scheduler has different admission,
preemption, token budgets, and mixed-forward behavior. The control intentionally
uses Python lists + pageable tensor construction, measuring the implementation
package (including C++ pinned staging), not the language in isolation.
"""
from collections import deque
from dataclasses import dataclass, field
import torch


@dataclass
class Request:
    id: int
    prompt: list
    limit: int
    computed: int = 0
    output: list = field(default_factory=list)
    blocks: list = field(default_factory=list)


class PythonControl:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.waiting = deque()
        self.running = []
        self.completed = []
        self.next_id = 0
        self.max_blocks = (config.max_context_length + config.block_size - 1) // config.block_size
        self.free = list(range(config.max_batch_size * self.max_blocks))
        self.max_context = 0

    def submit_request(self, prompt_ids, max_output_tokens):
        if not prompt_ids or max_output_tokens <= 0:
            raise ValueError("nonempty prompt and positive output length required")
        if len(prompt_ids) + max_output_tokens > self.config.max_context_length:
            raise ValueError("request exceeds context capacity")
        request = Request(self.next_id, list(prompt_ids), max_output_tokens)
        self.next_id += 1
        self.waiting.append(request)
        return request.id

    def num_pending(self):
        return len(self.waiting)

    def num_running(self):
        return len(self.running)

    def max_decode_context_length(self):
        return self.max_context

    def pop_completed(self):
        result, self.completed = self.completed, []
        return result

    def _slot(self, request, position):
        size = self.config.block_size
        return request.blocks[position // size] * size + position % size

    def _table(self, requests):
        return [r.blocks + [0] * (self.max_blocks - len(r.blocks)) for r in requests]

    def _tensor(self, values, dtype=torch.int64):
        return torch.tensor(values, dtype=dtype, device=self.device)

    @torch.no_grad()
    def step(self, forward):
        decode = [r for r in self.running if r.computed == len(r.prompt)]
        prefill = []
        budget = self.config.max_prefill_tokens_per_iter
        for request in self.running:
            remaining = len(request.prompt) - request.computed
            if remaining and budget:
                chunk = min(remaining, budget)
                prefill.append((request, chunk))
                budget -= chunk
        while self.waiting and budget and len(self.running) < self.config.max_batch_size:
            request = self.waiting[0]
            blocks = (len(request.prompt) + request.limit + self.config.block_size - 1) // self.config.block_size
            if blocks > len(self.free):
                break
            request.blocks = [self.free.pop() for _ in range(blocks)]
            self.waiting.popleft()
            self.running.append(request)
            chunk = min(len(request.prompt), budget)
            prefill.append((request, chunk))
            budget -= chunk
        if not decode and not prefill:
            return 0
        self.max_context = max((len(r.prompt) + len(r.output) for r in decode), default=0)
        calls = []
        if decode:
            positions = [len(r.prompt) + len(r.output) - 1 for r in decode]
            calls.append((self._tensor([r.output[-1] if r.output else r.prompt[-1] for r in decode]),
                          self._tensor(positions),
                          self._tensor([self._slot(r, p) for r, p in zip(decode, positions)]),
                          self._tensor([], torch.int32), self._tensor([p + 1 for p in positions], torch.int32),
                          self._tensor(self._table(decode), torch.int32), 1, True))
        if prefill:
            ids, positions, slots, cu, context = [], [], [], [0], []
            for request, chunk in prefill:
                for position in range(request.computed, request.computed + chunk):
                    ids.append(request.prompt[position])
                    positions.append(position)
                    slots.append(self._slot(request, position))
                cu.append(len(ids))
                context.append(request.computed + chunk)
            calls.append((self._tensor(ids), self._tensor(positions), self._tensor(slots),
                          self._tensor(cu, torch.int32), self._tensor(context, torch.int32),
                          self._tensor(self._table([r for r, _ in prefill]), torch.int32),
                          max(chunk for _, chunk in prefill), False))
        # Build both phases before executing either, as in C++.
        logits = [forward(*args) for args in calls]
        logits = torch.cat(logits, 0) if len(logits) == 2 else logits[0]
        tokens = logits.argmax(-1).cpu().tolist()
        done = set()
        def append(request, token):
            request.output.append(token)
            if token == self.config.eos_token_id or len(request.output) >= request.limit:
                done.add(request.id)
        cursor = 0
        for request in decode:
            append(request, tokens[cursor])
            cursor += 1
        for request, chunk in prefill:
            request.computed += chunk
            if request.computed == len(request.prompt):
                append(request, tokens[cursor])
            cursor += 1
        retained = []
        for request in self.running:
            if request.id in done:
                self.completed.append((request.id, request.output))
                self.free.extend(request.blocks)
            else:
                retained.append(request)
        self.running = retained
        return len(done)
