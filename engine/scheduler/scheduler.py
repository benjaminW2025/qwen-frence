"""
Continuous batching scheduler
"""

import sys, os
from contextlib import contextmanager
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "baseline"))
sys.path.insert(0, os.path.join(_HERE, "..", "kvcache"))
sys.path.insert(0, os.path.join(_HERE, "..", "model_runner"))
sys.path.insert(0, os.path.join(_HERE, "..", "graph"))

from collections import deque

import torch

from paged_kv_cache import PagedKVCache, CacheFullError
from ragged_prefill import ragged_prefill
from mixed_batch import mixed_batch_forward
from paged_graph_decoder import graph_decode_forward
from iteration_plan import (
    IterationPlan,
    max_chunk_for_attention_pairs,
    prefill_attention_pairs,
)
from request_state import Request, Status


@contextmanager
def _null_profile_region():
    yield


class Scheduler:
    def __init__(self, model, cfg, max_running, num_blocks, block_size,
                 eos_ids, device="cuda", dtype=torch.float16,
                 use_graph=False, graph_max_blocks=None, profile_prefill=False,
                 max_prefill_batch_size=None, prefill_attention_backend="triton",
                 max_prefill_chunk_size=None, max_num_batched_tokens=4096,
                 max_prefill_attention_pairs=None, prefill_tile_policy="static",
                 decode_attention_policy="production", enable_regime_fusions=False):
        self.model = model
        self.cfg = cfg
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.eos_ids = eos_ids
        self.max_running = max_running
        self.profile_prefill = profile_prefill
        self.max_prefill_batch_size = (
            max_running if max_prefill_batch_size is None else max_prefill_batch_size
        )
        if not 1 <= self.max_prefill_batch_size <= max_running:
            raise ValueError("max_prefill_batch_size must be between 1 and max_running")
        if prefill_attention_backend not in ("triton", "sdpa"):
            raise ValueError("prefill_attention_backend must be 'triton' or 'sdpa'")
        self.prefill_attention_backend = prefill_attention_backend
        if prefill_tile_policy not in ("static", "adaptive"):
            raise ValueError("prefill_tile_policy must be 'static' or 'adaptive'")
        self.prefill_tile_policy = prefill_tile_policy
        if decode_attention_policy not in ("production", "adaptive"):
            raise ValueError(
                "decode_attention_policy must be 'production' or 'adaptive'"
            )
        if use_graph and decode_attention_policy != "production":
            raise ValueError(
                "adaptive decode attention is not yet supported by CUDA graphs"
            )
        self.decode_attention_policy = decode_attention_policy
        if use_graph and enable_regime_fusions:
            raise ValueError("regime fusions are not yet supported by CUDA graphs")
        self.enable_regime_fusions = bool(enable_regime_fusions)
        if max_prefill_chunk_size is not None and max_prefill_chunk_size < 1:
            raise ValueError("max_prefill_chunk_size must be positive when provided")
        self.max_prefill_chunk_size = max_prefill_chunk_size
        if max_num_batched_tokens < max_running:
            raise ValueError(
                "max_num_batched_tokens must be at least max_running so every "
                "active decode can be scheduled"
            )
        self.max_num_batched_tokens = max_num_batched_tokens
        if (
            max_prefill_attention_pairs is not None
            and max_prefill_attention_pairs < 1
        ):
            raise ValueError("max_prefill_attention_pairs must be positive when provided")
        self.max_prefill_attention_pairs = max_prefill_attention_pairs

        # One shared pool of `num_blocks` blocks across `max_running` slots. Sizing the
        # pool BELOW max_running * max_len is what makes preemption possible (and real).
        self.cache = PagedKVCache(cfg, max_running, num_blocks, block_size, device, dtype)

        self.waiting: deque[Request] = deque()
        self.prefilling: dict[int, Request] = {}       # slot -> partially computed prompt
        self.running: dict[int, Request] = {}          # slot -> Request
        self.free_slots: list[int] = list(range(max_running))
        self.finished: dict[int, list[int]] = {}       # req_id -> output_ids
        self._next_id = 0
        self.n_preemptions = 0                          # observability: how often we evicted
        self.prefill_batch_sizes: list[int] = []
        self.prefill_batch_token_counts: list[int] = []
        self.iteration_decode_token_counts: list[int] = []
        self.iteration_decode_context_lengths: list[list[int]] = []
        self.iteration_prefill_token_counts: list[int] = []
        self.iteration_prefill_attention_pairs: list[int] = []
        self.iteration_prefill_prefix_lengths: list[list[int]] = []
        self.iteration_token_counts: list[int] = []
        self.iteration_kinds: list[str] = []

        # Optional bucketed CUDA-graph decode path (perf). Eager stays the reference.
        self.use_graph = use_graph
        self.graph_decoder = None
        if use_graph:
            from bucketed_graph_decoder import BucketedGraphDecoder
            mb = graph_max_blocks or num_blocks         # a seq can own at most the whole pool
            self.graph_decoder = BucketedGraphDecoder(model, self.cache, max_running, mb,
                                                      device, dtype)

    def _prefill_region(self, name):
        """Emit matching PyTorch-profiler and NVTX ranges only when requested."""
        if not self.profile_prefill:
            return _null_profile_region()

        @contextmanager
        def region():
            from torch.profiler import record_function
            with record_function(name):
                with torch.cuda.nvtx.range(name):
                    yield
        return region()

    def reset(self):
        """Reset request/scheduler state while preserving model, cache buffers, and graphs.

        This is primarily useful for repeated benchmarks: CUDA graphs retain pointers
        into the same cache pool, so rebuilding the scheduler would unnecessarily pay
        capture cost again and could move the addresses baked into those graphs.
        """
        self.cache.reset()
        self.waiting.clear()
        self.prefilling.clear()
        self.running.clear()
        self.free_slots = list(range(self.max_running))
        self.finished.clear()
        self._next_id = 0
        self.n_preemptions = 0
        self.prefill_batch_sizes.clear()
        self.prefill_batch_token_counts.clear()
        self.iteration_decode_token_counts.clear()
        self.iteration_decode_context_lengths.clear()
        self.iteration_prefill_token_counts.clear()
        self.iteration_prefill_attention_pairs.clear()
        self.iteration_prefill_prefix_lengths.clear()
        self.iteration_token_counts.clear()
        self.iteration_kinds.clear()

    def add_request(self, prompt_ids, max_tokens) -> int:
        req = Request(self._next_id, list(prompt_ids), max_tokens)
        self._next_id += 1
        self.waiting.append(req)
        return req.req_id

    @torch.no_grad()
    def run(self):
        """Drive the loop until every request is FINISHED. Returns {req_id: output_ids}."""
        while self.waiting or self.prefilling or self.running:
            self.step()
        return self.finished

    @torch.no_grad()
    def step(self):
        """Plan and execute one decode-priority, globally token-budgeted iteration."""
        plan = self._plan_iteration()
        decode_slots = list(plan.decode_slots)
        prefill_slots = list(plan.prefill_slots)
        chunk_lengths = list(plan.prefill_chunk_lengths)

        if plan.kind == "empty":
            if self.waiting:
                raise CacheFullError("cannot admit any waiting request; pool too small")
            if self.prefilling:
                prefix = next(iter(self.prefilling.values())).num_prompt_tokens_computed
                raise RuntimeError(
                    "max_prefill_attention_pairs cannot fit the next FCFS prompt "
                    f"token at prefix {prefix}; increase it to at least {prefix + 1}"
                )
            return

        decode_lens_before = {slot: self.cache.cur_lens[slot] for slot in decode_slots}
        decode_context_lengths = [decode_lens_before[slot] for slot in decode_slots]
        prefill_prefix_lengths = [
            self.prefilling[slot].num_prompt_tokens_computed for slot in prefill_slots
        ]
        planned_prefill_attention_pairs = sum(
            prefill_attention_pairs(
                self.prefilling[slot].num_prompt_tokens_computed, length
            )
            for slot, length in zip(prefill_slots, chunk_lengths)
        )
        try:
            if plan.kind == "mixed":
                self._mixed_step(decode_slots, prefill_slots, chunk_lengths)
            elif plan.kind == "prefill_only":
                self._advance_prefill_batch(prefill_slots, chunk_lengths)
            else:
                self._decode_step(decode_slots)
        except CacheFullError:
            if not self.running:
                raise
            self._preempt()
            return
        except Exception:
            for slot, old_len in decode_lens_before.items():
                if slot in self.running and self.cache.cur_lens[slot] > old_len:
                    self.cache.truncate(slot, old_len)
            self._rollback_prefills(prefill_slots)
            raise

        self.iteration_decode_token_counts.append(plan.decode_tokens)
        self.iteration_decode_context_lengths.append(decode_context_lengths)
        self.iteration_prefill_token_counts.append(plan.prefill_tokens)
        self.iteration_prefill_attention_pairs.append(planned_prefill_attention_pairs)
        self.iteration_prefill_prefix_lengths.append(prefill_prefix_lengths)
        self.iteration_token_counts.append(plan.total_tokens)
        self.iteration_kinds.append(plan.kind)

    def _plan_iteration(self) -> IterationPlan:
        """Create one immutable decode-first token-budget plan."""
        decode_slots = list(self.running)
        remaining_budget = self.max_num_batched_tokens - len(decode_slots)
        # The attention-work ceiling protects active decodes from sharing an
        # iteration with an oversized prefill. With no decodes to protect, use the
        # full token budget to maximize standalone prefill throughput.
        attention_pair_budget = (
            self.max_prefill_attention_pairs if decode_slots else None
        )
        prefill_slots, chunk_lengths = self._plan_prefill(
            remaining_budget, attention_pair_budget
        )
        plan = IterationPlan(
            tuple(decode_slots), tuple(prefill_slots), tuple(chunk_lengths)
        )
        plan.validate_budget(self.max_num_batched_tokens)
        if attention_pair_budget is not None:
            planned_pairs = sum(
                prefill_attention_pairs(
                    self.prefilling[slot].num_prompt_tokens_computed, length
                )
                for slot, length in zip(prefill_slots, chunk_lengths)
            )
            if planned_pairs > attention_pair_budget:
                raise AssertionError(
                    f"scheduled {planned_pairs} prefill attention pairs with budget "
                    f"{attention_pair_budget}"
                )
        return plan

    def _rollback_prefills(self, slots):
        requests = []
        for slot in slots:
            req = self.prefilling.pop(slot, None)
            if req is None:
                continue
            requests.append(req)
            if self.cache.block_tables[slot]:
                self.cache.free(slot)
            self.free_slots.append(slot)
            req.slot = None
            req.status = Status.WAITING
            req.num_prompt_tokens_computed = 0
        for req in reversed(requests):
            self.waiting.appendleft(req)

    def _admit(self):
        req = self.waiting.popleft()
        slot = self.free_slots.pop()
        req.slot = slot
        req.status = Status.PREFILLING
        self.prefilling[slot] = req

        try:
            self._advance_prefill_batch([slot])
        except Exception:
            # Match batched admission's transactional behavior. Allocation is atomic,
            # but a later model/kernel failure may have populated this slot.
            if self.cache.block_tables[slot]:
                self.cache.free(slot)
            self.prefilling.pop(slot, None)
            req.slot = None
            req.status = Status.WAITING
            req.num_prompt_tokens_computed = 0
            self.free_slots.append(slot)
            self.waiting.appendleft(req)
            raise

    def _planned_prefill_count(self) -> int:
        """Return how many waiting requests can join the current active set FCFS."""
        remaining_blocks = len(self.cache.free_blocks)
        existing_prefills = getattr(self, "prefilling", {})
        projected_running = len(self.running) + len(existing_prefills)

        # Partially prefetched requests own only their pages so far. Reserve the pages
        # needed to finish them before admitting another full prompt.
        for slot, req in existing_prefills.items():
            prompt_blocks = (len(req.prompt_ids) - 1) // self.block_size + 1
            remaining_blocks -= max(
                0, prompt_blocks - len(self.cache.block_tables[slot])
            )
        limit = min(
            len(self.waiting),
            len(self.free_slots),
            max(0, self.max_prefill_batch_size - len(existing_prefills)),
        )
        count = 0
        for req in list(self.waiting)[:limit]:
            prompt_blocks = (len(req.prompt_ids) - 1) // self.block_size + 1
            # This is exactly _can_admit evaluated after each preceding request in
            # the proposed batch has consumed its prompt blocks and become running.
            if prompt_blocks + projected_running > remaining_blocks:
                break
            remaining_blocks -= prompt_blocks
            projected_running += 1
            count += 1
        return count

    def _admit_waiting(self, count):
        """Move ``count`` FCFS requests into PREFILLING without executing them."""
        admitted_slots = []
        for _ in range(count):
            req = self.waiting.popleft()
            slot = self.free_slots.pop()
            req.slot = slot
            req.status = Status.PREFILLING
            self.prefilling[slot] = req
            admitted_slots.append(slot)
        return admitted_slots

    def _plan_prefill(self, token_budget, attention_pair_budget=None):
        """Fill FCFS chunks subject to token and optional attention-work ceilings."""
        if token_budget <= 0:
            return [], []

        admission_count = self._planned_prefill_count() if self.waiting else 0
        if admission_count:
            self._admit_waiting(admission_count)

        slots, chunk_lengths = [], []
        remaining = token_budget
        remaining_pairs = attention_pair_budget
        for slot, req in self.prefilling.items():
            if remaining == 0:
                break
            length = min(req.remaining_prompt_tokens, remaining)
            if self.max_prefill_chunk_size is not None:
                length = min(length, self.max_prefill_chunk_size)
            if remaining_pairs is not None:
                length = max_chunk_for_attention_pairs(
                    req.num_prompt_tokens_computed, length, remaining_pairs
                )
            if length:
                slots.append(slot)
                chunk_lengths.append(length)
                remaining -= length
                if remaining_pairs is not None:
                    remaining_pairs -= prefill_attention_pairs(
                        req.num_prompt_tokens_computed, length
                    )
            elif remaining_pairs is not None:
                # Preserve FCFS: do not bypass an expensive resumed request for a
                # cheaper fresh request later in the queue.
                break
        return slots, chunk_lengths

    def _admit_batch(self):
        """Admit the largest currently feasible FCFS prefill batch."""
        count = self._planned_prefill_count()
        if count < 1:
            raise CacheFullError("no waiting request can be admitted")
        if count == 1:
            self._admit()
            return

        requests = list(self.waiting)[:count]
        slots = self._admit_waiting(count)
        try:
            self._advance_prefill_batch(slots)
        except Exception:
            # Allocation itself is atomic, but a later model failure can leave the
            # selected slots allocated. Roll those slots back before restoring queues.
            for slot in slots:
                if self.cache.block_tables[slot]:
                    self.cache.free(slot)
                self.prefilling.pop(slot, None)
            for req in requests:
                req.slot = None
                req.status = Status.WAITING
                req.num_prompt_tokens_computed = 0
            for req in reversed(requests):
                self.waiting.appendleft(req)
            self.free_slots.extend(reversed(slots))
            raise

    def _finish(self, slot, req):
        req.status = Status.FINISHED
        self.cache.free(slot)                   # blocks back to the pool
        self.free_slots.append(slot)
        del self.running[slot]
        self.finished[req.req_id] = req.full_output()   # includes any pre-preemption tokens

    def _can_admit(self, req) -> bool:
        """
        Return True if the pool has room to prefill this request's prompt. Used
        only for prefill block allocation.
        """
        prompt_blocks = (len(req.prompt_ids) - 1) // self.block_size + 1
        headroom = len(self.running)
        return prompt_blocks + headroom <= len(self.cache.free_blocks)

    def _prefill(self, slot, req) -> int:
        """Compatibility helper: compute one request's next prefill chunk."""
        logits, _ = self._compute_prefill_batch([slot], [req])
        return int(logits[0].argmax())

    def _prefill_batch(self, slots, requests) -> list[int]:
        """Compatibility helper: compute each request's next prefill chunk."""
        logits, _ = self._compute_prefill_batch(slots, requests)
        with self._prefill_region("prefill/sample_batch"):
            return logits.argmax(dim=-1).tolist()

    def _compute_prefill_batch(self, slots, requests, chunk_lengths=None):
        starts = [req.num_prompt_tokens_computed for req in requests]
        if chunk_lengths is None:
            chunk_lengths = [
                min(
                    req.remaining_prompt_tokens,
                    self.max_prefill_chunk_size or req.remaining_prompt_tokens,
                )
                for req in requests
            ]
        else:
            chunk_lengths = list(chunk_lengths)
            if len(chunk_lengths) != len(requests):
                raise ValueError("chunk_lengths must match requests")
            if any(
                length < 1 or length > req.remaining_prompt_tokens
                for req, length in zip(requests, chunk_lengths)
            ):
                raise ValueError("invalid planned prefill chunk length")
        chunks = [
            req.prompt_ids[start:start + length]
            for req, start, length in zip(requests, starts, chunk_lengths)
        ]
        with self._prefill_region("prefill/ragged_batch"):
            logits = ragged_prefill(
                self.model,
                self.cache,
                slots,
                chunks,
                prompt_starts=starts,
                profile_region=self._prefill_region,
                attention_backend=self.prefill_attention_backend,
                prefill_tile_policy=self.prefill_tile_policy,
                enable_regime_fusions=self.enable_regime_fusions,
            )
        return logits, chunk_lengths

    def _advance_prefill_batch(self, slots=None, chunk_lengths=None):
        """Commit one chunk for each selected PREFILLING request."""
        slots = list(self.prefilling) if slots is None else list(slots)
        requests = [self.prefilling[slot] for slot in slots]
        if chunk_lengths is None:
            logits, chunk_lengths = self._compute_prefill_batch(slots, requests)
        else:
            logits, chunk_lengths = self._compute_prefill_batch(
                slots, requests, chunk_lengths
            )

        self.prefill_batch_sizes.append(len(requests))
        self.prefill_batch_token_counts.append(sum(chunk_lengths))
        self._commit_prefill_logits(slots, requests, logits, chunk_lengths)

    def _commit_prefill_logits(self, slots, requests, logits, chunk_lengths):
        """Advance prompt state and transition completed rows to decode-ready."""
        for req, chunk_length in zip(requests, chunk_lengths):
            req.num_prompt_tokens_computed += chunk_length

        completed_rows = [
            row for row, req in enumerate(requests) if req.remaining_prompt_tokens == 0
        ]
        if completed_rows:
            with self._prefill_region("prefill/sample_batch"):
                tokens = logits.index_select(
                    0, torch.tensor(completed_rows, device=logits.device)
                ).argmax(dim=-1).tolist()
            for row, token in zip(completed_rows, tokens):
                slot, req = slots[row], requests[row]
                del self.prefilling[slot]
                req.status = Status.RUNNING
                self.running[slot] = req
                req.output_ids.append(token)
                if req.is_finished(self.eos_ids):
                    self._finish(slot, req)

    def _commit_decode_logits(self, active, logits):
        # One reduction launch and one device-to-host synchronization for the whole
        # decode batch. Sampling each row independently serialized B argmax kernels
        # and B scalar reads even though all logits are already available together.
        with self._prefill_region("decode/sample_batch"):
            tokens = logits.argmax(dim=-1).tolist()
        for slot, token in zip(active, tokens):
            req = self.running[slot]
            req.output_ids.append(token)
            if req.is_finished(self.eos_ids):
                self._finish(slot, req)

    def _mixed_step(self, decode_slots, prefill_slots, chunk_lengths):
        """Execute decode tokens and prompt chunks in one packed transformer pass."""
        prefill_requests = [self.prefilling[slot] for slot in prefill_slots]
        starts = [req.num_prompt_tokens_computed for req in prefill_requests]
        chunks = [
            req.prompt_ids[start:start + length]
            for req, start, length in zip(prefill_requests, starts, chunk_lengths)
        ]
        decode_tokens = [self.running[slot].output_ids[-1] for slot in decode_slots]
        decode_logits, prefill_logits = mixed_batch_forward(
            self.model,
            self.cache,
            decode_slots=decode_slots,
            decode_tokens=decode_tokens,
            prefill_slots=prefill_slots,
            prefill_chunks=chunks,
            prefill_starts=starts,
            profile_region=self._prefill_region,
            prefill_attention_backend=self.prefill_attention_backend,
            prefill_tile_policy=self.prefill_tile_policy,
            decode_attention_policy=self.decode_attention_policy,
            enable_regime_fusions=self.enable_regime_fusions,
        )

        self.prefill_batch_sizes.append(len(prefill_requests))
        self.prefill_batch_token_counts.append(sum(chunk_lengths))
        self._commit_decode_logits(decode_slots, decode_logits)
        self._commit_prefill_logits(
            prefill_slots, prefill_requests, prefill_logits, chunk_lengths
        )

    def _build_step_inputs(self, active_slots, tokens):
        """
        Dense-gather the decode-step inputs over the active slots -> (num_active, ...).
        Call AFTER allocate_block has advanced the active slots' cur_lens.
        """
        bs = self.cache.block_size
        positions, seq_lens, slot_mapping = [], [], []
        for s in active_slots:
            cur = self.cache.cur_lens[s]
            p = cur - 1 # Get the new token
            positions.append(p)
            seq_lens.append(cur)
            slot_mapping.append(self.cache.block_tables[s][p // bs] * bs + (p % bs))

        max_blocks = max(len(self.cache.block_tables[s]) for s in active_slots)
        rows = [self.cache.block_tables[s] + [0] * (max_blocks - len(self.cache.block_tables[s]))
                for s in active_slots]

        B = len(active_slots)
        input_ids    = torch.tensor(tokens, device=self.device, dtype=torch.long).view(B, 1)
        positions    = torch.tensor(positions, device=self.device, dtype=torch.int32)
        seq_lens     = torch.tensor(seq_lens, device=self.device, dtype=torch.int32)
        block_table  = torch.tensor(rows, device=self.device, dtype=torch.int32)
        slot_mapping = torch.tensor(slot_mapping, device=self.device, dtype=torch.long)
        return input_ids, positions, seq_lens, block_table, slot_mapping

    def _decode_step(self, active=None):
        """One decode step over all running slots: forward, sample, finish sweep."""
        active = list(self.running.keys()) if active is None else list(active)

        # Grow every active slot by one token. Atomic -> raises CacheFullError before any
        # mutation, so step() can preempt and retry cleanly.
        n_news = [1 if s in self.running else 0 for s in range(self.max_running)]
        self.cache.allocate_block(n_news)

        tokens = [self.running[s].output_ids[-1] for s in active]  # Feed each seq's last token
        inputs = self._build_step_inputs(active, tokens)
        if self.use_graph:
            logits = self.graph_decoder.decode(*inputs)                  # bucketed graph replay
        else:
            logits = graph_decode_forward(
                self.model,
                self.cache,
                *inputs,
                decode_attention_policy=self.decode_attention_policy,
                max_decode_context_length=max(
                    self.cache.cur_lens[slot] for slot in active
                ),
                enable_regime_fusions=self.enable_regime_fusions,
            ) # (num_active,1,vocab) eager

        self._commit_decode_logits(active, logits[:, -1])

    def _preempt(self):
        """Evict a running request under memory pressure and re-queue it for recompute."""
        self.n_preemptions += 1
        # LIFO: evict the most-recently-admitted running request (avoids starving old ones).
        victim_slot = list(self.running)[-1] # Take the last (most recently deployed) running slot
        victim = self.running[victim_slot] # Grab that request

        self.cache.free(victim_slot) # Free the cache               
        self.free_slots.append(victim_slot) # Free the slot
        del self.running[victim_slot] # Get rid of running request

        # Recompute: fold generated tokens into the prompt so a later re-admit re-prefills
        # to where it left off. n_prompt is untouched, so full_output()/total_generated()
        # still account for these tokens.
        victim.prompt_ids = victim.prompt_ids + victim.output_ids # Now processes it as if one singular stored prompt
        victim.output_ids = [] # Reset output tokens
        victim.num_prompt_tokens_computed = 0
        victim.status = Status.WAITING
        victim.slot = None
        self.waiting.appendleft(victim)                      
